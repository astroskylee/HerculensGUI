from __future__ import annotations

import atexit
import inspect
import json
import os
import pickle
import shutil
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import arviz as az
import jax
import jax_lensing_profiles  # noqa: F401 - registers MULTI_GAUSSIAN_ELLIPSE
import jax.numpy as jnp
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpyro
import numpyro.infer as infer
import numpyro.infer.autoguide as autoguide
import optax
import xarray as xr
from astropy.io import fits
from corner import corner
from jax import lax
from matplotlib.gridspec import GridSpec
from numpyro import distributions as dist
from numpyro.distributions.util import lazy_property
from numpyro.handlers import condition
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares
import scipy
from scipy.special import roots_legendre
from skimage import measure

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF
from herculens.LensImage.lens_image import LensImage
from herculens.LensImage.lens_image_multiplane import MPLensImage
from herculens.LightModel.light_model import LightModel
from herculens.LightModel.light_model_multiplane import MPLightModel
from herculens.MassModel import mass_model_base
from herculens.MassModel.mass_model import MassModel
from herculens.MassModel.mass_model_multiplane import MPMassModel
from herculens.MassModel.Profiles.pixelated import PixelatedFixed
from herculens.PointSourceModel.point_source_model import PointSourceModel
from herculens.Util import model_util
from jax_lensing_profiles.MassModel.Profiles.CuspyNFW_ellipse_kappa import CuspyNFW_3D_fn
from jax_lensing_profiles.MassModel.Profiles.MGE import MGE


def pixel_scale_arcsec_from_header(header):
    if header.get("CD1_1") is not None:
        return 3600.0 * float(np.hypot(float(header["CD1_1"]), float(header.get("CD2_1", 0.0))))
    return 3600.0 * abs(float(header["CDELT1"]))


def exposure_time_from_header(header, fallback):
    for key in ("EXPTIME", "TEXPTIME", "EFFTIME", "DURATION"):
        if header.get(key) is not None:
            return float(header[key])
    return float(fallback)


def resolve_auto_sigma_lims(sigma_lims, data_shape, pix_scale):
    sigma_min = max(float(sigma_lims[0]), 1e-6)
    sigma_max = sigma_lims[1]
    if sigma_max is None or str(sigma_max).strip().lower() in {"", "auto", "half", "half_image", "half image"}:
        sigma_max = 0.5 * min(data_shape) * pix_scale
    return [sigma_min, max(float(sigma_max), sigma_min * 1.01)]


def format_elapsed_seconds(seconds):
    seconds = max(float(seconds), 0.0)
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def runtime_device_summary():
    requested_label = os.environ.get("HERCULENS_GPU_LABEL", "").strip()
    requested_gres = os.environ.get("HERCULENS_GPU_GRES", "").strip()
    requested_partition = os.environ.get("HERCULENS_GPU_PARTITION", "").strip()
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME") or ""
    try:
        backend = jax.default_backend()
        devices = jax.devices()
    except Exception as exc:
        backend = "unknown"
        devices = []
        device_error = str(exc)
    else:
        device_error = ""
    device_kinds = []
    for device in devices:
        kind = str(getattr(device, "device_kind", "") or getattr(device, "platform", "") or device)
        if kind not in device_kinds:
            device_kinds.append(kind)
    probe_text = " ".join([requested_label, requested_gres, *device_kinds]).upper()
    if "A100" in probe_text:
        device_type = "A100"
    elif "L40" in probe_text:
        device_type = "L40"
    elif "GPU" in probe_text or backend.lower() == "gpu":
        device_type = "GPU"
    elif backend:
        device_type = backend.upper()
    else:
        device_type = "unknown"
    return {
        "device_type": device_type,
        "backend": backend,
        "device_count": len(devices),
        "device_kinds": device_kinds,
        "requested_label": requested_label,
        "requested_gres": requested_gres,
        "requested_partition": requested_partition,
        "node": node,
        "device_error": device_error,
    }


def start_lens_runtime_timer(label="lens model", output_dir=None):
    timer = {
        "label": str(label),
        "output_dir": str(output_dir) if output_dir is not None else "",
        "start_perf": time.perf_counter(),
        "start_wall": datetime.now(),
        "device": runtime_device_summary(),
        "finished": False,
    }
    start_iso = timer["start_wall"].isoformat(timespec="seconds")
    device = timer["device"]
    print(f"RUN_TIMER_START label={timer['label']} wall_time={start_iso} output_dir={timer['output_dir']}", flush=True)
    print(
        "RUNTIME_DEVICE "
        f"type={device['device_type']} "
        f"backend={device['backend']} "
        f"count={device['device_count']} "
        f"kinds={json.dumps(device['device_kinds'])} "
        f"requested_label={json.dumps(device['requested_label'])} "
        f"requested_gres={json.dumps(device['requested_gres'])} "
        f"requested_partition={json.dumps(device['requested_partition'])} "
        f"node={json.dumps(device['node'])}",
        flush=True,
    )
    if device["device_error"]:
        print(f"RUNTIME_DEVICE_ERROR {device['device_error']}", flush=True)

    def finalize_if_needed():
        if not timer.get("finished"):
            finish_lens_runtime_timer(timer, status="exited")

    atexit.register(finalize_if_needed)
    return timer


def finish_lens_runtime_timer(timer, status="completed"):
    if timer is None:
        return {}
    if timer.get("finished"):
        return timer.get("summary", {})
    end_wall = datetime.now()
    elapsed_seconds = time.perf_counter() - float(timer.get("start_perf", time.perf_counter()))
    summary = {
        "label": timer.get("label", "lens model"),
        "status": str(status),
        "start_time": timer.get("start_wall").isoformat(timespec="seconds") if timer.get("start_wall") else None,
        "end_time": end_wall.isoformat(timespec="seconds"),
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_hms": format_elapsed_seconds(elapsed_seconds),
        "device": timer.get("device", {}),
    }
    timer["finished"] = True
    timer["summary"] = summary
    print(
        "RUN_TIMER_DONE "
        f"label={summary['label']} "
        f"status={summary['status']} "
        f"elapsed_seconds={summary['elapsed_seconds']:.3f} "
        f"elapsed_hms={summary['elapsed_hms']} "
        f"start_time={summary['start_time']} "
        f"end_time={summary['end_time']}",
        flush=True,
    )
    return summary


class Plot:
    @staticmethod
    def sanitize_label(label):
        return (
            str(label)
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('/', '_')
        )

    @staticmethod
    def lens_light_image(lens_image, kwargs):
        if 'kwargs_light' in kwargs:
            return lens_image.model(
                eta_flat=kwargs['eta_flat'],
                kwargs_mass=kwargs['kwargs_mass'],
                kwargs_light=kwargs['kwargs_light'],
                k_planes=(0,),
                apply_mask=False,
            )
        return lens_image.model(
            kwargs_lens=kwargs.get('kwargs_lens'),
            kwargs_source=kwargs.get('kwargs_source'),
            kwargs_lens_light=kwargs.get('kwargs_lens_light'),
            source_add=False,
            lens_light_add=True,
            point_source_add=False,
        )

    @staticmethod
    def lensed_arc_image(lens_image, kwargs, plane_index, apply_mask=True):
        return lens_image.model(
            eta_flat=kwargs['eta_flat'],
            kwargs_mass=kwargs['kwargs_mass'],
            kwargs_light=kwargs['kwargs_light'],
            k_planes=(plane_index,),
            apply_mask=apply_mask,
        )

    @staticmethod
    def plot_loss(losses, max_iterations, ax=None, axins=None, inset=True, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 3.5))
        ax.plot(losses, **kwargs)
        ax.set_yscale('asinh')

        if inset and axins is None:
            axins = ax.inset_axes([0.3, 0.5, 0.64, 0.45])
        n_end = max_iterations // 3
        x_plot = np.linspace(max_iterations - n_end, max_iterations, n_end)
        if inset:
            axins.plot(x_plot, losses[max_iterations - n_end:], **kwargs)
            ax.indicate_inset_zoom(axins, edgecolor='k')
        return ax

    @staticmethod
    def pixelize_plane(lens_image, herc_dict, num_pix, source_grid_scale=None):
        if source_grid_scale is None:
            source_grid_scale = lens_image._source_grid_scale
        x, y, extent = lens_image.get_source_coordinates(
            herc_dict['kwargs_lens'],
            force=True,
            npix_src=num_pix,
            source_grid_scale=source_grid_scale,
        )
        xgrid, ygrid = jnp.meshgrid(x, y)
        image_grid = lens_image.SourceModel.surface_brightness(
            xgrid,
            ygrid,
            herc_dict['kwargs_source'],
            pixels_x_coord=xgrid[0],
            pixels_y_coord=ygrid[:, 0],
        ) * lens_image.Grid.pixel_area
        return image_grid, extent

    @staticmethod
    def pixelize_plane_multiplane(lens_image, herc_dict, num_pix, source_grid_scale=1.0, N=1):
        _, _, extents = lens_image.get_source_coordinates(
            herc_dict['eta_flat'],
            herc_dict['kwargs_mass'],
            force=True,
            npix_src=num_pix,
            source_grid_scale=source_grid_scale,
        )
        extent = extents[N]
        x = jnp.linspace(extent[0], extent[1], num_pix)
        y = jnp.linspace(extent[2], extent[3], num_pix)
        xgrid, ygrid = jnp.meshgrid(x, y)
        image_grid = lens_image.MPLightModel.light_models[N].surface_brightness(
            xgrid.flatten(),
            ygrid.flatten(),
            herc_dict['kwargs_light'][N],
            pixels_x_coord=x,
            pixels_y_coord=y,
        ) * lens_image.Grid.pixel_area
        return image_grid.reshape(num_pix, num_pix), extent

    @staticmethod
    def source_plane_footprint_mask(
        lens_image,
        herc_dict,
        source_shape,
        source_extent,
        N=2,
        grow_pixels=0,
        close_pixels=2,
        fill_holes=True,
        convex_hull=False,
        boundary_fill=True,
        erode_pixels=0,
    ):
        """Map the full image-plane arc mask for plane N onto a source-plane grid."""
        ny, nx = [int(v) for v in source_shape]
        if ny <= 0 or nx <= 0:
            return np.zeros((max(ny, 0), max(nx, 0)), dtype=bool)
        # ``_source_arc_masks_flat_bool`` stores only the eroded mask boundary
        # and is used by Herculens to define adaptive source-grid extents.  A
        # source-plane visibility footprint must instead trace every masked
        # image-plane sample, including the mask interior.
        masks = getattr(lens_image, '_source_arc_masks_flat', None)
        if masks is None:
            masks = getattr(lens_image, '_source_arc_masks_flat_bool', None)
        if masks is None or N >= len(masks) or masks[N] is None:
            return np.ones((ny, nx), dtype=bool)
        image_mask = np.asarray(jax.device_get(masks[N]), dtype=bool).reshape(-1)
        if not np.any(image_mask):
            return np.zeros((ny, nx), dtype=bool)

        x_img, y_img = lens_image.ImageNumerics.coordinates_evaluate
        ra_planes, dec_planes = lens_image.MPMassModel.ray_shooting(
            x_img,
            y_img,
            herc_dict['eta_flat'],
            herc_dict['kwargs_mass'],
        )
        x_src = np.asarray(jax.device_get(ra_planes[N])).reshape(-1)[image_mask]
        y_src = np.asarray(jax.device_get(dec_planes[N])).reshape(-1)[image_mask]
        xmin, xmax, ymin, ymax = [float(v) for v in source_extent]
        if xmax <= xmin or ymax <= ymin:
            return np.zeros((ny, nx), dtype=bool)

        ix = np.floor((x_src - xmin) / (xmax - xmin) * nx).astype(int)
        iy = np.floor((y_src - ymin) / (ymax - ymin) * ny).astype(int)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        footprint = np.zeros((ny, nx), dtype=bool)
        footprint[iy[valid], ix[valid]] = True
        if boundary_fill:
            image_numerics = getattr(lens_image, 'ImageNumerics', None)
            grid_class = getattr(image_numerics, 'grid_class', None)
            grid_shape = getattr(grid_class, 'num_pixel_axes', None)
            if grid_shape is None or int(np.prod(grid_shape)) != image_mask.size:
                side = int(round(np.sqrt(image_mask.size)))
                grid_shape = (side, side) if side * side == image_mask.size else None
            if grid_shape is not None:
                x_grid = np.asarray(jax.device_get(x_img)).reshape(grid_shape)
                y_grid = np.asarray(jax.device_get(y_img)).reshape(grid_shape)
                mask_2d = image_mask.reshape(grid_shape).astype(float)
                fig, ax = plt.subplots()
                contour = ax.contour(x_grid, y_grid, mask_2d, levels=[0.5])
                plt.close(fig)
                yy, xx = np.mgrid[:ny, :nx]
                x_centers = xmin + (xx.ravel() + 0.5) * (xmax - xmin) / nx
                y_centers = ymin + (yy.ravel() + 0.5) * (ymax - ymin) / ny
                centers = np.column_stack([x_centers, y_centers])
                from matplotlib.path import Path as MplPath

                for segment in contour.allsegs[0]:
                    if len(segment) < 3:
                        continue
                    src_x, src_y = lens_image.MPMassModel.ray_shooting(
                        jnp.asarray(segment[:, 0]),
                        jnp.asarray(segment[:, 1]),
                        herc_dict['eta_flat'],
                        herc_dict['kwargs_mass'],
                    )
                    vertices = np.column_stack([
                        np.asarray(jax.device_get(src_x[N])),
                        np.asarray(jax.device_get(src_y[N])),
                    ])
                    if len(vertices) >= 3:
                        footprint |= MplPath(vertices, closed=True).contains_points(centers).reshape(ny, nx)
        if convex_hull and np.count_nonzero(valid) >= 3:
            from matplotlib.path import Path as MplPath
            from scipy.spatial import ConvexHull

            points = np.column_stack([ix[valid], iy[valid]])
            hull = ConvexHull(points)
            hull_path = MplPath(points[hull.vertices])
            yy, xx = np.mgrid[:ny, :nx]
            centers = np.column_stack([xx.ravel(), yy.ravel()])
            footprint |= hull_path.contains_points(centers).reshape(ny, nx)
        if close_pixels and int(close_pixels) > 0:
            footprint = scipy.ndimage.binary_closing(footprint, iterations=int(close_pixels))
        if fill_holes:
            footprint = scipy.ndimage.binary_fill_holes(footprint)
        if grow_pixels and int(grow_pixels) > 0:
            footprint = scipy.ndimage.binary_dilation(footprint, iterations=int(grow_pixels))
        if erode_pixels and int(erode_pixels) > 0:
            footprint = scipy.ndimage.binary_erosion(footprint, iterations=int(erode_pixels))
        return footprint

    @staticmethod
    def mask_source_plane_image(
        source_image,
        lens_image,
        herc_dict,
        source_extent,
        N=2,
        grow_pixels=0,
        close_pixels=2,
        fill_holes=True,
        convex_hull=False,
        boundary_fill=True,
        erode_pixels=0,
    ):
        """Set source-plane pixels outside the traced image-plane arc mask to zero."""
        image = np.asarray(jax.device_get(source_image), dtype=float)
        footprint = Plot.source_plane_footprint_mask(
            lens_image,
            herc_dict,
            image.shape,
            source_extent,
            N=N,
            grow_pixels=grow_pixels,
            close_pixels=close_pixels,
            fill_holes=fill_holes,
            convex_hull=convex_hull,
            boundary_fill=boundary_fill,
            erode_pixels=erode_pixels,
        )
        return jnp.asarray(np.where(footprint, image, 0.0)), footprint

    @staticmethod
    def mtf_transform(midtones, values):
        values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        if abs(midtones - 0.5) < 1e-12:
            return values
        if midtones <= 0:
            return np.where(values <= 0, 0.0, 1.0)
        if midtones >= 1:
            return np.where(values >= 1, 1.0, 0.0)
        return ((midtones - 1.0) * values) / (((2.0 * midtones - 1.0) * values) - midtones)

    @staticmethod
    def mtf_limits(reference_image):
        reference = np.asarray(reference_image, dtype=float)
        positive = reference[np.isfinite(reference) & (reference > 0)]
        if positive.size == 0:
            return 0.0, 1.0
        vmin = float(np.nanpercentile(positive, 1))
        vmax = float(np.nanpercentile(positive, 99.5))
        if not np.isfinite(vmin):
            vmin = float(np.nanmin(positive))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + max(abs(vmin), 1.0) * 1e-6
        return vmin, vmax

    @staticmethod
    def mtf_full_range_limits(reference_image):
        reference = np.asarray(reference_image, dtype=float)
        finite = reference[np.isfinite(reference)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + max(abs(vmin), 1.0) * 1e-6
        return vmin, vmax

    @staticmethod
    def mtf_midtones_for_target(value, target=0.25):
        value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
        target = float(np.clip(target, 1e-6, 1.0 - 1e-6))
        midtones = value * (1.0 - target) / (target + value - 2.0 * target * value)
        return float(np.clip(midtones, 1e-6, 1.0 - 1e-6))

    @staticmethod
    def mtf_median_target_params(image, target=0.25):
        image = np.asarray(image, dtype=float)
        vmin, vmax = Plot.mtf_full_range_limits(image)
        scaled = (image - vmin) / max(vmax - vmin, 1e-12)
        finite_scaled = scaled[np.isfinite(scaled)]
        median_scaled = float(np.nanmedian(finite_scaled)) if finite_scaled.size else 0.5
        midtones = Plot.mtf_midtones_for_target(median_scaled, target=target)
        return vmin, vmax, {'shadows': 0.0, 'midtones': midtones, 'highlights': 1.0}

    @staticmethod
    def mtf_median_quarter_params(image):
        return Plot.mtf_median_target_params(image, target=0.25)

    @staticmethod
    def mtf_stretch_median_quarter(image):
        vmin, vmax, mtf = Plot.mtf_median_quarter_params(image)
        return Plot.mtf_stretch(image, vmin, vmax, mtf)

    @staticmethod
    def mtf_stretch_with_params(image, mtf_params):
        vmin, vmax, mtf = mtf_params
        return Plot.mtf_stretch(
            image,
            vmin,
            vmax,
            mtf,
        )

    @staticmethod
    def mtf_stretch(image, vmin, vmax, mtf=None):
        mtf = mtf or {}
        shadows = float(mtf.get('shadows', 0.0))
        midtones = float(mtf.get('midtones', 0.5))
        highlights = float(mtf.get('highlights', 1.0))
        if highlights <= shadows + 0.005:
            highlights = min(1.0, shadows + 0.005)
        image = np.asarray(image, dtype=float)
        width = max(vmax - vmin, 1e-12)
        scaled = (image - vmin) / width
        clipped = np.clip((scaled - shadows) / max(highlights - shadows, 1e-6), 0.0, 1.0)
        stretched = Plot.mtf_transform(midtones, clipped)
        return np.ma.array(stretched, mask=~np.isfinite(image))

    @staticmethod
    def imshow_lognorm(ax, image, *, origin='lower', extent=None, cmap='twilight'):
        return ax.imshow(np.asarray(image), origin=origin, extent=extent, cmap=cmap, norm=colors.LogNorm())

    @staticmethod
    def source_display_mode(value):
        value = str(value or 'linear').strip().lower()
        return 'log' if value in {'log', 'logarithmic', 'lognorm'} else 'linear'

    @staticmethod
    def imshow_source(ax, image, *, origin='lower', extent=None, cmap='twilight', source_display='linear'):
        arr = np.asarray(image, dtype=float)
        if Plot.source_display_mode(source_display) != 'log':
            return ax.imshow(arr, origin=origin, extent=extent, cmap=cmap)
        finite_positive = np.isfinite(arr) & (arr > 0)
        if not np.any(finite_positive):
            return ax.imshow(arr, origin=origin, extent=extent, cmap=cmap)
        positive_values = arr[finite_positive]
        vmin = max(float(np.nanmin(positive_values)), 1e-12)
        vmax = float(np.nanmax(positive_values))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 1.01
        masked = np.ma.array(arr, mask=~finite_positive)
        return ax.imshow(masked, origin=origin, extent=extent, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax))

    @staticmethod
    def residual_imshow_params(image, display_mask=None, percentile=99, fallback=3.0):
        arr = np.asarray(image, dtype=float)
        finite = np.isfinite(arr)
        if display_mask is not None:
            display = np.asarray(display_mask, dtype=bool)
            if display.shape != arr.shape:
                raise ValueError(f"Residual display mask shape {display.shape} does not match image shape {arr.shape}.")
            finite = finite & display
        limit = float(fallback)
        return np.ma.array(arr, mask=~finite), -limit, limit

    @staticmethod
    def residual_cmap():
        cmap = plt.get_cmap('bwr').copy()
        cmap.set_bad(color='white', alpha=1.0)
        return cmap

    @staticmethod
    def format_tick_one_decimal(value, pos=None):
        text = f'{float(value):.1f}'.rstrip('0').rstrip('.')
        return '0' if text == '-0' else text

    @staticmethod
    def style_lens_panel(ax, title, title_fontsize=18, tick_labelsize=12):
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=5)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=3))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=3))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(Plot.format_tick_one_decimal))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(Plot.format_tick_one_decimal))
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=tick_labelsize, length=6, width=1.8)
        ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=3, width=1.2)

    @staticmethod
    def add_top_colorbar(fig, ax, image, tick_labelsize=11):
        cax = ax.inset_axes([0.0, 1.20, 1.0, 0.07], transform=ax.transAxes)
        cax.set_in_layout(False)
        cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=tick_labelsize, length=4, width=1.0)
        return cbar

    @staticmethod
    def add_top_colorbar_from_position(fig, ax, image, tick_labelsize=11, pad=0.04, height=0.022):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y1 + pad, pos.width, height])
        cax.set_in_layout(False)
        cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(
            direction='in',
            which='both',
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
            labelsize=tick_labelsize,
            length=3,
            width=0.8,
            pad=1,
        )
        return cbar

    @staticmethod
    def points_array(points):
        if points is None:
            return None
        arr = np.asarray(jax.device_get(points), dtype=float)
        if arr.size == 0:
            return None
        return arr.reshape(-1, 2)

    @staticmethod
    def trace_points_to_source_plane(lens_image, kwargs, points, N=1):
        arr = Plot.points_array(points)
        if arr is None or arr.shape[0] == 0:
            return None
        x = jnp.asarray(arr[:, 0], dtype=jnp.float64)
        y = jnp.asarray(arr[:, 1], dtype=jnp.float64)
        if 'eta_flat' in kwargs and 'kwargs_mass' in kwargs and hasattr(lens_image, 'MPMassModel'):
            ra_planes, dec_planes = lens_image.MPMassModel.ray_shooting(
                x,
                y,
                kwargs['eta_flat'],
                kwargs['kwargs_mass'],
            )
            return np.column_stack([
                np.asarray(jax.device_get(ra_planes[N]), dtype=float),
                np.asarray(jax.device_get(dec_planes[N]), dtype=float),
            ])
        if 'kwargs_lens' in kwargs and hasattr(lens_image, 'MassModel'):
            src_x, src_y = lens_image.MassModel.ray_shooting(
                x,
                y,
                kwargs['kwargs_lens'],
            )
            return np.column_stack([
                np.asarray(jax.device_get(src_x), dtype=float),
                np.asarray(jax.device_get(src_y), dtype=float),
            ])
        return None

    @staticmethod
    def plot_source_points(
        ax,
        points,
        color,
        markersize=2,
        extent=None,
        image_shape=None,
        pixel_fraction=0.5,
        labels=None,
        show_centroid=False,
    ):
        arr = Plot.points_array(points)
        if arr is None or arr.shape[0] == 0:
            return
        ax.plot(arr[:, 0], arr[:, 1], 'o', color=color, markersize=markersize, markeredgewidth=0)
        if labels is not None:
            label_offsets = ((7, -13), (7, 2), (-18, 8))
            for index, (x_point, y_point) in enumerate(arr):
                label = labels[index] if index < len(labels) else str(index + 1)
                offset = label_offsets[index] if index < len(label_offsets) else (6, 6)
                ax.annotate(
                    label,
                    (x_point, y_point),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=8,
                    color='black',
                    bbox={'boxstyle': 'round,pad=0.15', 'facecolor': 'white', 'edgecolor': '0.35', 'alpha': 0.8},
                    zorder=8,
                )
        if show_centroid and arr.shape[0] > 1:
            centroid = np.mean(arr, axis=0)
            ax.plot(
                centroid[0],
                centroid[1],
                marker='*',
                color='#35a853',
                markeredgecolor='black',
                markeredgewidth=0.5,
                markersize=max(7, markersize + 4),
                zorder=8,
            )

    @staticmethod
    def single_plane_caustics(lens_image, kwargs, supersampling=6):
        kwargs_lens = kwargs.get('kwargs_lens') if isinstance(kwargs, dict) else None
        if kwargs_lens is None:
            return []
        try:
            _, caustics = model_util.critical_lines_caustics(
                lens_image,
                kwargs_lens,
                supersampling=supersampling,
            )
            return caustics
        except Exception as exc:
            warnings.warn(f'Could not compute caustics for source preview: {exc}', RuntimeWarning)
            return []

    @staticmethod
    def multiplane_caustics(lens_image, kwargs, source_plane, supersampling=6):
        if not isinstance(kwargs, dict):
            return []
        eta_flat = kwargs.get('eta_flat')
        kwargs_mass = kwargs.get('kwargs_mass')
        if eta_flat is None or kwargs_mass is None:
            return []
        try:
            _, caustics = model_util.critical_lines_caustics(
                lens_image,
                kwargs_mass,
                eta_flat=eta_flat,
                k_plane=int(source_plane),
                supersampling=supersampling,
            )
            return caustics
        except Exception as exc:
            warnings.warn(
                f'Could not compute caustics for source plane {source_plane}: {exc}',
                RuntimeWarning,
            )
            return []

    @staticmethod
    def sis_outer_pseudo_caustics(
        lens_image,
        kwargs,
        mass_plane,
    ):
        """Return the SIS outer pseudo-caustic in the lens plane's own coordinates.

        This intentionally excludes the foreground main lens and eta scaling:
        the curve marks where the next source plane is multiply imaged by this SIS.
        """
        if not isinstance(kwargs, dict):
            return []
        kwargs_mass = kwargs.get('kwargs_mass')
        if kwargs_mass is None or mass_plane >= len(kwargs_mass):
            return []
        try:
            mass_model = lens_image.MPMassModel.mass_models[mass_plane]
            kwargs_plane = kwargs_mass[mass_plane]
            profile_types = getattr(mass_model, 'profile_type_list', [])
            pseudo_caustics = []
            angle = np.linspace(0.0, 2.0 * np.pi, 361)
            for profile_type, component in zip(profile_types, kwargs_plane):
                if str(profile_type).upper() != 'SIS':
                    continue
                center_x = float(np.asarray(component.get('center_x', 0.0)).reshape(-1)[0])
                center_y = float(np.asarray(component.get('center_y', 0.0)).reshape(-1)[0])
                theta_e = abs(float(np.asarray(component['theta_E']).reshape(-1)[0]))
                pseudo_caustics.append((
                    center_x + theta_e * np.cos(angle),
                    center_y + theta_e * np.sin(angle),
                ))
            return pseudo_caustics
        except Exception as exc:
            warnings.warn(
                f'Could not compute SIS outer pseudo-caustics for mass plane {mass_plane}: {exc}',
                RuntimeWarning,
            )
            return []

    @staticmethod
    def plot_caustics(ax, caustics, color='#ff3300', linestyle='--', linewidth=1.5, alpha=0.95):
        for curve in caustics or []:
            if curve is None or len(curve) < 2:
                continue
            x_curve = np.asarray(jax.device_get(curve[0]), dtype=float)
            y_curve = np.asarray(jax.device_get(curve[1]), dtype=float)
            finite = np.isfinite(x_curve) & np.isfinite(y_curve)
            if not np.any(finite):
                continue
            x_curve = x_curve[finite]
            y_curve = y_curve[finite]
            panel_span = max(abs(np.diff(ax.get_xlim()))[0], abs(np.diff(ax.get_ylim()))[0], 1e-12)
            curve_span = max(float(np.ptp(x_curve)), float(np.ptp(y_curve)))
            if curve_span < 0.005 * panel_span:
                ax.plot(
                    float(np.mean(x_curve)),
                    float(np.mean(y_curve)),
                    marker='x',
                    color=color,
                    markersize=4,
                    markeredgewidth=linewidth,
                    alpha=alpha,
                    zorder=5,
                )
                continue
            ax.plot(
                x_curve,
                y_curve,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                zorder=5,
            )

    @staticmethod
    def set_panel_extent_limits(ax, extent):
        if extent is None or len(extent) < 4:
            return
        ax.set_xlim(float(extent[0]), float(extent[1]))
        ax.set_ylim(float(extent[2]), float(extent[3]))

    @staticmethod
    def visualize_model_four_panel(
        lens_image,
        kwargs,
        data_obs,
        fit_mask,
        background_rms,
        pixel_grid_shape,
        source_grid_scale,
        image_extent,
        title,
        mtf=None,
        source_display='linear',
    ):
        model_image = lens_image.model(**kwargs)
        lens_light_image = Plot.lens_light_image(lens_image, kwargs)
        lensed_source_image = lens_image.model(
            kwargs_lens=kwargs.get('kwargs_lens'),
            kwargs_source=kwargs.get('kwargs_source'),
            kwargs_lens_light=kwargs.get('kwargs_lens_light'),
            kwargs_point_source=kwargs.get('kwargs_point_source'),
            source_add=True,
            lens_light_add=False,
            point_source_add=True,
        )
        noise = jnp.asarray(background_rms)
        if noise.ndim == 0:
            model_var = lens_image.Noise.C_D_model(model_image, background_rms=float(noise))
            noise = jnp.sqrt(jnp.maximum(model_var, 1e-12))
        else:
            noise = jnp.sqrt(jnp.maximum(noise**2, 1e-12))
        residual = (jnp.asarray(data_obs) - model_image) / noise
        lens_light_subtracted = jnp.asarray(data_obs) - lens_light_image
        source_image, source_extent = Plot.pixelize_plane(
            lens_image,
            kwargs,
            pixel_grid_shape,
            source_grid_scale=source_grid_scale,
        )

        data_panel = np.asarray(data_obs)
        model_panel = np.asarray(model_image)
        lens_light_subtracted_panel = np.asarray(lens_light_subtracted)
        lensed_source_panel = np.asarray(lensed_source_image)
        residual_panel = np.asarray(residual)
        source_panel = np.asarray(source_image)

        residual_plot, residual_vmin, residual_vmax = Plot.residual_imshow_params(residual_panel, display_mask=fit_mask)

        data_mtf_params = Plot.mtf_median_target_params(data_panel, target=0.125)
        lens_light_mtf_params = Plot.mtf_median_quarter_params(lens_light_subtracted_panel)

        fig, axes = plt.subplots(1, 6, figsize=(24, 4.2), constrained_layout=False)
        axes[0].imshow(Plot.mtf_stretch_with_params(data_panel, data_mtf_params), origin='lower', extent=image_extent, cmap='twilight', vmin=0, vmax=1)
        Plot.style_lens_panel(axes[0], 'Data')
        axes[1].imshow(Plot.mtf_stretch_with_params(model_panel, data_mtf_params), origin='lower', extent=image_extent, cmap='twilight', vmin=0, vmax=1)
        Plot.style_lens_panel(axes[1], 'LensModel')
        residual_artist = axes[2].imshow(
            residual_plot,
            origin='lower',
            extent=image_extent,
            cmap=Plot.residual_cmap(),
            vmin=residual_vmin,
            vmax=residual_vmax,
        )
        Plot.style_lens_panel(axes[2], 'Residual')
        Plot.imshow_source(axes[3], source_panel, origin='lower', extent=source_extent, cmap='twilight', source_display=source_display)
        Plot.plot_caustics(axes[3], Plot.single_plane_caustics(lens_image, kwargs))
        traced_points = Plot.trace_points_to_source_plane(
            lens_image,
            kwargs,
            getattr(lens_image, 'conjugate_points', None),
            N=1,
        )
        source_point_labels = (
            [chr(ord('A') + index) for index in range(len(traced_points))]
            if traced_points is not None
            else None
        )
        Plot.plot_source_points(
            axes[3],
            traced_points,
            'black',
            markersize=3,
            extent=source_extent,
            image_shape=source_panel.shape,
            labels=source_point_labels,
            show_centroid=True,
        )
        Plot.set_panel_extent_limits(axes[3], source_extent)
        Plot.style_lens_panel(axes[3], 'Source')
        axes[4].imshow(Plot.mtf_stretch_with_params(lens_light_subtracted_panel, lens_light_mtf_params), origin='lower', extent=image_extent, cmap='twilight', vmin=0, vmax=1)
        Plot.style_lens_panel(axes[4], 'Lens Light subtracted', title_fontsize=15)
        axes[5].imshow(Plot.mtf_stretch_with_params(lensed_source_panel, lens_light_mtf_params), origin='lower', extent=image_extent, cmap='twilight', vmin=0, vmax=1)
        Plot.style_lens_panel(axes[5], 'Lensed source without lens light', title_fontsize=15)

        if title:
            fig.suptitle(title, y=0.97, fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.88], pad=0.25, w_pad=0.45, h_pad=0.2)
        Plot.add_top_colorbar_from_position(fig, axes[2], residual_artist, pad=0.07)
        return fig

    @staticmethod
    def visualize_dspl_model(
        lens_image,
        kwargs,
        data_obs,
        fit_mask,
        background_rms,
        pixel_grid_shape_s1,
        pixel_grid_shape_s2,
        source_grid_scale_s1,
        source_grid_scale_s2,
        image_extent,
        title,
        show_lensed_arcs=False,
        apply_lensed_arc_mask=True,
        mtf=None,
        conjugate_points_source1=None,
        conjugate_points_source2=None,
        source_display='linear',
        show_caustics=True,
    ):
        model_image = lens_image.model(
            **kwargs,
            apply_mask=apply_lensed_arc_mask,
        )
        lens_light_image = Plot.lens_light_image(lens_image, kwargs)
        noise = jnp.asarray(background_rms)
        if noise.ndim == 0:
            model_var = lens_image.Noise.C_D_model(model_image, background_rms=float(noise))
            noise = jnp.sqrt(jnp.maximum(model_var, 1e-12))
        else:
            noise = jnp.sqrt(jnp.maximum(noise**2, 1e-12))
        residual = (jnp.asarray(data_obs) - model_image) / noise
        lens_light_subtracted = jnp.asarray(data_obs) - lens_light_image
        source1_image, source1_extent = Plot.pixelize_plane_multiplane(
            lens_image,
            kwargs,
            pixel_grid_shape_s1,
            source_grid_scale=source_grid_scale_s1,
            N=1,
        )
        source2_image, source2_extent = Plot.pixelize_plane_multiplane(
            lens_image,
            kwargs,
            pixel_grid_shape_s2,
            source_grid_scale=source_grid_scale_s2,
            N=2,
        )
        if show_lensed_arcs:
            lensed_arc1 = Plot.lensed_arc_image(lens_image, kwargs, 1, apply_mask=apply_lensed_arc_mask)
            lensed_arc2 = Plot.lensed_arc_image(lens_image, kwargs, 2, apply_mask=apply_lensed_arc_mask)

        if show_lensed_arcs:
            fig, axes_grid = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=False)
            axes = list(axes_grid[0, :])
            arc_axes = list(axes_grid[1, :])
        else:
            fig, axes = plt.subplots(1, 6, figsize=(20, 4.2), constrained_layout=False)
            arc_axes = []
        image_panels = [
            (np.asarray(data_obs), image_extent, 'data', 'twilight'),
            (np.asarray(lens_light_subtracted), image_extent, 'data - lens light', 'twilight'),
            (np.asarray(model_image), image_extent, 'model', 'twilight'),
            (np.asarray(residual), image_extent, 'data - model / rms', 'bwr'),
        ]
        source_panels = [
            (np.asarray(source1_image), source1_extent, 'source 1', 'twilight'),
            (np.asarray(source2_image), source2_extent, 'source 2', 'twilight'),
        ]
        if show_lensed_arcs:
            panels = [
                *image_panels,
                (np.asarray(lensed_arc1), image_extent, 'lensed arc 1', 'twilight'),
                (np.asarray(lensed_arc2), image_extent, 'lensed arc 2', 'twilight'),
                *source_panels,
            ]
        else:
            panels = [*image_panels, *source_panels]
        image_plane_titles = {'data', 'data - lens light', 'model', 'lensed arc 1', 'lensed arc 2'}
        data_mtf_params = Plot.mtf_median_target_params(np.asarray(data_obs), target=0.125)
        lens_light_mtf_params = Plot.mtf_median_quarter_params(np.asarray(lens_light_subtracted))
        title_map = {
            'data': 'Data',
            'data - lens light': 'Lens light subtracted',
            'model': 'Model',
            'data - model / rms': 'Residual',
            'lensed arc 1': 'Lensed arc 1',
            'lensed arc 2': 'Lensed arc 2',
            'source 1': 'Source1',
            'source 2': 'Source 2',
        }
        traced_source1_points = Plot.trace_points_to_source_plane(
            lens_image,
            kwargs,
            conjugate_points_source1,
            N=1,
        )
        traced_source2_points = Plot.trace_points_to_source_plane(
            lens_image,
            kwargs,
            conjugate_points_source2,
            N=2,
        )
        if show_caustics:
            source1_caustics = Plot.multiplane_caustics(lens_image, kwargs, source_plane=1)
            source2_caustics = Plot.sis_outer_pseudo_caustics(
                lens_image,
                kwargs,
                mass_plane=1,
            )
        else:
            source1_caustics = []
            source2_caustics = []

        residual_ax = None
        residual_artist = None
        for ax, (image, extent, panel_title, cmap) in zip([*axes, *arc_axes], panels):
            if panel_title == 'data - model / rms':
                residual_plot, residual_vmin, residual_vmax = Plot.residual_imshow_params(image, display_mask=fit_mask)
                residual_artist = ax.imshow(residual_plot, origin='lower', extent=extent, cmap=Plot.residual_cmap(), vmin=residual_vmin, vmax=residual_vmax)
                Plot.style_lens_panel(ax, title_map.get(panel_title, panel_title))
                residual_ax = ax
                continue
            elif panel_title in {'data', 'model'}:
                ax.imshow(Plot.mtf_stretch_with_params(image, data_mtf_params), origin='lower', extent=extent, cmap=cmap, vmin=0, vmax=1)
            elif panel_title in image_plane_titles:
                ax.imshow(Plot.mtf_stretch_with_params(image, lens_light_mtf_params), origin='lower', extent=extent, cmap=cmap, vmin=0, vmax=1)
            elif panel_title in {'source 1', 'source 2'}:
                Plot.imshow_source(ax, image, origin='lower', extent=extent, cmap=cmap, source_display=source_display)
            else:
                ax.imshow(image, origin='lower', extent=extent, cmap=cmap)
            if panel_title == 'source 1':
                Plot.plot_caustics(ax, source1_caustics)
                Plot.plot_source_points(ax, traced_source1_points, 'black', extent=extent, image_shape=image.shape)
            if panel_title == 'source 2':
                Plot.plot_caustics(ax, source2_caustics)
                Plot.plot_source_points(ax, traced_source2_points, 'C1', extent=extent, image_shape=image.shape)
            if panel_title in {'source 1', 'source 2'}:
                Plot.set_panel_extent_limits(ax, extent)
            Plot.style_lens_panel(ax, title_map.get(panel_title, panel_title))
        if title:
            fig.suptitle(title, y=0.97, fontsize=18)
        if show_lensed_arcs:
            fig.tight_layout(rect=[0, 0, 1, 0.93], pad=0.25, w_pad=0.02, h_pad=1.1)
        else:
            fig.tight_layout(rect=[0, 0, 1, 0.88], pad=0.25, w_pad=0.12, h_pad=0.2)
        if residual_ax is not None and residual_artist is not None:
            Plot.add_top_colorbar_from_position(fig, residual_ax, residual_artist)
        return fig


class LensResultWriter:
    @staticmethod
    def svi_tail_mean_loss(state, tail=1000):
        losses = np.asarray(state["losses"], dtype=float)
        if losses.size == 0:
            return np.inf
        tail_size = min(int(tail), losses.size)
        return float(np.nanmean(losses[-tail_size:]))

    @staticmethod
    def best_svi_chain_index(states):
        return int(np.nanargmin([
            LensResultWriter.svi_tail_mean_loss(state)
            for state in states
        ]))

    @staticmethod
    def best_svi_chain_index_by_residual(lens_image, kwargs_list, data, fit_mask, states, rms_map, rms):
        """Select the median model with the lowest normalized residual RMS."""
        scores = []
        data = jnp.asarray(data)
        fit_mask = jnp.asarray(fit_mask, dtype=bool)
        for kwargs, state in zip(kwargs_list, states):
            model = lens_image.model(**kwargs)
            noise = LensResultWriter.residual_noise_from_state(state, rms_map, rms)
            noise = jnp.asarray(noise)
            if noise.ndim == 0:
                variance = lens_image.Noise.C_D_model(model, background_rms=noise)
                noise = jnp.sqrt(jnp.maximum(variance, 1e-12))
            residual = (data - model) / jnp.maximum(noise, 1e-12)
            values = residual[fit_mask]
            score = float(jax.device_get(jnp.sqrt(jnp.mean(values ** 2))))
            scores.append(score if np.isfinite(score) else np.inf)
        best = int(np.argmin(scores))
        print(f"Residual-RMS chain scores: {scores}; selected chain {best + 1}.", flush=True)
        return best

    @staticmethod
    def residual_noise_from_state(state, rms_map, rms):
        if rms_map is not None:
            return rms_map
        return float(state["median"].get("RMS", rms))

    @staticmethod
    def json_value(value):
        array = np.asarray(jax.device_get(value)).reshape(-1)
        if array.size == 1:
            return float(array[0])
        return [float(item) for item in array]

    @staticmethod
    def single_plane_mass_parameter_summary(kwargs):
        epl = kwargs["kwargs_lens"][0]
        shear = kwargs["kwargs_lens"][1]
        summary = {
            "theta_E": LensResultWriter.json_value(epl["theta_E"]),
            "gamma": LensResultWriter.json_value(epl["gamma"]),
            "e1": LensResultWriter.json_value(epl["e1"]),
            "e2": LensResultWriter.json_value(epl["e2"]),
            "center_x": LensResultWriter.json_value(epl["center_x"]),
            "center_y": LensResultWriter.json_value(epl["center_y"]),
            "gamma1_ext": LensResultWriter.json_value(shear["gamma1"]),
            "gamma2_ext": LensResultWriter.json_value(shear["gamma2"]),
        }
        for index, component in enumerate(kwargs["kwargs_lens"][2:], start=1):
            if "theta_E" not in component:
                prefix = f"FIXED_MASS_MAP_{index}"
                func_pixels = component.get("func_pixels")
                summary[f"{prefix}_fixed"] = True
                if func_pixels is not None:
                    summary[f"{prefix}_shape"] = list(
                        np.asarray(jax.device_get(func_pixels)).shape
                    )
                continue
            profile = "SIE" if "e1" in component else "SIS"
            prefix = f"{profile}_{index}"
            summary[f"{prefix}_theta_E"] = LensResultWriter.json_value(component["theta_E"])
            summary[f"{prefix}_center_x"] = LensResultWriter.json_value(component["center_x"])
            summary[f"{prefix}_center_y"] = LensResultWriter.json_value(component["center_y"])
            if profile == "SIE":
                summary[f"{prefix}_e1"] = LensResultWriter.json_value(component["e1"])
                summary[f"{prefix}_e2"] = LensResultWriter.json_value(component["e2"])
        return summary

    @staticmethod
    def save_loss_comparison(
        output_dir,
        model_config,
        parametric_states,
        pixelated_states,
        selected_chain_index,
        stage2_states=None,
    ):
        """Plot one panel per SVI stage.

        ``pixelated_states`` always carries the *final* stage, so when the exact-solver
        stage ran it is passed here and the plain pixelated stage arrives as
        ``stage2_states``.  The panels are not on a common scale: the solver stage
        replaces the conjugate-point likelihood with a constrained parameterisation plus
        a Jacobian term, so its ELBO is not comparable with the earlier stages.
        """
        output_dir = Path(output_dir)
        max_iter_solver = model_config["svi"].get("max_iter_solver", model_config["svi"]["max_iter_pixelated"])
        ncols = 2 if stage2_states is None else 3
        fig, axes = plt.subplots(1, ncols, figsize=(7.5*ncols, 4.2), constrained_layout=True)
        stages = [
            (axes[0], "Parametric SVI loss", parametric_states, model_config["svi"]["max_iter_parametric"], "tab:blue"),
        ]
        if stage2_states is None:
            stages.append(
                (axes[1], "Pixelated SVI loss", pixelated_states, model_config["svi"]["max_iter_pixelated"], "tab:orange")
            )
        else:
            stages.append(
                (axes[1], "Pixelated SVI loss", stage2_states, model_config["svi"]["max_iter_pixelated"], "tab:orange")
            )
            stages.append(
                (axes[2], "Exact-solver SVI loss (not comparable)", pixelated_states, max_iter_solver, "tab:green")
            )
        for ax, title, states, max_iter, selected_color in stages:
            for chain_index, state in enumerate(states):
                is_selected = chain_index == selected_chain_index
                Plot.plot_loss(
                    np.asarray(state["losses"]),
                    max_iter,
                    ax=ax,
                    inset=False,
                    color=selected_color if is_selected else "0.55",
                    alpha=0.95 if is_selected else 0.5,
                    linewidth=2.0 if is_selected else 1.0,
                    label="Chain " + str(chain_index + 1) + (" selected" if is_selected else ""),
                )
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
        output = output_dir / "losses_comparison.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output

    @staticmethod
    def write_panel_fits(project_root, path, image, extent=None, panel_name=""):
        project_root = Path(project_root)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(jax.device_get(image), dtype=np.float32)
        array = np.squeeze(array)
        header = fits.Header()
        header["PANEL"] = (panel_name[:68], "Lens model panel")
        if extent is not None and len(extent) >= 4:
            header["XMIN"] = float(extent[0])
            header["XMAX"] = float(extent[1])
            header["YMIN"] = float(extent[2])
            header["YMAX"] = float(extent[3])
            header["CTYPE1"] = "arcsec"
            header["CTYPE2"] = "arcsec"
        fits.writeto(path, array, header=header, overwrite=True)
        return "/" + path.relative_to(project_root).as_posix()

    @staticmethod
    def save_single_panel_fits(
        project_root,
        chain_dir,
        lens_image,
        kwargs,
        data_obs,
        fit_mask,
        background_rms,
        pixel_grid_shape,
        source_grid_scale,
        image_extent,
        stage_name="pixelated",
    ):
        panel_dir = Path(chain_dir) / stage_name
        model_image = lens_image.model(**kwargs)
        lens_light_image = Plot.lens_light_image(lens_image, kwargs)
        lensed_source_image = lens_image.model(
            kwargs_lens=kwargs.get('kwargs_lens'),
            kwargs_source=kwargs.get('kwargs_source'),
            kwargs_lens_light=kwargs.get('kwargs_lens_light'),
            kwargs_point_source=kwargs.get('kwargs_point_source'),
            source_add=True,
            lens_light_add=False,
            point_source_add=True,
        )
        noise = jnp.asarray(background_rms)
        if noise.ndim == 0:
            model_var = lens_image.Noise.C_D_model(model_image, background_rms=float(noise))
            noise = jnp.sqrt(jnp.maximum(model_var, 1e-12))
        else:
            noise = jnp.sqrt(jnp.maximum(noise**2, 1e-12))
        residual = (jnp.asarray(data_obs) - model_image) / noise
        lens_light_subtracted = jnp.asarray(data_obs) - lens_light_image
        source_image, source_extent = Plot.pixelize_plane(
            lens_image,
            kwargs,
            pixel_grid_shape,
            source_grid_scale=source_grid_scale,
        )
        return {
            "data": LensResultWriter.write_panel_fits(project_root, panel_dir / "data.fits", data_obs, image_extent, "data"),
            "model": LensResultWriter.write_panel_fits(project_root, panel_dir / "model.fits", model_image, image_extent, "model"),
            "data_minus_model_over_rms": LensResultWriter.write_panel_fits(project_root, panel_dir / "data_minus_model_over_rms.fits", residual, image_extent, "data - model / rms"),
            "source": LensResultWriter.write_panel_fits(project_root, panel_dir / "source.fits", source_image, source_extent, "source image"),
            "data_minus_lens_light": LensResultWriter.write_panel_fits(project_root, panel_dir / "data_minus_lens_light.fits", lens_light_subtracted, image_extent, "data - lens light"),
            "lensed_source_without_lens_light": LensResultWriter.write_panel_fits(
                project_root,
                panel_dir / "lensed_source_without_lens_light.fits",
                lensed_source_image,
                image_extent,
                "lensed source without lens light",
            ),
        }

    @staticmethod
    def compose_lens_preview(output_path, panels):
        images = [(title, Image.open(path).convert("RGB")) for title, path in panels]
        max_width = max(image.width for _, image in images)
        pad_x = 40
        pad_y = 22
        title_h = 38
        gap = 22
        canvas_width = max_width + 2 * pad_x
        canvas_height = 2 * pad_y + sum(image.height for _, image in images) + len(images) * title_h + (len(images) - 1) * gap
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 34)
        except OSError:
            font = ImageFont.load_default()
        y = pad_y
        for title, image in images:
            bbox = draw.textbbox((0, 0), title, font=font)
            draw.text(((canvas_width - (bbox[2] - bbox[0])) / 2, y), title, fill="black", font=font)
            y += title_h
            canvas.paste(image, ((canvas_width - image.width) // 2, y))
            y += image.height + gap
        canvas.save(output_path)

    @staticmethod
    def save_single_plane_svi_outputs(
        *,
        project_root,
        run_output_dir,
        gui_status,
        run_timer,
        model_config,
        num_chains,
        lens_image_parametric,
        lens_image_pixelated,
        parametric_states,
        parametric_kwargs_list,
        pixelated_states,
        pixelated_kwargs_list,
        data,
        fit_mask,
        rms_map,
        rms,
        pixel_grid_shape,
        image_extent,
        data_file,
        mask_1_file,
        mask_out_file,
        psf_file,
        rms_file,
        rms_prior_low,
        rms_prior_high,
        background_offset,
        semilinear_result=None,
        semilinear_source_path=None,
        stage2_states=None,
    ):
        project_root = Path(project_root)
        run_output_dir = Path(run_output_dir)
        data_file = Path(data_file)
        mask_1_file = Path(mask_1_file)
        mask_out_file = Path(mask_out_file)
        psf_file = Path(psf_file)
        rms_file = Path(rms_file)

        if model_config.get("chain_selection") == "residual_rms":
            best_chain_index = LensResultWriter.best_svi_chain_index_by_residual(
                lens_image_pixelated,
                pixelated_kwargs_list,
                data,
                fit_mask,
                pixelated_states,
                rms_map,
                rms,
            )
        else:
            best_chain_index = LensResultWriter.best_svi_chain_index(pixelated_states)
        parametric_state = parametric_states[best_chain_index]
        parametric_kwargs = parametric_kwargs_list[best_chain_index]
        pixelated_state = pixelated_states[best_chain_index]
        pixelated_kwargs = pixelated_kwargs_list[best_chain_index]
        print(f"Selected chain {best_chain_index + 1}/{num_chains} for visualization.", flush=True)

        # Save completed SVI products before visualization, so plotting errors do not discard the chain.
        np.save(run_output_dir / "parametric_losses.npy", np.asarray([state["losses"] for state in parametric_states]))
        np.save(run_output_dir / "pixelated_losses.npy", np.asarray([state["losses"] for state in pixelated_states]))
        np.save(run_output_dir / "selected_chain_index.npy", np.asarray(best_chain_index))
        with open(run_output_dir / "parametric_state.pkl", "wb") as f:
            pickle.dump(parametric_state, f)
        with open(run_output_dir / "pixelated_state.pkl", "wb") as f:
            pickle.dump(pixelated_state, f)
        with open(run_output_dir / "parametric_states.pkl", "wb") as f:
            pickle.dump(parametric_states, f)
        with open(run_output_dir / "pixelated_states.pkl", "wb") as f:
            pickle.dump(pixelated_states, f)
        with open(run_output_dir / "pixelated_kwargs.pkl", "wb") as f:
            pickle.dump(pixelated_kwargs, f)

        losses_comparison_png = LensResultWriter.save_loss_comparison(
            run_output_dir,
            model_config,
            parametric_states,
            pixelated_states,
            best_chain_index,
            stage2_states=stage2_states,
        )

        parametric_fig = Plot.visualize_model_four_panel(
            lens_image_parametric,
            parametric_kwargs,
            data,
            fit_mask,
            LensResultWriter.residual_noise_from_state(parametric_state, rms_map, rms),
            pixel_grid_shape,
            model_config["source_grid"]["scale"],
            image_extent,
            title="parametric source",
            mtf=model_config["display"]["mtf"],
            source_display=model_config.get("display", {}).get("source_display", "linear"),
        )
        pixelated_fig = Plot.visualize_model_four_panel(
            lens_image_pixelated,
            pixelated_kwargs,
            data,
            fit_mask,
            LensResultWriter.residual_noise_from_state(pixelated_state, rms_map, rms),
            pixel_grid_shape,
            model_config["source_grid"]["scale"],
            image_extent,
            title="",
            mtf=model_config["display"]["mtf"],
            source_display=model_config.get("display", {}).get("source_display", "linear"),
        )
        semilinear_kwargs = None
        semilinear_fig = None
        semilinear_summary = None
        if semilinear_result is not None:
            semilinear_kwargs = semilinear_result.kwargs
            semilinear_summary = semilinear_result.summary()
            semilinear_summary["chain"] = int(best_chain_index + 1)
            semilinear_fig = Plot.visualize_model_four_panel(
                lens_image_pixelated,
                semilinear_kwargs,
                data,
                fit_mask,
                LensResultWriter.residual_noise_from_state(pixelated_state, rms_map, rms),
                pixel_grid_shape,
                model_config["source_grid"]["scale"],
                image_extent,
                title="semilinear source",
                mtf=model_config["display"]["mtf"],
                source_display=model_config.get("display", {}).get("source_display", "linear"),
            )

        run_summary = {
            "run_output_dir": str(run_output_dir),
            "seed": model_config["svi"]["seed"],
            "num_chains": int(num_chains),
            "selected_chain_index": int(best_chain_index),
            "max_iter_parametric": model_config["svi"]["max_iter_parametric"],
            "max_iter_pixelated": model_config["svi"]["max_iter_pixelated"],
            "pixel_grid_shape": int(pixel_grid_shape),
            "data_file": str(data_file),
            "mask_1_file": str(mask_1_file),
            "mask_out_file": str(mask_out_file),
            "psf_file": str(psf_file),
            "rms_file": str(rms_file) if rms_file.exists() else None,
            "conjugate_points_source1": model_config.get("conjugate_points_source1", []),
            "residual_noise": "scalar RMS posterior" if model_config["data"].get("use_scalar_rms_loguniform", False) else ("RMS_map.fits" if rms_map is not None else "SVI posterior RMS"),
            "rms_prior": [float(rms_prior_low), float(rms_prior_high)],
            "background_offset": float(background_offset),
            "losses_url": "/" + losses_comparison_png.relative_to(project_root).as_posix(),
            "mass_parameters": LensResultWriter.single_plane_mass_parameter_summary(pixelated_kwargs),
        }
        if semilinear_summary is not None:
            run_summary["semilinear"] = semilinear_summary
        (run_output_dir / "summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        (run_output_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")

        parametric_png = run_output_dir / "parametric_model_comparison.png"
        pixelated_png = run_output_dir / "pixelated_model_comparison.png"
        semilinear_png = (
            run_output_dir
            / f"chain_{best_chain_index + 1:02d}_semilinear_model_comparison.png"
        )
        parametric_fig.savefig(parametric_png, dpi=180, bbox_inches="tight")
        pixelated_fig.savefig(pixelated_png, dpi=180, bbox_inches="tight")
        if semilinear_fig is not None:
            semilinear_fig.savefig(
                semilinear_png,
                dpi=180,
                bbox_inches="tight",
            )

        for num in plt.get_fignums():
            fig = plt.figure(num)
            out_png = run_output_dir / f"figure_{num:02d}.png"
            fig.savefig(out_png, dpi=180, bbox_inches="tight")

        latest = run_output_dir / "latest_four_panel_comparison.png"
        latest_panels = [
            ("Parametric model", parametric_png),
            ("Pixelated SVI model", pixelated_png),
        ]
        if semilinear_fig is not None:
            latest_panels.append(("Semilinear source model", semilinear_png))
        LensResultWriter.compose_lens_preview(
            latest,
            latest_panels,
        )

        chain_previews = []
        for chain_index in range(num_chains):
            chain_tag = f"chain_{chain_index + 1:02d}"
            chain_dir = run_output_dir / chain_tag
            chain_parametric_fig = Plot.visualize_model_four_panel(
                lens_image_parametric,
                parametric_kwargs_list[chain_index],
                data,
                fit_mask,
                LensResultWriter.residual_noise_from_state(parametric_states[chain_index], rms_map, rms),
                pixel_grid_shape,
                model_config["source_grid"]["scale"],
                image_extent,
                title="parametric source",
                mtf=model_config["display"]["mtf"],
                source_display=model_config.get("display", {}).get("source_display", "linear"),
            )
            chain_pixelated_fig = Plot.visualize_model_four_panel(
                lens_image_pixelated,
                pixelated_kwargs_list[chain_index],
                data,
                fit_mask,
                LensResultWriter.residual_noise_from_state(pixelated_states[chain_index], rms_map, rms),
                pixel_grid_shape,
                model_config["source_grid"]["scale"],
                image_extent,
                title="",
                mtf=model_config["display"]["mtf"],
                source_display=model_config.get("display", {}).get("source_display", "linear"),
            )
            chain_parametric_panel_fits = LensResultWriter.save_single_panel_fits(
                project_root,
                chain_dir,
                lens_image_parametric,
                parametric_kwargs_list[chain_index],
                data,
                fit_mask,
                LensResultWriter.residual_noise_from_state(parametric_states[chain_index], rms_map, rms),
                pixel_grid_shape,
                model_config["source_grid"]["scale"],
                image_extent,
                stage_name="parametric",
            )
            chain_panel_fits = LensResultWriter.save_single_panel_fits(
                project_root,
                chain_dir,
                lens_image_pixelated,
                pixelated_kwargs_list[chain_index],
                data,
                fit_mask,
                LensResultWriter.residual_noise_from_state(pixelated_states[chain_index], rms_map, rms),
                pixel_grid_shape,
                model_config["source_grid"]["scale"],
                image_extent,
            )
            chain_semilinear_panel_fits = None
            if (
                semilinear_kwargs is not None
                and chain_index == best_chain_index
            ):
                chain_semilinear_panel_fits = (
                    LensResultWriter.save_single_panel_fits(
                        project_root,
                        chain_dir,
                        lens_image_pixelated,
                        semilinear_kwargs,
                        data,
                        fit_mask,
                        LensResultWriter.residual_noise_from_state(
                            pixelated_states[chain_index],
                            rms_map,
                            rms,
                        ),
                        pixel_grid_shape,
                        model_config["source_grid"]["scale"],
                        image_extent,
                        stage_name="semilinear",
                    )
                )
            chain_parametric_png = run_output_dir / f"{chain_tag}_parametric_model_comparison.png"
            chain_pixelated_png = run_output_dir / f"{chain_tag}_pixelated_model_comparison.png"
            chain_latest = run_output_dir / f"{chain_tag}_lens_model_comparison.png"
            chain_parametric_fig.savefig(chain_parametric_png, dpi=180, bbox_inches="tight")
            chain_pixelated_fig.savefig(chain_pixelated_png, dpi=180, bbox_inches="tight")
            chain_comparison_panels = [
                (f"Chain {chain_index + 1} parametric model", chain_parametric_png),
                (f"Chain {chain_index + 1} pixelated SVI model", chain_pixelated_png),
            ]
            if (
                semilinear_fig is not None
                and chain_index == best_chain_index
            ):
                chain_comparison_panels.append(
                    (
                        f"Chain {chain_index + 1} semilinear source model",
                        semilinear_png,
                    )
                )
            LensResultWriter.compose_lens_preview(
                chain_latest,
                chain_comparison_panels,
            )
            plt.close(chain_parametric_fig)
            plt.close(chain_pixelated_fig)
            chain_preview = {
                "chain": int(chain_index + 1),
                "selected": bool(chain_index == best_chain_index),
                "preview_url": "/" + chain_latest.relative_to(project_root).as_posix(),
                "parametric_url": "/" + chain_parametric_png.relative_to(project_root).as_posix(),
                "pixelated_url": "/" + chain_pixelated_png.relative_to(project_root).as_posix(),
                "parametric_panel_fits": chain_parametric_panel_fits,
                "pixelated_panel_fits": chain_panel_fits,
                "mass_parameters": LensResultWriter.single_plane_mass_parameter_summary(pixelated_kwargs_list[chain_index]),
                "final_parametric_loss": float(np.asarray(parametric_states[chain_index]["losses"])[-1]),
                "final_pixelated_loss": float(np.asarray(pixelated_states[chain_index]["losses"])[-1]),
            }
            if chain_semilinear_panel_fits is not None:
                chain_preview.update({
                    "semilinear_url": "/" + semilinear_png.relative_to(project_root).as_posix(),
                    "semilinear_panel_fits": chain_semilinear_panel_fits,
                })
                if semilinear_source_path is not None:
                    source_path = Path(semilinear_source_path)
                    chain_preview["semilinear_source_pixels_url"] = (
                        "/" + source_path.relative_to(project_root).as_posix()
                    )
            chain_previews.append(chain_preview)

        run_summary["chain_previews"] = chain_previews
        if semilinear_summary is not None:
            run_summary["semilinear"].update({
                "figure_url": "/" + semilinear_png.relative_to(project_root).as_posix(),
                "panel_fits": chain_previews[best_chain_index].get(
                    "semilinear_panel_fits",
                    {},
                ),
            })
            if semilinear_source_path is not None:
                source_path = Path(semilinear_source_path)
                run_summary["semilinear"]["source_pixels_url"] = (
                    "/" + source_path.relative_to(project_root).as_posix()
                )
        if semilinear_fig is not None:
            plt.close(semilinear_fig)
        runtime_summary = finish_lens_runtime_timer(run_timer, status="completed")
        run_summary["runtime"] = runtime_summary
        (run_output_dir / "summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        preview_url = "/" + latest.relative_to(project_root).as_posix()
        gui_status.write(
            state="completed",
            message="Lens model completed.",
            preview_url=preview_url,
            losses_url="/" + losses_comparison_png.relative_to(project_root).as_posix(),
            chain_previews=chain_previews,
            runtime=runtime_summary,
            progress={"fraction": 1.0, "message": "Lens model completed."},
        )
        print(f"LONG_SVI_RUN_DONE output_dir={run_output_dir}", flush=True)
        return run_summary

    @staticmethod
    def save_single_plane_svi_outputs_from_context(context):
        return LensResultWriter.save_single_plane_svi_outputs(
            project_root=context["PROJECT_ROOT"],
            run_output_dir=context["RUN_OUTPUT_DIR"],
            gui_status=context["GUI_STATUS"],
            run_timer=context["RUN_TIMER"],
            model_config=context["MODEL_CONFIG"],
            num_chains=context["num_chains"],
            lens_image_parametric=context["lens_image_parametric"],
            lens_image_pixelated=context["lens_image_pixelated"],
            parametric_states=context["parametric_states"],
            parametric_kwargs_list=context["parametric_kwargs_list"],
            pixelated_states=context["pixelated_states"],
            pixelated_kwargs_list=context["pixelated_kwargs_list"],
            data=context["data"],
            fit_mask=context["fit_mask"],
            rms_map=context["rms_map"],
            rms=context["rms"],
            pixel_grid_shape=context["pixel_grid_shape"],
            image_extent=context["extent"],
            data_file=context["DATA_FILE"],
            mask_1_file=context["MASK_1_FILE"],
            mask_out_file=context["MASK_OUT_FILE"],
            psf_file=context["PSF_FILE"],
            rms_file=context["RMS_FILE"],
            rms_prior_low=context["rms_prior_low"],
            rms_prior_high=context["rms_prior_high"],
            background_offset=context["background_offset"],
            semilinear_result=context.get("semilinear_result"),
            semilinear_source_path=context.get("semilinear_source_path"),
            stage2_states=context.get("stage2_states"),
        )


class LensImageExtension(LensImage):
    def __init__(
        self,
        grid_class,
        psf_class,
        noise_class=None,
        lens_mass_model_class=None,
        source_model_class=None,
        lens_light_model_class=None,
        point_source_model_class=None,
        source_arc_mask=None,
        source_grid_scale=1.0,
        conjugate_points=None,
        kwargs_numerics=None,
    ):
        super().__init__(
            grid_class,
            psf_class,
            noise_class=noise_class,
            lens_mass_model_class=lens_mass_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
            point_source_model_class=point_source_model_class,
            source_arc_mask=source_arc_mask,
            kwargs_numerics=kwargs_numerics,
        )
        self._source_grid_scale = source_grid_scale
        self.conjugate_points = conjugate_points

        ssf = self.ImageNumerics.grid_supersampling_factor
        s_ones = np.ones([ssf, ssf])
        self.source_arc_mask_ss = np.kron(self.source_arc_mask, s_ones)
        self._source_arc_mask_flat = self.source_arc_mask_ss.flatten()
        self._source_arc_mask_outline_flat = (
            self.source_arc_mask_ss - scipy.ndimage.binary_erosion(self.source_arc_mask_ss)
        ).flatten().astype(bool)

    def source_surface_brightness(
        self,
        kwargs_source,
        kwargs_lens=None,
        de_lensed=False,
        k=None,
        k_lens=None,
    ):
        if len(self.SourceModel.profile_type_list) == 0:
            return jnp.zeros(self.Grid.num_pixel_axes)

        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        if (self._src_adaptive_grid) or (not de_lensed):
            x_grid_src, y_grid_src = self.MassModel.ray_shooting(
                x_grid_img,
                y_grid_img,
                kwargs_lens,
                k=k_lens,
            )
            pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(
                x_grid_src,
                y_grid_src,
            )
        else:
            pixels_x_coord, pixels_y_coord = None, None
        if de_lensed:
            source_light = self.SourceModel.surface_brightness(
                x_grid_img,
                y_grid_img,
                kwargs_source,
                k=k,
                pixels_x_coord=pixels_x_coord,
                pixels_y_coord=pixels_y_coord,
            )
        else:
            source_light = self.SourceModel.surface_brightness(
                x_grid_src,
                y_grid_src,
                kwargs_source,
                k=k,
                pixels_x_coord=pixels_x_coord,
                pixels_y_coord=pixels_y_coord,
            )
        return source_light

    def lens_surface_brightness(self, kwargs_lens_light, k=None):
        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        return self.LensLightModel.surface_brightness(
            x_grid_img,
            y_grid_img,
            kwargs_lens_light,
            k=k,
        )

    @partial(jax.jit, static_argnums=(0, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    def model(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_point_source=None,
        unconvolved=False,
        supersampled=False,
        source_add=True,
        lens_light_add=True,
        point_source_add=True,
        k_lens=None,
        k_source=None,
        k_lens_light=None,
        k_point_source=None,
        psf_noise_fft=None,
    ):
        model = jnp.zeros((self.ImageNumerics.grid_class.num_grid_points,)).flatten()
        if source_add is True:
            source_model = self.source_surface_brightness(
                kwargs_source,
                kwargs_lens,
                k=k_source,
                k_lens=k_lens,
            )
            if self._source_arc_mask_flat is not None:
                source_model *= self._source_arc_mask_flat
            model += source_model
        if lens_light_add is True:
            model += self.lens_surface_brightness(
                kwargs_lens_light,
                k=k_lens_light,
            )
        if not supersampled:
            model = self.ImageNumerics.re_size_convolve(
                model,
                unconvolved=unconvolved,
                kwargs_psf=psf_noise_fft,
            )
        if point_source_add:
            model += self.point_source_image(
                kwargs_point_source,
                kwargs_lens,
                kwargs_solver=self.kwargs_lens_equation_solver,
                k=k_point_source,
                kwargs_psf=psf_noise_fft,
            )
        return model

    def trace_conjugate_points(self, kwargs_lens, k_lens=None):
        if self.conjugate_points is not None:
            x, y = self.conjugate_points.T
            conj_x, conj_y = self.MassModel.ray_shooting(x, y, kwargs_lens, k=k_lens)
            return jnp.vstack([conj_x, conj_y]).T
        return None

    def mask_extent(self, x_grid_src, y_grid_src, npix_src, grid_scale=1):
        x_left, x_right = x_grid_src.min(), x_grid_src.max()
        y_bottom, y_top = y_grid_src.min(), y_grid_src.max()
        cx = 0.5 * (x_left + x_right)
        cy = 0.5 * (y_bottom + y_top)
        width = jnp.abs(x_left - x_right)
        height = jnp.abs(y_bottom - y_top)
        half_size = 0.5 * grid_scale * jnp.maximum(height, width)
        x_left = cx - half_size
        x_right = cx + half_size
        y_bottom = cy - half_size
        y_top = cy + half_size
        x_adapt = jnp.linspace(x_left, x_right, npix_src)
        y_adapt = jnp.linspace(y_bottom, y_top, npix_src)
        extent_adapt = [x_adapt[0], x_adapt[-1], y_adapt[0], y_adapt[-1]]
        return x_adapt, y_adapt, extent_adapt

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def adapt_source_coordinates(
        self,
        x_grid_src,
        y_grid_src,
        force=False,
        npix_src=100,
        source_grid_scale=1,
    ):
        if self._src_adaptive_grid or force:
            if not force:
                npix_src, npix_src_y = self.SourceModel.pixel_grid.num_pixel_axes
                if npix_src_y != npix_src:
                    raise ValueError('Adaptive source plane grid only works with square grids')
                grid_scale = self._source_grid_scale
            else:
                grid_scale = source_grid_scale
            if self.Grid.x_is_inverted or self.Grid.y_is_inverted:
                raise NotImplementedError('invert x and y not yet supported for adaptive source grid')
            return self.mask_extent(
                x_grid_src[self._source_arc_mask_outline_flat],
                y_grid_src[self._source_arc_mask_outline_flat],
                npix_src,
                grid_scale,
            )
        return None, None, None

    def get_source_coordinates(
        self,
        kwargs_lens,
        force=False,
        npix_src=100,
        source_grid_scale=1.0,
        k_lens=None,
    ):
        if (not self._src_adaptive_grid) and (self.SourceModel.pixel_grid is not None):
            x_grid, y_grid = self.SourceModel.pixel_grid.pixel_coordinates
            extent = self.SourceModel.pixel_grid.extent
        else:
            x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
            x_grid_src, y_grid_src = self.MassModel.ray_shooting(
                x_grid_img,
                y_grid_img,
                kwargs_lens,
                k=k_lens,
            )
            x_grid, y_grid, extent = self.adapt_source_coordinates(
                x_grid_src,
                y_grid_src,
                force=force,
                npix_src=npix_src,
                source_grid_scale=source_grid_scale,
            )
        return x_grid, y_grid, extent


class ResumeInit:
    @staticmethod
    def select_init_values(params, allowed_keys):
        return {k: params[k] for k in allowed_keys if k in params}

    @staticmethod
    def stack_or_none(*xs):
        first = xs[0]
        return None if first is None else jnp.stack(xs)

    @staticmethod
    def stack_dicts(dict_list):
        return jax.tree.map(lambda *xs: jnp.stack(xs), *dict_list)

    @staticmethod
    def existing_batch_indices(output_dir, suffix_hmc):
        prefix = 'WFI2033_'
        indices = []
        for path in Path(output_dir).glob(f'WFI2033_[0-9]*{suffix_hmc}.nc'):
            name = path.name
            idx_str = name[len(prefix):name.index(suffix_hmc)]
            indices.append(int(idx_str))
        return sorted(indices)

    @staticmethod
    def save_resume_state(path, state):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as fh:
            pickle.dump(jax.device_get(state), fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_resume_state(path):
        with Path(path).open('rb') as fh:
            return pickle.load(fh)

    @staticmethod
    @jax.jit
    def get_value_from_index(xs, i):
        i = jnp.asarray(i)
        return jax.tree.map(lambda x: x[i], xs)

    @staticmethod
    def init_to_value_or_defer(site=None, values=None, defer=numpyro.infer.init_to_median):
        if values is None:
            values = {}
        if site is None:
            return partial(ResumeInit.init_to_value_or_defer, values=values, defer=defer)

        if site["type"] == "sample" and not site["is_observed"]:
            if site["name"] in values:
                return values[site["name"]]
            return defer(site)

    @staticmethod
    def pixelated_stage_init_from_parametric(params):
        allowed_prefixes = (
            'theta_E_1',
            'gamma_1',
            'e_1',
            'center_1',
            'gamma_sheer_1',
            'shear_strength_1',
            'shear_position_angle_1',
            'A_lens',
            'amp_lens',
            'sigma_lens',
            'e_lens',
            'center_lens',
            'ra_ps',
            'dec_ps',
            'log10_amp_ps',
            'RMS',
        )
        return {key: value for key, value in params.items() if key.startswith(allowed_prefixes)}

    @staticmethod
    def _inside(value, low, high, margin=1e-6):
        """Pull a value strictly inside an open interval.

        numpyro maps a constrained site to an unconstrained one with a logit-style
        transform, so a value sitting exactly on a bound becomes +-inf and
        ``initialize_model`` rejects the whole init with "Cannot find valid initial
        parameters".  A converged stage can legitimately land on a bound (zero external
        shear, zero position angle), so clamp before handing the value over.
        """
        span = high - low
        pad = jnp.minimum(margin, 1e-3*span)
        return jnp.clip(value, low + pad, high - pad)

    @staticmethod
    def solver_stage_init_from_pixelated(params, param_name='1', prior=None):
        """Re-express soft-constraint mass parameters for the exact-solver stage.

        ``ThetaEllipticitySolver`` turns ``theta_E`` and the ellipticity magnitude into
        deterministic outputs of the two-image solve and samples a position angle in
        their place, so those two sites must be dropped from the init dict and
        ``epl_position_angle`` synthesised from the cartesian ellipticity.  Every other
        site (lens light, point sources, source grid, shear, centre) keeps its name and
        transfers unchanged.
        """
        prior = prior or {}
        theta_key = f'theta_E_{param_name}'
        e_key = f'e_{param_name}'
        init_values = {key: value for key, value in params.items() if key not in (theta_key, e_key)}
        if e_key in params:
            e_flat = jnp.ravel(jnp.asarray(params[e_key]))
            # e = |e| * (cos 2phi, sin 2phi).  atan2 lands in (-pi/2, pi/2] after the
            # halving, but the solver's prior is Uniform(0, pi), so wrap into support.
            angle = jnp.mod(0.5*jnp.arctan2(e_flat[1], e_flat[0]), jnp.pi)
            init_values[f'epl_position_angle_{param_name}'] = jnp.atleast_1d(
                ResumeInit._inside(angle, 0.0, float(jnp.pi))
            )
        angle_key = f'shear_position_angle_{param_name}'
        if angle_key in init_values:
            init_values[angle_key] = ResumeInit._inside(
                jnp.asarray(init_values[angle_key]), 0.0, float(jnp.pi)
            )
        for key, low, high in (
            (f'shear_strength_{param_name}', 'shear_strength_low', 'shear_strength_high'),
            (f'gamma_{param_name}', 'gamma_low', 'gamma_high'),
        ):
            if key in init_values and low in prior:
                init_values[key] = ResumeInit._inside(
                    jnp.asarray(init_values[key]), float(prior[low]), float(prior[high])
                )
        return init_values

    @staticmethod
    def solver_initial_q_from_pixelated(params, fallback, param_name='1'):
        """Seed the Newton solve with this chain's own converged (theta_E, |e|).

        Falls back to the GUI-supplied global guess when the earlier stage did not
        produce the two sites.  Bounds match the clip box used inside
        ``ThetaEllipticitySolver.newton``.
        """
        theta_key = f'theta_E_{param_name}'
        e_key = f'e_{param_name}'
        if theta_key not in params or e_key not in params:
            return jnp.asarray(fallback, dtype=jnp.float64)
        theta = jnp.ravel(jnp.asarray(params[theta_key]))[0]
        e_flat = jnp.ravel(jnp.asarray(params[e_key]))
        return jnp.stack([
            jnp.clip(theta, 1e-4, 5.0),
            jnp.clip(jnp.hypot(e_flat[0], e_flat[1]), 1e-6, 0.95),
        ])


class Geometry:
    @staticmethod
    def get_pixel_grid(data, pix_scale, ss=1):
        ny, nx = data.shape
        ny *= ss
        nx *= ss
        pix_scale /= ss
        half_size_x = nx * pix_scale / 2
        half_size_y = ny * pix_scale / 2
        ra_at_xy_0 = -half_size_x + pix_scale / 2
        dec_at_xy_0 = -half_size_y + pix_scale / 2
        transform_pix2angle = pix_scale * np.eye(2)
        kwargs_pixel = {
            'nx': nx,
            'ny': ny,
            'ra_at_xy_0': ra_at_xy_0,
            'dec_at_xy_0': dec_at_xy_0,
            'transform_pix2angle': transform_pix2angle,
        }
        pixel_grid = PixelGrid(**kwargs_pixel)
        xgrid, ygrid = pixel_grid.pixel_coordinates
        x_axis = xgrid[0]
        y_axis = ygrid[:, 0]
        extent = pixel_grid.extent
        return pixel_grid, xgrid, ygrid, x_axis, y_axis, extent, nx, ny

    @staticmethod
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def reduced_distance(a, i, j):
        u = jax.lax.dynamic_slice_in_dim(a, i, 1)
        v = jax.lax.dynamic_slice_in_dim(a, j, 1)
        return jnp.linalg.norm(u - v, ord=2)

    @staticmethod
    def reduced_distance_matrix(a):
        i, j = jnp.triu_indices(a.shape[0], k=1)
        return Geometry.reduced_distance(a, i, j)

    @staticmethod
    def get_best_pixel_size(lens_image, herc_dict, source_grid_scale, return_full=False):
        from sklearn.neighbors import NearestNeighbors

        x_ss_grid, y_ss_grid = lens_image.ImageNumerics.coordinates_evaluate
        mask = lens_image._source_arc_mask_flat.astype(bool)
        x_ss_trace, y_ss_trace = lens_image.MassModel.ray_shooting(
            x_ss_grid[mask].flatten(),
            y_ss_grid[mask].flatten(),
            herc_dict['kwargs_lens'],
        )

        _, _, extent = lens_image.mask_extent(
            x_ss_trace,
            y_ss_trace,
            100,
            grid_scale=source_grid_scale,
        )

        full_size = jax.device_get(extent[1] - extent[0])
        tdx = (
            (x_ss_trace >= extent[0])
            & (x_ss_trace <= extent[1])
            & (y_ss_trace >= extent[2])
            & (y_ss_trace <= extent[3])
        )

        jax.block_until_ready(x_ss_trace)
        jax.block_until_ready(y_ss_trace)
        x_trim = jax.device_get(x_ss_trace[tdx])
        y_trim = jax.device_get(y_ss_trace[tdx])
        samples = np.vstack([x_trim, y_trim]).T
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(samples)
        distances, _ = nbrs.kneighbors(samples)

        mean_distance = np.mean(distances[:, 1])
        pixel_grid_shape = int(full_size / (5 * mean_distance) + 1)
        if return_full:
            return pixel_grid_shape, x_trim, y_trim, mean_distance, full_size
        return pixel_grid_shape

    @staticmethod
    def get_best_pixel_size_multiplane(lens_image, herc_dict, source_grid_scale, N=1, return_full=False):
        from sklearn.neighbors import NearestNeighbors

        x_ss_grid, y_ss_grid = lens_image.ImageNumerics.coordinates_evaluate
        ra_grid_planes, dec_grid_planes = lens_image.MPMassModel.ray_shooting(
            x_ss_grid,
            y_ss_grid,
            herc_dict['eta_flat'],
            herc_dict['kwargs_mass'],
        )
        mask = lens_image._source_arc_masks_flat_bool[N]
        x_ss_trace = ra_grid_planes[N][mask].flatten()
        y_ss_trace = dec_grid_planes[N][mask].flatten()

        _, _, extent = lens_image.mask_extent(
            x_ss_trace,
            y_ss_trace,
            100,
            source_grid_scale,
        )

        full_size = jax.device_get(extent[1] - extent[0])
        tdx = (
            (x_ss_trace >= extent[0])
            & (x_ss_trace <= extent[1])
            & (y_ss_trace >= extent[2])
            & (y_ss_trace <= extent[3])
        )

        jax.block_until_ready(x_ss_trace)
        jax.block_until_ready(y_ss_trace)
        x_trim = jax.device_get(x_ss_trace[tdx])
        y_trim = jax.device_get(y_ss_trace[tdx])
        samples = np.vstack([x_trim, y_trim]).T
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(samples)
        distances, _ = nbrs.kneighbors(samples)

        mean_distance = np.mean(distances[:, 1])
        pixel_grid_shape = int(full_size / (5 * mean_distance) + 1)
        if return_full:
            return pixel_grid_shape, x_trim, y_trim, mean_distance, full_size
        return pixel_grid_shape


class Mass:
    _fixed_mass_map_cache = {}

    @staticmethod
    def load_fixed_mass_map(path):
        """Load and cache a six-extension FITS field for PIXELATED_FIXED."""
        resolved_path = str(Path(path).expanduser().resolve())
        if resolved_path in Mass._fixed_mass_map_cache:
            return Mass._fixed_mass_map_cache[resolved_path]

        required = ('POTENTIAL', 'ALPHA_X', 'ALPHA_Y', 'HESS_XX', 'HESS_YY', 'HESS_XY')
        with fits.open(resolved_path, memmap=False) as hdul:
            missing = [name for name in required if name not in hdul]
            if missing:
                raise ValueError(
                    f'Fixed mass map {resolved_path} is missing FITS extensions: {missing}'
                )
            arrays = {
                name: np.asarray(hdul[name].data, dtype=np.float64)
                for name in required
            }
            header = hdul['POTENTIAL'].header

        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(f'Fixed mass map extensions have inconsistent shapes: {sorted(shapes)}')
        shape = next(iter(shapes))
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError(f'Fixed mass map must contain 2-D arrays of at least 2x2 pixels; got {shape}')
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f'Fixed mass map contains non-finite values: {resolved_path}')

        pixel_scale = float(header['PIXSCALE'])
        x_min = float(header['XMIN'])
        y_min = float(header['YMIN'])
        if not np.isfinite(pixel_scale) or pixel_scale <= 0:
            raise ValueError(f'Fixed mass map has invalid PIXSCALE={pixel_scale!r}')
        ny, nx = shape
        pixel_grid = PixelGrid(
            nx=nx,
            ny=ny,
            ra_at_xy_0=x_min,
            dec_at_xy_0=y_min,
            transform_pix2angle=pixel_scale * np.eye(2),
        )
        profile = PixelatedFixed(
            func_pixel_grid=pixel_grid,
            deriv_pixel_grid=pixel_grid,
            hess_pixel_grid=pixel_grid,
        )
        kwargs = {
            'func_pixels': jnp.asarray(arrays['POTENTIAL']),
            'deriv_pixels': (
                jnp.asarray(arrays['ALPHA_X']),
                jnp.asarray(arrays['ALPHA_Y']),
            ),
            'hess_pixels': (
                jnp.asarray(arrays['HESS_XX']),
                jnp.asarray(arrays['HESS_YY']),
                jnp.asarray(arrays['HESS_XY']),
            ),
        }
        Mass._fixed_mass_map_cache[resolved_path] = (profile, kwargs)
        return profile, kwargs

    @staticmethod
    def scale_theta_E_from_g2(theta_E_g2, target_theta_mean, g2_theta_mean):
        return theta_E_g2 * target_theta_mean / g2_theta_mean

    @staticmethod
    def EPL_w_shear(
        plate_name,
        param_name,
        theta_low=0.0,
        theta_high=3.0,
        gamma_low=1.2,
        gamma_high=2.8,
        gamma_fixed=None,
        center_low=-0.2,
        center_high=0.2,
        center_sigma=0.1,
        e_low=-0.2,
        e_high=0.2,
        e_sigma=0.25,
        shear_strength_low=0.0,
        shear_strength_high=0.2,
        shear_low=None,
        shear_high=None,
    ):
        # ``shear_low``/``shear_high`` are retained only so older saved GUI
        # configurations remain readable.  They now define the maximum shear
        # strength rather than a rectangular prior on (gamma1, gamma2).
        if shear_low is not None or shear_high is not None:
            legacy_bounds = [value for value in (shear_low, shear_high) if value is not None]
            shear_strength_high = max(abs(float(value)) for value in legacy_bounds)
        with numpyro.plate(f'{plate_name} scalers - [1]', 1):
            theta_E = numpyro.sample(f'theta_E_{param_name}', dist.Uniform(theta_low, theta_high))
            if gamma_fixed is None:
                gamma = numpyro.sample(f'gamma_{param_name}', dist.Uniform(gamma_low, gamma_high))
            else:
                gamma = jnp.asarray([float(gamma_fixed)], dtype=jnp.float64)
            shear_strength = numpyro.sample(
                f'shear_strength_{param_name}',
                dist.Uniform(shear_strength_low, shear_strength_high),
            )
            shear_position_angle = numpyro.sample(
                f'shear_position_angle_{param_name}',
                dist.Uniform(0.0, jnp.pi),
            )
            with numpyro.plate(f'{plate_name} vectors - [2]', 2):
                e_mass = numpyro.sample(
                    f'e_{param_name}',
                    dist.TruncatedNormal(0, e_sigma, low=e_low, high=e_high),
                )
        center = numpyro.sample(
            f'center_{param_name}',
            dist.TruncatedNormal(0, center_sigma, low=center_low, high=center_high).expand([2]),
        )
        gamma1_shear = shear_strength[0] * jnp.cos(2.0 * shear_position_angle[0])
        gamma2_shear = shear_strength[0] * jnp.sin(2.0 * shear_position_angle[0])
        return [
            {
                'theta_E': theta_E[0],
                'gamma': gamma[0],
                'e1': e_mass[0],
                'e2': e_mass[1],
                'center_x': center[0],
                'center_y': center[1],
            },
            {
                'gamma1': gamma1_shear,
                'gamma2': gamma2_shear,
                'ra_0': center[0],
                'dec_0': center[1],
            },
        ]

    @staticmethod
    def params2kwargs_EPL_w_shear(params, param_name, gamma_fixed=None):
        gamma = (
            params[f'gamma_{param_name}'][0]
            if gamma_fixed is None
            else jnp.asarray(float(gamma_fixed), dtype=jnp.float64)
        )
        if f'shear_strength_{param_name}' in params:
            shear_strength = params[f'shear_strength_{param_name}'][0]
            shear_position_angle = params[f'shear_position_angle_{param_name}'][0]
            gamma1_shear = shear_strength * jnp.cos(2.0 * shear_position_angle)
            gamma2_shear = shear_strength * jnp.sin(2.0 * shear_position_angle)
        else:
            # Backward compatibility for results produced with the former
            # component-wise external-shear parameterisation.
            gamma1_shear = params[f'gamma_sheer_{param_name}'][0]
            gamma2_shear = params[f'gamma_sheer_{param_name}'][1]
        return [
            {
                'theta_E': params[f'theta_E_{param_name}'][0],
                'gamma': gamma,
                'e1': params[f'e_{param_name}'][0],
                'e2': params[f'e_{param_name}'][1],
                'center_x': params[f'center_{param_name}'][0],
                'center_y': params[f'center_{param_name}'][1],
            },
            {
                'gamma1': gamma1_shear,
                'gamma2': gamma2_shear,
                'ra_0': params[f'center_{param_name}'][0],
                'dec_0': params[f'center_{param_name}'][1],
            },
        ]

    @staticmethod
    def SIS(
        plate_name,
        param_name,
        origin,
        theta_low=0.0,
        theta_high=0.01,
        theta_mean=None,
        theta_sigma=None,
    ):
        with numpyro.plate(f'{plate_name} scalers - [1]', 1):
            if theta_mean is not None and theta_sigma is not None:
                theta_E = numpyro.sample(
                    f'theta_E_{param_name}',
                    dist.Normal(theta_mean, theta_sigma),
                )
            else:
                theta_E = numpyro.sample(
                    f'theta_E_{param_name}',
                    dist.Uniform(theta_low, theta_high),
                )
            center_0 = numpyro.deterministic(f'center_1_{param_name}', jnp.array([origin[0]]))
            center_1 = numpyro.deterministic(f'center_2_{param_name}', jnp.array([origin[1]]))

        return [{
            'theta_E': theta_E[0],
            'center_x': center_0[0],
            'center_y': center_1[0],
        }]

    @staticmethod
    def params2kwargs_SIS(params, param_name, fallback_origin=None):
        if f'center_1_{param_name}' in params and f'center_2_{param_name}' in params:
            center_x = params[f'center_1_{param_name}'][0]
            center_y = params[f'center_2_{param_name}'][0]
        elif fallback_origin is not None:
            center_x, center_y = fallback_origin
        else:
            raise KeyError(f'Missing SIS center sites for {param_name}.')
        return [{
            'theta_E': params[f'theta_E_{param_name}'][0],
            'center_x': center_x,
            'center_y': center_y,
        }]

    @staticmethod
    def SIE(
        plate_name,
        param_name,
        origin,
        theta_low=0.0,
        theta_high=1.0,
        e_low=-0.3,
        e_high=0.3,
        e_sigma=0.15,
    ):
        with numpyro.plate(f'{plate_name} scalers - [1]', 1):
            theta_E = numpyro.sample(
                f'theta_E_{param_name}',
                dist.Uniform(theta_low, theta_high),
            )
            with numpyro.plate(f'{plate_name} vectors - [2]', 2):
                e_mass = numpyro.sample(
                    f'e_{param_name}',
                    dist.TruncatedNormal(0, e_sigma, low=e_low, high=e_high),
                )
            center_0 = numpyro.deterministic(f'center_1_{param_name}', jnp.array([origin[0]]))
            center_1 = numpyro.deterministic(f'center_2_{param_name}', jnp.array([origin[1]]))

        return [{
            'theta_E': theta_E[0],
            'e1': e_mass[0],
            'e2': e_mass[1],
            'center_x': center_0[0],
            'center_y': center_1[0],
        }]

    @staticmethod
    def params2kwargs_SIE(params, param_name, fallback_origin=None):
        if f'center_1_{param_name}' in params and f'center_2_{param_name}' in params:
            center_x = params[f'center_1_{param_name}'][0]
            center_y = params[f'center_2_{param_name}'][0]
        elif fallback_origin is not None:
            center_x, center_y = fallback_origin
        else:
            raise KeyError(f'Missing SIE center sites for {param_name}.')
        return [{
            'theta_E': params[f'theta_E_{param_name}'][0],
            'e1': params[f'e_{param_name}'][0],
            'e2': params[f'e_{param_name}'][1],
            'center_x': center_x,
            'center_y': center_y,
        }]

    @staticmethod
    def lens_plane_component_profiles(components):
        profiles = []
        for component in components:
            profile = str(component['profile']).upper()
            if profile == 'FIXED_MASS_MAP':
                fixed_profile, _ = Mass.load_fixed_mass_map(component['path'])
                profiles.append(fixed_profile)
            else:
                profiles.append(profile)
        return profiles

    @staticmethod
    def lens_plane_components(components):
        kwargs_mass = []
        for index, component in enumerate(components, start=1):
            profile = str(component['profile']).upper()
            if profile == 'FIXED_MASS_MAP':
                _, fixed_kwargs = Mass.load_fixed_mass_map(component['path'])
                kwargs_mass.append(fixed_kwargs)
                continue
            param_name = str(component['param_name'])
            origin = (float(component['center_x']), float(component['center_y']))
            common = {
                'plate_name': f'Lens-plane {profile} {index}',
                'param_name': param_name,
                'origin': origin,
                'theta_low': float(component['theta_low']),
                'theta_high': float(component['theta_high']),
            }
            if profile == 'SIS':
                kwargs_mass.extend(Mass.SIS(**common))
            elif profile == 'SIE':
                kwargs_mass.extend(
                    Mass.SIE(
                        **common,
                        e_low=float(component['e_low']),
                        e_high=float(component['e_high']),
                        e_sigma=float(component['e_sigma']),
                    )
                )
            else:
                raise ValueError(f'Unsupported lens-plane mass profile: {profile}')
        return kwargs_mass

    @staticmethod
    def params2kwargs_lens_plane_components(params, components):
        kwargs_mass = []
        for component in components:
            profile = str(component['profile']).upper()
            if profile == 'FIXED_MASS_MAP':
                _, fixed_kwargs = Mass.load_fixed_mass_map(component['path'])
                kwargs_mass.append(fixed_kwargs)
                continue
            param_name = str(component['param_name'])
            origin = (float(component['center_x']), float(component['center_y']))
            if profile == 'SIS':
                kwargs_mass.extend(Mass.params2kwargs_SIS(params, param_name, fallback_origin=origin))
            elif profile == 'SIE':
                kwargs_mass.extend(Mass.params2kwargs_SIE(params, param_name, fallback_origin=origin))
            else:
                raise ValueError(f'Unsupported lens-plane mass profile: {profile}')
        return kwargs_mass

    @staticmethod
    def lens_plane_component_init_values(params, components):
        names = Mass.lens_plane_component_hmc_vars(components)
        return {name: params[name] for name in names if name in params}

    @staticmethod
    def lens_plane_component_hmc_vars(components):
        names = []
        for component in components:
            if str(component['profile']).upper() == 'FIXED_MASS_MAP':
                continue
            param_name = str(component['param_name'])
            names.append(f'theta_E_{param_name}')
            if str(component['profile']).upper() == 'SIE':
                names.append(f'e_{param_name}')
        return names

    @staticmethod
    def lens_plane_component_dense_mass_groups(components):
        groups = []
        for component in components:
            if str(component['profile']).upper() != 'SIE':
                continue
            param_name = str(component['param_name'])
            groups.append((f'theta_E_{param_name}', f'e_{param_name}'))
        return groups

    @staticmethod
    def GNFW_w_shear(
        plate_name,
        param_name,
        gamma_in_up=2,
        gamma_in_low=0.5,
        Rs_high=None,
        Rs_low=None,
        Rs_mean=None,
        Rs_std=None,
        Rs_value=None,
        e_low=-0.2,
        e_high=0.2,
        center_x=None,
        center_y=None,
        kappa_s_low=0.0,
        kappa_s_high=1,
        sph=False,
        gamma_sheer_low=-0.2,
        gamma_sheer_high=0.2,
        gamma_sheer_value=None,
    ):
        if Rs_value is not None:
            Rs = numpyro.deterministic(f'Rs_{param_name}', jnp.float64(Rs_value))
        elif Rs_low is not None:
            Rs = numpyro.sample(f'Rs_{param_name}', dist.Uniform(Rs_low, Rs_high))
        elif Rs_mean is not None:
            Rs = numpyro.sample(
                f'Rs_{param_name}',
                dist.TruncatedNormal(Rs_mean, Rs_std, low=Rs_mean - 1 * Rs_std, high=Rs_mean + 1 * Rs_std),
            )
        else:
            raise ValueError('GNFW_w_shear requires one of Rs_value, Rs_low/Rs_high, or Rs_mean/Rs_std')

        kappa_s = numpyro.sample(f'kappa_s_{param_name}', dist.Uniform(kappa_s_low, kappa_s_high))
        gamma_in = numpyro.sample(f'gammain_{param_name}', dist.Uniform(gamma_in_low, gamma_in_up))

        if gamma_sheer_value is None:
            with numpyro.plate(f'{plate_name} vectors - [2]', 2):
                gamma_sheer = numpyro.sample(
                    f'gamma_sheer_{param_name}',
                    dist.Uniform(gamma_sheer_low, gamma_sheer_high),
                )
        else:
            gamma_sheer = numpyro.deterministic(
                f'gamma_sheer_{param_name}',
                jnp.asarray(gamma_sheer_value, dtype=jnp.float64).reshape(2,),
            )

        if sph is False:
            with numpyro.plate(f'{plate_name} vectors - [2]', 2):
                e_mass = numpyro.sample(
                    f'e_{param_name}',
                    dist.TruncatedNormal(0, 0.25, low=e_low, high=e_high),
                )
        else:
            e_mass = numpyro.deterministic(f"e_{param_name}", jnp.array([0.0001, -0.0001]))

        if center_x is None:
            center = numpyro.sample(
                f'center_{param_name}',
                dist.TruncatedNormal(0, 1, low=-0.4, high=0.4).expand([2]),
            )
        else:
            center = numpyro.deterministic(f"center_{param_name}", jnp.array([center_x, center_y]))

        return [{
            'R_s': Rs,
            'gamma': gamma_in,
            'kappa_s': kappa_s,
            'e1': e_mass[0],
            'e2': e_mass[1],
            'center_x': center[0],
            'center_y': center[1],
        }, {
            'gamma1': gamma_sheer[0],
            'gamma2': gamma_sheer[1],
            'ra_0': center[0],
            'dec_0': center[1],
        }]


class PowerSpectrum:
    @staticmethod
    def matern_prior_kwargs(config):
        """Return only supported Matérn source options from a GUI config."""
        config = config or {}
        keys = (
            'k_zero', 'positive', 'nonlinear_brightness', 'log_brightness',
            'log_mean_loc', 'log_mean_scale', 'log_n_low', 'log_n_high',
            'log_rho_low', 'log_rho_high', 'log_sigma_scale',
        )
        return {key: config[key] for key in keys if key in config}


    class K_grid:
        def __init__(self, shape, scale=1):
            self.Ny, self.Nx = shape
            self.scale = scale

        @lazy_property
        def rk(self):
            kx = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.scale)
            ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=self.scale)
            return np.sqrt(ky.reshape(-1, 1) ** 2 + kx ** 2)

        @lazy_property
        def k(self):
            kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=self.scale)
            ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=self.scale)
            return np.sqrt(ky.reshape(-1, 1) ** 2 + kx ** 2)

    @staticmethod
    @partial(jax.jit, static_argnums=(5,))
    def P_Matern(k, n, sigma, rho, c=1e-20, k_zero=None):
        r = 2 * n / rho ** 2
        norm = sigma ** 2 * 4 * jnp.pi * n * jnp.power(r, n)
        power = norm * jnp.power(r + k ** 2, -(n + 1))
        if k_zero is not None:
            power = jnp.where(k == 0, k_zero, power)
        return power

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _odd_pack(values, n_pix):
        n1 = n_pix // 2 + 1
        thin_real = jax.lax.dynamic_slice(values, (0, 1), (n_pix, n1 - 1))
        thin_imag = jnp.flip(jax.lax.dynamic_slice(values, (0, n1), (n_pix, n1 - 1)), axis=1)

        first_real_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 1, 1))
        first_real = jnp.vstack([
            2 * values[0, 0].reshape(1, 1),
            first_real_slice,
            jnp.flip(first_real_slice, axis=0),
        ])

        first_imag_slice = jax.lax.dynamic_slice(values, (n1, 0), (n1 - 1, 1))
        first_imag = jnp.vstack([
            jnp.zeros((1, 1)),
            -jnp.flip(first_imag_slice, axis=0),
            first_imag_slice,
        ])

        fft_real = jnp.hstack([first_real[:thin_real.shape[0]], thin_real])
        fft_imag = jnp.hstack([first_imag[:thin_imag.shape[0]], thin_imag])
        return fft_real + 1j * fft_imag

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _even_pack(values, n_pix):
        n1 = n_pix // 2 + 1
        thin_real = jax.lax.dynamic_slice(values, (0, 1), (n_pix, n1 - 2))
        thin_imag = jnp.flip(jax.lax.dynamic_slice(values, (0, n1), (n_pix, n1 - 2)), axis=1)

        first_real_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 2, 1))
        first_real = jnp.vstack([
            2 * jax.lax.dynamic_slice(values, (0, 0), (1, 1)),
            first_real_slice,
            2 * jax.lax.dynamic_slice(values, (n1 - 1, 0), (1, 1)),
            jnp.flip(first_real_slice, axis=0),
        ])

        last_real_slice = jax.lax.dynamic_slice(values, (1, n1 - 1), (n1 - 2, 1))
        last_real = jnp.vstack([
            2 * jax.lax.dynamic_slice(values, (0, n1 - 1), (1, 1)),
            last_real_slice,
            2 * jax.lax.dynamic_slice(values, (n1 - 1, n1 - 1), (1, 1)),
            jnp.flip(last_real_slice, axis=0),
        ])

        first_imag_slice = jax.lax.dynamic_slice(values, (n1, 0), (n1 - 2, 1))
        first_imag = jnp.vstack([
            jnp.zeros((1, 1)),
            -jnp.flip(first_imag_slice, axis=0),
            jnp.zeros((1, 1)),
            first_imag_slice,
        ])

        last_imag_slice = jax.lax.dynamic_slice(values, (n1, n1 - 1), (n1 - 2, 1))
        last_imag = jnp.vstack([
            jnp.zeros((1, 1)),
            -jnp.flip(last_imag_slice, axis=0),
            jnp.zeros((1, 1)),
            last_imag_slice,
        ])

        delta = thin_real.shape[0] - first_real.shape[0]
        first_real = jnp.pad(first_real, ((0, delta), (0, 0)))
        last_real = jnp.pad(last_real, ((0, delta), (0, 0)))
        first_imag = jnp.pad(first_imag, ((0, delta), (0, 0)))
        last_imag = jnp.pad(last_imag, ((0, delta), (0, 0)))

        fft_real = jnp.hstack([first_real, thin_real, last_real])
        fft_imag = jnp.hstack([first_imag, thin_imag, last_imag])
        return fft_real + 1j * fft_imag

    @staticmethod
    @jax.jit
    def pack_fft_values(values):
        ny, nx = values.shape
        assert ny == nx, 'Input array must be square'
        return jax.lax.cond(
            nx % 2 == 0,
            partial(PowerSpectrum._even_pack, n_pix=nx),
            partial(PowerSpectrum._odd_pack, n_pix=nx),
            jnp.sqrt(0.5) * values,
        )

    class TruncatedWedge(dist.Distribution):
        def __init__(self, a, low, b):
            self.a = a
            self.b = b
            self.low = low
            batch_shape = jax.lax.broadcast_shapes(
                jnp.shape(a),
                jnp.shape(low),
                jnp.shape(b),
            )
            self._support = dist.constraints.interval(low, b)
            self.norm = (self.b - self.a) ** 2 - (self.low - self.a) ** 2
            super().__init__(batch_shape=batch_shape, event_shape=())

        @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
        def support(self):
            return self._support

        def log_prob(self, value):
            return jnp.log(2) + jnp.log(value - self.a) - jnp.log(self.norm)

        def sample(self, key, sample_shape=()):
            shape = sample_shape + self.batch_shape
            u = jax.random.uniform(key, shape=shape, minval=0, maxval=1)
            return self.a + jnp.sqrt(self.norm * u + (self.low - self.a) ** 2)

    @staticmethod
    def matern_power_spectrum(
        plate_name,
        param_name,
        k,
        k_zero=None,
        n_high=100,
        n_value=None,
        sigma_value=None,
        rho_value=None,
        sigma_low=1e-5,
        sigma_high=10,
        positive=True,
        nonlinear_brightness=True,
        nonlinear_terms=10,
        log_brightness=False,
        log_mean_loc=-3.0,
        log_mean_scale=5.0,
        log_n_low=0.05,
        log_n_high=3.0,
        log_rho_low=0.5,
        log_rho_high=40.0,
        log_sigma_scale=4.0,
    ):
        with numpyro.plate(f'{plate_name} power spectrum params - [1]', 1):
            if n_value is None:
                n_prior = (
                    dist.LogUniform(log_n_low, log_n_high)
                    if log_brightness
                    else PowerSpectrum.TruncatedWedge(-1, 0.0001, n_high)
                )
                n = numpyro.sample(f'n_{param_name}', n_prior)
            else:
                n = numpyro.deterministic(f'n_{param_name}', jnp.atleast_1d(n_value))
            if sigma_value is None:
                sigma_prior = dist.HalfNormal(log_sigma_scale) if log_brightness else dist.LogUniform(sigma_low, sigma_high)
                sigma = numpyro.sample(f'sigma_{param_name}', sigma_prior)
            else:
                sigma = numpyro.deterministic(f'sigma_{param_name}', jnp.atleast_1d(sigma_value))
            if rho_value is None:
                rho_prior = dist.LogUniform(log_rho_low, log_rho_high) if log_brightness else dist.LogNormal(2.1, 1.1)
                rho = numpyro.sample(f'rho_{param_name}', rho_prior)
            else:
                rho = numpyro.deterministic(f'rho_{param_name}', jnp.atleast_1d(rho_value))
            if log_brightness:
                log_mean = numpyro.sample(f'log_mean_{param_name}', dist.Normal(log_mean_loc, log_mean_scale))

        P = PowerSpectrum.P_Matern(k, n[0], sigma[0], rho[0], k_zero=k_zero)
        scale = jnp.sqrt(P)

        ny, nx = scale.shape
        with numpyro.plate(f'{plate_name} fft y - [{ny}]', ny):
            with numpyro.plate(f'{plate_name} fft x - [{nx}]', nx):
                pixels_wn = numpyro.sample(
                    f'pixels_wn_{param_name}',
                    dist.Normal(0, 1),
                )

        gp = jnp.fft.irfft2(
            PowerSpectrum.pack_fft_values(pixels_wn * scale),
            s=scale.shape,
            norm='ortho',
        )
        if log_brightness:
            gp = jnp.exp(log_mean[0] + gp)
        elif positive:
            gp = jax.nn.softplus(100 * gp) / 100.0 + 1e-30
        if nonlinear_brightness and positive and not log_brightness:
            nonlinear_terms = int(nonlinear_terms)
            pow_lam = jnp.sort(numpyro.sample(
                f'pow_lam_{param_name}',
                dist.Uniform(
                    0.5 * jnp.ones((nonlinear_terms,)),
                    2.0 * jnp.ones((nonlinear_terms,)),
                ),
            ))
            scale_lam = numpyro.sample(
                f'scale_lam_{param_name}',
                dist.LogUniform(
                    1e-2 * jnp.ones((nonlinear_terms,)),
                    1e2 * jnp.ones((nonlinear_terms,)),
                ),
            )
            gp = PowerSpectrum.nonlinear_brightness_transform(gp, pow_lam, scale_lam)
        pixels = numpyro.deterministic(f'pixels_{param_name}', gp)
        return {'pixels': pixels}

    @staticmethod
    def nonlinear_brightness_transform(pixels, powers, amplitudes):
        powers = jnp.asarray(powers, dtype=jnp.float64).reshape((-1, 1, 1))
        amplitudes = jnp.asarray(amplitudes, dtype=jnp.float64).reshape((-1, 1, 1))
        return jnp.sum(amplitudes * jnp.power(pixels[None, :, :], powers), axis=0)

    @staticmethod
    def pixels_from_params(
        params,
        param_name,
        k_values,
        *,
        positive=True,
        n_value=None,
        sigma_value=None,
        rho_value=None,
        k_zero=None,
        nonlinear_brightness=True,
        log_brightness=False,
        log_mean_loc=-3.0,
        log_mean_scale=5.0,
        log_n_low=0.05,
        log_n_high=3.0,
        log_rho_low=0.5,
        log_rho_high=40.0,
        log_sigma_scale=4.0,
    ):
        pixel_key = f'pixels_{param_name}'
        if pixel_key in params:
            return jnp.asarray(params[pixel_key], dtype=jnp.float64)

        if f'n_{param_name}' in params:
            n = jnp.ravel(jnp.asarray(params[f'n_{param_name}']))[0]
        elif n_value is not None:
            n = jnp.asarray(n_value, dtype=jnp.float64)
        else:
            raise KeyError(f'Missing n_{param_name}; provide n_value when n is fixed in the model.')

        if sigma_value is None:
            sigma = jnp.ravel(jnp.asarray(params[f'sigma_{param_name}']))[0]
        else:
            sigma = jnp.asarray(sigma_value, dtype=jnp.float64)
        if rho_value is None:
            rho = jnp.ravel(jnp.asarray(params[f'rho_{param_name}']))[0]
        else:
            rho = jnp.asarray(rho_value, dtype=jnp.float64)
        pixels_wn = jnp.asarray(params[f'pixels_wn_{param_name}'], dtype=jnp.float64)
        scale = jnp.sqrt(PowerSpectrum.P_Matern(k_values, n, sigma, rho, k_zero=k_zero))
        pixels = jnp.fft.irfft2(PowerSpectrum.pack_fft_values(pixels_wn * scale), s=scale.shape, norm='ortho')
        if log_brightness:
            mean_key = f'log_mean_{param_name}'
            if mean_key not in params:
                raise KeyError(f'Missing {mean_key} for log-brightness Matérn reconstruction.')
            pixels = jnp.exp(jnp.ravel(jnp.asarray(params[mean_key]))[0] + pixels)
        elif positive:
            pixels = jax.nn.softplus(100 * pixels) / 100.0 + 1e-30
        if nonlinear_brightness and positive and not log_brightness:
            power_key = f'pow_lam_{param_name}'
            amplitude_key = f'scale_lam_{param_name}'
            has_power = power_key in params
            has_amplitude = amplitude_key in params
            if has_power != has_amplitude:
                raise KeyError(
                    f'Nonlinear brightness reconstruction requires both {power_key} and {amplitude_key}.'
                )
            if has_power:
                powers = jnp.sort(jnp.asarray(params[power_key], dtype=jnp.float64))
                amplitudes = jnp.asarray(params[amplitude_key], dtype=jnp.float64)
                pixels = PowerSpectrum.nonlinear_brightness_transform(pixels, powers, amplitudes)
        return pixels

    @staticmethod
    def params2kwargs_power_spectrum(
        params,
        param_name,
        k_values,
        *,
        positive=True,
        n_value=None,
        sigma_value=None,
        rho_value=None,
        k_zero=None,
        nonlinear_brightness=True,
        log_brightness=False,
        log_mean_loc=-3.0,
        log_mean_scale=5.0,
        log_n_low=0.05,
        log_n_high=3.0,
        log_rho_low=0.5,
        log_rho_high=40.0,
        log_sigma_scale=4.0,
    ):
        return {
            'pixels': PowerSpectrum.pixels_from_params(
                params,
                param_name,
                k_values,
                positive=positive,
                n_value=n_value,
                sigma_value=sigma_value,
                rho_value=rho_value,
                k_zero=k_zero,
                nonlinear_brightness=nonlinear_brightness,
                log_brightness=log_brightness,
            )
        }

    @staticmethod
    def fit_power_spectrum_init(
        image,
        k_values,
        pixelated_prior,
        seed,
        max_iterations=30000,
        learning_rate=0.01,
        noise_factor=0.001,
        progress_bar=True,
        param_name='source_grid',
    ):
        image = jnp.asarray(np.asarray(image, dtype=np.float64), dtype=jnp.float64)
        noise_level = max(noise_factor * float(np.nanmax(np.asarray(image))), 1e-6)

        def power_init_model(image_obs):
            source = PowerSpectrum.matern_power_spectrum(
                'Source grid',
                param_name,
                k_values,
                k_zero=pixelated_prior.get('power_init_k_zero', None),
                n_value=pixelated_prior.get('n_value'),
                sigma_low=float(pixelated_prior.get('sigma_low', 1e-5)),
                sigma_high=float(pixelated_prior.get('sigma_high', 10.0)),
                positive=bool(pixelated_prior.get('positive', True)),
                nonlinear_brightness=bool(pixelated_prior.get('nonlinear_brightness', True)),
                log_brightness=bool(pixelated_prior.get('log_brightness', False)),
                log_mean_loc=float(pixelated_prior.get('log_mean_loc', -3.0)),
                log_mean_scale=float(pixelated_prior.get('log_mean_scale', 5.0)),
                log_n_low=float(pixelated_prior.get('log_n_low', 0.05)),
                log_n_high=float(pixelated_prior.get('log_n_high', 3.0)),
                log_rho_low=float(pixelated_prior.get('log_rho_low', 0.5)),
                log_rho_high=float(pixelated_prior.get('log_rho_high', 40.0)),
                log_sigma_scale=float(pixelated_prior.get('log_sigma_scale', 4.0)),
            )
            numpyro.sample('obs', dist.Normal(source['pixels'], noise_level).to_event(2), obs=image_obs)

        guide = autoguide.AutoDiagonalNormal(power_init_model, init_loc_fn=infer.init_to_median(num_samples=25))
        scheduler = SVI.split_scheduler(
            max_iterations,
            init_value=learning_rate,
            decay_rates=(0.99, 0.995),
            transition_steps=(100, 25),
        )
        svi = infer.SVI(power_init_model, guide, optax.adabelief(learning_rate=scheduler), infer.TraceMeanField_ELBO())
        result = svi.run(
            jax.random.PRNGKey(seed),
            max_iterations,
            image,
            progress_bar=progress_bar,
            stable_update=True,
        )
        return guide.median(result.params)

    @staticmethod
    def fit_power_spectrum_init_from_parametric_source(
        lens_image,
        parametric_kwargs,
        num_pix,
        source_grid_scale,
        k_values,
        pixelated_prior,
        seed,
        max_iterations=1200,
        learning_rate=0.02,
        noise_factor=0.001,
        progress_bar=True,
        param_name='source_grid',
        source_plane=None,
    ):
        if parametric_kwargs is None:
            return {}

        if source_plane is None:
            source_image, _ = Plot.pixelize_plane(
                lens_image,
                parametric_kwargs,
                num_pix,
                source_grid_scale=source_grid_scale,
            )
        else:
            source_image, _ = Plot.pixelize_plane_multiplane(
                lens_image,
                parametric_kwargs,
                num_pix,
                source_grid_scale=source_grid_scale,
                N=source_plane,
            )
        return PowerSpectrum.fit_power_spectrum_init(
            source_image,
            k_values,
            pixelated_prior,
            seed=seed,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            noise_factor=noise_factor,
            progress_bar=progress_bar,
            param_name=param_name,
        )


class MultiplicativePSFCorrection:
    """Positive, unit-sum, zero-centroid multiplicative Matérn PSF correction."""
    def __init__(self, base_kernel, field_shape=(31, 31), log_clip=1.0):
        self.base = jnp.asarray(base_kernel, dtype=jnp.float64)
        self.base /= self.base.sum()
        self.log_clip = float(log_clip)
        y, x = jnp.indices(self.base.shape, dtype=jnp.float64)
        self.x, self.y = x-(self.base.shape[1]-1)/2, y-(self.base.shape[0]-1)/2
        self.basis = jnp.stack([jnp.ones_like(self.x), self.x, self.y], axis=-1).reshape(-1, 3)
        self.k = PowerSpectrum.K_grid(tuple(field_shape)).k

    def centroid(self, kernel):
        return jnp.array([jnp.sum(kernel*self.x), jnp.sum(kernel*self.y)])

    def project(self, field):
        coefficients = jnp.linalg.solve(self.basis.T@self.basis+1e-12*jnp.eye(3), self.basis.T@field.reshape(-1))
        return (field.reshape(-1)-self.basis@coefficients).reshape(field.shape)

    def recenter(self, kernel):
        kernel, tilt = jnp.maximum(kernel, 1e-30), jnp.zeros(2, dtype=jnp.float64)
        for _ in range(8):
            weights = kernel*jnp.exp(jnp.clip(tilt[0]*self.x+tilt[1]*self.y, -20, 20)); weights /= weights.sum()
            centroid = self.centroid(weights); x, y = self.x-centroid[0], self.y-centroid[1]
            covariance = jnp.array([[jnp.sum(weights*x*x), jnp.sum(weights*x*y)], [jnp.sum(weights*x*y), jnp.sum(weights*y*y)]])
            tilt -= jnp.linalg.solve(covariance+1e-12*jnp.eye(2), centroid)
        weights = kernel*jnp.exp(jnp.clip(tilt[0]*self.x+tilt[1]*self.y, -20, 20))
        return weights/weights.sum()

    def build(self, coarse):
        field = self.project(jax.image.resize(coarse, self.base.shape, method='linear'))
        return self.recenter(self.base*jnp.exp(jnp.clip(field, -self.log_clip, self.log_clip)))

    def sample(self):
        field = PowerSpectrum.matern_power_spectrum('PSF correction', 'psf_corr', self.k, k_zero=0.0,
            sigma_low=1e-4, sigma_high=0.5, positive=False, nonlinear_brightness=False)['pixels']
        return self.build(field)

    def from_params(self, params):
        field = PowerSpectrum.pixels_from_params(params, 'psf_corr', self.k, positive=False,
            k_zero=0.0, nonlinear_brightness=False)
        return self.build(field)


class Light:
    class ExternalLensLight:
        """Lens-light helper for optional external MGE start/fixed modes."""

        def __init__(self, model_config, project_root, param_name='lens'):
            self.model_config = model_config
            self.project_root = Path(project_root)
            self.param_name = param_name
            self.lens_cfg = self.model_config["light"][self.param_name]
            self.mode = str(self.lens_cfg.get("external_mode") or "free").strip().lower()
            self.stage_modes = {
                "parametric": str(
                    self.lens_cfg.get("parametric_mode") or self.mode
                ).strip().lower(),
                "pixelated": str(
                    self.lens_cfg.get("pixelated_mode") or self.mode
                ).strip().lower(),
            }
            self.kwargs = self._load_external_kwargs()

        def mode_for_stage(self, stage=None):
            if stage is None:
                return self.mode
            return self.stage_modes[stage]

        def _load_external_kwargs(self):
            requested_modes = {self.mode, *self.stage_modes.values()}
            if requested_modes.isdisjoint({"start", "fixed"}):
                return None
            path_value = self.lens_cfg.get("external_kwargs_path") or "lens_light_subtraction_result/kwargs_lens_light.pkl"
            path = Path(path_value)
            if not path.is_absolute():
                path = self.project_root / path
            if not path.exists():
                raise FileNotFoundError(
                    f"Lens light {self.mode} requested, but kwargs file does not exist: {path}"
                )
            with open(path, "rb") as handle:
                kwargs = pickle.load(handle)
            if isinstance(kwargs, dict):
                kwargs = [kwargs]
            if not isinstance(kwargs, (list, tuple)) or not kwargs:
                raise ValueError(f"Invalid lens light kwargs in {path}")

            first = kwargs[0]
            required = ("amp", "sigma", "e1", "e2", "center_x", "center_y")
            missing = [key for key in required if key not in first]
            if missing:
                raise ValueError(f"Lens light kwargs missing keys {missing}: {path}")

            normalized = [{
                "amp": jnp.asarray(first["amp"], dtype=jnp.float64),
                "sigma": jnp.asarray(first["sigma"], dtype=jnp.float64),
                "e1": jnp.asarray(first["e1"], dtype=jnp.float64),
                "e2": jnp.asarray(first["e2"], dtype=jnp.float64),
                "center_x": jnp.asarray(first["center_x"], dtype=jnp.float64),
                "center_y": jnp.asarray(first["center_y"], dtype=jnp.float64),
            }]
            self.lens_cfg["n_gauss"] = int(normalized[0]["sigma"].shape[0])
            print(
                f"Lens light external modes: parametric={self.stage_modes['parametric']}, "
                f"pixelated={self.stage_modes['pixelated']}; loaded "
                f"{self.lens_cfg['n_gauss']} Gaussian(s) from {path}",
                flush=True,
            )
            return normalized

        def init_values(self, stage=None):
            if self.mode_for_stage(stage) != "start" or self.kwargs is None:
                return {}
            kwargs = self.kwargs[0]
            sigma = jnp.asarray(kwargs["sigma"], dtype=jnp.float64)
            return {
                "amp_lens": jnp.asarray(kwargs["amp"], dtype=jnp.float64),
                "sigma_lens": sigma,
                "e_lens": jnp.stack([
                    jnp.asarray(kwargs["e1"], dtype=jnp.float64),
                    jnp.asarray(kwargs["e2"], dtype=jnp.float64),
                ]),
                "center_lens": jnp.stack([
                    jnp.asarray(kwargs["center_x"], dtype=jnp.float64),
                    jnp.asarray(kwargs["center_y"], dtype=jnp.float64),
                ]),
            }

        def for_model(self, stage=None):
            mode = self.mode_for_stage(stage)
            if mode == "fixed":
                return self.kwargs
            if mode == "start":
                kwargs = self.kwargs[0]
                n_gauss = int(self.lens_cfg["n_gauss"])
                center_max_offset = float(self.lens_cfg.get("center_max_offset", 0.4))
                with numpyro.plate(f"Lens light external start - [{n_gauss}]", n_gauss):
                    amp = numpyro.sample(
                        "amp_lens",
                        dist.LogNormal(jnp.log(jnp.maximum(kwargs["amp"], 1e-30)), 0.8),
                    )
                    sigma = numpyro.sample(
                        "sigma_lens",
                        dist.LogNormal(jnp.log(jnp.maximum(kwargs["sigma"], 1e-30)), 0.35),
                    )
                e0 = jnp.stack([kwargs["e1"], kwargs["e2"]])
                center0 = jnp.stack([kwargs["center_x"], kwargs["center_y"]])
                e = numpyro.sample(
                    "e_lens",
                    dist.TruncatedNormal(e0, 0.12, low=-0.6, high=0.6).to_event(2),
                )
                center = numpyro.sample(
                    "center_lens",
                    dist.TruncatedNormal(
                        center0,
                        0.08,
                        low=-center_max_offset,
                        high=center_max_offset,
                    ).to_event(2),
                )
                return [{
                    "amp": amp,
                    "sigma": sigma,
                    "e1": e[0],
                    "e2": e[1],
                    "center_x": center[0],
                    "center_y": center[1],
                }]
            return Light.multi_gauss_light(
                "Lens light",
                self.param_name,
                self.lens_cfg["n_gauss"],
                self.lens_cfg["sigma_lims"],
                center_low=self.lens_cfg["center_low"],
                center_high=self.lens_cfg["center_high"],
            )

        def kwargs_from_params(self, params, stage=None):
            mode = self.mode_for_stage(stage)
            if mode == "fixed":
                return self.kwargs
            if mode == "start":
                return [{
                    "amp": params["amp_lens"],
                    "sigma": params["sigma_lens"],
                    "e1": params["e_lens"][0],
                    "e2": params["e_lens"][1],
                    "center_x": params["center_lens"][0],
                    "center_y": params["center_lens"][1],
                }]
            return Light.params2kwargs_multi_gauss_light(
                params,
                self.param_name,
                self.lens_cfg["n_gauss"],
            )

    @staticmethod
    def multi_gauss_light(
        plate_name,
        param_name,
        n_gauss,
        sigma_lims,
        center_low=None,
        center_high=None,
        e_normal=0.2,
        e_low=None,
        e_high=None,
        amp_high=10000.0,
    ):
        sigma_bins = jnp.logspace(
            jnp.log10(sigma_lims[0]),
            jnp.log10(sigma_lims[1]),
            n_gauss + 1,
        )
        with numpyro.plate(f'{plate_name} - [{n_gauss}]', n_gauss):
            A = numpyro.sample(f'A_{param_name}', dist.LogUniform(1e-5, amp_high))
            sigma = numpyro.sample(
                f'sigma_{param_name}',
                dist.LogUniform(sigma_bins[:-1], sigma_bins[1:]),
            )
            with numpyro.plate(f'{plate_name} vectors - [2]', 2):
                e = numpyro.sample(
                    f'e_{param_name}',
                    dist.TruncatedNormal(0, e_normal, low=e_low, high=e_high),
                )
                if center_low is not None or center_high is not None:
                    center = numpyro.sample(
                        f'center_{param_name}',
                        dist.TruncatedNormal(0.0, 0.2, low=center_low, high=center_high),
                    )
                else:
                    center = numpyro.sample(f'center_{param_name}', dist.Normal(0.0, 0.5))

        amp = numpyro.deterministic(f'amp_{param_name}', A * sigma**2)
        return [{
            'amp': amp,
            'sigma': sigma,
            'e1': e[0],
            'e2': e[1],
            'center_x': center[0],
            'center_y': center[1],
        }]

    @staticmethod
    def params2kwargs_multi_gauss_light(params, param_name, n_gauss=None):
        sigma = params[f'sigma_{param_name}']
        amp = params.get(f'amp_{param_name}', params[f'A_{param_name}'] * sigma**2)
        center = params[f'center_{param_name}']
        e = params[f'e_{param_name}']
        return [{
            'amp': amp,
            'sigma': sigma,
            'e1': e[0],
            'e2': e[1],
            'center_x': center[0],
            'center_y': center[1],
        }]


class ModelHelper:
    @staticmethod
    def params2kwargs_EPL_w_shear_lens_light(
        params,
        source_kwargs,
        n_gauss_lens,
        lens_param_name='1',
        lens_light_param_name='lens',
    ):
        return {
            'kwargs_lens': Mass.params2kwargs_EPL_w_shear(params, lens_param_name),
            'kwargs_source': source_kwargs,
            'kwargs_lens_light': Light.params2kwargs_multi_gauss_light(
                params,
                lens_light_param_name,
                n_gauss_lens,
            ),
        }


class Numpyro_function:
    @staticmethod
    def split_normal_logpdf(x, mean, sigma_minus, sigma_plus):
        x = jnp.asarray(x, dtype=jnp.float64)
        mean = jnp.asarray(mean, dtype=jnp.float64)
        sigma_minus = jnp.asarray(sigma_minus, dtype=jnp.float64)
        sigma_plus = jnp.asarray(sigma_plus, dtype=jnp.float64)

        sigma = jnp.where(x < mean, sigma_minus, sigma_plus)
        log_norm = jnp.log(jnp.sqrt(2.0 / jnp.pi)) - jnp.log(sigma_minus + sigma_plus)
        return log_norm - 0.5 * ((x - mean) / sigma) ** 2


class SVI:
    @staticmethod
    def split_scheduler(
        max_iterations,
        init_value=0.01,
        decay_rates=(0.99, 0.99),
        transition_steps=(200, 10),
        boundary=0.5,
    ):
        boundary = int(max_iterations * boundary)
        scheduler1 = optax.exponential_decay(
            init_value=init_value,
            decay_rate=decay_rates[0],
            transition_steps=transition_steps[0],
        )
        scheduler2 = optax.exponential_decay(
            init_value=scheduler1(boundary),
            decay_rate=decay_rates[1],
            transition_steps=transition_steps[1],
        )
        return optax.join_schedules([scheduler1, scheduler2], boundaries=[boundary])

    @staticmethod
    def run_one_chain_svi(
        model,
        data_obs,
        *,
        max_iterations,
        seed,
        init_values=None,
        learning_rate=0.01,
        init_scale=0.1,
        loss_kind='trace_elbo',
        num_particles=10,
        model_args=(),
        progress_bar=True,
        progress_callback=None,
        progress_interval=None,
        init_strategy='median',
    ):
        if init_strategy not in {'median', 'sample'}:
            raise ValueError(f"Unknown SVI init_strategy: {init_strategy}")
        deferred_init = (
            infer.init_to_sample
            if init_strategy == 'sample'
            else infer.init_to_median(num_samples=25)
        )
        init_fun = (
            ResumeInit.init_to_value_or_defer(values=init_values, defer=deferred_init)
            if init_values
            else deferred_init
        )
        guide = autoguide.AutoLowRankMultivariateNormal(model, init_loc_fn=init_fun, init_scale=init_scale)
        scheduler = SVI.split_scheduler(max_iterations, init_value=learning_rate, transition_steps=(200, 10))
        optim = optax.adabelief(learning_rate=scheduler)
        if loss_kind == 'trace_meanfield_elbo':
            loss = infer.TraceMeanField_ELBO()
        elif loss_kind == 'trace_elbo':
            loss = infer.Trace_ELBO(num_particles=num_particles)
        else:
            raise ValueError(f"Unknown SVI loss_kind: {loss_kind}")
        svi = infer.SVI(model, guide, optim, loss)
        if progress_callback is None:
            result = svi.run(
                jax.random.PRNGKey(seed),
                max_iterations,
                data_obs,
                *model_args,
                progress_bar=progress_bar,
                stable_update=True,
            )
        else:
            def body_fn(svi_state):
                svi_state, loss_value = svi.stable_update(svi_state, data_obs, *model_args)
                return svi_state, loss_value

            update_fn = jax.jit(body_fn)
            svi_state = svi.init(jax.random.PRNGKey(seed), data_obs, *model_args)
            losses = []
            batch = int(progress_interval or max(max_iterations // 100, 1))
            progress_callback(0, max_iterations, None, None)
            for step in range(1, max_iterations + 1):
                svi_state, loss_value = update_fn(svi_state)
                loss_float = float(jax.device_get(loss_value))
                losses.append(loss_float)
                if step % batch == 0 or step == max_iterations:
                    recent = [value for value in losses[-batch:] if value == value]
                    avg_loss = float(np.mean(recent)) if recent else float('nan')
                    progress_callback(step, max_iterations, loss_float, avg_loss)
            result = infer.svi.SVIRunResult(svi.get_params(svi_state), svi_state, jnp.asarray(losses))
        median = guide.median(result.params)
        return {
            'result': result,
            'guide': guide,
            'median': median,
            'losses': np.asarray(result.losses),
        }


class Cosmo:
    c_km_s = 299792.458

    @staticmethod
    def sample_cosmology_from_prior(stage_kwargs, cosmo_priors):
        prior_name = stage_kwargs['cosmo_prior_name']
        prior = cosmo_priors[prior_name]
        cosmo_vec = numpyro.sample(
            'cosmo_vec',
            dist.MultivariateNormal(
                loc=jnp.asarray(prior['mean_vec'], dtype=jnp.float64),
                covariance_matrix=jnp.asarray(prior['cov'], dtype=jnp.float64),
            ),
        )
        omega_m = numpyro.deterministic('omega_m_cosmo', cosmo_vec[0])
        h0 = numpyro.deterministic('H0_cosmo', cosmo_vec[1])
        return {
            'Omegam': omega_m,
            'Omegak': jnp.asarray(0.0, dtype=jnp.float64),
            'w0': jnp.asarray(-1.0, dtype=jnp.float64),
            'wa': jnp.asarray(0.0, dtype=jnp.float64),
            'h0': h0,
        }

    @staticmethod
    def func(z, Omegam, Omegak, w0, wa=0):
        Omegal = 1.0 - Omegam - Omegak
        zp1 = 1.0 + z
        de = zp1 ** (3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * z / zp1)
        Ez2 = Omegam * zp1 ** 3 + Omegak * zp1 ** 2 + Omegal * de
        return Ez2 ** -0.5

    @staticmethod
    def nth_order_quad(n=20):
        xval, weights = map(jnp.array, roots_legendre(n))
        xval = xval.reshape(-1, 1)
        weights = weights.reshape(-1, 1)

        def integrate(func, a, b, *args):
            return 0.5 * (b - a) * jnp.sum(
                weights * func(0.5 * ((b - a) * xval + (b + a)), *args),
                axis=0,
            )

        return integrate

    @staticmethod
    def integrate(func, a, b, *args, n=20):
        quad = Cosmo.nth_order_quad(n)
        return quad(func, a, b, *args)

    @staticmethod
    def Dplus(Omegak, Es, El, zs, zl):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        Ds = jnp.sinh(sqrt_ok * Es) / sqrt_ok / (1 + zs)
        Dls = jnp.sinh(sqrt_ok * (Es - El)) / sqrt_ok / (1 + zs)
        Dl = jnp.sinh(sqrt_ok * El) / sqrt_ok / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dminus(Omegak, Es, El, zs, zl):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        Ds = jnp.sin(sqrt_ok * Es) / sqrt_ok / (1 + zs)
        Dls = jnp.sin(sqrt_ok * (Es - El)) / sqrt_ok / (1 + zs)
        Dl = jnp.sin(sqrt_ok * El) / sqrt_ok / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dflat(Es, El, zs, zl):
        Ds = Es / (1 + zs)
        Dls = (Es - El) / (1 + zs)
        Dl = El / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dpos(Omegak, E, z):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        return jnp.sinh(sqrt_ok * E) / sqrt_ok / (1 + z)

    @staticmethod
    def Dneg(Omegak, E, z):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        return jnp.sin(sqrt_ok * E) / sqrt_ok / (1 + z)

    @staticmethod
    def Dzero(E, z):
        return E / (1 + z)

    @staticmethod
    def angular_diameter_distance(z, cosmology, n=20):
        Omegam = cosmology["Omegam"]
        Omegak = cosmology["Omegak"]
        w0 = cosmology["w0"]
        wa = cosmology["wa"]
        h = cosmology["h0"]
        E = Cosmo.integrate(Cosmo.func, 0, z, Omegam, Omegak, w0, wa, n=n)

        Dl = lax.cond(
            Omegak > 0,
            lambda _: Cosmo.Dpos(Omegak, E, z),
            lambda _: lax.cond(
                Omegak < 0,
                lambda _: Cosmo.Dneg(Omegak, E, z),
                lambda _: Cosmo.Dzero(E, z),
                None,
            ),
            None,
        )
        return Dl * Cosmo.c_km_s / h

    @staticmethod
    def dldsdls(zl, zs, cosmology, n=20):
        Omegam = cosmology["Omegam"]
        Omegak = cosmology["Omegak"]
        w0 = cosmology["w0"]
        wa = cosmology["wa"]
        h = cosmology["h0"]

        El = Cosmo.integrate(Cosmo.func, 0, zl, Omegam, Omegak, w0, wa, n=n)
        Es = Cosmo.integrate(Cosmo.func, 0, zs, Omegam, Omegak, w0, wa, n=n)

        Dl, Ds, Dls = lax.cond(
            Omegak > 0,
            lambda _: Cosmo.Dplus(Omegak, Es, El, zs, zl),
            lambda _: lax.cond(
                Omegak < 0,
                lambda _: Cosmo.Dminus(Omegak, Es, El, zs, zl),
                lambda _: Cosmo.Dflat(Es, El, zs, zl),
                None,
            ),
            None,
        )
        return Dl * Cosmo.c_km_s / h, Ds * Cosmo.c_km_s / h, Dls * Cosmo.c_km_s / h

    @staticmethod
    def compute_time_delay_distances(cosmology, z_lens, z_source):
        dl, ds, dls = Cosmo.dldsdls(z_lens, z_source, cosmology)
        return (1.0 + z_lens) * dl * ds / dls


class GUIStatus:
    def __init__(self, status_path, job_dir):
        self.status_path = Path(status_path)
        self.job_dir = Path(job_dir)

    def write(self, **kwargs):
        payload = {}
        if self.status_path.exists():
            try:
                payload = json.loads(self.status_path.read_text(encoding='utf-8'))
            except Exception:
                payload = {}
        payload.update({'job_dir': str(self.job_dir), **kwargs})
        self.status_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return payload


def report_semilinear_progress(
    gui_status,
    fraction,
    message,
    overall_start=0.94,
    overall_span=0.05,
):
    fraction = float(fraction)
    overall_fraction = float(overall_start) + float(overall_span) * fraction
    gui_status.write(
        state='running',
        message=message,
        progress={'fraction': overall_fraction, 'message': message},
    )
    print(
        f'LENS_SEMILINEAR {100.0 * fraction:.1f}% {message}',
        flush=True,
    )


def import_function(namespace=None):
    if namespace is None:
        namespace = inspect.currentframe().f_back.f_globals

    namespace.update({
        'os': os,
        'sys': sys,
        'json': json,
        'pickle': pickle,
        'shutil': shutil,
        'warnings': warnings,
        'datetime': datetime,
        'Path': Path,
        'deepcopy': deepcopy,
        'np': np,
        'xr': xr,
        'jax': jax,
        'jnp': jnp,
        'numpyro': numpyro,
        'dist': dist,
        'infer': infer,
        'autoguide': autoguide,
        'optax': optax,
        'condition': condition,
        'az': az,
        'matplotlib': matplotlib,
        'plt': plt,
        'colors': colors,
        'GridSpec': GridSpec,
        'corner': corner,
        'fits': fits,
        'least_squares': least_squares,
        'Noise': Noise,
        'PSF': PSF,
        'LightModel': LightModel,
        'MPLightModel': MPLightModel,
        'MassModel': MassModel,
        'MPMassModel': MPMassModel,
        'MPLensImage': MPLensImage,
        'LensImageExtension': LensImageExtension,
        'PointSourceModel': PointSourceModel,
        'mass_model_base': mass_model_base,
        'CuspyNFW_3D_fn': CuspyNFW_3D_fn,
        'MGE': MGE,
        'Plot': Plot,
        'ResumeInit': ResumeInit,
        'Geometry': Geometry,
        'PowerSpectrum': PowerSpectrum,
        'Light': Light,
        'Mass': Mass,
        'ModelHelper': ModelHelper,
        'Numpyro_function': Numpyro_function,
        'SVI': SVI,
        'Cosmo': Cosmo,
        'GUIStatus': GUIStatus,
    })
    return namespace
