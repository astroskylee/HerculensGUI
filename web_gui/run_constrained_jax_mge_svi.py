#!/usr/bin/env python
"""Semi-linear JAX/NumPyro MGE lens-light SVI.

The starting point is a saved PyAutoLens-style linear MGE solution. The script
samples only shared MGE shape parameters in SVI. Gaussian sigmas are fixed and
amplitudes are solved by a positive-only JAX linear solver inside each model
evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if (candidate / "Tian_infra.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from Tian_infra import (  # noqa: E402
    Geometry,
    SVI as Tian_SVI,
    import_function,
    pixel_scale_arcsec_from_header,
)
from herculens.LensImage.lens_image import LensImage  # noqa: E402

import_function(globals())
jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

import jaxnnls  # noqa: E402

MIN_ELLIPTICITY_START = 1.0e-3


def load_mask(path, shape):
    if not path.exists():
        raise FileNotFoundError(f"Required mask file is missing: {path}")
    mask = np.asarray(fits.getdata(path)).astype(bool)
    if mask.shape != shape:
        raise ValueError(f"Mask shape mismatch for {path}: expected {shape}, got {mask.shape}")
    return mask


def corner_background(data, size=10):
    ny, nx = data.shape
    s = int(min(size, ny // 2, nx // 2))
    if s < 1:
        raise ValueError(f"Corner background size is invalid for data shape {data.shape}: {size}")
    values = np.concatenate([
        data[:s, :s].ravel(),
        data[:s, -s:].ravel(),
        data[-s:, :s].ravel(),
        data[-s:, -s:].ravel(),
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate corner background: no finite corner pixels.")
    return float(np.nanmedian(values))


def make_lens_image(data, header, psf_kernel):
    pix_scale = pixel_scale_arcsec_from_header(header)
    pixel_grid, xgrid, ygrid, x_axis, y_axis, extent, nx, ny = Geometry.get_pixel_grid(data, pix_scale)
    psf = PSF(psf_type="PIXEL", kernel_point_source=psf_kernel)
    noise = Noise(nx, ny, exposure_time=1.0)
    lens_image = LensImage(
        deepcopy(pixel_grid),
        deepcopy(psf),
        noise_class=noise,
        lens_light_model_class=LightModel(["MULTI_GAUSSIAN_ELLIPSE"]),
    )
    return lens_image, pix_scale, extent, np.asarray(xgrid), np.asarray(ygrid)


def load_basis_params(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    kwargs = payload["kwargs_lens_light"]
    return {
        "amp": np.asarray(kwargs["amp"], dtype=np.float64),
        "sigma": np.asarray(kwargs["sigma"], dtype=np.float64),
        "e1": np.asarray(kwargs["e1"], dtype=np.float64),
        "e2": np.asarray(kwargs["e2"], dtype=np.float64),
        "center_x": np.asarray(kwargs["center_x"], dtype=np.float64),
        "center_y": np.asarray(kwargs["center_y"], dtype=np.float64),
    }


def build_kwargs_from_arrays(amp, sigma, e1, e2, center_x, center_y):
    return [{
        "amp": amp,
        "sigma": sigma,
        "e1": e1,
        "e2": e2,
        "center_x": center_x,
        "center_y": center_y,
    }]


def convolved_lens_light(lens_image, kwargs_lens_light):
    return np.asarray(
        jax.device_get(
            lens_image.model(
                kwargs_lens_light=kwargs_lens_light,
                source_add=False,
                lens_light_add=True,
                point_source_add=False,
            )
        ),
        dtype=float,
    )


def lens_light_basis_cube(lens_image, sigma, e1, e2, center_x, center_y):
    n_gauss = int(sigma.shape[0])
    e1_array = jnp.full_like(sigma, e1)
    e2_array = jnp.full_like(sigma, e2)
    center_x_array = jnp.full_like(sigma, center_x)
    center_y_array = jnp.full_like(sigma, center_y)
    images = []
    for index in range(n_gauss):
        amp = jnp.zeros_like(sigma).at[index].set(1.0)
        basis_kwargs = build_kwargs_from_arrays(
            amp=amp,
            sigma=sigma,
            e1=e1_array,
            e2=e2_array,
            center_x=center_x_array,
            center_y=center_y_array,
        )
        images.append(
            lens_image.model(
                kwargs_lens_light=basis_kwargs,
                source_add=False,
                lens_light_add=True,
                point_source_add=False,
            )
        )
    return jnp.stack(images, axis=0)


def positive_linear_amplitudes_from_basis(
    basis_cube,
    data_obs,
    rms_map,
    fit_mask,
    *,
    ridge=1.0e-8,
    target_kappa=1.0e-11,
    active_rtol=1.0e-10,
    active_atol=0.0,
):
    basis_fit = jnp.swapaxes(basis_cube[:, fit_mask], 0, 1)
    inv_rms = 1.0 / rms_map[fit_mask]
    matrix = basis_fit * inv_rms[:, None]
    target = data_obs[fit_mask] * inv_rms

    q = matrix.T @ target
    n_gauss = matrix.shape[1]
    q_matrix = matrix.T @ matrix

    column_norm = jnp.linalg.norm(matrix, axis=0)
    max_column_norm = jnp.max(column_norm)
    active = column_norm > jnp.maximum(float(active_atol), float(active_rtol) * max_column_norm)
    active_float = active.astype(q_matrix.dtype)

    # Interior-point NNLS is unstable for effectively invisible basis columns.
    # Keep the static shape, but force inactive columns to solve to amp=0.
    q = q * active_float
    q_matrix = q_matrix * active_float[:, None] * active_float[None, :]
    diagonal_boost = jnp.where(active, float(ridge), 1.0)
    q_matrix = q_matrix + diagonal_boost * jnp.eye(n_gauss, dtype=q_matrix.dtype)

    diag = jnp.diag(q_matrix)
    scale = 1.0 / jnp.sqrt(diag)
    q_matrix_pc = (q_matrix * scale[:, None]) * scale[None, :]
    q_pc = q * scale
    amp_pc, _, _, _, _ = jaxnnls.solve_nnls(q_matrix_pc, q_pc)
    return amp_pc * scale


def resolve_sigma_max(sigma_max, data_shape, pix_scale, sigma_min):
    if sigma_max is None:
        return 0.5 * min(data_shape) * float(pix_scale)
    text = str(sigma_max).strip().lower()
    if text in {"", "auto", "half", "half_image", "half image", "none"}:
        return 0.5 * min(data_shape) * float(pix_scale)
    return max(float(sigma_max), float(sigma_min) * 1.01)


def initial_center_from_data(data_fit, fit_mask, xgrid, ygrid):
    valid = np.asarray(fit_mask, dtype=bool) & np.isfinite(data_fit)
    if np.any(valid):
        score = np.where(valid, data_fit, -np.inf)
    else:
        score = np.where(np.isfinite(data_fit), data_fit, -np.inf)
    if not np.isfinite(score).any():
        return 0.0, 0.0
    y_peak, x_peak = np.unravel_index(int(np.nanargmax(score)), score.shape)
    return float(xgrid[y_peak, x_peak]), float(ygrid[y_peak, x_peak])


def make_auto_basis_params(
    basis_path,
    *,
    args,
    lens_image,
    data_fit,
    data_obs,
    rms_map,
    fit_mask_np,
    fit_mask,
    pix_scale,
    xgrid,
    ygrid,
):
    n_gauss = max(1, int(getattr(args, "n_gauss", 8) or 8))
    sigma_min = max(float(getattr(args, "sigma_min", 0.01) or 0.01), 1.0e-6)
    sigma_max = resolve_sigma_max(getattr(args, "sigma_max", None), data_fit.shape, pix_scale, sigma_min)
    if n_gauss == 1:
        sigmas = np.asarray([sigma_min], dtype=np.float64)
    else:
        sigmas = np.geomspace(sigma_min, sigma_max, n_gauss).astype(np.float64)

    center_x_raw, center_y_raw = initial_center_from_data(data_fit, fit_mask_np, xgrid, ygrid)
    center_max_offset = max(float(getattr(args, "center_max_offset", 0.4) or 0.4), 1.0e-6)
    center_x0 = float(np.clip(center_x_raw, -center_max_offset, center_max_offset))
    center_y0 = float(np.clip(center_y_raw, -center_max_offset, center_max_offset))
    e1 = jnp.asarray(MIN_ELLIPTICITY_START, dtype=jnp.float64)
    e2 = jnp.asarray(MIN_ELLIPTICITY_START, dtype=jnp.float64)
    sigma_jax = jnp.asarray(sigmas, dtype=jnp.float64)
    search_radius = min(center_max_offset, 2.0 * float(pix_scale))
    candidate_offsets = np.linspace(-search_radius, search_radius, 5)
    candidate_center_x = np.unique(np.clip(center_x0 + candidate_offsets, -center_max_offset, center_max_offset))
    candidate_center_y = np.unique(np.clip(center_y0 + candidate_offsets, -center_max_offset, center_max_offset))
    best = None
    for center_x in candidate_center_x:
        for center_y in candidate_center_y:
            center_x = float(center_x)
            center_y = float(center_y)
            basis_cube = lens_light_basis_cube(
                lens_image,
                sigma_jax,
                e1,
                e2,
                jnp.asarray(center_x, dtype=jnp.float64),
                jnp.asarray(center_y, dtype=jnp.float64),
            )
            amp = positive_linear_amplitudes_from_basis(
                basis_cube,
                data_obs,
                rms_map,
                fit_mask,
                ridge=float(getattr(args, "linear_solver_ridge", 1.0e-8)),
                target_kappa=float(getattr(args, "linear_solver_target_kappa", 1.0e-11)),
                active_rtol=float(getattr(args, "linear_solver_active_rtol", 1.0e-10)),
                active_atol=float(getattr(args, "linear_solver_active_atol", 0.0)),
            )
            model = jnp.tensordot(amp, basis_cube, axes=(0, 0))
            residual = (data_obs - model) / rms_map
            objective = float(jax.device_get(jnp.nanmean(residual[fit_mask] ** 2)))
            if best is None or objective < best["objective"]:
                best = {
                    "center_x": center_x,
                    "center_y": center_y,
                    "amp": np.asarray(jax.device_get(amp), dtype=float),
                    "objective": objective,
                }

    if best is None:
        raise RuntimeError("Could not generate an automatic MGE basis.")

    kwargs_lens_light = {
        "amp": best["amp"].tolist(),
        "sigma": sigmas.tolist(),
        "e1": np.full(n_gauss, MIN_ELLIPTICITY_START, dtype=float).tolist(),
        "e2": np.full(n_gauss, MIN_ELLIPTICITY_START, dtype=float).tolist(),
        "center_x": np.full(n_gauss, best["center_x"], dtype=float).tolist(),
        "center_y": np.full(n_gauss, best["center_y"], dtype=float).tolist(),
    }
    basis_path.parent.mkdir(parents=True, exist_ok=True)
    basis_payload = {
        "auto_generated": True,
        "method": "logspace_sigma_shared_center_jaxnnls",
        "sigmas": sigmas.tolist(),
        "amps": best["amp"].tolist(),
        "kwargs_lens_light": kwargs_lens_light,
    }
    basis_path.write_text(json.dumps(basis_payload, indent=2), encoding="utf-8")
    metrics = {
        "auto_generated": True,
        "n_gauss": n_gauss,
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "center_max_offset": center_max_offset,
        "center_x_peak": float(center_x_raw),
        "center_y_peak": float(center_y_raw),
        "center_x_start": float(center_x0),
        "center_y_start": float(center_y0),
        "center_x": float(best["center_x"]),
        "center_y": float(best["center_y"]),
        "fit_pixels": int(np.asarray(fit_mask_np, dtype=bool).sum()),
        "nonzero_amplitudes": int(np.count_nonzero(best["amp"] > 0)),
        "max_amp": float(np.max(best["amp"])) if best["amp"].size else 0.0,
        "objective_mean_residual2": float(best["objective"]),
    }
    (basis_path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return basis_payload


def params_from_state(
    fixed,
    state,
    lens_image,
    data_obs,
    rms_map,
    fit_mask,
    linear_solver_ridge,
    linear_solver_target_kappa,
    linear_solver_active_rtol=1.0e-10,
    linear_solver_active_atol=0.0,
):
    sigma = np.asarray(fixed["sigma"], dtype=float)
    center_x_scalar = float(np.asarray(state["center_x_lens"], dtype=float))
    center_y_scalar = float(np.asarray(state["center_y_lens"], dtype=float))
    e1_scalar = float(np.asarray(state["e1_lens"], dtype=float))
    e2_scalar = float(np.asarray(state["e2_lens"], dtype=float))
    sigma_jax = jnp.asarray(sigma, dtype=jnp.float64)
    basis_cube = lens_light_basis_cube(
        lens_image,
        sigma_jax,
        jnp.asarray(e1_scalar, dtype=jnp.float64),
        jnp.asarray(e2_scalar, dtype=jnp.float64),
        jnp.asarray(center_x_scalar, dtype=jnp.float64),
        jnp.asarray(center_y_scalar, dtype=jnp.float64),
    )
    amp = np.asarray(
        jax.device_get(
            positive_linear_amplitudes_from_basis(
                basis_cube,
                data_obs,
                rms_map,
                fit_mask,
                ridge=linear_solver_ridge,
                target_kappa=linear_solver_target_kappa,
                active_rtol=linear_solver_active_rtol,
                active_atol=linear_solver_active_atol,
            )
        ),
        dtype=float,
    )
    e1 = np.full_like(sigma, e1_scalar, dtype=float)
    e2 = np.full_like(sigma, e2_scalar, dtype=float)
    center_x = np.full_like(sigma, center_x_scalar, dtype=float)
    center_y = np.full_like(sigma, center_y_scalar, dtype=float)
    return {
        "amp": amp,
        "sigma": sigma,
        "e1": e1,
        "e2": e2,
        "center_x": center_x,
        "center_y": center_y,
    }


def full_kwargs_from_state(state):
    return {
        "amp": np.asarray(state["amp_lens"], dtype=float),
        "sigma": np.asarray(state["sigma_lens"], dtype=float),
        "e1": np.asarray(state["e1_lens"], dtype=float),
        "e2": np.asarray(state["e2_lens"], dtype=float),
        "center_x": np.asarray(state["center_x_lens"], dtype=float),
        "center_y": np.asarray(state["center_y_lens"], dtype=float),
    }


class LensLightMgeSVIFitter:
    def __init__(self, args):
        self.args = args

    @classmethod
    def from_cli(cls):
        return cls(parse_args())

    def fit_lenslight(self, progress_callback=None):
        return _fit_lenslight_from_args(self.args, progress_callback=progress_callback)


def _fit_lenslight_from_args(args, progress_callback=None):
    center_max_offset_value = getattr(args, "center_max_offset", 0.4)
    if center_max_offset_value in ("", None):
        center_max_offset_value = 0.4
    center_max_offset = float(center_max_offset_value)
    if not np.isfinite(center_max_offset) or center_max_offset <= 0:
        raise ValueError("Gaussian center max offset must be a finite positive value in arcsec.")
    args.center_max_offset = center_max_offset

    project_dir = Path(args.project_dir).expanduser().resolve() if args.project_dir else SCRIPT_DIR
    init_dir = Path(args.init_dir).expanduser()
    basis_path = init_dir / "basis_params.json" if init_dir.is_absolute() else project_dir / init_dir / "basis_params.json"

    data, header = fits.getdata(project_dir / "Data_cutout.fits", header=True)
    data = np.asarray(data, dtype=np.float64)
    output_header = header.copy()
    if not isinstance(output_header.get("EXTNAME"), str):
        output_header.remove("EXTNAME", ignore_missing=True)

    mask_1 = load_mask(project_dir / "mask_1.fits", data.shape)
    mask_2 = load_mask(project_dir / "mask_2.fits", data.shape)
    mask_out = load_mask(project_dir / "mask_out.fits", data.shape)
    source_mask = mask_1 | mask_2
    fit_mask_np = ~(source_mask | mask_out)

    psf_kernel = np.asarray(fits.getdata(project_dir / "PSF_model.fits"), dtype=float)
    if psf_kernel.ndim != 2 or not np.all(np.isfinite(psf_kernel)):
        raise ValueError("PSF_model.fits must be a finite 2D array.")
    if np.any(psf_kernel < 0):
        raise ValueError("PSF_model.fits contains negative pixels.")
    psf_sum = float(psf_kernel.sum())
    if not np.isfinite(psf_sum) or psf_sum <= 0:
        raise ValueError("PSF_model.fits has a non-positive sum.")
    psf_kernel /= psf_sum

    rms_path = project_dir / "RMS_map.fits"
    if not rms_path.exists():
        raise FileNotFoundError(f"Required RMS map is missing: {rms_path}")
    rms_map_np = np.asarray(fits.getdata(rms_path), dtype=float)
    if rms_map_np.shape != data.shape:
        raise ValueError(f"RMS map shape mismatch: expected {data.shape}, got {rms_map_np.shape}")
    if not np.all(np.isfinite(rms_map_np)) or np.any(rms_map_np <= 0):
        raise ValueError("RMS_map.fits must be finite and strictly positive.")

    background = corner_background(data, size=args.corner_size) if args.background == "corner" else 0.0
    data_fit = data - background

    lens_image, pix_scale, extent, xgrid, ygrid = make_lens_image(data, header, psf_kernel)

    if not basis_path.exists():
        print(f"MGE basis file is missing; generating automatic basis: {basis_path}", flush=True)
        make_auto_basis_params(
            basis_path,
            args=args,
            lens_image=lens_image,
            data_fit=data_fit,
            data_obs=jnp.asarray(data_fit, dtype=jnp.float64),
            rms_map=jnp.asarray(rms_map_np, dtype=jnp.float64),
            fit_mask_np=fit_mask_np,
            fit_mask=jnp.asarray(fit_mask_np, dtype=bool),
            pix_scale=pix_scale,
            xgrid=xgrid,
            ygrid=ygrid,
        )
    fixed_np = load_basis_params(basis_path)
    n_gauss = int(fixed_np["amp"].size)
    if n_gauss < 1:
        raise RuntimeError("No Gaussian components found in basis_params.json.")
    expected_shape = (n_gauss,)
    for key, value in fixed_np.items():
        if value.shape != expected_shape:
            raise ValueError(f"Basis parameter {key!r} has shape {value.shape}; expected {expected_shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Basis parameter {key!r} contains non-finite values.")
    if np.any(fixed_np["sigma"] <= 0):
        raise ValueError("Basis sigma values must be strictly positive.")

    fixed = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in fixed_np.items()}
    data_obs = jnp.asarray(data_fit, dtype=jnp.float64)
    fit_mask = jnp.asarray(fit_mask_np, dtype=bool)
    rms_map = jnp.asarray(rms_map_np, dtype=jnp.float64)
    linear_solver_ridge = float(getattr(args, "linear_solver_ridge", 1.0e-8))
    linear_solver_target_kappa = float(getattr(args, "linear_solver_target_kappa", 1.0e-11))
    linear_solver_active_rtol = float(getattr(args, "linear_solver_active_rtol", 1.0e-10))
    linear_solver_active_atol = float(getattr(args, "linear_solver_active_atol", 0.0))

    center_x_start_value = float(np.nanmedian(fixed_np["center_x"]))
    center_y_start_value = float(np.nanmedian(fixed_np["center_y"]))
    center_max_offset = max(float(getattr(args, "center_max_offset", 0.4) or 0.4), 1.0e-6)
    if abs(center_x_start_value) > center_max_offset or abs(center_y_start_value) > center_max_offset:
        raise ValueError(
            "Semilinear MGE center lies outside the configured Gaussian center max offset: "
            f"center=({center_x_start_value:.6g}, {center_y_start_value:.6g}), "
            f"max={center_max_offset:.6g} arcsec."
        )
    e1_start_value = MIN_ELLIPTICITY_START
    e2_start_value = MIN_ELLIPTICITY_START
    if abs(e1_start_value) >= args.e_abs_max or abs(e2_start_value) >= args.e_abs_max:
        raise ValueError("MIN_ELLIPTICITY_START must be smaller than e_abs_max.")

    center_x_start = jnp.asarray(center_x_start_value, dtype=jnp.float64)
    center_y_start = jnp.asarray(center_y_start_value, dtype=jnp.float64)
    e1_start = jnp.asarray(e1_start_value, dtype=jnp.float64)
    e2_start = jnp.asarray(e2_start_value, dtype=jnp.float64)

    npix_fit = int(fit_mask_np.sum())

    def model_lens_light(data_obs):
        center_x_delta = numpyro.sample(
            "center_x_delta_lens",
            dist.TruncatedNormal(
                0.0,
                args.center_delta_sigma,
                low=-center_max_offset - center_x_start,
                high=center_max_offset - center_x_start,
            ),
        )
        center_y_delta = numpyro.sample(
            "center_y_delta_lens",
            dist.TruncatedNormal(
                0.0,
                args.center_delta_sigma,
                low=-center_max_offset - center_y_start,
                high=center_max_offset - center_y_start,
            ),
        )
        e1 = numpyro.sample(
            "e1_lens",
            dist.TruncatedNormal(
                e1_start,
                args.e_delta_sigma,
                low=-args.e_abs_max,
                high=args.e_abs_max,
            ),
        )
        e2 = numpyro.sample(
            "e2_lens",
            dist.TruncatedNormal(
                e2_start,
                args.e_delta_sigma,
                low=-args.e_abs_max,
                high=args.e_abs_max,
            ),
        )
        center_x = numpyro.deterministic("center_x_lens", center_x_start + center_x_delta)
        center_y = numpyro.deterministic("center_y_lens", center_y_start + center_y_delta)

        basis_cube = lens_light_basis_cube(lens_image, fixed["sigma"], e1, e2, center_x, center_y)
        amp = jax.lax.stop_gradient(
            positive_linear_amplitudes_from_basis(
                basis_cube,
                data_obs,
                rms_map,
                fit_mask,
                ridge=linear_solver_ridge,
                target_kappa=linear_solver_target_kappa,
                active_rtol=linear_solver_active_rtol,
                active_atol=linear_solver_active_atol,
            ),
        )
        numpyro.deterministic("amp_lens", amp)
        sigma = numpyro.deterministic("sigma_lens", fixed["sigma"])
        model_image = jnp.tensordot(amp, basis_cube, axes=(0, 0))
        numpyro.deterministic("lens_light_model", model_image)
        with numpyro.plate(f"Lens light fit pixels - [{npix_fit}]", npix_fit):
            numpyro.sample("obs", dist.Normal(model_image[fit_mask], rms_map[fit_mask]), obs=data_obs[fit_mask])

    init_values = {
        "center_x_delta_lens": np.asarray(0.0, dtype=np.float64),
        "center_y_delta_lens": np.asarray(0.0, dtype=np.float64),
        "e1_lens": np.asarray(e1_start_value, dtype=np.float64),
        "e2_lens": np.asarray(e2_start_value, dtype=np.float64),
    }

    tag = args.tag or "free_shape_e03_jax_svi"
    out_dir = project_dir / "lens_light_constrained_jax_mge_svi" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    semilinear_mge_steps = max(0, int(args.steps))
    total_svi_steps = semilinear_mge_steps + max(0, int(getattr(args, "unconstrained_steps", 0)))
    config = {
        "tag": tag,
        "mode": "semi_linear_jaxnnls",
        "project_dir": str(project_dir),
        "init_basis_path": str(basis_path),
        "n_gauss": n_gauss,
        "semilinear_mge_steps": int(semilinear_mge_steps),
        "constrained_steps": int(semilinear_mge_steps),
        "unconstrained_steps": max(0, int(getattr(args, "unconstrained_steps", 0))),
        "steps": int(semilinear_mge_steps) + max(0, int(getattr(args, "unconstrained_steps", 0))),
        "learning_rate": float(args.learning_rate),
        "init_scale": float(args.init_scale),
        "center_delta_sigma": float(args.center_delta_sigma),
        "center_max_offset": center_max_offset,
        "e_delta_sigma": float(args.e_delta_sigma),
        "e_abs_max": float(args.e_abs_max),
        "fixed_sigmas": [float(value) for value in fixed_np["sigma"]],
        "linear_solver": "jaxnnls.solve_nnls",
        "linear_solver_ridge": float(linear_solver_ridge),
        "linear_solver_target_kappa": float(linear_solver_target_kappa),
        "linear_solver_active_rtol": float(linear_solver_active_rtol),
        "linear_solver_active_atol": float(linear_solver_active_atol),
        "linear_solver_gradient": "stop_gradient_envelope",
        "background_mode": args.background,
        "background_subtracted_for_fit": float(background),
        "fit_pixels": int(npix_fit),
        "source_mask_pixels": int(source_mask.sum()),
        "mask_out_pixels": int(mask_out.sum()),
        "initial_e1": float(e1_start_value),
        "initial_e2": float(e2_start_value),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    init_params = params_from_state(
        fixed_np,
        {
            "center_x_lens": center_x_start_value,
            "center_y_lens": center_y_start_value,
            "e1_lens": e1_start_value,
            "e2_lens": e2_start_value,
        },
        lens_image,
        data_obs,
        rms_map,
        fit_mask,
        linear_solver_ridge,
        linear_solver_target_kappa,
        linear_solver_active_rtol,
        linear_solver_active_atol,
    )
    init_kwargs = build_kwargs_from_arrays(**init_params)
    init_model = convolved_lens_light(lens_image, init_kwargs)
    init_residual = (data_fit - init_model) / rms_map_np

    def emit_stage_progress(stage_name, offset, step, total, latest_loss=None, avg_loss=None):
        if progress_callback is None:
            return
        try:
            progress_callback(
                step,
                total,
                latest_loss,
                avg_loss,
                stage=stage_name,
                offset=offset,
                total_steps=max(total_svi_steps, 1),
            )
        except TypeError:
            progress_callback(offset + step, max(total_svi_steps, 1), latest_loss, avg_loss)

    state = None
    constrained_losses = 0
    if semilinear_mge_steps > 0:
        state = Tian_SVI.run_one_chain_svi(
            model_lens_light,
            data_obs,
            max_iterations=semilinear_mge_steps,
            seed=args.seed,
            init_values=init_values,
            learning_rate=args.learning_rate,
            init_scale=args.init_scale,
            progress_bar=True,
            progress_callback=lambda step, total, latest_loss=None, avg_loss=None: emit_stage_progress(
                "constrained", 0, step, total, latest_loss, avg_loss
            ),
            progress_interval=max(1, semilinear_mge_steps // 100),
            loss_kind="trace_elbo",
            num_particles=args.num_particles,
        )
        constrained_losses = len(state["losses"])
        final_params = params_from_state(
            fixed_np,
            jax.device_get(state["median"]),
            lens_image,
            data_obs,
            rms_map,
            fit_mask,
            linear_solver_ridge,
            linear_solver_target_kappa,
            linear_solver_active_rtol,
            linear_solver_active_atol,
        )
    else:
        final_params = init_params
    final_kwargs = build_kwargs_from_arrays(**final_params)
    final_model = convolved_lens_light(lens_image, final_kwargs)
    final_subtracted = data - final_model
    final_residual = (data_fit - final_model) / rms_map_np

    constrained_kwargs = final_kwargs
    constrained_model = final_model
    constrained_subtracted = final_subtracted
    constrained_residual = final_residual

    unconstrained_steps = max(0, int(getattr(args, "unconstrained_steps", 0)))
    unconstrained_state = None
    if unconstrained_steps > 0:
        start = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in final_kwargs[0].items()}
        amp_start = jnp.maximum(start["amp"], 1.0e-12)
        sigma_start = jnp.maximum(start["sigma"], 1.0e-6)

        def model_full_lens_light(data_obs):
            with numpyro.plate(f"Full lens light MGE - [{n_gauss}]", n_gauss):
                amp = numpyro.sample("amp_lens", dist.LogNormal(jnp.log(amp_start), 0.5))
                sigma = numpyro.sample("sigma_lens", dist.LogNormal(jnp.log(sigma_start), 0.2))
                e1 = numpyro.sample(
                    "e1_lens",
                    dist.TruncatedNormal(start["e1"], 0.08, low=-args.e_abs_max, high=args.e_abs_max),
                )
                e2 = numpyro.sample(
                    "e2_lens",
                    dist.TruncatedNormal(start["e2"], 0.08, low=-args.e_abs_max, high=args.e_abs_max),
                )
                center_x = numpyro.sample(
                    "center_x_lens",
                    dist.TruncatedNormal(
                        start["center_x"],
                        args.center_delta_sigma,
                        low=-center_max_offset,
                        high=center_max_offset,
                    ),
                )
                center_y = numpyro.sample(
                    "center_y_lens",
                    dist.TruncatedNormal(
                        start["center_y"],
                        args.center_delta_sigma,
                        low=-center_max_offset,
                        high=center_max_offset,
                    ),
                )
            kwargs_lens_light = build_kwargs_from_arrays(
                amp=amp,
                sigma=sigma,
                e1=e1,
                e2=e2,
                center_x=center_x,
                center_y=center_y,
            )
            model_image = lens_image.model(
                kwargs_lens_light=kwargs_lens_light,
                source_add=False,
                lens_light_add=True,
                point_source_add=False,
            )
            numpyro.deterministic("lens_light_model", model_image)
            with numpyro.plate(f"Full lens light fit pixels - [{npix_fit}]", npix_fit):
                numpyro.sample("obs", dist.Normal(model_image[fit_mask], rms_map[fit_mask]), obs=data_obs[fit_mask])

        unconstrained_init = {
            "amp_lens": np.asarray(np.maximum(final_kwargs[0]["amp"], 1.0e-12), dtype=np.float64),
            "sigma_lens": np.asarray(np.maximum(final_kwargs[0]["sigma"], 1.0e-6), dtype=np.float64),
            "e1_lens": np.asarray(final_kwargs[0]["e1"], dtype=np.float64),
            "e2_lens": np.asarray(final_kwargs[0]["e2"], dtype=np.float64),
            "center_x_lens": np.asarray(final_kwargs[0]["center_x"], dtype=np.float64),
            "center_y_lens": np.asarray(final_kwargs[0]["center_y"], dtype=np.float64),
        }
        unconstrained_state = Tian_SVI.run_one_chain_svi(
            model_full_lens_light,
            data_obs,
            max_iterations=unconstrained_steps,
            seed=args.seed + 1000,
            init_values=unconstrained_init,
            learning_rate=args.learning_rate,
            init_scale=args.init_scale,
            progress_bar=True,
            progress_callback=lambda step, total, latest_loss=None, avg_loss=None: emit_stage_progress(
                "unconstrained", semilinear_mge_steps, step, total, latest_loss, avg_loss
            ),
            progress_interval=max(1, unconstrained_steps // 100),
            loss_kind="trace_elbo",
            num_particles=args.num_particles,
        )
        unconstrained_kwargs = build_kwargs_from_arrays(**full_kwargs_from_state(jax.device_get(unconstrained_state["median"])))
        final_kwargs = unconstrained_kwargs
        final_model = convolved_lens_light(lens_image, final_kwargs)
        final_subtracted = data - final_model
        final_residual = (data_fit - final_model) / rms_map_np

    fits.writeto(out_dir / "init_lens_light_model.fits", init_model.astype(np.float32), header=output_header, overwrite=True)
    fits.writeto(out_dir / "init_fit_residual_over_rms.fits", init_residual.astype(np.float32), header=output_header, overwrite=True)
    fits.writeto(out_dir / "constrained_lens_light_model.fits", constrained_model.astype(np.float32), header=output_header, overwrite=True)
    fits.writeto(out_dir / "constrained_lens_light_subtracted.fits", constrained_subtracted.astype(np.float32), header=output_header, overwrite=True)
    fits.writeto(out_dir / "constrained_fit_residual_over_rms.fits", constrained_residual.astype(np.float32), header=output_header, overwrite=True)
    fits.writeto(out_dir / "svi_lens_light_model.fits", final_model.astype(np.float32), header=output_header, overwrite=True)
    fits.writeto(out_dir / "svi_lens_light_subtracted.fits", final_subtracted.astype(np.float32), header=output_header, overwrite=True)
    fits.writeto(out_dir / "svi_fit_residual_over_rms.fits", final_residual.astype(np.float32), header=output_header, overwrite=True)
    fits.writeto(out_dir / "fit_mask.fits", fit_mask_np.astype(np.uint8), header=output_header, overwrite=True)
    with open(out_dir / "kwargs_lens_light_constrained.pkl", "wb") as handle:
        pickle.dump(constrained_kwargs, handle)
    with open(out_dir / "kwargs_lens_light.pkl", "wb") as handle:
        pickle.dump(final_kwargs, handle)

    if unconstrained_state is not None:
        final_stage_label = "SVI"
    elif semilinear_mge_steps > 0:
        final_stage_label = "Semilinear MGE SVI"
    else:
        final_stage_label = "Semilinear MGE"
    panels = [
        (data, "Data", "magma", None),
        (init_model, "Init model", "magma", None),
        (final_model, f"{final_stage_label} model", "magma", None),
        (data - init_model, "Data - init", "twilight", None),
        (final_subtracted, f"Data - {final_stage_label}", "twilight", None),
        (final_residual, "Residual / RMS", "bwr", (-5, 5)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=170)
    for ax, (image, title, cmap, limits) in zip(axes.ravel(), panels):
        kwargs = {"origin": "lower", "extent": extent, "cmap": cmap, "interpolation": "nearest"}
        if limits is not None:
            kwargs.update({"vmin": limits[0], "vmax": limits[1]})
        im = ax.imshow(image, **kwargs)
        if mask_1.any():
            ax.contour(mask_1, levels=[0.5], colors="#9933ff", linestyles="dashed", origin="lower", extent=extent, linewidths=1.0)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.tick_params(labelsize=8)
        if title == "Residual / RMS":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{tag}: semilinear MGE + optional full MGE SVI, e1/e2 in +/-{args.e_abs_max:g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic.png", bbox_inches="tight")
    plt.close(fig)
    result = {
        "output_dir": str(out_dir),
        "kwargs_lens_light": str(out_dir / "kwargs_lens_light.pkl"),
        "diagnostic": str(out_dir / "diagnostic.png"),
        "model_fits": str(out_dir / "svi_lens_light_model.fits"),
        "subtracted_fits": str(out_dir / "svi_lens_light_subtracted.fits"),
        "residual_fits": str(out_dir / "svi_fit_residual_over_rms.fits"),
        "fit_mask_fits": str(out_dir / "fit_mask.fits"),
        "constrained_model_fits": str(out_dir / "constrained_lens_light_model.fits"),
        "constrained_subtracted_fits": str(out_dir / "constrained_lens_light_subtracted.fits"),
        "constrained_residual_fits": str(out_dir / "constrained_fit_residual_over_rms.fits"),
        "constrained_kwargs_lens_light": str(out_dir / "kwargs_lens_light_constrained.pkl"),
        "semilinear_mge_steps": int(semilinear_mge_steps),
        "constrained_steps": int(semilinear_mge_steps),
        "constrained_losses": constrained_losses,
        "unconstrained_steps": unconstrained_steps,
        "unconstrained_losses": len(unconstrained_state["losses"]) if unconstrained_state is not None else 0,
        "center_max_offset": center_max_offset,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run semi-linear JAX NNLS MGE with optional SVI and e1/e2 bounded to +/-0.3.")
    parser.add_argument("--tag", default="semi_linear_jaxnnls_mge_svi")
    parser.add_argument(
        "--mode",
        choices=("free_shape", "semi_linear_jaxnnls"),
        default="semi_linear_jaxnnls",
        help="Compatibility option; all modes run the semi-linear JAX NNLS solver.",
    )
    parser.add_argument(
        "--project-dir",
        default="",
        help="Project folder containing Data_cutout.fits, PSF_model.fits, RMS_map.fits, and masks.",
    )
    parser.add_argument(
        "--init-dir",
        default="lens_light_linear_mge_pyautolens_style/gauss30_smin0030_centeropt_nnls",
    )
    parser.add_argument("--n-gauss", type=int, default=30, help="Legacy compatibility; ignored.")
    parser.add_argument("--sigma-min", type=float, default=0.03, help="Legacy compatibility; ignored.")
    parser.add_argument("--sigma-max", default=None, help="Legacy compatibility; ignored.")
    parser.add_argument("--steps", type=int, default=0, help="Semilinear MGE SVI steps; 0 uses the direct JAX NNLS solution.")
    parser.add_argument("--unconstrained-steps", type=int, default=0, help="Full MGE SVI steps after the semilinear solution.")
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--init-scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num-particles", type=int, default=10)
    parser.add_argument("--amp-delta-sigma", type=float, default=0.8, help="Legacy compatibility; ignored.")
    parser.add_argument("--sigma-delta-sigma", type=float, default=0.35, help="Legacy compatibility; ignored.")
    parser.add_argument("--sigma-delta-limit", type=float, default=1.0, help="Legacy compatibility; ignored.")
    parser.add_argument("--center-delta-sigma", type=float, default=0.08)
    parser.add_argument(
        "--center-max-offset",
        "--center-delta-limit",
        dest="center_max_offset",
        type=float,
        default=0.4,
        help="Maximum absolute Gaussian center x/y coordinate in arcsec relative to the image center.",
    )
    parser.add_argument("--e-delta-sigma", type=float, default=0.12)
    parser.add_argument("--e-abs-max", type=float, default=0.3)
    parser.add_argument("--linear-solver-ridge", type=float, default=1.0e-8)
    parser.add_argument("--linear-solver-target-kappa", type=float, default=1.0e-11)
    parser.add_argument("--linear-solver-active-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-solver-active-atol", type=float, default=0.0)
    parser.add_argument("--background", choices=("corner", "none"), default="corner")
    parser.add_argument("--corner-size", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    LensLightMgeSVIFitter.from_cli().fit_lenslight()
