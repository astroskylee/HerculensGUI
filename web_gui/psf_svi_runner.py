import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from astropy.io import fits

GUI_ROOT = Path(__file__).resolve().parents[1]
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import numpyro.infer.autoguide as autoguide
import optax
from scipy.optimize import least_squares
from scipy.ndimage import shift as scipy_shift
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.PointSourceModel.point_source_model import PointSourceModel
from Tian_infra import PowerSpectrum

jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

PSF_POS_FLOOR = 1e-20
XPOS_PRIOR_SIGMA = 0.08
YPOS_PRIOR_SIGMA = 0.08
XPOS_BOUNDS = (-0.75, 0.75)
YPOS_BOUNDS = (-0.75, 0.75)
PSF_FWHM_FALLBACK = 1.6
MOFFAT_BETA_DEFAULT = 3.0
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
PSF_CORR_SIGMA_LOW = 1e-9
PSF_CORR_SIGMA_HIGH = 1e-4
PSF_CORR_PIXEL_SIGMA = 1e-4
PSF_STAGE1_CORR_SIGMA_LOW = 1e-7
PSF_STAGE1_CORR_SIGMA_HIGH = 5e-3
PSF_CORR_MODEL_DEFAULT = "pixel"
PSF_POSITIVE_TRANSFORM_DEFAULT = "hard_clip"
PSF_SOFTPLUS_TEMPERATURE = 1e-5
PSF_MULT_LOG_CLIP = 12.0


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def update_status(output_dir, **kwargs):
    status_path = Path(output_dir) / "status.json"
    payload = {}
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload.update(kwargs)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(status_path, payload)


def load_image(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=float)
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data = np.squeeze(np.asarray(hdu.data, dtype=float))
                if data.ndim > 2:
                    data = data[0]
                return data
    raise ValueError(f"No image data found in {path}")


def normalize_kernel(kernel, eps=1e-20):
    kernel = np.nan_to_num(np.asarray(kernel, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    kernel = np.clip(kernel, 0.0, np.inf)
    total = float(np.sum(kernel))
    if total <= eps:
        raise ValueError("Kernel normalization failed.")
    return kernel / total


def normalize_signed_kernel(kernel, eps=1e-20):
    kernel = np.nan_to_num(np.asarray(kernel, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(kernel))
    if abs(total) <= eps:
        raise ValueError("Signed kernel normalization failed.")
    return kernel / total


def center_crop_or_pad(arr, target_shape):
    ty, tx = target_shape
    ay, ax = arr.shape
    if ay > ty:
        y0 = (ay - ty) // 2
        arr = arr[y0 : y0 + ty, :]
    if ax > tx:
        x0 = (ax - tx) // 2
        arr = arr[:, x0 : x0 + tx]
    ay, ax = arr.shape
    if ay < ty or ax < tx:
        py0 = (ty - ay) // 2
        py1 = ty - ay - py0
        px0 = (tx - ax) // 2
        px1 = tx - ax - px0
        arr = np.pad(arr, ((py0, py1), (px0, px1)), mode="constant", constant_values=0.0)
    return arr


def fixed_cutout(data, x_center, y_center, size):
    half = size // 2
    x_center = int(round(float(x_center)))
    y_center = int(round(float(y_center)))
    y0, y1 = y_center - half, y_center + half + 1
    x0, x1 = x_center - half, x_center + half + 1
    if x0 < 0 or y0 < 0 or x1 > data.shape[1] or y1 > data.shape[0]:
        return None
    cutout = np.asarray(data[y0:y1, x0:x1], dtype=float)
    if cutout.shape != (size, size):
        return None
    return cutout


def estimate_background_and_error(cutout, bg_stats=None):
    if bg_stats is not None:
        bkg = float(bg_stats["background"])
        sigma = max(float(bg_stats["sigma"]), 1e-8)
        source_term = np.clip(np.nan_to_num(cutout - bkg, nan=0.0, posinf=0.0, neginf=0.0), 0.0, np.inf)
        err = np.sqrt(sigma**2 + source_term)
        err = np.clip(err, max(0.25 * sigma, 1e-8), np.inf)
        return bkg, err, sigma

    border = np.concatenate([cutout[:5, :].ravel(), cutout[-5:, :].ravel(), cutout[:, :5].ravel(), cutout[:, -5:].ravel()])
    finite = np.isfinite(border)
    if not np.any(finite):
        finite_cut = cutout[np.isfinite(cutout)]
        bkg = float(np.nanmedian(finite_cut)) if finite_cut.size else 0.0
        sigma = float(np.nanstd(finite_cut)) if finite_cut.size else 1.0
    else:
        vals = border[finite]
        bkg = float(np.nanmedian(vals))
        mad = float(np.nanmedian(np.abs(vals - bkg)))
        sigma = max(1.4826 * mad, 1e-8)
    source_term = np.clip(np.nan_to_num(cutout - bkg, nan=0.0, posinf=0.0, neginf=0.0), 0.0, np.inf)
    err = np.sqrt(sigma**2 + source_term)
    err = np.clip(err, max(0.25 * sigma, 1e-8), np.inf)
    return bkg, err, sigma


def extract_box(data, x_center, y_center, size):
    size = max(4, int(round(float(size))))
    half = size // 2
    x_center = int(round(float(x_center)))
    y_center = int(round(float(y_center)))
    x0 = max(0, x_center - half)
    x1 = min(data.shape[1], x0 + size)
    x0 = max(0, x1 - size)
    y0 = max(0, y_center - half)
    y1 = min(data.shape[0], y0 + size)
    y0 = max(0, y1 - size)
    return np.asarray(data[y0:y1, x0:x1], dtype=float), {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def background_stats_from_config(data, config):
    bg_box = config.get("bg_box") or None
    if not bg_box:
        return None
    box, bounds = extract_box(data, bg_box["x"], bg_box["y"], bg_box["size"])
    finite = box[np.isfinite(box)]
    if finite.size == 0:
        raise ValueError("Selected background box has no finite pixels.")
    bkg = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - bkg)))
    sigma = max(1.4826 * mad, 1e-8)
    return {
        "background": bkg,
        "sigma": sigma,
        "bounds": bounds,
        "size": int(bg_box["size"]),
        "x": int(bg_box["x"]),
        "y": int(bg_box["y"]),
    }


def selected_cutouts(config):
    data = load_image(config["image_path"])
    finite = np.isfinite(data)
    fill = float(np.nanmedian(data[finite])) if np.any(finite) else 0.0
    data = np.where(finite, data, fill)
    bg_stats = background_stats_from_config(data, config)
    size = int(config["kernel_size"])
    star_map = {int(star["id"]): star for star in config.get("stars", [])}
    background_subtracted = bool(config.get("background_subtracted", False))

    sci, err, mask, used = [], [], [], []
    x_loc, y_loc = [], []
    flux_loc, bkg_loc, bkg_scale = [], [], []
    for star_id in config["selected_ids"]:
        star = star_map.get(int(star_id))
        if not star:
            continue
        x_center = int(round(float(star["x"])))
        y_center = int(round(float(star["y"])))
        cutout = fixed_cutout(data, star["x"], star["y"], size)
        if cutout is None:
            continue
        bkg, err_i, sigma_i = estimate_background_and_error(cutout, bg_stats=bg_stats)
        sci_clean = np.nan_to_num(cutout, nan=bkg, posinf=bkg, neginf=bkg)
        sci_i = sci_clean - bkg if background_subtracted else sci_clean
        flux_est = float(np.sum(np.clip(sci_clean - bkg, 0.0, None)))
        sci.append(sci_i)
        err.append(err_i)
        mask.append(np.isfinite(cutout).astype(float))
        used.append(int(star_id))
        x_loc.append(float(star["x"]) - x_center)
        y_loc.append(float(star["y"]) - y_center)
        flux_loc.append(max(flux_est, 1e-8))
        bkg_loc.append(0.0 if background_subtracted else bkg)
        bkg_scale.append(max(sigma_i, 1e-8))
    if not sci:
        raise ValueError("No selected stars could be extracted.")
    return (
        np.stack(sci),
        np.stack(err),
        np.stack(mask),
        used,
        np.asarray(x_loc, dtype=float),
        np.asarray(y_loc, dtype=float),
        np.asarray(flux_loc, dtype=float),
        np.asarray(bkg_loc, dtype=float),
        np.asarray(bkg_scale, dtype=float),
        bg_stats,
    )


def estimate_cutout_fwhm(cutout, bg_stats=None, background_subtracted=False):
    if background_subtracted:
        bkg = 0.0
        sigma = float(bg_stats["sigma"]) if bg_stats else float(np.nanstd(cutout))
    else:
        bkg, _, sigma = estimate_background_and_error(cutout, bg_stats=bg_stats)
    signal = np.nan_to_num(np.asarray(cutout, dtype=float) - bkg, nan=0.0, posinf=0.0, neginf=0.0)
    signal = np.clip(signal, 0.0, np.inf)
    peak = float(np.nanmax(signal))
    if not np.isfinite(peak) or peak <= 0.0:
        return None

    core_threshold = max(3.0 * float(sigma), 0.02 * peak)
    core = np.where(signal > core_threshold, signal, 0.0)
    total_core = float(np.sum(core))
    if total_core <= 0.0:
        core = signal
        total_core = float(np.sum(core))
    if total_core <= 0.0:
        return None

    yy, xx = np.indices(signal.shape, dtype=float)
    cx = float(np.sum(core * xx) / total_core)
    cy = float(np.sum(core * yy) / total_core)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    fit_mask = signal > max(3.0 * float(sigma), 0.05 * peak)
    if np.count_nonzero(fit_mask) >= 4:
        x = r2[fit_mask].reshape(-1)
        y = np.log(np.clip(signal[fit_mask].reshape(-1), PSF_POS_FLOOR, np.inf))
        w = np.clip(signal[fit_mask].reshape(-1) / peak, 1e-3, np.inf)
        design = np.stack([np.ones_like(x), x], axis=1)
        lhs = design.T @ (w[:, None] * design)
        rhs = design.T @ (w * y)
        try:
            _, slope = np.linalg.solve(lhs, rhs)
            if np.isfinite(slope) and slope < 0.0:
                sigma_fit = np.sqrt(-0.5 / slope)
                fwhm_fit = sigma_fit / FWHM_TO_SIGMA
                if 0.3 <= fwhm_fit <= min(cutout.shape):
                    return float(fwhm_fit)
        except np.linalg.LinAlgError:
            pass

    sigma_moment = np.sqrt(float(np.sum(core * r2) / max(2.0 * total_core, PSF_POS_FLOOR)))
    fwhm_moment = sigma_moment / FWHM_TO_SIGMA
    if np.isfinite(fwhm_moment) and fwhm_moment > 0.0:
        return float(fwhm_moment)
    return None


def estimate_stack_fwhm(sci_stack, bg_stats=None, fallback=PSF_FWHM_FALLBACK, background_subtracted=False):
    estimates = [
        value
        for value in (estimate_cutout_fwhm(cutout, bg_stats=bg_stats, background_subtracted=background_subtracted) for cutout in sci_stack)
        if value is not None and np.isfinite(value) and value > 0.0
    ]
    if not estimates:
        return float(fallback)
    high = max(0.5, min(sci_stack.shape[-2:]) / 2.0)
    return float(np.clip(np.nanmedian(estimates), 0.3, high))


def centered_gaussian_kernel(shape, fwhm):
    ny, nx = shape
    sigma = max(float(fwhm) * FWHM_TO_SIGMA, 1e-6)
    yy, xx = np.indices((ny, nx), dtype=float)
    cx = (nx - 1.0) / 2.0
    cy = (ny - 1.0) / 2.0
    radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
    kernel = np.exp(-0.5 * radius2 / sigma**2)
    kernel = np.maximum(kernel, PSF_POS_FLOOR)
    return normalize_kernel(kernel)


def shifted_moffat_kernel(shape, fwhm, beta=MOFFAT_BETA_DEFAULT, x0=None, y0=None):
    ny, nx = shape
    beta = max(float(beta), 1.01)
    alpha = max(float(fwhm) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0)), 1e-6)
    yy, xx = np.indices((ny, nx), dtype=float)
    cx = (nx - 1.0) / 2.0 if x0 is None else float(x0)
    cy = (ny - 1.0) / 2.0 if y0 is None else float(y0)
    radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
    kernel = (1.0 + radius2 / alpha**2) ** (-beta)
    kernel = np.maximum(kernel, PSF_POS_FLOOR)
    return normalize_kernel(kernel)


def centered_moffat_kernel(shape, fwhm, beta=MOFFAT_BETA_DEFAULT):
    return shifted_moffat_kernel(shape, fwhm, beta=beta)


def fit_cutout_moffat_params(cutout, bg_stats=None, background_subtracted=False):
    if background_subtracted:
        bkg = 0.0
    else:
        bkg, _, _ = estimate_background_and_error(cutout, bg_stats=bg_stats)
    target = np.nan_to_num(np.asarray(cutout, dtype=float) - bkg, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.clip(target, 0.0, np.inf)
    if float(np.sum(target)) <= 0.0:
        return None
    target = target / np.sum(target)

    ny, nx = target.shape
    yy, xx = np.indices(target.shape, dtype=float)
    total = float(np.sum(target))
    x0 = float(np.sum(target * xx) / total)
    y0 = float(np.sum(target * yy) / total)
    fwhm0 = estimate_cutout_fwhm(cutout, bg_stats=bg_stats, background_subtracted=background_subtracted) or PSF_FWHM_FALLBACK
    beta0 = MOFFAT_BETA_DEFAULT
    alpha0 = fwhm0 / (2.0 * np.sqrt(2.0 ** (1.0 / beta0) - 1.0))

    positive = target[target > 0]
    floor = float(np.nanpercentile(positive, 5)) if positive.size else 1e-8
    weight = 1.0 / np.sqrt(np.maximum(target, floor))
    weight = weight / np.nanmedian(weight)

    def moffat(params):
        xc, yc, log_alpha, log_beta_minus_one = params
        alpha = np.exp(log_alpha)
        beta = 1.0 + np.exp(log_beta_minus_one)
        rr2 = (xx - xc) ** 2 + (yy - yc) ** 2
        model = (1.0 + rr2 / alpha**2) ** (-beta)
        return model / np.sum(model)

    def residual(params):
        return ((moffat(params) - target) / weight).ravel()

    try:
        result = least_squares(
            residual,
            x0=np.asarray([x0, y0, np.log(alpha0), np.log(beta0 - 1.0)]),
            bounds=(
                [x0 - 3.0, y0 - 3.0, np.log(0.2), np.log(1e-3)],
                [x0 + 3.0, y0 + 3.0, np.log(max(nx, ny)), np.log(100.0)],
            ),
            max_nfev=1000,
        )
    except Exception:
        return None

    alpha = float(np.exp(result.x[2]))
    beta = float(1.0 + np.exp(result.x[3]))
    fwhm = float(2.0 * alpha * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    if not np.isfinite(fwhm) or not np.isfinite(beta) or fwhm <= 0.0:
        return None
    return {
        "x0": float(result.x[0]),
        "y0": float(result.x[1]),
        "fwhm": fwhm,
        "beta": beta,
        "cost": float(result.cost),
    }


def estimate_stack_moffat_params(sci_stack, bg_stats=None, background_subtracted=False):
    params = [fit_cutout_moffat_params(cutout, bg_stats=bg_stats, background_subtracted=background_subtracted) for cutout in sci_stack]
    params = [p for p in params if p is not None]
    if not params:
        return None
    return {
        "x0": float(np.nanmedian([p["x0"] for p in params])),
        "y0": float(np.nanmedian([p["y0"] for p in params])),
        "fwhm": float(np.nanmedian([p["fwhm"] for p in params])),
        "beta": float(np.nanmedian([p["beta"] for p in params])),
        "cost": float(np.nanmedian([p["cost"] for p in params])),
    }


def recenter_kernel_to_frame(kernel, x0_det, y0_det, factor=1):
    if x0_det is None or y0_det is None:
        return normalize_kernel(kernel)
    factor = max(1, int(factor))
    ny, nx = kernel.shape
    cx = (nx - 1.0) / 2.0
    cy = (ny - 1.0) / 2.0
    x0 = (float(x0_det) + 0.5) * factor - 0.5
    y0 = (float(y0_det) + 0.5) * factor - 0.5
    shifted = scipy_shift(
        np.asarray(kernel, dtype=float),
        shift=(cy - y0, cx - x0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return normalize_kernel(shifted)


def weighted_centroid_offset(kernel, factor=1):
    kernel = np.nan_to_num(np.asarray(kernel, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    factor = max(1, int(factor))
    ny, nx = kernel.shape
    yy, xx = np.indices(kernel.shape, dtype=float)
    cx = (nx - 1.0) / 2.0
    cy = (ny - 1.0) / 2.0
    total = float(np.sum(kernel))
    if abs(total) <= 1e-20:
        return {
            "x": cx,
            "y": cy,
            "dx_ss": 0.0,
            "dy_ss": 0.0,
            "dx_det": 0.0,
            "dy_det": 0.0,
            "sum": total,
        }
    x = float(np.sum(kernel * xx) / total)
    y = float(np.sum(kernel * yy) / total)
    dx_ss = x - cx
    dy_ss = y - cy
    return {
        "x": x,
        "y": y,
        "dx_ss": float(dx_ss),
        "dy_ss": float(dy_ss),
        "dx_det": float(dx_ss / factor),
        "dy_det": float(dy_ss / factor),
        "sum": total,
    }


def recenter_kernel_weighted_centroid(kernel, factor=1):
    before = weighted_centroid_offset(kernel, factor=factor)
    shifted = scipy_shift(
        np.asarray(kernel, dtype=float),
        shift=(-before["dy_ss"], -before["dx_ss"]),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    shifted = normalize_kernel(shifted)
    after = weighted_centroid_offset(shifted, factor=factor)
    return shifted, {"before": before, "after": after}


def build_correction_init(sci_stack, base_psf_det, factor=1, base_psf_ss=None):
    corrections = []
    factor = max(1, int(factor))
    base_psf_det = normalize_kernel(base_psf_det)
    if base_psf_ss is not None:
        base_psf_ss = normalize_kernel(base_psf_ss)
    for cutout in sci_stack:
        target = np.nan_to_num(np.asarray(cutout, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        target = np.clip(target, 0.0, np.inf)
        total = float(np.sum(target))
        if total <= 0.0:
            continue
        target = target / total
        if factor > 1 and base_psf_ss is not None:
            target = supersample_repeat_np(target, factor) / (factor**2)
            corr = target - base_psf_ss
        else:
            corr = target - base_psf_det
        corr = corr - np.mean(corr)
        corrections.append(corr)
    if not corrections:
        return np.zeros_like(base_psf_ss if base_psf_ss is not None else base_psf_det)
    corr = np.median(np.stack(corrections), axis=0)
    corr = corr - np.mean(corr)
    if corr.shape == base_psf_det.shape:
        corr_ss = supersample_repeat_np(corr, factor) / (factor**2)
    else:
        corr_ss = corr
    corr_ss = corr_ss - np.mean(corr_ss)
    return corr_ss


def project_zero_moments_np(corr, eps=1e-12):
    corr = np.nan_to_num(np.asarray(corr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ny, nx = corr.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    xx = xx - (nx - 1.0) / 2.0
    yy = yy - (ny - 1.0) / 2.0
    basis = np.stack([np.ones_like(corr).ravel(), xx.ravel(), yy.ravel()], axis=1)
    data = corr.ravel()
    coeff = np.linalg.solve(basis.T @ basis + eps * np.eye(3), basis.T @ data)
    return (data - basis @ coeff).reshape(ny, nx)


def build_rendered_correction_init(sci_stack, base_psf_ss, x_pos, y_pos, flux, factor=1, remove_moments=True):
    corrections = []
    factor = max(1, int(factor))
    base_psf_ss = normalize_signed_kernel(base_psf_ss)
    render_single_star = make_render_functions(get_pixel_grid(sci_stack.shape[-2:], pix_scale=1.0), ss_factor=factor)
    for cutout, x_i, y_i, flux_i in zip(sci_stack, x_pos, y_pos, flux):
        target = np.nan_to_num(np.asarray(cutout, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        unit_base = np.asarray(
            jax.device_get(
                render_single_star(
                    jnp.asarray(base_psf_ss, dtype=jnp.float64),
                    float(x_i),
                    float(y_i),
                    1.0,
                    0.0,
                )
            ),
            dtype=float,
        )
        residual_det = target / max(float(flux_i), 1e-8) - unit_base
        residual_ss = supersample_repeat_np(residual_det, factor) / (factor**2)
        corr = scipy_shift(
            residual_ss,
            shift=(-float(y_i) * factor, -float(x_i) * factor),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        corr = project_zero_moments_np(corr) if remove_moments else corr - np.mean(corr)
        corrections.append(corr)
    corr = np.median(np.stack(corrections), axis=0)
    return project_zero_moments_np(corr) if remove_moments else corr - np.mean(corr)


def build_base_psf(config, target_det_shape, sci_stack, bg_stats=None, factor=1):
    base_path = str(config.get("base_psf_path") or "").strip()
    if base_path:
        base_raw = load_image(base_path)
        base_ss, base_det, base_mode = prepare_base_psf(base_raw, target_det_shape, factor)
        return base_ss, base_det, base_path, base_mode, None, None, None, None

    profile = str(config.get("analytic_psf_profile", "moffat")).lower()
    background_subtracted = bool(config.get("background_subtracted", True))
    fallback_fwhm = float(config.get("fwhm", PSF_FWHM_FALLBACK))
    target_ss_shape = (target_det_shape[0] * factor, target_det_shape[1] * factor)
    if profile == "gaussian":
        fwhm_det = estimate_stack_fwhm(sci_stack, bg_stats=bg_stats, fallback=fallback_fwhm, background_subtracted=background_subtracted)
        base_ss = centered_gaussian_kernel(target_ss_shape, fwhm_det * factor)
        beta = None
        x0_det = None
        y0_det = None
        mode = "centered_gaussian"
        source = f"centered_gaussian_fwhm_{fwhm_det:.4g}px"
    else:
        moffat_params = estimate_stack_moffat_params(sci_stack, bg_stats=bg_stats, background_subtracted=background_subtracted) or {}
        fwhm_det = float(config.get("moffat_fwhm", moffat_params.get("fwhm", fallback_fwhm)))
        beta = float(config.get("moffat_beta", moffat_params.get("beta", MOFFAT_BETA_DEFAULT)))
        if bool(config.get("psf_model_centered", True)):
            x0_det = (target_det_shape[1] - 1.0) / 2.0
            y0_det = (target_det_shape[0] - 1.0) / 2.0
        else:
            x0_det = float(config.get("moffat_x0", moffat_params.get("x0", (target_det_shape[1] - 1.0) / 2.0)))
            y0_det = float(config.get("moffat_y0", moffat_params.get("y0", (target_det_shape[0] - 1.0) / 2.0)))
        x0_ss = (x0_det + 0.5) * factor - 0.5
        y0_ss = (y0_det + 0.5) * factor - 0.5
        base_ss = shifted_moffat_kernel(target_ss_shape, fwhm_det * factor, beta=beta, x0=x0_ss, y0=y0_ss)
        mode = "centered_moffat" if bool(config.get("psf_model_centered", True)) else "shifted_moffat"
        source = f"{mode}_fwhm_{fwhm_det:.4g}px_beta_{beta:.4g}_x{x0_det:.3f}_y{y0_det:.3f}"
    base_det = normalize_kernel(downsample_mean_np(base_ss, factor))
    return base_ss, base_det, source, mode, fwhm_det, beta, x0_det, y0_det


def psf_initialization_stack(config, sci_stack, used_ids):
    strategy = str(config.get("psf_init_strategy", "first_selected")).lower()
    base_path = str(config.get("base_psf_path") or "").strip()
    if strategy == "first_selected" and not base_path:
        return sci_stack[:1], 0, int(used_ids[0]), "first_selected"
    return sci_stack, None, None, "stack"


def model_position_priors(config, x_loc, y_loc, reference_index=None):
    reference_mode = str(config.get("psf_position_reference", "absolute")).lower()
    if reference_mode == "absolute":
        return x_loc, y_loc
    if bool(config.get("psf_base_uses_cutout_position", True)):
        if reference_index is None:
            return np.zeros_like(x_loc), np.zeros_like(y_loc)
        return x_loc - x_loc[reference_index], y_loc - y_loc[reference_index]
    return x_loc, y_loc


def downsample_mean_np(kernel_ss, factor=1):
    if factor == 1:
        return np.asarray(kernel_ss)
    ny_ss, nx_ss = kernel_ss.shape
    return kernel_ss.reshape(ny_ss // factor, factor, nx_ss // factor, factor).mean(axis=(1, 3))


def supersample_repeat_np(kernel_det, factor=1):
    if factor == 1:
        return np.asarray(kernel_det)
    return np.repeat(np.repeat(kernel_det, factor, axis=0), factor, axis=1)


def prepare_base_psf(base_raw, target_det_shape, factor=1):
    target_ss = (target_det_shape[0] * factor, target_det_shape[1] * factor)
    if base_raw.shape == target_ss:
        base_ss = normalize_kernel(base_raw)
        base_det = normalize_kernel(downsample_mean_np(base_ss, factor))
        return base_ss, base_det, "input_already_supersampled"
    if factor > 1 and base_raw.shape[0] % factor == 0 and base_raw.shape[1] % factor == 0:
        base_ss = normalize_kernel(center_crop_or_pad(base_raw, target_ss))
        base_det = normalize_kernel(downsample_mean_np(base_ss, factor))
        return base_ss, base_det, "input_supersampled_recentered"
    base_det = normalize_kernel(center_crop_or_pad(base_raw, target_det_shape))
    base_ss = normalize_kernel(supersample_repeat_np(base_det, factor))
    return base_ss, base_det, "detector_to_supersampled"


def downsample_mean(kernel_ss, factor=1):
    if factor == 1:
        return kernel_ss
    ny_ss, nx_ss = kernel_ss.shape
    return kernel_ss.reshape(ny_ss // factor, factor, nx_ss // factor, factor).mean(axis=(1, 3))


def downsample_sum(image_ss, factor=1):
    if factor == 1:
        return image_ss
    ny_ss, nx_ss = image_ss.shape
    return image_ss.reshape(ny_ss // factor, factor, nx_ss // factor, factor).sum(axis=(1, 3))


def normalize_kernel_jax(kernel):
    kernel = jnp.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
    kernel = jnp.clip(kernel, 0.0, jnp.inf)
    total = jnp.sum(kernel)
    return jnp.where(total > 0.0, kernel / total, kernel)


def normalize_signed_kernel_jax(kernel):
    kernel = jnp.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
    total = jnp.sum(kernel)
    return kernel / total


def get_pixel_grid(shape, pix_scale=1.0):
    ny, nx = shape
    half_size_x = nx * pix_scale / 2
    half_size_y = ny * pix_scale / 2
    kwargs_pixel = {
        "nx": nx,
        "ny": ny,
        "ra_at_xy_0": -half_size_x + pix_scale / 2,
        "dec_at_xy_0": -half_size_y + pix_scale / 2,
        "transform_pix2angle": pix_scale * np.eye(2),
    }
    return PixelGrid(**kwargs_pixel)


def split_scheduler(max_iterations, init_value=0.01, decay_rates=(0.99, 0.99), transition_steps=(200, 20), boundary=0.5):
    boundary = int(max_iterations * boundary)
    scheduler1 = optax.exponential_decay(init_value=init_value, decay_rate=decay_rates[0], transition_steps=transition_steps[0])
    scheduler2 = optax.exponential_decay(scheduler1(boundary), decay_rate=decay_rates[1], transition_steps=transition_steps[1])
    return optax.join_schedules([scheduler1, scheduler2], boundaries=[boundary])


def project_zero_moments(corr, eps=1e-12):
    ny, nx = corr.shape
    yy, xx = jnp.indices((ny, nx), dtype=corr.dtype)
    xx = xx - (nx - 1) / 2.0
    yy = yy - (ny - 1) / 2.0
    basis = jnp.stack([jnp.ones_like(corr).reshape(-1), xx.reshape(-1), yy.reshape(-1)], axis=1)
    data = corr.reshape(-1)
    coeff = jnp.linalg.solve(basis.T @ basis + eps * jnp.eye(3, dtype=corr.dtype), basis.T @ data)
    return (data - basis @ coeff).reshape(ny, nx)


def positive_psf_transform(raw_psf, mode=PSF_POSITIVE_TRANSFORM_DEFAULT, temperature=PSF_SOFTPLUS_TEMPERATURE):
    if mode == "none":
        return normalize_signed_kernel_jax(raw_psf)
    elif mode == "hard_clip":
        psf = jnp.maximum(raw_psf, 0.0)
    else:
        temperature = jnp.asarray(max(float(temperature), PSF_POS_FLOOR), dtype=raw_psf.dtype)
        psf = jax.nn.softplus(raw_psf / temperature) * temperature + PSF_POS_FLOOR
    return psf / jnp.sum(psf)


def positive_psf_transform_np(raw_psf, mode=PSF_POSITIVE_TRANSFORM_DEFAULT, temperature=PSF_SOFTPLUS_TEMPERATURE):
    raw_psf = np.asarray(raw_psf, dtype=float)
    if mode == "none":
        return normalize_signed_kernel(raw_psf)
    elif mode == "hard_clip":
        psf = np.maximum(raw_psf, 0.0)
    else:
        temperature = max(float(temperature), PSF_POS_FLOOR)
        psf = np.logaddexp(0.0, raw_psf / temperature) * temperature + PSF_POS_FLOOR
    return normalize_kernel(psf)


def normalize_for_positivity_mode(kernel, mode):
    if str(mode) == "none":
        return normalize_signed_kernel(kernel)
    return normalize_kernel(kernel)


def sample_psf_correction(
    corr_model,
    k_values,
    corr_sigma_low,
    corr_sigma_high,
    corr_pixel_sigma,
):
    if corr_model != "matern":
        ny, nx = k_values.shape
        sigma = numpyro.deterministic("sigma_psf_corr_pixel", jnp.asarray(corr_pixel_sigma))
        with numpyro.plate(f"PSF correction pixel y - [{ny}]", ny):
            with numpyro.plate(f"PSF correction pixel x - [{nx}]", nx):
                pixels_wn = numpyro.sample("pixels_wn_psf_corr_pixel", dist.Normal(0, 1))
        return numpyro.deterministic("pixels_psf_corr", sigma * pixels_wn)

    corr_dict = PowerSpectrum.matern_power_spectrum(
        "PSF correction",
        "psf_corr",
        k_values,
        n_value=None,
        sigma_low=corr_sigma_low,
        sigma_high=corr_sigma_high,
        positive=False,
    )
    return corr_dict["pixels"]


def make_render_functions(pixel_grid, ss_factor=1):
    point_source_model = PointSourceModel(["IMAGE_POSITIONS"])
    ss_factor = max(1, int(ss_factor))
    det_nx, det_ny = pixel_grid.num_pixel_axes
    render_grid = pixel_grid
    if ss_factor > 1:
        render_grid = get_pixel_grid((det_ny * ss_factor, det_nx * ss_factor), pix_scale=1.0 / ss_factor)
    render_nx, render_ny = render_grid.num_pixel_axes
    x_center_offset = (0.5 / ss_factor) if render_nx % 2 == 0 else 0.0
    y_center_offset = (0.5 / ss_factor) if render_ny % 2 == 0 else 0.0

    def render_point_sources_from_kernel(kernel_ss, theta_x, theta_y, amplitude):
        theta_x = jnp.atleast_1d(theta_x)
        theta_y = jnp.atleast_1d(theta_y)
        amplitude = jnp.atleast_1d(amplitude)
        x_pix, y_pix = render_grid.map_coord2pix(theta_x + x_center_offset, theta_y + y_center_offset)
        kernel_t = kernel_ss.T
        xrange = jnp.arange(render_nx) + kernel_t.shape[0] // 2
        yrange = jnp.arange(render_ny) + kernel_t.shape[1] // 2
        result_ss = jnp.zeros((render_nx, render_ny), dtype=kernel_ss.dtype)
        for x0, y0, amp in zip(x_pix, y_pix, amplitude):
            xy_grid = jnp.meshgrid(xrange - x0, yrange - y0)
            result_ss = result_ss + amp * map_coordinates(kernel_t, xy_grid, order=1, mode="nearest")
        return downsample_sum(result_ss, factor=ss_factor)

    def render_single_star(kernel_ss, x_pos, y_pos, flux, background):
        kwargs_point_source = [{"ra": x_pos, "dec": y_pos, "amp": flux}]
        theta_x_list, theta_y_list, amp_list = point_source_model.get_multiple_images(
            kwargs_point_source,
            kwargs_lens=None,
            kwargs_solver=None,
            k=0,
            with_amplitude=True,
            zero_amp_duplicates=False,
        )
        return render_point_sources_from_kernel(kernel_ss, theta_x_list[0], theta_y_list[0], amp_list[0]) + background

    return render_single_star


def weighted_ls_flux(unit_model_stack, data_minus_bkg, err_stack, mask_data, eps=1e-12):
    inv_var = mask_data / (err_stack**2 + eps)
    numer = jnp.sum(unit_model_stack * data_minus_bkg * inv_var, axis=(1, 2))
    denom = jnp.sum((unit_model_stack**2) * inv_var, axis=(1, 2)) + eps
    return jnp.clip(numer / denom, eps, jnp.inf)


def model_psf_svi(
    sci_data,
    err_data,
    mask_data,
    x_loc,
    y_loc,
    flux_loc,
    bkg_loc,
    bkg_scale,
    base_psf_ss,
    corr_init_ss,
    k_values,
    render_single_star,
    ss_factor,
    corr_sigma_low=PSF_CORR_SIGMA_LOW,
    corr_sigma_high=PSF_CORR_SIGMA_HIGH,
    corr_pixel_sigma=PSF_CORR_PIXEL_SIGMA,
    corr_model=PSF_CORR_MODEL_DEFAULT,
    corr_zero_moments=True,
    corr_zero_moments_mode="delta",
    positivity_mode=PSF_POSITIVE_TRANSFORM_DEFAULT,
    softplus_temperature=PSF_SOFTPLUS_TEMPERATURE,
    correction_mode="additive",
    mult_log_clip=PSF_MULT_LOG_CLIP,
    fit_background=False,
    solve_flux=False,
):
    corr_ss_raw = sample_psf_correction(
        corr_model,
        k_values,
        corr_sigma_low,
        corr_sigma_high,
        corr_pixel_sigma,
    )
    numpyro.deterministic("pixels_psf_corr_raw", corr_ss_raw)
    numpyro.deterministic("pixels_psf_corr_init", corr_init_ss)
    if corr_zero_moments and corr_zero_moments_mode == "total":
        corr_ss_unproj = corr_init_ss + corr_ss_raw
        numpyro.deterministic("pixels_psf_corr_total_unproj", corr_ss_unproj)
        corr_ss = project_zero_moments(corr_ss_unproj)
        corr_delta = corr_ss - corr_init_ss
    else:
        corr_delta = project_zero_moments(corr_ss_raw) if corr_zero_moments else corr_ss_raw
        corr_ss = corr_init_ss + corr_delta
    numpyro.deterministic("pixels_psf_corr_delta", corr_delta)
    corr_ss = corr_ss - jnp.mean(corr_ss)
    numpyro.deterministic("pixels_psf_corr_proj", corr_ss)

    if correction_mode == "multiplicative":
        logmult = jnp.clip(corr_ss, -mult_log_clip, mult_log_clip)
        numpyro.deterministic("psf_ss_logmult", logmult)
        psf_ss_raw = base_psf_ss * jnp.exp(logmult)
        numpyro.deterministic("psf_ss_raw", psf_ss_raw)
        psf_ss_model = jnp.maximum(psf_ss_raw, PSF_POS_FLOOR)
        psf_ss_model = psf_ss_model / jnp.sum(psf_ss_model)
    else:
        psf_ss_raw = base_psf_ss + corr_ss
        numpyro.deterministic("psf_ss_raw", psf_ss_raw)
        psf_ss_model = positive_psf_transform(psf_ss_raw, mode=positivity_mode, temperature=softplus_temperature)
    numpyro.deterministic("psf_ss_model", psf_ss_model)

    psf_det_model = downsample_mean(psf_ss_model, factor=ss_factor)
    if correction_mode != "multiplicative" and positivity_mode == "none":
        psf_det_model = normalize_signed_kernel_jax(psf_det_model)
    else:
        psf_det_model = jnp.maximum(psf_det_model, PSF_POS_FLOOR)
        psf_det_model = psf_det_model / jnp.sum(psf_det_model)
    numpyro.deterministic("psf_det_model", psf_det_model)

    base_det = downsample_mean(base_psf_ss, factor=ss_factor)
    base_det = base_det / jnp.sum(base_det)
    numpyro.deterministic("psf_corr_ss_eff", psf_ss_model - base_psf_ss)
    numpyro.deterministic("psf_corr_det_eff", psf_det_model - base_det)

    n_star = sci_data.shape[0]
    with numpyro.plate("stars", n_star):
        x_pos = numpyro.sample(
            "x_pos",
            dist.TruncatedNormal(
                loc=x_loc,
                scale=XPOS_PRIOR_SIGMA,
                low=XPOS_BOUNDS[0],
                high=XPOS_BOUNDS[1],
            ),
        )
        y_pos = numpyro.sample(
            "y_pos",
            dist.TruncatedNormal(
                loc=y_loc,
                scale=YPOS_PRIOR_SIGMA,
                low=YPOS_BOUNDS[0],
                high=YPOS_BOUNDS[1],
            ),
        )
        if fit_background:
            background = numpyro.sample("background", dist.Normal(loc=bkg_loc, scale=bkg_scale))
        else:
            background = numpyro.deterministic("background", jnp.zeros(n_star, dtype=sci_data.dtype))

    unit_model_stack = jax.vmap(render_single_star, in_axes=(None, 0, 0, 0, 0))(
        psf_ss_model,
        x_pos,
        y_pos,
        jnp.ones_like(x_pos),
        jnp.zeros_like(x_pos),
    )
    if solve_flux:
        flux_fixed = weighted_ls_flux(unit_model_stack, sci_data - background[:, None, None], err_data, mask_data)
    else:
        flux_fixed = jnp.clip(flux_loc, 1e-12, jnp.inf)
    numpyro.deterministic("flux_fixed", flux_fixed)
    numpyro.deterministic("flux_opt", flux_fixed)
    numpyro.deterministic("log10_flux_fixed", jnp.log10(flux_fixed))
    numpyro.deterministic("log10_flux_opt", jnp.log10(flux_fixed))

    model_stack = unit_model_stack * flux_fixed[:, None, None] + background[:, None, None]
    logp = dist.Normal(model_stack, err_data).log_prob(sci_data)
    logp = jnp.where(mask_data > 0.5, logp, 0.0)
    numpyro.factor("obs_masked", jnp.sum(logp))


def run_svi_model(config, sci_stack, err_stack, mask_stack, x_loc, y_loc, flux_loc, bkg_loc, bkg_scale, base_psf_ss_np, corr_init_ss_np):
    steps = int(config.get("svi_steps", 1000))
    num_chains = max(1, int(config.get("num_chains", 1)))
    ss_factor = max(1, int(config.get("ss_factor", 2)))
    seed = int(config.get("seed", 42))
    progress_chunks = max(1, int(config.get("progress_chunks", 40)))
    output_dir = config["output_dir"]
    corr_model = str(config.get("psf_corr_model", PSF_CORR_MODEL_DEFAULT))
    corr_sigma_low = float(config.get("psf_corr_sigma_low", PSF_CORR_SIGMA_LOW))
    corr_sigma_high = float(config.get("psf_corr_sigma_high", PSF_CORR_SIGMA_HIGH))
    corr_pixel_sigma = float(config.get("psf_corr_pixel_sigma", PSF_CORR_PIXEL_SIGMA))
    corr_zero_moments = bool(config.get("psf_corr_zero_moments", True))
    corr_zero_moments_mode = str(config.get("psf_corr_zero_moments_mode", "delta"))
    positivity_mode = str(config.get("psf_positive_transform", PSF_POSITIVE_TRANSFORM_DEFAULT))
    softplus_temperature = float(config.get("psf_softplus_temperature", PSF_SOFTPLUS_TEMPERATURE))
    correction_mode = str(config.get("psf_correction_mode", "additive"))
    mult_log_clip = float(config.get("psf_mult_log_clip", PSF_MULT_LOG_CLIP))
    fit_background = bool(config.get("psf_fit_background", False))
    solve_flux = bool(config.get("psf_solve_flux", False))
    progress_label = str(config.get("progress_label", "PSF SVI"))

    pixel_grid = get_pixel_grid(sci_stack.shape[-2:], pix_scale=1.0)
    render_single_star = make_render_functions(pixel_grid, ss_factor=ss_factor)
    if corr_model == "matern":
        k_values_ss = jnp.asarray(PowerSpectrum.K_grid(base_psf_ss_np.shape).k, dtype=jnp.float64)
    else:
        k_values_ss = jnp.zeros(base_psf_ss_np.shape, dtype=jnp.float64)
    if positivity_mode == "none":
        base_psf_ss = normalize_signed_kernel_jax(jnp.asarray(base_psf_ss_np, dtype=jnp.float64))
    else:
        base_psf_ss = normalize_kernel_jax(jnp.asarray(base_psf_ss_np, dtype=jnp.float64))
    corr_init_ss = jnp.asarray(corr_init_ss_np, dtype=jnp.float64)
    corr_init_ss = corr_init_ss - jnp.mean(corr_init_ss)

    sci_data = jnp.asarray(sci_stack, dtype=jnp.float64)
    err_data = jnp.asarray(err_stack, dtype=jnp.float64)
    mask_data = jnp.asarray(mask_stack, dtype=jnp.float64)
    x_loc = jnp.asarray(x_loc, dtype=jnp.float64)
    y_loc = jnp.asarray(y_loc, dtype=jnp.float64)
    flux_loc = jnp.asarray(flux_loc, dtype=jnp.float64)
    bkg_loc = jnp.asarray(bkg_loc, dtype=jnp.float64)
    bkg_scale = jnp.asarray(bkg_scale, dtype=jnp.float64)

    def model(sci, err, mask, x0, y0, flux0, bkg0, bkg_sig, base_ss, corr_init, k_values):
        return model_psf_svi(
            sci,
            err,
            mask,
            x0,
            y0,
            flux0,
            bkg0,
            bkg_sig,
            base_ss,
            corr_init,
            k_values,
            render_single_star,
            ss_factor,
            corr_sigma_low=corr_sigma_low,
            corr_sigma_high=corr_sigma_high,
            corr_pixel_sigma=corr_pixel_sigma,
            corr_model=corr_model,
            corr_zero_moments=corr_zero_moments,
            corr_zero_moments_mode=corr_zero_moments_mode,
            positivity_mode=positivity_mode,
            softplus_temperature=softplus_temperature,
            correction_mode=correction_mode,
            mult_log_clip=mult_log_clip,
            fit_background=fit_background,
            solve_flux=solve_flux,
        )

    init_fun = infer.init_to_median(num_samples=15)
    guide = autoguide.AutoDiagonalNormal(model, init_loc_fn=init_fun, init_scale=0.02)
    scheduler = split_scheduler(steps, init_value=0.01, transition_steps=(200, 20))
    optim = optax.adabelief(learning_rate=scheduler)
    svi = infer.SVI(model, guide, optim, infer.TraceMeanField_ELBO())

    best = None
    losses_all = []
    for chain in range(num_chains):
        update_status(
            output_dir,
            state="running",
            message=f"{progress_label} chain {chain + 1}/{num_chains}: step 0/{steps}.",
            progress={"chain": chain + 1, "chains": num_chains, "step": 0, "steps": steps, "fraction": 0.0},
        )
        rng_key = jax.random.PRNGKey(seed + chain)
        svi_state = svi.init(rng_key, sci_data, err_data, mask_data, x_loc, y_loc, flux_loc, bkg_loc, bkg_scale, base_psf_ss, corr_init_ss, k_values_ss)
        losses_parts = []
        completed = 0
        chunk_size = max(1, int(np.ceil(max(steps, 1) / progress_chunks)))

        def run_chunk(state, n_steps):
            def body_fn(carry, _):
                next_state, loss_value = svi.update(
                    carry,
                    sci_data,
                    err_data,
                    mask_data,
                    x_loc,
                    y_loc,
                    flux_loc,
                    bkg_loc,
                    bkg_scale,
                    base_psf_ss,
                    corr_init_ss,
                    k_values_ss,
                )
                return next_state, loss_value

            return jax.lax.scan(body_fn, state, jnp.arange(n_steps))

        while completed < steps:
            n_step_chunk = min(chunk_size, steps - completed)
            svi_state, losses_chunk = run_chunk(svi_state, n_step_chunk)
            losses_parts.append(np.asarray(jax.device_get(losses_chunk), dtype=float))
            completed += n_step_chunk
            update_status(
                output_dir,
                state="running",
                message=f"{progress_label} chain {chain + 1}/{num_chains}: step {completed}/{steps}.",
                progress={
                    "chain": chain + 1,
                    "chains": num_chains,
                    "step": completed,
                    "steps": steps,
                    "fraction": float(completed / max(steps, 1)),
                    "latest_loss": float(losses_parts[-1][-1]) if losses_parts[-1].size else None,
                },
            )

        losses = np.concatenate(losses_parts) if losses_parts else np.asarray([], dtype=float)
        params = svi.get_params(svi_state)
        losses_all.append(losses)
        final_loss = float(losses[-1]) if losses.size else np.inf
        if best is None or final_loss < best["final_loss"]:
            best = {"params": params, "final_loss": final_loss, "chain": chain}

    params_best = best["params"]
    median_sites = guide.median(params_best)
    posterior_samples = {name: value[None, ...] for name, value in median_sites.items()}
    predictive = infer.Predictive(
        model,
        posterior_samples=posterior_samples,
        return_sites=(
            "psf_det_model",
            "psf_ss_model",
            "psf_ss_raw",
            "pixels_psf_corr_init",
            "pixels_psf_corr_delta",
            "pixels_psf_corr_raw",
            "pixels_psf_corr_proj",
            "psf_corr_det_eff",
            "flux_fixed",
            "flux_opt",
            "background",
            "x_pos",
            "y_pos",
        ),
    )
    posterior = predictive(
        jax.random.PRNGKey(seed + 10000),
        sci_data,
        err_data,
        mask_data,
        x_loc,
        y_loc,
        flux_loc,
        bkg_loc,
        bkg_scale,
        base_psf_ss,
        corr_init_ss,
        k_values_ss,
    )

    det_psf = np.asarray(jax.device_get(posterior["psf_det_model"][0]), dtype=float)
    ss_psf = np.asarray(jax.device_get(posterior["psf_ss_model"][0]), dtype=float)
    corr_proj = np.asarray(jax.device_get(posterior["pixels_psf_corr_proj"][0]), dtype=float)
    raw_psf = np.asarray(jax.device_get(posterior["psf_ss_raw"][0]), dtype=float)
    x_pos = np.asarray(jax.device_get(posterior["x_pos"][0]), dtype=float)
    y_pos = np.asarray(jax.device_get(posterior["y_pos"][0]), dtype=float)
    flux_fit = np.asarray(jax.device_get(posterior["flux_fixed"][0]), dtype=float)
    background_fit = np.asarray(jax.device_get(posterior["background"][0]), dtype=float)
    losses_np = np.stack(losses_all, axis=0)
    summary = {
        "best_chain": int(best["chain"]),
        "final_losses": [float(v[-1]) if len(v) else None for v in losses_all],
        "psf_correction_mode": correction_mode,
        "psf_positive_transform": positivity_mode,
        "psf_softplus_temperature": softplus_temperature,
        "psf_mult_log_clip": mult_log_clip,
        "psf_fit_background": fit_background,
        "psf_solve_flux": solve_flux,
        "psf_corr_model": corr_model,
        "psf_corr_sigma_low": corr_sigma_low,
        "psf_corr_sigma_high": corr_sigma_high,
        "psf_corr_pixel_sigma": corr_pixel_sigma,
        "psf_corr_zero_moments": corr_zero_moments,
        "psf_corr_zero_moments_mode": corr_zero_moments_mode,
        "psf_corr_init_std": float(np.std(np.asarray(jax.device_get(corr_init_ss), dtype=float))),
        "raw_psf_negative_pixels": int(np.count_nonzero(raw_psf <= 0.0)),
    }
    return (
        normalize_for_positivity_mode(det_psf, positivity_mode),
        normalize_for_positivity_mode(ss_psf, positivity_mode),
        corr_proj,
        x_pos,
        y_pos,
        flux_fit,
        background_fit,
        losses_np,
        summary,
    )


def robust_stack_psf(sci_stack, bg_stats=None):
    kernels = []
    for cutout in sci_stack:
        bkg, _, _ = estimate_background_and_error(cutout, bg_stats=bg_stats)
        kernel = np.clip(cutout - bkg, 0.0, np.inf)
        kernels.append(normalize_kernel(kernel))
    return normalize_kernel(np.median(np.stack(kernels), axis=0))


def render_psf_stack(psf_kernel, sci_stack, err_stack, mask_stack, x_pos, y_pos, flux_fixed, ss_factor=1):
    x_pos = np.asarray(x_pos, dtype=float)
    y_pos = np.asarray(y_pos, dtype=float)
    flux_fixed = np.asarray(flux_fixed, dtype=float)
    render_single_star = make_render_functions(get_pixel_grid(sci_stack.shape[-2:], pix_scale=1.0), ss_factor=ss_factor)
    render_kernel_jax = jnp.asarray(psf_kernel, dtype=jnp.float64)

    models = []
    residuals = []
    metrics = []
    for i, cutout in enumerate(sci_stack):
        unit_model = np.asarray(
            jax.device_get(render_single_star(render_kernel_jax, x_pos[i], y_pos[i], 1.0, 0.0)),
            dtype=float,
        )
        flux = max(float(flux_fixed[i]), 0.0)
        model = unit_model * flux
        residual = np.asarray(cutout, dtype=float) - model
        valid = np.asarray(mask_stack[i], dtype=float) > 0.5
        resid_valid = residual[valid]
        data_valid = np.asarray(cutout, dtype=float)[valid]
        err_valid = np.asarray(err_stack[i], dtype=float)[valid]
        denom = max(float(np.sum(np.abs(data_valid))), 1e-20)
        chi2 = float(np.sum((resid_valid / np.maximum(err_valid, 1e-12)) ** 2))
        dof = max(int(resid_valid.size), 1)
        metrics.append(
            {
                "index": int(i),
                "x_pos": float(x_pos[i]),
                "y_pos": float(y_pos[i]),
                "flux": flux,
                "data_sum": float(np.sum(data_valid)),
                "model_sum": float(np.sum(model[valid])),
                "residual_sum": float(np.sum(resid_valid)),
                "l1_fraction": float(np.sum(np.abs(resid_valid)) / denom),
                "rmse": float(np.sqrt(np.mean(resid_valid**2))),
                "max_abs_residual": float(np.max(np.abs(resid_valid))),
                "chi2_per_pixel": chi2 / dof,
            }
        )
        models.append(model)
        residuals.append(residual)
    return np.asarray(models), np.asarray(residuals), metrics


def save_outputs(
    config,
    sci_stack,
    err_stack,
    mask_stack,
    used_ids,
    det_psf,
    base_psf,
    ss_psf=None,
    corr_proj=None,
    first_model_init=None,
    x_pos=None,
    y_pos=None,
    flux_fixed=None,
    background_fixed=None,
    losses=None,
    summary=None,
    bg_stats=None,
    stage1_det_psf=None,
    stage1_ss_psf=None,
    stage1_models=None,
    stage1_residuals=None,
    stage1_metrics=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = output_dir / "PSF_model_gui_fit.fits"
    npz_path = output_dir / "PSF_model_gui_fit_info.npz"
    png_path = output_dir / "PSF_model_gui_fit_preview.png"
    stage1_preview_path = output_dir / "stage1_initial_psf_all_stars.png"
    stage1_metrics_path = output_dir / "stage1_initial_psf_all_stars_metrics.json"

    hdr = fits.Header()
    hdr["NSTAR"] = int(len(used_ids))
    hdr["KERNEL"] = int(config["kernel_size"])
    hdr["NITER"] = int(config.get("svi_steps", 0))
    hdr["NCHAIN"] = int(config.get("num_chains", 1))
    hdr["SSFACT"] = int(config.get("ss_factor", 2))
    hdr["METHOD"] = "GUI_PSF_FIT"
    hdr["RENDER"] = "SS_BLOCKSUM" if int(config.get("ss_factor", 2)) > 1 else "DET"
    hdr["BGSUB"] = (bool(config.get("background_subtracted", True)), "Whether SCI_STACK is background-subtracted")
    hdr["CENTERED"] = (bool(config.get("psf_model_centered", True)), "DET_PSF_MODEL is recentered for export")
    if config.get("psf_init_strategy_used"):
        hdr["INITSTR"] = str(config["psf_init_strategy_used"])
    if config.get("psf_init_reference_star_id") is not None:
        hdr["INITID"] = (int(config["psf_init_reference_star_id"]), "Reference star for PSF initialization")
    if config.get("psf_init_reference_x") is not None:
        hdr["INITX"] = (float(config["psf_init_reference_x"]), "Reference star x offset in detector pixels")
    if config.get("psf_init_reference_y") is not None:
        hdr["INITY"] = (float(config["psf_init_reference_y"]), "Reference star y offset in detector pixels")
    if config.get("base_psf_mode"):
        hdr["BASEMODE"] = str(config["base_psf_mode"])
    if config.get("base_psf_fwhm") is not None:
        hdr["BASEFWHM"] = (float(config["base_psf_fwhm"]), "Analytic base FWHM in detector pixels")
    if config.get("base_psf_beta") is not None:
        hdr["BASEBETA"] = (float(config["base_psf_beta"]), "Analytic Moffat beta")
    if config.get("base_psf_x0") is not None:
        hdr["BASEX0"] = (float(config["base_psf_x0"]), "Analytic base x0 in detector pixels")
    if config.get("base_psf_y0") is not None:
        hdr["BASEY0"] = (float(config["base_psf_y0"]), "Analytic base y0 in detector pixels")
    if summary:
        hdr["BESTCHN"] = int(summary.get("best_chain", 0))
    if bg_stats:
        hdr["BGSIGMA"] = float(bg_stats["sigma"])
        hdr["BGMED"] = float(bg_stats["background"])
        hdr["BGBOX"] = (int(bg_stats["size"]), "Background box side length")

    x_pos = np.zeros(len(sci_stack), dtype=float) if x_pos is None else np.asarray(x_pos, dtype=float)
    y_pos = np.zeros(len(sci_stack), dtype=float) if y_pos is None else np.asarray(y_pos, dtype=float)
    flux_fixed = None if flux_fixed is None else np.asarray(flux_fixed, dtype=float)
    background_fixed = np.zeros(len(sci_stack), dtype=float) if background_fixed is None else np.asarray(background_fixed, dtype=float)
    ss_factor = max(1, int(config.get("ss_factor", 2)))
    det_psf_fit = np.asarray(det_psf, dtype=float)
    ss_psf_fit = None if ss_psf is None else np.asarray(ss_psf, dtype=float)
    det_psf_output = det_psf_fit
    ss_psf_output = ss_psf_fit
    if bool(config.get("psf_export_recenter", False)) and config.get("base_psf_x0") is not None and config.get("base_psf_y0") is not None:
        det_psf_output = recenter_kernel_to_frame(det_psf_fit, config["base_psf_x0"], config["base_psf_y0"], factor=1)
        if ss_psf_fit is not None:
            ss_psf_output = recenter_kernel_to_frame(ss_psf_fit, config["base_psf_x0"], config["base_psf_y0"], factor=ss_factor)

    render_kernel = ss_psf_fit if ss_psf_fit is not None else det_psf_fit
    render_factor = ss_factor if ss_psf_fit is not None else 1
    render_single_star = make_render_functions(get_pixel_grid(det_psf.shape, pix_scale=1.0), ss_factor=render_factor)
    render_kernel_jax = jnp.asarray(render_kernel, dtype=jnp.float64)

    residuals = []
    models = []
    for i, cutout in enumerate(sci_stack):
        unit_model = np.asarray(jax.device_get(render_single_star(render_kernel_jax, x_pos[i], y_pos[i], 1.0, 0.0)), dtype=float)
        if flux_fixed is None:
            inv_var = mask_stack[i] / (err_stack[i] ** 2 + 1e-12)
            flux = np.sum(unit_model * cutout * inv_var) / max(np.sum(unit_model**2 * inv_var), 1e-20)
        else:
            flux = flux_fixed[i]
        model = unit_model * max(float(flux), 0.0) + float(background_fixed[i])
        models.append(model)
        residuals.append(cutout - model)

    hdus = [
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(data=np.asarray(det_psf_output, dtype=np.float32), name="DET_PSF_MODEL"),
        fits.ImageHDU(data=np.asarray(base_psf, dtype=np.float32), name="BASE_PSF_DET"),
        fits.ImageHDU(data=np.asarray(sci_stack, dtype=np.float32), name="SCI_STACK"),
        fits.ImageHDU(data=np.asarray(err_stack, dtype=np.float32), name="ERR_STACK"),
        fits.ImageHDU(data=np.asarray(mask_stack, dtype=np.float32), name="MASK_STACK"),
        fits.ImageHDU(data=np.asarray(models, dtype=np.float32), name="MODEL_STACK"),
        fits.ImageHDU(data=np.asarray(residuals, dtype=np.float32), name="RESIDUAL_STACK"),
    ]
    if ss_psf_output is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(ss_psf_output, dtype=np.float32), name="SS_PSF_MODEL"))
    if corr_proj is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(corr_proj, dtype=np.float32), name="CORR_PROJ_SS"))
    if first_model_init is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(first_model_init, dtype=np.float32), name="FIRST_MODEL_INIT"))
    hdus.append(fits.ImageHDU(data=np.asarray(background_fixed, dtype=np.float32), name="STAR_BACKGROUNDS"))
    if stage1_det_psf is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(stage1_det_psf, dtype=np.float32), name="STAGE1_DET_PSF"))
    if stage1_ss_psf is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(stage1_ss_psf, dtype=np.float32), name="STAGE1_SS_PSF"))
    if stage1_models is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(stage1_models, dtype=np.float32), name="STAGE1_MODEL_STACK"))
    if stage1_residuals is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(stage1_residuals, dtype=np.float32), name="STAGE1_RESIDUAL"))
    hdus.append(fits.ImageHDU(data=np.asarray([x_pos, y_pos], dtype=np.float32), name="STAR_OFFSETS"))
    if losses is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(losses, dtype=np.float32), name="FIT_LOSSES"))
    fits.HDUList(hdus).writeto(fits_path, overwrite=True)

    if stage1_metrics is not None:
        metrics_payload = {
            "selected_ids": [int(v) for v in used_ids],
            "metrics": stage1_metrics,
        }
        write_json(stage1_metrics_path, metrics_payload)
        if summary is not None:
            summary["stage1_all_star_metrics"] = stage1_metrics
            summary["stage1_all_star_metrics_path"] = str(stage1_metrics_path)

    np.savez(
        npz_path,
        selected_ids=np.asarray(used_ids),
        psf_det_median=det_psf_output,
        psf_ss_median=ss_psf_output,
        corr_proj=corr_proj,
        stage1_det_psf=stage1_det_psf,
        stage1_ss_psf=stage1_ss_psf,
        stage1_model_stack=stage1_models,
        stage1_residual_stack=stage1_residuals,
        x_pos=x_pos,
        y_pos=y_pos,
        flux_fixed=flux_fixed,
        background_fixed=background_fixed,
        losses=losses,
        summary=json.dumps(summary or {}),
        base_fwhm=float(config["base_psf_fwhm"]) if config.get("base_psf_fwhm") is not None else np.nan,
        base_beta=float(config["base_psf_beta"]) if config.get("base_psf_beta") is not None else np.nan,
        base_x0=float(config["base_psf_x0"]) if config.get("base_psf_x0") is not None else np.nan,
        base_y0=float(config["base_psf_y0"]) if config.get("base_psf_y0") is not None else np.nan,
        base_mode=str(config.get("base_psf_mode", "")),
        psf_init_strategy=str(config.get("psf_init_strategy_used", "")),
        psf_init_reference_star_id=int(config["psf_init_reference_star_id"]) if config.get("psf_init_reference_star_id") is not None else -1,
        psf_init_reference_x=float(config["psf_init_reference_x"]) if config.get("psf_init_reference_x") is not None else np.nan,
        psf_init_reference_y=float(config["psf_init_reference_y"]) if config.get("psf_init_reference_y") is not None else np.nan,
    )

    first_psf_image = np.clip(np.nan_to_num(np.asarray(sci_stack[0], dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, np.inf)
    if float(np.sum(first_psf_image)) > 0.0:
        first_psf_image = first_psf_image / np.sum(first_psf_image)
    if first_model_init is None:
        first_model_init = models[0]
    first_model_image = np.clip(np.nan_to_num(np.asarray(first_model_init, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, np.inf)
    if float(np.sum(first_model_image)) > 0.0:
        first_model_image = first_model_image / np.sum(first_model_image)

    positive_psf = np.concatenate([
        first_psf_image.ravel(),
        np.asarray(det_psf_output, dtype=float).ravel(),
    ])
    positive_psf = positive_psf[np.isfinite(positive_psf) & (positive_psf > 0)]
    if positive_psf.size:
        psf_vmin = max(float(np.nanpercentile(positive_psf, 0.5)), 1e-20)
        psf_vmax = max(float(np.nanpercentile(positive_psf, 99.8)), psf_vmin * 10)
    else:
        psf_vmin, psf_vmax = 1e-20, 1.0
    psf_norm = LogNorm(vmin=psf_vmin, vmax=psf_vmax)

    median_residual = np.median(np.asarray(residuals), axis=0)
    finite_residual = median_residual[np.isfinite(median_residual)]
    if finite_residual.size:
        residual_sigma = float(np.nanstd(finite_residual))
        if residual_sigma <= 0:
            mad = float(np.nanmedian(np.abs(finite_residual - np.nanmedian(finite_residual))))
            residual_sigma = 1.4826 * mad
        residual_limit = max(3.0 * residual_sigma, 1e-12)
    else:
        residual_limit = 1.0
    residual_norm = Normalize(vmin=-residual_limit, vmax=residual_limit)

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), constrained_layout=True)
    axes[0].imshow(first_psf_image, origin="lower", norm=psf_norm, cmap="twilight")
    axes[0].set_title(f"First PSF image #{used_ids[0]}")
    axes[1].imshow(det_psf_output, origin="lower", norm=psf_norm, cmap="twilight")
    axes[1].set_title("Model PSF")
    axes[2].imshow(median_residual, origin="lower", norm=residual_norm, cmap="bwr")
    axes[2].set_title("Median residual")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    if stage1_models is not None and stage1_residuals is not None:
        stage1_models = np.asarray(stage1_models, dtype=float)
        stage1_residuals = np.asarray(stage1_residuals, dtype=float)
        n_star = len(used_ids)
        stage1_positive = np.concatenate([
            np.clip(np.asarray(sci_stack, dtype=float), 0.0, np.inf).ravel(),
            np.clip(stage1_models, 0.0, np.inf).ravel(),
        ])
        stage1_positive = stage1_positive[np.isfinite(stage1_positive) & (stage1_positive > 0)]
        if stage1_positive.size:
            stage1_vmin = max(float(np.nanpercentile(stage1_positive, 0.5)), 1e-20)
            stage1_vmax = max(float(np.nanpercentile(stage1_positive, 99.8)), stage1_vmin * 10)
        else:
            stage1_vmin, stage1_vmax = 1e-20, 1.0
        stage1_data_norm = LogNorm(vmin=stage1_vmin, vmax=stage1_vmax)
        stage1_resid_values = stage1_residuals[np.isfinite(stage1_residuals)]
        stage1_resid_limit = max(3.0 * float(np.nanstd(stage1_resid_values)), 1e-12) if stage1_resid_values.size else 1.0
        stage1_resid_norm = Normalize(vmin=-stage1_resid_limit, vmax=stage1_resid_limit)
        fig_height = max(2.4, 2.1 * n_star)
        fig, axes = plt.subplots(n_star, 3, figsize=(9.6, fig_height), squeeze=False, constrained_layout=True)
        for i, star_id in enumerate(used_ids):
            axes[i, 0].imshow(np.clip(sci_stack[i], 0.0, np.inf), origin="lower", norm=stage1_data_norm, cmap="twilight")
            axes[i, 0].set_title(f"Star #{star_id} data")
            axes[i, 1].imshow(np.clip(stage1_models[i], 0.0, np.inf), origin="lower", norm=stage1_data_norm, cmap="twilight")
            axes[i, 1].set_title("Stage1 model")
            axes[i, 2].imshow(stage1_residuals[i], origin="lower", norm=stage1_resid_norm, cmap="bwr")
            metric = stage1_metrics[i] if stage1_metrics and i < len(stage1_metrics) else {}
            l1_value = metric.get("l1_fraction")
            title = "Residual" if l1_value is None else f"Residual L1={l1_value:.3g}"
            axes[i, 2].set_title(title)
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.savefig(stage1_preview_path, dpi=160)
        plt.close(fig)
        if summary is not None:
            summary["stage1_all_star_preview_path"] = str(stage1_preview_path)

    return fits_path, npz_path, png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = config["output_dir"]
    config.setdefault("background_subtracted", False)
    update_status(output_dir, state="running", message="Preparing selected PSF cutouts.")
    try:
        sci_stack, err_stack, mask_stack, used_ids, x_loc, y_loc, flux_loc, bkg_loc, bkg_scale, bg_stats = selected_cutouts(config)
        background_subtracted = bool(config.get("background_subtracted", False))
        if background_subtracted:
            sci_stack_init = sci_stack
        else:
            sci_stack_init = sci_stack - bkg_loc[:, None, None]
        init_stack, init_reference_index, init_reference_star_id, init_strategy_used = psf_initialization_stack(config, sci_stack_init, used_ids)
        config["psf_init_strategy_used"] = init_strategy_used
        if init_reference_star_id is not None:
            config["psf_init_reference_star_id"] = int(init_reference_star_id)
            config["psf_init_reference_x"] = float(x_loc[init_reference_index])
            config["psf_init_reference_y"] = float(y_loc[init_reference_index])
        if init_reference_star_id is None:
            update_status(output_dir, state="running", message="Building stack PSF initialization.")
        else:
            update_status(
                output_dir,
                state="running",
                message=f"Building Moffat+correction initialization from the first selected PSF, ID #{init_reference_star_id}.",
            )
        ss_factor = max(1, int(config.get("ss_factor", 2)))
        x_model_loc, y_model_loc = model_position_priors(config, x_loc, y_loc, reference_index=init_reference_index)
        x_model_loc = np.asarray(x_model_loc, dtype=float).copy()
        y_model_loc = np.asarray(y_model_loc, dtype=float).copy()
        flux_loc = np.asarray(flux_loc, dtype=float).copy()
        base_psf_ss, base_psf_det, base_source, base_mode, base_fwhm, base_beta, base_x0, base_y0 = build_base_psf(
            config,
            sci_stack.shape[-2:],
            init_stack,
            bg_stats=bg_stats,
            factor=ss_factor,
        )
        config["base_psf_mode"] = base_mode
        if base_fwhm is not None:
            config["base_psf_fwhm"] = float(base_fwhm)
        if base_beta is not None:
            config["base_psf_beta"] = float(base_beta)
        if base_x0 is not None:
            config["base_psf_x0"] = float(base_x0)
        if base_y0 is not None:
            config["base_psf_y0"] = float(base_y0)
        stage1_moffat_fit = None
        stage1_moffat_flux = None
        if init_reference_index is not None and not str(config.get("base_psf_path") or "").strip():
            stage1_moffat_fit = fit_cutout_moffat_params(
                sci_stack_init[init_reference_index],
                bg_stats=bg_stats,
                background_subtracted=True,
            )
            center_x = (sci_stack.shape[-1] - 1.0) / 2.0
            center_y = (sci_stack.shape[-2] - 1.0) / 2.0
            x_model_loc[init_reference_index] = float(stage1_moffat_fit["x0"]) - center_x
            y_model_loc[init_reference_index] = float(stage1_moffat_fit["y0"]) - center_y
            stage1_moffat_flux = float(flux_loc[init_reference_index])
            flux_loc[init_reference_index] = stage1_moffat_flux
        corr_proj = None
        first_model_init = None
        stage1_det_output = None
        stage1_ss_output = None
        stage1_models = None
        stage1_residuals = None
        stage1_metrics = None
        flux_fit = None
        background_fit = None
        psf_corr_model = str(config.get("psf_corr_model", PSF_CORR_MODEL_DEFAULT))
        positivity_mode = str(config.get("psf_positive_transform", PSF_POSITIVE_TRANSFORM_DEFAULT))
        stage1_zero_moments = bool(config.get("psf_stage1_zero_moments", False))
        stage1_recenter_weighted = bool(config.get("psf_stage1_recenter_weighted", True))
        two_stage = bool(config.get("psf_two_stage", True)) and psf_corr_model != "fixed_init"
        if len(used_ids) < 2 and two_stage:
            psf_corr_model = "fixed_init"
            config["psf_corr_model"] = psf_corr_model
            config["psf_single_star_fixed_init"] = True
            two_stage = False

        if two_stage:
            update_status(
                output_dir,
                state="running",
                message=f"Stage 1: building centered Moffat+residual PSF from first selected star, ID #{used_ids[0]}.",
            )
            stage1_count = len(init_stack)
            stage1_corr_proj = build_rendered_correction_init(
                init_stack,
                base_psf_ss,
                x_model_loc[:stage1_count],
                y_model_loc[:stage1_count],
                flux_loc[:stage1_count],
                factor=ss_factor,
                remove_moments=stage1_zero_moments,
            )
            stage1_ss_psf = positive_psf_transform_np(base_psf_ss + stage1_corr_proj, mode="hard_clip")
            stage1_recenter_info = None
            if stage1_recenter_weighted:
                stage1_ss_psf, stage1_recenter_info = recenter_kernel_weighted_centroid(stage1_ss_psf, factor=ss_factor)
                x_model_loc = x_model_loc + stage1_recenter_info["before"]["dx_det"]
                y_model_loc = y_model_loc + stage1_recenter_info["before"]["dy_det"]
            stage1_det_psf = normalize_kernel(downsample_mean_np(stage1_ss_psf, ss_factor))
            stage1_x_pos = np.asarray(x_model_loc[:1], dtype=float)
            stage1_y_pos = np.asarray(y_model_loc[:1], dtype=float)
            stage1_summary = {
                "method": "rendered_centered_moffat_residual",
                "psf_positive_transform": "hard_clip",
                "moffat_fit": stage1_moffat_fit,
                "moffat_flux": stage1_moffat_flux,
                "zero_moments": stage1_zero_moments,
                "weighted_recenter": stage1_recenter_weighted,
                "weighted_recenter_info": stage1_recenter_info,
                "psf_corr_init_std": float(np.std(stage1_corr_proj)),
                "raw_psf_negative_pixels": int(np.count_nonzero(base_psf_ss + stage1_corr_proj < 0.0)),
            }
            stage1_models, stage1_residuals, stage1_metrics = render_psf_stack(
                stage1_ss_psf,
                sci_stack_init,
                err_stack,
                mask_stack,
                x_model_loc,
                y_model_loc,
                flux_loc,
                ss_factor=ss_factor,
            )
            for metric, star_id in zip(stage1_metrics, used_ids):
                metric["star_id"] = int(star_id)
            stage1_det_output = stage1_det_psf
            stage1_ss_output = stage1_ss_psf
            first_init_render = make_render_functions(get_pixel_grid(sci_stack.shape[-2:], pix_scale=1.0), ss_factor=ss_factor)
            first_model_init = np.asarray(
                jax.device_get(
                    first_init_render(
                        jnp.asarray(stage1_ss_psf, dtype=jnp.float64),
                        float(stage1_x_pos[0]),
                        float(stage1_y_pos[0]),
                        1.0,
                        0.0,
                    )
                ),
                dtype=float,
            ) * float(flux_loc[0])

            update_status(output_dir, state="running", message="Stage 2: jointly refining PSF on all selected stars.")
            stage2_config = {
                **config,
                "progress_label": "Step 3 PSF SVI",
                "psf_correction_mode": str(config.get("psf_correction_mode", "multiplicative")),
                "psf_corr_model": str(config.get("psf_corr_model", "matern")),
                "psf_corr_zero_moments": bool(config.get("psf_corr_zero_moments", True)),
                "psf_corr_zero_moments_mode": str(config.get("psf_corr_zero_moments_mode", "delta")),
                "psf_fit_background": bool(config.get("psf_fit_background", not background_subtracted)),
                "psf_solve_flux": bool(config.get("psf_solve_flux", True)),
            }
            stage2_base_ss = normalize_kernel(stage1_ss_psf)
            stage2_base_det = normalize_kernel(downsample_mean_np(stage2_base_ss, ss_factor))
            stage2_corr_init_ss = np.zeros_like(stage2_base_ss)
            det_psf, ss_psf, corr_proj, x_pos, y_pos, flux_fit, background_fit, losses, summary = run_svi_model(
                stage2_config,
                sci_stack,
                err_stack,
                mask_stack,
                x_model_loc,
                y_model_loc,
                flux_loc,
                bkg_loc,
                bkg_scale,
                stage2_base_ss,
                stage2_corr_init_ss,
            )
            base_psf_det = stage2_base_det
            base_source = f"stage1_centered_moffat_plus_correction_star_{used_ids[0]}"
            base_mode = "stage1_centered_psf"
            base_x0 = (sci_stack.shape[-1] - 1.0) / 2.0
            base_y0 = (sci_stack.shape[-2] - 1.0) / 2.0
            config["base_psf_mode"] = base_mode
            config["base_psf_x0"] = float(base_x0)
            config["base_psf_y0"] = float(base_y0)
            summary["psf_workflow"] = "step3_svi_stage1_base_multiplicative_correction"
            summary["psf_init_strategy"] = init_strategy_used
            summary["psf_init_reference_star_id"] = init_reference_star_id
            summary["stage1"] = {
                "steps": 0,
                "x_pos": stage1_x_pos.tolist(),
                "y_pos": stage1_y_pos.tolist(),
                "summary": stage1_summary,
            }
        elif psf_corr_model == "fixed_init":
            stage1_count = len(init_stack)
            corr_init_ss = build_rendered_correction_init(
                init_stack,
                base_psf_ss,
                x_model_loc[:stage1_count],
                y_model_loc[:stage1_count],
                flux_loc[:stage1_count],
                factor=ss_factor,
                remove_moments=stage1_zero_moments,
            )
            ss_psf = positive_psf_transform_np(
                base_psf_ss + corr_init_ss,
                mode=positivity_mode,
                temperature=float(config.get("psf_softplus_temperature", PSF_SOFTPLUS_TEMPERATURE)),
            )
            det_psf = normalize_for_positivity_mode(downsample_mean_np(ss_psf, ss_factor), positivity_mode)
            x_pos = np.asarray(x_model_loc, dtype=float)
            y_pos = np.asarray(y_model_loc, dtype=float)
            losses = np.asarray([[0.0]], dtype=float)
            stage1_models, stage1_residuals, stage1_metrics = render_psf_stack(
                ss_psf,
                sci_stack,
                err_stack,
                mask_stack,
                x_pos,
                y_pos,
                flux_loc,
                ss_factor=ss_factor,
            )
            for metric, star_id in zip(stage1_metrics, used_ids):
                metric["star_id"] = int(star_id)
            stage1_det_output = det_psf
            stage1_ss_output = ss_psf
            first_init_render = make_render_functions(get_pixel_grid(sci_stack.shape[-2:], pix_scale=1.0), ss_factor=ss_factor)
            first_model_init = np.asarray(
                jax.device_get(
                    first_init_render(
                        jnp.asarray(ss_psf, dtype=jnp.float64),
                        float(x_model_loc[0]),
                        float(y_model_loc[0]),
                        1.0,
                        0.0,
                    )
                ),
                dtype=float,
            ) * float(flux_loc[0])
            corr_proj = corr_init_ss
            flux_fit = np.asarray(flux_loc, dtype=float)
            background_fit = np.zeros(len(used_ids), dtype=float)
            summary = {
                "best_chain": 0,
                "final_losses": [0.0],
                "psf_corr_model": "fixed_init",
                "psf_init_strategy": init_strategy_used,
                "psf_init_reference_star_id": init_reference_star_id,
                "method": "rendered_centered_moffat_residual",
                "moffat_fit": stage1_moffat_fit,
                "moffat_flux": stage1_moffat_flux,
                "psf_corr_init_std": float(np.std(corr_init_ss)),
                "raw_psf_negative_pixels": int(np.count_nonzero(base_psf_ss + corr_init_ss <= 0.0)),
            }
        else:
            stage1_count = len(init_stack)
            corr_init_ss = build_rendered_correction_init(
                init_stack,
                base_psf_ss,
                x_model_loc[:stage1_count],
                y_model_loc[:stage1_count],
                flux_loc[:stage1_count],
                factor=ss_factor,
                remove_moments=stage1_zero_moments,
            )
            first_init_ss = positive_psf_transform_np(
                base_psf_ss + corr_init_ss,
                mode=positivity_mode,
                temperature=float(config.get("psf_softplus_temperature", PSF_SOFTPLUS_TEMPERATURE)),
            )
            first_init_det = normalize_for_positivity_mode(downsample_mean_np(first_init_ss, ss_factor), positivity_mode)
            stage1_models, stage1_residuals, stage1_metrics = render_psf_stack(
                first_init_ss,
                sci_stack,
                err_stack,
                mask_stack,
                x_model_loc,
                y_model_loc,
                flux_loc,
                ss_factor=ss_factor,
            )
            for metric, star_id in zip(stage1_metrics, used_ids):
                metric["star_id"] = int(star_id)
            stage1_det_output = first_init_det
            stage1_ss_output = first_init_ss
            first_init_render = make_render_functions(get_pixel_grid(sci_stack.shape[-2:], pix_scale=1.0), ss_factor=ss_factor)
            first_model_init = np.asarray(
                jax.device_get(
                    first_init_render(
                        jnp.asarray(first_init_ss, dtype=jnp.float64),
                        float(x_model_loc[0]),
                        float(y_model_loc[0]),
                        1.0,
                        0.0,
                    )
                ),
                dtype=float,
            ) * float(flux_loc[0])
            det_psf, ss_psf, corr_proj, x_pos, y_pos, flux_fit, background_fit, losses, summary = run_svi_model(
                config,
                sci_stack,
                err_stack,
                mask_stack,
                x_model_loc,
                y_model_loc,
                flux_loc,
                bkg_loc,
                bkg_scale,
                base_psf_ss,
                corr_init_ss,
            )
            summary["psf_init_strategy"] = init_strategy_used
            summary["psf_init_reference_star_id"] = init_reference_star_id
        fits_path, npz_path, png_path = save_outputs(
            config,
            sci_stack,
            err_stack,
            mask_stack,
            used_ids,
            det_psf,
            base_psf_det,
            ss_psf=ss_psf,
            corr_proj=corr_proj,
            first_model_init=first_model_init,
            x_pos=x_pos,
            y_pos=y_pos,
            flux_fixed=flux_fit if flux_fit is not None else flux_loc,
            background_fixed=background_fit if background_fit is not None else np.zeros(len(used_ids), dtype=float),
            losses=losses,
            summary=summary,
            bg_stats=bg_stats,
            stage1_det_psf=stage1_det_output,
            stage1_ss_psf=stage1_ss_output,
            stage1_models=stage1_models,
            stage1_residuals=stage1_residuals,
            stage1_metrics=stage1_metrics,
        )
        stage1_preview_path = summary.get("stage1_all_star_preview_path") if summary else None
        stage1_preview_url = (
            "/" + Path(stage1_preview_path).resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
            if stage1_preview_path
            else None
        )
        update_status(
            output_dir,
            state="completed",
            message=f"PSF SVI completed from {len(used_ids)} selected stars.",
            selected_ids=used_ids,
            psf_init_strategy=init_strategy_used,
            psf_init_reference_star_id=init_reference_star_id,
            psf_init_reference_position=(
                None
                if init_reference_index is None
                else {"x": float(x_loc[init_reference_index]), "y": float(y_loc[init_reference_index])}
            ),
            base_psf_source=base_source,
            base_psf_mode=base_mode,
            base_psf_fwhm=base_fwhm,
            base_psf_beta=base_beta,
            base_psf_x0=base_x0,
            base_psf_y0=base_y0,
            psf_model_centered=bool(config.get("psf_model_centered", True)),
            psf_position_reference=str(config.get("psf_position_reference", "first_star")),
            psf_center_fit=(
                None
                if base_x0 is None or base_y0 is None
                else {"x0": float(base_x0), "y0": float(base_y0)}
            ),
            position_prior_center={"x": x_loc.tolist(), "y": y_loc.tolist()},
            model_position_prior_center={"x": x_model_loc.tolist(), "y": y_model_loc.tolist()},
            fitted_positions={"x": x_pos.tolist(), "y": y_pos.tolist()},
            background_stats=bg_stats,
            svi_summary=summary,
            psf_positive_transform=str(config.get("psf_positive_transform", PSF_POSITIVE_TRANSFORM_DEFAULT)),
            psf_softplus_temperature=float(config.get("psf_softplus_temperature", PSF_SOFTPLUS_TEMPERATURE)),
            psf_correction_mode=str(config.get("psf_correction_mode", "multiplicative")),
            psf_fit_background=bool(config.get("psf_fit_background", not bool(config.get("background_subtracted", False)))),
            psf_solve_flux=bool(config.get("psf_solve_flux", True)),
            psf_corr_model=str(config.get("psf_corr_model", PSF_CORR_MODEL_DEFAULT)),
            psf_corr_sigma_low=float(config.get("psf_corr_sigma_low", PSF_CORR_SIGMA_LOW)),
            psf_corr_sigma_high=float(config.get("psf_corr_sigma_high", PSF_CORR_SIGMA_HIGH)),
            psf_corr_pixel_sigma=float(config.get("psf_corr_pixel_sigma", PSF_CORR_PIXEL_SIGMA)),
            psf_corr_zero_moments=bool(config.get("psf_corr_zero_moments", True)),
            fits_path=str(fits_path),
            npz_path=str(npz_path),
            stage1_all_star_preview_url=stage1_preview_url,
            stage1_all_star_metrics_path=(summary or {}).get("stage1_all_star_metrics_path"),
            preview_url="/" + fits_path.with_name(png_path.name).resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix(),
        )
    except Exception as exc:
        update_status(output_dir, state="failed", message=str(exc))
        raise


if __name__ == "__main__":
    main()
