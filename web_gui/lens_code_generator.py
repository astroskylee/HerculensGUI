from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path


AUTO_SIGMA_VALUES = {"", "auto", "half", "half_image", "half image"}
DEFAULT_EXPOSURE_TIME = 565.0 * 4.0


def _dict_literal(mapping, value_indent=12):
    inner = "\n".join(f"{' ' * value_indent}{key!r}: {value!r}," for key, value in mapping.items())
    return "{\n" + inner + "\n" + " " * (value_indent - 4) + "}"


def _compact_literal(value):
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key!r}: {_compact_literal(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_compact_literal(item) for item in value) + "]"
    return repr(value)


def _model_config_assignment(mapping):
    lines = ["MODEL_CONFIG = {"]
    lines.extend(f"    {key!r}: {_compact_literal(value)}," for key, value in mapping.items())
    lines.append("}")
    return "\n".join(lines)


def _dspl_model_config(config):
    return {
        "mode": "double_source_plane",
        "svi": {
            "seed": config["seed"],
            "num_chains": config["num_chains"],
            "max_iter_parametric": config["max_iter_parametric"],
            "max_iter_pixelated": config["max_iter_pixelated"],
            "max_iter_solver": config["max_iter_solver"],
        },
        "data": {
            "corner_pixel": config["background_subtract_corner"],
            "use_existing_rms_map": config["use_existing_rms_map"],
            "use_scalar_rms_loguniform": config["use_scalar_rms_loguniform"],
            "automatic_background_rms": config["automatic_background_rms"],
            "background_rms": config["background_rms"],
            "background_subtract_enabled": config["background_subtract_enabled"],
            "background_subtract_corner": config["background_subtract_corner"],
            "exposure_time_if_missing": config["exposure_time"],
        },
        "light": {
            "lens": {
                "profile": config["light_profile"],
                "n_gauss": config["n_gauss_lens"],
                "sigma_lims": config["lens_sigma_lims"],
                "center_max_offset": config["lens_light_center_max_offset"],
                "center_low": -config["lens_light_center_max_offset"],
                "center_high": config["lens_light_center_max_offset"],
                "external_mode": config["lens_light_external_mode"],
                "parametric_mode": config["lens_light_parametric_mode"],
                "pixelated_mode": config["lens_light_pixelated_mode"],
                "external_kwargs_path": config["lens_light_external_path"],
            },
            "source1": {
                "profile": config["light_profile"],
                "n_gauss": config["n_gauss_source"],
                "sigma_lims": config["source_sigma_lims"],
            },
            "source2": {
                "profile": config["light_profile"],
                "n_gauss": config["n_gauss_source2"],
                "sigma_lims": config["source2_sigma_lims"],
            },
        },
        "source_grid": {
            "scale1": config["source_grid_scale"],
            "scale2": config["source2_grid_scale"],
            "pixel_grid_shape1": config["source_pixel_grid_shape"],
            "pixel_grid_shape2": config["source2_pixel_grid_shape"],
            "manual1": config["source_pixel_grid_manual"],
            "manual2": config["source2_pixel_grid_manual"],
            "default_pixel_grid_shape": config["source_pixel_grid_shape"],
            "use_best_pixel_size": config["use_best_pixel_size"],
        },
        "numerics": {"supersampling_factor": config["supersampling_factor"]},
        "display": {
            "mtf": config["display_mtf"],
            "source_display": config["source_display"],
            "apply_lensed_arc_mask": config["apply_lensed_arc_mask"],
            "lensed_arcs_unmasked": config["dspl_lensed_arcs_unmasked"],
            "show_caustics": True,
        },
        "mass_profile": config["mass_profile"],
        "mass_prior": config["mass_prior"],
        "lens_plane_mass_components": config["lens_plane_mass_components"],
        "eta_prior": config["eta_prior"],
        "sis_prior": config["sis_prior"],
        "pixelated_prior": config["pixelated_prior"],
        "run_power_init": config["run_power_init"],
        "conjugate_points_source1": config["conjugate_points_source1"],
        "conjugate_points_source2": config["conjugate_points_source2"],
    }


def _single_plane_model_config(config):
    return {
        "svi": {
            "seed": config["seed"],
            "num_chains": config["num_chains"],
            "max_iter_parametric": config["max_iter_parametric"],
            "max_iter_pixelated": config["max_iter_pixelated"],
            "max_iter_solver": config["max_iter_solver"],
        },
        "data": {
            "corner_pixel": config["background_subtract_corner"],
            "use_existing_rms_map": config["use_existing_rms_map"],
            "use_scalar_rms_loguniform": config["use_scalar_rms_loguniform"],
            "automatic_background_rms": config["automatic_background_rms"],
            "background_rms": config["background_rms"],
            "background_subtract_enabled": config["background_subtract_enabled"],
            "background_subtract_corner": config["background_subtract_corner"],
            "exposure_time_if_missing": config["exposure_time"],
        },
        "light": {
            "lens": {
                "profile": config["light_profile"],
                "n_gauss": config["n_gauss_lens"],
                "sigma_lims": config["lens_sigma_lims"],
                "center_max_offset": config["lens_light_center_max_offset"],
                "center_low": -config["lens_light_center_max_offset"],
                "center_high": config["lens_light_center_max_offset"],
                "external_mode": config["lens_light_external_mode"],
                "parametric_mode": config["lens_light_parametric_mode"],
                "pixelated_mode": config["lens_light_pixelated_mode"],
                "external_kwargs_path": config["lens_light_external_path"],
            },
            "source": {
                "profile": config["light_profile"],
                "n_gauss": config["n_gauss_source"],
                "sigma_lims": config["source_sigma_lims"],
            },
            "point_source": config["point_source"],
        },
        "source_grid": {
            "scale": config["source_grid_scale"],
            "pixel_grid_shape": config["source_pixel_grid_shape"],
            "manual": config["source_pixel_grid_manual"],
            "default_pixel_grid_shape": config["source_pixel_grid_shape"],
            "use_best_pixel_size": config["use_best_pixel_size"],
        },
        "numerics": {"supersampling_factor": config["supersampling_factor"]},
        "display": {
            "mtf": config["display_mtf"],
            "source_display": config["source_display"],
        },
        "mass_profile": config["mass_profile"],
        "mass_prior": config["mass_prior"],
        "lens_plane_mass_components": config["lens_plane_mass_components"],
        "pixelated_prior": config["pixelated_prior"],
        "run_power_init": config["run_power_init"],
        "conjugate_points_source1": config["conjugate_points_source1"],
        "exact_two_image_solver": {
            "enabled": config["exact_two_image_solver"],
            "initial_q": [config["solver_initial_theta_E"], config["solver_initial_ellipticity"]],
            "solved": ["theta_E", "ellipticity magnitude"],
        },
        "multiplicative_psf_correction": config["multiplicative_psf_correction"],
        "parametric_init_strategy": config["parametric_init_strategy"],
        "run_semilinear": config["run_semilinear"],
        "chain_selection": config["chain_selection"],
        "stable_parametric_init": config["stable_parametric_init"],
    }


def _number(config, key, default):
    value = config.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _number_or_auto(config, key, default):
    value = config.get(key, default)
    if isinstance(value, str) and value.strip().lower() in AUTO_SIGMA_VALUES:
        return None
    if value in ("", None):
        return default
    return float(value)


def _integer(config, key, default):
    value = config.get(key, default)
    if value in ("", None):
        return default
    return int(value)


def _boolean(config, key, default):
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _pair(config, key, default):
    value = config.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        value = default
    return [float(value[0]), float(value[1])]


def _sigma_pair(config, key, default):
    value = config.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        value = default
    high = value[1]
    if isinstance(high, str) and high.strip().lower() in AUTO_SIGMA_VALUES:
        high = None
    elif high is not None:
        high = float(high)
    return [float(value[0]), high]


def _mtf_dict(config):
    value = config.get("display_mtf") or config.get("mtf") or {}
    if not isinstance(value, dict):
        value = {}
    return {
        "shadows": _number(value, "shadows", 0.0),
        "midtones": _number(value, "midtones", 0.5),
        "highlights": _number(value, "highlights", 1.0),
    }


def _source_display(config):
    value = str(config.get("source_display") or "linear").strip().lower()
    return "log" if value in {"log", "logarithmic", "lognorm"} else "linear"


def _pixelated_prior_dict(config):
    value = config.get("pixelated_prior") or {}
    prior = dict(value) if isinstance(value, dict) else {}
    k_zero = prior.get("k_zero", config.get("k_zero", 0.0))
    if k_zero in ("", None):
        k_zero = 0.0
    prior["k_zero"] = float(k_zero)
    positive = prior.get("positive", config.get("source_positive", True))
    if isinstance(positive, str):
        positive = positive.strip().lower() not in {"", "0", "false", "no", "off"}
    prior["positive"] = bool(positive)
    nonlinear_brightness = prior.get(
        "nonlinear_brightness",
        config.get("source_nonlinear_brightness", True),
    )
    if isinstance(nonlinear_brightness, str):
        nonlinear_brightness = nonlinear_brightness.strip().lower() not in {"", "0", "false", "no", "off"}
    prior["nonlinear_brightness"] = bool(nonlinear_brightness and prior["positive"])
    log_brightness = prior.get("log_brightness", config.get("source_log_brightness", False))
    if isinstance(log_brightness, str):
        log_brightness = log_brightness.strip().lower() not in {"", "0", "false", "no", "off"}
    prior["log_brightness"] = bool(log_brightness)
    if prior["log_brightness"]:
        prior["positive"] = True
        prior["nonlinear_brightness"] = False
    for key, default in (
        ("log_mean_loc", -3.0), ("log_mean_scale", 5.0),
        ("log_n_low", 0.05), ("log_n_high", 3.0),
        ("log_rho_low", 0.5), ("log_rho_high", 40.0),
        ("log_sigma_scale", 4.0),
    ):
        prior[key] = float(prior.get(key, config.get(f"source_{key}", default)))
    if prior["log_n_low"] <= 0 or prior["log_n_low"] >= prior["log_n_high"]:
        raise ValueError("Log-Matérn n prior requires 0 < low < high.")
    if prior["log_rho_low"] <= 0 or prior["log_rho_low"] >= prior["log_rho_high"]:
        raise ValueError("Log-Matérn rho prior requires 0 < low < high.")
    if prior["log_mean_scale"] <= 0 or prior["log_sigma_scale"] <= 0:
        raise ValueError("Log-Matérn scale priors must be positive.")
    return prior


def _point_list(config, key):
    value = config.get(key, [])
    if value in ("", None):
        return []
    points = []
    for point in value:
        if isinstance(point, dict):
            if "arcsec" in point and isinstance(point["arcsec"], dict):
                x_value = point["arcsec"].get("x")
                y_value = point["arcsec"].get("y")
            else:
                x_value = point.get("x")
                y_value = point.get("y")
        else:
            x_value, y_value = point[:2]
        points.append([float(x_value), float(y_value)])
    return points


def _lens_plane_mass_components(config, *, base_dir=None, portable_root=None):
    value = config.get("lens_plane_mass_components", [])
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("lens_plane_mass_components must be valid JSON.") from exc
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("lens_plane_mass_components must be a list.")
    if len(value) > 12:
        raise ValueError("At most 12 additional lens-plane mass components are supported.")

    components = []
    for index, raw_component in enumerate(value, start=1):
        if not isinstance(raw_component, dict):
            raise ValueError(f"Lens-plane mass component {index} must be an object.")
        profile = str(raw_component.get("profile") or "SIS").strip().upper()
        if profile in {"PIXELATED_FIXED", "FIXED", "FIXED_MAP"}:
            profile = "FIXED_MASS_MAP"
        if profile not in {"SIS", "SIE", "FIXED_MASS_MAP"}:
            raise ValueError(f"Lens-plane mass component {index} has unsupported profile {profile!r}.")
        if profile == "FIXED_MASS_MAP":
            path_text = str(raw_component.get("path") or "").strip()
            if not path_text:
                raise ValueError(f"Fixed mass map component {index} requires a FITS path.")
            path = Path(path_text).expanduser()
            if not path.is_absolute() and base_dir is not None:
                path = Path(base_dir) / path
            path = path.resolve()
            if path.suffix.lower() not in {".fits", ".fit", ".fts"}:
                raise ValueError(f"Fixed mass map component {index} must use a FITS file.")
            if not path.is_file():
                raise ValueError(f"Fixed mass map FITS does not exist: {path}")
            stored_path = str(path)
            if portable_root is not None:
                try:
                    stored_path = str(path.relative_to(Path(portable_root).expanduser().resolve()))
                except ValueError:
                    pass
            components.append(
                {
                    "id": f"lens-mass-{index}",
                    "param_name": f"lens_plane_{index}",
                    "profile": profile,
                    "path": stored_path,
                }
            )
            continue
        theta_low = _number(raw_component, "theta_low", 0.0)
        theta_high = _number(raw_component, "theta_high", 1.0)
        center_x = _number(raw_component, "center_x", 0.0)
        center_y = _number(raw_component, "center_y", 0.0)
        e_lim = abs(_number(raw_component, "e_lim", 0.3))
        e_sigma = abs(_number(raw_component, "e_sigma", 0.15))
        if not all(math.isfinite(number) for number in (theta_low, theta_high, center_x, center_y, e_lim, e_sigma)):
            raise ValueError(f"Lens-plane mass component {index} contains a non-finite value.")
        if theta_high <= theta_low:
            raise ValueError(f"Lens-plane mass component {index} requires theta max > theta min.")
        if profile == "SIE" and (e_lim <= 0 or e_lim >= 1):
            raise ValueError(f"Lens-plane SIE component {index} requires 0 < e lim < 1.")
        if profile == "SIE" and e_sigma <= 0:
            raise ValueError(f"Lens-plane SIE component {index} requires sigma e > 0.")
        method = str(raw_component.get("position_method") or "brightest").strip().lower()
        components.append(
            {
                "id": f"lens-mass-{index}",
                "param_name": f"lens_plane_{index}",
                "profile": profile,
                "theta_low": theta_low,
                "theta_high": theta_high,
                "center_x": center_x,
                "center_y": center_y,
                "e_low": -e_lim,
                "e_high": e_lim,
                "e_sigma": e_sigma,
                "position_method": "gaussian" if method == "gaussian" else "brightest",
            }
        )
    return components


def _lens_light_external_helper_code():
    return textwrap.dedent(
        """
        LENS_LIGHT_EXTERNAL = Tian_infra.Light.ExternalLensLight(MODEL_CONFIG, PROJECT_ROOT)
        LENS_LIGHT_EXTERNAL_PARAMETRIC_INIT_VALUES = LENS_LIGHT_EXTERNAL.init_values("parametric")
        LENS_LIGHT_EXTERNAL_PIXELATED_INIT_VALUES = LENS_LIGHT_EXTERNAL.init_values("pixelated")
        """
    ).strip()


def _flatten_saved_model_config(payload):
    config = dict(payload or {})
    svi = config.get("svi") if isinstance(config.get("svi"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    light = config.get("light") if isinstance(config.get("light"), dict) else {}
    lens_light = light.get("lens") if isinstance(light.get("lens"), dict) else {}
    source_light = light.get("source") if isinstance(light.get("source"), dict) else {}
    point_source = (
        config.get("point_source")
        if isinstance(config.get("point_source"), dict)
        else light.get("point_source")
        if isinstance(light.get("point_source"), dict)
        else {}
    )
    source2_light = light.get("source2") if isinstance(light.get("source2"), dict) else {}
    source_grid = config.get("source_grid") if isinstance(config.get("source_grid"), dict) else {}
    numerics = config.get("numerics") if isinstance(config.get("numerics"), dict) else {}
    display = config.get("display") if isinstance(config.get("display"), dict) else {}
    mass_prior = config.get("mass_prior") if isinstance(config.get("mass_prior"), dict) else {}

    for key in ("seed", "num_chains", "max_iter_parametric", "max_iter_pixelated", "max_iter_solver"):
        if key in svi:
            config.setdefault(key, svi[key])
    for key in ("use_existing_rms_map", "use_scalar_rms_loguniform", "automatic_background_rms", "background_rms", "background_subtract_corner", "background_subtract_enabled"):
        if key in data:
            config.setdefault(key, data[key])
    if "exposure_time_if_missing" in data:
        config.setdefault("exposure_time", data["exposure_time_if_missing"])
    if lens_light:
        config.setdefault("light_profile", lens_light.get("profile", "MULTI_GAUSSIAN_ELLIPSE"))
        config.setdefault("n_gauss_lens", lens_light.get("n_gauss"))
        config.setdefault("lens_sigma_lims", lens_light.get("sigma_lims"))
        config.setdefault("lens_light_center_max_offset", lens_light.get("center_max_offset"))
        config.setdefault("lens_light_external_mode", lens_light.get("external_mode"))
        config.setdefault("lens_light_parametric_mode", lens_light.get("parametric_mode"))
        config.setdefault("lens_light_pixelated_mode", lens_light.get("pixelated_mode"))
        config.setdefault("lens_light_external_path", lens_light.get("external_kwargs_path"))
    if source_light:
        config.setdefault("n_gauss_source", source_light.get("n_gauss"))
        config.setdefault("source_sigma_lims", source_light.get("sigma_lims"))
    if point_source:
        config.setdefault("point_source_profile", point_source.get("profile"))
        config.setdefault("point_source_pos_sigma", point_source.get("pos_sigma"))
        config.setdefault("point_source_pos_window", point_source.get("pos_window"))
        config.setdefault("point_source_log10_amp_low", point_source.get("log10_amp_low"))
        config.setdefault("point_source_log10_amp_high", point_source.get("log10_amp_high"))
        config.setdefault("point_source_measured_fluxes", point_source.get("measured_fluxes"))
    if source2_light:
        config.setdefault("n_gauss_source2", source2_light.get("n_gauss"))
        config.setdefault("source2_sigma_lims", source2_light.get("sigma_lims"))
    if source_grid:
        config.setdefault("source_grid_scale", source_grid.get("scale"))
        config.setdefault("source2_grid_scale", source_grid.get("scale2", source_grid.get("scale")))
        config.setdefault("source_pixel_grid_shape", source_grid.get("pixel_grid_shape1", source_grid.get("pixel_grid_shape")))
        config.setdefault("source2_pixel_grid_shape", source_grid.get("pixel_grid_shape2", source_grid.get("pixel_grid_shape")))
        config.setdefault("source_pixel_grid_manual", source_grid.get("manual"))
        config.setdefault("source2_pixel_grid_manual", source_grid.get("manual2", source_grid.get("manual")))
        config.setdefault("default_pixel_grid_shape", source_grid.get("default_pixel_grid_shape"))
        config.setdefault("use_best_pixel_size", source_grid.get("use_best_pixel_size"))
    if "supersampling_factor" in numerics:
        config.setdefault("supersampling_factor", numerics["supersampling_factor"])
    if "mtf" in display:
        config.setdefault("display_mtf", display["mtf"])
    if "source_display" in display:
        config.setdefault("source_display", display["source_display"])
    if "apply_lensed_arc_mask" in display:
        config.setdefault("apply_lensed_arc_mask", display["apply_lensed_arc_mask"])
    if "lensed_arcs_unmasked" in display:
        config.setdefault("dspl_lensed_arcs_unmasked", display["lensed_arcs_unmasked"])
    for key, value in mass_prior.items():
        config.setdefault(key, value)
    return config


def normalized_lens_config(payload, *, job_dir=None, default_data_dir=None):
    payload = _flatten_saved_model_config(payload)
    light_profile = payload.get("light_profile", "MULTI_GAUSSIAN_ELLIPSE")
    lens_sigma_lims = _sigma_pair(payload, "lens_sigma_lims", [0.001, None])
    source_sigma_lims = _pair(payload, "source_sigma_lims", [0.001, 0.2])
    source2_sigma_lims = _pair(payload, "source2_sigma_lims", [0.001, 0.2])
    mass_prior = {
        "theta_low": _number(payload, "theta_low", 0.0),
        "theta_high": _number(payload, "theta_high", 3.0),
        "gamma_low": _number(payload, "gamma_low", 1.2),
        "gamma_high": _number(payload, "gamma_high", 2.8),
        "center_low": _number(payload, "center_low", -0.2),
        "center_high": _number(payload, "center_high", 0.2),
        "center_sigma": _number(payload, "center_sigma", 0.1),
        "e_low": _number(payload, "e_low", -0.2),
        "e_high": _number(payload, "e_high", 0.2),
        "e_sigma": _number(payload, "e_sigma", 0.25),
        "shear_strength_low": max(0.0, _number(payload, "shear_strength_low", 0.0)),
        "shear_strength_high": max(
            0.0,
            _number(
                payload,
                "shear_strength_high",
                _number(
                    payload,
                    "shear_lim",
                    max(
                        abs(_number(payload, "shear_low", -0.2)),
                        abs(_number(payload, "shear_high", 0.2)),
                    ),
                ),
            ),
        ),
    }
    if payload.get("gamma_fixed") not in ("", None):
        mass_prior["gamma_fixed"] = float(payload["gamma_fixed"])

    source_pixel_grid_shape = _integer(payload, "source_pixel_grid_shape", _integer(payload, "default_pixel_grid_shape", 20))
    source2_pixel_grid_shape = _integer(payload, "source2_pixel_grid_shape", source_pixel_grid_shape)
    source_pixel_grid_manual = _boolean(payload, "source_pixel_grid_manual", False)
    source2_pixel_grid_manual = _boolean(payload, "source2_pixel_grid_manual", False)
    data_folder = payload.get("project_folder") or payload.get("data_folder") or default_data_dir or "../Data/Slicelens"
    background_subtract_corner = _integer(payload, "background_subtract_corner", 5)
    lens_light_center_max_offset = _number(payload, "lens_light_center_max_offset", 0.4)
    if lens_light_center_max_offset <= 0:
        raise ValueError("Lens-light Gaussian center max offset must be positive.")
    point_source_profile = str(payload.get("point_source_profile") or "NONE")
    exact_two_image_solver = _boolean(payload, "exact_two_image_solver", False)
    point_source = {"profile": point_source_profile}
    if point_source_profile == "IMAGE_POSITIONS":
        amp_low = float(payload["point_source_log10_amp_low"])
        amp_high = float(payload["point_source_log10_amp_high"])
        if not (math.isfinite(amp_low) and math.isfinite(amp_high) and amp_low < amp_high):
            raise ValueError("Point-source log10 amplitude prior must have finite low < high.")
        fallback_flux = 10.0 ** ((amp_low + amp_high) / 2.0)
        raw_fluxes = payload.get("point_source_measured_fluxes") or []
        count = len(payload.get("conjugate_points_source1") or [])
        measured_fluxes = []
        for index in range(count):
            value = raw_fluxes[index] if index < len(raw_fluxes) else None
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
            measured_fluxes.append(value if value is not None and math.isfinite(value) and value > 0 else None)
        point_source.update(
            {
                "pos_sigma": float(payload["point_source_pos_sigma"]),
                "pos_window": float(payload["point_source_pos_window"]),
                "log10_amp_low": float(payload["point_source_log10_amp_low"]),
                "log10_amp_high": float(payload["point_source_log10_amp_high"]),
                "measured_fluxes": measured_fluxes,
                "initial_fluxes": [value if value is not None else fallback_flux for value in measured_fluxes],
            }
        )
    if exact_two_image_solver:
        if point_source_profile != "IMAGE_POSITIONS" or len(payload.get("conjugate_points_source1") or []) != 2:
            raise ValueError("Exact two-image solver requires IMAGE_POSITIONS and exactly two Source 1 conjugate points.")
        if _boolean(payload, "dspl_enabled", False):
            raise ValueError("Exact two-image solver is currently available only for single-plane models.")
    lens_plane_mass_components = _lens_plane_mass_components(
        payload, base_dir=data_folder, portable_root=data_folder
    )
    if exact_two_image_solver and lens_plane_mass_components:
        raise ValueError("Exact two-image solver currently supports EPL plus shear without extra mass components.")
    if "apply_lensed_arc_mask" in payload:
        apply_lensed_arc_mask = _boolean(payload, "apply_lensed_arc_mask", True)
    else:
        apply_lensed_arc_mask = not _boolean(payload, "dspl_lensed_arcs_unmasked", False)
    return {
        "data_folder": str(data_folder),
        "output_dir": str(job_dir or payload.get("output_dir") or "runs/slicelens_svi"),
        "run_power_init": _boolean(payload, "run_power_init", True),
        "seed": _integer(payload, "seed", 100),
        "num_chains": _integer(payload, "num_chains", 1),
        "max_iter_parametric": _integer(payload, "max_iter_parametric", 10000),
        "max_iter_pixelated": _integer(payload, "max_iter_pixelated", 10000),
        # The exact-solver stage refines an already converged model, so it needs far
        # fewer steps than the two cold-start stages.
        "max_iter_solver": _integer(payload, "max_iter_solver", 5000),
        "use_existing_rms_map": _boolean(payload, "use_existing_rms_map", True),
        "use_scalar_rms_loguniform": _boolean(payload, "use_scalar_rms_loguniform", False),
        "automatic_background_rms": True if _boolean(payload, "use_scalar_rms_loguniform", False) else _boolean(payload, "automatic_background_rms", True),
        "exposure_time": _number(payload, "exposure_time", DEFAULT_EXPOSURE_TIME),
        "background_rms": _number(payload, "background_rms", 0.01),
        "background_subtract_enabled": _boolean(payload, "background_subtract_enabled", False),
        "background_subtract_corner": background_subtract_corner,
        "n_gauss_lens": _integer(payload, "n_gauss_lens", 8),
        "lens_light_center_max_offset": lens_light_center_max_offset,
        "lens_light_external_mode": str(payload.get("lens_light_external_mode") or "free"),
        "lens_light_parametric_mode": str(
            payload.get("lens_light_parametric_mode")
            or payload.get("lens_light_external_mode")
            or "free"
        ),
        "lens_light_pixelated_mode": str(
            payload.get("lens_light_pixelated_mode")
            or payload.get("lens_light_external_mode")
            or "free"
        ),
        "lens_light_external_path": str(
            payload.get("lens_light_external_path")
            or "lens_light_subtraction_result/kwargs_lens_light.pkl"
        ),
        "n_gauss_source": _integer(payload, "n_gauss_source", 1),
        "point_source": point_source,
        "exact_two_image_solver": exact_two_image_solver,
        "solver_initial_theta_E": _number(payload, "solver_initial_theta_E", 0.8),
        "solver_initial_ellipticity": _number(payload, "solver_initial_ellipticity", 0.24),
        "multiplicative_psf_correction": _boolean(payload, "multiplicative_psf_correction", False),
        "parametric_init_strategy": (
            "sample" if str(payload.get("parametric_init_strategy") or "median").strip().lower()
            in {"sample", "random", "prior"} else "median"
        ),
        "run_semilinear": _boolean(payload, "run_semilinear", True),
        "chain_selection": str(payload.get("chain_selection") or "tail_loss"),
        "stable_parametric_init": (
            dict(payload.get("stable_parametric_init"))
            if isinstance(payload.get("stable_parametric_init"), dict) else {}
        ),
        "light_profile": light_profile,
        "lens_sigma_lims": [
            _number(payload, "lens_sigma_min", lens_sigma_lims[0]),
            _number_or_auto(payload, "lens_sigma_max", lens_sigma_lims[1]),
        ],
        "source_sigma_lims": [
            _number(payload, "source_sigma_min", source_sigma_lims[0]),
            _number(payload, "source_sigma_max", source_sigma_lims[1]),
        ],
        "source_grid_scale": _number(payload, "source_grid_scale", 1.0),
        "source_pixel_grid_shape": source_pixel_grid_shape,
        "source_pixel_grid_manual": source_pixel_grid_manual,
        "default_pixel_grid_shape": source_pixel_grid_shape,
        "use_best_pixel_size": _boolean(payload, "use_best_pixel_size", True),
        "supersampling_factor": _integer(payload, "supersampling_factor", 2),
        "display_mtf": _mtf_dict(payload),
        "source_display": _source_display(payload),
        "pixelated_prior": _pixelated_prior_dict(payload),
        "mass_profile": payload.get("mass_profile", "EPL_w_shear"),
        "mass_prior": mass_prior,
        "lens_plane_mass_components": lens_plane_mass_components,
        "dspl_enabled": _boolean(payload, "dspl_enabled", False),
        "apply_lensed_arc_mask": apply_lensed_arc_mask,
        "dspl_lensed_arcs_unmasked": not apply_lensed_arc_mask,
        "n_gauss_source2": _integer(payload, "n_gauss_source2", 1),
        "source2_sigma_lims": [
            _number(payload, "source2_sigma_min", source2_sigma_lims[0]),
            _number(payload, "source2_sigma_max", source2_sigma_lims[1]),
        ],
        "source2_grid_scale": _number(payload, "source2_grid_scale", _number(payload, "source_grid_scale", 1.0)),
        "source2_pixel_grid_shape": source2_pixel_grid_shape,
        "source2_pixel_grid_manual": source2_pixel_grid_manual,
        "eta_prior": {
            "low": _number(payload, "eta_low", 1.0),
            "high": _number(payload, "eta_high", 2.5),
        },
        "sis_prior": {
            "theta_low": _number(payload, "sis_theta_low", 0.0),
            "theta_high": _number(payload, "sis_theta_high", 1.0),
        },
        "conjugate_points_source1": _point_list(payload, "conjugate_points_source1"),
        "conjugate_points_source2": _point_list(payload, "conjugate_points_source2"),
    }


def _generate_dspl_lens_script(config, mass_prior_code):
    model_config_code = _model_config_assignment(_dspl_model_config(config))
    has_conjugate_points_source1 = len(config["conjugate_points_source1"]) > 1
    has_conjugate_points_source2 = len(config["conjugate_points_source2"]) > 1
    source2_conjugate_likelihood_code = ""
    if has_conjugate_points_source2:
        source2_conjugate_likelihood_code = """
        conj_points_at_s2 = lens_image.trace_conjugate_points(eta_flat=eta_flat, kwargs_mass=kwargs_mass, N=2)
        conj_distance_2 = Geometry.reduced_distance_matrix(conj_points_at_s2)
        with numpyro.plate(f"Conjugate points to source 2 - [{conj_distance_2.shape[0]}]", conj_distance_2.shape[0]):
            numpyro.sample("conjugate_points_source2", dist.Exponential(1000), obs=conj_distance_2)
"""
    code = f"""
    # Auto-generated Herculens Double Source Plane Lens Model script.
    # Source: web_gui/lens_code_generator.py

    # %% Runtime setup and imports
    from pathlib import Path
    import pickle
    import sys
    from PIL import Image, ImageDraw, ImageFont

    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import Tian_infra
    Tian_infra.import_function(globals())

    # %% Paths and run configuration
    DATA_DIR = PROJECT_ROOT
    DATA_FILE = DATA_DIR / "Data_cutout.fits"
    RMS_FILE = DATA_DIR / "RMS_map.fits"
    MASK_1_FILE = DATA_DIR / "mask_1.fits"
    MASK_2_FILE = DATA_DIR / "mask_2.fits"
    MASK_OUT_FILE = DATA_DIR / "mask_out.fits"
    PSF_FILE = DATA_DIR / "PSF_model.fits"

    RUN_OUTPUT_DIR = PROJECT_ROOT / "lens_model_result"
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH = RUN_OUTPUT_DIR / "status.json"
    GUI_STATUS = Tian_infra.GUIStatus(STATUS_PATH, RUN_OUTPUT_DIR)

    jax.config.update("jax_enable_x64", True)
    numpyro.enable_x64()
    RUN_TIMER = Tian_infra.start_lens_runtime_timer("DSPL lens model", RUN_OUTPUT_DIR)

    __MODEL_CONFIG_LITERAL__

{textwrap.indent(_lens_light_external_helper_code(), "    ")}

    CONJUGATE_POINTS_SOURCE1 = jnp.asarray(MODEL_CONFIG["conjugate_points_source1"], dtype=jnp.float64)
    CONJUGATE_POINTS_SOURCE2 = jnp.asarray(MODEL_CONFIG["conjugate_points_source2"], dtype=jnp.float64)
    HAS_CONJUGATE_POINTS_SOURCE1 = {has_conjugate_points_source1!r}
    HAS_CONJUGATE_POINTS_SOURCE2 = {has_conjugate_points_source2!r}

    def load_primary_data():
        data, header = fits.getdata(DATA_FILE, header=True)
        data = np.asarray(data, dtype=np.float64)
        pix_scale = Tian_infra.pixel_scale_arcsec_from_header(header)
        exposure_time = Tian_infra.exposure_time_from_header(header, MODEL_CONFIG["data"]["exposure_time_if_missing"])

        subtract_corner = int(MODEL_CONFIG["data"]["background_subtract_corner"])
        subtract_background = bool(MODEL_CONFIG["data"].get("background_subtract_enabled", False))
        background_offset = 0.0
        if subtract_background and subtract_corner > 0:
            background_offset = float(np.nanmedian(data[:subtract_corner, :subtract_corner]))
            data = data - background_offset

        corner_pixel = int(MODEL_CONFIG["data"]["corner_pixel"])
        if MODEL_CONFIG["data"]["automatic_background_rms"]:
            rms = float(np.nanstd(data[:corner_pixel, :corner_pixel]))
        else:
            rms = float(MODEL_CONFIG["data"]["background_rms"])
        use_rms_map = MODEL_CONFIG["data"]["use_existing_rms_map"] and not MODEL_CONFIG["data"].get("use_scalar_rms_loguniform", False)
        rms_map = fits.getdata(RMS_FILE).astype(np.float64) if use_rms_map and RMS_FILE.exists() else None

        source1_arc_mask = fits.getdata(MASK_1_FILE).astype(bool)
        source2_arc_mask = fits.getdata(MASK_2_FILE).astype(bool)
        fit_mask_out = fits.getdata(MASK_OUT_FILE).astype(bool)
        fit_mask = ~fit_mask_out
        psf_kernel = fits.getdata(PSF_FILE).astype(np.float64)
        psf_kernel = psf_kernel / float(psf_kernel.sum())
        return data, header, pix_scale, source1_arc_mask, source2_arc_mask, fit_mask, psf_kernel, rms, rms_map, exposure_time, background_offset


    # %% Load science image, masks, PSF, and build the DSPL parametric LensImage object
    GUI_STATUS.write(
        state="running",
        message="Initializing generated DSPL Lens SVI runner.",
        progress={{"fraction": 0.01, "message": "Initializing generated DSPL Lens SVI runner."}},
    )

    data, header, pix_scale, source1_arc_mask_np, source2_arc_mask_np, fit_mask_np, psf_kernel, rms, rms_map_np, exposure_time, background_offset = load_primary_data()
    MODEL_CONFIG["light"]["lens"]["sigma_lims"] = Tian_infra.resolve_auto_sigma_lims(MODEL_CONFIG["light"]["lens"]["sigma_lims"], data.shape, pix_scale)
    pixel_grid, xgrid, ygrid, x_axis, y_axis, extent, nx, ny = Geometry.get_pixel_grid(data, pix_scale)
    psf = PSF(psf_type="PIXEL", kernel_point_source=psf_kernel)
    noise = Noise(nx, ny, exposure_time=exposure_time)

    source1_arc_mask = jnp.asarray(source1_arc_mask_np, dtype=bool)
    source2_arc_mask = jnp.asarray(source2_arc_mask_np, dtype=bool)
    fit_mask = jnp.asarray(fit_mask_np, dtype=bool)
    rms_map = jnp.asarray(rms_map_np, dtype=jnp.float64) if rms_map_np is not None else None
    npix_fit = int(np.asarray(fit_mask_np).sum())
    rms_prior_factors = (0.8, 1.2) if MODEL_CONFIG["data"].get("use_scalar_rms_loguniform", False) else (0.5, 1.5)
    rms_prior_low = max(float(rms) * rms_prior_factors[0], 1e-12)
    rms_prior_high = max(float(rms) * rms_prior_factors[1], rms_prior_low * 1.01)
    rms_source = "scalar RMS LogUniform" if MODEL_CONFIG["data"].get("use_scalar_rms_loguniform", False) else ("RMS_map.fits" if rms_map_np is not None else "background RMS")
    print(f"DSPL project data: shape={{data.shape}}, pix_scale={{pix_scale:.5g}} arcsec, fit_pixels={{npix_fit}}, rms={{rms:.4g}}, rms_source={{rms_source}}, exposure_time={{exposure_time:.4g}}, source1_conj={{len(MODEL_CONFIG['conjugate_points_source1'])}}, source2_conj={{len(MODEL_CONFIG['conjugate_points_source2'])}}", flush=True)

    LENS_PLANE_COMPONENTS = MODEL_CONFIG["lens_plane_mass_components"]
    LENS_PLANE_MASS_PROFILES = ["EPL", "SHEAR"] + Tian_infra.Mass.lens_plane_component_profiles(
        LENS_PLANE_COMPONENTS
    )


    def lens_plane_mass_from_model():
        return (
            Tian_infra.Mass.EPL_w_shear("Main lens mass", "1", **MODEL_CONFIG["mass_prior"])
            + Tian_infra.Mass.lens_plane_components(LENS_PLANE_COMPONENTS)
        )


    def lens_plane_mass_from_params(params):
        return (
            Tian_infra.Mass.params2kwargs_EPL_w_shear(
                params,
                "1",
                gamma_fixed=MODEL_CONFIG["mass_prior"].get("gamma_fixed"),
            )
            + Tian_infra.Mass.params2kwargs_lens_plane_components(params, LENS_PLANE_COMPONENTS)
        )


    def build_dspl_lens_image(source_type, pixel_grid_shape_s1=None, pixel_grid_shape_s2=None):
        mass_model = MPMassModel([
            MassModel(LENS_PLANE_MASS_PROFILES),
            MassModel(["SIS"]),
        ])
        if source_type == "parametric":
            light_model = MPLightModel([
                LightModel([MODEL_CONFIG["light"]["lens"]["profile"]]),
                LightModel([MODEL_CONFIG["light"]["source1"]["profile"]]),
                LightModel([MODEL_CONFIG["light"]["source2"]["profile"]]),
            ])
        else:
            light_model = MPLightModel([
                LightModel([MODEL_CONFIG["light"]["lens"]["profile"]]),
                LightModel(
                    ["PIXELATED"],
                    pixel_adaptive_grid=True,
                    pixel_interpol="fast_bilinear",
                    kwargs_pixelated={{"num_pixels": int(pixel_grid_shape_s1)}},
                ),
                LightModel(
                    ["PIXELATED"],
                    pixel_adaptive_grid=True,
                    pixel_interpol="fast_bilinear",
                    kwargs_pixelated={{"num_pixels": int(pixel_grid_shape_s2)}},
                ),
            ])
        return MPLensImage(
            deepcopy(pixel_grid),
            deepcopy(psf),
            noise_class=noise,
            light_model_class=light_model,
            mass_model_class=mass_model,
            conjugate_points=[CONJUGATE_POINTS_SOURCE1, CONJUGATE_POINTS_SOURCE2 if HAS_CONJUGATE_POINTS_SOURCE2 else None],
            source_arc_masks=[None, source1_arc_mask, source2_arc_mask],
            kwargs_numerics={{"supersampling_factor": MODEL_CONFIG["numerics"]["supersampling_factor"]}},
            source_grid_scale=[1.0, MODEL_CONFIG["source_grid"]["scale1"], MODEL_CONFIG["source_grid"]["scale2"]],
        )


    lens_image_parametric = build_dspl_lens_image("parametric")


    def dspl_mass_from_params(lens_image, eta_flat, main_lens_mass, sis_theta_E):
        conj_points_at_s1 = lens_image.trace_conjugate_points(
            eta_flat=eta_flat,
            kwargs_mass=[main_lens_mass],
            N=1,
        )
        sis_origin = conj_points_at_s1.mean(axis=0)
        sis_mass = [{{
            "theta_E": sis_theta_E,
            "center_x": sis_origin[0],
            "center_y": sis_origin[1],
        }}]
        return [main_lens_mass, sis_mass]


    def model_lens(data_obs, source_type, k_values_s1=None, k_values_s2=None):
        lens_image = lens_image_parametric if source_type == "parametric" else lens_image_pixelated
        eta = numpyro.sample("eta", dist.Uniform(MODEL_CONFIG["eta_prior"]["low"], MODEL_CONFIG["eta_prior"]["high"]))
        eta_flat = jnp.asarray([eta], dtype=jnp.float64)

        main_lens_mass = lens_plane_mass_from_model()
        lens_light = LENS_LIGHT_EXTERNAL.for_model(source_type)
        conj_points_at_s1 = lens_image.trace_conjugate_points(eta_flat=eta_flat, kwargs_mass=[main_lens_mass], N=1)
        sis_origin = conj_points_at_s1.mean(axis=0)
        if HAS_CONJUGATE_POINTS_SOURCE1:
            conj_distance_1 = Geometry.reduced_distance_matrix(conj_points_at_s1)
            with numpyro.plate(f"Conjugate points to source 1 - [{{conj_distance_1.shape[0]}}]", conj_distance_1.shape[0]):
                numpyro.sample("conjugate_points_source1", dist.Exponential(1000), obs=conj_distance_1)
        sis_mass = Tian_infra.Mass.SIS(
            "Source1 SIS",
            "SIS_s1",
            origin=sis_origin,
            **MODEL_CONFIG["sis_prior"],
        )
        kwargs_mass = [main_lens_mass, sis_mass]
{textwrap.indent(textwrap.dedent(source2_conjugate_likelihood_code).rstrip(), "        ")}

        if source_type == "parametric":
            source1_light = Tian_infra.Light.multi_gauss_light(
                "Source 1 light",
                "source1",
                MODEL_CONFIG["light"]["source1"]["n_gauss"],
                MODEL_CONFIG["light"]["source1"]["sigma_lims"],
            )
            source2_light = Tian_infra.Light.multi_gauss_light(
                "Source 2 light",
                "source2",
                MODEL_CONFIG["light"]["source2"]["n_gauss"],
                MODEL_CONFIG["light"]["source2"]["sigma_lims"],
            )
        else:
            source1_light = [PowerSpectrum.matern_power_spectrum(
                "Source 1 grid",
                "source_grid_s1",
                k_values_s1,
                **PowerSpectrum.matern_prior_kwargs(MODEL_CONFIG["pixelated_prior"]),
            )]
            source2_light = [PowerSpectrum.matern_power_spectrum(
                "Source 2 grid",
                "source_grid_s2",
                k_values_s2,
                **PowerSpectrum.matern_prior_kwargs(MODEL_CONFIG["pixelated_prior"]),
            )]

        model_image = lens_image.model(
            eta_flat=eta_flat,
            kwargs_mass=kwargs_mass,
            kwargs_light=[lens_light, source1_light, source2_light],
            apply_mask=MODEL_CONFIG["display"]["apply_lensed_arc_mask"],
        )
        numpyro.deterministic("model_image", model_image)
        if rms_map is not None:
            model_std = rms_map
        else:
            background_rms_model = numpyro.sample("RMS", dist.LogUniform(rms_prior_low, rms_prior_high))
            model_var = lens_image.Noise.C_D_model(model_image, background_rms=background_rms_model)
            model_std = jnp.sqrt(jnp.maximum(model_var, 1e-12))
        with numpyro.plate(f"Data masked - [{{npix_fit}}]", npix_fit):
            numpyro.sample("obs", dist.Normal(model_image[fit_mask], model_std[fit_mask]), obs=data_obs[fit_mask])


    def params2kwargs_dspl(params, lens_image, source1_kwargs, source2_kwargs, source_type):
        eta_flat = jnp.asarray([params["eta"]], dtype=jnp.float64)
        main_lens_mass = lens_plane_mass_from_params(params)
        kwargs_mass = dspl_mass_from_params(
            lens_image,
            eta_flat,
            main_lens_mass,
            params["theta_E_SIS_s1"][0],
        )
        return {{
            "eta_flat": eta_flat,
            "kwargs_mass": kwargs_mass,
            "kwargs_light": [
                LENS_LIGHT_EXTERNAL.kwargs_from_params(params, source_type),
                source1_kwargs,
                source2_kwargs,
            ],
        }}


    def dspl_pixelated_init_from_parametric(params):
        allowed_prefixes = (
            "eta",
            "theta_E_1",
            "gamma_1",
            "e_1",
            "center_1",
            "gamma_sheer_1",
            "shear_strength_1",
            "shear_position_angle_1",
            "theta_E_SIS_s1",
            "A_lens",
            "amp_lens",
            "sigma_lens",
            "e_lens",
            "center_lens",
            "RMS",
        )
        init_values = {{key: value for key, value in params.items() if key.startswith(allowed_prefixes)}}
        init_values.update(
            Tian_infra.Mass.lens_plane_component_init_values(params, LENS_PLANE_COMPONENTS)
        )
        return init_values


    # %% Parametric two-source model and SVI
    num_chains = int(MODEL_CONFIG["svi"]["num_chains"])
    data_jax = jnp.asarray(data, dtype=jnp.float64)
    parametric_states = []
    parametric_kwargs_list = []

    print("LENS_STAGE parametric", flush=True)
    GUI_STATUS.write(
        state="running",
        message=f"Running DSPL parametric SVI for {{num_chains}} chain(s).",
        progress={{"fraction": 0.02, "message": f"Running DSPL parametric SVI for {{num_chains}} chain(s)."}},
    )
    for chain_index in range(num_chains):
        print(f"LENS_CHAIN parametric {{chain_index + 1}} {{num_chains}}", flush=True)
        parametric_state_i = SVI.run_one_chain_svi(
            model_lens,
            data_jax,
            max_iterations=MODEL_CONFIG["svi"]["max_iter_parametric"],
            seed=MODEL_CONFIG["svi"]["seed"] + chain_index,
            init_values=LENS_LIGHT_EXTERNAL_PARAMETRIC_INIT_VALUES or None,
            learning_rate=0.01,
            init_scale=0.1,
            loss_kind="trace_elbo",
            num_particles=10,
            model_args=("parametric",),
        )
        parametric_kwargs_i = params2kwargs_dspl(
            parametric_state_i["median"],
            lens_image_parametric,
            Tian_infra.Light.params2kwargs_multi_gauss_light(
                parametric_state_i["median"],
                "source1",
                MODEL_CONFIG["light"]["source1"]["n_gauss"],
            ),
            Tian_infra.Light.params2kwargs_multi_gauss_light(
                parametric_state_i["median"],
                "source2",
                MODEL_CONFIG["light"]["source2"]["n_gauss"],
            ),
            "parametric",
        )
        parametric_states.append(parametric_state_i)
        parametric_kwargs_list.append(parametric_kwargs_i)

    parametric_state = parametric_states[0]
    parametric_kwargs = parametric_kwargs_list[0]

    # %% Pixelated source 1 and source 2 model and SVI
    source1_has_mask = bool(np.any(np.asarray(lens_image_parametric._source_arc_masks_flat_bool[1])))
    source2_has_mask = bool(np.any(np.asarray(lens_image_parametric._source_arc_masks_flat_bool[2])))
    if MODEL_CONFIG["source_grid"]["use_best_pixel_size"] and not MODEL_CONFIG["source_grid"]["manual1"] and source1_has_mask:
        pixel_grid_shape_s1 = Geometry.get_best_pixel_size_multiplane(
            lens_image_parametric,
            parametric_kwargs,
            MODEL_CONFIG["source_grid"]["scale1"],
            N=1,
        )
    else:
        pixel_grid_shape_s1 = int(MODEL_CONFIG["source_grid"]["pixel_grid_shape1"])
    if MODEL_CONFIG["source_grid"]["use_best_pixel_size"] and not MODEL_CONFIG["source_grid"]["manual2"] and source2_has_mask:
        pixel_grid_shape_s2 = Geometry.get_best_pixel_size_multiplane(
            lens_image_parametric,
            parametric_kwargs,
            MODEL_CONFIG["source_grid"]["scale2"],
            N=2,
        )
    else:
        pixel_grid_shape_s2 = int(MODEL_CONFIG["source_grid"]["pixel_grid_shape2"])

    k_grid_s1 = PowerSpectrum.K_grid((pixel_grid_shape_s1, pixel_grid_shape_s1))
    k_grid_s2 = PowerSpectrum.K_grid((pixel_grid_shape_s2, pixel_grid_shape_s2))
    lens_image_pixelated = build_dspl_lens_image("pixelated", pixel_grid_shape_s1, pixel_grid_shape_s2)
    pixelated_states = []
    pixelated_kwargs_list = []

    print("LENS_STAGE pixelated", flush=True)
    GUI_STATUS.write(
        state="running",
        message=f"Running DSPL pixelated SVI for {{num_chains}} chain(s).",
        progress={{"fraction": 0.52, "message": f"Running DSPL pixelated SVI for {{num_chains}} chain(s)."}},
    )
    for chain_index, parametric_state_i, parametric_kwargs_i in zip(range(num_chains), parametric_states, parametric_kwargs_list):
        print(f"LENS_CHAIN pixelated {{chain_index + 1}} {{num_chains}}", flush=True)
        pixelated_init_values_i = dspl_pixelated_init_from_parametric(parametric_state_i["median"])
        pixelated_init_values_i.update(LENS_LIGHT_EXTERNAL_PIXELATED_INIT_VALUES)
        if MODEL_CONFIG["run_power_init"]:
            pixelated_init_values_i.update(
                PowerSpectrum.fit_power_spectrum_init_from_parametric_source(
                    lens_image_parametric,
                    parametric_kwargs_i,
                    pixel_grid_shape_s1,
                    MODEL_CONFIG["source_grid"]["scale1"],
                    k_grid_s1.k,
                    MODEL_CONFIG["pixelated_prior"],
                    seed=MODEL_CONFIG["svi"]["seed"] + 7919 + chain_index,
                    param_name="source_grid_s1",
                    source_plane=1,
                )
            )
            pixelated_init_values_i.update(
                PowerSpectrum.fit_power_spectrum_init_from_parametric_source(
                    lens_image_parametric,
                    parametric_kwargs_i,
                    pixel_grid_shape_s2,
                    MODEL_CONFIG["source_grid"]["scale2"],
                    k_grid_s2.k,
                    MODEL_CONFIG["pixelated_prior"],
                    seed=MODEL_CONFIG["svi"]["seed"] + 7920 + chain_index,
                    param_name="source_grid_s2",
                    source_plane=2,
                )
            )
        pixelated_state_i = SVI.run_one_chain_svi(
            model_lens,
            data_jax,
            max_iterations=MODEL_CONFIG["svi"]["max_iter_pixelated"],
            seed=MODEL_CONFIG["svi"]["seed"] + 1000 + chain_index,
            init_values=pixelated_init_values_i,
            learning_rate=0.01,
            init_scale=0.01,
            loss_kind="trace_meanfield_elbo",
            model_args=("pixelated", k_grid_s1.k, k_grid_s2.k),
        )
        pixelated_kwargs_i = params2kwargs_dspl(
            pixelated_state_i["median"],
            lens_image_pixelated,
            [PowerSpectrum.params2kwargs_power_spectrum(
                pixelated_state_i["median"],
                "source_grid_s1",
                k_grid_s1.k,
                **PowerSpectrum.matern_prior_kwargs(MODEL_CONFIG["pixelated_prior"]),
            )],
            [PowerSpectrum.params2kwargs_power_spectrum(
                pixelated_state_i["median"],
                "source_grid_s2",
                k_grid_s2.k,
                **PowerSpectrum.matern_prior_kwargs(MODEL_CONFIG["pixelated_prior"]),
            )],
            "pixelated",
        )
        pixelated_states.append(pixelated_state_i)
        pixelated_kwargs_list.append(pixelated_kwargs_i)

    def svi_tail_mean_loss(state, tail=1000):
        losses = np.asarray(state["losses"], dtype=float)
        if losses.size == 0:
            return np.inf
        tail_size = min(int(tail), losses.size)
        return float(np.nanmean(losses[-tail_size:]))

    best_chain_index = int(np.nanargmin([svi_tail_mean_loss(state) for state in pixelated_states]))
    parametric_state = parametric_states[best_chain_index]
    parametric_kwargs = parametric_kwargs_list[best_chain_index]
    pixelated_state = pixelated_states[best_chain_index]
    pixelated_kwargs = pixelated_kwargs_list[best_chain_index]
    print(f"Selected chain {{best_chain_index + 1}}/{{num_chains}} for visualization.", flush=True)


    def save_loss_comparison(parametric_states_all, pixelated_states_all, selected_chain_index):
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), constrained_layout=True)
        stages = [
            (axes[0], "Parametric SVI loss", parametric_states_all, MODEL_CONFIG["svi"]["max_iter_parametric"], "tab:blue"),
            (axes[1], "Pixelated SVI loss", pixelated_states_all, MODEL_CONFIG["svi"]["max_iter_pixelated"], "tab:orange"),
        ]
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
        output = RUN_OUTPUT_DIR / "losses_comparison.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output

    # Save completed SVI products before visualization, so plotting errors do not discard the chain.
    np.save(RUN_OUTPUT_DIR / "parametric_losses.npy", np.asarray([state["losses"] for state in parametric_states]))
    np.save(RUN_OUTPUT_DIR / "pixelated_losses.npy", np.asarray([state["losses"] for state in pixelated_states]))
    np.save(RUN_OUTPUT_DIR / "selected_chain_index.npy", np.asarray(best_chain_index))
    with open(RUN_OUTPUT_DIR / "parametric_state.pkl", "wb") as f:
        pickle.dump(parametric_state, f)
    with open(RUN_OUTPUT_DIR / "pixelated_state.pkl", "wb") as f:
        pickle.dump(pixelated_state, f)
    with open(RUN_OUTPUT_DIR / "parametric_states.pkl", "wb") as f:
        pickle.dump(parametric_states, f)
    with open(RUN_OUTPUT_DIR / "pixelated_states.pkl", "wb") as f:
        pickle.dump(pixelated_states, f)
    with open(RUN_OUTPUT_DIR / "pixelated_kwargs.pkl", "wb") as f:
        pickle.dump(pixelated_kwargs, f)
    losses_comparison_png = save_loss_comparison(parametric_states, pixelated_states, best_chain_index)


    def residual_noise_from_state(state):
        if rms_map is not None:
            return rms_map
        return float(state["median"].get("RMS", rms))


    def write_panel_fits(path, image, extent=None, panel_name=""):
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
        return "/" + path.relative_to(PROJECT_ROOT).as_posix()


    def save_dspl_panel_fits(
        chain_dir,
        stage_name,
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
        include_lensed_arcs=False,
        apply_lensed_arc_mask=True,
    ):
        panel_dir = chain_dir / stage_name
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
        products = {{
            "data": write_panel_fits(panel_dir / "data.fits", data_obs, image_extent, "data"),
            "data_minus_lens_light": write_panel_fits(panel_dir / "data_minus_lens_light.fits", lens_light_subtracted, image_extent, "data - lens light"),
            "model": write_panel_fits(panel_dir / "model.fits", model_image, image_extent, "model"),
            "data_minus_model_over_rms": write_panel_fits(panel_dir / "data_minus_model_over_rms.fits", residual, image_extent, "data - model / rms"),
            "source1": write_panel_fits(panel_dir / "source1.fits", source1_image, source1_extent, "source 1"),
            "source2": write_panel_fits(panel_dir / "source2.fits", source2_image, source2_extent, "source 2"),
        }}
        if include_lensed_arcs:
            products["lensed_arc1"] = write_panel_fits(
                panel_dir / "lensed_arc1.fits",
                Plot.lensed_arc_image(lens_image, kwargs, 1, apply_mask=apply_lensed_arc_mask),
                image_extent,
                "lensed arc 1",
            )
            products["lensed_arc2"] = write_panel_fits(
                panel_dir / "lensed_arc2.fits",
                Plot.lensed_arc_image(lens_image, kwargs, 2, apply_mask=apply_lensed_arc_mask),
                image_extent,
                "lensed arc 2",
            )
        return products


    parametric_fig = Plot.visualize_dspl_model(
        lens_image_parametric,
        parametric_kwargs,
        data,
        fit_mask,
        residual_noise_from_state(parametric_state),
        MODEL_CONFIG["source_grid"]["pixel_grid_shape1"],
        MODEL_CONFIG["source_grid"]["pixel_grid_shape2"],
        MODEL_CONFIG["source_grid"]["scale1"],
        MODEL_CONFIG["source_grid"]["scale2"],
        extent,
        title="",
        apply_lensed_arc_mask=MODEL_CONFIG["display"]["apply_lensed_arc_mask"],
        mtf=MODEL_CONFIG["display"]["mtf"],
        conjugate_points_source1=CONJUGATE_POINTS_SOURCE1,
        conjugate_points_source2=CONJUGATE_POINTS_SOURCE2 if HAS_CONJUGATE_POINTS_SOURCE2 else None,
        source_display=MODEL_CONFIG["display"].get("source_display", "linear"),
        show_caustics=MODEL_CONFIG["display"].get("show_caustics", True),
    )
    pixelated_fig = Plot.visualize_dspl_model(
        lens_image_pixelated,
        pixelated_kwargs,
        data,
        fit_mask,
        residual_noise_from_state(pixelated_state),
        pixel_grid_shape_s1,
        pixel_grid_shape_s2,
        MODEL_CONFIG["source_grid"]["scale1"],
        MODEL_CONFIG["source_grid"]["scale2"],
        extent,
        title="",
        show_lensed_arcs=True,
        apply_lensed_arc_mask=MODEL_CONFIG["display"]["apply_lensed_arc_mask"],
        mtf=MODEL_CONFIG["display"]["mtf"],
        conjugate_points_source1=CONJUGATE_POINTS_SOURCE1,
        conjugate_points_source2=CONJUGATE_POINTS_SOURCE2 if HAS_CONJUGATE_POINTS_SOURCE2 else None,
        source_display=MODEL_CONFIG["display"].get("source_display", "linear"),
        show_caustics=MODEL_CONFIG["display"].get("show_caustics", True),
    )

    # %% Save products
    def json_value(value):
        array = np.asarray(jax.device_get(value)).reshape(-1)
        if array.size == 1:
            return float(array[0])
        return [float(item) for item in array]


    def dspl_mass_parameter_summary(kwargs):
        main_lens = kwargs["kwargs_mass"][0][0]
        shear = kwargs["kwargs_mass"][0][1]
        sis = kwargs["kwargs_mass"][1][0]
        summary = {{
            "eta": json_value(kwargs["eta_flat"][0]),
            "theta_E": json_value(main_lens["theta_E"]),
            "gamma": json_value(main_lens["gamma"]),
            "e1": json_value(main_lens["e1"]),
            "e2": json_value(main_lens["e2"]),
            "center_x": json_value(main_lens["center_x"]),
            "center_y": json_value(main_lens["center_y"]),
            "gamma1_ext": json_value(shear["gamma1"]),
            "gamma2_ext": json_value(shear["gamma2"]),
            "theta_E_SIS_s1": json_value(sis["theta_E"]),
            "SIS_s1_center_x": json_value(sis["center_x"]),
            "SIS_s1_center_y": json_value(sis["center_y"]),
        }}
        for index, component in enumerate(kwargs["kwargs_mass"][0][2:], start=1):
            profile = "SIE" if "e1" in component else "SIS"
            prefix = f"{{profile}}_{{index}}"
            summary[f"{{prefix}}_theta_E"] = json_value(component["theta_E"])
            summary[f"{{prefix}}_center_x"] = json_value(component["center_x"])
            summary[f"{{prefix}}_center_y"] = json_value(component["center_y"])
            if profile == "SIE":
                summary[f"{{prefix}}_e1"] = json_value(component["e1"])
                summary[f"{{prefix}}_e2"] = json_value(component["e2"])
        return summary


    run_summary = {{
        "run_output_dir": str(RUN_OUTPUT_DIR),
        "mode": MODEL_CONFIG["mode"],
        "mass_profile": MODEL_CONFIG["mass_profile"],
        "seed": MODEL_CONFIG["svi"]["seed"],
        "num_chains": int(num_chains),
        "selected_chain_index": int(best_chain_index),
        "max_iter_parametric": MODEL_CONFIG["svi"]["max_iter_parametric"],
        "max_iter_pixelated": MODEL_CONFIG["svi"]["max_iter_pixelated"],
        "pixel_grid_shape_s1": int(pixel_grid_shape_s1),
        "pixel_grid_shape_s2": int(pixel_grid_shape_s2),
        "data_file": str(DATA_FILE),
        "mask_1_file": str(MASK_1_FILE),
        "mask_2_file": str(MASK_2_FILE),
        "mask_out_file": str(MASK_OUT_FILE),
        "psf_file": str(PSF_FILE),
        "rms_file": str(RMS_FILE) if RMS_FILE.exists() else None,
        "residual_noise": "scalar RMS posterior" if MODEL_CONFIG["data"].get("use_scalar_rms_loguniform", False) else ("RMS_map.fits" if rms_map_np is not None else "SVI posterior RMS"),
        "rms_prior": [float(rms_prior_low), float(rms_prior_high)],
        "conjugate_points_source1": MODEL_CONFIG["conjugate_points_source1"],
        "conjugate_points_source2": MODEL_CONFIG["conjugate_points_source2"],
        "background_offset": float(background_offset),
        "losses_url": "/" + losses_comparison_png.relative_to(PROJECT_ROOT).as_posix(),
        "mass_parameters": dspl_mass_parameter_summary(pixelated_kwargs),
    }}
    (RUN_OUTPUT_DIR / "summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    (RUN_OUTPUT_DIR / "model_config.json").write_text(json.dumps(MODEL_CONFIG, indent=2), encoding="utf-8")

    parametric_png = RUN_OUTPUT_DIR / "parametric_model_comparison.png"
    pixelated_png = RUN_OUTPUT_DIR / "pixelated_model_comparison.png"
    parametric_fig.savefig(parametric_png, dpi=180, bbox_inches="tight")
    pixelated_fig.savefig(pixelated_png, dpi=180, bbox_inches="tight")

    for num in plt.get_fignums():
        fig = plt.figure(num)
        out_png = RUN_OUTPUT_DIR / f"figure_{{num:02d}}.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight")

    latest = RUN_OUTPUT_DIR / "latest_four_panel_comparison.png"

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

    compose_lens_preview(
        latest,
        [
            ("Parametric model", parametric_png),
            ("Pixelated model", pixelated_png),
        ],
    )
    chain_previews = []
    for chain_index in range(num_chains):
        chain_tag = f"chain_{{chain_index + 1:02d}}"
        chain_dir = RUN_OUTPUT_DIR / chain_tag
        chain_parametric_fig = Plot.visualize_dspl_model(
            lens_image_parametric,
            parametric_kwargs_list[chain_index],
            data,
            fit_mask,
            residual_noise_from_state(parametric_states[chain_index]),
            MODEL_CONFIG["source_grid"]["pixel_grid_shape1"],
            MODEL_CONFIG["source_grid"]["pixel_grid_shape2"],
            MODEL_CONFIG["source_grid"]["scale1"],
            MODEL_CONFIG["source_grid"]["scale2"],
            extent,
            title="",
            apply_lensed_arc_mask=MODEL_CONFIG["display"]["apply_lensed_arc_mask"],
            mtf=MODEL_CONFIG["display"]["mtf"],
            conjugate_points_source1=CONJUGATE_POINTS_SOURCE1,
            conjugate_points_source2=CONJUGATE_POINTS_SOURCE2 if HAS_CONJUGATE_POINTS_SOURCE2 else None,
            source_display=MODEL_CONFIG["display"].get("source_display", "linear"),
            show_caustics=MODEL_CONFIG["display"].get("show_caustics", True),
        )
        chain_pixelated_fig = Plot.visualize_dspl_model(
            lens_image_pixelated,
            pixelated_kwargs_list[chain_index],
            data,
            fit_mask,
            residual_noise_from_state(pixelated_states[chain_index]),
            pixel_grid_shape_s1,
            pixel_grid_shape_s2,
            MODEL_CONFIG["source_grid"]["scale1"],
            MODEL_CONFIG["source_grid"]["scale2"],
            extent,
            title="",
            show_lensed_arcs=True,
            apply_lensed_arc_mask=MODEL_CONFIG["display"]["apply_lensed_arc_mask"],
            mtf=MODEL_CONFIG["display"]["mtf"],
            conjugate_points_source1=CONJUGATE_POINTS_SOURCE1,
            conjugate_points_source2=CONJUGATE_POINTS_SOURCE2 if HAS_CONJUGATE_POINTS_SOURCE2 else None,
            source_display=MODEL_CONFIG["display"].get("source_display", "linear"),
            show_caustics=MODEL_CONFIG["display"].get("show_caustics", True),
        )
        chain_parametric_panel_fits = save_dspl_panel_fits(
            chain_dir,
            "parametric",
            lens_image_parametric,
            parametric_kwargs_list[chain_index],
            data,
            fit_mask,
            residual_noise_from_state(parametric_states[chain_index]),
            MODEL_CONFIG["source_grid"]["pixel_grid_shape1"],
            MODEL_CONFIG["source_grid"]["pixel_grid_shape2"],
            MODEL_CONFIG["source_grid"]["scale1"],
            MODEL_CONFIG["source_grid"]["scale2"],
            extent,
            include_lensed_arcs=False,
            apply_lensed_arc_mask=MODEL_CONFIG["display"]["apply_lensed_arc_mask"],
        )
        chain_pixelated_panel_fits = save_dspl_panel_fits(
            chain_dir,
            "pixelated",
            lens_image_pixelated,
            pixelated_kwargs_list[chain_index],
            data,
            fit_mask,
            residual_noise_from_state(pixelated_states[chain_index]),
            pixel_grid_shape_s1,
            pixel_grid_shape_s2,
            MODEL_CONFIG["source_grid"]["scale1"],
            MODEL_CONFIG["source_grid"]["scale2"],
            extent,
            include_lensed_arcs=True,
            apply_lensed_arc_mask=MODEL_CONFIG["display"]["apply_lensed_arc_mask"],
        )
        chain_parametric_png = RUN_OUTPUT_DIR / f"{{chain_tag}}_parametric_model_comparison.png"
        chain_pixelated_png = RUN_OUTPUT_DIR / f"{{chain_tag}}_pixelated_model_comparison.png"
        chain_latest = RUN_OUTPUT_DIR / f"{{chain_tag}}_lens_model_comparison.png"
        chain_parametric_fig.savefig(chain_parametric_png, dpi=180, bbox_inches="tight")
        chain_pixelated_fig.savefig(chain_pixelated_png, dpi=180, bbox_inches="tight")
        compose_lens_preview(
            chain_latest,
            [
                (f"Chain {{chain_index + 1}} parametric model", chain_parametric_png),
                (f"Chain {{chain_index + 1}} pixelated model", chain_pixelated_png),
            ],
        )
        plt.close(chain_parametric_fig)
        plt.close(chain_pixelated_fig)
        chain_previews.append({{
            "chain": int(chain_index + 1),
            "selected": bool(chain_index == best_chain_index),
            "preview_url": "/" + chain_latest.relative_to(PROJECT_ROOT).as_posix(),
            "parametric_url": "/" + chain_parametric_png.relative_to(PROJECT_ROOT).as_posix(),
            "pixelated_url": "/" + chain_pixelated_png.relative_to(PROJECT_ROOT).as_posix(),
            "parametric_panel_fits": chain_parametric_panel_fits,
            "pixelated_panel_fits": chain_pixelated_panel_fits,
            "mass_parameters": dspl_mass_parameter_summary(pixelated_kwargs_list[chain_index]),
            "final_parametric_loss": float(np.asarray(parametric_states[chain_index]["losses"])[-1]),
            "final_pixelated_loss": float(np.asarray(pixelated_states[chain_index]["losses"])[-1]),
        }})
    run_summary["chain_previews"] = chain_previews
    runtime_summary = Tian_infra.finish_lens_runtime_timer(RUN_TIMER, status="completed")
    run_summary["runtime"] = runtime_summary
    (RUN_OUTPUT_DIR / "summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    preview_url = "/" + latest.relative_to(PROJECT_ROOT).as_posix()
    GUI_STATUS.write(
        state="completed",
        message="DSPL lens model completed.",
        preview_url=preview_url,
        losses_url="/" + losses_comparison_png.relative_to(PROJECT_ROOT).as_posix(),
        chain_previews=chain_previews,
        runtime=runtime_summary,
        progress={{"fraction": 1.0, "message": "DSPL lens model completed."}},
    )
    print(f"LONG_SVI_RUN_DONE output_dir={{RUN_OUTPUT_DIR}}", flush=True)
    """
    return textwrap.dedent(code).replace("__MODEL_CONFIG_LITERAL__", model_config_code).strip() + "\n"


def _dspl_hmc_extension_code():
    return textwrap.dedent(
        """

        # %% Pixelated DSPL HMC from the completed SVI state
        # This cell follows the Herculens_3DSPL_EPL.py HMC/Gibbs pattern after # In[22].
        import os

        from numpyro.infer import MCMC, NUTS
        from custom_gibbs import MultiHMCGibbs

        HMC_OUTPUT_DIR = PROJECT_ROOT / "lens_model_hmc_result"
        HMC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        hmc_netcdf_root = os.environ.get("HERCULENS_HMC_NETCDF_ROOT")
        HMC_NETCDF_DIR = (
            Path(hmc_netcdf_root).expanduser() / PROJECT_ROOT.name / HMC_OUTPUT_DIR.name
            if hmc_netcdf_root
            else HMC_OUTPUT_DIR
        )
        HMC_NETCDF_DIR.mkdir(parents=True, exist_ok=True)
        HMC_STATUS_PATH = HMC_OUTPUT_DIR / "status.json"
        HMC_STATUS = Tian_infra.GUIStatus(HMC_STATUS_PATH, HMC_OUTPUT_DIR)
        HMC_TIMER = Tian_infra.start_lens_runtime_timer("DSPL pixelated HMC", HMC_OUTPUT_DIR)

        HMC_CONFIG = {
            "num_warmup": 2000,
            "num_samples": 1000,
            "batch_number": 4,
            "target_accept_lens_light": 0.8,
            "target_accept_source": 0.9,
            "target_accept_mass": 0.95,
            "max_tree_depth": 10,
            "netcdf_name": "Euclid_DSPL_HMC.nc",
        }
        (HMC_OUTPUT_DIR / "hmc_config.json").write_text(json.dumps(HMC_CONFIG, indent=2), encoding="utf-8")

        def hmc_write_status(state, message, fraction, **extra):
            payload = HMC_STATUS.write(
                state=state,
                message=message,
                progress={"fraction": float(fraction), "message": message},
                model_config=MODEL_CONFIG,
                **extra,
            )
            GUI_STATUS.write(
                state=state,
                message=message,
                hmc_output_dir=str(HMC_OUTPUT_DIR),
                hmc_netcdf_dir=str(HMC_NETCDF_DIR),
                hmc_status_path=str(HMC_STATUS_PATH),
                hmc_progress={"fraction": float(fraction), "message": message},
                **extra,
            )
            return payload

        def hmc_get_free_parameter_list(model_fn, args_tuple, values):
            _, model_trace = infer.util.log_density(
                model_fn,
                args_tuple,
                {},
                ResumeInit.get_value_from_index(values, 0),
            )
            return [
                name
                for name, site in dict(model_trace).items()
                if site.get("type") == "sample" and not site.get("is_observed", False)
            ]

        def hmc_keep_free(names, free_names):
            return [name for name in names if name in free_names]

        print("HMC_STAGE initialize", flush=True)
        hmc_write_status("running", "Preparing DSPL HMC from pixelated SVI median.", 0.02)

        if "pixelated_states" not in globals() or not pixelated_states:
            raise RuntimeError("Pixelated SVI states are missing; run the full SVI section before HMC.")
        if "lens_image_pixelated" not in globals():
            raise RuntimeError("lens_image_pixelated is missing; run the full DSPL pixelated SVI section before HMC.")

        multi_svi_pixel_median = jax.tree.map(
            lambda *xs: jnp.stack(xs),
            *[state["median"] for state in pixelated_states],
        )
        hmc_model_args = (data_jax, "pixelated", k_grid_s1.k, k_grid_s2.k)
        free_params_keys = hmc_get_free_parameter_list(model_lens, hmc_model_args, multi_svi_pixel_median)
        print(f"HMC_FREE_PARAMS {free_params_keys}", flush=True)

        multi_svi_pixel_median_vars = {
            key: multi_svi_pixel_median[key]
            for key in free_params_keys
            if key in multi_svi_pixel_median
        }
        unconstrained_svi_pixel_median = jax.vmap(
            lambda params: infer.util.unconstrain_fn(model_lens, hmc_model_args, {}, params)
        )(multi_svi_pixel_median_vars)
        unconstrained_svi_pixel_median = {
            key: jnp.asarray(value, dtype=jnp.float64)
            for key, value in unconstrained_svi_pixel_median.items()
        }

        init_fun_pixel = ResumeInit.init_to_value_or_defer(
            values=ResumeInit.get_value_from_index(multi_svi_pixel_median, 0)
        )

        vars_lens_light = hmc_keep_free(
            ["A_lens", "sigma_lens", "e_lens", "center_lens"],
            free_params_keys,
        )
        vars_pixel_s1 = hmc_keep_free(["pixels_wn_source_grid_s1"], free_params_keys)
        vars_pixel_s2 = hmc_keep_free(["pixels_wn_source_grid_s2"], free_params_keys)
        vars_power_s1 = hmc_keep_free(
            [
                "n_source_grid_s1",
                "rho_source_grid_s1",
                "sigma_source_grid_s1",
                "pow_lam_source_grid_s1",
                "scale_lam_source_grid_s1",
            ],
            free_params_keys,
        )
        vars_power_s2 = hmc_keep_free(
            [
                "n_source_grid_s2",
                "rho_source_grid_s2",
                "sigma_source_grid_s2",
                "pow_lam_source_grid_s2",
                "scale_lam_source_grid_s2",
            ],
            free_params_keys,
        )
        vars_source = vars_pixel_s1 + vars_power_s1 + vars_pixel_s2 + vars_power_s2
        vars_lens_mass = hmc_keep_free(
            ["theta_E_1", "gamma_1", "e_1", "center_1", "shear_strength_1", "shear_position_angle_1", "theta_E_SIS_s1"]
            + Tian_infra.Mass.lens_plane_component_hmc_vars(LENS_PLANE_COMPONENTS),
            free_params_keys,
        )
        vars_others = hmc_keep_free(["eta", "RMS"], free_params_keys)
        gibbs_sites_list = [vars_lens_light, vars_source, vars_lens_mass + vars_others]
        assigned = {name for group in gibbs_sites_list for name in group}
        unassigned = [name for name in free_params_keys if name not in assigned]
        if unassigned:
            gibbs_sites_list[-1].extend(unassigned)
        gibbs_sites_list = [group for group in gibbs_sites_list if group]
        if not gibbs_sites_list:
            raise RuntimeError("No free HMC parameters were found.")
        print(f"HMC_GIBBS_SITES {gibbs_sites_list}", flush=True)

        inner_kernels = [
            NUTS(
                model_lens,
                init_strategy=init_fun_pixel,
                target_accept_prob=HMC_CONFIG["target_accept_lens_light"],
                max_tree_depth=HMC_CONFIG["max_tree_depth"],
                dense_mass=[("A_lens", "center_lens", "e_lens", "sigma_lens")],
            ),
            NUTS(
                model_lens,
                init_strategy=init_fun_pixel,
                target_accept_prob=HMC_CONFIG["target_accept_source"],
                max_tree_depth=HMC_CONFIG["max_tree_depth"],
                dense_mass=[
                    ("n_source_grid_s1", "rho_source_grid_s1", "sigma_source_grid_s1"),
                    ("n_source_grid_s2", "rho_source_grid_s2", "sigma_source_grid_s2"),
                ],
            ),
            NUTS(
                model_lens,
                init_strategy=init_fun_pixel,
                target_accept_prob=HMC_CONFIG["target_accept_mass"],
                max_tree_depth=HMC_CONFIG["max_tree_depth"],
                dense_mass=[
                    ("theta_E_1", "gamma_1"),
                    ("theta_E_SIS_s1", "eta"),
                    ("e_1", "shear_strength_1", "shear_position_angle_1"),
                ] + Tian_infra.Mass.lens_plane_component_dense_mass_groups(LENS_PLANE_COMPONENTS),
            ),
        ][:len(gibbs_sites_list)]

        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            gibbs_sites_list=gibbs_sites_list,
        )
        mcmc_pixel = MCMC(
            outer_kernel,
            num_warmup=int(HMC_CONFIG["num_warmup"]),
            num_samples=int(HMC_CONFIG["num_samples"]),
            num_chains=num_chains,
            progress_bar=True,
            chain_method="vectorized",
        )

        hmc_rng_key = jax.random.PRNGKey(int(MODEL_CONFIG["svi"]["seed"]) + 100000)
        batch_list = []
        batch_paths = []
        last_states = []
        for batch_index in range(int(HMC_CONFIG["batch_number"])):
            batch_id = batch_index + 1
            print(f"HMC_BATCH {batch_id}/{HMC_CONFIG['batch_number']}", flush=True)
            hmc_write_status(
                "running",
                f"Running DSPL HMC batch {batch_id}/{HMC_CONFIG['batch_number']}.",
                0.05 + 0.85 * batch_index / max(int(HMC_CONFIG["batch_number"]), 1),
                hmc_batch=batch_id,
            )
            if batch_index == 0:
                hmc_rng_key, run_key = jax.random.split(hmc_rng_key)
                mcmc_pixel.run(
                    run_key,
                    *hmc_model_args,
                    init_params=unconstrained_svi_pixel_median,
                )
            else:
                mcmc_pixel.post_warmup_state = mcmc_pixel.last_state
                mcmc_pixel.run(
                    mcmc_pixel.post_warmup_state.rng_key,
                    *hmc_model_args,
                )

            last_states.append(jax.device_get(mcmc_pixel.last_state))
            mcmc_pixel._states = jax.device_get(mcmc_pixel._states)
            mcmc_pixel._states_flat = jax.device_get(mcmc_pixel._states_flat)
            current_batch = az.from_numpyro(mcmc_pixel)
            if int(HMC_CONFIG["batch_number"]) == 1:
                batch_path = HMC_NETCDF_DIR / HMC_CONFIG["netcdf_name"]
            else:
                batch_path = HMC_NETCDF_DIR / f"DSPL_EPL_HMC_batch_{batch_id:02d}.nc"
            current_batch.to_netcdf(batch_path)
            print(f"HMC_BATCH_DONE {batch_id}/{HMC_CONFIG['batch_number']} path={batch_path}", flush=True)
            batch_list.append(current_batch)
            batch_paths.append(str(batch_path))

        print("HMC_STAGE concat", flush=True)
        hmc_write_status("running", "Concatenating DSPL HMC batches.", 0.94)
        if len(batch_list) == 1:
            inf_data_pixel = batch_list[0]
            hmc_netcdf_path = Path(batch_paths[0])
        else:
            inf_data_pixel = az.concat(*batch_list, dim="draw")
            hmc_netcdf_path = HMC_NETCDF_DIR / HMC_CONFIG["netcdf_name"]
            inf_data_pixel.to_netcdf(hmc_netcdf_path)
        divergences = inf_data_pixel.sample_stats.diverging.values.sum(axis=1).T
        print(f"HMC_DIVERGENCES_PER_CHAIN_PER_STEP {divergences}", flush=True)

        hmc_summary = {
            "output_dir": str(HMC_OUTPUT_DIR),
            "netcdf_dir": str(HMC_NETCDF_DIR),
            "netcdf_path": str(hmc_netcdf_path),
            "batch_paths": batch_paths,
            "batch_number": int(HMC_CONFIG["batch_number"]),
            "num_warmup": int(HMC_CONFIG["num_warmup"]),
            "num_samples": int(HMC_CONFIG["num_samples"]),
            "num_chains": int(num_chains),
            "free_params": free_params_keys,
            "gibbs_sites_list": gibbs_sites_list,
            "divergences_per_chain_per_step": np.asarray(divergences).tolist(),
        }
        runtime_summary_hmc = Tian_infra.finish_lens_runtime_timer(HMC_TIMER, status="completed")
        hmc_summary["runtime"] = runtime_summary_hmc
        (HMC_OUTPUT_DIR / "summary.json").write_text(json.dumps(hmc_summary, indent=2), encoding="utf-8")
        hmc_write_status(
            "completed",
            "DSPL HMC completed.",
            1.0,
            hmc_summary=hmc_summary,
            runtime=runtime_summary_hmc,
            netcdf_path=str(hmc_netcdf_path),
        )
        print(f"HMC_RUN_DONE output_dir={HMC_OUTPUT_DIR} netcdf={hmc_netcdf_path}", flush=True)
        jax.clear_caches()
        """
    ).strip() + "\n"


def _single_plane_hmc_extension_code():
    return textwrap.dedent(
        """

        # %% Pixelated single-plane HMC from the completed SVI state
        # This cell starts from the pixelated SVI median and runs Gibbs-grouped NUTS.
        import os

        from numpyro.infer import MCMC, NUTS
        from custom_gibbs import MultiHMCGibbs

        HMC_OUTPUT_DIR = PROJECT_ROOT / "lens_model_hmc_result"
        HMC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        hmc_netcdf_root = os.environ.get("HERCULENS_HMC_NETCDF_ROOT")
        HMC_NETCDF_DIR = (
            Path(hmc_netcdf_root).expanduser() / PROJECT_ROOT.name / HMC_OUTPUT_DIR.name
            if hmc_netcdf_root
            else HMC_OUTPUT_DIR
        )
        HMC_NETCDF_DIR.mkdir(parents=True, exist_ok=True)
        HMC_STATUS_PATH = HMC_OUTPUT_DIR / "status.json"
        HMC_STATUS = Tian_infra.GUIStatus(HMC_STATUS_PATH, HMC_OUTPUT_DIR)
        HMC_TIMER = Tian_infra.start_lens_runtime_timer("single-plane pixelated HMC", HMC_OUTPUT_DIR)

        HMC_CONFIG = {
            "num_warmup": 2000,
            "num_samples": 1000,
            "batch_number": 4,
            "target_accept_lens_light": 0.8,
            "target_accept_source": 0.9,
            "target_accept_mass": 0.95,
            "max_tree_depth": 10,
            "netcdf_name": "single_plane_HMC.nc",
        }
        (HMC_OUTPUT_DIR / "hmc_config.json").write_text(json.dumps(HMC_CONFIG, indent=2), encoding="utf-8")

        def hmc_write_status(state, message, fraction, **extra):
            payload = HMC_STATUS.write(
                state=state,
                message=message,
                progress={"fraction": float(fraction), "message": message},
                model_config=MODEL_CONFIG,
                **extra,
            )
            GUI_STATUS.write(
                state=state,
                message=message,
                hmc_output_dir=str(HMC_OUTPUT_DIR),
                hmc_netcdf_dir=str(HMC_NETCDF_DIR),
                hmc_status_path=str(HMC_STATUS_PATH),
                hmc_progress={"fraction": float(fraction), "message": message},
                **extra,
            )
            return payload

        def hmc_get_free_parameter_list(model_fn, args_tuple, values):
            _, model_trace = infer.util.log_density(
                model_fn,
                args_tuple,
                {},
                ResumeInit.get_value_from_index(values, 0),
            )
            return [
                name
                for name, site in dict(model_trace).items()
                if site.get("type") == "sample" and not site.get("is_observed", False)
            ]

        def hmc_keep_free(names, free_names):
            return [name for name in names if name in free_names]

        def hmc_dense_mass(candidate_groups, free_names):
            dense_groups = []
            for candidate_group in candidate_groups:
                group = tuple(name for name in candidate_group if name in free_names)
                if len(group) > 1:
                    dense_groups.append(group)
            return dense_groups or False

        print("HMC_STAGE initialize", flush=True)
        hmc_write_status("running", "Preparing single-plane HMC from pixelated SVI median.", 0.02)

        if "pixelated_states" not in globals() or not pixelated_states:
            raise RuntimeError("Pixelated SVI states are missing; run the full SVI section before HMC.")
        if "lens_image_pixelated" not in globals():
            raise RuntimeError("lens_image_pixelated is missing; run the full pixelated SVI section before HMC.")

        multi_svi_pixel_median = jax.tree.map(
            lambda *xs: jnp.stack(xs),
            *[state["median"] for state in pixelated_states],
        )
        hmc_model_args = (data_jax, "pixelated", k_grid.k)
        free_params_keys = hmc_get_free_parameter_list(model_lens, hmc_model_args, multi_svi_pixel_median)
        print(f"HMC_FREE_PARAMS {free_params_keys}", flush=True)

        multi_svi_pixel_median_vars = {
            key: multi_svi_pixel_median[key]
            for key in free_params_keys
            if key in multi_svi_pixel_median
        }
        unconstrained_svi_pixel_median = jax.vmap(
            lambda params: infer.util.unconstrain_fn(model_lens, hmc_model_args, {}, params)
        )(multi_svi_pixel_median_vars)
        unconstrained_svi_pixel_median = {
            key: jnp.asarray(value, dtype=jnp.float64)
            for key, value in unconstrained_svi_pixel_median.items()
        }

        init_fun_pixel = ResumeInit.init_to_value_or_defer(
            values=ResumeInit.get_value_from_index(multi_svi_pixel_median, 0)
        )

        vars_lens_light = hmc_keep_free(
            ["A_lens", "sigma_lens", "e_lens", "center_lens"],
            free_params_keys,
        )
        vars_point_source = hmc_keep_free(
            ["ra_ps", "dec_ps", "log10_amp_ps"],
            free_params_keys,
        )
        vars_pixel_source = hmc_keep_free(["pixels_wn_source_grid"], free_params_keys)
        vars_power_source = hmc_keep_free(
            [
                "n_source_grid",
                "rho_source_grid",
                "sigma_source_grid",
                "pow_lam_source_grid",
                "scale_lam_source_grid",
            ],
            free_params_keys,
        )
        vars_source = vars_pixel_source + vars_power_source + vars_point_source
        vars_lens_mass = hmc_keep_free(
            ["theta_E_1", "gamma_1", "e_1", "center_1", "shear_strength_1", "shear_position_angle_1"]
            + Tian_infra.Mass.lens_plane_component_hmc_vars(LENS_PLANE_COMPONENTS),
            free_params_keys,
        )
        vars_others = hmc_keep_free(["RMS"], free_params_keys)
        gibbs_sites_list = [vars_lens_light, vars_source, vars_lens_mass + vars_others]
        assigned = {name for group in gibbs_sites_list for name in group}
        unassigned = [name for name in free_params_keys if name not in assigned]
        if unassigned:
            gibbs_sites_list[-1].extend(unassigned)
        gibbs_sites_list = [group for group in gibbs_sites_list if group]
        if not gibbs_sites_list:
            raise RuntimeError("No free HMC parameters were found.")
        print(f"HMC_GIBBS_SITES {gibbs_sites_list}", flush=True)

        kernel_specs = [
            {
                "sites": vars_lens_light,
                "target_accept": HMC_CONFIG["target_accept_lens_light"],
                "dense_mass": hmc_dense_mass(
                    [("A_lens", "center_lens", "e_lens", "sigma_lens")],
                    free_params_keys,
                ),
            },
            {
                "sites": vars_source,
                "target_accept": HMC_CONFIG["target_accept_source"],
                "dense_mass": hmc_dense_mass(
                    [
                        ("n_source_grid", "rho_source_grid", "sigma_source_grid"),
                        ("ra_ps", "dec_ps", "log10_amp_ps"),
                    ],
                    free_params_keys,
                ),
            },
            {
                "sites": vars_lens_mass + vars_others + unassigned,
                "target_accept": HMC_CONFIG["target_accept_mass"],
                "dense_mass": hmc_dense_mass(
                    [
                        ("theta_E_1", "gamma_1"),
                        ("e_1", "shear_strength_1", "shear_position_angle_1"),
                        ("theta_E_1", "center_1"),
                        *Tian_infra.Mass.lens_plane_component_dense_mass_groups(LENS_PLANE_COMPONENTS),
                    ],
                    free_params_keys,
                ),
            },
        ]
        kernel_specs = [spec for spec in kernel_specs if spec["sites"]]
        inner_kernels = [
            NUTS(
                model_lens,
                init_strategy=init_fun_pixel,
                target_accept_prob=spec["target_accept"],
                max_tree_depth=HMC_CONFIG["max_tree_depth"],
                dense_mass=spec["dense_mass"],
            )
            for spec in kernel_specs
        ]

        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            gibbs_sites_list=gibbs_sites_list,
        )
        mcmc_pixel = MCMC(
            outer_kernel,
            num_warmup=int(HMC_CONFIG["num_warmup"]),
            num_samples=int(HMC_CONFIG["num_samples"]),
            num_chains=num_chains,
            progress_bar=True,
            chain_method="vectorized",
        )

        hmc_rng_key = jax.random.PRNGKey(int(MODEL_CONFIG["svi"]["seed"]) + 100000)
        batch_list = []
        batch_paths = []
        last_states = []
        for batch_index in range(int(HMC_CONFIG["batch_number"])):
            batch_id = batch_index + 1
            print(f"HMC_BATCH {batch_id}/{HMC_CONFIG['batch_number']}", flush=True)
            hmc_write_status(
                "running",
                f"Running single-plane HMC batch {batch_id}/{HMC_CONFIG['batch_number']}.",
                0.05 + 0.85 * batch_index / max(int(HMC_CONFIG["batch_number"]), 1),
                hmc_batch=batch_id,
            )
            if batch_index == 0:
                hmc_rng_key, run_key = jax.random.split(hmc_rng_key)
                mcmc_pixel.run(
                    run_key,
                    *hmc_model_args,
                    init_params=unconstrained_svi_pixel_median,
                )
            else:
                mcmc_pixel.post_warmup_state = mcmc_pixel.last_state
                mcmc_pixel.run(
                    mcmc_pixel.post_warmup_state.rng_key,
                    *hmc_model_args,
                )

            last_states.append(jax.device_get(mcmc_pixel.last_state))
            mcmc_pixel._states = jax.device_get(mcmc_pixel._states)
            mcmc_pixel._states_flat = jax.device_get(mcmc_pixel._states_flat)
            current_batch = az.from_numpyro(mcmc_pixel)
            if int(HMC_CONFIG["batch_number"]) == 1:
                batch_path = HMC_NETCDF_DIR / HMC_CONFIG["netcdf_name"]
            else:
                batch_path = HMC_NETCDF_DIR / f"single_plane_HMC_batch_{batch_id:02d}.nc"
            current_batch.to_netcdf(batch_path)
            print(f"HMC_BATCH_DONE {batch_id}/{HMC_CONFIG['batch_number']} path={batch_path}", flush=True)
            batch_list.append(current_batch)
            batch_paths.append(str(batch_path))

        print("HMC_STAGE concat", flush=True)
        hmc_write_status("running", "Concatenating single-plane HMC batches.", 0.94)
        if len(batch_list) == 1:
            inf_data_pixel = batch_list[0]
            hmc_netcdf_path = Path(batch_paths[0])
        else:
            inf_data_pixel = az.concat(*batch_list, dim="draw")
            hmc_netcdf_path = HMC_NETCDF_DIR / HMC_CONFIG["netcdf_name"]
            inf_data_pixel.to_netcdf(hmc_netcdf_path)
        divergences = inf_data_pixel.sample_stats.diverging.values.sum(axis=1).T
        print(f"HMC_DIVERGENCES_PER_CHAIN_PER_STEP {divergences}", flush=True)

        hmc_summary = {
            "output_dir": str(HMC_OUTPUT_DIR),
            "netcdf_dir": str(HMC_NETCDF_DIR),
            "netcdf_path": str(hmc_netcdf_path),
            "batch_paths": batch_paths,
            "batch_number": int(HMC_CONFIG["batch_number"]),
            "num_warmup": int(HMC_CONFIG["num_warmup"]),
            "num_samples": int(HMC_CONFIG["num_samples"]),
            "num_chains": int(num_chains),
            "free_params": free_params_keys,
            "gibbs_sites_list": gibbs_sites_list,
            "divergences_per_chain_per_step": np.asarray(divergences).tolist(),
        }
        runtime_summary_hmc = Tian_infra.finish_lens_runtime_timer(HMC_TIMER, status="completed")
        hmc_summary["runtime"] = runtime_summary_hmc
        (HMC_OUTPUT_DIR / "summary.json").write_text(json.dumps(hmc_summary, indent=2), encoding="utf-8")
        hmc_write_status(
            "completed",
            "Single-plane HMC completed.",
            1.0,
            hmc_summary=hmc_summary,
            runtime=runtime_summary_hmc,
            netcdf_path=str(hmc_netcdf_path),
        )
        print(f"HMC_RUN_DONE output_dir={HMC_OUTPUT_DIR} netcdf={hmc_netcdf_path}", flush=True)
        jax.clear_caches()
        """
    ).strip() + "\n"


def generate_lens_hmc_script(payload, *, job_dir=None, default_data_dir=None):
    config = normalized_lens_config(payload, job_dir=job_dir, default_data_dir=default_data_dir)
    if config["mass_profile"] != "EPL_w_shear":
        raise ValueError("Only EPL_w_shear is supported for the generated HMC script.")
    if config["light_profile"] != "MULTI_GAUSSIAN_ELLIPSE":
        raise ValueError("Only MULTI_GAUSSIAN_ELLIPSE is supported for the generated HMC script.")
    mass_prior_code = _dict_literal(config["mass_prior"])
    if config["dspl_enabled"]:
        svi_code = _generate_dspl_lens_script(config, mass_prior_code)
        return svi_code.rstrip() + "\n\n" + _dspl_hmc_extension_code(), config
    svi_code, _ = generate_lens_script(payload, job_dir=job_dir, default_data_dir=default_data_dir)
    return svi_code.rstrip() + "\n\n" + _single_plane_hmc_extension_code(), config


def _single_plane_semilinear_output_code(enabled):
    if not enabled:
        return textwrap.dedent(
            """
            # %% Save SVI outputs without semilinear source inversion
            print("LENS_STAGE outputs (semilinear disabled)", flush=True)
            Tian_infra.LensResultWriter.save_single_plane_svi_outputs_from_context(locals())
            """
        ).strip()

    return textwrap.dedent(
        """
        # %% Fixed-mass semilinear source inversion for the best SVI chain
        best_chain_index = Tian_infra.LensResultWriter.best_svi_chain_index(
            pixelated_states
        )
        semilinear_noise = Tian_infra.LensResultWriter.residual_noise_from_state(
            pixelated_states[best_chain_index],
            rms_map,
            rms,
        )

        print(
            f"LENS_STAGE semilinear chain {best_chain_index + 1}",
            flush=True,
        )
        semilinear_result = semisolve(
            lens_image=lens_image_pixelated,
            fixed_kwargs=pixelated_kwargs_list[best_chain_index],
            data=data,
            noise=semilinear_noise,
            fit_mask=fit_mask,
            source_shape=(pixel_grid_shape, pixel_grid_shape),
            positive=MODEL_CONFIG["pixelated_prior"].get("positive", True),
            progress_callback=partial(
                Tian_infra.report_semilinear_progress,
                GUI_STATUS,
            ),
        )
        semilinear_source_dir = (
            RUN_OUTPUT_DIR
            / f"chain_{best_chain_index + 1:02d}"
            / "semilinear"
        )
        semilinear_source_dir.mkdir(parents=True, exist_ok=True)
        semilinear_source_path = (
            semilinear_source_dir / "source_pixels_raw.fits"
        )
        semilinear_source_header = fits.Header()
        semilinear_source_header["METHOD"] = "SEMILINEAR"
        semilinear_source_header["REGULAR"] = "GRADIENT"
        semilinear_source_header["CHAIN"] = int(best_chain_index + 1)
        semilinear_source_header["RLAMBDA"] = float(
            semilinear_result.selected_relative_lambda
        )
        semilinear_source_header["ALAMBDA"] = float(
            semilinear_result.selected_actual_lambda
        )
        fits.writeto(
            semilinear_source_path,
            np.asarray(semilinear_result.source_pixels, dtype=np.float32),
            header=semilinear_source_header,
            overwrite=True,
        )

        Tian_infra.LensResultWriter.save_single_plane_svi_outputs_from_context(locals())
        """
    ).strip()


def generate_lens_script(payload, *, job_dir=None, default_data_dir=None):
    config = normalized_lens_config(payload, job_dir=job_dir, default_data_dir=default_data_dir)
    if config["light_profile"] != "MULTI_GAUSSIAN_ELLIPSE":
        raise ValueError("Only MULTI_GAUSSIAN_ELLIPSE is supported in the first Light Model builder version.")

    mass_prior_code = _dict_literal(config["mass_prior"])
    if config["dspl_enabled"]:
        if config["mass_profile"] != "EPL_w_shear":
            raise ValueError("Only EPL_w_shear is supported in the DSPL Lens Model builder.")
        return _generate_dspl_lens_script(config, mass_prior_code), config
    if config["mass_profile"] != "EPL_w_shear":
        raise ValueError("Only EPL_w_shear is supported in the single-plane Lens Model builder.")
    model_config_code = _model_config_assignment(_single_plane_model_config(config))
    has_conjugate_points_source1 = len(config["conjugate_points_source1"]) > 0
    has_conjugate_constraint_source1 = len(config["conjugate_points_source1"]) > 1
    has_point_sources = config["point_source"]["profile"] == "IMAGE_POSITIONS"

    semilinear_import = "from semilinear_solver import semisolve" if config["run_semilinear"] else ""
    semilinear_output_code = _single_plane_semilinear_output_code(config["run_semilinear"])

    code = f"""
    # Auto-generated Herculens Lens Model script.
    # Source: web_gui/lens_code_generator.py
    # This file is intentionally notebook-like: each "# %%" block can be pasted
    # into a Jupyter cell and read from top to bottom.

    # %% Runtime setup and imports
    from functools import partial
    from pathlib import Path
    import sys
    from PIL import Image, ImageDraw, ImageFont

    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import Tian_infra
    Tian_infra.import_function(globals())
    __SEMILINEAR_IMPORT__
    from theta_e_solver import ThetaEllipticitySolver

    # %% Paths and run configuration
    DATA_DIR = PROJECT_ROOT
    DATA_FILE = DATA_DIR / "Data_cutout.fits"
    RMS_FILE = DATA_DIR / "RMS_map.fits"
    MASK_1_FILE = DATA_DIR / "mask_1.fits"
    MASK_OUT_FILE = DATA_DIR / "mask_out.fits"
    PSF_FILE = DATA_DIR / "PSF_model.fits"

    RUN_OUTPUT_DIR = PROJECT_ROOT / "lens_model_result"
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH = RUN_OUTPUT_DIR / "status.json"
    GUI_STATUS = Tian_infra.GUIStatus(STATUS_PATH, RUN_OUTPUT_DIR)

    jax.config.update("jax_enable_x64", True)
    numpyro.enable_x64()
    RUN_TIMER = Tian_infra.start_lens_runtime_timer("single-plane lens model", RUN_OUTPUT_DIR)

    __MODEL_CONFIG_LITERAL__

{textwrap.indent(_lens_light_external_helper_code(), "    ")}

    CONJUGATE_POINTS_SOURCE1 = jnp.asarray(MODEL_CONFIG["conjugate_points_source1"], dtype=jnp.float64)
    HAS_CONJUGATE_POINTS_SOURCE1 = {has_conjugate_points_source1!r}
    HAS_CONJUGATE_CONSTRAINT_SOURCE1 = {has_conjugate_constraint_source1!r}
    HAS_POINT_SOURCES = {has_point_sources!r}
    # The exact two-image solver now defines a third SVI stage rather than replacing the
    # conjugate-point likelihood throughout: stages 1-2 fit with the smooth conjugate
    # likelihood, and stage 3 re-fits with theta_E and |e| solved exactly from the two
    # image positions, warm-started from stage 2 so the Newton solve begins inside its
    # convergence basin.
    RUN_SOLVER_STAGE = bool(MODEL_CONFIG["exact_two_image_solver"]["enabled"])
    INITIAL_SOLVER_Q = jnp.asarray(MODEL_CONFIG["exact_two_image_solver"]["initial_q"], dtype=jnp.float64)
    MASS_SOLVER = ThetaEllipticitySolver() if RUN_SOLVER_STAGE else None
    if HAS_POINT_SOURCES:
        POINT_SOURCE_MEASURED_FLUXES = jnp.asarray(
            MODEL_CONFIG["light"]["point_source"]["initial_fluxes"],
            dtype=jnp.float64,
        )
        POINT_SOURCE_INIT_VALUES = {{
            "ra_ps": CONJUGATE_POINTS_SOURCE1[:, 0],
            "dec_ps": CONJUGATE_POINTS_SOURCE1[:, 1],
            "log10_amp_ps": jnp.log10(POINT_SOURCE_MEASURED_FLUXES),
        }}
    else:
        POINT_SOURCE_INIT_VALUES = {{}}


    def load_primary_data():
        data, header = fits.getdata(DATA_FILE, header=True)
        data = np.asarray(data, dtype=np.float64)
        pix_scale = Tian_infra.pixel_scale_arcsec_from_header(header)
        exposure_time = Tian_infra.exposure_time_from_header(header, MODEL_CONFIG["data"]["exposure_time_if_missing"])

        subtract_corner = int(MODEL_CONFIG["data"]["background_subtract_corner"])
        subtract_background = bool(MODEL_CONFIG["data"].get("background_subtract_enabled", False))
        background_offset = 0.0
        if subtract_background and subtract_corner > 0:
            corner = data[:subtract_corner, :subtract_corner]
            background_offset = float(np.nanmedian(corner))
            data = data - background_offset

        corner_pixel = int(MODEL_CONFIG["data"]["corner_pixel"])
        if MODEL_CONFIG["data"]["automatic_background_rms"]:
            noise_pixels = data[:corner_pixel, :corner_pixel]
            rms = float(np.nanstd(noise_pixels))
        else:
            rms = float(MODEL_CONFIG["data"]["background_rms"])
        use_rms_map = MODEL_CONFIG["data"]["use_existing_rms_map"] and not MODEL_CONFIG["data"].get("use_scalar_rms_loguniform", False)
        rms_map = fits.getdata(RMS_FILE).astype(np.float64) if use_rms_map and RMS_FILE.exists() else None

        source_arc_mask = fits.getdata(MASK_1_FILE).astype(bool)
        fit_mask_out = fits.getdata(MASK_OUT_FILE).astype(bool)
        fit_mask = ~fit_mask_out
        psf_kernel = fits.getdata(PSF_FILE).astype(np.float64)
        psf_kernel = psf_kernel / float(psf_kernel.sum())
        return data, header, pix_scale, source_arc_mask, fit_mask, psf_kernel, rms, rms_map, exposure_time, background_offset


    # %% Load science image, masks, PSF, and build the first LensImage object
    GUI_STATUS.write(
        state="running",
        message="Initializing generated Lens SVI runner.",
        progress={{"fraction": 0.01, "message": "Initializing generated Lens SVI runner."}},
    )

    data, header, pix_scale, source_arc_mask_np, fit_mask_np, psf_kernel, rms, rms_map_np, exposure_time, background_offset = load_primary_data()
    MODEL_CONFIG["light"]["lens"]["sigma_lims"] = Tian_infra.resolve_auto_sigma_lims(MODEL_CONFIG["light"]["lens"]["sigma_lims"], data.shape, pix_scale)
    pixel_grid, xgrid, ygrid, x_axis, y_axis, extent, nx, ny = Geometry.get_pixel_grid(data, pix_scale)
    psf = PSF(psf_type="PIXEL", kernel_point_source=psf_kernel)
    PSF_CORRECTION = Tian_infra.MultiplicativePSFCorrection(psf_kernel)
    noise = Noise(nx, ny, exposure_time=exposure_time)

    source_arc_mask = jnp.asarray(source_arc_mask_np, dtype=bool)
    fit_mask = jnp.asarray(fit_mask_np, dtype=bool)
    rms_map = jnp.asarray(rms_map_np, dtype=jnp.float64) if rms_map_np is not None else None
    npix_fit = int(np.asarray(fit_mask_np).sum())
    rms_prior_factors = (0.8, 1.2) if MODEL_CONFIG["data"].get("use_scalar_rms_loguniform", False) else (0.5, 1.5)
    rms_prior_low = max(float(rms) * rms_prior_factors[0], 1e-12)
    rms_prior_high = max(float(rms) * rms_prior_factors[1], rms_prior_low * 1.01)
    rms_source = "scalar RMS LogUniform" if MODEL_CONFIG["data"].get("use_scalar_rms_loguniform", False) else ("RMS_map.fits" if rms_map_np is not None else "background RMS")
    print(f"Project data: shape={{data.shape}}, pix_scale={{pix_scale:.5g}} arcsec, fit_pixels={{npix_fit}}, rms={{rms:.4g}}, rms_source={{rms_source}}, exposure_time={{exposure_time:.4g}}, background_offset={{background_offset:.4g}}, source_conj={{len(MODEL_CONFIG['conjugate_points_source1'])}}", flush=True)

    LENS_PLANE_COMPONENTS = MODEL_CONFIG["lens_plane_mass_components"]
    LENS_PLANE_MASS_PROFILES = ["EPL", "SHEAR"] + Tian_infra.Mass.lens_plane_component_profiles(
        LENS_PLANE_COMPONENTS
    )


    def lens_plane_mass_from_model(point_source=None, use_solver=False, solver_initial_q=None):
        if use_solver:
            points = jnp.stack([point_source[0]["ra"], point_source[0]["dec"]], axis=1)
            initial_q = INITIAL_SOLVER_Q if solver_initial_q is None else solver_initial_q
            return MASS_SOLVER.sample_numpyro(points, MODEL_CONFIG["mass_prior"], initial_q)
        return (
            Tian_infra.Mass.EPL_w_shear("Mass model", "1", **MODEL_CONFIG["mass_prior"])
            + Tian_infra.Mass.lens_plane_components(LENS_PLANE_COMPONENTS)
        )


    def lens_plane_mass_from_params(params, use_solver=False, solver_initial_q=None):
        if use_solver:
            initial_q = INITIAL_SOLVER_Q if solver_initial_q is None else solver_initial_q
            return MASS_SOLVER.from_params(params, initial_q)
        return (
            Tian_infra.Mass.params2kwargs_EPL_w_shear(
                params,
                "1",
                gamma_fixed=MODEL_CONFIG["mass_prior"].get("gamma_fixed"),
            )
            + Tian_infra.Mass.params2kwargs_lens_plane_components(params, LENS_PLANE_COMPONENTS)
        )


    def point_source_from_model():
        if not HAS_POINT_SOURCES:
            return None
        point_source_config = MODEL_CONFIG["light"]["point_source"]
        ra_ps = numpyro.sample(
            "ra_ps",
            dist.TruncatedNormal(
                loc=CONJUGATE_POINTS_SOURCE1[:, 0],
                scale=point_source_config["pos_sigma"],
                low=CONJUGATE_POINTS_SOURCE1[:, 0] - point_source_config["pos_window"],
                high=CONJUGATE_POINTS_SOURCE1[:, 0] + point_source_config["pos_window"],
            ).to_event(1),
        )
        dec_ps = numpyro.sample(
            "dec_ps",
            dist.TruncatedNormal(
                loc=CONJUGATE_POINTS_SOURCE1[:, 1],
                scale=point_source_config["pos_sigma"],
                low=CONJUGATE_POINTS_SOURCE1[:, 1] - point_source_config["pos_window"],
                high=CONJUGATE_POINTS_SOURCE1[:, 1] + point_source_config["pos_window"],
            ).to_event(1),
        )
        log10_amp_ps = numpyro.sample(
            "log10_amp_ps",
            dist.Uniform(
                point_source_config["log10_amp_low"],
                point_source_config["log10_amp_high"],
            ).expand([CONJUGATE_POINTS_SOURCE1.shape[0]]).to_event(1),
        )
        return [{{"ra": ra_ps, "dec": dec_ps, "amp": jnp.power(10.0, log10_amp_ps)}}]


    def point_source_from_params(params):
        if not HAS_POINT_SOURCES:
            return None
        return [{{
            "ra": params["ra_ps"],
            "dec": params["dec_ps"],
            "amp": jnp.power(10.0, params["log10_amp_ps"]),
        }}]

    # %% Parametric source model and SVI
    mass_model_parametric = MassModel(LENS_PLANE_MASS_PROFILES)
    point_source_model_parametric = (
        PointSourceModel(
            [MODEL_CONFIG["light"]["point_source"]["profile"]],
            mass_model=mass_model_parametric,
            image_plane=pixel_grid,
        )
        if HAS_POINT_SOURCES
        else None
    )
    lens_image_parametric = LensImageExtension(
        deepcopy(pixel_grid),
        deepcopy(psf),
        noise_class=noise,
        lens_light_model_class=LightModel([MODEL_CONFIG["light"]["lens"]["profile"]]),
        lens_mass_model_class=mass_model_parametric,
        source_model_class=LightModel([MODEL_CONFIG["light"]["source"]["profile"]]),
        point_source_model_class=point_source_model_parametric,
        source_arc_mask=source_arc_mask,
        conjugate_points=CONJUGATE_POINTS_SOURCE1 if HAS_CONJUGATE_POINTS_SOURCE1 else None,
        kwargs_numerics={{"supersampling_factor": MODEL_CONFIG["numerics"]["supersampling_factor"]}},
        source_grid_scale=MODEL_CONFIG["source_grid"]["scale"],
    )


    def model_lens(data_obs, source_type, k_values=None, use_solver=False, solver_initial_q=None):
        point_source = point_source_from_model()
        mass_params = lens_plane_mass_from_model(point_source, use_solver, solver_initial_q)
        lens_light = LENS_LIGHT_EXTERNAL.for_model(source_type)
        if source_type == "parametric":
            lens_image = lens_image_parametric
            source_light = Tian_infra.Light.multi_gauss_light(
                "Source light",
                "source",
                MODEL_CONFIG["light"]["source"]["n_gauss"],
                MODEL_CONFIG["light"]["source"]["sigma_lims"],
            )
        else:
            lens_image = lens_image_pixelated
            source_light = [PowerSpectrum.matern_power_spectrum(
                "Source grid",
                "source_grid",
                k_values,
                **PowerSpectrum.matern_prior_kwargs(MODEL_CONFIG["pixelated_prior"]),
            )]

        psf_noise_fft = None
        if source_type != "parametric" and MODEL_CONFIG["multiplicative_psf_correction"]:
            psf_noise_fft = {{"pixels": PSF_CORRECTION.sample()}}

        if HAS_CONJUGATE_CONSTRAINT_SOURCE1 and not use_solver:
            if HAS_POINT_SOURCES:
                source_x_ps, source_y_ps = lens_image.MassModel.ray_shooting(
                    point_source[0]["ra"],
                    point_source[0]["dec"],
                    mass_params,
                )
                conj_points_at_source = jnp.stack([source_x_ps, source_y_ps], axis=1)
            else:
                conj_points_at_source = lens_image.trace_conjugate_points(kwargs_lens=mass_params)
            conj_distance = Geometry.reduced_distance_matrix(conj_points_at_source)
            with numpyro.plate(f"Conjugate points to source - [{{conj_distance.shape[0]}}]", conj_distance.shape[0]):
                numpyro.sample("conjugate_points_source1", dist.Exponential(1000), obs=conj_distance)

        model_image = lens_image.model(
            kwargs_lens=mass_params,
            kwargs_source=source_light,
            kwargs_lens_light=lens_light,
            kwargs_point_source=point_source,
            source_add=True,
            point_source_add=HAS_POINT_SOURCES,
            psf_noise_fft=psf_noise_fft,
        )
        numpyro.deterministic("model_image", model_image)
        if rms_map is not None:
            model_std = rms_map
        else:
            background_rms_model = numpyro.sample("RMS", dist.LogUniform(rms_prior_low, rms_prior_high))
            model_var = lens_image.Noise.C_D_model(model_image, background_rms=background_rms_model)
            model_std = jnp.sqrt(jnp.maximum(model_var, 1e-12))
        with numpyro.plate(f"Data masked - [{{npix_fit}}]", npix_fit):
            numpyro.sample("obs", dist.Normal(model_image[fit_mask], model_std[fit_mask]), obs=data_obs[fit_mask])


    num_chains = int(MODEL_CONFIG["svi"]["num_chains"])
    data_jax = jnp.asarray(data, dtype=jnp.float64)
    parametric_init_values = {{}}
    if MODEL_CONFIG["parametric_init_strategy"] != "sample":
        parametric_init_values.update(LENS_LIGHT_EXTERNAL_PARAMETRIC_INIT_VALUES)
        parametric_init_values.update(POINT_SOURCE_INIT_VALUES)
        parametric_init_values.update({{
            key: jnp.asarray(value, dtype=jnp.float64)
            for key, value in MODEL_CONFIG.get("stable_parametric_init", {{}}).items()
        }})
    parametric_states = []
    parametric_kwargs_list = []

    print("LENS_STAGE parametric", flush=True)
    GUI_STATUS.write(
        state="running",
        message=f"Running parametric SVI for {{num_chains}} chain(s).",
        progress={{"fraction": 0.02, "message": f"Running parametric SVI for {{num_chains}} chain(s)."}},
    )
    for chain_index in range(num_chains):
        print(f"LENS_CHAIN parametric {{chain_index + 1}} {{num_chains}}", flush=True)
        parametric_state_i = Tian_infra.SVI.run_one_chain_svi(
            model_lens,
            data_jax,
            max_iterations=MODEL_CONFIG["svi"]["max_iter_parametric"],
            seed=MODEL_CONFIG["svi"]["seed"] + chain_index,
            init_values=parametric_init_values,
            learning_rate=0.01,
            init_scale=0.1,
            loss_kind="trace_elbo",
            num_particles=10,
            model_args=("parametric",),
            init_strategy=MODEL_CONFIG["parametric_init_strategy"],
        )
        parametric_kwargs_i = {{
            "kwargs_lens": lens_plane_mass_from_params(parametric_state_i["median"]),
            "kwargs_lens_light": LENS_LIGHT_EXTERNAL.kwargs_from_params(parametric_state_i["median"], "parametric"),
            "kwargs_point_source": point_source_from_params(parametric_state_i["median"]),
            "kwargs_source": Tian_infra.Light.params2kwargs_multi_gauss_light(
                parametric_state_i["median"],
                "source",
                MODEL_CONFIG["light"]["source"]["n_gauss"],
            ),
        }}
        parametric_states.append(parametric_state_i)
        parametric_kwargs_list.append(parametric_kwargs_i)

    parametric_state = parametric_states[0]
    parametric_kwargs = parametric_kwargs_list[0]

    # %% Pixelated source model and SVI
    source_has_mask = bool(np.any(np.asarray(lens_image_parametric._source_arc_mask_flat)))
    if MODEL_CONFIG["source_grid"]["use_best_pixel_size"] and not MODEL_CONFIG["source_grid"]["manual"] and source_has_mask:
        pixel_grid_shape = Geometry.get_best_pixel_size(lens_image_parametric, parametric_kwargs, MODEL_CONFIG["source_grid"]["scale"])
    else:
        pixel_grid_shape = int(MODEL_CONFIG["source_grid"]["pixel_grid_shape"])
    k_grid = PowerSpectrum.K_grid((pixel_grid_shape, pixel_grid_shape))

    pixel_grid_pixelated = deepcopy(pixel_grid)
    mass_model_pixelated = MassModel(LENS_PLANE_MASS_PROFILES)
    point_source_model_pixelated = (
        PointSourceModel(
            [MODEL_CONFIG["light"]["point_source"]["profile"]],
            mass_model=mass_model_pixelated,
            image_plane=pixel_grid_pixelated,
        )
        if HAS_POINT_SOURCES
        else None
    )
    lens_image_pixelated = LensImageExtension(
        pixel_grid_pixelated,
        deepcopy(psf),
        noise_class=noise,
        lens_light_model_class=LightModel([MODEL_CONFIG["light"]["lens"]["profile"]]),
        lens_mass_model_class=mass_model_pixelated,
        source_model_class=LightModel(
            ["PIXELATED"],
            pixel_adaptive_grid=True,
            pixel_interpol="fast_bilinear",
            kwargs_pixelated={{"num_pixels": pixel_grid_shape}},
        ),
        point_source_model_class=point_source_model_pixelated,
        source_arc_mask=source_arc_mask,
        conjugate_points=CONJUGATE_POINTS_SOURCE1 if HAS_CONJUGATE_POINTS_SOURCE1 else None,
        kwargs_numerics={{"supersampling_factor": MODEL_CONFIG["numerics"]["supersampling_factor"]}},
        source_grid_scale=MODEL_CONFIG["source_grid"]["scale"],
    )

    pixelated_states = []
    pixelated_kwargs_list = []

    print("LENS_STAGE pixelated", flush=True)
    GUI_STATUS.write(
        state="running",
        message=f"Running pixelated SVI for {{num_chains}} chain(s).",
        progress={{"fraction": 0.52, "message": f"Running pixelated SVI for {{num_chains}} chain(s)."}},
    )
    for chain_index, parametric_state_i, parametric_kwargs_i in zip(range(num_chains), parametric_states, parametric_kwargs_list):
        print(f"LENS_CHAIN pixelated {{chain_index + 1}} {{num_chains}}", flush=True)
        pixelated_init_values_i = ResumeInit.pixelated_stage_init_from_parametric(parametric_state_i["median"])
        pixelated_init_values_i.update(LENS_LIGHT_EXTERNAL_PIXELATED_INIT_VALUES)
        pixelated_init_values_i.update(
            Tian_infra.Mass.lens_plane_component_init_values(
                parametric_state_i["median"],
                LENS_PLANE_COMPONENTS,
            )
        )
        if MODEL_CONFIG["run_power_init"]:
            pixelated_init_values_i.update(
                PowerSpectrum.fit_power_spectrum_init_from_parametric_source(
                    lens_image_parametric,
                    parametric_kwargs_i,
                    pixel_grid_shape,
                    MODEL_CONFIG["source_grid"]["scale"],
                    k_grid.k,
                    MODEL_CONFIG["pixelated_prior"],
                    seed=MODEL_CONFIG["svi"]["seed"] + 7919 + chain_index,
                )
            )
        pixelated_state_i = Tian_infra.SVI.run_one_chain_svi(
            model_lens,
            data_jax,
            max_iterations=MODEL_CONFIG["svi"]["max_iter_pixelated"],
            seed=MODEL_CONFIG["svi"]["seed"] + 1000 + chain_index,
            init_values=pixelated_init_values_i,
            learning_rate=0.01,
            init_scale=0.01,
            loss_kind="trace_meanfield_elbo",
            model_args=("pixelated", k_grid.k),
        )
        pixelated_kwargs_i = {{
            "kwargs_lens": lens_plane_mass_from_params(pixelated_state_i["median"]),
            "kwargs_lens_light": LENS_LIGHT_EXTERNAL.kwargs_from_params(pixelated_state_i["median"], "pixelated"),
            "kwargs_point_source": point_source_from_params(pixelated_state_i["median"]),
            "kwargs_source": [PowerSpectrum.params2kwargs_power_spectrum(
                pixelated_state_i["median"],
                "source_grid",
                k_grid.k,
                **PowerSpectrum.matern_prior_kwargs(MODEL_CONFIG["pixelated_prior"]),
            )],
            "psf_noise_fft": (
                {{"pixels": PSF_CORRECTION.from_params(pixelated_state_i["median"])}}
                if MODEL_CONFIG["multiplicative_psf_correction"] else None
            ),
        }}
        pixelated_states.append(pixelated_state_i)
        pixelated_kwargs_list.append(pixelated_kwargs_i)

    # %% Checkpoint stages 1-2
    # The result writer only runs at the very end, so without this a failure in the
    # solver stage would discard both completed SVI stages.  Written before stage 3
    # starts so the expensive part is always recoverable.
    # Store plain arrays only.  An SVI state holds the guide, which closes over
    # `model_lens`, so a pickled state cannot be loaded outside this script - useless for
    # offline debugging or for resuming stage 3.  Medians, losses and kwargs are enough
    # for both.
    STAGE12_CHECKPOINT = RUN_OUTPUT_DIR / "stage12_checkpoint.pkl"
    with open(STAGE12_CHECKPOINT, "wb") as f:
        pickle.dump(
            jax.device_get({{
                "parametric_medians": [state["median"] for state in parametric_states],
                "parametric_losses": [np.asarray(state["losses"]) for state in parametric_states],
                "parametric_kwargs_list": parametric_kwargs_list,
                "pixelated_medians": [state["median"] for state in pixelated_states],
                "pixelated_losses": [np.asarray(state["losses"]) for state in pixelated_states],
                "pixelated_kwargs_list": pixelated_kwargs_list,
                "pixel_grid_shape": pixel_grid_shape,
                "num_chains": num_chains,
                "mass_prior": MODEL_CONFIG["mass_prior"],
                "initial_solver_q": np.asarray(INITIAL_SOLVER_Q),
            }}),
            f,
        )
    print(f"LENS_CHECKPOINT stages 1-2 saved {{STAGE12_CHECKPOINT}}", flush=True)

    # %% Exact two-image solver SVI
    # Stages 1-2 tie the two conjugate images together through a smooth Exponential
    # likelihood.  This stage drops that term and instead solves theta_E and the
    # ellipticity magnitude exactly from the two image positions, so the constraint is
    # enforced instead of penalised.  Each chain is warm-started from its own stage-2
    # solution, which is what keeps the Newton solve inside its convergence basin.
    solver_states = []
    solver_kwargs_list = []
    stage2_states = None

    if RUN_SOLVER_STAGE:
        print("LENS_STAGE solver", flush=True)
        GUI_STATUS.write(
            state="running",
            message=f"Running exact-solver SVI for {{num_chains}} chain(s).",
            progress={{"fraction": 0.78, "message": f"Running exact-solver SVI for {{num_chains}} chain(s)."}},
        )
        for chain_index, pixelated_state_i in zip(range(num_chains), pixelated_states):
            print(f"LENS_CHAIN solver {{chain_index + 1}} {{num_chains}}", flush=True)
            solver_initial_q_i = ResumeInit.solver_initial_q_from_pixelated(
                pixelated_state_i["median"],
                INITIAL_SOLVER_Q,
            )
            solver_init_values_i = ResumeInit.solver_stage_init_from_pixelated(
                pixelated_state_i["median"],
                prior=MODEL_CONFIG["mass_prior"],
            )
            solver_state_i = Tian_infra.SVI.run_one_chain_svi(
                model_lens,
                data_jax,
                max_iterations=MODEL_CONFIG["svi"]["max_iter_solver"],
                seed=MODEL_CONFIG["svi"]["seed"] + 2000 + chain_index,
                init_values=solver_init_values_i,
                learning_rate=0.01,
                init_scale=0.01,
                loss_kind="trace_meanfield_elbo",
                model_args=("pixelated", k_grid.k, True, solver_initial_q_i),
            )
            solver_kwargs_i = {{
                "kwargs_lens": lens_plane_mass_from_params(
                    solver_state_i["median"], True, solver_initial_q_i
                ),
                "kwargs_lens_light": LENS_LIGHT_EXTERNAL.kwargs_from_params(solver_state_i["median"], "pixelated"),
                "kwargs_point_source": point_source_from_params(solver_state_i["median"]),
                "kwargs_source": [PowerSpectrum.params2kwargs_power_spectrum(
                    solver_state_i["median"],
                    "source_grid",
                    k_grid.k,
                    **PowerSpectrum.matern_prior_kwargs(MODEL_CONFIG["pixelated_prior"]),
                )],
                "psf_noise_fft": (
                    {{"pixels": PSF_CORRECTION.from_params(solver_state_i["median"])}}
                    if MODEL_CONFIG["multiplicative_psf_correction"] else None
                ),
            }}
            solver_states.append(solver_state_i)
            solver_kwargs_list.append(solver_kwargs_i)

        np.save(
            RUN_OUTPUT_DIR / "pixelated_stage2_losses.npy",
            np.asarray([state["losses"] for state in pixelated_states]),
        )
        with open(RUN_OUTPUT_DIR / "pixelated_stage2_states.pkl", "wb") as f:
            pickle.dump(pixelated_states, f)
        # Everything downstream - previews, mass summaries, chain selection and the
        # semilinear inversion - reports the last stage that ran, so the solver results
        # take over the `pixelated_*` names and stage 2 is retained for the loss plot.
        stage2_states = pixelated_states
        pixelated_states = solver_states
        pixelated_kwargs_list = solver_kwargs_list

    __SEMILINEAR_OUTPUT_BLOCK__
    """
    code = (
        textwrap.dedent(code)
        .replace("__MODEL_CONFIG_LITERAL__", model_config_code)
        .replace("__SEMILINEAR_IMPORT__", semilinear_import)
        .replace("__SEMILINEAR_OUTPUT_BLOCK__", semilinear_output_code)
    )
    return code.strip() + "\n", config
