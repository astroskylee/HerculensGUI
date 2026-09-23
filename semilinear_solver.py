"""Single-plane semilinear source inversion for a fixed Herculens model."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.optimize import minimize


DEFAULT_LAMBDA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
ProgressCallback = Callable[[float, str], None]


@dataclass
class SemilinearResult:
    """Numerical products from a fixed-mass single-plane source inversion."""

    source_pixels: np.ndarray
    kwargs: dict
    selected_relative_lambda: float
    selected_actual_lambda: float
    lambda_scan: list[dict]
    metrics: dict

    def summary(self) -> dict:
        return {
            "method": (
                "fixed mass and lens light; direct pixel source with "
                "first-order gradient regularization"
            ),
            "regularization": "gradient",
            "positive": bool(self.metrics["positive"]),
            "source_shape": list(self.source_pixels.shape),
            "selected_relative_lambda": self.selected_relative_lambda,
            "selected_actual_lambda": self.selected_actual_lambda,
            **self.metrics,
        }


def _report(
    callback: ProgressCallback | None,
    fraction: float,
    message: str,
) -> None:
    if callback is not None:
        callback(float(np.clip(fraction, 0.0, 1.0)), message)


def _response_matrix(
    lens_image,
    fixed_kwargs: dict,
    source_shape: tuple[int, int],
    image_shape: tuple[int, int],
    chunk_size: int,
    progress_callback: ProgressCallback | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Differentiate the exact PSF-convolved model with respect to source pixels."""
    n_source = int(np.prod(source_shape))

    def source_model(flat_pixels):
        return lens_image.model(
            kwargs_lens=fixed_kwargs["kwargs_lens"],
            kwargs_lens_light=fixed_kwargs["kwargs_lens_light"],
            kwargs_point_source=fixed_kwargs["kwargs_point_source"],
            kwargs_source=[{"pixels": flat_pixels.reshape(source_shape)}],
            psf_noise_fft=fixed_kwargs.get("psf_noise_fft"),
        ).reshape(-1)

    started = time.perf_counter()
    zero_source = jnp.zeros(n_source, dtype=jnp.float64)
    zero_model, linearized = jax.linearize(source_model, zero_source)
    zero_model.block_until_ready()
    linearized_batch = jax.jit(jax.vmap(linearized))
    response = np.empty((zero_model.size, n_source), dtype=np.float32)
    report_stride = max(chunk_size, int(np.ceil(n_source / 20)))
    next_report = report_stride

    for start in range(0, n_source, chunk_size):
        stop = min(start + chunk_size, n_source)
        basis = np.zeros((stop - start, n_source), dtype=np.float64)
        basis[np.arange(stop - start), np.arange(start, stop)] = 1.0
        block = linearized_batch(jnp.asarray(basis))
        block.block_until_ready()
        response[:, start:stop] = np.asarray(block).T.astype(np.float32)
        if stop >= next_report or stop == n_source:
            _report(
                progress_callback,
                0.2 * stop / n_source,
                f"Building semilinear response matrix {stop}/{n_source}.",
            )
            next_report += report_stride

    return (
        response,
        np.asarray(zero_model, dtype=float).reshape(image_shape),
        float(time.perf_counter() - started),
    )


def _gradient_regularization(
    active_full_indices: np.ndarray,
    source_shape: tuple[int, int],
    column_scales: np.ndarray,
) -> sparse.csr_matrix:
    """First-order horizontal and vertical source-pixel differences."""
    active_position = {
        int(full_index): active_index
        for active_index, full_index in enumerate(active_full_indices)
    }
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    row_index = 0
    ny, nx = source_shape
    for iy in range(ny):
        for ix in range(nx):
            first = iy * nx + ix
            neighbors = []
            if ix + 1 < nx:
                neighbors.append(first + 1)
            if iy + 1 < ny:
                neighbors.append(first + nx)
            for second in neighbors:
                if first not in active_position or second not in active_position:
                    continue
                first_active = active_position[first]
                second_active = active_position[second]
                rows.extend((row_index, row_index))
                columns.extend((first_active, second_active))
                values.extend(
                    (
                        1.0 / column_scales[first_active],
                        -1.0 / column_scales[second_active],
                    )
                )
                row_index += 1
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(row_index, len(active_full_indices)),
    )


def _lcurve_curvature(
    relative_lambdas: np.ndarray,
    data_norms: np.ndarray,
    regularization_norms: np.ndarray,
) -> np.ndarray:
    if relative_lambdas.size < 3:
        return np.zeros_like(relative_lambdas)
    parameter = np.log(relative_lambdas)
    x = np.log(np.maximum(data_norms, 1e-300))
    y = np.log(np.maximum(regularization_norms, 1e-300))
    dx = np.gradient(x, parameter)
    dy = np.gradient(y, parameter)
    ddx = np.gradient(dx, parameter)
    ddy = np.gradient(dy, parameter)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.divide(
        np.abs(dx * ddy - dy * ddx),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    curvature[0] = 0.0
    curvature[-1] = 0.0
    return curvature


def _solve_lambda_grid(
    weighted_response: np.ndarray,
    weighted_data: np.ndarray,
    source_shape: tuple[int, int],
    relative_lambdas: np.ndarray,
    positive: bool,
    max_iterations: int,
    progress_callback: ProgressCallback | None,
) -> tuple[np.ndarray, dict, list[dict]]:
    column_norms = np.linalg.norm(weighted_response, axis=0)
    maximum_norm = float(np.max(column_norms))
    if not np.isfinite(maximum_norm) or maximum_norm <= 0:
        raise ValueError("The source response matrix contains no active columns.")
    active = column_norms > maximum_norm * 1e-10
    active_full_indices = np.flatnonzero(active)
    column_scales = column_norms[active]
    scaled_response = weighted_response[:, active] / column_scales
    regularization = _gradient_regularization(
        active_full_indices,
        source_shape,
        column_scales,
    )
    regularization_diagonal = np.asarray(
        regularization.power(2).sum(axis=0)
    ).reshape(-1)
    positive_diagonal = regularization_diagonal[regularization_diagonal > 0]
    lambda_scale = (
        1.0 / float(np.median(positive_diagonal))
        if positive_diagonal.size
        else 1.0
    )
    response_transpose = scaled_response.T
    warm_start = response_transpose @ weighted_data
    if positive:
        warm_start = np.maximum(warm_start, 0.0)
    solutions: list[np.ndarray] = []
    scan_records: list[dict] = []

    for index, relative_lambda in enumerate(relative_lambdas):
        actual_lambda = float(relative_lambda * lambda_scale)

        def objective_and_gradient(scaled_source):
            data_residual = scaled_response @ scaled_source - weighted_data
            regularization_residual = regularization @ scaled_source
            objective = 0.5 * (
                np.dot(data_residual, data_residual)
                + actual_lambda
                * np.dot(regularization_residual, regularization_residual)
            )
            gradient = (
                response_transpose @ data_residual
                + actual_lambda
                * (regularization.T @ regularization_residual)
            )
            return float(objective), np.asarray(gradient, dtype=float)

        solve_started = time.perf_counter()
        result = minimize(
            objective_and_gradient,
            warm_start,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * warm_start.size if positive else None,
            options={
                "ftol": 1e-12,
                "gtol": 1e-7,
                "maxiter": int(max_iterations),
                "maxls": 50,
            },
        )
        scaled_solution = np.asarray(result.x, dtype=float)
        if positive:
            scaled_solution = np.maximum(scaled_solution, 0.0)
        warm_start = scaled_solution
        physical_solution = scaled_solution / column_scales
        data_residual = (
            weighted_response[:, active] @ physical_solution - weighted_data
        )
        regularization_residual = regularization @ scaled_solution
        raw_gradient = np.asarray(result.jac, dtype=float)
        if positive:
            projected_gradient = np.where(
                scaled_solution > 0.0,
                raw_gradient,
                np.minimum(raw_gradient, 0.0),
            )
        else:
            projected_gradient = raw_gradient
        record = {
            "relative_lambda": float(relative_lambda),
            "actual_lambda": actual_lambda,
            "data_norm": float(np.linalg.norm(data_residual)),
            "chi2": float(np.dot(data_residual, data_residual)),
            "regularization_norm": float(
                np.linalg.norm(regularization_residual)
            ),
            "cost": float(result.fun),
            "iterations": int(result.nit),
            "status": int(result.status),
            "message": str(result.message),
            "optimality": float(
                np.linalg.norm(projected_gradient, ord=np.inf)
            ),
            "solve_seconds": float(time.perf_counter() - solve_started),
        }
        solutions.append(physical_solution)
        scan_records.append(record)
        _report(
            progress_callback,
            0.2 + 0.8 * (index + 1) / len(relative_lambdas),
            (
                f"Semilinear lambda {index + 1}/{len(relative_lambdas)} "
                f"(relative={relative_lambda:g}, chi2={record['chi2']:.4g})."
            ),
        )

    data_norms = np.asarray([record["data_norm"] for record in scan_records])
    regularization_norms = np.asarray(
        [record["regularization_norm"] for record in scan_records]
    )
    curvature = _lcurve_curvature(
        relative_lambdas,
        data_norms,
        regularization_norms,
    )
    if relative_lambdas.size >= 3:
        selected_index = int(np.nanargmax(curvature))
    else:
        selected_index = int(np.nanargmin(data_norms))
    for index, record in enumerate(scan_records):
        record["lcurve_curvature"] = float(curvature[index])
        record["selected"] = index == selected_index

    full_solution = np.zeros(weighted_response.shape[1], dtype=float)
    full_solution[active] = solutions[selected_index]
    metadata = {
        "active_source_pixels": int(np.count_nonzero(active)),
        "inactive_source_pixels": int(active.size - np.count_nonzero(active)),
        "lambda_scale": lambda_scale,
        "selected_relative_lambda": float(relative_lambdas[selected_index]),
        "selected_actual_lambda": float(
            relative_lambdas[selected_index] * lambda_scale
        ),
        "regularization_rows": int(regularization.shape[0]),
    }
    return full_solution, metadata, scan_records


def _fixed_noise_map(
    lens_image,
    fixed_kwargs: dict,
    noise,
    image_shape: tuple[int, int],
) -> np.ndarray:
    noise_array = np.asarray(jax.device_get(noise), dtype=float)
    if noise_array.ndim > 0:
        if noise_array.shape != image_shape:
            raise ValueError(
                f"Noise map shape {noise_array.shape} does not match data "
                f"shape {image_shape}."
            )
        return noise_array

    baseline_model = lens_image.model(**fixed_kwargs)
    model_variance = np.asarray(
        lens_image.Noise.C_D_model(
            baseline_model,
            background_rms=float(noise_array),
        ),
        dtype=float,
    )
    if model_variance.shape != image_shape:
        raise ValueError("Model variance shape does not match the data.")
    if np.any(~np.isfinite(model_variance)) or np.any(model_variance <= 0):
        raise ValueError("Model variance must be finite and strictly positive.")
    return np.sqrt(model_variance)


def semisolve(
    *,
    lens_image,
    fixed_kwargs: dict,
    data,
    noise,
    fit_mask,
    source_shape: int | tuple[int, int],
    regularization: str = "gradient",
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    positive: bool = True,
    response_chunk_size: int = 64,
    max_iterations: int = 1000,
    progress_callback: ProgressCallback | None = None,
) -> SemilinearResult:
    """Solve direct source pixels with fixed mass, lens light, PSF, and geometry."""
    started = time.perf_counter()
    if regularization != "gradient":
        raise ValueError(
            "Only first-order gradient regularization is currently supported."
        )
    if isinstance(source_shape, int):
        source_shape = (source_shape, source_shape)
    source_shape = tuple(int(value) for value in source_shape)
    if len(source_shape) != 2 or min(source_shape) < 1:
        raise ValueError(f"Invalid source shape: {source_shape}")
    relative_lambdas = np.asarray(tuple(lambda_grid), dtype=float)
    if (
        relative_lambdas.size < 1
        or np.any(~np.isfinite(relative_lambdas))
        or np.any(relative_lambdas <= 0)
        or np.any(np.diff(relative_lambdas) <= 0)
    ):
        raise ValueError(
            "lambda_grid must contain strictly increasing positive values."
        )

    data_array = np.asarray(jax.device_get(data), dtype=float)
    fit_mask_array = np.asarray(jax.device_get(fit_mask), dtype=bool)
    if fit_mask_array.shape != data_array.shape:
        raise ValueError("fit_mask shape does not match data.")
    noise_map = _fixed_noise_map(
        lens_image,
        fixed_kwargs,
        noise,
        data_array.shape,
    )
    _report(progress_callback, 0.0, "Building semilinear response matrix.")
    response, lens_light, response_seconds = _response_matrix(
        lens_image,
        fixed_kwargs,
        source_shape,
        data_array.shape,
        max(1, int(response_chunk_size)),
        progress_callback,
    )
    valid = (
        fit_mask_array
        & np.isfinite(data_array)
        & np.isfinite(noise_map)
        & (noise_map > 0)
        & np.isfinite(lens_light)
    )
    if not np.any(valid):
        raise ValueError("No valid image pixels are available for semisolve.")
    weighted_response = (
        response[valid.reshape(-1), :].astype(np.float64)
        / noise_map[valid, None]
    )
    weighted_data = (data_array - lens_light)[valid] / noise_map[valid]
    source_flat, metadata, scan_records = _solve_lambda_grid(
        weighted_response,
        weighted_data,
        source_shape,
        relative_lambdas,
        bool(positive),
        max(1, int(max_iterations)),
        progress_callback,
    )
    source_pixels = source_flat.reshape(source_shape)
    semilinear_kwargs = deepcopy(fixed_kwargs)
    semilinear_kwargs["kwargs_source"] = [
        {"pixels": jnp.asarray(source_pixels)}
    ]
    model = np.asarray(lens_image.model(**semilinear_kwargs), dtype=float)
    residual = (data_array - model) / noise_map
    linear_model = lens_light + (response @ source_flat).reshape(data_array.shape)
    selected_record = next(
        record for record in scan_records if record["selected"]
    )
    metrics = {
        "positive": bool(positive),
        "fit_pixels": int(np.count_nonzero(valid)),
        "active_source_pixels": metadata["active_source_pixels"],
        "inactive_source_pixels": metadata["inactive_source_pixels"],
        "regularization_rows": metadata["regularization_rows"],
        "chi2": float(np.sum(residual[valid] ** 2)),
        "chi2_per_fit_pixel": float(np.mean(residual[valid] ** 2)),
        "residual_std": float(np.std(residual[valid])),
        "response_seconds": response_seconds,
        "solve_seconds": float(
            sum(record["solve_seconds"] for record in scan_records)
        ),
        "total_seconds": float(time.perf_counter() - started),
        "linearity_max_abs": float(np.max(np.abs(linear_model - model))),
        "selected_solver_status": selected_record["status"],
        "selected_solver_message": selected_record["message"],
        "selected_solver_iterations": selected_record["iterations"],
        "selected_solver_optimality": selected_record["optimality"],
    }
    _report(progress_callback, 1.0, "Semilinear source inversion completed.")
    return SemilinearResult(
        source_pixels=source_pixels,
        kwargs=semilinear_kwargs,
        selected_relative_lambda=metadata["selected_relative_lambda"],
        selected_actual_lambda=metadata["selected_actual_lambda"],
        lambda_scan=scan_records,
        metrics=metrics,
    )
