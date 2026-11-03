"""Pose configuration and reference-area metrics for compensation experiments."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from physical_model.compensation_utils.photometry import reference_geometry


def pose_config(base_config: dict[str, Any], radius_mm: float | None,
                pose: dict[str, Any] | None = None) -> dict[str, Any]:
    """Copy a physical snapshot and select curvature plus sphere pose."""
    config = copy.deepcopy(base_config)
    config['surface'].setdefault('bending', {'axis': 'x', 'direction_z': -1})
    config['surface']['bending']['radius_mm'] = radius_mm
    if pose is None:
        config['indentation'].update(center_uv=[0.0, 0.0], depth_mm=0.0)
    else:
        config['indentation'].update(center_uv=list(pose['center_uv']),
                                     depth_mm=float(pose['depth_mm']))
    return config


def grid_axis_weights(axis: np.ndarray) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() \
            or np.any(np.diff(values) <= 0.0):
        raise ValueError('grid axes must be finite, increasing, and have at least two entries')
    edges = np.r_[values[0], (values[1:] + values[:-1]) / 2.0, values[-1]]
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError('grid axis has nonpositive reference area weights')
    return widths


def reference_area_weights(x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """Area weights for the arclength chart, in square millimetres."""
    return np.outer(grid_axis_weights(y_axis), grid_axis_weights(x_axis))


def lut_support(color_delta: np.ndarray, lut: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return per-color in-range and original-node-support flags."""
    color = np.asarray(color_delta, dtype=np.float64)
    if color.ndim != 2 or color.shape[1] != 3 or not np.isfinite(color).all():
        raise ValueError('color_delta must be finite with shape (N,3)')
    inside = np.all((color >= lut.color_min) & (color <= lut.color_max), axis=-1)
    supported = np.ones(len(color), dtype=bool)
    if lut.original_valid is None:
        supported[:] = False
        return inside, supported
    span = lut.color_max - lut.color_min
    coordinate = np.clip((color - lut.color_min) / span, 0.0, 1.0) * (lut.size - 1)
    lower = np.floor(coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, lut.size - 1)
    fraction = coordinate - lower
    for red in (0, 1):
        ir = upper[:, 0] if red else lower[:, 0]
        wr = fraction[:, 0] if red else 1.0 - fraction[:, 0]
        for green in (0, 1):
            ig = upper[:, 1] if green else lower[:, 1]
            wg = fraction[:, 1] if green else 1.0 - fraction[:, 1]
            for blue in (0, 1):
                ib = upper[:, 2] if blue else lower[:, 2]
                wb = fraction[:, 2] if blue else 1.0 - fraction[:, 2]
                weight = wr * wg * wb
                node_valid = np.asarray(lut.original_valid)[ir, ig, ib]
                supported &= (weight <= 0.0) | node_valid
    return inside, supported


def pose_metric(beta_prediction: np.ndarray, beta_truth: np.ndarray,
                area_mm2: np.ndarray, tolerances: list[float]) -> dict[str, Any]:
    """Area-weighted local-slope vector metrics for one pose."""
    prediction = np.asarray(beta_prediction, dtype=np.float64)
    truth = np.asarray(beta_truth, dtype=np.float64)
    area = np.asarray(area_mm2, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 2 \
            or area.shape != (len(prediction),) or np.any(area <= 0.0):
        raise ValueError('slope predictions, labels, and positive area weights disagree')
    weight = area / np.sum(area)
    error = prediction - truth
    error_norm = np.linalg.norm(error, axis=-1)
    sorted_index = np.argsort(error_norm, kind='stable')
    cumulative = np.cumsum(weight[sorted_index])
    p95_index = min(np.searchsorted(cumulative, 0.95, side='left'), len(error_norm) - 1)
    rmse = float(np.sqrt(np.sum(weight * error_norm ** 2)))
    result: dict[str, Any] = {
        'slope_rmse': rmse,
        'slope_component_rmse': np.sqrt(np.sum(weight[:, None] * error ** 2, axis=0)).tolist(),
        'slope_bias': np.sum(weight[:, None] * error, axis=0).tolist(),
        'slope_error_p95': float(error_norm[sorted_index[p95_index]]),
        'slope_mean_error': float(np.sum(weight * error_norm)),
    }
    for tolerance in tolerances:
        result[f'within_{format(float(tolerance), ".12g")}_fraction'] = float(
            np.sum(weight * (error_norm <= float(tolerance))))
    return result


def depth_rmse(depth_prediction: np.ndarray, depth_truth: np.ndarray,
               area_mm2: np.ndarray) -> float:
    prediction = np.asarray(depth_prediction, dtype=np.float64)
    truth = np.asarray(depth_truth, dtype=np.float64)
    area = np.asarray(area_mm2, dtype=np.float64)
    weight = area / np.sum(area)
    return float(np.sqrt(np.sum(weight * (prediction - truth) ** 2)))
