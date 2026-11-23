"""Unified lookup for legacy normal LUTs and continuous slope models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.normal_mapping import NormalCalibration, lookup_slopes_raw
from utils.normal_mlp import MLPCalibration, MODEL_KIND, predict_mlp


@dataclass(frozen=True)
class SlopePrediction:
    slopes: np.ndarray
    variance: np.ndarray | None
    confidence: np.ndarray | None
    supported_mask: np.ndarray


def load_normal_model(path: str | Path) -> NormalCalibration | MLPCalibration:
    """Dispatch validated NPZ models without loading pickled objects."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        kind = str(data['model_kind']) if 'model_kind' in data else 'table'
    if kind == MODEL_KIND:
        return MLPCalibration.load(path)
    if kind == 'table':
        return NormalCalibration.load(path)
    raise ValueError(f'Unsupported normal model kind: {kind}')


def predict_normal_model(
    colors: np.ndarray,
    model: NormalCalibration | MLPCalibration,
    valid_mask: np.ndarray | None = None,
    *,
    st: np.ndarray | None = None,
    curvature_per_mm: np.ndarray | None = None,
    curvature_valid_mask: np.ndarray | None = None,
) -> SlopePrediction:
    """Predict raw slopes, preserving legacy table interpolation behavior."""
    if isinstance(model, NormalCalibration):
        slopes, variance, confidence = lookup_slopes_raw(colors, model, valid_mask)
        supported = np.isfinite(np.asarray(colors)).all(axis=-1)
        if valid_mask is not None:
            supported &= np.asarray(valid_mask, dtype=bool)
        return SlopePrediction(slopes, variance, confidence, supported)
    if isinstance(model, MLPCalibration):
        slopes, supported = predict_mlp(
            colors, model, valid_mask, st=st, curvature_per_mm=curvature_per_mm,
            curvature_valid_mask=curvature_valid_mask,
        )
        return SlopePrediction(slopes, None, None, supported)
    raise TypeError('Expected a normal LUT or MLP calibration')
