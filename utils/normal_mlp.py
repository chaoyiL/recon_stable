"""Continuous local-slope calibration with NumPy-only inference.

Inputs are signed linear RGB residuals, optionally frozen material coordinates
and curvature in the same tangent basis as the output slopes. Coordinates and
curvature are never substituted for one another.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


FEATURE_ORDERS = {
    'mlp3': ('delta_r', 'delta_g', 'delta_b'),
    'mlp5': ('delta_r', 'delta_g', 'delta_b', 's', 't'),
    'mlp6': ('delta_r', 'delta_g', 'delta_b', 'k11', 'k12', 'k22'),
    'mlp8': ('delta_r', 'delta_g', 'delta_b', 's', 't', 'k11', 'k12', 'k22'),
}
MODEL_KIND = 'mlp_local_slopes_v1'
NORMAL_CONVENTION = 'normalize(n0-E[p,q])'


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')


def _feature_array(value: np.ndarray | None, shape: tuple[int, ...], width: int,
                   name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f'{name} is required for this model')
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (*shape, width):
        raise ValueError(f'{name} must have shape {(*shape, width)}')
    return result


def assemble_normal_features(
    colors: np.ndarray,
    method: str,
    *,
    st: np.ndarray | None = None,
    curvature_per_mm: np.ndarray | None = None,
    curvature_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return features (..., D) and their validity mask (...).

    The color order is signed linear ΔR, ΔG, ΔB. Curvature is S11, S12,
    S22 in inverse millimetres in the same tangent frame as local slopes.
    M6 and M8 require an explicit curvature validity mask.
    """
    if method not in FEATURE_ORDERS:
        raise ValueError(f'Unknown MLP method: {method}')
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim < 1 or colors.shape[-1] != 3:
        raise ValueError('colors must have shape (..., 3)')
    shape = colors.shape[:-1]
    components = [colors]
    valid = np.isfinite(colors).all(axis=-1)
    if method in ('mlp5', 'mlp8'):
        coordinates = _feature_array(st, shape, 2, 'st')
        components.append(coordinates)
        valid &= np.isfinite(coordinates).all(axis=-1)
    if method in ('mlp6', 'mlp8'):
        curvature = _feature_array(curvature_per_mm, shape, 3, 'curvature_per_mm')
        if curvature_valid_mask is None:
            raise ValueError(f'curvature_valid_mask is required for {method}')
        curvature_valid = np.asarray(curvature_valid_mask, dtype=bool)
        if curvature_valid.shape != shape:
            raise ValueError('curvature_valid_mask shape mismatch')
        components.append(curvature)
        valid &= curvature_valid & np.isfinite(curvature).all(axis=-1)
    features = np.concatenate(components, axis=-1)
    return np.where(valid[..., None], features, 0), valid


@dataclass(frozen=True)
class MLPCalibration:
    """Tanh MLP parameters and training-only support for local p/q slopes."""

    method: str
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    color_min: np.ndarray
    color_max: np.ndarray
    support_features: np.ndarray
    support_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)
    training_info: dict[str, Any] = field(default_factory=dict)
    _support_tree: cKDTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.method not in FEATURE_ORDERS:
            raise ValueError(f'Unknown MLP method: {self.method}')
        dimension = len(FEATURE_ORDERS[self.method])
        widths = (dimension, 32, 32, 32, 2)
        weights = tuple(np.asarray(w, np.float32) for w in self.weights)
        biases = tuple(np.asarray(b, np.float32) for b in self.biases)
        if len(weights) != 4 or len(biases) != 4:
            raise ValueError('MLP requires four weight matrices and biases')
        for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
            if weight.shape != (widths[index], widths[index + 1]) or bias.shape != (widths[index + 1],):
                raise ValueError(f'Invalid MLP layer {index} shape')
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError('MLP weights must be finite')
        mean = np.asarray(self.feature_mean, np.float32)
        scale = np.asarray(self.feature_scale, np.float32)
        lower = np.asarray(self.color_min, np.float32)
        upper = np.asarray(self.color_max, np.float32)
        support = np.asarray(self.support_features, np.float32)
        if mean.shape != (dimension,) or scale.shape != (dimension,):
            raise ValueError('MLP normalization shape mismatch')
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('MLP normalization must be finite with positive scale')
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all() \
                or not np.isfinite(upper).all() or np.any(upper < lower):
            raise ValueError('Invalid training RGB range')
        if support.ndim != 2 or support.shape[1] != dimension or len(support) == 0 \
                or not np.isfinite(support).all():
            raise ValueError('Support features must be a finite nonempty NxD array')
        if not np.isfinite(self.support_threshold) or self.support_threshold < 0:
            raise ValueError('Support threshold must be finite and nonnegative')
        if not isinstance(self.metadata, dict) or not isinstance(self.training_info, dict):
            raise ValueError('MLP metadata and training_info must be mappings')
        metadata = dict(self.metadata)
        if metadata.get('method', self.method) != self.method:
            raise ValueError('MLP metadata method does not match model method')
        metadata['method'] = self.method
        for key, value in dict(weights=weights, biases=biases, feature_mean=mean,
                               feature_scale=scale, color_min=lower, color_max=upper,
                               support_features=support,
                               support_threshold=float(self.support_threshold),
                               metadata=metadata,
                               training_info=dict(self.training_info),
                               _support_tree=cKDTree(support)).items():
            object.__setattr__(self, key, value)

    @property
    def feature_order(self) -> tuple[str, ...]:
        return FEATURE_ORDERS[self.method]

    def save(self, path: str | Path, **metadata: Any) -> Path:
        """Atomically save parameters and declarations without object arrays."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        combined = dict(self.metadata)
        combined.update(metadata)
        if combined.get('method') != self.method:
            raise ValueError('MLP metadata method does not match model method')
        values = dict(
            format_version=np.asarray(1, np.int32), model_kind=np.asarray(MODEL_KIND),
            method=np.asarray(self.method), output_kind=np.asarray('local_slopes'),
            normal_convention=np.asarray(NORMAL_CONVENTION),
            color_residual_mode=np.asarray('signed'), activation=np.asarray('tanh'),
            output_activation=np.asarray('linear'), anchor_zero=np.asarray(True),
            feature_order=np.asarray(self.feature_order), feature_mean=self.feature_mean,
            feature_scale=self.feature_scale, color_min=self.color_min,
            color_max=self.color_max, support_features=self.support_features,
            support_threshold=np.asarray(self.support_threshold, np.float64),
            curvature_units=np.asarray('mm^-1'),
            curvature_basis=np.asarray('local_tangent_e1_e2'),
            curvature_source=np.asarray('current_overall_contact_surface'),
            metadata=np.asarray(json.dumps(combined, ensure_ascii=False, default=_json_default)),
            training_info=np.asarray(json.dumps(self.training_info, ensure_ascii=False,
                                                default=_json_default)),
        )
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            values[f'weight_{index}'] = weight
            values[f'bias_{index}'] = bias
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as stream:
                temporary_path = Path(stream.name)
                np.savez_compressed(stream, **values)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> MLPCalibration:
        """Load and validate a self-describing MLP NPZ without pickle."""
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            expected = dict(model_kind=MODEL_KIND, output_kind='local_slopes',
                            normal_convention=NORMAL_CONVENTION,
                            color_residual_mode='signed', activation='tanh',
                            output_activation='linear', curvature_units='mm^-1',
                            curvature_basis='local_tangent_e1_e2',
                            curvature_source='current_overall_contact_surface')
            if int(data['format_version']) != 1 or any(
                str(data[key]) != value for key, value in expected.items()
            ) or not bool(data['anchor_zero']):
                raise ValueError('Unsupported MLP model declaration')
            method = str(data['method'])
            if method not in FEATURE_ORDERS or tuple(data['feature_order'].tolist()) != FEATURE_ORDERS[method]:
                raise ValueError('MLP feature order does not match its method')
            return cls(
                method=method,
                weights=tuple(data[f'weight_{index}'] for index in range(4)),
                biases=tuple(data[f'bias_{index}'] for index in range(4)),
                feature_mean=data['feature_mean'], feature_scale=data['feature_scale'],
                color_min=data['color_min'], color_max=data['color_max'],
                support_features=data['support_features'],
                support_threshold=float(data['support_threshold']),
                metadata=json.loads(str(data['metadata'])),
                training_info=json.loads(str(data['training_info'])),
            )


def predict_mlp(
    colors: np.ndarray,
    model: MLPCalibration,
    valid_mask: np.ndarray | None = None,
    *,
    st: np.ndarray | None = None,
    curvature_per_mm: np.ndarray | None = None,
    curvature_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw slopes and empirical support; never clip a finite input."""
    features, valid = assemble_normal_features(
        colors, model.method, st=st, curvature_per_mm=curvature_per_mm,
        curvature_valid_mask=curvature_valid_mask,
    )
    shape = features.shape[:-1]
    if valid_mask is not None:
        requested = np.asarray(valid_mask, dtype=bool)
        if requested.shape != shape:
            raise ValueError('valid_mask shape mismatch')
        valid &= requested
    flat = features.reshape(-1, features.shape[-1])
    active = valid.reshape(-1)
    slopes = np.zeros((len(flat), 2), np.float32)
    supported = np.zeros(len(flat), dtype=bool)
    indices = np.flatnonzero(active)
    source = flat[indices]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = (source - model.feature_mean) / model.feature_scale
        zero = normalized.copy()
        zero[:, :3] = -model.feature_mean[:3] / model.feature_scale[:3]
    computable = np.isfinite(normalized).all(axis=1) & np.isfinite(zero).all(axis=1)
    source, normalized, zero, indices = (
        source[computable], normalized[computable], zero[computable], indices[computable]
    )
    if len(indices) == 0:
        return slopes.reshape(*shape, 2), supported.reshape(shape)

    def evaluate(values):
        for weight, bias in zip(model.weights[:-1], model.biases[:-1], strict=True):
            values = np.tanh(values @ weight + bias)
        return values @ model.weights[-1] + model.biases[-1]

    with np.errstate(over='ignore', invalid='ignore'):
        prediction = evaluate(normalized) - evaluate(zero)
    finite = np.isfinite(prediction).all(axis=1)
    source, normalized, indices, prediction = (
        source[finite], normalized[finite], indices[finite], prediction[finite]
    )
    if len(indices) == 0:
        return slopes.reshape(*shape, 2), supported.reshape(shape)
    slopes[indices] = prediction
    in_range = np.all(
        (source[:, :3] >= model.color_min) & (source[:, :3] <= model.color_max), axis=1
    )
    distance = model._support_tree.query(normalized, k=1)[0]
    if model.method in ('mlp5', 'mlp8'):
        in_range &= np.all((source[:, 3:5] >= 0) & (source[:, 3:5] <= 1), axis=1)
    supported[indices] = in_range & (distance <= model.support_threshold)
    return slopes.reshape(*shape, 2), supported.reshape(shape)
