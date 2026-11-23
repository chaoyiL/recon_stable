"""Train local-slope MLPs from image-grouped calibration samples."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from utils.normal_mlp import FEATURE_ORDERS, MLPCalibration, assemble_normal_features


DEFAULT_OPTIONS: dict[str, Any] = {
    'seed': 42,
    'epochs': 200,
    'patience': 20,
    'learning_rate': 1e-3,
    'batch_size': 1024,
    'validation_fraction': 0.2,
    'min_delta_degrees': 0.0,
    'support_quantile': 0.99,
}


def _options(options: dict[str, Any] | None) -> dict[str, Any]:
    if options is not None and not isinstance(options, dict):
        raise ValueError('MLP training options must be a mapping')
    result = dict(DEFAULT_OPTIONS)
    unknown = set(options or {}) - set(result)
    if unknown:
        raise ValueError(f'Unknown MLP training options: {sorted(unknown)}')
    result.update(options or {})
    for name in ('seed', 'epochs', 'patience', 'batch_size'):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        result[name] = int(value)
    if result['seed'] < 0 or min(result['epochs'], result['patience'], result['batch_size']) < 1:
        raise ValueError('seed must be nonnegative and training counts positive')
    for name in ('learning_rate', 'validation_fraction', 'min_delta_degrees', 'support_quantile'):
        result[name] = float(result[name])
        if not np.isfinite(result[name]):
            raise ValueError(f'{name} must be finite')
    if result['learning_rate'] <= 0 or result['min_delta_degrees'] < 0 \
            or not 0 < result['validation_fraction'] < 1 \
            or not 0 < result['support_quantile'] <= 1:
        raise ValueError('Invalid MLP training option')
    return result


def _normalization(features: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(features.shape[1], np.float32)
    scale = np.ones(features.shape[1], np.float32)
    mean[:3] = np.mean(features[:, :3], axis=0, dtype=np.float64)
    scale[:3] = np.maximum(np.std(features[:, :3], axis=0, dtype=np.float64), 1e-6)
    if method in ('mlp5', 'mlp8'):
        mean[3:5] = 0.5
        scale[3:5] = 0.5
    if method in ('mlp6', 'mlp8'):
        mean[-3:] = np.mean(features[:, -3:], axis=0, dtype=np.float64)
        scale[-3:] = np.maximum(np.std(features[:, -3:], axis=0, dtype=np.float64), 1e-6)
    return mean, scale


def _initialize(dimension: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    widths = (dimension, 32, 32, 32, 2)
    layers = []
    for n_in, n_out in zip(widths[:-1], widths[1:], strict=True):
        limit = np.sqrt(6.0 / (n_in + n_out))
        weights = rng.uniform(-limit, limit, (n_in, n_out)).astype(np.float32)
        layers.append((weights, np.zeros(n_out, np.float32)))
    return layers


def _predict_numpy(params, features: np.ndarray, zero_rgb: np.ndarray) -> np.ndarray:
    zero = features.copy()
    zero[:, :3] = zero_rgb
    def network(values):
        for weights, bias in params[:-1]:
            values = np.tanh(values @ weights + bias)
        weights, bias = params[-1]
        return values @ weights + bias
    return network(features) - network(zero)


def _normal_angles(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, np.float64)
    target = np.asarray(target, np.float64)
    dot = 1 + np.sum(predicted * target, axis=1)
    denominator = np.sqrt(1 + np.sum(predicted * predicted, axis=1)) * np.sqrt(
        1 + np.sum(target * target, axis=1)
    )
    return np.degrees(np.arccos(np.clip(dot / denominator, -1, 1)))


def _image_rms(predicted: np.ndarray, target: np.ndarray, groups: np.ndarray) -> float:
    angles = _normal_angles(predicted, target)
    return float(np.mean([
        np.sqrt(np.mean(angles[groups == group] ** 2)) for group in np.unique(groups)
    ]))


def _fit(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    zero_rgb: np.ndarray,
    options: dict[str, Any],
    epochs: int,
    *,
    development: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, float]], int]:
    """Fit the image-balanced cosine loss; optionally select an epoch by image."""
    import jax
    import jax.numpy as jnp
    from jax.example_libraries import optimizers

    rng = np.random.default_rng(options['seed'])
    params = _initialize(features.shape[1], options['seed'])
    init, update, parameters = optimizers.adam(options['learning_rate'])
    state = init(params)
    zero_rgb_jax = jnp.asarray(zero_rgb)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    sample_weights = (len(features) / (len(counts) * counts[inverse])).astype(np.float32)

    def network(layers, values):
        for weights, bias in layers[:-1]:
            values = jnp.tanh(values @ weights + bias)
        weights, bias = layers[-1]
        return values @ weights + bias

    def loss(layers, x, y, weight):
        zero = x.at[:, :3].set(zero_rgb_jax)
        prediction = network(layers, x) - network(layers, zero)
        dot = 1 + jnp.sum(prediction * y, axis=1)
        norm = jnp.sqrt(1 + jnp.sum(prediction ** 2, axis=1)) * jnp.sqrt(
            1 + jnp.sum(y ** 2, axis=1)
        )
        return jnp.mean(weight * (1 - dot / norm))

    @jax.jit
    def step(iteration, current, x, y, weight):
        gradients = jax.grad(loss)(parameters(current), x, y, weight)
        return update(iteration, gradients, current)

    best_rms = float('inf')
    best_params = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    step_index = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(features))
        for start in range(0, len(order), options['batch_size']):
            selected = order[start:start + options['batch_size']]
            state = step(step_index, state, features[selected], targets[selected], sample_weights[selected])
            step_index += 1
        current = [(np.asarray(w), np.asarray(b)) for w, b in parameters(state)]
        prediction = _predict_numpy(current, features, zero_rgb)
        train_rms = _image_rms(prediction, targets, groups)
        if not np.isfinite(train_rms):
            raise ValueError('MLP training produced non-finite values')
        row = dict(epoch=epoch, train_image_rms_degrees=train_rms)
        if development is None:
            history.append(row)
            continue
        dev_features, dev_targets, dev_groups = development
        dev_prediction = _predict_numpy(current, dev_features, zero_rgb)
        dev_rms = _image_rms(dev_prediction, dev_targets, dev_groups)
        if not np.isfinite(dev_rms):
            raise ValueError('MLP development evaluation produced non-finite values')
        row['development_image_rms_degrees'] = dev_rms
        history.append(row)
        if dev_rms < best_rms - options['min_delta_degrees']:
            best_rms = dev_rms
            best_epoch = epoch
            best_params = [(w.copy(), b.copy()) for w, b in current]
        elif epoch - best_epoch >= options['patience']:
            break
    if development is None:
        return current, history, epochs
    if best_params is None:
        raise ValueError('MLP development did not produce a valid model')
    return best_params, history, best_epoch


def _support_threshold(features: np.ndarray, groups: np.ndarray, quantile: float) -> float:
    """Determine feature distance from other training images, not the same image."""
    distances = np.empty(len(features), np.float64)
    for group in np.unique(groups):
        selected = groups == group
        distances[selected] = cKDTree(features[~selected]).query(features[selected], k=1)[0]
    return float(np.quantile(distances, quantile))


def train_normal_mlp(
    colors: np.ndarray,
    slopes: np.ndarray,
    image_index: np.ndarray,
    *,
    method: str,
    st: np.ndarray | None = None,
    curvature_per_mm: np.ndarray | None = None,
    curvature_valid_mask: np.ndarray | None = None,
    options: dict[str, Any] | None = None,
) -> MLPCalibration:
    """Fit a local-slope MLP using only the supplied training images.

    Each image is equally weighted. A held-out subset of these images selects
    the epoch; a fresh model is then fitted on all supplied valid images for
    that number of epochs. The caller must keep its external test images out
    of these arrays. Invalid feature/label points are excluded and counted.
    """
    if method not in FEATURE_ORDERS:
        raise ValueError(f'Unknown MLP method: {method}')
    settings = _options(options)
    features, valid = assemble_normal_features(
        colors, method, st=st, curvature_per_mm=curvature_per_mm,
        curvature_valid_mask=curvature_valid_mask,
    )
    if features.ndim != 2:
        raise ValueError('Training colors must have shape (N, 3)')
    targets = np.asarray(slopes, np.float32)
    groups = np.asarray(image_index)
    if targets.shape != (len(features), 2) or groups.shape != (len(features),):
        raise ValueError('slopes and image_index must match training colors')
    if groups.dtype.kind not in 'iuUS':
        raise ValueError('image_index must contain integer or string image identifiers')
    if groups.dtype.kind == 'S':
        groups = groups.astype('U')
    if method in ('mlp5', 'mlp8'):
        valid &= np.all((features[:, 3:5] >= 0) & (features[:, 3:5] <= 1), axis=1)
    valid &= np.isfinite(targets).all(axis=1)
    excluded = int(np.count_nonzero(~valid))
    features, targets, groups = features[valid], targets[valid], groups[valid]
    image_ids = np.unique(groups)
    if len(image_ids) < 2:
        raise ValueError('MLP training requires at least two valid image groups')

    rng = np.random.default_rng(settings['seed'])
    shuffled = rng.permutation(image_ids)
    development_count = min(len(image_ids) - 1, max(1, math.ceil(
        settings['validation_fraction'] * len(image_ids)
    )))
    development_ids = shuffled[:development_count]
    development_mask = np.isin(groups, development_ids)
    inner_train = ~development_mask
    inner_mean, inner_scale = _normalization(features[inner_train], method)
    normalized = (features - inner_mean) / inner_scale
    zero_rgb = -inner_mean[:3] / inner_scale[:3]
    _, development_history, best_epoch = _fit(
        normalized[inner_train], targets[inner_train], groups[inner_train], zero_rgb,
        settings, settings['epochs'], development=(
            normalized[development_mask], targets[development_mask], groups[development_mask]
        ),
    )

    mean, scale = _normalization(features, method)
    normalized = (features - mean) / scale
    zero_rgb = -mean[:3] / scale[:3]
    params, refit_history, _ = _fit(
        normalized, targets, groups, zero_rgb, settings, best_epoch
    )
    threshold = _support_threshold(normalized, groups, settings['support_quantile'])
    info = dict(
        options=settings,
        optimizer='jax.example_libraries.optimizers.adam',
        loss='image_balanced_cosine_normal',
        zero_anchor=True,
        sample_count=int(len(features)), excluded_invalid_count=excluded,
        training_images=image_ids.tolist(),
        inner_training_images=shuffled[development_count:].tolist(),
        development_images=development_ids.tolist(),
        best_epoch=best_epoch,
        development_history=development_history,
        refit_history=refit_history,
        support_definition='normalized_full_features_nearest_other_image',
        support_quantile=settings['support_quantile'],
        support_threshold=threshold,
    )
    return MLPCalibration(
        method=method,
        weights=tuple(weights for weights, _ in params),
        biases=tuple(bias for _, bias in params),
        feature_mean=mean, feature_scale=scale,
        color_min=np.minimum(np.min(features[:, :3], axis=0), 0),
        color_max=np.maximum(np.max(features[:, :3], axis=0), 0),
        support_features=normalized.astype(np.float32), support_threshold=threshold,
        metadata=dict(method=method, validation_status='training_only'), training_info=info,
    )
