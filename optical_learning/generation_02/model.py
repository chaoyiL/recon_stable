"""Fixed-surface RGB regression with Fourier features of material coordinates."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np
from scipy.special import expit

from .data import FEATURE_ORDER


FOURIER_FREQUENCIES = (1, 2, 4, 8)  # Cycles across a normalized material coordinate.
WIDTHS = (len(FEATURE_ORDER) + 4 * len(FOURIER_FREQUENCIES), 32, 32, 3)
POSITION_ENCODING = 'raw_st_and_fourier_sin_cos'


@dataclass(frozen=True)
class TrainingOptions:
    seed: int = 42
    epochs: int = 200
    batch_size: int = 1024
    learning_rate: float = 0.001


def logits(layers, features):
    values = features
    for weight, bias in layers[:-1]:
        values = (values @ weight + bias).clip(0)
    weight, bias = layers[-1]
    return values @ weight + bias


def encode_inputs(
    features: np.ndarray, curvature_mean: np.ndarray, curvature_scale: np.ndarray,
) -> np.ndarray:
    """Expand (..., 5) raw inputs into st, per-axis sin/cos bands, and scaled curvature."""
    raw = np.asarray(features, dtype=np.float32)
    if raw.ndim < 1 or raw.shape[-1] != len(FEATURE_ORDER):
        raise ValueError('Features must end in s, t, k11, k12, k22')
    st = raw[..., :2]
    angular_frequencies = np.float32(2 * np.pi) * np.asarray(FOURIER_FREQUENCIES, dtype=np.float32)
    phase = st[..., None] * angular_frequencies
    bands = np.concatenate((np.sin(phase), np.cos(phase)), axis=-1)
    bands = bands.reshape(st.shape[:-1] + (4 * len(FOURIER_FREQUENCIES),))
    curvature = (raw[..., 2:] - curvature_mean) / curvature_scale
    return np.concatenate((st, bands, curvature), axis=-1)


@dataclass(frozen=True)
class BackgroundMLP:
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    curvature_mean: np.ndarray
    curvature_scale: np.ndarray
    curvature_smoothing_half_width_mm: float = 0.0

    def predict(self, features: np.ndarray, *, curvature_smoothing_half_width_mm: float = 0.0) -> np.ndarray:
        """Predict linear RGB from float material coordinates and curvature in mm^-1."""
        if curvature_smoothing_half_width_mm != self.curvature_smoothing_half_width_mm:
            raise ValueError('Input curvature smoothing half-width differs from training')
        inputs = encode_inputs(features, self.curvature_mean, self.curvature_scale)
        return expit(logits(self.layers, inputs))

    def save(self, path: Path, chart_id: str) -> None:
        arrays = {}
        for index, (weight, bias) in enumerate(self.layers):
            arrays[f'weight_{index}'] = weight
            arrays[f'bias_{index}'] = bias
        np.savez_compressed(path, format_version=1, chart_id=chart_id, feature_order=FEATURE_ORDER,
                            widths=WIDTHS, hidden_activation='relu', output_activation='sigmoid',
                            position_encoding=POSITION_ENCODING, fourier_frequencies=FOURIER_FREQUENCIES,
                            color_space='linear_rgb', curvature_mean=self.curvature_mean,
                            curvature_scale=self.curvature_scale,
                            curvature_smoothing_half_width_mm=self.curvature_smoothing_half_width_mm,
                            **arrays)

    @classmethod
    def load(cls, path: Path) -> 'BackgroundMLP':
        with np.load(path, allow_pickle=False) as data:
            if (int(data['format_version']) != 1 or tuple(data['widths']) != WIDTHS
                    or str(data['position_encoding']) != POSITION_ENCODING
                    or tuple(data['fourier_frequencies']) != FOURIER_FREQUENCIES
                    or tuple(data['feature_order']) != FEATURE_ORDER
                    or str(data['hidden_activation']) != 'relu' or str(data['output_activation']) != 'sigmoid'
                    or str(data['color_space']) != 'linear_rgb'):
                raise ValueError('Unsupported background MLP artifact')
            layers = tuple((data[f'weight_{i}'], data[f'bias_{i}']) for i in range(len(WIDTHS) - 1))
            half_width = float(data.get('curvature_smoothing_half_width_mm', 0.0))
            if not np.isfinite(half_width) or half_width < 0:
                raise ValueError('Invalid curvature smoothing half-width in model')
            return cls(layers, data['curvature_mean'], data['curvature_scale'], half_width)


def fit_background(
    features: np.ndarray,
    mean_rgb: np.ndarray,
    counts: np.ndarray,
    options: TrainingOptions,
    *,
    curvature_smoothing_half_width_mm: float = 0.0,
) -> BackgroundMLP:
    """Fit MSE over all valid frame samples using per-point sufficient statistics."""
    observed = counts > 0
    raw = features[observed]
    if not len(raw):
        raise ValueError('Training requires valid background observations')
    curvature = raw[:, 2:]
    mean = curvature.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(curvature.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    inputs = encode_inputs(raw, mean, scale)
    targets = mean_rgb[observed]
    sample_weights = counts[observed].astype(np.float32)
    # Weight by observation count, which preserves the original per-sample MSE gradients.
    sample_weights /= sample_weights.mean()

    rng = np.random.default_rng(options.seed)
    layers = []
    for n_in, n_out in zip(WIDTHS[:-1], WIDTHS[1:]):
        limit = np.sqrt(6.0 / (n_in + n_out))
        layers.append((rng.uniform(-limit, limit, (n_in, n_out)).astype(np.float32),
                       np.zeros(n_out, dtype=np.float32)))
    init, update, parameters = optimizers.adam(options.learning_rate)
    state = init(layers)

    def loss(current, x, y, weight):
        rgb = jax.nn.sigmoid(logits(current, x))
        return jnp.mean(weight * jnp.mean((rgb - y) ** 2, axis=-1))

    @jax.jit
    def step(iteration, current, x, y, weight):
        gradient = jax.grad(loss)(parameters(current), x, y, weight)
        return update(iteration, gradient, current)

    iteration = 0
    for _ in range(options.epochs):
        order = rng.permutation(len(inputs))
        for start in range(0, len(order), options.batch_size):
            selected = order[start:start + options.batch_size]
            state = step(iteration, state, inputs[selected], targets[selected], sample_weights[selected])
            iteration += 1

    fitted = tuple((np.asarray(weight), np.asarray(bias)) for weight, bias in parameters(state))
    if not all(np.isfinite(array).all() for layer in fitted for array in layer):
        raise ValueError('Background training produced non-finite parameters')
    return BackgroundMLP(fitted, mean, scale, curvature_smoothing_half_width_mm)
