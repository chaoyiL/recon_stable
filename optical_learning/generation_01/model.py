"""Direct fixed-surface background RGB regression."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np
from scipy.special import expit

from .data import FEATURE_ORDER


WIDTHS = (5, 32, 32, 3)


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


@dataclass(frozen=True)
class BackgroundMLP:
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    curvature_mean: np.ndarray
    curvature_scale: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict linear RGB from float material coordinates and curvature in mm^-1."""
        inputs = np.array(features, dtype=np.float32, copy=True)
        inputs[..., 2:] = (inputs[..., 2:] - self.curvature_mean) / self.curvature_scale
        return expit(logits(self.layers, inputs))

    def save(self, path: Path, chart_id: str) -> None:
        arrays = {}
        for index, (weight, bias) in enumerate(self.layers):
            arrays[f'weight_{index}'] = weight
            arrays[f'bias_{index}'] = bias
        np.savez_compressed(path, format_version=1, chart_id=chart_id, feature_order=FEATURE_ORDER,
                            widths=WIDTHS, hidden_activation='relu', output_activation='sigmoid',
                            color_space='linear_rgb', curvature_mean=self.curvature_mean,
                            curvature_scale=self.curvature_scale, **arrays)

    @classmethod
    def load(cls, path: Path) -> 'BackgroundMLP':
        with np.load(path, allow_pickle=False) as data:
            if (int(data['format_version']) != 1 or tuple(data['widths']) != WIDTHS
                    or tuple(data['feature_order']) != FEATURE_ORDER
                    or str(data['hidden_activation']) != 'relu' or str(data['output_activation']) != 'sigmoid'
                    or str(data['color_space']) != 'linear_rgb'):
                raise ValueError('Unsupported background MLP artifact')
            layers = tuple((data[f'weight_{i}'], data[f'bias_{i}']) for i in range(len(WIDTHS) - 1))
            return cls(layers, data['curvature_mean'], data['curvature_scale'])


def fit_background(
    features: np.ndarray,
    mean_rgb: np.ndarray,
    counts: np.ndarray,
    options: TrainingOptions,
) -> BackgroundMLP:
    """Fit MSE over all valid frame samples using per-point sufficient statistics."""
    observed = counts > 0
    inputs = features[observed].copy()
    if not len(inputs):
        raise ValueError('Training requires valid background observations')
    curvature = inputs[:, 2:]
    mean = curvature.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(curvature.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    inputs[:, 2:] = (curvature - mean) / scale
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
    return BackgroundMLP(fitted, mean, scale)
