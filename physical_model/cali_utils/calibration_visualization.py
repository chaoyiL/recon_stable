"""Small diagnostic plots generated from saved calibration samples."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.normal_mapping import lookup_slopes_raw


def _pyplot():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def _angle_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    first = np.concatenate((-predicted, np.ones((len(predicted), 1))), axis=-1)
    second = np.concatenate((-truth, np.ones((len(truth), 1))), axis=-1)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-15)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-15)
    return np.degrees(np.arccos(np.clip(np.sum(first * second, axis=-1), -1, 1)))


def _lookup_chunked(colors: np.ndarray, lut, chunk_size: int = 500_000) -> np.ndarray:
    result = []
    for start in range(0, len(colors), chunk_size):
        result.append(lookup_slopes_raw(colors[start:start + chunk_size], lut)[0])
    return np.concatenate(result) if result else np.empty((0, 2), np.float32)


def generate_calibration_visualizations(samples: dict, lut, validation: dict,
                                        output_dir: Path) -> Path:
    """Write a fit/error summary and RGB LUT slices without re-rendering."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(samples['color_residual_samples'], np.float32)
    truth = np.asarray(samples['slope_samples'], np.float32)
    predicted = _lookup_chunked(colors, lut)
    angles = _angle_errors(predicted, truth)
    plt = _pyplot()

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for component, axis in enumerate(axes[0]):
        axis.scatter(truth[:, component], predicted[:, component], s=3, alpha=.2,
                     rasterized=True)
        low = min(float(np.min(truth[:, component])), float(np.min(predicted[:, component])))
        high = max(float(np.max(truth[:, component])), float(np.max(predicted[:, component])))
        axis.plot((low, high), (low, high), 'k--', linewidth=1)
        axis.set(xlabel=f'geometry {"p" if component == 0 else "q"}',
                 ylabel='fitted LUT', title='Full-sample fit')
        axis.set_aspect('equal', adjustable='datalim')
    axes[1, 0].hist(angles, bins=50, color='#4477aa')
    axes[1, 0].set(xlabel='normal angular error (degrees)', ylabel='samples',
                   title='Full-sample fit error')
    rows = validation.get('images', [])
    labels = [row['image_key'] for row in rows]
    axes[1, 1].bar(labels, [row.get('rms_degrees', 0.) or 0. for row in rows],
                   color=['#55a868' if row.get('passed') else '#c44e52' for row in rows])
    axes[1, 1].axhline(validation['thresholds']['rms_degrees'], color='black', linestyle='--')
    axes[1, 1].set(title=f"Held-out poses: {validation['status']}",
                   xlabel='pose key', ylabel='RMS angular error (degrees)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    figure.tight_layout()
    figure.savefig(output_dir / 'calibration_summary.png', dpi=140, bbox_inches='tight')
    plt.close(figure)

    names = ('R', 'G', 'B')
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    bound = max(float(np.max(np.abs(lut.slopes))), 1e-6)
    for channel in range(3):
        remaining = [value for value in range(3) if value != channel]
        first = np.linspace(lut.color_min[remaining[0]], lut.color_max[remaining[0]], lut.size)
        second = np.linspace(lut.color_min[remaining[1]], lut.color_max[remaining[1]], lut.size)
        a, b = np.meshgrid(first, second)
        query = np.zeros((*a.shape, 3), np.float32)
        query[..., remaining[0]] = a
        query[..., remaining[1]] = b
        values = lookup_slopes_raw(query, lut)[0]
        for component in range(2):
            image = axes[component, channel].imshow(
                values[..., component].T, origin='lower', cmap='coolwarm',
                vmin=-bound, vmax=bound, aspect='auto')
            axes[component, channel].set(
                title=f'{"p" if component == 0 else "q"}: delta {names[channel]}=0',
                xlabel=f'delta {names[remaining[0]]} bin',
                ylabel=f'delta {names[remaining[1]]} bin')
            figure.colorbar(image, ax=axes[component, channel], shrink=.75)
    figure.tight_layout()
    figure.savefig(output_dir / 'lut_slices.png', dpi=140, bbox_inches='tight')
    plt.close(figure)
    return output_dir
