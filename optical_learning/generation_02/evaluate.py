"""Shared-mask comparisons and material-grid diagnostics for generation 1."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


@dataclass(frozen=True)
class BackgroundStatistics:
    counts: np.ndarray
    mean: np.ndarray
    standard_deviation: np.ndarray


def background_statistics(rgb: np.ndarray, valid: np.ndarray) -> BackgroundStatistics:
    counts = valid.sum(axis=0)
    colors = np.where(valid[..., None], rgb, 0).astype(np.float64)
    mean = colors.sum(axis=0) / np.maximum(counts[..., None], 1)
    centered = np.where(valid[..., None], colors - mean, 0)
    variance = (centered ** 2).sum(axis=0) / np.maximum(counts[..., None] - 1, 1)
    std = np.where(counts[..., None] > 1, np.sqrt(variance), np.nan)
    return BackgroundStatistics(counts, mean.astype(np.float32), std.astype(np.float32))


def color_metrics(error: np.ndarray) -> dict:
    if error.size == 0:
        return {'n': 0}
    error = error.astype(np.float64)
    magnitude = np.abs(error)
    mse = np.mean(error ** 2, axis=0)
    return dict(n=len(error), mae=magnitude.mean(axis=0).tolist(), rmse=np.sqrt(mse).tolist(),
                p95=np.quantile(magnitude, .95, axis=0).tolist(), bias=error.mean(axis=0).tolist(),
                rgb_rmse=float(np.sqrt(mse.mean())))


def count_metrics(statistics: BackgroundStatistics, valid: np.ndarray) -> dict:
    counts = statistics.counts[valid]
    repeated = statistics.standard_deviation[valid & (statistics.counts > 1)]
    result = dict(covered=int(np.count_nonzero(counts)), domain=len(counts),
                  min_median_max=[int(counts.min()), float(np.median(counts)), int(counts.max())],
                  repeatable_points=len(repeated))
    if len(repeated):
        result['repeat_std_rms'] = np.sqrt(np.mean(repeated.astype(np.float64)**2, axis=0)).tolist()
    return result


def save_comparison(path: Path, arrays: dict[str, np.ndarray]) -> None:
    ref_valid = arrays['reference_valid']
    domain = arrays['geometry_valid']
    counts = arrays['all_counts']

    def orient(values):
        # Material t runs along the image width, while increasing s points upward.
        return np.swapaxes(values, 0, 1)[::-1]

    def preview(rgb, valid):
        srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1 / 2.4) - 0.055)
        return orient(np.where(valid[..., None], srgb, .13))

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.3), layout='constrained')
    for ax, title, rgb, valid in (
        (axes[0, 0], 'Measured reference', arrays['reference_rgb'], ref_valid),
        (axes[0, 1], 'MLP prediction', arrays['prediction'], domain),
        (axes[0, 2], 'Per-point RGB mean', arrays['mean_rgb'], domain & (counts > 0)),
    ):
        ax.imshow(preview(rgb, valid), extent=(0, 1, 0, 1), aspect='auto')
        ax.set_title(title)

    error = np.sqrt(np.mean((arrays['prediction'] - arrays['reference_rgb'])**2, axis=-1))
    repeat_std = np.sqrt(np.mean(arrays['repeat_std']**2, axis=-1))
    for ax, title, values, valid, color_map in (
        (axes[1, 0], 'MLP vs. reference RGB RMSE', error, ref_valid, 'magma'),
        (axes[1, 1], 'Non-contact observations per point', counts, domain, 'viridis'),
        (axes[1, 2], 'Across-frame RGB standard deviation', repeat_std, domain & (counts > 1), 'magma'),
    ):
        image = ax.imshow(orient(np.where(valid, values, np.nan)), extent=(0, 1, 0, 1),
                          aspect='auto', cmap=color_map, vmin=0)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=.82)
    for ax in axes.ravel():
        ax.set_xlabel('t')
        ax.set_ylabel('s')
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
    fig.savefig(path, dpi=160)
    plt.close(fig)
