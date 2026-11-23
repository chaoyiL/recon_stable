"""Summarize and plot converged sphere-fitting errors across images."""

from collections import defaultdict
from math import ceil, fsum, isfinite, sqrt
from numbers import Integral, Real
from pathlib import Path
from typing import Sequence

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


_METRICS = (
    ('rmse', 'converged_rmse_mm', 'mm', 'mm^2'),
    ('mse', 'converged_mse_mm2', 'mm^2', 'mm^4'),
)
_STATISTIC_FIELDS = (
    'mean', 'variance', 'sample_variance', 'std', 'median', 'min', 'max',
    'q25', 'q75', 'p05', 'p95', 'iqr', 'pooled',
)


def _read_metric(record: dict, field: str) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    label = f"{record.get('image', '?')!r} / {record.get('method', '?')!r}: {field}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{label} must be a finite nonnegative number or None')
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{label} must be a finite nonnegative number or None') from exc
    if not isfinite(value) or value < 0:
        raise ValueError(f'{label} must be a finite nonnegative number or None')
    return value


def _fit_count(record: dict) -> int:
    count = record.get('fit_point_count')
    if (isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral)
            or count <= 0):
        label = f"{record.get('image', '?')!r} / {record.get('method', '?')!r}"
        raise ValueError(f'{label}: fit_point_count must be a positive integer for a valid metric')
    return int(count)


def _group_observations(records: Sequence[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        observation = {'image': record['image']}
        for metric, field, _, _ in _METRICS:
            observation[metric] = _read_metric(record, field)
        observation['count'] = _fit_count(record) if any(
            observation[metric] is not None for metric, _, _, _ in _METRICS) else None
        grouped[record['method']].append(observation)
    return grouped


def _pooled(values: list[float], counts: list[int], metric: str) -> float:
    maximum = max(values)
    if maximum == 0:
        return 0.0
    total = sum(counts)
    power = 2 if metric == 'rmse' else 1
    normalized = fsum(count / total * (value / maximum)**power
                      for value, count in zip(values, counts, strict=True))
    return maximum * (sqrt(normalized) if metric == 'rmse' else normalized)


def _summarize_metric(method: str, metric: str, unit: str, variance_unit: str,
                      observations: list[dict]) -> dict:
    present = [observation for observation in observations if observation[metric] is not None]
    values = [observation[metric] for observation in present]
    row = dict(method=method, metric=metric, unit=unit, variance_unit=variance_unit,
               n_total=len(observations), n=len(values))
    row.update(dict.fromkeys(_STATISTIC_FIELDS))
    if not values:
        return row

    array = np.asarray(values, dtype=float)
    q05, q25, median, q75, q95 = np.percentile(array, [5, 25, 50, 75, 95])
    row.update(
        mean=float(np.mean(array)),
        variance=float(np.var(array, ddof=0)),
        sample_variance=float(np.var(array, ddof=1)) if len(values) > 1 else None,
        std=float(np.std(array, ddof=0)),
        median=float(median), min=float(np.min(array)), max=float(np.max(array)),
        q25=float(q25), q75=float(q75), p05=float(q05), p95=float(q95),
        iqr=float(q75-q25),
        pooled=_pooled(values, [observation['count'] for observation in present], metric),
    )
    return row


def summarize_rmse(records: Sequence[dict], methods: Sequence[str]) -> list[dict]:
    """Summarize converged RMSE in mm and MSE in mm² for each method.

    Each image has equal weight in the distribution; variance and standard
    deviation use ddof=0, with sample variance also given for n > 1. ``pooled``
    weights residuals by fit-point count across images. Missing metrics stay
    absent from distributions; projection failures can retain fitted metrics.
    """
    grouped = _group_observations(records)
    return [_summarize_metric(method, metric, unit, variance_unit, grouped.get(method, []))
            for method in methods for metric, _, unit, variance_unit in _METRICS]


def _draw_distribution(axis, metric: str, title: str, ylabel: str, methods: list[str],
                       grouped: dict[str, list[dict]], summaries: dict[tuple[str, str], dict],
                       image_styles: dict[str, tuple]) -> None:
    positions = np.arange(1, len(methods)+1)
    for position, method in zip(positions, methods, strict=True):
        observations = [observation for observation in grouped.get(method, [])
                        if observation[metric] is not None]
        if not observations:
            continue
        values = [observation[metric] for observation in observations]
        axis.boxplot([values], positions=[position], widths=.48, patch_artist=True,
                     showfliers=False, boxprops=dict(facecolor='#d6e7f5', edgecolor='#477598'),
                     whiskerprops=dict(color='#477598'), capprops=dict(color='#477598'),
                     medianprops=dict(color='#244c69', linewidth=1.5))
        for observation in observations:
            color, offset = image_styles[observation['image']]
            axis.scatter(position + offset, observation[metric], s=34, color=color,
                         edgecolors='white', linewidths=.5, zorder=3)
        summary = summaries[method, metric]
        axis.errorbar(position, summary['mean'], yerr=summary['std'],
                      fmt='D', markersize=5, color='#202020', capsize=3,
                      linewidth=1.2, zorder=4)

    labels = [f"{method}\n({summaries[method, metric]['n']}/{summaries[method, metric]['n_total']})"
              for method in methods]
    axis.set_xticks(positions, labels)
    if methods:
        axis.set_xlim(.5, len(methods)+.5)
    axis.set(title=title, xlabel='Method (valid images / total images)', ylabel=ylabel)
    axis.grid(axis='y', alpha=.22)
    axis.set_axisbelow(True)


def write_rmse_distribution(path: Path, records: Sequence[dict], summaries: Sequence[dict]) -> None:
    """Plot convergence RMSE and MSE with a consistent color for each image."""
    grouped = _group_observations(records)
    methods = list(dict.fromkeys(row['method'] for row in summaries))
    summary_by_metric = {(row['method'], row['metric']): row for row in summaries}
    images = sorted({record['image'] for record in records})
    offsets = np.linspace(-.18, .18, len(images)) if len(images) > 1 else np.array([0.])
    palette = colormaps['tab20']
    image_styles = {name: (palette(index % palette.N), float(offset))
                    for index, (name, offset) in enumerate(zip(images, offsets, strict=True))}
    handles = [
        Patch(facecolor='#d6e7f5', edgecolor='#477598', label='Quartiles and whiskers'),
        Line2D([], [], marker='D', color='#202020', label='Mean ± population std'),
    ] + [Line2D([], [], marker='o', linestyle='', color=image_styles[name][0], label=name)
         for name in images]
    columns = min(6, len(handles))
    legend_rows = ceil(len(handles)/columns)
    figure = Figure(figsize=(max(11, 1.7*len(methods)), 6.2 + .22*legend_rows))
    FigureCanvasAgg(figure)
    figure.subplots_adjust(left=.07, right=.98, top=.87,
                           bottom=min(.2 + .04*legend_rows, .5), wspace=.27)
    rmse_axis, mse_axis = figure.subplots(1, 2)
    _draw_distribution(rmse_axis, 'rmse', 'Converged RMSE', 'Radial RMSE (mm)',
                       methods, grouped, summary_by_metric, image_styles)
    _draw_distribution(mse_axis, 'mse', 'Converged MSE', 'Radial MSE (mm²)',
                       methods, grouped, summary_by_metric, image_styles)
    figure.suptitle('Depth-converged sphere fitting across images', fontsize=14)
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .09),
                  ncol=columns, frameon=False)
    figure.text(.5, .025,
                'Each available image has equal weight; variance and std use ddof=0. '
                'Point colors and offsets identify the same image in both panels.',
                ha='center', va='bottom', fontsize=9)
    figure.savefig(path, dpi=160, bbox_inches='tight')
    figure.clear()
