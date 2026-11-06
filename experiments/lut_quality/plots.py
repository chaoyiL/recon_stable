"""Diagnostic plots for the signed RGB-to-slope LUT quality experiment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import PercentFormatter

from .metric import bin_samples


_NO_SAMPLE_COLOR = '#f3ead8'


def _figure(size: tuple[float, float]) -> Figure:
    figure = Figure(figsize=size, layout='constrained')
    FigureCanvasAgg(figure)
    return figure


def _save(figure: Figure, path: Path) -> None:
    figure.savefig(path, dpi=160, bbox_inches='tight')
    figure.clear()


def _detail_index(taus: np.ndarray, detail_tau: float) -> int:
    matches = np.flatnonzero(np.isclose(taus, detail_tau, rtol=1e-12, atol=1e-12))
    if len(matches) != 1:
        raise ValueError('detail_tau must occur exactly once in taus')
    return int(matches[0])


def _curves(report: dict, name: str) -> list[dict]:
    rows = report.get('curves')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{name} must contain nonempty curves')
    return rows


def _plot_curves(output: Path, reference_report: dict, holdout_report: dict | None) -> Path:
    reference = _curves(reference_report, 'reference_report')
    taus = np.asarray([row['tau'] for row in reference], np.float64)
    values = np.asarray([row['incompatibility'] for row in reference], np.float64)
    if not np.isfinite(taus).all() or not np.isfinite(values).all():
        raise ValueError('Reference curves must be finite')

    figure = _figure((8.2, 5.2))
    axis = figure.subplots()
    axis.plot(taus, values, marker='o', linewidth=2,
              label='Reference: optimal single-value incompatibility A(τ)')

    if holdout_report is not None:
        holdout_rows = _curves(holdout_report, 'holdout_report')
        holdout_taus = np.asarray([row['tau'] for row in holdout_rows], np.float64)
        if not np.array_equal(holdout_taus, taus):
            raise ValueError('Reference and holdout tau values differ')
        transferred = np.asarray([
            row['transferred_disc_failure_or_unseen'] for row in holdout_rows
        ], np.float64)
        known = np.asarray([
            np.nan if row['transferred_disc_known_bin_error'] is None
            else row['transferred_disc_known_bin_error']
            for row in holdout_rows
        ], np.float64)
        if not np.isfinite(transferred).all() or not np.isfinite(known[~np.isnan(known)]).all():
            raise ValueError('Holdout curves must be finite or null for known-bin error')
        axis.plot(holdout_taus, transferred, marker='s', linestyle='--',
                  label='Holdout: transferred-disc failure or unseen bin')
        if np.isfinite(known).any():
            axis.plot(holdout_taus, known, marker='^', linestyle=':',
                      label='Holdout: transferred-disc error on known bins')
        unseen = holdout_report.get('unseen_bin_fraction')
        if unseen is not None:
            if not np.isfinite(unseen):
                raise ValueError('unseen_bin_fraction must be finite or null')
            axis.axhline(float(unseen), color='#666666', linestyle='-.', linewidth=1.2,
                         label='Holdout: unseen-bin fraction (independent of τ)')

    axis.set(xlabel='Slope tolerance τ (Euclidean distance in dimensionless p, q)',
             ylabel='Incompatibility / transfer failure fraction',
             title='LUT single-value incompatibility and holdout transfer')
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_ylim(0, 1)
    axis.grid(alpha=.25)
    axis.legend(loc='best', fontsize=8)
    path = output/'curves.png'
    _save(figure, path)
    return path


def _plot_top_bins(output: Path, bins: dict, colors: np.ndarray, slopes: np.ndarray,
                   lut, detail_index: int, detail_tau: float, top_bins: int) -> Path:
    bin_ids = np.asarray(bins['bin_ids'], np.int64)
    sample_counts = np.asarray(bins['sample_count'], np.int64)
    centers = np.asarray(bins['optimal_center'], np.float64)[detail_index]
    incompatibility = np.asarray(bins['incompatibility'], np.float64)[detail_index]
    coordinates = np.asarray(bins['coordinates'], np.int64)
    if not (len(bin_ids) == len(sample_counts) == len(centers) == len(incompatibility)
            == len(coordinates)):
        raise ValueError('Per-bin arrays have inconsistent lengths')

    row_ids = bin_samples(colors, lut)['bin_ids']
    rank = sorted(range(len(bin_ids)), key=lambda index: (
        -incompatibility[index], -sample_counts[index], int(bin_ids[index])))
    shown_count = min(4, max(0, int(top_bins)), len(rank))
    selected = rank[:shown_count]
    figure = _figure((5.8*max(1, min(2, shown_count)),
                      4.9*max(1, int(np.ceil(shown_count/2)))))
    axes = np.atleast_1d(figure.subplots(max(1, int(np.ceil(shown_count/2))),
                                        min(2, max(1, shown_count)), squeeze=False)).ravel()

    node_values = [np.linspace(float(lut.color_min[channel]), float(lut.color_max[channel]), lut.size)
                   for channel in range(3)]
    for panel_index, row_index in enumerate(selected):
        axis = axes[panel_index]
        bin_id = int(bin_ids[row_index])
        points = slopes[row_ids == bin_id]
        if len(points) != sample_counts[row_index]:
            raise ValueError(f'Sample count does not match reference bin {bin_id}')
        center = centers[row_index]
        distances = np.linalg.norm(points-center, axis=1)
        if detail_tau == 0:
            inside = np.all(points == center, axis=1)
        else:
            radius = detail_tau + 1e-12*max(1., detail_tau)
            inside = distances <= radius
        axis.scatter(points[~inside, 0], points[~inside, 1], s=12, color='#e27a35', alpha=.68,
                     linewidths=0, rasterized=True)
        axis.scatter(points[inside, 0], points[inside, 1], s=14, color='#2878a0', alpha=.75,
                     linewidths=0, rasterized=True)
        axis.scatter([center[0]], [center[1]], marker='*', s=100, color='black', zorder=4)
        if detail_tau > 0:
            axis.add_patch(Circle(center, detail_tau, fill=False, edgecolor='black',
                                  linestyle='--', linewidth=1.2))
        rgb_index = coordinates[row_index]
        rgb_value = [node_values[channel][rgb_index[channel]] for channel in range(3)]
        axis.set_title(
            f"Rank {panel_index+1} | bin {bin_id}, RGB node {tuple(map(int, rgb_index))}\n"
            f"ΔRGB=({rgb_value[0]:.3g}, {rgb_value[1]:.3g}, {rgb_value[2]:.3g}) linear, unitless; "
            f"N={sample_counts[row_index]} | Aᵦ={incompatibility[row_index]:.1%}", fontsize=9)
        axis.set(xlabel='Slope p (dimensionless)', ylabel='Slope q (dimensionless)',
                 aspect='equal')
        axis.grid(alpha=.18)
        if len(points):
            lo = np.minimum(points.min(axis=0), center-detail_tau)
            hi = np.maximum(points.max(axis=0), center+detail_tau)
            margin = np.maximum((hi-lo)*.06, .05)
            axis.set_xlim(lo[0]-margin[0], hi[0]+margin[0])
            axis.set_ylim(lo[1]-margin[1], hi[1]+margin[1])

    for axis in axes[shown_count:]:
        axis.set_visible(False)
    handles = [
        Line2D([], [], marker='o', linestyle='none', color='#2878a0', label='Samples within τ of center'),
        Line2D([], [], marker='o', linestyle='none', color='#e27a35', label='Samples outside τ of center'),
        Line2D([], [], marker='*', linestyle='none', color='black', markersize=10, label='Optimal center'),
    ]
    if detail_tau > 0:
        handles.append(Line2D([], [], color='black', linestyle='--', label='τ boundary'))
    figure.legend(handles=handles, loc='lower center', ncol=2, fontsize=8)
    figure.suptitle(f'Highest-incompatibility occupied RGB bins | detail τ={detail_tau:g} '
                    '(dimensionless slope distance)', fontsize=12)
    path = output/'top_bins.png'
    _save(figure, path)
    return path


def _selected_blue_layers(coordinates: np.ndarray) -> np.ndarray:
    levels = np.unique(coordinates[:, 2])
    if len(levels) <= 3:
        return levels
    return levels[np.linspace(0, len(levels)-1, 3, dtype=int)]


def _slice_panels(output: Path, bins: dict, lut, detail_index: int,
                  detail_tau: float) -> tuple[Path, Path]:
    coordinates = np.asarray(bins['coordinates'], np.int64)
    values = np.asarray(bins['incompatibility'], np.float64)[detail_index]
    counts = np.asarray(bins['sample_count'], np.int64)
    layers = _selected_blue_layers(coordinates)
    size = lut.size
    red = np.linspace(float(lut.color_min[0]), float(lut.color_max[0]), size)
    green = np.linspace(float(lut.color_min[1]), float(lut.color_max[1]), size)
    blue = np.linspace(float(lut.color_min[2]), float(lut.color_max[2]), size)
    heatmap_cmap = colormaps['magma'].with_extremes(bad=_NO_SAMPLE_COLOR)
    count_cmap = colormaps['viridis'].with_extremes(bad=_NO_SAMPLE_COLOR)

    heat_figure = _figure((5.2*len(layers), 5.0))
    heat_axes = np.atleast_1d(heat_figure.subplots(1, len(layers), squeeze=False)).ravel()
    count_figure = _figure((5.2*len(layers), 5.0))
    count_axes = np.atleast_1d(count_figure.subplots(1, len(layers), squeeze=False)).ravel()
    count_max = max(1, int(counts.max()))
    count_norm = LogNorm(vmin=1, vmax=max(2, count_max))
    for axis, count_axis, blue_index in zip(heat_axes, count_axes, layers, strict=True):
        selected = coordinates[:, 2] == blue_index
        heat = np.full((size, size), np.nan, np.float64)
        sample_count = np.full((size, size), np.nan, np.float64)
        red_index, green_index = coordinates[selected, 0], coordinates[selected, 1]
        heat[green_index, red_index] = values[selected]
        sample_count[green_index, red_index] = counts[selected]
        artist = axis.pcolormesh(red, green, np.ma.masked_invalid(heat), shading='nearest',
                                 cmap=heatmap_cmap, vmin=0, vmax=1)
        count_artist = count_axis.pcolormesh(red, green, np.ma.masked_invalid(sample_count),
            shading='nearest', cmap=count_cmap, norm=count_norm)
        b_value = blue[blue_index]
        axis.set(title=f'B node {blue_index} | ΔB={b_value:.4g} (linear, unitless)',
                 xlabel='ΔR (linear RGB residual, unitless)', ylabel='ΔG (linear RGB residual, unitless)')
        count_axis.set(title=f'B node {blue_index} | ΔB={b_value:.4g} (linear, unitless)',
                       xlabel='ΔR (linear RGB residual, unitless)',
                       ylabel='ΔG (linear RGB residual, unitless)')
        axis.set_aspect('auto')
        count_axis.set_aspect('auto')
        heat_figure.colorbar(artist, ax=axis, shrink=.82,
                             format=PercentFormatter(xmax=1.0), label=f'Per-bin incompatibility Aᵦ (τ={detail_tau:g})')
        count_figure.colorbar(count_artist, ax=count_axis, shrink=.82,
                              label='Raw reference sample count per bin (log scale)')
    missing = Patch(facecolor=_NO_SAMPLE_COLOR, edgecolor='#888888', label='No reference samples')
    heat_figure.legend(handles=[missing], loc='upper center', bbox_to_anchor=(.5, -.02), fontsize=8)
    count_figure.legend(handles=[missing], loc='upper center', bbox_to_anchor=(.5, -.02), fontsize=8)
    heat_figure.suptitle('Per-bin incompatibility on occupied RGB slices', fontsize=12)
    count_figure.suptitle('Reference sample counts on occupied RGB slices', fontsize=12)
    heat_path, count_path = output/'bin_heatmaps.png', output/'sample_counts.png'
    _save(heat_figure, heat_path)
    _save(count_figure, count_path)
    return heat_path, count_path


def _plot_normal_coverage(output: Path, bins: dict) -> Path:
    counts = np.asarray(bins['normal_coverage_count'], np.int64)
    theta_edges = np.asarray(bins['theta_bin_edges_degrees'], np.float64)
    phi_edges = np.asarray(bins['phi_bin_edges_degrees'], np.float64)
    if counts.shape != (len(theta_edges)-1, len(phi_edges)-1):
        raise ValueError('Normal coverage counts do not match their angle edges')
    figure = _figure((9.0, 5.0))
    axis = figure.subplots()
    artist = axis.pcolormesh(phi_edges, theta_edges, counts, shading='flat', cmap='viridis')
    axis.set(xlabel='Azimuth φ = atan2(−q, −p) (degrees)',
             ylabel='Tilt θ = atan(√(p² + q²)) (degrees)',
             title='Unweighted reference normal coverage')
    figure.colorbar(artist, ax=axis, label='Raw reference sample count')
    path = output/'normal_coverage.png'
    _save(figure, path)
    return path


def render_plots(
    output_dir: str | Path,
    reference_report: dict,
    holdout_report: dict | None,
    bins: dict,
    colors: np.ndarray,
    slopes: np.ndarray,
    reference_poses: np.ndarray,
    lut,
    taus: np.ndarray,
    *,
    detail_tau: float,
    top_bins: int = 8,
) -> dict[str, str]:
    """Write the five figures described by the quality report format.

    ``colors``, ``slopes`` and ``reference_poses`` must already be restricted
    to the reference split. Counts and normal coverage remain unweighted.
    """
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(colors, np.float64)
    slopes = np.asarray(slopes, np.float64)
    reference_poses = np.asarray(reference_poses)
    taus = np.asarray(taus, np.float64)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError('colors must have shape [N,3]')
    if slopes.shape != (len(colors), 2) or reference_poses.shape != (len(colors),):
        raise ValueError('Reference colors, slopes and poses must have matching sample counts')
    if not len(colors) or not np.isfinite(colors).all() or not np.isfinite(slopes).all():
        raise ValueError('Reference samples must be nonempty and finite')
    if top_bins < 1:
        raise ValueError('top_bins must be positive')
    detail_index = _detail_index(taus, detail_tau)
    bins_taus = np.asarray(bins['taus'], np.float64)
    if not np.array_equal(bins_taus, taus):
        raise ValueError('Per-bin tau values differ from the requested taus')

    paths = [
        _plot_curves(output, reference_report, holdout_report),
        _plot_top_bins(output, bins, colors, slopes, lut, detail_index, detail_tau, top_bins),
    ]
    paths.extend(_slice_panels(output, bins, lut, detail_index, detail_tau))
    paths.append(_plot_normal_coverage(output, bins))
    return {path.name: str(path) for path in paths}
