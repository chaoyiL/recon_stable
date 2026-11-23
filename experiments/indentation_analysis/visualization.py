"""Inspect depth-converged fitting points and fixed-radius spheres."""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
import numpy as np

from .analysis import ContactResult, Reconstruction, _baseline_direction
from .geometry import SphereFit, sphere_indentation_depth


@dataclass(frozen=True)
class SphereFitView:
    """Saved fit geometry in camera XYZ millimetres, with observed point sets."""

    label: str
    candidate_xyz_mm: np.ndarray
    fit_xyz_mm: np.ndarray
    excluded_xyz_mm: np.ndarray
    fit: SphereFit | None
    projection_xyz_mm: np.ndarray | None
    center_height_mm: float | None = None
    indentation_depth_mm: float | None = None
    status: str | None = None


def _depth_label(view: SphereFitView) -> str:
    if view.center_height_mm is None or view.indentation_depth_mm is None:
        return 'd=N/A'
    return f'h_c={view.center_height_mm:+.4f} mm | d={view.indentation_depth_mm:.4f} mm'


def _sphere_mesh(center: np.ndarray, radius: float) -> np.ndarray:
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 33), np.linspace(0, np.pi, 17))
    return center+radius*np.stack((np.sin(v)*np.cos(u), np.sin(v)*np.sin(u), np.cos(v)), axis=-1)


def _fit_bounds(views: Sequence[SphereFitView], *, include_candidates: bool):
    clouds = []
    for view in views:
        if include_candidates and len(view.candidate_xyz_mm):
            clouds.append(view.candidate_xyz_mm)
        if view.fit is not None:
            center, radius = view.fit.center_xyz_mm, view.fit.radius_mm
            clouds.extend((np.stack((center-radius, center+radius)), view.fit_xyz_mm))
            if view.projection_xyz_mm is not None:
                clouds.append(view.projection_xyz_mm[None])
    if not clouds:
        clouds = [view.candidate_xyz_mm for view in views if len(view.candidate_xyz_mm)]
    if not clouds:
        raise ValueError('Sphere-fit overview requires observed points or a fitted sphere')
    points = np.concatenate(clouds)
    lower, upper = points.min(axis=0), points.max(axis=0)
    spans = np.maximum(upper-lower, .1)
    midpoint = (upper+lower)/2
    return midpoint-.55*spans, midpoint+.55*spans


def _set_fit_axes(axis, bounds) -> None:
    lower, upper = bounds
    for setter, j in ((axis.set_xlim, 0), (axis.set_ylim, 1), (axis.set_zlim, 2)):
        setter(lower[j], upper[j])
    axis.set_box_aspect(upper-lower)
    axis.view_init(elev=25, azim=-60)
    axis.set(xlabel='Camera X (mm)', ylabel='Camera Y (mm)', zlabel='Camera Z (mm)')
    axis.tick_params(labelsize=7)


def _draw_sphere_fit(axis, view: SphereFitView, norm: TwoSlopeNorm, bounds):
    fit = view.fit
    assert fit is not None
    lower, upper = bounds
    excluded = view.excluded_xyz_mm
    # Keep the comparison frame around the spheres; distant background points
    # would stretch the axes and obscure the measured cap.
    visible = np.all((excluded >= lower) & (excluded <= upper), axis=1)
    if visible.any():
        axis.scatter(*excluded[visible].T, color='#999999', s=3, alpha=.2, depthshade=False)
    residuals = np.linalg.norm(view.fit_xyz_mm-fit.center_xyz_mm, axis=1)-fit.radius_mm
    artist = axis.scatter(*view.fit_xyz_mm.T, c=residuals, cmap='coolwarm',
                          norm=norm, s=6, alpha=.9, depthshade=False)
    sphere = _sphere_mesh(fit.center_xyz_mm, fit.radius_mm)
    axis.plot_wireframe(sphere[..., 0], sphere[..., 1], sphere[..., 2], color='#222222',
                        linewidth=.45, alpha=.55, rstride=2, cstride=2)
    axis.scatter(*fit.center_xyz_mm, color='black', marker='x', s=40)
    if view.projection_xyz_mm is not None:
        axis.plot(*np.stack((view.projection_xyz_mm, fit.center_xyz_mm)).T,
                  color='magenta', linewidth=2)
    _set_fit_axes(axis, bounds)
    return artist


def write_sphere_fit_overview(path: Path, views: Sequence[SphereFitView]) -> None:
    """Compare saved cap fits using shared physical bounds and residual colors.

    Colored points are the original fitted subset. Gray points are excluded
    candidates inside the displayed spatial window. Fits and projections are
    supplied by the saved analysis, independently of the depth-curve limits.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    if not views:
        raise ValueError('Sphere-fit overview requires at least one view')
    columns = min(3, len(views))
    rows = (len(views)+columns-1)//columns
    figure = Figure(figsize=(6.1*columns+.5, 6.0*rows), layout='constrained')
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(rows, columns+1, width_ratios=[1]*columns+[.045], hspace=.08)
    bounds = _fit_bounds(views, include_candidates=False)
    fitted = [view for view in views if view.fit is not None and len(view.fit_xyz_mm)]
    limit = max((float(np.max(np.abs(np.linalg.norm(view.fit_xyz_mm-view.fit.center_xyz_mm, axis=1)
                                    - view.fit.radius_mm))) for view in fitted), default=1e-6)
    norm = TwoSlopeNorm(vmin=-max(limit, 1e-6), vcenter=0., vmax=max(limit, 1e-6))
    for index, view in enumerate(views):
        axis = figure.add_subplot(grid[index//columns, index % columns], projection='3d')
        if view.fit is None:
            lower, upper = bounds
            visible = np.all((view.candidate_xyz_mm >= lower) & (view.candidate_xyz_mm <= upper), axis=1)
            axis.scatter(*view.candidate_xyz_mm[visible].T, color='#999999', s=3, alpha=.3)
            _set_fit_axes(axis, bounds)
            status = 'Depth scan not converged' if view.status == 'depth_not_converged' else 'Fit unavailable'
            axis.set_title(f'{view.label}\n{status}', fontsize=11)
        else:
            _draw_sphere_fit(axis, view, norm, bounds)
            axis.set_title(f'{view.label} | R={view.fit.radius_mm:g} mm\n'
                           f'N={len(view.fit_xyz_mm)}/{len(view.candidate_xyz_mm)} | '
                           f'Radial RMS={view.fit.radial_rms_mm:.4f} mm\n'
                           f'{_depth_label(view)}', fontsize=11)
    handles = [
        Line2D([], [], color='#999999', marker='o', linestyle='', label='Nearby excluded candidates'),
        Line2D([], [], color='#3b4cc0', marker='o', linestyle='', label='Depth fit: negative radial residual'),
        Line2D([], [], color='#b40426', marker='o', linestyle='', label='Depth fit: positive radial residual'),
        Line2D([], [], color='#222222', label='Fixed-radius sphere'),
        Line2D([], [], color='black', marker='x', linestyle='', label='Fitted sphere center'),
        Line2D([], [], color='magenta', label='Center-to-baseline segment'),
    ]
    if len(views) < rows*columns:
        legend_axis = figure.add_subplot(grid[-1, -2])
        legend_axis.set_axis_off()
        legend_axis.legend(handles=handles, loc='upper left', frameon=False, fontsize=11)
        legend_axis.text(.03, .35, 'Radial residual = ||P - C|| - R\n'
                         'N = depth-fit / all candidate points\n\n'
                         'Common camera XYZ limits and view angle\n'
                         'Common residual color scale\n'
                         'Gray points shown inside the spatial window',
                         transform=legend_axis.transAxes, fontsize=11, linespacing=1.6, va='top')
    else:
        figure.legend(handles=handles, loc='outside lower center', ncols=min(columns, 3), fontsize=9)
    if fitted:
        color_axis = figure.add_subplot(grid[:, -1])
        figure.colorbar(ScalarMappable(norm=norm, cmap='coolwarm'), cax=color_axis,
                        label='Radial residual (mm)')
    figure.suptitle('Depth-converged spherical-cap fits across LUTs\n'
                   'h_c > 0 along positive baseline normal | indentation d = R - h_c (mm)', fontsize=13)
    figure.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.2)
    figure.clear()


def write_analysis_figure(path: Path, reconstruction: Reconstruction,
                          results: list[ContactResult]) -> None:
    figure = Figure(figsize=(16, 6), layout='constrained')
    FigureCanvasAgg(figure)
    residuals, image, lines = figure.subplots(1, 3)
    colors = colormaps['tab10']
    for i, result in enumerate(results):
        color = colors(i % 10)
        index = result.contact.index
        samples = result.depth_samples
        label = f'Contact {index}'
        if result.fit is None:
            label += ' (not converged)'
        residuals.plot([sample.depth_mm for sample in samples],
                       [np.nan if sample.fit is None else sample.fit.radial_rms_mm
                        for sample in samples], color=color, label=label)
        if result.fit is not None:
            residuals.scatter(result.selection['converged_depth_mm'], result.fit.radial_rms_mm,
                              color=color, marker='*', s=90, zorder=4)
        image.contour(result.contact.region_mask.astype(float), levels=[.5],
                      colors=['#888888'], linewidths=.8, linestyles='--')
        if result.fit_mask.any():
            image.contour(result.fit_mask.astype(float), levels=[.5],
                          colors=[color], linewidths=1.2)
            row, column = np.argwhere(result.fit_mask).mean(axis=0)
        else:
            row, column = result.contact.center_rc
        image.scatter(column, row, color=color, edgecolor='white', s=35, zorder=5)
        image.annotate(str(index), (column, row), xytext=(6, -6),
                       textcoords='offset points', color=color, fontweight='bold')
        if result.projection is not None:
            point = result.projection.xyz_mm
            center = result.fit.center_xyz_mm
            lines.plot([point[0], center[0]], [point[2], center[2]], color=color, linewidth=2.2)
            lines.scatter(point[0], point[2], color=color, marker='x', s=50)
            lines.scatter(center[0], center[2], color=color, marker='o', s=25)
            lines.annotate(str(index), (center[0], center[2]),
                           xytext=(4, 5), textcoords='offset points', color=color)
    residuals.set(xlabel='Depth range d from deepest point (mm)', ylabel='Radial RMSE (mm)',
                  title='Depth scan; star = first convergence')
    residuals.grid(axis='y', alpha=.2)
    if residuals.get_legend_handles_labels()[0]:
        residuals.legend(fontsize=8)
    values = np.ma.array(reconstruction.displacement, mask=~reconstruction.valid)
    limit = max(float(np.max(np.abs(values.compressed()))), 1e-6)
    shown = image.imshow(values, cmap='coolwarm', vmin=-limit, vmax=limit,
                         origin='upper', aspect='equal', zorder=-1)
    image.set(xlabel='Material grid column', ylabel='Material grid row',
              title='Dashed: candidates; colored: converged depth subsets')
    figure.colorbar(shown, ax=image, label='Normal displacement (mm)', shrink=.75)
    lines.set(xlabel='Camera X (mm)', ylabel='Camera Z (mm)',
              title='Center-to-surface segments projected onto X-Z')
    lines.grid(alpha=.2)
    figure.suptitle('Sphere fitting and normal-angle analysis')
    figure.savefig(path, dpi=160)
    figure.clear()


def write_fit_check(path: Path, reconstruction: Reconstruction,
                    results: list[ContactResult]) -> None:
    figure = Figure(figsize=(max(12, 5.5*len(results)), 6), layout='constrained')
    FigureCanvasAgg(figure)
    for i, result in enumerate(results, start=1):
        axis = figure.add_subplot(1, len(results), i, projection='3d')
        candidate = result.contact.region_mask & reconstruction.photometric_valid
        fit = result.fit
        if fit is None:
            points = reconstruction.xyz[candidate]
            if len(points):
                axis.scatter(*points.T, color='#aaaaaa', s=3, alpha=.4)
                axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), .1))
            axis.set(xlabel='Camera X (mm)', ylabel='Camera Y (mm)', zlabel='Camera Z (mm)',
                     title=f'Contact {result.contact.index}: {result.selection["method"]}')
            axis.text2D(.05, .03, result.error or 'Fit failed', fontsize=7, transform=axis.transAxes)
            continue
        points = reconstruction.xyz[result.fit_mask]
        height, depth = None, None
        if result.projection is not None:
            height, depth = sphere_indentation_depth(
                fit.center_xyz_mm, fit.radius_mm, result.projection.xyz_mm,
                _baseline_direction(reconstruction, result.fit_mask))
        view = SphereFitView(
            label=f'Contact {result.contact.index}', candidate_xyz_mm=reconstruction.xyz[candidate],
            fit_xyz_mm=points, excluded_xyz_mm=reconstruction.xyz[candidate & ~result.fit_mask],
            fit=fit, projection_xyz_mm=None if result.projection is None else result.projection.xyz_mm,
            center_height_mm=height, indentation_depth_mm=depth,
            status=result.selection['method'])
        residuals = np.linalg.norm(points-fit.center_xyz_mm, axis=1)-fit.radius_mm
        limit = max(float(np.max(np.abs(residuals))), 1e-6)
        artist = _draw_sphere_fit(axis, view, TwoSlopeNorm(vmin=-limit, vcenter=0., vmax=limit),
                                  _fit_bounds([view], include_candidates=True))
        axis.set_title(f'Contact {result.contact.index} | R={fit.radius_mm:g} mm | '
                       f'N={len(points)}/{int(candidate.sum())}\nRadial RMS={fit.radial_rms_mm:.4f} mm\n'
                       f'{_depth_label(view)}')
        figure.colorbar(artist, ax=axis, shrink=.5, pad=.02, label='Radial residual (mm)')
    figure.suptitle('Depth-converged fitting points and fixed-radius spheres')
    figure.supxlabel('Gray: shallower candidates; colored: depth-fit points; black mesh: sphere; magenta: projection segment')
    figure.savefig(path, dpi=150)
    figure.clear()
