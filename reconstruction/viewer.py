"""Display saved local reconstruction geometry and signed displacement in mm."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from utils.surface_geometry import surface_grid_curve_lengths


def _surface_quads(xyz, displacement, valid, maximum_samples=90):
    """Coarsen display geometry without spanning invalid source-grid cells."""
    rows, columns = valid.shape
    ri = np.linspace(0, rows-1, min(rows, maximum_samples)).astype(int)
    ci = np.linspace(0, columns-1, min(columns, maximum_samples)).astype(int)
    points = xyz[np.ix_(ri, ci)]
    values = displacement[np.ix_(ri, ci)]
    quads = np.stack((points[:-1, :-1], points[:-1, 1:],
                      points[1:, 1:], points[1:, :-1]), axis=-2)
    colors = (values[:-1, :-1]+values[:-1, 1:]
              + values[1:, 1:]+values[1:, :-1])/4
    # Checking only coarse corners would hide narrow holes between those corners.
    integral = np.pad((~valid).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    r0, r1 = ri[:-1, None], ri[1:, None]+1
    c0, c1 = ci[None, :-1], ci[None, 1:]+1
    clear = integral[r1, c1]-integral[r0, c1]-integral[r1, c0]+integral[r0, c0] == 0
    return quads[clear], colors[clear]


def _surface_height_width_ratio(xyz: np.ndarray, valid: np.ndarray) -> float:
    """Use physical lengths, retaining grid proportions for degenerate previews."""
    try:
        width, height = surface_grid_curve_lengths(xyz, valid)
    except ValueError:
        return valid.shape[0]/valid.shape[1]
    return height/width


def create_reconstruction_figure(
    xyz_out: np.ndarray,
    displacement: np.ndarray,
    valid_mask: np.ndarray,
    *,
    figure=None,
):
    """Plot camera XYZ and signed normal displacement in mm.

    Both panels use the geometry mask, including displacement inferred at markers.
    This function builds a figure without opening a native GUI window.
    """
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    xyz = np.asarray(xyz_out, np.float64)
    displacement = np.asarray(displacement, np.float64)
    valid = np.asarray(valid_mask, bool)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or min(xyz.shape[:2]) < 2:
        raise ValueError('xyz_out must be rows x columns x 3, with at least two rows and columns')
    if any(array.shape != xyz.shape[:2] for array in (displacement, valid)):
        raise ValueError('displacement and valid_mask must match the XYZ grid')
    valid = valid & np.isfinite(xyz).all(axis=-1) & np.isfinite(displacement)
    if not valid.any():
        raise ValueError('Reconstruction contains no valid geometry to display')

    if figure is None:
        figure = Figure(figsize=(12, 6), layout='constrained')
    figure.suptitle('Local surface reconstruction')
    grid = figure.add_gridspec(1, 2, width_ratios=(1.5, 1), wspace=.25)
    surface_axis = figure.add_subplot(grid[0, 0], projection='3d')
    displacement_axis = figure.add_subplot(grid[0, 1])
    limit = max(float(np.max(np.abs(displacement[valid]))), 1e-6)
    normalization = Normalize(-limit, limit)
    cmap = colormaps['coolwarm']
    quads, values = _surface_quads(xyz, displacement, valid)
    if len(quads):
        surface_axis.add_collection3d(Poly3DCollection(
            quads, facecolors=cmap(normalization(values)), edgecolors='none',
            linewidths=0, antialiased=False))
    else:
        surface_axis.scatter(*xyz[valid].T, c=displacement[valid], cmap=cmap,
                             norm=normalization, s=2)
    points = xyz[valid]
    lower, upper = points.min(0), points.max(0)
    center = (lower+upper)/2
    spans = np.maximum(upper-lower, max(float(np.max(upper-lower)), 1e-3)*.04)
    surface_axis.set_xlim(center[0]-spans[0]/2, center[0]+spans[0]/2)
    surface_axis.set_ylim(center[1]-spans[1]/2, center[1]+spans[1]/2)
    surface_axis.set_zlim(center[2]-spans[2]/2, center[2]+spans[2]/2)
    surface_axis.set_box_aspect(spans)
    surface_axis.set_zticks([lower[2], upper[2]] if upper[2] > lower[2] else [center[2]])
    surface_axis.tick_params(axis='both', labelsize=8, pad=1)
    surface_axis.view_init(elev=30, azim=-60)
    surface_axis.set(xlabel='Camera X (mm)', ylabel='Camera Y (mm)', zlabel='Camera Z (mm)',
                     title='Reconstructed surface\nColor: normal displacement (mm)')

    displacement_image = displacement_axis.imshow(
        np.ma.array(displacement, mask=~valid), cmap=cmap, norm=normalization,
        origin='upper', interpolation='nearest', aspect='auto')
    displacement_axis.set_box_aspect(_surface_height_width_ratio(xyz, valid))
    displacement_axis.set_title('Normal displacement (mm)')
    figure.colorbar(displacement_image, ax=displacement_axis, shrink=.75, label='mm')
    displacement_axis.set(xlabel='Material grid column', ylabel='Material grid row')
    displacement_axis.set_facecolor('#454545')
    figure.supxlabel(
        f'Normal displacement: {displacement[valid].min():+.4f} to '
        f'{displacement[valid].max():+.4f} mm; positive = along baseline normal')
    return figure


def view_reconstruction(
    path: str | Path,
    *,
    image_output: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    """Read a local reconstruction NPZ; optionally save a figure and open a viewer."""
    source = Path(path).expanduser()
    fields = ('xyz_out', 'displacement', 'valid_mask')
    with np.load(source, allow_pickle=False) as data:
        missing = set(fields)-set(data.files)
        if missing:
            raise ValueError(f'{source} is not a reconstruction result; missing {", ".join(sorted(missing))}')
        arrays = {key: data[key] for key in fields}
    output = Path(image_output).expanduser() if image_output is not None else None
    if output is not None:
        if output.resolve() == source.resolve() or output.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}:
            raise ValueError('Preview output must be a PNG/JPG/SVG/PDF path distinct from the input NPZ')
        output.parent.mkdir(parents=True, exist_ok=True)
    figure = None
    if show:
        import matplotlib.pyplot as plt
        figure = plt.figure(figsize=(12, 6), layout='constrained')
    try:
        figure = create_reconstruction_figure(**arrays, figure=figure)
        if output is not None:
            figure.savefig(output, dpi=160)
            print(f'重建预览已保存: {output}')
        if show:
            print('拖动三维曲面可旋转视角；关闭窗口退出。')
            plt.show()
    finally:
        if show and figure is not None:
            plt.close(figure)
    return output
