"""Render signed color differences, raw LUT p/q maps and calibration diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from utils.color_difference import signed_difference_preview
from utils.normal_mapping import NormalCalibration, find_ambiguous_color_pairs, lookup_slopes_raw
from utils.surface_geometry import local_slopes_to_normals
from utils.surface_sampling import rasterize_surface_grid


DECLARATIONS = ('coordinate_kind', 'coordinate_chart_id', 'frame_convention',
                'residual_domain', 'color_space', 'reference_image')


def _check_declarations(data, calibration, keys=DECLARATIONS):
    for key in keys:
        if str(data[key]) != calibration.metadata.get(key):
            raise ValueError(f'Visualization input/LUT {key} mismatch')
    try:
        offsets = np.asarray([data['contact_offset_mm'], calibration.metadata['contact_offset_mm']], np.float64)
    except (KeyError, TypeError, ValueError):
        raise ValueError('Visualization input/LUT requires contact_offset_mm') from None
    if (offsets.shape != (2,) or not np.isfinite(offsets).all()
            or not np.isclose(offsets[0], offsets[1], rtol=0, atol=1e-9)):
        raise ValueError('Visualization input/LUT contact_offset_mm mismatch')
    if 'color_space' in keys:
        transfer = str(data['rgb_transfer_function']) if 'rgb_transfer_function' in data else None
        if transfer != calibration.metadata.get('rgb_transfer_function'):
            raise ValueError('Visualization input/LUT rgb_transfer_function mismatch')


def _load_samples(path, calibration):
    with np.load(path, allow_pickle=False) as data:
        _check_declarations(data, calibration)
        fields = ('color_residual_samples', 'slope_samples', 'reference_normal_samples',
                  'tangent_frame_samples', 'image_index', 'image_paths')
        samples = {key: data[key] for key in fields}
    n = len(samples['color_residual_samples'])
    shapes = dict(color_residual_samples=(n, 3), slope_samples=(n, 2),
                  reference_normal_samples=(n, 3), tangent_frame_samples=(n, 3, 2), image_index=(n,))
    for key, shape in shapes.items():
        if samples[key].shape != shape or not np.isfinite(samples[key]).all():
            raise ValueError(f'Invalid calibration sample field: {key}')
    indices, names = samples['image_index'], samples['image_paths']
    if (not n or names.ndim != 1 or len(set(names.tolist())) != len(names)
            or indices.dtype.kind not in 'iu' or indices.min() < 0 or indices.max() >= len(names)
            or set(indices.tolist()) != set(range(len(names)))):
        raise ValueError('Calibration samples require nonempty, uniquely grouped images')
    if calibration.metadata.get('total_sample_count') != n:
        raise ValueError('LUT/sample count mismatch; use the samples that produced this LUT')
    return samples


def _predict(calibration, colors, truth, reference_normals, tangent_frames, valid):
    """Return raw LUT slopes for finite colors; valid only masks truth comparisons."""
    predicted, _, _ = lookup_slopes_raw(colors, calibration)
    predicted_normals, vp = local_slopes_to_normals(predicted, reference_normals, tangent_frames)
    true_normals, vt = local_slopes_to_normals(truth, reference_normals, tangent_frames)
    valid = np.asarray(valid, bool) & vp & vt
    dot = np.sum(predicted_normals * true_normals, axis=-1)
    angles = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    return predicted, np.where(valid, angles, np.nan), valid


def _statistics(angles):
    values = np.asarray(angles)[np.isfinite(angles)]
    if not len(values):
        raise ValueError('No valid normal comparisons are available')
    return dict(count=int(len(values)), rms_degrees=float(np.sqrt(np.mean(values**2))),
                median_degrees=float(np.median(values)), p95_degrees=float(np.percentile(values, 95)))


def _validation_label(calibration):
    status = calibration.metadata['validation_status']
    label = f'Validation: {status.upper()}'
    return label+' | diagnostic LUT; not published' if status == 'failed' else label


def _figure(size):
    figure = Figure(figsize=size, layout='constrained')
    FigureCanvasAgg(figure)
    return figure


def _save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, format='png')
    figure.clear()


def _scalar(figure, axis, values, valid, title, limits, *, extent=None, cmap='coolwarm'):
    palette = colormaps[cmap].with_extremes(bad='black')
    artist = axis.imshow(np.ma.array(values, mask=~valid), cmap=palette,
                         vmin=limits[0], vmax=limits[1], extent=extent, interpolation='nearest')
    axis.set(title=title, xlabel='image x (px)', ylabel='image y (px)')
    figure.colorbar(artist, ax=axis, shrink=.7, extend='both' if limits[0] < 0 else 'max')


def _ambiguity_figure(calibration, samples, output, color_distance_threshold, normal_angle_threshold_degrees):
    slopes = samples['slope_samples']
    pairs, distances, angles = find_ambiguous_color_pairs(samples['color_residual_samples'], slopes,
        color_distance_threshold=color_distance_threshold,
        normal_angle_threshold_degrees=normal_angle_threshold_degrees)
    ambiguous = np.zeros(len(slopes), bool)
    ambiguous[pairs.ravel()] = True
    count = int(ambiguous.sum())
    fraction = float(ambiguous.mean())
    limit = max(.5, float(np.ceil(np.abs(slopes).max()*1.05*2)/2))
    views = [(limit, 'Full gradient range')]
    if limit > 2:
        views.append((2., 'Central region | p, q in [-2, 2]'))
    figure = _figure((6*len(views), 6.6))
    axes = np.atleast_1d(figure.subplots(1, len(views)))
    for axis, (extent, title) in zip(axes, views, strict=True):
        visible = (np.abs(slopes) <= extent).all(axis=-1)
        for mask, color, marker, size, label in (
                (~ambiguous, '#777777', '.', 4, 'No conflicting neighbour'),
                (ambiguous, '#e31a1c', '*', 18, 'Ambiguous sample')):
            points = slopes[mask & visible]
            axis.scatter(points[:, 0], points[:, 1], s=size, color=color,
                         marker=marker, alpha=.65, linewidths=0, rasterized=True, label=label)
        # Match the reference figure's axis orientation without changing p/q signs.
        axis.set(xlabel='p (dimensionless)', ylabel='q (dimensionless)', title=title,
                 xlim=(-extent, extent), ylim=(extent, -extent), aspect='equal')
        axis.tick_params(direction='in', top=True, right=True)
        axis.legend(loc='upper right', fontsize=8, markerscale=1.5)
    figure.suptitle('Ambiguous color-to-gradient mappings | sphere-labelled samples\n'
        f'RGB L2 distance <= {color_distance_threshold:g} and local normal angle > '
        f'{normal_angle_threshold_degrees:g} deg | all unordered pairs, within and across images\n'
        f'Red: {count:,} / {len(slopes):,} samples ({fraction:.2%}) | '
        f'{len(pairs):,} conflicting pairs\n{_validation_label(calibration)}', fontsize=11)
    _save_figure(figure, output/'pq_ambiguities.png')
    np.savez_compressed(output/'ambiguities.npz', pairs=pairs, color_distances=distances,
        normal_angles_degrees=angles, ambiguous_mask=ambiguous,
        color_residual_samples=samples['color_residual_samples'], slope_samples=slopes,
        image_index=samples['image_index'], image_paths=samples['image_paths'],
        color_distance_threshold=color_distance_threshold,
        normal_angle_threshold_degrees=normal_angle_threshold_degrees)
    cross_image = samples['image_index'][pairs[:, 0]] != samples['image_index'][pairs[:, 1]]
    return dict(color_distance_threshold=color_distance_threshold,
        normal_angle_threshold_degrees=normal_angle_threshold_degrees,
        color_metric='Euclidean distance in signed linear RGB',
        normal_metric='Angle between normalize([-p, -q, 1]) in the shared local frame convention',
        pair_scope='All unordered pairs i < j, within and across images; both endpoints marked',
        sample_source='Raw paired sphere labels, before LUT fitting and outlier rejection',
        sample_count=len(slopes), ambiguous_sample_count=count, ambiguous_sample_fraction=fraction,
        ambiguous_pair_count=len(pairs), cross_image_pair_count=int(cross_image.sum()),
        within_image_pair_count=int((~cross_image).sum()),
        figure='pq_ambiguities.png', arrays='ambiguities.npz')


def _mapping_figures(calibration, samples, predicted, output, slope_limit):
    colors = samples['color_residual_samples']
    selected = np.linspace(0, len(colors)-1, min(6000, len(colors)), dtype=int)
    figure = _figure((13, 6))
    for component, name in enumerate(('p', 'q')):
        axis = figure.add_subplot(1, 2, component+1, projection='3d')
        points = axis.scatter(*colors[selected].T, c=predicted[selected, component], s=4,
                              cmap='coolwarm', vmin=-slope_limit, vmax=slope_limit, alpha=.7)
        axis.set(xlabel='delta R (linear)', ylabel='delta G (linear)', zlabel='delta B (linear)',
                 title=f'Color difference -> raw all-sample LUT {name}')
        figure.colorbar(points, ax=axis, shrink=.65, label=f'{name} (dimensionless)')
    figure.suptitle(f'Calibration color domain | {len(colors)} paired samples | at most 6000 shown\n'
                   f'{_validation_label(calibration)}')
    _save_figure(figure, output/'color_pq_mapping.png')

    figure = _figure((14, 8))
    axes = figure.subplots(2, 3)
    channel_names = ('R', 'G', 'B')
    for fixed in range(3):
        varying = [i for i in range(3) if i != fixed]
        coordinates = [np.linspace(calibration.color_min[i], calibration.color_max[i], calibration.size)
                       for i in varying]
        xx, yy = np.meshgrid(*coordinates)
        query = np.zeros((*xx.shape, 3))
        query[..., varying[0]], query[..., varying[1]] = xx, yy
        values, _, _ = lookup_slopes_raw(query, calibration)
        extent = (coordinates[0][0], coordinates[0][-1], coordinates[1][-1], coordinates[1][0])
        for component, name in enumerate(('p', 'q')):
            axis = axes[component, fixed]
            _scalar(figure, axis, values[..., component], np.ones(xx.shape, bool),
                    f'{name} | delta {channel_names[fixed]} = 0', (-slope_limit, slope_limit), extent=extent)
            axis.set(xlabel=f'delta {channel_names[varying[0]]} (linear)',
                     ylabel=f'delta {channel_names[varying[1]]} (linear)', aspect='auto')
    figure.suptitle('Completed LUT zero-channel slices | includes filled nodes | raw p/q\n'
                   f'{_validation_label(calibration)}')
    _save_figure(figure, output/'lut_slices.png')


def _validation_predictions(calibration_file, calibration, samples):
    report = calibration.metadata.get('validation', {})
    names = samples['image_paths']
    held_out = set(report.get('validation_images', []))
    training = set(report.get('training_images', []))
    if not held_out or training & held_out or training | held_out != set(names.tolist()):
        raise ValueError('LUT must declare the training/held-out image split')
    train_path = calibration_file.with_name(calibration_file.stem+'_train.npz')
    train = NormalCalibration.load(train_path)
    if (train.metadata.get('validation_status') != 'training_only'
            or set(train.metadata.get('training_images', [])) != training
            or set(train.metadata.get('validation_images', [])) != held_out):
        raise ValueError('Training LUT does not match the final LUT validation split')
    _check_declarations(train.metadata, calibration)
    mask = np.isin(names[samples['image_index']], list(held_out))
    predicted, angles, valid = _predict(train, samples['color_residual_samples'], samples['slope_samples'],
                                      samples['reference_normal_samples'], samples['tangent_frame_samples'], mask)
    if not valid[mask].all():
        raise ValueError('Held-out samples contain invalid local normal frames')
    return predicted, angles, mask


def _summary_figure(calibration, samples, predicted, angles, held_prediction, held_angles, held_mask, output):
    figure = _figure((14, 9))
    axes = figure.subplots(2, 3)
    truth = samples['slope_samples']
    for row, (values, selected, label) in enumerate((
        (predicted, np.ones(len(truth), bool), 'All-sample LUT: calibration fit'),
        (held_prediction, held_mask, 'Training-only LUT: held-out images'),
    )):
        ids = np.flatnonzero(selected)
        shown = ids[np.linspace(0, len(ids)-1, min(8000, len(ids)), dtype=int)]
        for component, name in enumerate(('p', 'q')):
            axis = axes[row, component]
            axis.scatter(truth[shown, component], values[shown, component], s=3, alpha=.25)
            lo = float(min(truth[ids, component].min(), values[ids, component].min()))
            hi = float(max(truth[ids, component].max(), values[ids, component].max()))
            axis.plot([lo, hi], [lo, hi], '--', color='black', linewidth=1)
            axis.set(title=f'{label}\n{name}', xlabel=f'Sphere truth {name}', ylabel=f'LUT predicted {name}')
            axis.grid(alpha=.2)
    axes[0, 2].hist(angles[np.isfinite(angles)], bins=50, color='#2878a0')
    axes[0, 2].set(title='All-sample LUT normal-angle differences', xlabel='Angle (degrees)', ylabel='Samples')
    keys = samples['image_paths']
    metrics = [_statistics(held_angles[(samples['image_index'] == i) & held_mask])
               for i in range(len(keys)) if held_mask[samples['image_index'] == i].any()]
    labels = [f'{i:06d}' for i in range(len(keys)) if held_mask[samples['image_index'] == i].any()]
    axis = axes[1, 2]
    axis.bar(labels, [m['rms_degrees'] for m in metrics], color='#c57a24', label='Held-out RMS')
    threshold = calibration.metadata['validation']['thresholds']['rms_degrees']
    axis.axhline(threshold, color='red', linestyle='--', label=f'Limit {threshold:g} deg')
    axis.set(title='Independent held-out validation', xlabel='Image index (see summary.json)', ylabel='RMS (degrees)')
    axis.legend(fontsize=8)
    supported = int(np.count_nonzero(calibration.original_valid)) if calibration.original_valid is not None else 0
    final_rms = _statistics(angles)['rms_degrees']
    held_rms = _statistics(held_angles[held_mask])['rms_degrees']
    figure.suptitle(f'LUT {calibration.size}^3 | supported nodes {supported}/{calibration.size**3} | '
                   f'{_validation_label(calibration)}\n'
                   f'All-sample fit RMS: {final_rms:.2f} deg | held-out RMS: {held_rms:.2f} deg')
    _save_figure(figure, output/'calibration_summary.png')


def _frame_paths(work_dir, samples, calibration):
    wanted = set(samples['image_paths'].tolist())
    paths = {}
    for path in sorted((work_dir/'frames').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            key = str(data['image_key'])
            if key not in wanted or str(data['coordinate_chart_id']) != calibration.metadata['coordinate_chart_id']:
                continue
            _check_declarations(data, calibration)
            if str(data['geometry_stage']) != 'final' or key in paths:
                raise ValueError(f'Duplicate or unfinished calibration frame: {key}')
            paths[key] = path
    if set(paths) != wanted:
        raise ValueError(f'Missing matching final frame caches: {sorted(wanted-set(paths))}')
    return paths


def _project_frame(path, sample_index, samples, calibration, work_dir):
    with np.load(path, allow_pickle=False) as data:
        frame = {key: data[key] for key in data.files}
    photo, paired = frame['photometric_valid_mask'], frame['sample_valid_mask']
    group = samples['image_index'] == sample_index
    if (not np.array_equal(frame['color_residual_grid'][paired], samples['color_residual_samples'][group])
            or not np.array_equal(frame['slope_grid'][paired], samples['slope_samples'][group])):
        raise ValueError(f'Frame/sample values differ: {frame["image_key"]}; use matching calibration caches')
    with np.load(work_dir/'overall'/path.name, allow_pickle=False) as data:
        if str(data['image_key']) != str(frame['image_key']):
            raise ValueError('Overall geometry cache refers to a different image')
        _check_declarations(data, calibration, ('coordinate_kind', 'coordinate_chart_id', 'frame_convention'))
        image_shape = tuple(data['image_shape'])
    predicted, angles, comparison = _predict(calibration, frame['color_residual_grid'], frame['slope_grid'],
                                           frame['reference_normals'], frame['tangent_frames'], paired)
    # Show p/q over all usable colors; compare truth only at paired samples.
    color = rasterize_surface_grid(np.dstack((frame['color_residual_grid'], predicted)),
                                   frame['uv'], photo, image_shape)
    truth = rasterize_surface_grid(np.dstack((frame['slope_grid'], angles)),
                                   frame['uv'], comparison, image_shape)
    uv = frame['uv'][photo]
    lower = np.maximum(np.floor(uv.min(0)).astype(int)-4, 0)
    upper = np.minimum(np.ceil(uv.max(0)).astype(int)+5, image_shape[::-1])
    x0, y0 = lower
    x1, y1 = upper
    crop = np.s_[y0:y1, x0:x1]
    extent = (x0-.5, x1-.5, y1-.5, y0-.5)
    return color.values[crop], color.valid[crop], truth.values[crop], truth.valid[crop], extent


def _frame_figures(path, index, key, samples, calibration, work_dir, output, slope_limit):
    values, photo, truth, paired, extent = _project_frame(path, index, samples, calibration, work_dir)
    figure = _figure((18, 4))
    axes = figure.subplots(1, 4)
    axes[0].imshow(signed_difference_preview(values[..., :3], photo)[..., ::-1], extent=extent)
    axes[0].set(title='Signed delta RGB | display = 0.5 + 2 delta', xlabel='image x (px)', ylabel='image y (px)')
    for i, channel in enumerate(('R', 'G', 'B')):
        limit = max(abs(float(calibration.color_min[i])), abs(float(calibration.color_max[i])), 1e-6)
        _scalar(figure, axes[i+1], values[..., i], photo, f'delta {channel} (linear RGB)', (-limit, limit), extent=extent)
    figure.suptitle(f'{index:06d} | {key} | observed minus reference; black = invalid\n'
                   f'{_validation_label(calibration)}')
    _save_figure(figure, output/'color_difference'/f'{index:06d}.png')

    figure = _figure((15, 7))
    axes = figure.subplots(2, 3)
    for component, name in enumerate(('p', 'q')):
        _scalar(figure, axes[0, component], values[..., 3+component], photo,
                f'All-sample LUT {name} (dimensionless)', (-slope_limit, slope_limit), extent=extent)
        _scalar(figure, axes[1, component], truth[..., component], paired,
                f'Sphere truth {name} (paired samples)', (-slope_limit, slope_limit), extent=extent)
    _scalar(figure, axes[0, 2], np.linalg.norm(values[..., 3:5], axis=-1), photo,
            'All-sample LUT sqrt(p^2 + q^2)', (0, np.sqrt(2)*slope_limit), extent=extent, cmap='viridis')
    angle_limit = max(5., calibration.metadata['validation']['thresholds']['rms_degrees'])
    _scalar(figure, axes[1, 2], truth[..., 2], paired, f'Normal error (deg; >{angle_limit:g} clipped)',
            (0, angle_limit), extent=extent, cmap='magma')
    figure.suptitle(f'{index:06d} | {key} | raw all-sample LUT; no zero-color protection\n'
                   f'{_validation_label(calibration)} | error only where sphere labels are valid')
    _save_figure(figure, output/'pq_mapping'/f'{index:06d}.png')


def _check_output_directory(output, inputs):
    if any(p == output or output in p.parents for p in inputs):
        raise ValueError('Visualization output directory must not contain calibration inputs')
    if not output.exists() or not any(output.iterdir()):
        return
    summary = output/'summary.json'
    report = json.loads(summary.read_text(encoding='utf-8')) if summary.is_file() else {}
    if report.get('report_kind') != 'normal_calibration_visualization':
        raise ValueError('Visualization output directory already contains unrelated files')
    # Accept the previous SVG only so report replacement can remove it.
    generated = {'summary.json', 'calibration_summary.png', 'color_pq_mapping.png', 'lut_slices.png',
                 'pq_ambiguities.png', 'pq_ambiguities.svg', 'ambiguities.npz'}
    for row in report['images']:
        generated.update((row['color_figure'], row['pq_figure']))
    existing = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if existing-generated:
        raise ValueError(f'Visualization directory contains unrelated files: {sorted(existing-generated)}')


def generate_calibration_visualizations(
    calibration_file: str | Path,
    sample_file: str | Path,
    work_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    color_distance_threshold: float = .01,
    normal_angle_threshold_degrees: float = 20.,
) -> Path:
    """Render a passed image LUT or a failed diagnostic LUT without changing inputs.

    Per-image plots use original distorted image pixels and conservative display
    interpolation; statistics use original material-grid samples. Held-out plots
    query the sibling *_train.npz, never the final all-sample LUT.
    """
    calibration_file, sample_file, work_dir = (Path(p).expanduser().resolve()
                                               for p in (calibration_file, sample_file, work_dir))
    output = (work_dir/'visualization' if output_dir is None else Path(output_dir).expanduser().resolve())
    _check_output_directory(output, (calibration_file, sample_file, work_dir/'frames', work_dir/'overall'))
    calibration = NormalCalibration.load(calibration_file)
    status = calibration.metadata.get('validation_status')
    if status not in {'passed', 'failed'}:
        raise ValueError('Visualization requires a validated or failed diagnostic image LUT and its paired samples')
    if status == 'failed' and calibration.metadata.get('validation', {}).get('status') != 'failed':
        raise ValueError('Diagnostic LUT requires its failed validation report')
    samples = _load_samples(sample_file, calibration)
    paths = _frame_paths(work_dir, samples, calibration)
    predicted, angles, valid = _predict(calibration, samples['color_residual_samples'], samples['slope_samples'],
                                       samples['reference_normal_samples'], samples['tangent_frame_samples'],
                                       np.ones(len(samples['image_index']), bool))
    if not valid.all():
        raise ValueError('Calibration samples contain invalid local normal frames')
    held_prediction, held_angles, held_mask = _validation_predictions(calibration_file, calibration, samples)
    slope_limit = max(float(np.abs(calibration.slopes).max()), float(np.abs(samples['slope_samples']).max()), 1e-6)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.normal-visualization-', dir=output.parent) as temporary:
        staging = Path(temporary)/'report'
        staging.mkdir()
        _mapping_figures(calibration, samples, predicted, staging, slope_limit)
        _summary_figure(calibration, samples, predicted, angles, held_prediction, held_angles, held_mask, staging)
        ambiguities = _ambiguity_figure(calibration, samples, staging, color_distance_threshold,
                                       normal_angle_threshold_degrees)
        rows = []
        for index, key in enumerate(samples['image_paths'].tolist()):
            group = samples['image_index'] == index
            row = dict(index=index, image_key=key, final_LUT=_statistics(angles[group]),
                       color_figure=f'color_difference/{index:06d}.png', pq_figure=f'pq_mapping/{index:06d}.png')
            if (held_mask & group).any():
                row['held_out_training_LUT'] = _statistics(held_angles[group & held_mask])
            rows.append(row)
            _frame_figures(paths[key], index, key, samples, calibration, work_dir, staging, slope_limit)
        report = dict(report_kind='normal_calibration_visualization',
                      coordinate_chart_id=calibration.metadata['coordinate_chart_id'],
                      calibration_file=str(calibration_file), sample_file=str(sample_file),
                      validation_status=status, validation=calibration.metadata['validation'],
                      original_valid_node_count=(int(np.count_nonzero(calibration.original_valid))
                                                 if calibration.original_valid is not None else None),
                      retained_sample_count=(int(np.sum(calibration.sample_counts))
                                             if calibration.sample_counts is not None else None),
                      ambiguities=ambiguities,
                      final_LUT=_statistics(angles), held_out_training_LUT=_statistics(held_angles[held_mask]),
                      images=rows, note='Raw LUT. All-sample plots describe fit; held-out metrics use training-only LUT. '
                                       'Failed diagnostic LUTs are not published for reconstruction.')
        (staging/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        previous = Path(temporary)/'previous'
        if output.exists():
            output.replace(previous)
        try:
            staging.replace(output)
        except OSError:
            if previous.exists():
                previous.replace(output)
            raise
    return output
