"""Measure the curvature support of the paired training data at test-image material positions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from calibrate.utils.norm_pipeline import parse_settings
from optical_learning.generation_02.data import load_fixed_surface
from optical_learning.generation_02.model import BackgroundMLP
from optical_learning.generation_03.curvature_coverage import (
    COMPONENTS, compare_to_frame_set, distribution, envelopes, tensor_squared,
)
from utils.normal_models import load_normal_model
from utils.surface_sampling import bilinear_sample

from .compare_target import matched_inputs
from .run import CONFIG, HERE, ROOT, load_curvature_observations


ARCHIVE = HERE.parent / 'generation_03/runs/curvature_coverage/curvature_coverage.npz'
BASELINE = HERE.parent / 'generation_03/runs/configured_40px/margin_40/3b2'
IMAGE_3B1 = Path('assets/test_pics/3b/3b1.npz')


def range_summary(values: np.ndarray) -> dict:
    return dict(minimum=values.min(axis=0), **distribution(values))


def plot(path: Path, image: str, diameter: np.ndarray, nearest: np.ndarray, gap: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout='constrained')
    panels = (
        ('Training curvature span', diameter, np.nanquantile(diameter, .98)),
        (f'{image} to nearest training curvature', nearest, np.nanquantile(nearest, .98)),
        (f'{image} outside local training ranges', gap, np.nanquantile(nearest, .98)),
    )
    for ax, (title, values, limit) in zip(axes, panels, strict=True):
        panel = ax.imshow(values.T, origin='lower', extent=(0, 1, 0, 1), aspect='auto',
                          cmap='magma', vmin=0, vmax=limit)
        ax.set(title=title, xlabel='t', ylabel='s')
        figure.colorbar(panel, ax=ax, label='mm⁻¹', extend='max')
    figure.savefig(path, dpi=160)
    plt.close(figure)


def load_3b2_target(directory: Path, chart_id: str, half_width: float,
                    training_st: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the saved overall curvature and check the target grid against the reconstruction archives."""
    current, baseline = [json.loads((path / 'comparison.json').read_text()) for path in (directory, BASELINE)]
    for key in ('image', 'reference_image', 'grid_shape', 'contact_offset_mm', 'coordinate_chart_id',
                'background_smoothing_half_width_mm'):
        if current[key] != baseline[key]:
            raise ValueError(f'Archived 3b2 reconstruction differs in {key}')
    if current['coordinate_chart_id'] != chart_id or current['background_smoothing_half_width_mm'] != half_width:
        raise ValueError('Training and 3b2 use different material charts or curvature smoothing')
    coverage = json.loads(ARCHIVE.with_suffix('.json').read_text())
    if coverage['smoothing_half_width_mm'] != half_width:
        raise ValueError('Archived 3b2 curvature uses a different smoothing half-width')
    with np.load(ARCHIVE, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['training_st'], training_st)
        test, st, uv = saved['test_curvature_per_mm'], saved['test_st'], saved['test_uv']
    with np.load(directory / 'reference/3b2.npz', allow_pickle=False) as actual, \
            np.load(BASELINE / 'reference/3b2.npz', allow_pickle=False) as previous, \
            np.load(directory / 'reference/fit_selection.npz', allow_pickle=False) as fits:
        for key in ('xyz_out', 'photometric_valid_mask'):
            np.testing.assert_array_equal(actual[key], previous[key])
        np.testing.assert_array_equal(uv, fits['uv'])
        photo = actual['photometric_valid_mask']
    if not np.isfinite(test[photo]).all():
        raise ValueError('Archived 3b2 curvature is missing at photometric points')
    return test, st, uv, photo


def load_3b1_target(training_run: Path, model: BackgroundMLP,
                    chart_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the curvature from the same geometry and sampling as the saved 3b1 reconstruction."""
    model_dir = training_run / 'per_frame'
    archived = training_run.parent / 'compare_3b1'
    provenance = json.loads((archived / 'comparison.json').read_text())
    baseline = json.loads((model_dir / '3b2/comparison.json').read_text())
    if (Path(provenance['image']) != IMAGE_3B1 or provenance['coordinate_chart_id'] != chart_id
            or provenance['curvature_smoothing_half_width_mm'] != model.curvature_smoothing_half_width_mm):
        raise ValueError('3b1 reconstruction uses a different image, material chart or curvature smoothing')
    table = load_normal_model(model_dir / '3b2/reference/table.npz')
    data, _, _ = matched_inputs(ROOT / IMAGE_3B1, baseline, table, model)
    if data['coordinate_chart_id'] != chart_id:
        raise ValueError('3b1 material chart differs from training')
    with np.load(archived / 'reference/3b1.npz', allow_pickle=False) as saved, \
            np.load(archived / 'reference/fit_selection.npz', allow_pickle=False) as fits:
        photo = data['photometric_valid_mask']
        np.testing.assert_array_equal(photo, saved['photometric_valid_mask'])
        np.testing.assert_array_equal(data['uv'], fits['uv'])
        xyz = saved['xyz_out'].astype(np.float64) - saved['displacement_vectors']
        np.testing.assert_allclose(data['xyz'][photo], xyz[photo], atol=1e-5, rtol=0)
    return data['curvature_grid_per_mm'], data['st'], data['uv'], photo


def run(training_run: Path, output: Path, *, image: str = '3b2') -> dict:
    settings = parse_settings(CONFIG)
    parameters = json.loads((training_run / 'training.json').read_text())
    margin = parameters['contact_exclusion_margin_px']
    model_path = training_run / 'per_frame/model.npz'
    model = BackgroundMLP.load(model_path)
    half_width = model.curvature_smoothing_half_width_mm
    surface = load_fixed_surface(settings, half_width)
    observed, curvature, valid = load_curvature_observations(settings, surface, half_width, margin_px=margin)
    train = curvature[valid]
    if (len(train) != parameters['observations']['all'] or surface.chart_id != parameters['chart_id']
            or set(observed.keys) != set(parameters['partition']['train'] + parameters['partition']['held'])):
        raise ValueError('Reloaded curvature data differ from the training record')
    np.testing.assert_array_equal(train.mean(axis=0, dtype=np.float64).astype(np.float32), model.curvature_mean)
    scale = np.maximum(train.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    np.testing.assert_array_equal(scale, model.curvature_scale)

    counts, lower, upper, diameter = envelopes(curvature, valid)
    repeated = counts >= 2
    span = upper - lower
    train_range = range_summary(train)

    if image == '3b1':
        test, st, uv, photo = load_3b1_target(training_run, model, surface.chart_id)
        source = IMAGE_3B1
    elif image == '3b2':
        test, st, uv, photo = load_3b2_target(training_run / 'per_frame/3b2', surface.chart_id, half_width, surface.st)
        source = ARCHIVE.relative_to(ROOT)
    else:
        raise ValueError(f'Unsupported target image: {image}')
    query = st * [counts.shape[1] - 1, counts.shape[0] - 1]
    comparable_frames, local_min, local_max, nearest = compare_to_frame_set(curvature, valid, query, test)
    paired = photo & (comparable_frames > 0)
    local_range = photo & (comparable_frames >= 2)
    gap = test - np.clip(test, local_min, local_max)
    outside = local_range & np.any(gap != 0, axis=-1)
    gap_distance = np.sqrt(tensor_squared(gap))
    global_outside = np.any((test < train_range['minimum']) | (test > train_range['maximum']), axis=-1) & photo

    fixed = bilinear_sample(surface.features[..., 2:], query, counts > 0, allow_boundary=True)
    common = paired & fixed.valid
    fixed_distance = np.sqrt(tensor_squared(test - fixed.values))
    right = uv[..., 0] >= 445
    report = dict(
        model=str(model_path.relative_to(ROOT)), target_curvature_source=str(source),
        components=COMPONENTS, units='mm^-1', frames=len(observed.keys),
        contact_exclusion_margin_px=margin, smoothing_half_width_mm=half_width,
        training=dict(samples=len(train), material_points=int((counts > 0).sum()),
                      global_curvature=train_range, standardization_scale=scale,
                      pointwise_component_span=distribution(span[repeated]),
                      pointwise_tensor_span=distribution(diameter[repeated])),
    )
    report[f'test_{image}'] = dict(
        photometric_points=int(photo.sum()), global_curvature=range_summary(test[photo]),
        outside_global_ranges_points=int(global_outside.sum()),
        nearest_training_distance=distribution(nearest[paired]),
        nearest_training_distance_right=distribution(nearest[paired & right]),
        local_range_points=int(local_range.sum()), outside_local_range_points=int(outside.sum()),
        outside_local_range_fraction=float(outside[local_range].mean()),
        outside_local_range_by_component=np.count_nonzero(gap[local_range] != 0, axis=0),
        outside_local_range_distance=distribution(gap_distance[outside]) if outside.any() else None,
        nearest_training_distance_on_fixed_support=distribution(nearest[common]),
        fixed_reference_distance_on_same_support=distribution(fixed_distance[common]),
    )
    nearest_grid = np.where(paired, nearest, np.nan)
    gap_grid = np.where(local_range, gap_distance, np.nan)
    np.savez_compressed(output / 'curvature_coverage.npz',
        training_st=surface.st, training_counts=counts, training_min_per_mm=lower, training_max_per_mm=upper,
        training_span_per_mm=span, training_tensor_span_per_mm=diameter,
        test_st=st, test_uv=uv, test_curvature_per_mm=np.where(photo[..., None], test, np.nan),
        test_comparable_frames=comparable_frames, test_nearest_training_per_mm=nearest_grid,
        test_local_range_valid=local_range, test_outside_local_range=outside,
        test_local_min_per_mm=np.where(local_range[..., None], local_min, np.nan),
        test_local_max_per_mm=np.where(local_range[..., None], local_max, np.nan),
        test_gap_per_mm=np.where(local_range[..., None], gap, np.nan), test_gap_distance_per_mm=gap_grid,
        test_fixed_reference_distance_per_mm=np.where(common, fixed_distance, np.nan))
    plot(output / 'curvature_coverage.png', image, diameter, nearest_grid, gap_grid)
    (output / 'curvature_coverage.json').write_text(json.dumps(report, indent=2, allow_nan=False,
        default=lambda value: value.tolist()) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', choices=('3b1', '3b2'), default='3b2')
    parser.add_argument('--training-run', type=Path, default=HERE / 'runs/paired_curvature')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    name = 'curvature_coverage' if args.image == '3b2' else 'curvature_coverage_3b1'
    output = args.output_dir or HERE / 'runs' / name
    output.mkdir(parents=True)
    run(args.training_run.resolve(), output, image=args.image)
    print(f'Saved curvature coverage to {output}.')


if __name__ == '__main__':
    main()
