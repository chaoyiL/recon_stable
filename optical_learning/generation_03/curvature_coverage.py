"""Audit pointwise curvature coverage and 3b2 distances in the frozen material chart."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from calibrate.utils.norm_pipeline import load_frame_calibration_data, parse_settings
from optical_learning.generation_02.data import FixedSurface, load_observations
from optical_learning.generation_02.model import BackgroundMLP
from utils.normal_models import load_normal_model
from utils.surface_curvature import compute_surface_curvature
from utils.surface_sampling import bilinear_sample


HERE = Path(__file__).resolve().parent
RUN = HERE/'runs/configured_40px/margin_40'
COMPONENTS = ('k11', 'k12', 'k22')


def tensor_squared(values: np.ndarray) -> np.ndarray:
    """Squared Frobenius norm with the mixed term counted twice."""
    return (values[..., 0]**2 + 2*values[..., 1]**2 + values[..., 2]**2)


def distribution(values: np.ndarray) -> dict:
    if not len(values):
        raise ValueError('Curvature comparison has no supported points')
    low, middle, high, p99 = np.percentile(values, [5, 50, 95, 99], axis=0)
    return dict(n=len(values), p05=low, median=middle, p95=high, p99=p99, maximum=values.max(axis=0))


def frame_curvatures(settings, surface, masks, keys, half_width):
    manifest = json.loads((settings.work_dir/'manifest.json').read_text())
    indices = {row['key']: row['index'] for row in manifest['images']}
    fields, valid = [], []
    for name, rgb_mask in zip(keys, masks, strict=True):
        path = settings.work_dir/'overall'/f'{indices[name]:06d}.npz'
        frame = load_frame_calibration_data(path, settings)
        grid = frame.systems.observation
        if (frame.image_key != name or frame.geometry_stage != 'final' or grid.chart_id != surface.chart_id
                or not np.allclose(grid.st, surface.st, atol=1e-7, rtol=0)):
            raise ValueError(f'{name}: fitted geometry and background sample identities differ')
        curvature = compute_surface_curvature(grid.xyz, grid.valid_mask, grid.coordinates, frame.template_xy_mm,
                                              smoothing_half_width_mm=half_width)
        fields.append(curvature.values)
        valid.append(curvature.valid & rgb_mask)
    return np.stack(fields), np.stack(valid)


def envelopes(curvatures, valid):
    counts = valid.sum(axis=0)
    bounds = valid[..., None]
    lo = np.min(np.where(bounds, curvatures, np.inf), axis=0)
    hi = np.max(np.where(bounds, curvatures, -np.inf), axis=0)
    lo[counts == 0] = np.nan
    hi[counts == 0] = np.nan
    diameter2 = np.zeros(counts.shape, dtype=np.float32)
    for index in range(len(curvatures)-1):
        pair_valid = valid[index] & valid[index+1:]
        candidate = tensor_squared(curvatures[index] - curvatures[index+1:])
        diameter2 = np.maximum(diameter2, np.where(pair_valid, candidate, 0).max(axis=0))
    return counts, lo, hi, np.where(counts >= 2, np.sqrt(diameter2), np.nan)


def compare_to_frame_set(curvatures, valid, query, test):
    counts = np.zeros(test.shape[:2], dtype=np.int32)
    nearest = np.full(test.shape[:2], np.inf)
    lower, upper = np.full(test.shape, np.inf), np.full(test.shape, -np.inf)
    for grid, mask in zip(curvatures, valid, strict=True):
        sample = bilinear_sample(grid, query, mask, allow_boundary=True)
        counts += sample.valid
        support = sample.valid[..., None]
        lower = np.minimum(lower, np.where(support, sample.values, np.inf))
        upper = np.maximum(upper, np.where(support, sample.values, -np.inf))
        distance2 = tensor_squared(test - sample.values)
        nearest = np.minimum(nearest, np.where(sample.valid, distance2, np.inf))
    lower[counts == 0] = np.nan
    upper[counts == 0] = np.nan
    nearest[counts == 0] = np.nan
    return counts, lower, upper, np.sqrt(nearest)


def plot(path: Path, train_diameter, delta, distance, nearest):
    figure, axes = plt.subplots(2, 3, figsize=(12, 5.2), layout='constrained')
    physical = (train_diameter, distance, nearest)
    abs_limit = float(np.nanquantile(np.abs(delta), .98))
    distance_limit = max(float(np.nanquantile(values, .98)) for values in physical)
    titles = ('Calibration frame curvature span', '3b2 distance to training', '3b2 distance to nearest frame')
    for column, ax in enumerate(axes[0]):
        view = physical[column].T[::-1]
        top = ax.imshow(view, cmap='magma', extent=(0, 1, 0, 1), vmin=0, vmax=distance_limit, aspect='auto')
        ax.set_title(titles[column])
    for column, ax in enumerate(axes[1]):
        view = delta[..., column].T[::-1]
        bottom = ax.imshow(view, cmap='coolwarm', extent=(0, 1, 0, 1), vmin=-abs_limit, vmax=abs_limit, aspect='auto')
        ax.set_title(f'3b2 minus training {COMPONENTS[column]}')
    for ax in axes.flat:
        ax.set(xlabel='t', ylabel='s')
    figure.colorbar(top, ax=axes[0], label='mm⁻¹')
    figure.colorbar(bottom, ax=axes[1], label='mm⁻¹')
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(output: Path) -> dict:
    from optical_learning.lut_background_3b2.experiment import CONFIG, ROOT, matched_inputs

    settings = parse_settings(CONFIG)
    model_path = RUN/'background/model.npz'
    model = BackgroundMLP.load(model_path)
    half_width = model.curvature_smoothing_half_width_mm
    with np.load(RUN/'background/evaluation_grids.npz', allow_pickle=False) as saved:
        st, reference, support = saved['st'], saved['curvature_per_mm'], saved['geometry_valid']
        train_count, morning_count = saved['all_counts'], saved['train_counts']
        reference_valid = saved['reference_valid']
    with np.load(model_path, allow_pickle=False) as saved:
        chart_id = str(saved['chart_id'])
    summary = json.loads((RUN/'background/metrics.json').read_text())
    margin = summary['contact_exclusion_margin_px']
    surface = FixedSurface(np.concatenate((st, reference), axis=-1), support, chart_id)
    observed = load_observations(settings, ROOT/'assets/cali_norm_pics', surface, contact_exclusion_margin_px=margin)
    np.testing.assert_array_equal(train_count, observed.valid.sum(axis=0))
    morning = set(summary['partition']['train'])
    afternoon = set(summary['partition']['held'])
    if morning & afternoon or morning | afternoon != set(observed.keys):
        raise ValueError('Saved session split differs from the available training frames')
    morning_frames = np.asarray([key in morning for key in observed.keys])
    np.testing.assert_array_equal(morning_count, observed.valid[morning_frames].sum(axis=0))
    afternoon_support = observed.valid[~morning_frames].any(axis=0)
    afternoon_paired = afternoon_support & (morning_count > 0)
    reference_paired = reference_valid & (train_count > 0)
    print(f'Computing per-frame curvature ranges from {len(observed.keys)} calibration frames.', flush=True)
    measured, measured_valid = frame_curvatures(settings, surface, observed.valid, observed.keys, half_width)
    span_count, lo, hi, diameter = envelopes(measured, measured_valid)

    lut = load_normal_model(RUN/'3b2/reference/table.npz')
    data, _, uv = matched_inputs(settings.reference_image, half_width, lut, background_file=model_path)
    if data['coordinate_chart_id'] != chart_id:
        raise ValueError('Training curvature and 3b2 use different material coordinate charts')
    with np.load(settings.work_dir/'reference_surface.npz', allow_pickle=False) as ref:
        template = ref['template_xy_mm']
    sampled = compute_surface_curvature(data['xyz'], data['geometry_valid_mask'], data['material_coordinates'],
                                        template, smoothing_half_width_mm=half_width)
    test = sampled.values
    query = data['st'] * [st.shape[1]-1, st.shape[0]-1]
    trained = bilinear_sample(reference, query, train_count > 0, allow_boundary=True)
    paired = data['photometric_valid_mask'] & sampled.valid & trained.valid
    delta = test - trained.values
    distance = np.sqrt(tensor_squared(delta))
    standardized = np.linalg.norm(delta / model.curvature_scale, axis=-1)
    near_count, near_lo, near_hi, nearest = compare_to_frame_set(measured, measured_valid, query, test)
    can_compare = paired & (near_count >= 2)
    out_of_range = ((test < near_lo) | (test > near_hi)).any(axis=-1) & can_compare

    report = dict(
        feature_components=COMPONENTS, units='mm^-1', training_frames=len(observed.keys),
        training_margin_px=margin, smoothing_half_width_mm=half_width,
        actual_training=dict(points=int((train_count > 0).sum()), per_point_span=[0., 0., 0.],
                             overall_min=reference[train_count > 0].min(axis=0),
                             overall_max=reference[train_count > 0].max(axis=0),
                             standardization_scale=model.curvature_scale),
        fixed_curvature_evaluation=dict(morning_frames=len(morning), afternoon_frames=len(afternoon),
                                       morning_points=int((morning_count > 0).sum()),
                                       afternoon_points=int(afternoon_support.sum()),
                                       afternoon_paired_points=int(afternoon_paired.sum()),
                                       reference_points=int(reference_valid.sum()),
                                       reference_paired_points=int(reference_paired.sum()),
                                       paired_curvature_distance=0.),
        fitted_frames=dict(valid_points=int((span_count > 0).sum()),
                           span_per_component=distribution((hi-lo)[span_count >= 2]),
                           tensor_diameter=distribution(diameter[span_count >= 2])),
        test_3b2=dict(photometric_points=int(data['photometric_valid_mask'].sum()),
                      same_position_distance=distribution(distance[paired]),
                      same_position_standardized_distance=distribution(standardized[paired]),
                      same_position_absolute_component_error=distribution(np.abs(delta[paired])),
                      fitted_frame_coverage_points=int(can_compare.sum()),
                      fitted_frame_envelope_outside_points=int(out_of_range.sum()),
                      fitted_frame_envelope_outside_fraction=float(out_of_range[can_compare].mean()),
                      nearest_fitted_frame_distance=distribution(nearest[can_compare]),
                      right_same_position_distance=distribution(distance[paired & (uv[..., 0] >= 445.)])),
    )
    span_mask = train_count > 0
    np.savez_compressed(output/'curvature_coverage.npz', training_st=st, training_counts=train_count,
        morning_training_counts=morning_count, afternoon_holdout_support=afternoon_support,
        reference_support=reference_valid,
        training_curvature_per_mm=np.where(span_mask[..., None], reference, np.nan),
        training_span_per_mm=np.where(span_mask[..., None], np.zeros_like(reference), np.nan),
        fitted_frame_counts=span_count, fitted_frame_min_per_mm=lo, fitted_frame_max_per_mm=hi,
        fitted_frame_diameter_per_mm=diameter, test_st=data['st'], test_uv=uv, test_paired=paired,
        test_curvature_per_mm=np.where(sampled.valid[..., None], test, np.nan),
        test_minus_training_per_mm=np.where(paired[..., None], delta, np.nan),
        test_training_distance_per_mm=np.where(paired, distance, np.nan),
        test_training_standardized_distance=np.where(paired, standardized, np.nan),
        test_fitted_frame_counts=near_count, test_fitted_frame_min_per_mm=near_lo, test_fitted_frame_max_per_mm=near_hi,
        test_outside_fitted_frame_envelope=out_of_range,
        test_nearest_fitted_frame_per_mm=np.where(can_compare, nearest, np.nan))
    plot(output/'curvature_coverage.png', diameter, np.where(paired[..., None], delta, np.nan),
         np.where(paired, distance, np.nan), np.where(can_compare, nearest, np.nan))
    (output/'curvature_coverage.json').write_text(json.dumps(report, indent=2, allow_nan=False,
        default=lambda value: value.tolist())+'\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE/'runs/curvature_coverage')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True)
    run(args.output_dir)
    print(f'Curvature coverage saved to {args.output_dir}.')


if __name__ == '__main__':
    main()
