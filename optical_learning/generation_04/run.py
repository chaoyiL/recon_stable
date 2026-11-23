"""Train on paired frame geometry and RGB, with a fixed-curvature control."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from calibrate.utils.norm_pipeline import CalibrationSettings, parse_settings
from optical_learning.generation_02.data import (
    FixedSurface, Observations, load_fixed_surface, load_observations, sample_reference,
)
from optical_learning.generation_02.evaluate import background_statistics, color_metrics, save_comparison
from optical_learning.generation_02.model import FOURIER_FREQUENCIES, WIDTHS, TrainingOptions, fit_background
from optical_learning.generation_03.curvature_coverage import frame_curvatures
from reconstruction.pipeline import DEFAULT_CONFIG_PATH as CONFIG
from utils.config import get_curvature_smoothing_half_width_mm, load_config


HERE = Path(__file__).resolve().parent
ROOT = CONFIG.parent
OPTIONS = TrainingOptions(epochs=12)


def load_curvature_observations(
    settings: CalibrationSettings, surface: FixedSurface, half_width: float, *, margin_px: int | None = None,
) -> tuple[Observations, np.ndarray, np.ndarray]:
    """Load RGB and curvature on the same frame and material indices, with their joint validity mask."""
    observed = load_observations(settings, ROOT / 'assets/cali_norm_pics', surface,
                                 contact_exclusion_margin_px=margin_px)
    curvature, valid = frame_curvatures(settings, surface, observed.valid, observed.keys, half_width)
    if not valid.any(axis=(1, 2)).all():
        raise ValueError('Each accepted calibration frame needs paired RGB and curvature')
    return observed, curvature, valid


def fit_samples(features: np.ndarray, rgb: np.ndarray, half_width: float):
    # Distinct frame observations retain their own targets and enter MSE with unit weight.
    return fit_background(features, rgb, np.ones(len(rgb), dtype=np.uint8), OPTIONS,
                          curvature_smoothing_half_width_mm=half_width)


def train_backgrounds(output: Path) -> dict:
    settings = parse_settings(CONFIG)
    half_width = get_curvature_smoothing_half_width_mm(load_config(CONFIG))
    prediction_options = dict(curvature_smoothing_half_width_mm=half_width)
    surface = load_fixed_surface(settings, half_width)
    observed, curvature, valid = load_curvature_observations(settings, surface, half_width)
    frames, rows, columns = np.nonzero(valid)
    fixed_inputs = surface.features[rows, columns]
    actual_inputs = np.concatenate((fixed_inputs[:, :2], curvature[valid]), axis=-1)
    rgb = observed.rgb[valid]
    held_frames = observed.sessions == observed.sessions.max()
    held = held_frames[frames]

    reference = sample_reference(settings, surface)
    all_stats = background_statistics(observed.rgb, valid)
    train_counts = valid[~held_frames].sum(axis=0)
    held_at_trained_points = train_counts[rows[held], columns[held]] > 0
    lower = np.min(np.where(valid[..., None], curvature, np.inf), axis=0)
    upper = np.max(np.where(valid[..., None], curvature, -np.inf), axis=0)
    repeated = all_stats.counts >= 2
    span = upper[repeated] - lower[repeated]
    if not (span > 0).any():
        raise ValueError('Training samples must contain distinct curvatures at repeated material points')

    report = dict(
        model=dict(widths=WIDTHS, fourier_frequencies=FOURIER_FREQUENCIES), training=asdict(OPTIONS),
        **prediction_options, contact_exclusion_margin_px=settings.background_contact_exclusion_margin_px,
        chart_id=surface.chart_id,
        partition=dict(train=[key for key, is_held in zip(observed.keys, held_frames) if not is_held],
                       held=[key for key, is_held in zip(observed.keys, held_frames) if is_held]),
        observations=dict(train=int((~held).sum()), held=int(held.sum()), all=len(rgb),
                          excluded_invalid_frame_curvature=int(observed.valid.sum()) - len(rgb)),
        material_points=int(np.count_nonzero(all_stats.counts)), repeated_material_points=int(repeated.sum()),
        curvature_span_median_per_mm=np.median(span, axis=0).tolist(),
        curvature_span_p95_per_mm=np.percentile(span, 95, axis=0).tolist(), backgrounds={},
    )
    for name, inputs in (('per_frame', actual_inputs), ('fixed_control', fixed_inputs)):
        directory = output / name
        directory.mkdir()
        start = perf_counter()
        print(f'{name}: training on {int((~held).sum()):,} pairs for session holdout.', flush=True)
        session_model = fit_samples(inputs[~held], rgb[~held], half_width)
        train_seconds = perf_counter() - start
        actual_error = session_model.predict(actual_inputs[held], **prediction_options) - rgb[held]
        fixed_error = session_model.predict(fixed_inputs[held], **prediction_options) - rgb[held]
        holdout_actual = color_metrics(actual_error)
        holdout_fixed = color_metrics(fixed_error)
        print(f'{name}: held-out actual-curvature RMSE {holdout_actual["rgb_rmse"]:.6f}. '
              f'Refitting on all {len(rgb):,} pairs.', flush=True)
        start = perf_counter()
        model = fit_samples(inputs, rgb, half_width)
        all_seconds = perf_counter() - start
        session_model.save(directory / 'session_model.npz', surface.chart_id)
        model.save(directory / 'model.npz', surface.chart_id)
        prediction = model.predict(surface.features, **prediction_options)
        reference_error = color_metrics((prediction - reference.values)[reference.valid])
        report['backgrounds'][name] = dict(
            holdout_actual=holdout_actual, holdout_fixed=holdout_fixed,
            holdout_actual_at_trained_points=color_metrics(actual_error[held_at_trained_points]),
            reference=reference_error, fit_seconds=dict(train=train_seconds, all=all_seconds),
            curvature_mean_per_mm=model.curvature_mean.tolist(), curvature_scale_per_mm=model.curvature_scale.tolist(),
        )
        save_comparison(directory / 'reference.png', dict(
            reference_valid=reference.valid, geometry_valid=surface.valid, all_counts=all_stats.counts,
            reference_rgb=reference.values, prediction=prediction, mean_rgb=all_stats.mean,
            repeat_std=all_stats.standard_deviation,
        ))
    (output / 'training.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def run(output: Path) -> None:
    from optical_learning.lut_background_3b2.experiment import run as compare_backgrounds
    from optical_learning.lut_background_3b2.table.run import fit_models

    train_backgrounds(output)
    for name in ('per_frame', 'fixed_control'):
        directory = output / name
        compare_backgrounds(directory / '3b2', 'table', fit_models, background_file=directory / 'model.npz')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='paired_curvature')
    args = parser.parse_args()
    if args.run_name in {'', '.', '..'} or Path(args.run_name).name != args.run_name:
        parser.error('--run-name must be a single directory name')
    output = HERE / 'runs' / args.run_name
    output.mkdir(parents=True)
    run(output)
    print(f'Saved {output.relative_to(ROOT)}.', flush=True)


if __name__ == '__main__':
    main()
