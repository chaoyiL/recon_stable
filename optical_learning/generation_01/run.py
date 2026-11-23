"""Run the fixed-surface experiment from the repository root."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import jax
import numpy as np

from calibrate.utils.norm_pipeline import CONTACT_OUTER_MARGIN_PX, parse_settings
from .data import SESSION_GAP_SECONDS, load_fixed_surface, load_observations, sample_reference
from .evaluate import background_statistics, color_metrics, count_metrics, save_comparison
from .model import WIDTHS, TrainingOptions, fit_background


GENERATION = Path(__file__).resolve().parent
ROOT = GENERATION.parents[1]


def run(output: Path) -> dict:
    settings = parse_settings(ROOT / 'config.yaml')
    surface = load_fixed_surface(settings)
    observations = load_observations(settings, ROOT / 'assets' / 'cali_norm_pics', surface)
    options = TrainingOptions()
    held = observations.sessions == observations.sessions.max()
    partition = dict(train=[key for key, selected in zip(observations.keys, ~held) if selected],
                     held=[key for key, selected in zip(observations.keys, held) if selected])
    train = background_statistics(observations.rgb[~held], observations.valid[~held])
    all_frames = background_statistics(observations.rgb, observations.valid)
    features = surface.features
    print(f'Sampled {len(observations.keys)} valid frames. Training on {len(partition["train"])} frames; '
          f'holding out {len(partition["held"])} frames.', flush=True)

    start = perf_counter()
    model = fit_background(features, train.mean, train.counts, options)
    train_time = perf_counter() - start
    held_prediction = model.predict(features)
    held_rgb = observations.rgb[held]
    held_valid = observations.valid[held]
    held_common = held_valid & (train.counts > 0)
    if not held_common.any():
        raise ValueError('The held-out session and training sessions have no common valid points')
    held_unsupported = held_valid & (train.counts == 0)
    heldout_metrics = dict(
        mlp=color_metrics((held_prediction - held_rgb)[held_common]),
        point_mean=color_metrics((train.mean - held_rgb)[held_common]),
        mlp_unseen=color_metrics((held_prediction - held_rgb)[held_unsupported]),
    )
    print(f'Held-out RGB RMSE is {heldout_metrics["mlp"]["rgb_rmse"]:.6f}. Refitting on all frames.',
          flush=True)

    start = perf_counter()
    final_model = fit_background(features, all_frames.mean, all_frames.counts, options)
    final_time = perf_counter() - start
    prediction = np.where(surface.valid[..., None], final_model.predict(features), 0)
    reference = sample_reference(settings, surface)
    if not reference.valid.any():
        raise ValueError('The measured reference has no valid evaluation points')
    common = reference.valid & (all_frames.counts > 0)
    unseen = reference.valid & (all_frames.counts == 0)
    reference_metrics = dict(
        mlp=color_metrics((prediction - reference.values)[common]),
        point_mean=color_metrics((all_frames.mean - reference.values)[common]),
        mlp_full_valid=color_metrics((prediction - reference.values)[reference.valid]),
        mlp_unseen=color_metrics((prediction - reference.values)[unseen]),
    )

    domain_inputs = features[surface.valid]
    final_model.predict(domain_inputs)
    timings = []
    for _ in range(30):
        start = perf_counter()
        final_model.predict(domain_inputs)
        timings.append((perf_counter() - start)*1000)

    summary = dict(
        model=dict(widths=WIDTHS, parameters=sum(w.size + b.size for w, b in final_model.layers)),
        training=asdict(options), device=str(jax.devices()[0]),
        versions=dict(jax=jax.__version__, numpy=np.__version__),
        chart_id=surface.chart_id, grid_shape=list(surface.valid.shape),
        contact_exclusion_margin_px=CONTACT_OUTER_MARGIN_PX, session_gap_seconds=SESSION_GAP_SECONDS,
        reference_image=str(settings.reference_image.relative_to(ROOT)),
        sessions=dict(Counter(observations.sessions.tolist())), partition=partition,
        skipped=observations.skipped,
        observations=dict(train=int(train.counts.sum()), held=int(held_valid.sum()),
                          all=int(all_frames.counts.sum())),
        coverage=dict(train=count_metrics(train, surface.valid),
                      all=count_metrics(all_frames, surface.valid)),
        holdout=heldout_metrics, reference=reference_metrics,
        fit_seconds=dict(train=train_time, all=final_time),
        predict_ms=dict(points=len(domain_inputs), median=float(np.median(timings)),
                        p95=float(np.quantile(timings, .95))),
    )
    arrays = dict(st=surface.st, curvature_per_mm=features[..., 2:], geometry_valid=surface.valid,
                  reference_rgb=reference.values, reference_valid=reference.valid, prediction=prediction,
                  mean_rgb=all_frames.mean, repeat_std=all_frames.standard_deviation,
                  all_counts=all_frames.counts, train_counts=train.counts)
    final_model.save(output / 'model.npz', surface.chart_id)
    np.savez_compressed(output / 'evaluation_grids.npz', **arrays)
    save_comparison(output / 'background_comparison.png', arrays)
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                     allow_nan=False) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='initial', help='Name of a new run directory')
    args = parser.parse_args()
    if args.run_name in {'', '.', '..'} or Path(args.run_name).name != args.run_name:
        parser.error('--run-name must be a single directory name')
    runs = GENERATION / 'runs'
    runs.mkdir(exist_ok=True)
    destination = runs / args.run_name
    if destination.exists():
        parser.error(f'Run already exists: {destination.relative_to(ROOT)}')
    with TemporaryDirectory(dir=runs) as temporary:
        staging = Path(temporary) / 'result'
        staging.mkdir()
        summary = run(staging)
        staging.rename(destination)

    print(f'Reference RGB RMSE is {summary["reference"]["mlp_full_valid"]["rgb_rmse"]:.6f}. '
          f'Saved {destination.relative_to(ROOT)}.', flush=True)


if __name__ == '__main__':
    main()
