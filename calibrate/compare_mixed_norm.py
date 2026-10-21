"""Evaluate pooling deep/shallow calibration, using the existing paired holdouts.

Run after calibrate.compare_norm. Outputs stay in its comparison directory;
the reconstruction calibration and its configuration are never replaced.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import yaml

from calibrate.compare_norm import (
    SAMPLE_FIELDS, TILT_BINS, background_metrics, color_support, dataset_summary,
    fit_dataset, leave_one_image_out, near_color_consistency, noise_response,
    save_csv, save_figure, score_samples, source_signature, summarize_common_tests,
)
from calibrate.norm_pipeline import parse_settings, run_prepare, run_samples
from utils.normal_mapping import NormalCalibration, build_normal_lut, lookup_slopes_raw


def prepare_mixed(config, base, directories, work):
    """Rebuild all images against one reference chart, retaining their annotations."""
    work.mkdir(parents=True, exist_ok=True)
    image_root = Path(os.path.commonpath(list(directories.values())))
    annotations = yaml.safe_load(base.annotations_file.read_text())
    images = sorted(image for directory in directories.values() for image in directory.glob('*.png'))
    rows = []
    for image in images:
        matches = [row for row in annotations['regions'] if Path(row['image']).name == image.name]
        if len(matches) != 1 or matches[0]['status'] != 'accepted':
            raise ValueError(f'Expected one accepted annotation for {image}')
        rows.append(dict(matches[0], image=image.relative_to(image_root).as_posix()))
    annotation_file = work/'manual_regions.yaml'
    annotation_file.write_text(yaml.safe_dump(dict(version=2,
        image_root=os.path.relpath(image_root, work), regions=rows), sort_keys=False, allow_unicode=True))
    settings = replace(base, images=tuple(str(directory/'*.png') for directory in directories.values()),
        image_root=image_root, annotations_file=annotation_file, work_dir=work,
        input_file=work/'normal_samples.npz', output=work/'normal_lut.npz')
    manifest = run_prepare(config, settings)
    if any(row['status'] != 'prepared' for row in manifest['images']):
        raise ValueError('Mixed preparation rejected images; inspect its manifest')
    records = run_samples(settings)
    if len(records) != len(images):
        raise ValueError('Mixed preparation lost images; no silent exclusion allowed')
    return settings


def check_same_samples(mixed, datasets):
    """Only the shared chart UUID and image keys may differ after preparation."""
    for name, samples in datasets.items():
        for group, image in enumerate(samples['image_paths']):
            key = f'cali_norm_pics({name})/{image}'
            mixed_ids = np.flatnonzero(mixed['image_paths'] == key)
            if len(mixed_ids) != 1:
                raise ValueError(f'Missing or duplicate mixed image: {key}')
            mask = mixed['image_index'] == mixed_ids[0]
            original = samples['image_index'] == group
            for field in SAMPLE_FIELDS[:4]:
                np.testing.assert_array_equal(mixed[field][mask], samples[field][original], err_msg=key)


def repeated_mixed_tests(datasets, baseline, options):
    """Use the baseline's exact test images; fit 4+4 and 8+8 image pools."""
    rows = deepcopy(baseline['results'])
    for row in rows:
        row['model'] += '8'
        domain = row['model'][:-1]
        row['training_images'] = [f'{domain}/{image}' for image in row['training_images']]
        row['test_images'] = [f"{row['target']}/{image}" for image in row['test_images']]
    seeds = sorted({row['seed'] for row in rows})
    for seed in seeds:
        train_keys, test_keys = {}, {}
        for name in datasets:
            row = next(row for row in rows if row['seed'] == seed and row['model'] == name+'8' and row['target'] == name)
            train_keys[name] = [key.split('/', 1)[1] for key in row['training_images']]
            test_keys[name] = [key.split('/', 1)[1] for key in row['test_images']]
            if set(train_keys[name]) & set(test_keys[name]):
                raise ValueError('Training/test image leakage')
        rng = np.random.default_rng(1000+seed)
        balanced_keys = {name: rng.permutation(keys)[:4].tolist() for name, keys in train_keys.items()}
        for model_name, chosen in (('mixed8', balanced_keys), ('mixed16', train_keys)):
            training = {}
            for name, samples in datasets.items():
                ids = np.flatnonzero(np.isin(samples['image_paths'], chosen[name]))
                training[name] = np.isin(samples['image_index'], ids)
            colors = np.concatenate([samples['color_residual_samples'][training[name]] for name, samples in datasets.items()])
            slopes = np.concatenate([samples['slope_samples'][training[name]] for name, samples in datasets.items()])
            model = build_normal_lut(colors, slopes, **options)
            for name, samples in datasets.items():
                ids = np.flatnonzero(np.isin(samples['image_paths'], test_keys[name]))
                selected = np.isin(samples['image_index'], ids)
                target_colors = samples['color_residual_samples'][selected]
                metrics, angles = score_samples(lookup_slopes_raw(target_colors, model)[0], samples, selected)
                support, _ = color_support(target_colors, model, colors)
                tilt = np.degrees(np.arctan(np.linalg.norm(samples['slope_samples'][selected], axis=-1)))
                bins = []
                for low, high in TILT_BINS:
                    in_bin = (tilt >= low) & (tilt < high)
                    if in_bin.any():
                        bins.append(dict(tilt_range=[low, high], count=int(in_bin.sum()),
                            squared_error_sum=float(np.sum(angles[in_bin]**2))))
                rows.append(dict(seed=seed, model=model_name, target=name, metrics=metrics, support=support, tilt_bins=bins,
                    training_sample_count=len(colors),
                    training_images=[f'{domain}/{image}' for domain, keys in chosen.items() for image in keys],
                    test_images=[f'{name}/{image}' for image in test_keys[name]]))
        print(f'Mixed common test {seed+1}/{len(seeds)} complete', flush=True)
    models = ['deep8', 'shallow8', 'mixed8', 'mixed16']
    return dict(rounds=len(seeds), training_images=dict(deep8=8, shallow8=8, mixed8=8, mixed16=16),
        summary=summarize_common_tests(rows, models, list(datasets)), results=rows)


def mixed_figures(output, common):
    names = list(common['summary'])
    figure = Figure(figsize=(15, 5), layout='constrained')
    axes = figure.subplots(1, 3)
    matrix = np.array([[common['summary'][name][domain]['rms_mean'] for domain in ('deep', 'shallow')] for name in names])
    artist = axes[0].imshow(matrix, cmap='viridis_r', vmin=0, vmax=22)
    axes[0].set(xticks=range(2), xticklabels=['deep', 'shallow'], yticks=range(len(names)),
        yticklabels=names, title='Same held-out images | 10 rounds', xlabel='Target dataset', ylabel='Training pool')
    for index in np.ndindex(matrix.shape):
        axes[0].text(index[1], index[0], f'{matrix[index]:.2f}', ha='center', va='center', color='black')
    figure.colorbar(artist, ax=axes[0], label='RMS normal error (degrees)')
    axes[1].bar(names, [common['summary'][name]['equal_domain_rms']['mean'] for name in names],
        yerr=[common['summary'][name]['equal_domain_rms']['std'] for name in names], capsize=4)
    axes[1].set(title='Deep/shallow weighted equally', ylabel='RMS normal error (degrees)')
    for name in names:
        rows = [row for row in common['summary'][name]['common_target_tilt_bins'] if row['repeated_test_sample_count'] >= 100]
        axes[2].plot([sum(row['tilt_range'])/2 for row in rows], [row['rms_degrees'] for row in rows], 'o-', label=name)
    axes[2].set(title='Same test points, grouped by true tilt', xlabel='Target tilt midpoint (degrees)', ylabel='RMS normal error (degrees)')
    axes[2].grid(alpha=.2)
    axes[2].legend()
    save_figure(figure, output/'mixed_common_tests.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output', default='assets/normal_calibration_comparison')
    parser.add_argument('--reuse-samples', action='store_true')
    args = parser.parse_args()
    config = Path(args.config).resolve()
    output = Path(args.output).resolve()
    base = parse_settings(config)
    directories = {name: config.parent/f'assets/cali_norm_pics({name})' for name in ('deep', 'shallow')}
    signature = source_signature(config, directories, base)
    if signature != json.loads((output/'source_hashes.json').read_text()):
        raise ValueError('Baseline source data/config changed; rerun calibrate.compare_norm first')
    protected = [config, base.annotations_file, base.reference_image, base.output, base.work_dir/'lut_material_grid.npz']
    protected += [output/'comparison.json']
    protected += [output/name/'normal_lut.npz' for name in directories]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    baseline = json.loads((output/'comparison.json').read_text())
    datasets, models, settings = {}, {}, {}
    for name in directories:
        work = output/name
        settings[name] = replace(base, work_dir=work, annotations_file=work/'manual_regions.yaml',
            input_file=work/'normal_samples.npz', output=work/'normal_lut.npz')
        models[name] = NormalCalibration.load(settings[name].output)
        with np.load(settings[name].input_file, allow_pickle=False) as data:
            datasets[name] = {field: data[field] for field in SAMPLE_FIELDS}
    work = output/'mixed'
    if args.reuse_samples:
        if json.loads((work/'source_hashes.json').read_text()) != signature:
            raise ValueError('Mixed source data/config changed; regenerate samples')
        mixed_settings = replace(base, work_dir=work, annotations_file=work/'manual_regions.yaml',
            input_file=work/'normal_samples.npz', output=work/'normal_lut.npz')
    else:
        mixed_settings = prepare_mixed(config, base, directories, work)
        (work/'source_hashes.json').write_text(json.dumps(signature, indent=2)+'\n')
    with np.load(work/'reference_coordinates.npz') as mixed_chart:
        for name in directories:
            with np.load(output/name/'reference_coordinates.npz') as chart:
                for field in ('st_table', 'arc_lengths_mm', 'a_axis', 'b_axis'):
                    np.testing.assert_array_equal(mixed_chart[field], chart[field])
    model, samples = fit_dataset(mixed_settings)
    check_same_samples(samples, datasets)
    print('Mixed samples exactly match both independently prepared datasets', flush=True)
    loio, prediction, angles, supports = leave_one_image_out('mixed', samples, base.lut_options())
    summary = dataset_summary(samples, model, loio)
    by_domain = {}
    for name in datasets:
        selected_images = np.flatnonzero(np.char.startswith(samples['image_paths'], f'cali_norm_pics({name})/'))
        selected = np.isin(samples['image_index'], selected_images)
        metrics, _ = score_samples(prediction[selected], samples, selected)
        by_domain[name] = dict(metrics=metrics, out_of_range_fraction=float(supports['outside'][selected].mean()),
            entirely_filled_corner_fraction=float(np.mean(supports['corner_weight'][selected] < 1e-8)))
    summary['loio_by_domain'] = by_domain
    consistency, pair_distance, pair_angles = near_color_consistency(samples['color_residual_samples'], samples['slope_samples'], samples['image_index'])
    summary['near_color_consistency'] = consistency
    # Pair against the other acquisition dataset specifically, rather than any other image.
    domains = np.char.startswith(samples['image_paths'][samples['image_index']], 'cali_norm_pics(shallow)/')
    summary['cross_dataset_color_consistency'] = near_color_consistency(samples['color_residual_samples'], samples['slope_samples'], domains)[0]
    models['mixed'] = model
    pooled = samples['color_residual_samples']
    lower = np.max([model.color_min for model in models.values()], axis=0)+.005
    upper = np.min([model.color_max for model in models.values()], axis=0)-.005
    common_colors = pooled[np.all((pooled >= lower) & (pooled <= upper), axis=-1)]
    noises = {name: noise_response(common_colors, model)[0] for name, model in models.items()}
    background = {domain: {name: background_metrics(settings[domain], model) for name, model in models.items()} for domain in datasets}
    common = repeated_mixed_tests(datasets, baseline['common_tests'], base.lut_options())
    report = dict(lut_options=base.lut_options(), mixed=summary, common_tests=common,
        common_noise_color_count=len(common_colors), common_color_noise_sensitivity=noises, background=background,
        definitions=dict(
            mixed='All currently present deep/shallow images and existing annotations; no resampling or special weighting.',
            fixed_size='mixed8 uses four deep plus four shallow training images; deep8/shallow8 retain their previous eight-image splits. All use the exact same two deep and three shallow held-out images per round. Mixed subset RNG seed = 1000 + round seed.',
            extra_data='mixed16 uses all eight remaining training images per dataset on those same held-out targets. Its improvement combines complementarity and additional data.',
            loio='Mixed LOIO trains on twenty images and tests the twenty-first. Baseline LOIO trained on nine/ten, so use fixed_size for training-count-controlled comparison.',
            equal_domain_rms='Per round sqrt((deep_RMS^2 + shallow_RMS^2)/2), then mean/std across rounds.',
            color_consistency='Directed nearest-colour pairs, absolute signed linear RGB L2. Compare canonical normalize([-p,-q,1]) label normals; cross_dataset pairs always use the other acquisition dataset.',
            limitations=baseline['definitions']['labels']),
        protected_file_hashes={str(path): digest for path, digest in before.items()})
    (output/'mixed_comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    save_csv(output/'mixed_common_test_rounds.csv', [dict(seed=row['seed'], model=row['model'], target=row['target'],
        rms_degrees=row['metrics']['rms_degrees'], out_of_range_fraction=row['support']['out_of_range_fraction']) for row in common['results']])
    np.savez_compressed(output/'mixed_evaluation.npz', prediction=prediction, angles=angles, pair_distance=pair_distance, pair_angles=pair_angles, **supports)
    mixed_figures(output, common)
    for path, digest in before.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Unexpected modification outside mixed comparison outputs: {path}')
    print(f'Mixed comparison complete: {output}/mixed_comparison.json', flush=True)


if __name__ == '__main__':
    main()
