"""Calibrate deep/shallow image sets and compare their inverse photometric LUTs.

Run from the repository root: uv run python -u -m calibrate.compare_norm
All outputs are isolated from the reconstruction calibration in config.yaml.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy.spatial import cKDTree
import yaml

from calibrate.norm_pipeline import (
    _LUTValidationFailure, fit_and_validate_samples, parse_settings, run_prepare, run_samples,
)
from calibrate.visualize_norm import generate_calibration_visualizations
from utils.normal_mapping import NormalCalibration, build_normal_lut, lookup_slopes_raw
from utils.region_annotations import load_regions, region_masks
from utils.surface_geometry import local_slopes_to_normals
from utils.surface_sampling import sample_mask_to_surface


SAMPLE_FIELDS = ('color_residual_samples', 'slope_samples', 'reference_normal_samples',
                 'tangent_frame_samples', 'image_index', 'image_paths')
TILT_BINS = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90))


def distribution(values):
    values = np.asarray(values, float)
    if not values.size:
        return dict(count=0, mean=None, median=None, p95=None, minimum=None, maximum=None)
    return dict(count=int(values.size), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), minimum=float(values.min()), maximum=float(values.max()))


def local_normals(slopes):
    return local_slopes_to_normals(slopes, np.array([0., 0., 1.]), np.eye(3)[:, :2])[0]


def normal_angles(first, second):
    return np.degrees(np.arccos(np.clip(np.sum(first*second, axis=-1), -1, 1)))


def score_slopes(predicted, truth, reference_normals, tangent_frames):
    pn, vp = local_slopes_to_normals(predicted, reference_normals, tangent_frames)
    tn, vt = local_slopes_to_normals(truth, reference_normals, tangent_frames)
    if not (vp & vt).all():
        raise ValueError('Invalid local normal frames in comparison samples')
    angles = normal_angles(pn, tn)
    error = np.asarray(predicted, float)-truth
    spread = np.sum((truth-truth.mean(axis=0))**2, axis=0)
    metrics = dict(angular_degrees=distribution(angles), rms_degrees=float(np.sqrt(np.mean(angles**2))),
                   fraction_over_10_degrees=float(np.mean(angles > 10)),
                   fraction_over_20_degrees=float(np.mean(angles > 20)),
                   pq_rmse=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                   pq_r2=[float(1-np.sum(error[:, i]**2)/spread[i]) if spread[i] > 1e-12 else None
                          for i in range(2)])
    return metrics, angles


def score_samples(predicted, samples, mask):
    return score_slopes(predicted, samples['slope_samples'][mask],
                        samples['reference_normal_samples'][mask], samples['tangent_frame_samples'][mask])


def color_support(colors, calibration, training_colors):
    """Absolute RGB distances permit comparison across differently scaled LUTs."""
    outside = ((colors < calibration.color_min) | (colors > calibration.color_max)).any(axis=-1)
    span = calibration.color_max-calibration.color_min
    coordinate = np.clip((colors-calibration.color_min)/span, 0, 1)*(calibration.size-1)
    lower = np.floor(coordinate).astype(int)
    upper = np.minimum(lower+1, calibration.size-1)
    fraction = coordinate-lower
    weight = np.zeros(len(colors))
    for red in (0, 1):
        for green in (0, 1):
            for blue in (0, 1):
                corner = np.where([red, green, blue], upper, lower)
                factor = np.prod(np.where([red, green, blue], fraction, 1-fraction), axis=-1)
                weight += factor*calibration.original_valid[tuple(corner.T)]
    # A clipped query does not have direct measured support at its actual colour.
    weight[outside] = 0
    valid_coordinates = np.argwhere(calibration.original_valid)
    node_colors = calibration.color_min+valid_coordinates/(calibration.size-1)*span
    nearest_sample = cKDTree(training_colors).query(colors)[0]
    nearest_node = cKDTree(node_colors).query(colors)[0]
    metrics = dict(out_of_range_fraction=float(outside.mean()),
                   original_corner_weight=distribution(weight),
                   entirely_filled_corner_fraction=float(np.mean(weight < 1e-8)),
                   nearest_training_color_rgb_distance=distribution(nearest_sample),
                   nearest_valid_node_rgb_distance=distribution(nearest_node))
    return metrics, dict(outside=outside, corner_weight=weight,
                         nearest_sample=nearest_sample, nearest_node=nearest_node)


def near_color_consistency(colors, slopes, groups):
    """Directed nearest-colour pairs always come from different source images."""
    distance = np.empty(len(colors))
    neighbor_slopes = np.empty_like(slopes)
    for group in np.unique(groups):
        selected = groups == group
        other = np.flatnonzero(~selected)
        distances, ids = cKDTree(colors[other]).query(colors[selected])
        distance[selected] = distances
        neighbor_slopes[selected] = slopes[other[ids]]
    angles = normal_angles(local_normals(slopes), local_normals(neighbor_slopes))
    thresholds = []
    for limit in (.005, .01, .02):
        selected = distance <= limit
        thresholds.append(dict(rgb_distance_limit=limit, fraction_of_samples=float(selected.mean()),
            label_angle_degrees=distribution(angles[selected]),
            fraction_over_20_degrees=(float(np.mean(angles[selected] > 20)) if selected.any() else None)))
    return dict(nearest_cross_image_rgb_distance=distribution(distance), thresholds=thresholds), distance, angles


def noise_response(colors, calibration, delta=.005):
    center = local_normals(lookup_slopes_raw(colors, calibration)[0])
    responses = []
    for channel in range(3):
        for sign in (-1, 1):
            perturbed = colors.copy()
            perturbed[:, channel] += sign*delta
            predicted = local_normals(lookup_slopes_raw(perturbed, calibration)[0])
            responses.append(normal_angles(center, predicted))
    maximum = np.max(responses, axis=0)
    return dict(rgb_perturbation_per_channel=delta, worst_axis_angle_degrees=distribution(maximum)), maximum


def source_signature(config, datasets, settings):
    paths = [config, settings.annotations_file, settings.reference_image]
    # Provider geometry/camera settings are also part of reproducibility.
    from utils.config import load_config
    provider = Path(load_config(config)['get_surface']['overall_reconstruction']['config_file'])
    paths.append(provider if provider.is_absolute() else config.parent/provider)
    for directory in datasets.values():
        paths.extend(sorted(directory.glob('*.png')))
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def prepare_dataset(name, directory, output, config, base):
    work = output/name
    work.mkdir(parents=True, exist_ok=True)
    images = sorted(directory.glob('*.png'))
    annotations = yaml.safe_load(base.annotations_file.read_text())
    rows = []
    for image in images:
        matches = [row for row in annotations['regions'] if Path(row['image']).name == image.name]
        if len(matches) != 1 or matches[0]['status'] != 'accepted':
            raise ValueError(f'{name}: expected one accepted annotation for {image.name}')
        rows.append(dict(matches[0], image=image.name))
    import os
    annotation_file = work/'manual_regions.yaml'
    annotation_file.write_text(yaml.safe_dump(dict(version=2, image_root=os.path.relpath(directory, work),
                               regions=rows), sort_keys=False, allow_unicode=True))
    settings = replace(base, images=(str(directory/'*.png'),), image_root=directory,
                       annotations_file=annotation_file, work_dir=work,
                       input_file=work/'normal_samples.npz', output=work/'normal_lut.npz')
    print(f'[{name}] prepare {len(images)} images', flush=True)
    manifest = run_prepare(config, settings)
    if any(row['status'] != 'prepared' for row in manifest['images']):
        raise ValueError(f'{name}: preparation rejected images; inspect its manifest')
    records = run_samples(settings)
    if len(records) != len(images):
        raise ValueError(f'{name}: accepted {len(records)} of {len(images)} images; no silent exclusion allowed')
    del records
    return settings


def fit_dataset(settings):
    try:
        fit_and_validate_samples(settings.input_file, settings)
    except _LUTValidationFailure as exc:
        train = NormalCalibration.load(settings.output.with_name(settings.output.stem+'_train.npz'))
        with np.load(settings.input_file, allow_pickle=False) as data:
            model = build_normal_lut(data['color_residual_samples'], data['slope_samples'], **settings.lut_options())
            count = len(data['slope_samples'])
        metadata = {key: value for key, value in train.metadata.items()
                    if key not in {'training_images', 'validation_images', 'total_sample_count'}}
        metadata.update(validation_status='failed', validation=exc.report)
        replace(model, metadata=metadata).save(settings.output, total_sample_count=count)
        generate_calibration_visualizations(settings.output, settings.input_file, settings.work_dir)
    with np.load(settings.input_file, allow_pickle=False) as data:
        samples = {key: data[key] for key in SAMPLE_FIELDS}
    return NormalCalibration.load(settings.output), samples


def leave_one_image_out(name, samples, options):
    colors, slopes, groups = (samples[key] for key in ('color_residual_samples', 'slope_samples', 'image_index'))
    predicted = np.zeros_like(slopes)
    supports = {key: np.empty(len(colors)) for key in ('outside', 'corner_weight', 'nearest_sample', 'nearest_node')}
    rows = []
    for group, image in enumerate(samples['image_paths'].tolist()):
        selected = groups == group
        model = build_normal_lut(colors[~selected], slopes[~selected], **options)
        predicted[selected] = lookup_slopes_raw(colors[selected], model)[0]
        metrics, _ = score_samples(predicted[selected], samples, selected)
        support, arrays = color_support(colors[selected], model, colors[~selected])
        for key in supports:
            supports[key][selected] = arrays[key]
        rows.append(dict(dataset=name, image=image, metrics=metrics, support=support))
        print(f'[{name}] leave out {group+1}/{len(samples["image_paths"])}: RMS={metrics["rms_degrees"]:.3f}', flush=True)
    metrics, angles = score_samples(predicted, samples, np.ones(len(colors), bool))
    rms_values = np.array([row['metrics']['rms_degrees'] for row in rows])
    metrics['per_image_rms'] = dict(distribution(rms_values), std=float(rms_values.std()))
    support = dict(out_of_range_fraction=float(supports['outside'].mean()),
                   entirely_filled_corner_fraction=float(np.mean(supports['corner_weight'] < 1e-8)),
                   original_corner_weight=distribution(supports['corner_weight']),
                   nearest_training_color_rgb_distance=distribution(supports['nearest_sample']),
                   nearest_valid_node_rgb_distance=distribution(supports['nearest_node']))
    return dict(metrics=metrics, support=support, images=rows), predicted, angles, supports


def repeated_common_tests(datasets, options, rounds=10, training_images=8):
    rows = []
    for seed in range(rounds):
        rng = np.random.default_rng(seed)
        train_masks, test_masks = {}, {}
        for name, samples in datasets.items():
            selected = rng.permutation(len(samples['image_paths']))[:training_images]
            train_masks[name] = np.isin(samples['image_index'], selected)
            test_masks[name] = ~train_masks[name]
        models = {name: build_normal_lut(samples['color_residual_samples'][train_masks[name]],
                  samples['slope_samples'][train_masks[name]], **options) for name, samples in datasets.items()}
        for model_name, model in models.items():
            for target_name, samples in datasets.items():
                selected = test_masks[target_name]
                colors = samples['color_residual_samples'][selected]
                predicted = lookup_slopes_raw(colors, model)[0]
                metrics, angles = score_samples(predicted, samples, selected)
                tilt = np.degrees(np.arctan(np.linalg.norm(samples['slope_samples'][selected], axis=-1)))
                bins = []
                for low, high in TILT_BINS:
                    in_bin = (tilt >= low) & (tilt < high)
                    if in_bin.any():
                        bins.append(dict(tilt_range=[low, high], count=int(in_bin.sum()),
                                         squared_error_sum=float(np.sum(angles[in_bin]**2))))
                support, _ = color_support(colors, model, datasets[model_name]['color_residual_samples'][train_masks[model_name]])
                rows.append(dict(seed=seed, model=model_name, target=target_name, metrics=metrics, support=support, tilt_bins=bins,
                    training_images=datasets[model_name]['image_paths'][np.unique(datasets[model_name]['image_index'][train_masks[model_name]])].tolist(),
                    test_images=samples['image_paths'][np.unique(samples['image_index'][selected])].tolist()))
        print(f'Common test round {seed+1}/{rounds}: 8 training images per model, same held-out targets', flush=True)
    summary = summarize_common_tests(rows, list(datasets), list(datasets))
    return dict(rounds=rounds, training_images_per_model=training_images, summary=summary, results=rows)


def summarize_common_tests(rows, model_names, target_names):
    """Aggregate paired image holdouts, giving each target dataset equal weight."""
    summary = {}
    seeds = sorted({row['seed'] for row in rows})
    for model_name in model_names:
        summary[model_name] = {}
        for target_name in target_names:
            selected = [row for row in rows if row['model'] == model_name and row['target'] == target_name]
            rms = np.array([row['metrics']['rms_degrees'] for row in selected])
            summary[model_name][target_name] = dict(rms_mean=float(rms.mean()), rms_std=float(rms.std()),
                out_of_range_fraction_mean=float(np.mean([row['support']['out_of_range_fraction'] for row in selected])))
        balanced = []
        for seed in seeds:
            selected = [row for row in rows if row['model'] == model_name and row['seed'] == seed]
            balanced.append(np.sqrt(np.mean([row['metrics']['rms_degrees']**2 for row in selected])))
        summary[model_name]['equal_domain_rms'] = dict(distribution(balanced), std=float(np.std(balanced)))
        common_bins = []
        for bounds in TILT_BINS:
            selected = [item for row in rows if row['model'] == model_name
                        for item in row['tilt_bins'] if tuple(item['tilt_range']) == bounds]
            count = sum(item['count'] for item in selected)
            if count:
                common_bins.append(dict(tilt_range=list(bounds), repeated_test_sample_count=count,
                    rms_degrees=float(np.sqrt(sum(item['squared_error_sum'] for item in selected)/count))))
        summary[model_name]['common_target_tilt_bins'] = common_bins
    return summary


def background_metrics(settings, model):
    manifest = json.loads((settings.work_dir/'manifest.json').read_text())
    regions = load_regions(settings.annotations_file, image_root=manifest['image_root'])
    residuals, tilts, rows = [], [], []
    for entry in manifest['images']:
        with np.load(settings.work_dir/'frames'/f"{entry['index']:06d}.npz", allow_pickle=False) as frame:
            masks = region_masks(regions[entry['key']], (480, 640), outer_margin_px=20)
            valid = frame['photometric_valid_mask'] & sample_mask_to_surface(~masks['exclusion'], frame['uv'])
            colors = frame['color_residual_grid'][valid]
            tilt = np.degrees(np.arctan(np.linalg.norm(lookup_slopes_raw(colors, model)[0], axis=-1)))
            residuals.append(np.max(np.abs(colors), axis=-1))
            tilts.append(tilt)
            rows.append(dict(image=entry['key'], fit_info=json.loads(str(frame['fit_info'])),
                             median_background_abs_color=float(np.median(residuals[-1]))))
    return dict(max_abs_rgb=distribution(np.concatenate(residuals)),
                raw_false_tilt_degrees=distribution(np.concatenate(tilts)), images=rows)


def dataset_summary(samples, model, loio):
    colors, slopes = samples['color_residual_samples'], samples['slope_samples']
    counts = model.sample_counts[model.original_valid]
    fit, _ = score_samples(lookup_slopes_raw(colors, model)[0], samples, np.ones(len(colors), bool))
    tilt = np.degrees(np.arctan(np.linalg.norm(slopes, axis=-1)))
    return dict(image_count=len(samples['image_paths']), sample_count=len(colors),
        valid_node_count=int(model.original_valid.sum()), retained_samples=int(model.sample_counts.sum()),
        sample_retention_fraction=float(model.sample_counts.sum()/len(colors)),
        singleton_node_fraction=float(np.mean(counts == 1)), sample_counts_per_node=distribution(counts),
        target_tilt_degrees=distribution(tilt), color_min=model.color_min.tolist(), color_max=model.color_max.tolist(),
        all_sample_fit=fit, default_validation=model.metadata['validation'], leave_one_image_out=loio)


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, path):
    FigureCanvasAgg(figure)
    figure.savefig(path, dpi=160)
    figure.clear()


def comparison_figures(output, datasets, report, arrays):
    figure = Figure(figsize=(15, 9), layout='constrained')
    axes = figure.subplots(2, 3)
    for name, samples in datasets.items():
        data = arrays[name]
        panels = [(data['angles'], axes[0, 0], 'Leave-one-image-out normal error (degrees)'),
                  (data['tilt'], axes[0, 1], 'Sphere-labelled target tilt (degrees)'),
                  (data['nearest_sample'], axes[0, 2], 'Nearest training colour: absolute RGB distance'),
                  (data['pair_angles'][data['pair_distance'] <= .01], axes[1, 0], 'Cross-image label disagreement | RGB distance <= 0.01'),
                  (data['noise'], axes[1, 1], 'Worst normal change for +/-0.005 in one RGB channel')]
        for values, axis, title in panels:
            values = np.sort(values)
            if len(values):
                axis.plot(values, np.arange(1, len(values)+1)/len(values), label=name)
            axis.set(title=title, ylabel='Cumulative fraction')
            axis.grid(alpha=.2)
            axis.legend()
    axes[0, 0].set_xlim(0, 60)
    axes[0, 2].set_xlim(0, .1)
    axes[1, 0].set_xlim(0, 90)
    axes[1, 1].set_xlim(0, 30)
    names = list(datasets)
    axes[1, 2].bar(names, [report['datasets'][name]['sample_retention_fraction']*100 for name in names])
    axes[1, 2].set(title='Samples retained by the all-sample LUT', ylabel='Percent')
    figure.suptitle('Deep versus shallow | identical LUT algorithm and reference | raw predictions')
    save_figure(figure, output/'comparison_overview.png')

    figure = Figure(figsize=(12, 9), layout='constrained')
    axes = figure.subplots(2, 2)
    limit = max(float(np.abs(samples['slope_samples']).max()) for samples in datasets.values())
    for row, (name, samples) in enumerate(datasets.items()):
        ids = np.linspace(0, len(samples['slope_samples'])-1, min(6000, len(samples['slope_samples'])), dtype=int)
        for component, label in enumerate(('p', 'q')):
            axis = axes[row, component]
            axis.scatter(samples['slope_samples'][ids, component], arrays[name]['prediction'][ids, component], s=2, alpha=.2)
            axis.plot([-limit, limit], [-limit, limit], 'k--', linewidth=1)
            axis.set(title=f'{name}: {label}', xlabel='Sphere-labelled truth', ylabel='LOIO prediction',
                     xlim=(-limit, limit), ylim=(-limit, limit))
    save_figure(figure, output/'pq_leave_one_out.png')

    figure = Figure(figsize=(12, 5), layout='constrained')
    axes = figure.subplots(1, 2)
    matrix = np.array([[report['common_tests']['summary'][model][target]['rms_mean'] for target in names] for model in names])
    artist = axes[0].imshow(matrix, cmap='magma', vmin=0)
    axes[0].set(xticks=range(2), xticklabels=names, yticks=range(2), yticklabels=names,
                xlabel='Held-out target dataset', ylabel='Training dataset', title='8 images/model | mean RMS over 10 rounds')
    for index in np.ndindex(matrix.shape):
        axes[0].text(index[1], index[0], f'{matrix[index]:.2f} deg', ha='center', va='center', color='#111111')
    figure.colorbar(artist, ax=axes[0], label='Normal RMS (degrees)')
    for name in names:
        rows = [row for row in report['common_tests']['summary'][name]['common_target_tilt_bins']
                if row['repeated_test_sample_count'] >= 100]
        axes[1].plot([sum(row['tilt_range'])/2 for row in rows], [row['rms_degrees'] for row in rows], 'o-', label=name)
    axes[1].set(title='Same held-out targets: error by true tilt', xlabel='Target tilt bin midpoint (degrees)', ylabel='RMS normal error (degrees)')
    axes[1].legend()
    axes[1].grid(alpha=.2)
    save_figure(figure, output/'common_tests_and_tilt.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output', default='assets/normal_calibration_comparison')
    parser.add_argument('--reuse-samples', action='store_true', help='Reuse samples only when source hashes still match')
    args = parser.parse_args()
    config = Path(args.config).resolve()
    base = parse_settings(config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    directories = {name: config.parent/f'assets/cali_norm_pics({name})' for name in ('deep', 'shallow')}
    signature = source_signature(config, directories, base)
    signature_file = output/'source_hashes.json'
    if args.reuse_samples:
        if json.loads(signature_file.read_text()) != signature:
            raise ValueError('Source data/configuration changed; regenerate samples')
    else:
        signature_file.write_text(json.dumps(signature, indent=2)+'\n')
    protected = [config, base.annotations_file, base.reference_image, base.output, base.work_dir/'lut_material_grid.npz']
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected if path.exists()}
    settings, models, datasets = {}, {}, {}
    for name, directory in directories.items():
        if args.reuse_samples:
            work = output/name
            settings[name] = replace(base, work_dir=work, annotations_file=work/'manual_regions.yaml',
                                     input_file=work/'normal_samples.npz', output=work/'normal_lut.npz')
        else:
            settings[name] = prepare_dataset(name, directory, output, config, base)
        print(f'[{name}] fit and render calibration', flush=True)
        models[name], datasets[name] = fit_dataset(settings[name])

    # Both references must contain the same numeric chart, despite distinct UUIDs.
    with np.load(output/'deep/reference_coordinates.npz') as deep, np.load(output/'shallow/reference_coordinates.npz') as shallow:
        for key in ('st_table', 'arc_lengths_mm', 'a_axis', 'b_axis'):
            np.testing.assert_array_equal(deep[key], shallow[key])
    report = dict(lut_options=base.lut_options(), reference_image=str(base.reference_image),
                  calibration_geometry=dict(sphere_radius_mm=base.sphere_radius_mm,
                      sphere_side=base.sphere_side, boundary_model=base.boundary_model),
                  source_hashes=str(signature_file), datasets={}, definitions=dict(
        normal_error='Angle between predicted and sphere-labelled normals, degrees; raw LUT without reconstruction protection.',
        loio='Leave one complete image out; every source image is evaluated independently.',
        common_tests='Ten seeded rounds. Each model trains on eight images. Both predict the same union of held-out deep and shallow images. No target image enters either training set.',
        common_target_tilt_bins='Both models evaluated at exactly the same held-out sample points, pooled across the ten rounds. Counts include repeated held-out appearances. Bins with fewer than 100 test appearances are omitted from the plot.',
        support='Euclidean distances in absolute signed linear RGB, without dataset-specific rescaling. Direct corner weights outside the color box are zero.',
        near_color='Each sample pairs with its nearest colour from another image. Disagreement compares canonical local normals normalize([-p,-q,1]). Thresholds 0.005,0.01,0.02 are absolute RGB L2 distances.',
        sensitivity='Maximum normal change under +/-0.005 perturbation of one RGB channel, on identical pooled colors inside both models ranges with a 0.005 margin.',
        labels='Sphere-derived labels depend on existing contact annotations and zero_boundary_displacement; no independent physical normal/height ground truth.',
        dataset_selection='Use exactly the images currently present in each directory; shallow excludes the image previously removed by the user.'))
    arrays, image_rows = {}, []
    for name, samples in datasets.items():
        result, prediction, angles, supports = leave_one_image_out(name, samples, base.lut_options())
        report['datasets'][name] = dataset_summary(samples, models[name], result)
        consistency, pair_distance, pair_angles = near_color_consistency(samples['color_residual_samples'], samples['slope_samples'], samples['image_index'])
        report['datasets'][name]['near_color_consistency'] = consistency
        report['datasets'][name]['background'] = background_metrics(settings[name], models[name])
        tilt = np.degrees(np.arctan(np.linalg.norm(samples['slope_samples'], axis=-1)))
        bins = []
        for low, high in TILT_BINS:
            selected = (tilt >= low) & (tilt < high)
            if selected.any():
                bins.append(dict(tilt_range=[low, high], count=int(selected.sum()),
                                 rms_degrees=float(np.sqrt(np.mean(angles[selected]**2)))))
        report['datasets'][name]['tilt_bins'] = bins
        arrays[name] = dict(prediction=prediction, angles=angles, tilt=tilt, pair_distance=pair_distance,
                            pair_angles=pair_angles, **supports)
        for row in result['images']:
            image_rows.append(dict(dataset=name, image=row['image'], count=row['metrics']['angular_degrees']['count'],
                rms_degrees=row['metrics']['rms_degrees'], p95_degrees=row['metrics']['angular_degrees']['p95'],
                out_of_range_fraction=row['support']['out_of_range_fraction'],
                nearest_training_color_p95=row['support']['nearest_training_color_rgb_distance']['p95']))
    common_bins = [bounds for bounds in TILT_BINS if all(any(tuple(row['tilt_range']) == bounds and row['count'] >= 30
                   for row in report['datasets'][name]['tilt_bins']) for name in datasets)]
    report['common_tilt_bins'] = common_bins
    for name in datasets:
        rows = [row for row in report['datasets'][name]['tilt_bins'] if tuple(row['tilt_range']) in common_bins]
        report['datasets'][name]['equal_tilt_bin_rms_degrees'] = float(np.sqrt(np.mean([row['rms_degrees']**2 for row in rows])))
    report['common_tests'] = repeated_common_tests(datasets, base.lut_options())
    pooled = np.concatenate([samples['color_residual_samples'] for samples in datasets.values()])
    lower = np.max([model.color_min for model in models.values()], axis=0)+.005
    upper = np.min([model.color_max for model in models.values()], axis=0)-.005
    common_colors = pooled[np.all((pooled >= lower) & (pooled <= upper), axis=-1)]
    for name in datasets:
        metrics, responses = noise_response(common_colors, models[name])
        report['datasets'][name]['common_color_noise_sensitivity'] = metrics
        arrays[name]['noise'] = responses
    report['common_noise_color_count'] = len(common_colors)
    save_csv(output/'per_image_leave_one_out.csv', image_rows)
    save_csv(output/'common_test_rounds.csv', [dict(seed=row['seed'], model=row['model'], target=row['target'],
             rms_degrees=row['metrics']['rms_degrees'], out_of_range_fraction=row['support']['out_of_range_fraction'])
             for row in report['common_tests']['results']])
    (output/'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    comparison_figures(output, datasets, report, arrays)
    for name, data in arrays.items():
        np.savez_compressed(output/f'{name}_evaluation.npz', **data)
    for path, digest in before.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Unexpected modification outside comparison outputs: {path}')
    print(f'Comparison complete: {output}/comparison.json', flush=True)


if __name__ == '__main__':
    main()
