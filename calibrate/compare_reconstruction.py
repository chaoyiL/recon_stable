"""Compare training-only normal models on one or more captured images."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from reconstruction.pipeline import reconstruct_image
from reconstruction.viewer import _surface_quads
from utils.calibration_store import resolve_artifacts
from utils.config import load_config
from utils.normal_models import load_normal_model
from utils.surface_geometry import surface_grid_curve_lengths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _models(comparison: dict, config_path: Path) -> dict:
    models = {}
    reports = comparison['results']
    if not isinstance(reports, dict) or not reports:
        raise ValueError('Comparison must contain at least one model result')
    for method, report in reports.items():
        path = Path(report['training_file']).expanduser().resolve()
        digest = _sha256(path)
        if digest != report['training_model_sha256']:
            raise ValueError(f'{method} training-model identity differs from comparison')
        artifacts = resolve_artifacts(config_path, model_override=path)
        if artifacts.method != method or artifacts.grid_file is None:
            raise ValueError(f'{method} model or material grid differs from comparison')
        grid_digest = _sha256(artifacts.grid_file)
        if grid_digest != report['grid_sha256']:
            raise ValueError(f'{method} material-grid identity differs from comparison')
        metadata = load_normal_model(path).metadata
        with np.load(artifacts.grid_file, allow_pickle=False) as grid:
            for field in ('coordinate_kind', 'coordinate_chart_id', 'domain_kind',
                          'frame_convention', 'residual_domain', 'color_space'):
                if str(grid[field]) != metadata.get(field):
                    raise ValueError(f'{method} model/material-grid {field} mismatch')
        models[method] = dict(file=str(path), sha256=digest,
                              grid_file=str(artifacts.grid_file), grid_sha256=grid_digest,
                              validation_status=report['status'])
    if len({model['grid_sha256'] for model in models.values()}) != 1:
        raise ValueError('Models must share one material grid')
    return models


def _manifest_records(path: Path, config_path: Path, image_files: list[Path], models: dict) -> dict:
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (Path(manifest['config_file']).resolve() != config_path
            or manifest['config_sha256'] != _sha256(config_path)):
        raise ValueError('Saved results use a different configuration')
    records = {}
    for entry in manifest['results']:
        key = (Path(entry['image_file']).resolve(), entry['method'])
        if key in records:
            raise ValueError(f'Duplicate saved result: {key}')
        records[key] = entry
    for image in image_files:
        image_digest = _sha256(image)
        for method, model in models.items():
            entry = records[(image, method)]
            output = Path(entry['output_file']).resolve()
            if (entry['image_sha256'] != image_digest
                    or Path(entry['model_file']).resolve() != Path(model['file'])
                    or entry['model_sha256'] != model['sha256']
                    or entry['output_sha256'] != _sha256(output)):
                raise ValueError(f'Saved result identity differs for {image.name} / {method}')
    return records


def _read_result(path: Path) -> dict:
    required = ('xyz_out', 'displacement', 'displacement_vectors', 'reference_normals',
                'slopes', 'valid_mask', 'boundary_mask', 'photometric_valid_mask',
                'model_valid_mask', 'prior_valid_mask', 'equation_valid_mask',
                'inferred_mask', 'solver_istop', 'solver_iterations', 'solver_residual_norm')
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in required}


def _verify_geometry(results: dict) -> None:
    remaining = iter(results.items())
    _, first = next(remaining)
    baseline = first['xyz_out'].astype(np.float64) - first['displacement_vectors']
    for method, result in remaining:
        for key in ('valid_mask', 'boundary_mask', 'photometric_valid_mask', 'reference_normals'):
            if not np.array_equal(first[key], result[key]):
                raise ValueError(f'{method} does not share the comparison geometry: {key}')
        candidate = result['xyz_out'].astype(np.float64) - result['displacement_vectors']
        if candidate.shape != baseline.shape or not np.allclose(candidate, baseline, rtol=0, atol=2e-5):
            raise ValueError(f'{method} does not share the baseline geometry')


def _display_geometry(results: dict) -> tuple[float, float, np.ndarray, np.ndarray]:
    first = next(iter(results.values()))
    baseline = first['xyz_out'].astype(np.float64) - first['displacement_vectors']
    width, height = surface_grid_curve_lengths(baseline, first['valid_mask'])
    points = np.concatenate([item['xyz_out'][item['valid_mask']] for item in results.values()])
    lower, upper = points.min(axis=0), points.max(axis=0)
    spans = np.maximum(upper-lower, max(float(np.max(upper-lower)), 1e-3)*.04)
    return width, height, (lower+upper)/2, spans


def _limit(results: dict, field: str) -> float:
    return max(max(float(np.max(np.abs(item[field][item['valid_mask']])))
                   for item in results.values()), 1e-6)


def _surface(axis, result: dict, norm: Normalize, center: np.ndarray, spans: np.ndarray) -> None:
    valid = result['valid_mask']
    xyz, displacement = result['xyz_out'], result['displacement']
    quads, values = _surface_quads(xyz, displacement, valid, maximum_samples=70)
    cmap = colormaps['coolwarm']
    if len(quads):
        axis.add_collection3d(Poly3DCollection(
            quads, facecolors=cmap(norm(values)), edgecolors='none',
            linewidths=0, antialiased=False))
    else:
        axis.scatter(*xyz[valid].T, c=displacement[valid], cmap=cmap, norm=norm, s=2)
    axis.set_xlim(center[0]-spans[0]/2, center[0]+spans[0]/2)
    axis.set_ylim(center[1]-spans[1]/2, center[1]+spans[1]/2)
    axis.set_zlim(center[2]-spans[2]/2, center[2]+spans[2]/2)
    axis.set_box_aspect(spans)
    axis.view_init(elev=30, azim=-60)
    axis.set(xlabel='X', ylabel='Y', zlabel='Z')
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)
    axis.zaxis.label.set_size(8)
    axis.tick_params(labelsize=7, pad=1)


def _heatmap(axis, values, valid, norm, ratio):
    image = axis.imshow(np.ma.array(values, mask=~valid), cmap='coolwarm', norm=norm,
                        origin='upper', interpolation='nearest', aspect='auto')
    axis.set_box_aspect(ratio)
    axis.set_facecolor('#cccccc')
    return image


def _figure(figsize):
    figure = Figure(figsize=figsize, layout='constrained')
    FigureCanvasAgg(figure)
    return figure


def _save_image_comparisons(image: Path, results: dict, output: Path, previews: Path) -> dict:
    width, height, center, spans = _display_geometry(results)
    ratio = height/width
    limit = _limit(results, 'displacement')
    norm = Normalize(-limit, limit)
    paths = {}

    figure = _figure((max(13, 3.25*len(results)), 8))
    axes = figure.subplots(1, len(results), squeeze=False)[0]
    for axis, (method, result) in zip(axes, results.items(), strict=True):
        artist = _heatmap(axis, result['displacement'], result['valid_mask'], norm, ratio)
        axis.set(title=method, xlabel='Material-grid column')
    axes[0].set_ylabel('Material-grid row')
    figure.colorbar(artist, ax=list(axes), label='Normal displacement (mm)', shrink=.85)
    figure.suptitle(f'{image.stem}: normal displacement')
    paths['displacement'] = str(output/'displacement_comparison.png')
    figure.savefig(paths['displacement'], dpi=150)
    figure.clear()

    columns = math.ceil(math.sqrt(len(results)))
    rows = math.ceil(len(results)/columns)
    figure = _figure((7*columns, 5*rows))
    axes = [figure.add_subplot(rows, columns, i+1, projection='3d') for i in range(len(results))]
    for axis, (method, result) in zip(axes, results.items(), strict=True):
        _surface(axis, result, norm, center, spans)
        axis.set_title(method)
    figure.colorbar(ScalarMappable(norm=norm, cmap='coolwarm'), ax=axes,
                    label='Normal displacement (mm)', shrink=.7)
    figure.suptitle(f'{image.stem}: reconstructed surfaces (camera XYZ, mm)')
    paths['surface_3d'] = str(output/'surface_3d_comparison.png')
    figure.savefig(paths['surface_3d'], dpi=150)
    figure.clear()

    slope_values = []
    for result in results.values():
        mask = result['equation_valid_mask'] & ~result['boundary_mask']
        slope_values.append(np.abs(result['slopes'][mask]).reshape(-1))
    slope_limit = max(float(np.percentile(np.concatenate(slope_values), 99.5)), 1e-6)
    slope_norm = Normalize(-slope_limit, slope_limit)
    figure = _figure((max(13, 3.25*len(results)), 15))
    axes = figure.subplots(2, len(results), squeeze=False)
    for row, component in enumerate(('p', 'q')):
        for column, (method, result) in enumerate(results.items()):
            mask = result['equation_valid_mask'] & ~result['boundary_mask']
            artist = _heatmap(axes[row, column], result['slopes'][..., row], mask, slope_norm, ratio)
            axes[row, column].set(title=f'{method} | {component}', xlabel='Material-grid column')
        axes[row, 0].set_ylabel('Material-grid row')
    figure.colorbar(artist, ax=list(axes.flat), label='Local slope', shrink=.85, extend='both')
    figure.suptitle(f'{image.stem}: slopes used in reconstruction')
    paths['slopes'] = str(output/'slope_field_comparison.png')
    figure.savefig(paths['slopes'], dpi=150)
    figure.clear()
    paths['displacement_limit_mm'] = limit
    paths['slope_limit'] = slope_limit
    paths['camera_elevation_degrees'] = 30
    paths['camera_azimuth_degrees'] = -60
    paths['camera_xyz_center_mm'] = center.tolist()
    paths['camera_xyz_spans_mm'] = spans.tolist()

    previews.mkdir(parents=True, exist_ok=True)
    for method, result in results.items():
        preview = previews/f'{image.stem}_{method}_local_result.png'
        figure = _figure((12, 6))
        grid = figure.add_gridspec(1, 2, width_ratios=(1.5, 1))
        surface_axis = figure.add_subplot(grid[0, 0], projection='3d')
        _surface(surface_axis, result, norm, center, spans)
        surface_axis.set_title('Reconstructed surface\nCamera XYZ (mm)')
        axis = figure.add_subplot(grid[0, 1])
        artist = _heatmap(axis, result['displacement'], result['valid_mask'], norm, ratio)
        axis.set(title='Normal displacement (mm)', xlabel='Material-grid column',
                 ylabel='Material-grid row')
        figure.colorbar(artist, ax=axis, label='mm', shrink=.75)
        figure.suptitle(f'{image.stem} | {method}')
        figure.savefig(preview, dpi=150)
        figure.clear()
    return paths


def _save_overview(images: list[Path], all_results: dict, output: Path) -> Path:
    methods = tuple(all_results[images[0].stem])
    figure = _figure((max(14, 3.5*len(methods)), max(5, 4*len(images))))
    axes = figure.subplots(len(images), len(methods), squeeze=False)
    for row, image in enumerate(images):
        results = all_results[image.stem]
        width, height, _, _ = _display_geometry(results)
        limit = _limit(results, 'displacement')
        norm = Normalize(-limit, limit)
        for column, method in enumerate(methods):
            result = results[method]
            artist = _heatmap(axes[row, column], result['displacement'], result['valid_mask'],
                              norm, height/width)
            axes[row, column].set_title(method)
            if column == 0:
                axes[row, column].set_ylabel(f'{image.stem}\nMaterial-grid row')
            if row == len(images)-1:
                axes[row, column].set_xlabel('Material-grid column')
        figure.colorbar(artist, ax=list(axes[row]), label='mm', shrink=.8)
    figure.suptitle('Normal displacement: image and model comparison')
    path = output/'displacement_overview.png'
    figure.savefig(path, dpi=150)
    figure.clear()
    return path


def _row(image: Path, method: str, model: dict, result: dict, path: Path) -> dict:
    displacement = result['displacement'][result['valid_mask']]
    boundary = result['boundary_mask']
    return dict(image=image.stem, image_file=str(image), image_sha256=_sha256(image),
                method=method, model_file=model['file'], model_sha256=model['sha256'],
                model_validation_status=model['validation_status'], output_file=str(path),
                output_sha256=_sha256(path), shape=list(result['valid_mask'].shape),
                valid_points=int(result['valid_mask'].sum()),
                photometric_points=int(result['photometric_valid_mask'].sum()),
                model_points=int(result['model_valid_mask'].sum()),
                prior_points=int(result['prior_valid_mask'].sum()),
                equation_points=int(result['equation_valid_mask'].sum()),
                inferred_points=int(result['inferred_mask'].sum()),
                minimum_displacement_mm=float(displacement.min()),
                maximum_displacement_mm=float(displacement.max()),
                fixed_max_abs_displacement_mm=float(np.max(np.abs(result['displacement'][boundary]))),
                fixed_max_abs_slope=float(np.max(np.abs(result['slopes'][boundary]))),
                solver_istop=int(result['solver_istop']),
                solver_iterations=int(result['solver_iterations']),
                solver_residual_norm=float(result['solver_residual_norm']))


def compare_reconstructions(comparison_file, image_files, config_path, output_dir=None,
                            *, result_root=None, reuse_results=False, manifest_file=None):
    """Compare candidate models or verified saved results and refresh per-model previews.

    Reuse validates the manifest's input, model, configuration and NPZ hashes; previews
    are rendered again with a shared per-image color scale and 3D view.
    """
    comparison_file = Path(comparison_file).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    if isinstance(image_files, (str, Path)):
        image_files = [image_files]
    images = [Path(path).expanduser().resolve() for path in image_files]
    if not images or len({path.stem for path in images}) != len(images):
        raise ValueError('Input images must have distinct stems')
    comparison = json.loads(comparison_file.read_text(encoding='utf-8'))
    models = _models(comparison, config_path)
    if reuse_results and manifest_file is None:
        raise ValueError('A manifest is required to reuse saved results')
    records = (_manifest_records(Path(manifest_file).expanduser().resolve(), config_path, images, models)
               if reuse_results else {})
    configured_root = load_config(config_path)['reconstruction']['output_dir']
    root = (Path(result_root).expanduser().resolve() if result_root else
            (config_path.parent/str(configured_root)).resolve())
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = (Path(output_dir).expanduser().resolve() if output_dir else
              comparison_file.parent/'reconstruction_batch'/timestamp)
    if not reuse_results:
        for image in images:
            for method in models:
                target = root/'pc'/f'{image.stem}_{method}_local_result.npz'
                preview = root/'pics'/f'{image.stem}_{method}_local_result.png'
                if target.exists() or preview.exists():
                    raise FileExistsError(f'Output already exists: {target} or {preview}')
    targets = [output/name for name in ('reconstruction_comparison.json',
                                       'reconstruction_comparison.csv', 'displacement_overview.png')]
    targets.extend(output/image.stem/name for image in images for name in (
        'displacement_comparison.png', 'surface_3d_comparison.png', 'slope_field_comparison.png'))
    for target in targets:
        if target.exists():
            raise FileExistsError(f'Comparison output already exists: {target}')
    output.mkdir(parents=True, exist_ok=True)
    (root/'pc').mkdir(parents=True, exist_ok=True)
    all_results, rows, figures = {}, [], {}
    for image in images:
        image_results = {}
        for method, model in models.items():
            target = (Path(records[(image, method)]['output_file']).resolve() if reuse_results else
                      root/'pc'/f'{image.stem}_{method}_local_result.npz')
            print(f'{image.stem} / {method}: {"reuse" if reuse_results else "reconstruct"}', flush=True)
            if not reuse_results:
                result, _ = reconstruct_image(image, config_path, calibration_file=model['file'],
                                              output=str(target))
                result.save(target)
            image_results[method] = _read_result(target)
            rows.append(_row(image, method, model, image_results[method], target))
        _verify_geometry(image_results)
        image_dir = output/image.stem
        image_dir.mkdir(exist_ok=True)
        figures[image.stem] = _save_image_comparisons(image, image_results, image_dir, root/'pics')
        all_results[image.stem] = image_results
    overview = _save_overview(images, all_results, output)
    summary = dict(comparison_file=str(comparison_file), comparison_sha256=_sha256(comparison_file),
                   config_file=str(config_path), config_sha256=_sha256(config_path),
                   input_images=[dict(file=str(image), sha256=_sha256(image)) for image in images],
                   models=models, results=rows, figures=figures, overview=str(overview))
    (output/'reconstruction_comparison.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output/'reconstruction_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description='对多张图像比较法向模型的重建结果')
    parser.add_argument('--config', default=Path(__file__).resolve().parents[1]/'config.yaml')
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--image', action='append', required=True, help='图像路径，可重复指定')
    parser.add_argument('--output-dir', help='批次对比输出目录')
    parser.add_argument('--result-root', help='单模型 pc/ 和 pics/ 输出目录')
    parser.add_argument('--reuse-results', action='store_true', help='复用已保存结果并刷新共享尺度预览')
    parser.add_argument('--manifest', help='复用结果的输入、模型和结果哈希清单')
    args = parser.parse_args()
    summary = compare_reconstructions(args.comparison, args.image, args.config, args.output_dir,
        result_root=args.result_root, reuse_results=args.reuse_results, manifest_file=args.manifest)
    print(f"重建对比总览: {summary['overview']}")


if __name__ == '__main__':
    main()
