"""Reconstruct single-sphere images and summarize depth-converged fits."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from numbers import Real
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from reconstruction.local_recon import default_reconstruction_output
from reconstruction.pipeline import DEFAULT_CONFIG_PATH
from reconstruction.recon_batch import load_batch_calibrations, reconstruct_with_calibration
from utils.calibration_store import METHODS, CalibrationArtifacts
from utils.config import load_config
from .analysis import _baseline_direction, analyze_contacts, find_contacts, load_reconstruction
from .depth_fitting import DepthScanOptions, depth_scan_options
from .geometry import sphere_indentation_depth
from .runner import save_contact_analysis, select_largest_contact
from .sphere_rmse_statistics import summarize_rmse, write_rmse_distribution


def discover_images(directory: Path, pattern: str = '*.npz') -> list[Path]:
    """Read one directory in filename order; each image owns a unique output folder."""
    if not directory.is_dir():
        raise ValueError(f'输入目录应为图像文件夹：{directory}')
    images = sorted(path.resolve() for path in directory.glob(pattern)
                    if path.is_file() and path.suffix.lower() in ('.npz', '.png', '.jpg', '.jpeg'))
    if not images:
        raise ValueError(f'{directory}: 需要匹配 {pattern} 的重建输入图像')
    if len({image.stem for image in images}) != len(images):
        raise ValueError('图像主文件名应唯一，请通过 --pattern 选择同一格式的输入')
    return images


def _positive_value(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) \
            or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} 必须为有限正数')
    return float(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def _case_record(image: Path, method: str, directory: Path, radius: float) -> dict:
    return dict(
        image=image.name, method=method, radius_mm=radius, status='pending', error_stage='', error='',
        source_image=str(image), point_cloud=str(directory/f'{method}.npz'), point_cloud_sha256=None,
        preview=str(directory/'pics'/f'{method}.png'), analysis_dir=str(directory/f'{method}_analysis'),
        solver_iterations=None, solver_residual_norm=None,
        candidate_point_count=None, fit_point_count=None, retained_fraction=None,
        selection_method=None, converged_depth_mm=None, converged_rmse_mm=None,
        converged_mse_mm2=None, scanned_levels=None,
        center_x_mm=None, center_y_mm=None, center_z_mm=None,
        center_height_mm=None, indentation_depth_mm=None, projection_status='not_available',
        warnings='', excluded_region_count=None,
    )


def _solver_metadata(path: Path) -> tuple[int | None, float | None]:
    """Read solver metadata saved with a point cloud when it is present."""
    with np.load(path, allow_pickle=False) as data:
        iterations = int(data['solver_iterations']) if 'solver_iterations' in data else None
        residual = float(data['solver_residual_norm']) if 'solver_residual_norm' in data else None
    if iterations is not None and iterations < 0:
        raise ValueError('重建求解器迭代次数应为非负数')
    if residual is not None and (not np.isfinite(residual) or residual < 0):
        raise ValueError('重建求解器残差应为有限非负数')
    return iterations, residual


def _process_case(image: Path, method: str, config_path: Path,
                  artifacts: CalibrationArtifacts | None, directory: Path, *,
                  radius_mm: float, threshold_mm: float, min_region_points: int,
                  reference_image: str | None, scan_options: DepthScanOptions,
                  analysis_only: bool) -> dict:
    record = _case_record(image, method, directory, radius_mm)
    stage = 'reconstruction'
    try:
        point_cloud = Path(record['point_cloud'])
        if analysis_only:
            iterations, residual = _solver_metadata(point_cloud)
        else:
            result, point_cloud = reconstruct_with_calibration(
                image, config_path, artifacts, output=point_cloud,
                preview_output=Path(record['preview']), reference_image=reference_image)
            iterations, residual = int(result.solver_iterations), float(result.solver_residual_norm)
            if iterations < 0 or not np.isfinite(residual) or residual < 0:
                raise ValueError('重建求解器迭代次数与残差应为有限非负数')
        record.update(solver_iterations=iterations, solver_residual_norm=residual,
                      point_cloud_sha256=_sha256(Path(point_cloud)))
        stage = 'contact_detection'
        reconstruction = load_reconstruction(point_cloud)
        contacts = find_contacts(reconstruction, count=None, threshold_mm=threshold_mm,
                                 min_region_points=min_region_points)
        contacts, excluded = select_largest_contact(contacts, reconstruction.photometric_valid)
        record['excluded_region_count'] = len(excluded)
        stage = 'sphere_fit'
        results = analyze_contacts(reconstruction, contacts, (radius_mm,), scan_options=scan_options)
        contact = results[0]
        selection = contact.selection
        record.update(candidate_point_count=int(selection['candidate_count']),
                      fit_point_count=contact.sample_count,
                      selection_method=selection['method'],
                      converged_depth_mm=selection['converged_depth_mm'],
                      converged_rmse_mm=selection['converged_rmse_mm'],
                      converged_mse_mm2=selection['converged_mse_mm2'],
                      scanned_levels=selection['scanned_levels'],
                      warnings='; '.join(contact.warnings))
        if record['candidate_point_count']:
            record['retained_fraction'] = contact.sample_count/record['candidate_point_count']
        if selection['method'] == 'depth_not_converged':
            if contact.fit is not None or any(record[field] is not None for field in
                    ('converged_depth_mm', 'converged_rmse_mm', 'converged_mse_mm2')):
                raise ValueError('未收敛的扫描应保留空的收敛拟合指标')
        elif selection['method'] == 'depth_convergence' and contact.fit is not None:
            if any(record[field] is None for field in
                    ('converged_depth_mm', 'converged_rmse_mm', 'converged_mse_mm2')):
                raise ValueError('收敛拟合缺少深度或误差指标')
            record.update(zip(('center_x_mm', 'center_y_mm', 'center_z_mm'),
                              contact.fit.center_xyz_mm.tolist(), strict=True))
        else:
            raise ValueError(f"无效的深度扫描状态：{selection['method']}")
        stage = 'analysis_save'
        save_contact_analysis(
            Path(point_cloud), Path(record['analysis_dir']), reconstruction, results,
            threshold_mm=threshold_mm, min_region_points=min_region_points,
            contact_selection='largest_connected_region', excluded_contact_regions=excluded,
            figures=False)
        if selection['method'] == 'depth_not_converged':
            record.update(status='depth_not_converged', error_stage='depth_scan',
                          error=contact.error or '深度扫描未收敛')
        elif contact.projection is None:
            record.update(status='projection_failed', error_stage='projection',
                          error=contact.error or '基准曲面投影失败', projection_status='failed')
        else:
            stage = 'projection'
            height, depth = sphere_indentation_depth(
                contact.fit.center_xyz_mm, radius_mm, contact.projection.xyz_mm,
                _baseline_direction(reconstruction, contact.fit_mask))
            record.update(status='ok', projection_status='ok', center_height_mm=height,
                          indentation_depth_mm=depth)
    except Exception as error:
        # Keep projection failures after a converged fit; other failed stages
        # do not contribute a metric to the distribution.
        if stage != 'projection':
            record.update(converged_depth_mm=None, converged_rmse_mm=None, converged_mse_mm2=None)
        record.update(status=f'{stage}_failed', error_stage=stage,
                      error=f'{type(error).__name__}: {error}')
        if stage == 'projection':
            record['projection_status'] = 'failed'
    return record


def _analysis_provenance(output: Path, images: list[dict]) -> tuple[dict | None, dict | None]:
    """Preserve prior reconstruction provenance and verify recorded inputs."""
    path = output/'rmse_summary.json'
    if not path.is_file():
        return None, None
    previous = json.loads(path.read_text(encoding='utf-8'))
    saved_images = previous.get('images')
    if saved_images is None:
        return None, None
    saved = {item['image']: item['sha256'] for item in saved_images}
    current = {item['image']: item['sha256'] for item in images}
    if saved != current:
        changed = sorted(name for name in saved.keys() | current.keys()
                         if saved.get(name) != current.get(name))
        raise ValueError('现有点云与输入图像哈希记录不一致：'+', '.join(changed))
    for record in previous.get('records', []):
        recorded_hash = record.get('point_cloud_sha256')
        if recorded_hash:
            cloud = output/Path(record['image']).stem/f"{record['method']}.npz"
            if not cloud.is_file() or _sha256(cloud) != recorded_hash:
                raise ValueError(f'现有点云与记录的哈希不一致：{cloud}')
    provenance = previous.get('point_cloud_provenance')
    if provenance is None and not previous.get('analysis_only', False):
        provenance = dict(
            config_file=previous.get('config_file'), config_sha256=previous.get('config_sha256'),
            reference_image=previous.get('reference_image'),
            reference_sha256=previous.get('reference_sha256'), images=saved_images,
        )
    return previous.get('calibrations'), provenance


def _validate_existing_clouds(output: Path, images: list[Path]) -> None:
    missing = [str(output/image.stem/f'{method}.npz')
               for image in images for method in METHODS
               if not (output/image.stem/f'{method}.npz').is_file()]
    if missing:
        raise ValueError('复用点云时需要所有图像/方法的 NPZ，缺少：'+', '.join(missing))


def write_depth_scan_overview(path: Path, records: list[dict], methods: tuple[str, ...]) -> None:
    """Draw all saved depth scans in image rows and method columns."""
    images = list(dict.fromkeys(record['image'] for record in records))
    by_case = {(record['image'], record['method']): record for record in records}
    scans = {}
    for key, record in by_case.items():
        scan_file = Path(record['analysis_dir'])/'depth_scan.json'
        if record['status'] not in ('ok', 'depth_not_converged', 'projection_failed') or not scan_file.is_file():
            continue
        scan = json.loads(scan_file.read_text(encoding='utf-8'))
        samples = [sample for sample in scan['samples'] if sample['contact'] == 1]
        scans[key] = samples
    figure = Figure(figsize=(max(11., 2.8*len(methods)), max(3., 2.35*len(images))))
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(images), len(methods), squeeze=False)
    for row, image_name in enumerate(images):
        for column, method in enumerate(methods):
            axis = axes[row, column]
            record = by_case.get((image_name, method))
            samples = scans.get((image_name, method), [])
            last_depth = max((sample['depth_mm'] for sample in samples), default=0.)
            if samples:
                x = [sample['depth_mm'] for sample in samples]
                y = [np.nan if sample['radial_rmse_mm'] is None else sample['radial_rmse_mm']
                     for sample in samples]
                axis.plot(x, y, color='#326da0', linewidth=1.)
                finite_y = np.asarray(y)[np.isfinite(y)]
                if len(finite_y):
                    bottom, top = float(finite_y.min()), float(finite_y.max())
                    span = max(top-bottom, abs(top)*.05, 1e-9)
                    axis.set_ylim(bottom-.05*span, top+.4*span)
            if record is not None and record['converged_depth_mm'] is not None:
                depth, rmse = record['converged_depth_mm'], record['converged_rmse_mm']
                last_depth = max(last_depth, depth)
                axis.axvline(depth, color='#c6483b', linestyle='--', linewidth=.8)
                axis.scatter([depth], [rmse], color='#c6483b', s=17, zorder=3)
                axis.text(.03, .96, f'd={depth:.3f} mm\nRMSE={rmse:.4f} mm',
                          transform=axis.transAxes, ha='left', va='top', fontsize=7,
                          bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
            else:
                status = 'missing' if record is None else record['status']
                axis.text(.03, .96, status, transform=axis.transAxes,
                          ha='left', va='top', fontsize=8, color='#a42d28',
                          bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
            axis.set_xlim(0, last_depth*1.05 if last_depth > 0 else 1.)
            axis.grid(axis='y', alpha=.2)
            if row == 0:
                axis.set_title(method)
            if column == 0:
                axis.set_ylabel(f'{image_name}\nRMSE (mm)')
            if row == len(images)-1:
                axis.set_xlabel('Depth from deepest point (mm)')
    figure.suptitle('Depth scans: first convergence or complete range (individual horizontal scales)',
                    fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(path, dpi=160, bbox_inches='tight')
    figure.clear()


def run_sphere_batch(input_dir: str | Path, config_path: str | Path = DEFAULT_CONFIG_PATH, *,
                     output_dir: str | Path | None = None, pattern: str = '*.npz',
                     radius_mm: float | None = None, threshold_mm: float = .2,
                     min_region_points: int = 100, reference_image: str | None = None,
                     analysis_only: bool = False) -> dict:
    """Analyze every image/method pair, optionally reusing all existing point clouds."""
    source = Path(input_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    images = discover_images(source, pattern)
    config = load_config(config_path)
    radius = config.get('normal_calibration', {}).get('sphere_radius_mm') if radius_mm is None else radius_mm
    radius = _positive_value(radius, 'sphere_radius_mm')
    threshold = _positive_value(threshold_mm, 'threshold_mm')
    scan_options = depth_scan_options(config)
    if isinstance(min_region_points, (bool, np.bool_)) \
            or not isinstance(min_region_points, (int, np.integer)) or min_region_points < 4:
        raise ValueError('min_region_points 必须为至少 4 的整数')
    if not isinstance(analysis_only, bool):
        raise ValueError('analysis_only 必须为布尔值')
    output = (Path(output_dir).expanduser().resolve() if output_dir is not None else
              default_reconstruction_output(config_path, config['reconstruction']['output_dir'],
                                            source.name+'_series_batch').resolve())
    if output == source:
        raise ValueError('请使用与输入图像目录分开的输出目录')
    reference = Path(reference_image if reference_image is not None else
                     config['reconstruction']['reference_image']).expanduser()
    if not reference.is_absolute():
        reference = (Path.cwd() if reference_image is not None else config_path.parent)/reference
    reference = reference.resolve()
    image_info = [dict(image=image.name, source=str(image), sha256=_sha256(image)) for image in images]
    if analysis_only:
        _validate_existing_clouds(output, images)
        calibration_info, provenance = _analysis_provenance(output, image_info)
        calibrations = None
    else:
        calibrations = load_batch_calibrations(config_path)
        if tuple(calibrations) != METHODS:
            raise ValueError('批量实验需要按固定顺序提供全部五种 LUT')
        calibration_info = {
            method: dict(model_file=str(artifacts.model_file), grid_file=str(artifacts.grid_file),
                         version=artifacts.directory.name,
                         model_sha256=_sha256(artifacts.model_file), grid_sha256=_sha256(artifacts.grid_file))
            for method, artifacts in calibrations.items()
        }
        provenance = None
    reference_sha256 = _sha256(reference)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        state='running', analysis_only=analysis_only,
        input_dir=str(source), output_dir=str(output), config_file=str(config_path),
        config_sha256=_sha256(config_path), pattern=pattern,
        reference_image=str(reference), reference_sha256=reference_sha256,
        point_cloud_provenance=provenance,
        reconstruction_provenance_status=('current_run' if not analysis_only else
                                          'recorded' if provenance is not None else 'unknown'),
        radius_mm=radius, diameter_mm=2*radius, threshold_mm=threshold,
        min_region_points=min_region_points, contact_selection='largest_connected_region',
        depth_scan_options=asdict(scan_options),
        converged_rmse_definition='sqrt(mean((norm(P_i-C)-R)^2)) at the first converged depth, in mm',
        converged_mse_definition='mean((norm(P_i-C)-R)^2) at the first converged depth, in mm^2',
        statistics_definition='each available image has equal weight; variance/std use ddof=0; sample variance uses ddof=1',
        pooled_definition='RMSE: sqrt(sum(N_i*RMSE_i^2)/sum(N_i)); MSE: sum(N_i*MSE_i)/sum(N_i)',
        missing_values='unconverged or unavailable metrics are null and excluded; projection failures retain converged metrics',
        images=image_info, calibrations=calibration_info, records=[], summaries=[],
    )
    _write_json(output/'rmse_summary.json', report)
    records = []
    for image_index, image in enumerate(images, 1):
        directory = output/image.stem
        for method in METHODS:
            print(f'[{image_index}/{len(images)}] {image.name} / {method}', flush=True)
            artifacts = None if calibrations is None else calibrations[method]
            record = _process_case(image, method, config_path, artifacts, directory, radius_mm=radius,
                                   threshold_mm=threshold, min_region_points=min_region_points,
                                   reference_image=reference_image, scan_options=scan_options,
                                   analysis_only=analysis_only)
            records.append(record)
            metric = record['converged_rmse_mm']
            result = 'N/A' if metric is None else f'{metric:.6f} mm'
            print(f'  {record["status"]}: converged RMSE={result}'
                  + (f' | {record["error"]}' if record['error'] else ''), flush=True)
        report['records'] = records
        _write_csv(output/'rmse_per_image.csv', records)
        _write_json(output/'rmse_summary.json', report)
    summaries = summarize_rmse(records, METHODS)
    report.update(state='complete' if all(row['status'] == 'ok' for row in records)
                  else 'completed_with_failures', records=records, summaries=summaries)
    _write_csv(output/'rmse_statistics.csv', summaries)
    _write_json(output/'rmse_summary.json', report)
    write_rmse_distribution(output/'rmse_distribution.png', records, summaries)
    write_depth_scan_overview(output/'depth_scan_overview.png', records, METHODS)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='单球图像文件夹：五 LUT 深度收敛拟合与误差分布统计')
    parser.add_argument('input_dir', type=Path, help='单球压入图像文件夹')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--output-dir', type=Path, help='默认 reconstruction.output_dir/pc/<输入目录名>_series_batch')
    parser.add_argument('--pattern', default='*.npz', help='输入匹配模式，默认 *.npz')
    parser.add_argument('--radius-mm', type=float, help='已知球半径，默认读取 normal_calibration.sphere_radius_mm')
    parser.add_argument('--threshold-mm', type=float, default=.2)
    parser.add_argument('--min-region-points', type=int, default=100)
    parser.add_argument('--reference-image', help='覆盖重建参考图')
    parser.add_argument('--analysis-only', action='store_true', help='复用所有已有点云并重新执行深度扫描')
    args = parser.parse_args(argv)
    try:
        report = run_sphere_batch(
            args.input_dir, args.config, output_dir=args.output_dir, pattern=args.pattern,
            radius_mm=args.radius_mm, threshold_mm=args.threshold_mm,
            min_region_points=args.min_region_points, reference_image=args.reference_image,
            analysis_only=args.analysis_only)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f'批量单球实验失败：{error}\n')
    for row in report['summaries']:
        mean = 'N/A' if row['mean'] is None else f'{row["mean"]:.6f}'
        std = 'N/A' if row['std'] is None else f'{row["std"]:.6f}'
        print(f'{row["method"]} {row["metric"].upper()}: '
              f'n={row["n"]}/{row["n_total"]}, mean={mean} {row["unit"]}, std={std} {row["unit"]}')
    failures = [record for record in report['records'] if record['status'] != 'ok']
    if failures:
        print('失败明细：')
        for record in failures:
            print(f'  {record["image"]} / {record["method"]}: {record["status"]} | {record["error"]}')
    unconverged = [record for record in failures if record['status'] == 'depth_not_converged']
    if unconverged:
        options = report['depth_scan_options']
        print(f'深度扫描配置：step_mm={options["step_mm"]}, min_fit_points={options["min_fit_points"]}, '
              f'stable_steps={options["stable_steps"]}, rmse_atol_mm={options["rmse_atol_mm"]}, '
              f'rmse_rtol={options["rmse_rtol"]}')
    print(f'全部产物：{report["output_dir"]}')
    return 0 if report['state'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
