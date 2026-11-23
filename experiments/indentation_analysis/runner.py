"""CLI and reports for fixed-radius sphere fitting and normal-angle analysis."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from utils.config import load_config
from .analysis import (
    Contact, ContactResult, Reconstruction, analyze_contacts,
    find_contacts, load_reconstruction,
)
from .depth_fitting import depth_scan_options
from .geometry import analyze_lines
from .visualization import write_analysis_figure, write_fit_check

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config.yaml'


def _sphere_radii(config_path, overrides, count):
    if overrides is None:
        section = load_config(config_path).get('indentation_analysis')
        if not isinstance(section, dict) or 'sphere_radii_mm' not in section:
            raise ValueError('配置缺少 indentation_analysis.sphere_radii_mm')
        values = section['sphere_radii_mm']
    else:
        values = overrides
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f'需要按接触区顺序提供 {count} 个球半径（mm）')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError('球半径必须是有限正数（mm）')
    return tuple(float(value) for value in values)


def _native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot encode {type(value).__name__}')


def _fit_summary(fit):
    if fit is None:
        return None
    return dict(center_xyz_mm=fit.center_xyz_mm, radius_mm=fit.radius_mm,
                diameter_mm=2*fit.radius_mm, radial_rms_mm=fit.radial_rms_mm,
                jacobian_condition=fit.jacobian_condition)


def _build_report(source, results, *, threshold_mm, min_region_points,
                  contact_selection, excluded_contact_regions):
    contacts = [dict(index=result.contact.index, center_grid_rc=result.contact.center_rc,
                     sphere_radius_mm=result.radius_mm, sample_count=result.sample_count,
                     fit=_fit_summary(result.fit), selection=result.selection, warnings=result.warnings,
                     projection=asdict(result.projection) if result.projection is not None else None,
                     error=result.error) for result in results]
    successful = [result for result in results if result.projection is not None]
    plane = None
    plane_error = None
    geometry_pairs = {}
    if len(successful) >= 3:
        try:
            line_analysis = analyze_lines(
                [result.projection for result in successful],
                np.stack([result.fit.center_xyz_mm for result in successful]))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            plane_error = str(exc)
        else:
            plane = {key: value for key, value in asdict(line_analysis).items() if key != 'pairs'}
            for pair in line_analysis.pairs:
                i, j = pair.indices
                geometry_pairs[(successful[i].contact.index, successful[j].contact.index)] = pair

    pairs = []
    for i, left in enumerate(results):
        for right in results[i+1:]:
            pair = dict(contacts=[left.contact.index, right.contact.index], angle_3d_degrees=None,
                        projected_angle_degrees=None, extended_line_distance_mm=None,
                        warnings=[f'contact_{result.contact.index}: {warning}'
                                  for result in (left, right) for warning in result.warnings])
            if left.projection is not None and right.projection is not None:
                u, v = left.projection.unit_direction, right.projection.unit_direction
                pair['angle_3d_degrees'] = float(np.rad2deg(np.arccos(np.clip(u @ v, -1, 1))))
                cross = np.cross(u, v)
                norm = np.linalg.norm(cross)
                delta = right.projection.xyz_mm-left.projection.xyz_mm
                if norm > 1e-8:
                    pair['extended_line_distance_mm'] = float(abs(delta @ cross)/norm)
                geometry_pair = geometry_pairs.get((left.contact.index, right.contact.index))
                if geometry_pair is not None:
                    pair['projected_angle_degrees'] = geometry_pair.projected_angle_degrees
            pairs.append(pair)
    return dict(
        source=str(source), coordinate_frame='camera XYZ in millimetres',
        method=dict(
            baseline='xyz_out - displacement_vectors',
            detection_threshold_mm=threshold_mm, min_region_points=min_region_points,
            contact_selection=contact_selection,
            region_order='geometric centroid in material-grid row/column coordinates',
            sphere_radii_mm=[result.radius_mm for result in results],
            sphere_fit='expand candidate points from deepest to shallowest displacement; '
                       'independent fixed-radius radial least squares; '
                       'accept the first stable RMSE window',
            selection_file='fit_selection.npz',
            depth_scan_options=results[0].selection['convergence_options'],
            projection='nearest supported point on the baseline surface; direction points from surface to sphere center',
            angle_projection='onto the least-squares plane of the fitted center/projection endpoints',
        ),
        contacts=contacts, excluded_contact_regions=excluded_contact_regions,
        pairs=pairs, line_analysis=plane, line_analysis_error=plane_error,
        complete=all(result.error is None for result in results),
    )


def select_largest_contact(
    contacts: list[Contact], photometric_valid: np.ndarray,
) -> tuple[list[Contact], list[dict]]:
    """Choose the uniquely largest geometric region and describe the others.

    ``photometric_valid`` is a boolean mask on the same grid as each region.
    The selected contact is returned in a one-element list with index 1; a tie
    for the largest region raises ``ValueError``.
    """
    sizes = [int(contact.region_mask.sum()) for contact in contacts]
    largest_size = max(sizes)
    if sizes.count(largest_size) != 1:
        raise ValueError('最大接触区存在相同数量的几何点，无法唯一选择')
    largest_index = sizes.index(largest_size)
    excluded = [
        dict(center_grid_rc=contact.center_rc, geometric_point_count=size,
             photometric_valid_point_count=int((contact.region_mask & photometric_valid).sum()))
        for index, (contact, size) in enumerate(zip(contacts, sizes, strict=True))
        if index != largest_index
    ]
    return [replace(contacts[largest_index], index=1)], excluded


def _write_fits(path, results):
    names = ('contact', 'candidate_count', 'sample_count', 'selection_method', 'radius_mm',
             'diameter_mm', 'converged_depth_mm', 'converged_rmse_mm', 'converged_mse_mm2',
             'scanned_levels', 'radial_rms_mm', 'center_baseline_signed_distance_mm',
             'jacobian_condition', 'center_x_mm', 'center_y_mm', 'center_z_mm', 'warnings', 'error')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for result in results:
            row = dict(contact=result.contact.index, sample_count=result.sample_count,
                       radius_mm=result.radius_mm, diameter_mm=2*result.radius_mm,
                       selection_method=result.selection['method'], warnings='; '.join(result.warnings),
                       error=result.error or '')
            for key in ('candidate_count', 'converged_depth_mm', 'converged_rmse_mm',
                        'converged_mse_mm2', 'scanned_levels', 'center_baseline_signed_distance_mm'):
                row[key] = result.selection.get(key)
            if result.fit is not None:
                row.update(radial_rms_mm=result.fit.radial_rms_mm,
                           jacobian_condition=result.fit.jacobian_condition)
                row.update(zip(('center_x_mm', 'center_y_mm', 'center_z_mm'),
                               result.fit.center_xyz_mm, strict=True))
            writer.writerow(row)


def _write_depth_scan(output: Path, results: list[ContactResult]) -> None:
    rows = []
    for result in results:
        for sample in result.depth_samples:
            fit = sample.fit
            converged_depth = result.selection['converged_depth_mm']
            row = dict(
                contact=result.contact.index, depth_mm=sample.depth_mm,
                minimum_indentation_mm=sample.minimum_indentation_mm,
                sample_count=sample.sample_count,
                radial_rmse_mm=None if fit is None else fit.radial_rms_mm,
                radial_mse_mm2=None if fit is None else fit.radial_rms_mm**2,
                center_x_mm=None, center_y_mm=None, center_z_mm=None,
                jacobian_condition=None if fit is None else fit.jacobian_condition,
                status='ok' if fit is not None else sample.error,
                converged=converged_depth is not None and sample.depth_mm == converged_depth,
            )
            if fit is not None:
                row.update(zip(('center_x_mm', 'center_y_mm', 'center_z_mm'), fit.center_xyz_mm))
            rows.append(row)
    names = ('contact', 'depth_mm', 'minimum_indentation_mm', 'sample_count', 'radial_rmse_mm',
             'radial_mse_mm2', 'center_x_mm', 'center_y_mm', 'center_z_mm',
             'jacobian_condition', 'status', 'converged')
    with (output/'depth_scan.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    report = dict(
        definition='d = displacement - minimum candidate displacement, in mm; '
                   'include every photometric candidate with offset <= d',
        convergence='first window of stable_steps + 1 successful fits with new points at '
                    'each regular step; max(RMSE)-min(RMSE) <= max(atol, mean(RMSE)*rtol)',
        contacts=[dict(contact=result.contact.index, status=result.selection['method'],
                       converged_depth_mm=result.selection['converged_depth_mm'],
                       converged_rmse_mm=result.selection['converged_rmse_mm'],
                       converged_mse_mm2=result.selection['converged_mse_mm2'],
                       options=result.selection['convergence_options']) for result in results],
        samples=rows,
    )
    (output/'depth_scan.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_native, allow_nan=False)+'\n',
        encoding='utf-8')


def save_contact_analysis(
    source: Path, output: Path, reconstruction: Reconstruction, results: list[ContactResult], *,
    threshold_mm: float, min_region_points: int, contact_selection: str,
    excluded_contact_regions: list[dict], figures: bool = True,
) -> dict:
    """Write the analysis report, fitted spheres, and grid-aligned selection masks.

    The reconstruction and results must share the same material grid. Returns
    the report dictionary; ``figures`` controls the two visualizations.
    """
    report = _build_report(source, results, threshold_mm=threshold_mm,
                           min_region_points=min_region_points,
                           contact_selection=contact_selection,
                           excluded_contact_regions=excluded_contact_regions)
    output.mkdir(parents=True, exist_ok=True)
    _write_fits(output/'sphere_fits.csv', results)
    _write_depth_scan(output, results)
    if figures:
        write_analysis_figure(output/'analysis.png', reconstruction, results)
        write_fit_check(output/'fit_check.png', reconstruction, results)
    np.savez_compressed(output/'fit_selection.npz', source_file=np.asarray(str(source)),
        contact_index=np.asarray([result.contact.index for result in results], np.int32),
        candidate_masks=np.stack([result.contact.region_mask & reconstruction.photometric_valid
                                  for result in results]),
        fit_masks=np.stack([result.fit_mask for result in results]))
    (output/'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                  default=_native, allow_nan=False)+'\n', encoding='utf-8')
    return report


def _print_report(report, output):
    for contact in report['contacts']:
        row, column = contact['center_grid_rc']
        print(f"接触区 {contact['index']}：区域质心 ({row:.2f}, {column:.2f})，"
              f"拟合点数 {contact['sample_count']}，已知球半径 {contact['sphere_radius_mm']:.4f} mm")
        selection = contact['selection']
        print(f"  取点方式 {selection['method']}，候选点数 {selection['candidate_count']}")
        if selection['converged_depth_mm'] is not None:
            print(f"  收敛深度 d={selection['converged_depth_mm']:.4f} mm，"
                  f"MSE={selection['converged_mse_mm2']:.8f} mm²")
        else:
            print(f"  depth_not_converged：已扫描 {selection['scanned_levels']} 层，"
                  f"参数 {selection['convergence_options']}")
        fit = contact['fit']
        if fit is not None:
            center = ', '.join(f'{value:.4f}' for value in fit['center_xyz_mm'])
            print(f"  球心 ({center}) mm，径向 RMS {fit['radial_rms_mm']:.4f} mm，"
                  f"Jacobian 条件数 {fit['jacobian_condition']:.3f}")
        labels = {
            'center_negative_side': '球心位于整体基准面的负法向侧，有向连线与正法向反向',
            'center_near_baseline': '球心接近整体基准面，需结合球心敏感性检查方向',
        }
        for warning in contact['warnings']:
            print(f"  方向检查：{labels.get(warning, warning)}")
        if contact['error']:
            print(f"  拟合/投影状态：{contact['error']}")
    for pair in report['pairs']:
        a, b = pair['contacts']
        if pair['angle_3d_degrees'] is None:
            continue
        line = f"{a}–{b}：法向夹角 {pair['angle_3d_degrees']:.3f}°"
        if pair['projected_angle_degrees'] is not None:
            line += f"，投影夹角 {pair['projected_angle_degrees']:.3f}°"
        if pair['extended_line_distance_mm'] is not None:
            line += f"，延长线最短距离 {pair['extended_line_distance_mm']:.4f} mm"
        print(line)
    if report['line_analysis']:
        plane = report['line_analysis']
        print(f"线段端点最佳平面：RMS 偏差 {plane['endpoint_plane_rms_mm']:.4f} mm，"
              f"最大偏差 {plane['endpoint_plane_max_mm']:.4f} mm；"
              f"线段对该平面的最大偏角 {max(plane['line_angles_to_endpoint_plane_degrees']):.3f}°")
        print(f"单独拟合方向平面的最大偏离：{max(plane['direction_angles_to_plane_degrees']):.4f}°")
    if report['line_analysis_error']:
        print(f"共同平面状态：{report['line_analysis_error']}")
    print(f'结果目录：{output}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='已知球半径的球面拟合与局部法向夹角分析')
    parser.add_argument('point_cloud', help='待分析的重建结果 NPZ 路径')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='球半径配置文件，默认使用项目 config.yaml')
    parser.add_argument('--radii-mm', type=float, nargs='+', help='按区域顺序临时覆盖配置中的已知球半径（mm）')
    parser.add_argument('--output-dir', help='输出目录；默认在输入文件旁创建 <文件名>_analysis')
    parser.add_argument('--count', type=int, default=3, help='接触区域数量，默认 3')
    parser.add_argument('--largest-contact', action='store_true',
                        help='单球分析时按有效几何点数选取最大连通区，须与 --count 1 同用')
    parser.add_argument('--threshold-mm', type=float, default=.2,
                        help='分割独立接触区的负位移幅值阈值（mm），默认 0.2')
    parser.add_argument('--min-region-points', type=int, default=100,
                        help='候选接触区的最少几何点数，默认 100；拟合点数读取 depth_scan 配置')
    args = parser.parse_args(argv)
    if args.largest_contact and args.count != 1:
        parser.error('--largest-contact 须与 --count 1 同用')
    source = Path(args.point_cloud).expanduser().resolve()
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir
              else source.parent / (source.stem + '_analysis'))
    try:
        scan_options = depth_scan_options(load_config(args.config))
        radii = _sphere_radii(args.config, args.radii_mm, args.count)
        reconstruction = load_reconstruction(source)
        contacts = find_contacts(reconstruction, count=None if args.largest_contact else args.count,
                                 threshold_mm=args.threshold_mm,
                                 min_region_points=args.min_region_points)
        if args.largest_contact:
            contacts, excluded = select_largest_contact(contacts, reconstruction.photometric_valid)
        else:
            excluded = []
        results = analyze_contacts(reconstruction, contacts, radii, scan_options=scan_options)
        report = save_contact_analysis(
            source, output, reconstruction, results, threshold_mm=args.threshold_mm,
            min_region_points=args.min_region_points,
            contact_selection='largest_connected_region' if args.largest_contact else 'strict_count',
            excluded_contact_regions=excluded,
        )
        (output/'depth_sweep.csv').unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'分析失败：{exc}\n')
    _print_report(report, output)
    return 0 if report['complete'] else 1
