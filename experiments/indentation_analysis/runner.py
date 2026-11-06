"""CLI and reports for indentation-depth and sphere-normal analysis."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .analysis import analyze_contacts, find_contacts, load_reconstruction
from .geometry import analyze_lines


def _native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot encode {type(value).__name__}')


def _step_summary(step):
    summary = dict(fraction=step.fraction, threshold_mm=step.threshold_mm,
                   sample_count=step.sample_count, error=step.error)
    if step.fit is not None:
        summary.update(center_xyz_mm=step.fit.center_xyz_mm,
                       radius_mm=step.fit.radius_mm, diameter_mm=2*step.fit.radius_mm,
                       radial_rms_mm=step.fit.radial_rms_mm,
                       jacobian_condition=step.fit.jacobian_condition)
    return summary


def _build_report(source, reconstruction, results, *, threshold_mm, min_region_points):
    contacts = []
    successful = []
    for result in results:
        contact = result.contact
        row = dict(index=contact.index, peak_grid_rc=contact.peak_rc,
                   max_depth_mm=contact.max_depth_mm, error=result.error,
                   stable_fraction_start=None, selected_fit=None, projection=None)
        step = result.selected_step
        if step is not None:
            row['stable_fraction_start'] = result.steps[result.stable_index-2].fraction
            row['selected_fit'] = _step_summary(step)
        if result.projection is not None:
            row['projection'] = asdict(result.projection)
            successful.append(result)
        contacts.append(row)

    plane = None
    plane_error = None
    geometry_pairs = {}
    if len(successful) >= 3:
        try:
            line_analysis = analyze_lines(
                [result.projection for result in successful],
                np.stack([result.selected_step.fit.center_xyz_mm for result in successful]))
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
            pair = dict(contacts=[left.contact.index, right.contact.index],
                        depth_difference_mm=abs(left.contact.max_depth_mm-right.contact.max_depth_mm),
                        angle_3d_degrees=None, projected_angle_degrees=None,
                        extended_line_distance_mm=None)
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
            depth='negative signed normal displacement; maxima include all geometrically valid points',
            baseline='xyz_out - displacement_vectors',
            detection_threshold_mm=threshold_mm, min_region_points=min_region_points,
            region_partition='nearest detected peak in material-grid row/column coordinates',
            sphere_fit='radial least squares; photometrically valid points only',
            sphere_scan='minimum depth fraction 0.95 to 0.10 in steps of -0.05',
            stability='first three consecutive steps: diameter range <= 2% of mean, '
                      'maximum pairwise center distance <= 0.1 mm, at least 100 points '
                      'and Jacobian condition < 100 in every step',
            stability_scope='local window; the full scan may drift afterwards',
            projection='nearest supported point on the baseline surface; direction points from surface to sphere center',
            angle_projection='onto the least-squares plane of the fitted center/projection endpoints',
        ),
        contacts=contacts, pairs=pairs, line_analysis=plane, line_analysis_error=plane_error,
        complete=all(result.error is None for result in results),
    )


def _write_sweep(path, results):
    names = ('contact', 'fraction', 'threshold_mm', 'sample_count', 'diameter_mm',
             'radius_mm', 'radial_rms_mm', 'jacobian_condition', 'center_x_mm',
             'center_y_mm', 'center_z_mm', 'selected', 'error')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for result in results:
            for index, step in enumerate(result.steps):
                row = dict(contact=result.contact.index, fraction=step.fraction,
                           threshold_mm=step.threshold_mm, sample_count=step.sample_count,
                           selected=index == result.stable_index, error=step.error or '')
                if step.fit is not None:
                    row.update(diameter_mm=2*step.fit.radius_mm, radius_mm=step.fit.radius_mm,
                               radial_rms_mm=step.fit.radial_rms_mm,
                               jacobian_condition=step.fit.jacobian_condition)
                    row.update(zip(('center_x_mm', 'center_y_mm', 'center_z_mm'),
                                   step.fit.center_xyz_mm, strict=True))
                writer.writerow(row)


def _write_figure(path, reconstruction, results):
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(16, 6), layout='constrained')
    FigureCanvasAgg(figure)
    curve, image, lines = figure.subplots(1, 3)
    colors = colormaps['tab10']
    for i, result in enumerate(results):
        color = colors(i % 10)
        steps = [step for step in result.steps if step.fit is not None]
        curve.plot([100*(1-step.fraction) for step in steps],
                   [2*step.fit.radius_mm for step in steps], marker='.', color=color,
                   label=f'Contact {result.contact.index}')
        selected = result.selected_step
        if selected is not None:
            curve.scatter(100*(1-selected.fraction), 2*selected.fit.radius_mm,
                          marker='*', color=color, s=70, zorder=5)
            image.contour(selected.region_mask.astype(float), levels=[.5],
                          colors=[color], linewidths=1.5)
        row, column = result.contact.peak_rc
        image.scatter(column, row, color=color, edgecolor='white', s=35, zorder=5)
        image.annotate(str(result.contact.index), (column, row), xytext=(6, -6),
                       textcoords='offset points', color=color, fontweight='bold')
        if result.projection is not None:
            point = result.projection.xyz_mm
            center = selected.fit.center_xyz_mm
            lines.plot([point[0], center[0]], [point[2], center[2]], color=color, linewidth=2.2)
            lines.scatter(point[0], point[2], color=color, marker='x', s=50)
            lines.scatter(center[0], center[2], color=color, marker='o', s=25)
            lines.annotate(str(result.contact.index), (center[0], center[2]),
                           xytext=(4, 5), textcoords='offset points', color=color)
    curve.set(xlabel='Depth range from deepest point (% of peak)', ylabel='Sphere diameter (mm)',
              title='Sphere fitting as the contact region grows')
    curve.grid(alpha=.2)
    curve.legend(fontsize=8)
    # Display the displacement behind the labels and contour lines.
    values = np.ma.array(reconstruction.displacement, mask=~reconstruction.valid)
    limit = max(float(np.max(np.abs(values.compressed()))), 1e-6)
    shown = image.imshow(values, cmap='coolwarm', vmin=-limit, vmax=limit,
                         origin='upper', aspect='equal', zorder=-1)
    image.set(xlabel='Material grid column', ylabel='Material grid row', title='Contact regions')
    figure.colorbar(shown, ax=image, label='Normal displacement (mm)', shrink=.75)
    lines.set(xlabel='Camera X (mm)', ylabel='Camera Z (mm)',
              title='Center-to-surface segments projected onto X-Z')
    lines.grid(alpha=.2)
    figure.suptitle('Indentation depth and normal analysis')
    figure.savefig(path, dpi=160)
    figure.clear()


def _print_report(report, output):
    for contact in report['contacts']:
        step = contact['selected_fit']
        print(f"接触区 {contact['index']}：最大深度 {contact['max_depth_mm']:.4f} mm，"
              f"网格位置 {tuple(contact['peak_grid_rc'])}")
        if step is None:
            print(f"  球面拟合失败：{contact['error']}")
        else:
            print(f"  球直径 {step['diameter_mm']:.4f} mm，"
                  f"最低深度 {step['fraction']:.0%}，径向 RMS {step['radial_rms_mm']:.4f} mm")
            if contact['error']:
                print(f"  {contact['error']}")
    for pair in report['pairs']:
        a, b = pair['contacts']
        line = f"{a}–{b}：最大深度差 {pair['depth_difference_mm']:.4f} mm"
        if pair['angle_3d_degrees'] is not None:
            line += f"，法向夹角 {pair['angle_3d_degrees']:.3f}°"
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
        print(f"共同平面未确定：{report['line_analysis_error']}")
    print(f'结果目录：{output}')
    print('稳定结果仅表示局部扫描窗口；请检查完整球径变化曲线。')


def main(argv=None):
    parser = argparse.ArgumentParser(description='计算压入区域的最大深度差、球径和局部法向夹角')
    parser.add_argument('point_cloud', help='待分析的重建结果 NPZ 路径')
    parser.add_argument('--output-dir', help='输出目录；默认在输入文件旁创建 <文件名>_analysis')
    parser.add_argument('--count', type=int, default=3, help='接触区域数量，默认 3')
    parser.add_argument('--threshold-mm', type=float, default=.2,
                        help='识别独立压入区域的深度阈值（mm），默认 0.2')
    parser.add_argument('--min-region-points', type=int, default=100,
                        help='初始接触区域最少网格点数，默认 100')
    args = parser.parse_args(argv)
    source = Path(args.point_cloud).expanduser().resolve()
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir
              else source.parent / (source.stem + '_analysis'))
    try:
        reconstruction = load_reconstruction(source)
        contacts = find_contacts(reconstruction, count=args.count, threshold_mm=args.threshold_mm,
                                 min_region_points=args.min_region_points)
        results = analyze_contacts(reconstruction, contacts)
        report = _build_report(source, reconstruction, results, threshold_mm=args.threshold_mm,
                               min_region_points=args.min_region_points)
        output.mkdir(parents=True, exist_ok=True)
        _write_sweep(output/'depth_sweep.csv', results)
        _write_figure(output/'analysis.png', reconstruction, results)
        (output/'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                      default=_native, allow_nan=False)+'\n', encoding='utf-8')
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'分析失败：{exc}\n')
    _print_report(report, output)
    return 0 if report['complete'] else 1
