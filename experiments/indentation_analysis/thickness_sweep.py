"""Reconstruct and fit three spheres at each assumed contact-layer thickness.

The five saved normal models and material identities remain fixed. Every
thickness receives new two-image sampling, normal inference and displacement
integration, followed by new contact detection and fixed-radius depth fits.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from utils.config import load_config
from .analysis import Reconstruction, find_contacts
from .depth_fitting import depth_scan_options, find_convergence, scan_depths
from .thickness_reconstruction import _source_record, prepare_study


ROOT = Path(__file__).resolve().parents[2]
CENTER_KEYS = ('center_x_mm', 'center_y_mm', 'center_z_mm')


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _reconstruction(result) -> Reconstruction:
    xyz = result.xyz_out.astype(np.float64)
    return Reconstruction(xyz, result.displacement.astype(np.float64),
                          xyz-result.displacement_vectors.astype(np.float64),
                          result.valid_mask, result.photometric_valid_mask)


def _components(reconstruction: Reconstruction, uv: np.ndarray, *, min_points: int,
                 depth_mm: float) -> list[dict]:
    contacts = find_contacts(reconstruction, count=None, threshold_mm=.2,
                             min_region_points=min_points)
    components = []
    for contact in contacts:
        candidate = contact.region_mask & reconstruction.photometric_valid
        if not candidate.any():
            continue
        signed = reconstruction.displacement[candidate]
        cap = candidate & (reconstruction.displacement-signed.min() <= depth_mm)
        components.append(dict(contact=contact, candidate=candidate,
                               pixel_center=uv[cap].astype(np.float64).mean(axis=0),
                               geometry_count=int(contact.region_mask.sum()),
                               candidate_count=int(candidate.sum())))
    return components


def _associate(components: list[dict], anchors: np.ndarray) -> tuple[dict, list[dict]]:
    """Match detected components by source-image location, preserving sphere IDs."""
    if not components:
        return {}, []
    pixels = np.array([item['pixel_center'] for item in components])
    costs = cdist(anchors, pixels)
    spacing = cdist(anchors, anchors)
    np.fill_diagonal(spacing, np.inf)
    limits = .4*spacing.min(axis=1)
    eligible = costs <= limits[:, None]
    matches, diagnostics = {}, []
    for component_index, item in enumerate(components):
        nearby_anchors = np.flatnonzero(eligible[:, component_index])
        index = int(nearby_anchors[0]) if len(nearby_anchors) == 1 else None
        status = 'ambiguous_contact' if len(nearby_anchors) > 1 else 'outside_anchor_windows'
        if index is not None:
            if np.count_nonzero(eligible[index]) != 1:
                status = 'ambiguous_contact'
            else:
                matches[index+1] = item
                status = 'matched'
        diagnostics.append(dict(
            component=component_index+1, assigned_contact=None if index is None else index+1,
            status=status, geometry_count=item['geometry_count'],
            candidate_count=item['candidate_count'], pixel_u_px=float(item['pixel_center'][0]),
            pixel_v_px=float(item['pixel_center'][1]),
            candidate_contact_ids=(nearby_anchors+1).tolist(),
            anchor_distance_px=None if index is None else float(costs[index, component_index]),
            anchor_limit_px=None if index is None else float(limits[index]),
        ))
    return matches, diagnostics


def _empty_sphere(method: str, thickness: float, index: int, radius: float, depth_mm: float) -> dict:
    return dict(thickness_mm=thickness, method=method, contact=index, radius_mm=radius,
                fit_depth_mm=depth_mm, status='contact_not_detected', error=None,
                geometric_region_count=0, candidate_count=0, fit_count=0,
                meets_100_region_points=False, meets_100_fit_points=False,
                deepest_indentation_mm=None, maximum_depth_range_mm=None,
                radial_rmse_mm=None, **dict.fromkeys(CENTER_KEYS), jacobian_condition=None,
                convergence_status='not_available', converged_depth_mm=None,
                converged_rmse_mm=None, converged_fit_count=None)


def _fit_contacts(result, uv: np.ndarray, anchors: np.ndarray, radii: tuple[float, ...],
                   options, *, method: str, thickness: float, depth_mm: float,
                   min_region_points: int, min_fit_points: int):
    reconstruction = _reconstruction(result)
    try:
        components = _components(reconstruction, uv, min_points=min_region_points, depth_mm=depth_mm)
    except ValueError as exc:
        components, detection_error = [], str(exc)
    else:
        detection_error = None
    matched, component_rows = _associate(components, anchors)
    summaries, scan_rows = [], []
    candidates = np.zeros((3, *result.valid_mask.shape), bool)
    selected_masks = np.zeros_like(candidates)
    for index, radius in enumerate(radii, start=1):
        row = _empty_sphere(method, thickness, index, radius, depth_mm)
        item = matched.get(index)
        if item is None:
            ambiguous = any(entry['status'] == 'ambiguous_contact' and index in entry['candidate_contact_ids']
                            for entry in component_rows)
            message = ('Source-image component association is not unique' if ambiguous
                       else 'No supported component inside the image anchor window')
            row.update(status='ambiguous_contact' if ambiguous else 'contact_not_detected',
                       error=detection_error or message)
            summaries.append(row)
            continue
        candidate = item['candidate']
        candidates[index-1] = candidate
        signed = reconstruction.displacement[candidate]
        xyz = reconstruction.xyz[candidate]
        normals = (xyz-reconstruction.baseline_xyz[candidate])/signed[:, None]
        row.update(geometric_region_count=item['geometry_count'], candidate_count=len(xyz),
                   meets_100_region_points=item['geometry_count'] >= 100,
                   deepest_indentation_mm=float(-signed.min()),
                   maximum_depth_range_mm=float(np.ptp(signed)))
        try:
            samples = scan_depths(xyz, signed, normals, radius, step_mm=options.step_mm,
                                  min_fit_points=min(min_fit_points, options.min_fit_points))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row.update(status='depth_scan_failed', error=str(exc))
            summaries.append(row)
            continue
        converged = find_convergence(samples, options=options)
        for sample in samples:
            visible_fit = sample.fit if sample.sample_count >= min_fit_points else None
            entry = dict(thickness_mm=thickness, method=method, contact=index,
                         radius_mm=radius, depth_mm=sample.depth_mm,
                         minimum_indentation_mm=sample.minimum_indentation_mm,
                         sample_count=sample.sample_count, radial_rmse_mm=None,
                         **dict.fromkeys(CENTER_KEYS), jacobian_condition=None,
                         status='ok' if visible_fit is not None else sample.error or 'insufficient_points')
            if visible_fit is not None:
                entry.update(zip(CENTER_KEYS, visible_fit.center_xyz_mm.tolist(), strict=True))
                entry.update(radial_rmse_mm=visible_fit.radial_rms_mm,
                             jacobian_condition=visible_fit.jacobian_condition)
            scan_rows.append(entry)
        row.update(geometric_region_count=item['geometry_count'], candidate_count=len(xyz),
                   meets_100_region_points=item['geometry_count'] >= 100,
                   deepest_indentation_mm=float(-signed.min()),
                   maximum_depth_range_mm=samples[-1].depth_mm,
                   convergence_status='not_converged' if converged is None else 'converged')
        if converged is not None:
            row.update(converged_depth_mm=converged.depth_mm,
                       converged_rmse_mm=converged.fit.radial_rms_mm,
                       converged_fit_count=converged.sample_count)
        selected = next((sample for sample in samples if np.isclose(
            sample.depth_mm, depth_mm, rtol=0., atol=1e-12)), None)
        if selected is None:
            row.update(status='depth_out_of_range')
        else:
            row.update(fit_count=selected.sample_count,
                       meets_100_fit_points=selected.sample_count >= 100)
            selected_masks[index-1] = candidate & (reconstruction.displacement-signed.min() <= depth_mm)
            if selected.fit is None or selected.sample_count < min_fit_points:
                row.update(status=selected.error or 'insufficient_points')
            else:
                row.update(status='ok', radial_rmse_mm=selected.fit.radial_rms_mm,
                           jacobian_condition=selected.fit.jacobian_condition)
                row.update(zip(CENTER_KEYS, selected.fit.center_xyz_mm.tolist(), strict=True))
        summaries.append(row)
    return summaries, scan_rows, component_rows, candidates, selected_masks


def _pair_rows(spheres: list[dict], design: float) -> list[dict]:
    rows = []
    for left, right in combinations(spheres, 2):
        adjacent = right['contact'] == left['contact']+1
        target = design if adjacent else None
        distance = None
        if left['status'] == right['status'] == 'ok':
            first = np.array([left[key] for key in CENTER_KEYS])
            second = np.array([right[key] for key in CENTER_KEYS])
            distance = float(np.linalg.norm(second-first))
        rows.append(dict(thickness_mm=left['thickness_mm'], method=left['method'],
                         contact_a=left['contact'], contact_b=right['contact'],
                         status='ok' if distance is not None else 'sphere_fit_unavailable',
                         center_distance_mm=distance, design_distance_mm=target,
                         design_error_mm=None if distance is None or target is None else distance-target,
                         design_error_percent=None if distance is None or target is None else 100*(distance/target-1),
                         fit_count_a=left['fit_count'], fit_count_b=right['fit_count'],
                         meets_100_fit_points=left['meets_100_fit_points'] and right['meets_100_fit_points']))
    return rows


def _save_progress(output: Path, report: dict, depth_rows: list[dict], component_rows: list[dict]) -> None:
    for filename, rows in (('reconstructions.csv', report['reconstructions']),
                            ('sphere_fits.csv', report['spheres']), ('depth_scans.csv', depth_rows),
                            ('sphere_center_distances.csv', report['distances']),
                            ('contact_components.csv', component_rows)):
        _write_csv(output/filename, rows)
    (output/'experiment.json').write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                     allow_nan=False)+'\n', encoding='utf-8')


def _archived_check(result, source: Path) -> dict:
    with np.load(source, allow_pickle=False) as archived:
        same = np.asarray(result.valid_mask) & archived['valid_mask']
        difference = result.xyz_out.astype(np.float64)-archived['xyz_out'].astype(np.float64)
        displacement = result.displacement.astype(np.float64)-archived['displacement'].astype(np.float64)
        report = dict(
            xyz_max_error_mm=float(np.max(np.abs(difference[same]))),
            displacement_max_error_mm=float(np.max(np.abs(displacement[same]))),
            valid_mask_equal=bool(np.array_equal(result.valid_mask, archived['valid_mask'])),
            photometric_mask_equal=bool(np.array_equal(result.photometric_valid_mask, archived['photometric_valid_mask'])),
        )
    if (report['xyz_max_error_mm'] > 1e-4 or report['displacement_max_error_mm'] > 1e-4
            or not report['valid_mask_equal'] or not report['photometric_mask_equal']):
        raise ValueError(f'{source}: 7 mm replay differs from archived reconstruction: {report}')
    return report


def _plot(path: Path, pairs: list[dict], spheres: list[dict], methods: tuple,
          design: float, depth_mm: float) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(18, 10), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    for axis, pair in zip(axes[0], ((1, 2), (2, 3), (1, 3)), strict=True):
        for mi, method in enumerate(methods):
            rows = sorted((row for row in pairs if row['method'] == method
                           and (row['contact_a'], row['contact_b']) == pair), key=lambda row: row['thickness_mm'])
            axis.plot([row['thickness_mm'] for row in rows],
                      [np.nan if row['center_distance_mm'] is None else row['center_distance_mm'] for row in rows],
                      marker='o', markersize=3, color=f'C{mi}', label=method)
            low = [row for row in rows if row['center_distance_mm'] is not None and not row['meets_100_fit_points']]
            axis.scatter([row['thickness_mm'] for row in low], [row['center_distance_mm'] for row in low],
                         s=45, facecolors='white', edgecolors=f'C{mi}', zorder=4)
        if pair != (1, 3):
            axis.axhline(design, color='black', linestyle='--', linewidth=1., label=f'{design:g} mm design')
        axis.axvline(7., color='#777777', linestyle=':', linewidth=1.)
        axis.set(title=f'Fitted sphere centers {pair[0]}–{pair[1]}', ylabel='3D center distance (mm)',
                 xlabel='Assumed marker-to-contact thickness h (mm)')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    for index, axis in enumerate(axes[1], start=1):
        for mi, method in enumerate(methods):
            rows = sorted((row for row in spheres if row['method'] == method and row['contact'] == index),
                          key=lambda row: row['thickness_mm'])
            axis.plot([row['thickness_mm'] for row in rows],
                      [np.nan if row['radial_rmse_mm'] is None else row['radial_rmse_mm'] for row in rows],
                      color=f'C{mi}', marker='o', markersize=3, label=method)
        axis.axvline(7., color='#777777', linestyle=':', linewidth=1.)
        axis.set(title=f'Sphere {index}: fixed-radius residual', ylabel='Radial RMSE (mm)',
                 xlabel='Assumed marker-to-contact thickness h (mm)', ylim=(0, None))
        axis.grid(alpha=.25)
    figure.suptitle(f'Thickness → image resampling → local reconstruction → sphere fitting | cap depth d={depth_mm:g} mm\n'
                    'Five frozen 7 mm calibration models; circles with white centers have fewer than 100 fit points')
    figure.savefig(path, dpi=180)
    figure.clear()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='改变重建厚度，重新采样并重建五种 LUT 点云，再拟合三球和计算球心距')
    parser.add_argument('--config', type=Path, default=ROOT/'config.yaml')
    parser.add_argument('--image', type=Path, default=ROOT/'assets/test_pics/3b/3b2.npz')
    parser.add_argument('--reference', type=Path, default=ROOT/'assets/reference.npz')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT/'assets/reconstruction/pc/3b2_batch')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'accurate_models/thickness_sweep')
    parser.add_argument('--thicknesses-mm', type=float, nargs='+', default=list(range(11)))
    parser.add_argument('--fit-depth-mm', type=float, default=.4)
    parser.add_argument('--min-region-points', type=int, default=20)
    parser.add_argument('--min-fit-points', type=int, default=20)
    parser.add_argument('--design-distance-mm', type=float, default=9.33)
    args = parser.parse_args(argv)
    offsets = sorted(set(args.thicknesses_mm))
    if (not offsets or not np.isfinite(offsets).all() or offsets[0] < 0
            or min(args.min_region_points, args.min_fit_points) < 4
            or not np.isfinite(args.fit_depth_mm) or args.fit_depth_mm <= 0
            or not np.isfinite(args.design_distance_mm) or args.design_distance_mm <= 0):
        parser.error('厚度取有限非负值；拟合深度取正值；点数下限至少为4')
    options = depth_scan_options(load_config(args.config))
    if not np.isclose(args.fit_depth_mm/options.step_mm, round(args.fit_depth_mm/options.step_mm),
                       rtol=0, atol=1e-10):
        parser.error('拟合深度须为扫描步长的正整数倍')
    radii = tuple(load_config(args.config)['indentation_analysis']['sphere_radii_mm'])
    if len(radii) != 3:
        parser.error('三球实验需要三个已知半径')
    study = prepare_study(args.config, args.image, args.reference, args.baseline_dir)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reconstruction_rows, sphere_rows, depth_rows, distance_rows, component_rows = [], [], [], [], []
    anchors, baseline_displacements, baseline_valid = {}, {}, {}
    report = dict(
        provenance=study.provenance, offsets_mm=offsets, sphere_radii_mm=radii,
        fit_depth_mm=args.fit_depth_mm, design_adjacent_distance_mm=args.design_distance_mm,
        depth_scan_options=asdict(options), curve_min_fit_points=args.min_fit_points,
        contact_threshold_mm=.2, component_min_points=args.min_region_points,
        selection='At h=7, use the three original >=100-point connected regions to establish image anchors. '
                  'At every h, redetect regions meeting component_min_points and uniquely match source-image cap centroids to those anchors. '
                  'Fit all photometrically valid points within d of that region deepest candidate.',
        point_count_interpretation='The curve uses its configured minimum point count. '
                                   'Separate flags retain the original 100-point region and convergence requirements.',
        experiment='Frozen learned models, native parameters, st and grid resolution; '
                   'both-image projection, color differences, curvature, normal inference and displacement are recomputed.',
        depth_interpretation='h is marker-to-contact thickness. d is the range from each newly reconstructed deepest candidate; '
                             'it is distinct from h and from absolute indentation.',
        reconstructions=reconstruction_rows, spheres=sphere_rows, distances=distance_rows,
        source_programs=[_source_record(Path(__file__), ROOT),
                         _source_record(Path(__file__).with_name('thickness_reconstruction.py'), ROOT)],
    )
    order = [7.] + [h for h in offsets if h != 7.]
    for thickness in order:
        keep = thickness in offsets
        for method in study.methods:
            print(f'h={thickness:g} mm / {method}: reconstruct and fit', flush=True)
            try:
                result = study.reconstruct(method, thickness)
            except (OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                if thickness == 7.:
                    raise
                reconstruction_rows.append(dict(thickness_mm=thickness, method=method,
                                                 status='reconstruction_failed', error=str(exc)))
                failed = [_empty_sphere(method, thickness, index, radius, args.fit_depth_mm)
                          for index, radius in enumerate(radii, start=1)]
                for row in failed:
                    row.update(status='reconstruction_failed', error=str(exc))
                sphere_rows.extend(failed)
                distance_rows.extend(_pair_rows(failed, args.design_distance_mm))
                _save_progress(output, report, depth_rows, component_rows)
                print(f'  reconstruction_failed: {exc}', flush=True)
                continue
            uv = study.last_current_uv
            reconstruction = _reconstruction(result)
            check = None
            if thickness == 7.:
                check = _archived_check(result, args.baseline_dir/f'{method}.npz')
                original = _components(reconstruction, uv, min_points=100, depth_mm=args.fit_depth_mm)
                if len(original) != 3:
                    raise ValueError(f'{method}: h=7 requires exactly three original contact regions')
                anchors[method] = np.array([item['pixel_center'] for item in original])
                baseline_displacements[method] = result.displacement.copy()
                baseline_valid[method] = result.valid_mask.copy()
                report.setdefault('archived_7mm_reproduction', {})[method] = check
            if not keep:
                continue
            case = output/f'h_{thickness:05.2f}mm'
            case.mkdir(parents=True, exist_ok=True)
            pointcloud = case/f'{method}.npz'
            result.save(pointcloud)
            spheres, scans, components, candidates, selected = _fit_contacts(
                result, uv, anchors[method], radii, options, method=method, thickness=thickness,
                depth_mm=args.fit_depth_mm, min_region_points=args.min_region_points,
                min_fit_points=args.min_fit_points)
            np.savez_compressed(case/f'{method}_fit_selection.npz', candidate_masks=candidates,
                                fit_masks=selected, source_uv=uv,
                                source_model_offset_mm=study.models[method].calibration_offset_mm)
            shared = result.valid_mask & baseline_valid[method] & np.isfinite(baseline_displacements[method])
            change = result.displacement[shared].astype(float)-baseline_displacements[method][shared].astype(float)
            reconstruction_rows.append(dict(
                thickness_mm=thickness, method=method, status='ok', error=None,
                pointcloud=str(pointcloud.relative_to(output)),
                model_training_offset_mm=study.models[method].calibration_offset_mm,
                grid_rows=result.valid_mask.shape[0], grid_columns=result.valid_mask.shape[1],
                geometry_valid_count=int(result.valid_mask.sum()),
                photometric_valid_count=int(result.photometric_valid_mask.sum()),
                model_valid_count=int(result.model_valid_mask.sum()),
                equation_valid_count=int(result.equation_valid_mask.sum()),
                solver_istop=result.solver_istop, solver_iterations=result.solver_iterations,
                solver_residual_norm=result.solver_residual_norm,
                displacement_change_from_h7_rms_mm=float(np.sqrt(np.mean(change**2))),
                detected_components=len(components), matched_spheres=sum(row['candidate_count'] > 0 for row in spheres),
                successful_sphere_fits=sum(row['status'] == 'ok' for row in spheres),
            ))
            sphere_rows.extend(spheres)
            depth_rows.extend(scans)
            distance_rows.extend(_pair_rows(spheres, args.design_distance_mm))
            component_rows.extend(dict(thickness_mm=thickness, method=method, **row) for row in components)
            _save_progress(output, report, depth_rows, component_rows)
            print(f"  fitted={sum(row['status'] == 'ok' for row in spheres)}/3; "
                  f"points={[row['fit_count'] for row in spheres]}", flush=True)
    report['image_anchors_uv_px'] = {method: value.tolist() for method, value in anchors.items()}
    _save_progress(output, report, depth_rows, component_rows)
    _plot(output/'thickness_vs_sphere_distance.png', distance_rows, sphere_rows, study.methods,
           args.design_distance_mm, args.fit_depth_mm)
    print(f'结果目录：{output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
