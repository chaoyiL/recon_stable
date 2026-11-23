"""Compare three fixed-radius contact fits with the archived scan fit."""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from experiments.indentation_analysis.depth_fitting import DepthScanOptions, find_convergence, scan_depths
from experiments.indentation_analysis.geometry import fit_sphere
from reconstruction.local_recon import LocalReconstructionResult


DEPTH_MM = .40
THRESHOLD_MM = .20
ANCHOR_RADIUS_PX = 30.
MIN_REGION_POINTS = 100
MIN_FIT_POINTS = 20
PAIRS = tuple(combinations(range(1, 4), 2))


def scan_contacts(result: LocalReconstructionResult, uv: np.ndarray, anchors: np.ndarray,
                  radii: tuple[float, ...], options: DepthScanOptions, method: str, *, anchor_window: bool = False):
    """Associate connected caps with the image anchors, optionally cropping before labelling."""
    displacement = result.displacement
    pressed = result.valid_mask & (displacement < -THRESHOLD_MM)
    structure = np.ones((3, 3))
    all_regions, _ = label(pressed, structure=structure)
    candidate_masks, fitted_masks = [], []
    contacts, rows, centers_by_depth = [], [], {}
    windows = np.linalg.norm(uv[..., None, :] - anchors, axis=-1) <= ANCHOR_RADIUS_PX
    photo_windows = result.photometric_valid_mask[..., None] & windows

    for index, radius in enumerate(radii, 1):
        window = photo_windows[..., index - 1]
        if anchor_window:
            regions, _ = label(pressed & windows[..., index - 1], structure=structure)
        else:
            regions = all_regions
        sizes = np.bincount(regions.ravel())
        overlap = np.bincount(regions[window], minlength=len(sizes))
        overlap[sizes < MIN_REGION_POINTS] = 0
        overlap[0] = 0
        selected = int(np.argmax(overlap))
        geometry = (regions == selected) & (selected > 0)
        candidates = geometry & result.photometric_valid_mask
        candidate_masks.append(candidates)
        fit_mask = np.zeros_like(candidates)
        fitted_masks.append(fit_mask)
        summary = dict(method=method, contact=index, radius_mm=radius, fit_depth_mm=DEPTH_MM,
                       status='contact_not_detected', geometric_region_count=int(geometry.sum()),
                       candidate_count=int(candidates.sum()), fit_count=0, radial_rmse_mm=None,
                       center_x_mm=None, center_y_mm=None, center_z_mm=None,
                       deepest_indentation_mm=max(0., -float(displacement[window].min())) if window.any() else None,
                       window_points_over_0p2_mm=int((window & (displacement < -THRESHOLD_MM)).sum()),
                       convergence_status='not_converged', converged_depth_mm=None, converged_rmse_mm=None)
        contacts.append(summary)
        if not candidates.any():
            continue
        if np.count_nonzero(np.count_nonzero(photo_windows[candidates], axis=0) >= MIN_REGION_POINTS) > 1:
            summary['status'] = 'shared_contact_region'
            continue
        if displacement[candidates].min() != displacement[candidates & window].min():
            summary['status'] = 'deepest_point_outside_anchor'
            continue

        points, signed = result.xyz_out[candidates], displacement[candidates].astype(np.float64)
        normals = result.reference_normals[candidates].astype(np.float64)
        levels = scan_depths(points, signed, normals, radius, step_mm=options.step_mm, min_fit_points=MIN_FIT_POINTS)
        centers_by_depth[index] = {sample.depth_mm: sample.fit.center_xyz_mm for sample in levels if sample.fit}
        for sample in levels:
            center = sample.fit.center_xyz_mm.tolist() if sample.fit else [None] * 3
            rows.append(dict(method=method, contact=index, radius_mm=radius, depth_mm=sample.depth_mm,
                             minimum_indentation_mm=sample.minimum_indentation_mm, sample_count=sample.sample_count,
                             radial_rmse_mm=sample.fit.radial_rms_mm if sample.fit else None,
                             center_x_mm=center[0], center_y_mm=center[1], center_z_mm=center[2],
                             status='ok' if sample.fit else sample.error))
        stable = find_convergence(levels, options=options)
        if stable is not None:
            summary.update(convergence_status='converged', converged_depth_mm=stable.depth_mm,
                           converged_rmse_mm=stable.fit.radial_rms_mm)
        chosen = signed - signed.min() <= DEPTH_MM
        summary['fit_count'] = int(chosen.sum())
        fit_mask[candidates] = chosen
        if chosen.sum() < MIN_FIT_POINTS:
            summary['status'] = 'insufficient_points'
            continue
        unit_normals = normals[chosen] / np.linalg.norm(normals[chosen], axis=1, keepdims=True)
        try:
            fit = fit_sphere(points[chosen], radius, side_normal=unit_normals.mean(axis=0))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            summary['status'] = str(error)
            continue
        x, y, z = fit.center_xyz_mm.tolist()
        summary.update(status='ok', radial_rmse_mm=fit.radial_rms_mm, center_x_mm=x, center_y_mm=y, center_z_mm=z)
    return contacts, rows, centers_by_depth, np.stack(candidate_masks), np.stack(fitted_masks)


def distance_rows(method: str, centers_by_depth: dict, scan_distances: dict) -> list[dict]:
    rows = []
    for first, second in PAIRS:
        left, right = centers_by_depth.get(first, {}), centers_by_depth.get(second, {})
        scan_distance = scan_distances[f'{first}-{second}']
        for depth in sorted(left.keys() & right.keys()):
            distance = float(np.linalg.norm(left[depth] - right[depth]))
            rows.append(dict(method=method, depth_mm=depth, contact_a=first, contact_b=second,
                             distance_mm=distance, scan_distance_mm=scan_distance, error_mm=distance - scan_distance))
    return rows


def write_rows(path: Path, rows: list[dict], fields: tuple[str, ...] = ()) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]) if rows else fields)
        writer.writeheader()
        writer.writerows(rows)
