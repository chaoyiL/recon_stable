"""Scan sphere-fit RMSE, surface curvature and adjacent sphere distances."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from utils.config import load_config
from utils.surface_geometry import compute_surface_geometry
from .analysis import load_reconstruction
from .depth_fitting import DepthScanOptions, depth_scan_options, find_convergence, scan_depths

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config.yaml'


@dataclass(frozen=True)
class LocalCurvatures:
    """Per-query fit results; invalid curvatures and fit metrics are NaN.

    ``principal_curvatures`` has shape M×2, is ordered ascending, and has units
    of mm⁻¹. The remaining arrays have shape M; ``valid`` is boolean.
    ``neighbor_count`` counts support points within the 3-D search radius,
    including for invalid fits. ``fit_rmse_mm`` is the weighted height error,
    and ``fit_condition`` is the weighted design matrix's condition number.
    """

    principal_curvatures: np.ndarray
    valid: np.ndarray
    neighbor_count: np.ndarray
    fit_rmse_mm: np.ndarray
    fit_condition: np.ndarray


def _fit_local_curvatures(delta: np.ndarray, reference_normal: np.ndarray,
                          radius_mm: float) -> tuple[np.ndarray, float, float] | None:
    """Fit a normalized quadratic graph in an oriented, weighted PCA frame."""
    local = delta / radius_mm
    weights = np.exp(-2. * np.sum(local * local, axis=1))
    sqrt_weights = np.sqrt(weights)
    center = np.average(local, axis=0, weights=weights)
    try:
        _, spread, axes = np.linalg.svd((local - center) * sqrt_weights[:, None],
                                        full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if spread[0] <= 0 or spread[1] <= 1e-10 * spread[0]:
        return None

    tangent_u = axes[0]
    normal = axes[2]
    if normal @ reference_normal < 0:
        normal = -normal
    tangent_v = np.cross(normal, tangent_u)
    u = local @ tangent_u
    v = local @ tangent_v
    z = local @ normal
    design = np.column_stack((np.ones(len(local)), u, v, .5*u*u, u*v, .5*v*v))
    try:
        coefficients, _, rank, singular = np.linalg.lstsq(
            design * sqrt_weights[:, None], z * sqrt_weights, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 6 or singular[-1] <= 0:
        return None
    condition = singular[0] / singular[-1]
    if not np.isfinite(condition) or condition > 1e6:
        return None

    _, b, c, d, e, f = coefficients
    d, e, f = d / radius_mm, e / radius_mm, f / radius_mm
    metric = 1. + b*b + c*c
    mean = ((1. + c*c)*d - 2.*b*c*e + (1. + b*b)*f) / (2. * metric**1.5)
    gaussian = (d*f - e*e) / metric**2
    orientation = (normal - b*tangent_u - c*tangent_v) @ reference_normal
    if abs(orientation) <= 1e-12 * np.sqrt(metric):
        return None
    if orientation < 0:
        mean = -mean
    difference = np.sqrt(max(mean*mean - gaussian, 0.))
    curvatures = np.array([mean - difference, mean + difference])
    residual = z - design @ coefficients
    rmse = np.sqrt(np.average(residual*residual, weights=weights)) * radius_mm
    if not np.isfinite(curvatures).all() or not np.isfinite(rmse):
        return None
    return curvatures, float(rmse), float(condition)


def estimate_curvatures(support_points: np.ndarray, query_points: np.ndarray,
                        reference_normals: np.ndarray, *,
                        neighborhood_radius_mm: float = .5,
                        min_neighbors: int = 12) -> LocalCurvatures:
    """Estimate signed principal curvatures at queried XYZ points.

    ``support_points`` is finite N×3 observed XYZ in millimetres. The caller
    selects valid observations before calling. ``query_points`` is finite M×3
    XYZ in the same frame and ``reference_normals`` is M×3 finite, nonzero
    orientation vectors. The graph's oriented second fundamental form uses
    n·X''; a sphere has positive curvature when n points toward its center.

    Both PCA and a six-parameter quadratic fit use weights
    exp(-2*(distance/radius)²) in a 3-D Euclidean neighborhood. The graph is
    evaluated at the query's tangent coordinates. Fits with insufficient or
    collinear support, deficient rank, condition number above 1e6, or reference
    nearly perpendicular to the fitted normal (absolute cosine <= 1e-12) are marked
    invalid. Empty support or query arrays are accepted.
    """
    support = np.asarray(support_points, np.float64)
    queries = np.asarray(query_points, np.float64)
    normals = np.asarray(reference_normals, np.float64)
    if support.ndim != 2 or support.shape[1] != 3 or not np.isfinite(support).all():
        raise ValueError('support_points must be finite Nx3 XYZ points')
    if queries.ndim != 2 or queries.shape[1] != 3 or not np.isfinite(queries).all():
        raise ValueError('query_points must be finite Mx3 XYZ points')
    if (normals.shape != queries.shape or not np.isfinite(normals).all()
            or np.any(np.max(np.abs(normals), axis=1) == 0)):
        raise ValueError('reference_normals must be finite nonzero Mx3 vectors')
    if (not isinstance(neighborhood_radius_mm, Real)
            or isinstance(neighborhood_radius_mm, (bool, np.bool_))
            or not np.isfinite(neighborhood_radius_mm) or neighborhood_radius_mm <= 0):
        raise ValueError('neighborhood_radius_mm must be a finite positive real number')
    if (isinstance(min_neighbors, (bool, np.bool_))
            or not isinstance(min_neighbors, (int, np.integer)) or min_neighbors < 6):
        raise ValueError('min_neighbors must be an integer >= 6')

    # Rescaling avoids overflow or underflow when normalizing valid directions.
    normals = normals / np.max(np.abs(normals), axis=1, keepdims=True)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    radius = float(neighborhood_radius_mm)
    count = len(queries)
    principal = np.full((count, 2), np.nan)
    valid = np.zeros(count, dtype=bool)
    neighbor_count = np.zeros(count, dtype=int)
    rmse = np.full(count, np.nan)
    condition = np.full(count, np.nan)
    tree = cKDTree(support)
    for index, (point, reference) in enumerate(zip(queries, normals, strict=True)):
        neighbors = tree.query_ball_point(point, radius)
        neighbor_count[index] = len(neighbors)
        if len(neighbors) < min_neighbors:
            continue
        fit = _fit_local_curvatures(support[neighbors] - point, reference, radius)
        if fit is None:
            continue
        principal[index], rmse[index], condition[index] = fit
        valid[index] = True
    return LocalCurvatures(principal, valid, neighbor_count, rmse, condition)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray,
                        quantiles: tuple[float, ...]) -> np.ndarray:
    order = np.argsort(values)
    weights = weights[order]
    positions = (np.cumsum(weights)-.5*weights) / weights.sum()
    return np.interp(quantiles, positions, values[order])


def _curvature_statistics(selected, principal, valid, areas, radius, minimum):
    measured = selected & valid & (areas > 0)
    count = int(measured.sum())
    result = dict(candidate_count=int(selected.sum()), valid_count=count,
                  area_mm2=float(areas[measured].sum()),
                  gaussian_median_per_mm2=None, gaussian_q25_per_mm2=None, gaussian_q75_per_mm2=None,
                  gaussian_normalized_median=None, k_min_normalized_median=None,
                  k_max_normalized_median=None, negative_gaussian_area_fraction=None,
                  anisotropy_normalized_median=None, sphere_curvature_error_median=None)
    if count < minimum:
        return result
    k_min, k_max = principal[measured].T
    weights = areas[measured]
    gaussian = k_min*k_max
    q25, median, q75 = _weighted_quantiles(gaussian, weights, (.25, .5, .75))
    result.update(
        gaussian_median_per_mm2=float(median), gaussian_q25_per_mm2=float(q25), gaussian_q75_per_mm2=float(q75),
        gaussian_normalized_median=float(median*radius**2),
        k_min_normalized_median=float(_weighted_quantiles(k_min*radius, weights, (.5,))[0]),
        k_max_normalized_median=float(_weighted_quantiles(k_max*radius, weights, (.5,))[0]),
        negative_gaussian_area_fraction=float(weights[gaussian < 0].sum()/weights.sum()),
        anisotropy_normalized_median=float(_weighted_quantiles((k_max-k_min)*radius, weights, (.5,))[0]),
        sphere_curvature_error_median=float(_weighted_quantiles(
            np.sqrt(((radius*k_min-1)**2+(radius*k_max-1)**2)/2), weights, (.5,))[0]),
    )
    return result


def summarize_depth_curvature(offsets: np.ndarray, principal_curvatures: np.ndarray,
                              valid: np.ndarray, area_weights: np.ndarray, radius_mm: float,
                              depths: np.ndarray, *, bandwidth_mm: float = .1,
                              min_band_points: int = 8) -> list[dict]:
    """Area-weight local depth bands and cumulative subsets of measured curvature.

    offsets = displacement - minimum candidate displacement, in mm. Principal
    curvatures are ordered k_min <= k_max in mm^-1 with baseline-positive
    orientation. Areas are reconstructed-surface areas in mm². Curvature at d
    uses centers in [d-bandwidth/2, d+bandwidth/2]; cumulative uses offsets<=d.
    Both statistics require min_band_points finite, valid, positive-area points.
    """
    offsets = np.asarray(offsets, np.float64)
    principal = np.asarray(principal_curvatures, np.float64)
    valid = np.asarray(valid, bool)
    areas = np.asarray(area_weights, np.float64)
    depths = np.asarray(depths, np.float64)
    if (offsets.ndim != 1 or principal.shape != (len(offsets), 2)
            or valid.shape != offsets.shape or areas.shape != offsets.shape
            or depths.ndim != 1 or len(depths) == 0):
        raise ValueError('Require N offsets/valid/areas, Nx2 principal curvatures and 1D depths')
    if (not all(np.isfinite(value).all() for value in (offsets, areas, depths))
            or np.any(offsets < 0) or np.any(areas < 0) or np.any(depths < 0)
            or not np.isfinite(principal[valid]).all()
            or np.any(principal[valid, 0] > principal[valid, 1])):
        raise ValueError('Require finite nonnegative depths/areas and finite ordered valid curvatures')
    for name, value in (('radius_mm', radius_mm), ('bandwidth_mm', bandwidth_mm)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) \
                or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if isinstance(min_band_points, (bool, np.bool_)) \
            or not isinstance(min_band_points, (int, np.integer)) or min_band_points < 1:
        raise ValueError('min_band_points must be an integer >= 1')
    rows = []
    for depth in depths:
        row = dict(depth_mm=float(depth))
        for prefix, selected in (('band', np.abs(offsets-depth) <= bandwidth_mm/2),
                                 ('cumulative', offsets <= depth)):
            stats = _curvature_statistics(selected, principal, valid, areas, radius_mm, min_band_points)
            row.update({f'{prefix}_{key}': value for key, value in stats.items()})
        rows.append(row)
    return rows


def _scan_method(report_path: Path, *, options: DepthScanOptions, min_fit_points: int):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    method = report_path.parent.name.removesuffix('_analysis')
    source = report_path.parent.parent / f'{method}.npz'
    reconstruction = load_reconstruction(source)
    selection_path = report_path.parent / 'fit_selection.npz'
    with np.load(selection_path, allow_pickle=False) as data:
        candidates = data['candidate_masks']
    expected_shape = (len(report['contacts']), *reconstruction.valid.shape)
    if candidates.shape != expected_shape or candidates.dtype != np.bool_:
        raise ValueError(f'{selection_path}: candidate masks must be boolean with shape {expected_shape}')

    rows, summaries = [], []
    for candidate, contact in zip(candidates, report['contacts'], strict=True):
        if np.any(candidate & ~reconstruction.photometric_valid):
            raise ValueError(f'{method}: candidate masks must contain photometrically valid points')
        if int(candidate.sum()) != contact['selection']['candidate_count']:
            raise ValueError(f'{method}: saved candidate count disagrees with analysis.json')
        points = reconstruction.xyz[candidate]
        displacement = reconstruction.displacement[candidate]
        normals = (points-reconstruction.baseline_xyz[candidate]) / displacement[:, None]
        radius = contact['sphere_radius_mm']
        # The curve threshold can differ from the convergence threshold.
        # Fit at the lower threshold so both use the same complete depth scan.
        samples = scan_depths(points, displacement, normals, radius, step_mm=options.step_mm,
                              min_fit_points=min(min_fit_points, options.min_fit_points))
        converged = find_convergence(samples, options=options)
        converged_center = None if converged is None else converged.fit.center_xyz_mm
        offsets = displacement-displacement.min()
        for sample in samples:
            fit = sample.fit if sample.sample_count >= min_fit_points else None
            row = dict(method=method, contact=contact['index'], radius_mm=radius,
                       depth_mm=sample.depth_mm,
                       minimum_indentation_mm=sample.minimum_indentation_mm,
                       sample_count=sample.sample_count,
                       converged_fit_points_in_subset=None if converged is None else int(np.sum(
                           (offsets <= converged.depth_mm) & (offsets <= sample.depth_mm))),
                       radial_rmse_mm=None, center_x_mm=None, center_y_mm=None, center_z_mm=None,
                       center_shift_from_converged_mm=None, jacobian_condition=None,
                       status='ok' if fit is not None else
                              'insufficient_points' if sample.sample_count < min_fit_points else sample.error)
            if fit is not None:
                row.update(radial_rmse_mm=fit.radial_rms_mm,
                           jacobian_condition=fit.jacobian_condition)
                row.update(zip(('center_x_mm', 'center_y_mm', 'center_z_mm'),
                               fit.center_xyz_mm.tolist(), strict=True))
                if converged_center is not None:
                    row['center_shift_from_converged_mm'] = float(np.linalg.norm(
                        fit.center_xyz_mm-converged_center))
            rows.append(row)
        converged_rmse = None if converged is None else converged.fit.radial_rms_mm
        summaries.append(dict(
            method=method, contact=contact['index'], radius_mm=radius, source=str(source),
            selection_source=str(selection_path), analysis_source=str(report_path),
            detection_threshold_mm=report['method']['detection_threshold_mm'],
            candidate_count=int(candidate.sum()),
            convergence_status='not_converged' if converged is None else 'converged',
            converged_depth_mm=None if converged is None else converged.depth_mm,
            converged_rmse_mm=converged_rmse,
            converged_mse_mm2=None if converged_rmse is None else converged_rmse**2,
            converged_fit_count=None if converged is None else converged.sample_count,
            convergence_options=asdict(options),
            deepest_indentation_mm=float(-displacement.min()),
            shallowest_indentation_mm=float(-displacement.max()),
            maximum_depth_range_mm=samples[-1].depth_mm,
            scan_levels=len(samples),
            successful_fits=sum(sample.fit is not None and sample.sample_count >= min_fit_points
                                for sample in samples),
        ))
    return rows, summaries


def _analyze_method(batch: Path, method: str, contacts: list[dict], source_rows: list[dict], *,
                    neighborhood_radius_mm: float, min_neighbors: int,
                    bandwidth_mm: float, min_band_points: int):
    reconstruction = load_reconstruction(batch/f'{method}.npz')
    with np.load(batch/f'{method}_analysis/fit_selection.npz', allow_pickle=False) as data:
        candidates = data['candidate_masks'].astype(bool)
    if candidates.shape != (len(contacts), *reconstruction.valid.shape):
        raise ValueError(f'{method}: candidate mask shape disagrees with depth scan')
    target = np.any(candidates, axis=0)
    if np.any(target & ~reconstruction.photometric_valid):
        raise ValueError(f'{method}: curvature centers must be photometrically valid')
    points = reconstruction.xyz[target]
    displacement = reconstruction.displacement[target]
    reference = (points-reconstruction.baseline_xyz[target]) / displacement[:, None]
    estimates = estimate_curvatures(
        reconstruction.xyz[reconstruction.photometric_valid], points, reference,
        neighborhood_radius_mm=neighborhood_radius_mm, min_neighbors=min_neighbors)
    geometry = compute_surface_geometry(reconstruction.xyz, reconstruction.valid)
    shape = reconstruction.valid.shape
    principal = np.full((*shape, 2), np.nan)
    principal[target] = estimates.principal_curvatures
    valid = np.zeros(shape, bool)
    valid[target] = estimates.valid
    valid &= geometry['valid']
    principal[~valid] = np.nan
    neighbors = np.zeros(shape, np.int32)
    neighbors[target] = estimates.neighbor_count
    fit_rmse, condition = np.full(shape, np.nan), np.full(shape, np.nan)
    fit_rmse[target], condition[target] = estimates.fit_rmse_mm, estimates.fit_condition
    maps = dict(principal_curvatures=principal, valid=valid, neighbor_count=neighbors,
                fit_rmse_mm=fit_rmse, fit_condition=condition, area_mm2=geometry['area'])

    rows, summaries = [], []
    for contact in contacts:
        index, radius = contact['contact'], contact['radius_mm']
        candidate = candidates[index-1]
        signed = reconstruction.displacement[candidate]
        offsets = signed-signed.min()
        rmse_rows = [row for row in source_rows if row['method'] == method and row['contact'] == index]
        if len(signed) != contact['candidate_count'] or not np.isclose(
                -signed.min(), contact['deepest_indentation_mm'], rtol=0., atol=1e-10):
            raise ValueError(f'{method} contact {index}: candidate points disagree with saved depth scan')
        statistics = summarize_depth_curvature(
            offsets, principal[candidate], valid[candidate], geometry['area'][candidate], radius,
            np.array([row['depth_mm'] for row in rmse_rows]), bandwidth_mm=bandwidth_mm,
            min_band_points=min_band_points)
        for rmse_row, stats in zip(rmse_rows, statistics, strict=True):
            if stats['cumulative_candidate_count'] != rmse_row['sample_count']:
                raise ValueError(f'{method} contact {index}: cumulative depth masks disagree')
            rows.append(dict(rmse_row, **stats))
        measured = candidate & valid
        summaries.append(dict(
            method=method, contact=index, radius_mm=radius, candidate_count=int(candidate.sum()),
            valid_curvature_count=int(measured.sum()), deepest_indentation_mm=contact['deepest_indentation_mm'],
            median_neighbor_count=float(np.median(neighbors[measured])) if measured.any() else None,
            median_patch_fit_rmse_mm=float(np.median(fit_rmse[measured])) if measured.any() else None,
            median_patch_condition=float(np.median(condition[measured])) if measured.any() else None,
        ))
    return rows, summaries, maps


def _plot_curves(path: Path, rows: list[dict], summaries: list[dict], *,
                 distance_rows: list[dict],
                 min_fit_points: int, depth_limits: dict[int, float],
                 neighborhood_radius_mm: float, bandwidth_mm: float) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    contacts = sorted({row['contact'] for row in summaries})
    methods = list(dict.fromkeys(row['method'] for row in summaries))
    pairs = sorted({(row['contact_a'], row['contact_b']) for row in distance_rows})
    figure = Figure(figsize=(6*len(contacts), 13 if pairs else 10), layout='constrained')
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(4 if pairs else 3, len(contacts))
    for column, contact in enumerate(contacts):
        radius = next(row['radius_mm'] for row in summaries if row['contact'] == contact)
        limit = depth_limits.get(contact)
        rmse_axis = figure.add_subplot(grid[0, column])
        curvature_axis = figure.add_subplot(grid[1, column], sharex=rmse_axis)
        count_axis = figure.add_subplot(grid[-1, column], sharex=rmse_axis)
        for mi, method in enumerate(methods):
            full_series = [row for row in rows if row['method'] == method and row['contact'] == contact]
            summary = next(row for row in summaries
                           if row['method'] == method and row['contact'] == contact)
            series = [row for row in full_series if limit is None or row['depth_mm'] <= limit]
            depth = [row['depth_mm'] for row in series]
            color = f'C{mi}'
            values = {key: [np.nan if row[key] is None else row[key] for row in series]
                      for key in ('radial_rmse_mm', 'band_gaussian_normalized_median')}
            selected_depth = summary['converged_depth_mm']
            status = ('not converged' if selected_depth is None
                      else f'converged d={selected_depth:.3g} mm')
            rmse_axis.plot(depth, values['radial_rmse_mm'], color=color, linewidth=1.6,
                           label=f"{method.replace('mlp', 'm')}: {status}")
            if selected_depth is not None and (limit is None or selected_depth <= limit):
                rmse_axis.scatter([selected_depth], [summary['converged_rmse_mm']],
                                  color=color, marker='*', s=70, zorder=3)
            curvature_axis.plot(depth, values['band_gaussian_normalized_median'], color=color, linewidth=1.6)
            count_axis.plot(depth, [row['sample_count'] for row in series], color=color, linewidth=1.4)
        title = f'Contact {contact} | R = {radius:g} mm'
        if limit is not None:
            title += f' | d <= {limit:g} mm'
        rmse_axis.set(title=title, ylabel='Radial RMSE (mm)', ylim=(0, None))
        count_axis.axhline(min_fit_points, color='#777777', linewidth=.8, linestyle=':')
        count_axis.set(xlabel='d: depth range from deepest indentation (mm)',
                       ylabel='Included points', ylim=(0, None), xlim=(0, limit))
        curvature_axis.axhline(1., color='#555555', linestyle=':', linewidth=1.)
        curvature_axis.axhline(0., color='#888888', linewidth=.7)
        curvature_axis.set_ylabel('Band median Gaussian curvature: K R²')
        for axis in (rmse_axis, curvature_axis):
            axis.tick_params(labelbottom=False)
        for axis in (rmse_axis, curvature_axis, count_axis):
            axis.grid(alpha=.25)
        rmse_axis.legend(fontsize=9)

    if pairs:
        distance_grid = grid[2, :].subgridspec(1, len(pairs))
        for column, (left, right) in enumerate(pairs):
            axis = figure.add_subplot(distance_grid[0, column])
            limit = min((depth_limits[index] for index in (left, right) if index in depth_limits),
                        default=None)
            for mi, method in enumerate(methods):
                series = [row for row in distance_rows
                          if row['method'] == method and row['contact_a'] == left
                          and row['contact_b'] == right
                          and (limit is None or row['depth_mm'] <= limit)]
                axis.plot([row['depth_mm'] for row in series],
                          [np.nan if row['center_distance_mm'] is None else row['center_distance_mm']
                           for row in series], color=f'C{mi}', linewidth=1.6,
                          label=method.replace('mlp', 'm'))
            axis.set(title=f'Contacts {left}–{right}', ylabel='Sphere-center distance (mm)',
                     xlabel='d: depth range from each deepest indentation (mm)', xlim=(0, limit))
            axis.grid(alpha=.25)
            axis.legend(fontsize=9)
    title = ('Sphere-fit RMSE, Gaussian curvature and adjacent sphere distances' if pairs
             else 'Sphere-fit RMSE and local Gaussian curvature')
    figure.suptitle(title+'\n'
                   f'Spatial radius = {neighborhood_radius_mm:g} mm | depth band = {bandwidth_mm:g} mm', fontsize=11)
    figure.savefig(path, dpi=180)
    figure.clear()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _adjacent_sphere_distances(rows: list[dict]) -> list[dict]:
    """Measure adjacent contacts at shared depth levels in camera XYZ, in mm.

    Rows come from one fixed-step depth scan. Each contact measures depth from
    its own deepest candidate. Only equal depth levels are paired; individual
    full-range endpoints participate when both contacts have that exact level.
    Failed fits retain a null scalar distance, producing a gap in the curve.
    """
    grouped = {}
    for row in rows:
        contacts = grouped.setdefault(row['method'], {})
        contacts.setdefault(row['contact'], {})[row['depth_mm']] = row
    distances = []
    for method, contacts in grouped.items():
        for left in sorted(contacts):
            right = left+1
            if right not in contacts:
                continue
            for depth in sorted(contacts[left].keys() & contacts[right].keys()):
                first, second = contacts[left][depth], contacts[right][depth]
                distance = None
                if first['status'] == second['status'] == 'ok':
                    delta = [second[key]-first[key]
                             for key in ('center_x_mm', 'center_y_mm', 'center_z_mm')]
                    distance = float(np.linalg.norm(delta))
                distances.append(dict(method=method, contact_a=left, contact_b=right,
                                      depth_mm=depth, center_distance_mm=distance))
    return distances


def _depth_report(rows: list[dict], contacts: list[dict], *, options: DepthScanOptions,
                  min_fit_points: int, depth_limits: dict[int, float]) -> dict:
    report = dict(
        definition='D_i=-displacement_i; D_max=max(D_i) over photometric candidate points; '
                   'd increases from 0 to D_max-min(D_i); include D_i >= D_max-d',
        coordinate_frame='camera XYZ in millimetres',
        depth_coordinate='normal indentation relative to each point baseline; unit mm',
        fitting='independent fixed-radius equal-weight radial least squares at each depth',
        selection='all saved candidate points within depth range; nested subsets',
        rmse_definition='sqrt(mean((norm(P_i-C_d)-R)^2)) on the included points',
        convergence='first regular window of successful fits with newly added points and stable RMSE',
        step_mm=options.step_mm, min_fit_points=min_fit_points,
        convergence_options=asdict(options),
        plot_depth_limits_mm=depth_limits, contacts=contacts, samples=rows,
    )
    distances = _adjacent_sphere_distances(rows)
    if distances:
        report.update(
            adjacent_sphere_distance_definition='norm(C_(i+1)(d)-C_i(d)) in mm; '
                                                'contacts follow material-grid order; same method and exact d; '
                                                'each contact uses its own deepest point; failed fits give null',
            adjacent_sphere_distances=distances,
        )
    return report


def _parse_args(argv):
    parser = argparse.ArgumentParser(description='一次完成固定半径 RMSE、曲率和相邻球心距离的深度扫描与联合总图')
    parser.add_argument('batch_dir', type=Path, help='包含重建 NPZ 和各方法 *_analysis 的 batch 目录')
    parser.add_argument('--output-dir', type=Path, help='默认 batch_dir/analysis')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH,
                        help='平台判据配置文件，默认使用项目 config.yaml')
    parser.add_argument('--step-mm', type=float, help='深度扫描步长，默认使用配置')
    parser.add_argument('--min-fit-points', type=int,
                        help='曲线各步球面拟合最少点数，默认使用配置；平台点数阈值由配置决定')
    parser.add_argument('--neighborhood-radius-mm', type=float, default=.5, help='曲率拟合三维邻域半径，默认 0.5 mm')
    parser.add_argument('--depth-band-mm', type=float, default=.1, help='局部深度统计带宽，默认 0.1 mm')
    parser.add_argument('--min-neighbors', type=int, default=12)
    parser.add_argument('--min-band-points', type=int, default=8)
    parser.add_argument('--upper-plot-depth-mm', type=float,
                        help='覆盖第一个球冠展示上限；R=2 默认 0.8 mm，其余尺寸自适应；数据保留完整扫描')
    args = parser.parse_args(argv)
    if args.upper_plot_depth_mm is not None and (
            not np.isfinite(args.upper_plot_depth_mm) or args.upper_plot_depth_mm <= 0):
        parser.error('--upper-plot-depth-mm 需要有限正数')
    return parser, args


def main(argv=None) -> int:
    parser, args = _parse_args(argv)
    batch = args.batch_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() if args.output_dir else batch/'analysis'
    reports = sorted(batch.glob('*_analysis/analysis.json'),
                     key=lambda path: (path.parent.name != 'table_analysis', path.parent.name))
    if not reports:
        parser.error(f'{batch}: 需要各方法的 *_analysis/analysis.json 和 fit_selection.npz')
    depth_rows, depth_contacts, curvature_rows, curvature_contacts, maps = [], [], [], [], {}
    try:
        options = depth_scan_options(load_config(args.config))
        if args.step_mm is not None:
            options = replace(options, step_mm=args.step_mm)
        min_fit_points = options.min_fit_points if args.min_fit_points is None else args.min_fit_points
        if min_fit_points < 4:
            raise ValueError('min-fit-points must be an integer >= 4')
        for report_path in reports:
            method = report_path.parent.name.removesuffix('_analysis')
            method_rows, method_contacts = _scan_method(
                report_path, options=options, min_fit_points=min_fit_points)
            method_curvature, method_summaries, method_maps = _analyze_method(
                batch, method, method_contacts, method_rows,
                neighborhood_radius_mm=args.neighborhood_radius_mm, min_neighbors=args.min_neighbors,
                bandwidth_mm=args.depth_band_mm, min_band_points=args.min_band_points)
            depth_rows.extend(method_rows)
            depth_contacts.extend(method_contacts)
            curvature_rows.extend(method_curvature)
            curvature_contacts.extend(method_summaries)
            maps.update({f'{method}_{key}': value for key, value in method_maps.items()})
            print(f'{method}: {len(method_rows)} 个扫描层，'
                  f'{sum(row["status"] == "ok" for row in method_rows)} 次球面拟合成功，'
                  f'{sum(row["valid_curvature_count"] for row in method_summaries)} 个有效曲率点', flush=True)
            for contact in method_contacts:
                if contact['convergence_status'] == 'not_converged':
                    print(f'{method} contact {contact["contact"]}: 深度扫描未收敛，'
                          '可调整配置 indentation_analysis.depth_scan 的稳定阈值', flush=True)
        limits = {contact['contact']: .8 for contact in depth_contacts if contact['radius_mm'] == 2.}
        if args.upper_plot_depth_mm is not None:
            limits[1] = args.upper_plot_depth_mm
        depth_report = _depth_report(depth_rows, depth_contacts, options=options,
                                     min_fit_points=min_fit_points, depth_limits=limits)
        curvature_report = dict(
            source_depth_sweep=str(output/'depth_sweep.json'), depth_definition=depth_report['definition'],
            estimator='Gaussian-distance-weighted local quadratic Monge patch in a weighted PCA tangent frame',
            neighborhood_radius_mm=args.neighborhood_radius_mm,
            distance_weight_sigma_mm=args.neighborhood_radius_mm/2, min_neighbors=args.min_neighbors,
            support='all geometrically and photometrically valid observed XYZ points; centers in saved contact candidates',
            orientation='local normal aligned with baseline positive normal; sphere cap k_min=k_max=+1/R',
            curvature_units='principal curvature mm^-1; Gaussian curvature mm^-2',
            gaussian_definition='K=k_min*k_max; theoretical sphere K*R^2=1',
            principal_definition='eigenvalues of second fundamental form relative to first, including fitted first derivatives',
            depth_band_mm=args.depth_band_mm, min_band_points=args.min_band_points,
            band_definition='centers with abs((displacement-min(displacement))-d)<=depth_band_mm/2',
            cumulative_definition='centers with displacement-min(displacement)<=d',
            statistics='reconstructed-surface-area-weighted quantiles; negative-K fraction uses surface area',
            scale_interpretation='finite spatial fitting radius smooths changes and introduces quadratic-approximation bias; '
                                 'depth bandwidth groups patch centers and is separate from spatial resolution',
            plot_depth_limits_mm=limits, contacts=curvature_contacts, samples=curvature_rows,
        )
        output.mkdir(parents=True, exist_ok=True)
        _write_csv(output/'depth_sweep.csv', depth_rows)
        _write_csv(output/'curvature_depth_sweep.csv', curvature_rows)
        for filename, report in (('depth_sweep.json', depth_report),
                                 ('curvature_depth_sweep.json', curvature_report)):
            (output/filename).write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        methods = list(dict.fromkeys(contact['method'] for contact in depth_contacts))
        np.savez_compressed(output/'curvature_maps.npz', methods=np.array(methods), **maps)
        _plot_curves(output/'rmse_curvature_vs_depth.png', curvature_rows, depth_contacts,
                     distance_rows=depth_report.get('adjacent_sphere_distances', []),
                     min_fit_points=min_fit_points, depth_limits=limits,
                     neighborhood_radius_mm=args.neighborhood_radius_mm, bandwidth_mm=args.depth_band_mm)
        (output/'rmse_vs_depth.png').unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f'联合深度扫描失败：{exc}\n')
    print(f'结果目录：{output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
