"""Evaluate a reconstructed contact against one shared convex hexagon.

Grid coordinates are (row, column); XYZ, displacement and lengths are in mm.
Profile distances sum 3D segments on the baseline or reconstructed surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from utils.surface_geometry import compute_surface_geometry
from utils.surface_sampling import bilinear_sample


@dataclass(frozen=True)
class HexagonEvaluation:
    metrics: dict[str, Any]
    target_mask: np.ndarray
    predicted_mask: np.ndarray
    low_mask: np.ndarray | None = None
    high_mask: np.ndarray | None = None
    normalized_response: np.ndarray | None = None


def _hexagon_mask(vertices_rc: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    vertices = np.asarray(vertices_rc, np.float64)
    if vertices.shape != (6, 2) or not np.isfinite(vertices).all():
        raise ValueError('vertices_rc must be six finite (row, column) points')
    if np.any(vertices < 0) or np.any(vertices >= np.asarray(shape) - 1):
        raise ValueError('Hexagon vertices must lie within the surface grid')
    edges = np.roll(vertices, -1, axis=0) - vertices
    turn = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(np.linalg.norm(edges, axis=1) < 1e-6) or not (
        np.all(turn > 1e-8) or np.all(turn < -1e-8)
    ):
        raise ValueError('Hexagon vertices must be ordered and strictly convex')
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(vertices[:, ::-1]).astype(np.int32)], 1)
    return mask.astype(bool)


def _component(mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    overlap = np.bincount(labels[target].ravel(), minlength=count)
    overlap[0] = 0
    if overlap.max() == 0:
        return np.zeros_like(mask)
    return labels == int(np.argmax(overlap))


def _weighted_iou(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> float | None:
    union = first | second
    denominator = float(weights[union].sum())
    if denominator <= 0:
        return None
    return float(weights[first & second].sum() / denominator)


def _crossings(values: np.ndarray, level: float) -> np.ndarray:
    """Fractional sample indices where a profile passes a level outward."""
    index = np.flatnonzero((values[:-1] >= level) & (values[1:] < level))
    if not len(index):
        return np.empty(0, np.float64)
    fraction = (values[index] - level) / (values[index] - values[index + 1])
    return index + fraction


def _profile(
    rc: np.ndarray, outward_rc: np.ndarray, half_width_mm: float,
    surface_and_height: np.ndarray, support: np.ndarray, levels: tuple[float, float, float],
) -> tuple[float, float | None, float | None, np.ndarray] | None:
    # The parameter-space direction is locally unit length in the baseline 3D metric.
    half_samples = max(40, min(200, int(np.ceil(np.linalg.norm(outward_rc) * half_width_mm * 4))))
    offsets = np.linspace(-half_width_mm, half_width_mm, 2 * half_samples + 1)
    samples = bilinear_sample(surface_and_height, (rc + offsets[:, None] * outward_rc)[:, ::-1], support)
    if not samples.valid[half_samples]:
        return None
    first = half_samples
    while first > 0 and samples.valid[first - 1]:
        first -= 1
    last = half_samples + 1
    while last < len(offsets) and samples.valid[last]:
        last += 1
    if last - first < 3:
        return None
    values = samples.values[first:last]
    output_length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1))]
    baseline_length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(values[:, 3:6], axis=0), axis=1))]
    sample_indices = np.arange(len(values), dtype=float)
    low, middle, high = (_crossings(values[:, 6], level) for level in levels)
    if not len(middle):
        return None
    middle_distance = np.interp(middle, sample_indices, baseline_length) - baseline_length[half_samples - first]
    middle = float(middle[np.argmin(np.abs(middle_distance))])
    high = high[high <= middle]
    low = low[low >= middle]
    baseline_width = output_width = None
    if len(high) and len(low):
        endpoints = np.array([high.max(), low.min()])
        baseline_width = float(np.diff(np.interp(endpoints, sample_indices, baseline_length))[0])
        output_width = float(np.diff(np.interp(endpoints, sample_indices, output_length))[0])
    offset = float(np.interp(middle, sample_indices, baseline_length) - baseline_length[half_samples - first])
    return offset, baseline_width, output_width, samples.values[half_samples, 3:6]


def _side_profiles(
    start: np.ndarray, end: np.ndarray, centroid: np.ndarray,
    baseline: np.ndarray, surface_and_height: np.ndarray, support: np.ndarray,
    levels: tuple[float, float, float], profiles_per_side: int, half_width_mm: float | None,
) -> dict[str, Any]:
    def empty_side(status: str, width: float | None = None) -> dict[str, Any]:
        return dict(
            status=status, profiles_requested=profiles_per_side, profiles_with_midpoint=0,
            profiles_with_width=0, profile_half_width_mm=width, signed_offset_mean_mm=None,
            reference_deviation_rms_mm=None, straightness_rms_mm=None,
            transition_width_median_mm=None, transition_width_p90_mm=None,
            surface_transition_length_median_mm=None, surface_transition_length_p90_mm=None,
        )

    side = end - start
    midpoint = (start + end) / 2
    if half_width_mm is None:
        rc = midpoint + np.linspace(0, 1, 33)[:, None] * (centroid - midpoint)
        path = bilinear_sample(baseline, rc[:, ::-1], support)
        if not path.valid.all():
            return empty_side('insufficient_reference_support')
        half_width = .7 * float(np.linalg.norm(np.diff(path.values, axis=0), axis=1).sum())
    else:
        half_width = half_width_mm
    if half_width <= 1e-9:
        return empty_side('degenerate_reference_side', half_width)

    offsets, widths, output_widths, side_points = [], [], [], []
    for fraction in np.linspace(.2, .8, profiles_per_side):
        rc = start + fraction * side
        point = bilinear_sample(baseline, rc[::-1], support, with_jacobian=True)
        if not point.valid:
            continue
        # Solve t^T G v = 0, where G is the baseline 3D metric in row/column coordinates.
        jacobian = point.jacobian[:, ::-1]
        metric = jacobian.T @ jacobian
        if np.linalg.det(metric) <= 1e-14:
            continue
        outward = np.linalg.solve(metric, np.array([-side[1], side[0]]))
        if np.dot(outward, centroid - rc) > 0:
            outward = -outward
        speed = float(np.linalg.norm(jacobian @ outward))
        if speed <= 1e-12:
            continue
        outward /= speed
        result = _profile(rc, outward, half_width, surface_and_height, support, levels)
        if result is None:
            continue
        offset, width, output_width, point_xyz = result
        offsets.append(offset)
        side_points.append(point_xyz)
        if width is not None:
            widths.append(width)
            output_widths.append(output_width)

    minimum_profiles = max(3, (profiles_per_side + 1) // 2)
    status = ('ok' if len(offsets) >= minimum_profiles and len(widths) >= minimum_profiles
              else 'insufficient_profile_crossings')
    result = empty_side(status, half_width)
    result.update(profiles_with_midpoint=len(offsets), profiles_with_width=len(widths))
    if offsets:
        values = np.asarray(offsets)
        result['signed_offset_mean_mm'] = float(np.mean(values))
        result['reference_deviation_rms_mm'] = float(np.sqrt(np.mean(values ** 2)))
        if len(offsets) >= 3:
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(side_points, axis=0), axis=1))]
            if length[-1] > 1e-9:
                design = np.column_stack((np.ones(len(length)), length))
                error = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
                result['straightness_rms_mm'] = float(np.sqrt(np.mean(error ** 2)))
    if widths:
        result['transition_width_median_mm'] = float(np.median(widths))
        result['transition_width_p90_mm'] = float(np.percentile(widths, 90))
        result['surface_transition_length_median_mm'] = float(np.median(output_widths))
        result['surface_transition_length_p90_mm'] = float(np.percentile(output_widths, 90))
    return result


def evaluate_hexagon(
    baseline_xyz: np.ndarray,
    xyz_out: np.ndarray,
    valid_mask: np.ndarray,
    vertices_rc: np.ndarray,
    *,
    roi_mask: np.ndarray,
    contact_sign: int = 1,
    profiles_per_side: int = 15,
    profile_half_width_mm: float | None = None,
    response_levels_mm: tuple[float, float] | None = None,
) -> HexagonEvaluation:
    """Evaluate a surface using a common convex six-vertex target and ROI.

    The target vertices are ordered (row, column) coordinates on the baseline
    grid, supplied independently of the method being compared. ``contact_sign``
    is +1 for positive normal displacement and -1 for negative displacement.
    The response is the baseline-normal component of ``xyz_out - baseline_xyz``
    multiplied by that sign. Median response in the inner target and outer ROI
    sets common-to-the-surface 10%, 40%, 50%, 60%, and 90% thresholds. Optional
    ``response_levels_mm`` supplies (outside, inside) medians explicitly.

    The 50% connected contour is compared with the target using baseline surface
    area weights. Threshold stability is the area-weighted IoU of the 40% and
    60% connected contours. On each target edge, profiles run locally normal to
    the edge in the baseline surface metric. Edge offsets and the primary 10-90
    widths are 3D arc lengths on the baseline surface; the accompanying surface
    transition length measures the same crossings along ``xyz_out``. Straightness is the RMS residual
    after fitting an affine function to the signed 50% offsets along each edge;
    it is relative to the supplied target, with edge shift and tilt removed.

    ``status`` and per-edge counts identify insufficient contrast, ROI support,
    or threshold crossings. ``target_mask`` and ``predicted_mask`` share the
    input grid. Model-derived or uncertain reference vertices should be reported
    as such by the caller.
    """
    baseline = np.asarray(baseline_xyz, np.float64)
    output = np.asarray(xyz_out, np.float64)
    valid = np.asarray(valid_mask, bool)
    roi = np.asarray(roi_mask, bool)
    if baseline.ndim != 3 or baseline.shape[-1] != 3 or min(baseline.shape[:2]) < 4:
        raise ValueError('baseline_xyz must be an at-least-4x4 grid of XYZ points')
    if output.shape != baseline.shape or valid.shape != baseline.shape[:2] or roi.shape != valid.shape:
        raise ValueError('XYZ, valid_mask and roi_mask shapes must match')
    if np.any(valid & (~np.isfinite(baseline).all(axis=-1) | ~np.isfinite(output).all(axis=-1))):
        raise ValueError('Valid XYZ points must be finite')
    if contact_sign not in (-1, 1):
        raise ValueError('contact_sign must be +1 or -1')
    if isinstance(profiles_per_side, bool) or not isinstance(profiles_per_side, int) or profiles_per_side < 3:
        raise ValueError('profiles_per_side must be an integer >= 3')
    if profile_half_width_mm is not None and (
        not np.isfinite(profile_half_width_mm) or profile_half_width_mm <= 0
    ):
        raise ValueError('profile_half_width_mm must be positive and finite')
    if response_levels_mm is not None:
        response_levels = np.asarray(response_levels_mm, np.float64)
        if response_levels.shape != (2,) or not np.isfinite(response_levels).all():
            raise ValueError('response_levels_mm must contain finite outside and inside values')
    vertices = np.asarray(vertices_rc, np.float64)
    target_all = _hexagon_mask(vertices, valid.shape)
    geometry = compute_surface_geometry(baseline, valid)
    support = geometry['valid'] & roi
    target = target_all & support
    target_coverage = float(target.sum() / max(int(target_all.sum()), 1))
    inner_distance = cv2.distanceTransform(target_all.astype(np.uint8), cv2.DIST_L2, 5)
    outer_distance = cv2.distanceTransform((~target_all).astype(np.uint8), cv2.DIST_L2, 5)
    radius = float(inner_distance.max())
    inner = support & (inner_distance >= .35 * radius) & target_all
    outer = support & ~target_all & (outer_distance >= .25 * radius)
    metrics: dict[str, Any] = dict(
        status='failed', reason=None, target_coverage_fraction=target_coverage,
        inner_sample_count=int(inner.sum()), outer_sample_count=int(outer.sum()),
        contact_sign=int(contact_sign),
    )
    empty = np.zeros(valid.shape, bool)
    if radius <= 0 or inner.sum() < 6 or outer.sum() < 6:
        metrics['reason'] = 'insufficient_inner_or_outer_roi_support'
        return HexagonEvaluation(metrics, target, empty)
    height = contact_sign * np.sum((output - baseline) * geometry['normal'], axis=-1)
    if response_levels_mm is None:
        inside_level, outside_level = float(np.median(height[inner])), float(np.median(height[outer]))
    else:
        outside_level, inside_level = map(float, response_levels)
    contrast = inside_level - outside_level
    metrics.update(inner_median_mm=inside_level, outer_median_mm=outside_level, contrast_mm=contrast)
    if contrast <= 1e-6:
        metrics['reason'] = 'nonpositive_or_insufficient_contact_contrast'
        return HexagonEvaluation(metrics, target, empty)
    levels = tuple(outside_level + fraction * contrast for fraction in (.1, .5, .9))
    predicted = _component(support & (height >= levels[1]), target)
    if not predicted.any():
        metrics['reason'] = 'no_50_percent_component_overlaps_target'
        return HexagonEvaluation(metrics, target, predicted)
    lower = _component(support & (height >= outside_level + .4 * contrast), target)
    upper = _component(support & (height >= outside_level + .6 * contrast), target)
    weights = np.where(support, geometry['area'], 0.)
    roi_edge = support & ~cv2.erode(support.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    metrics.update(
        target_iou=_weighted_iou(predicted, target, weights),
        threshold_stability_iou_40_60=_weighted_iou(lower, upper, weights),
        contour_touches_roi_boundary=bool(np.any(predicted & roi_edge)),
        outside_positive_p95_mm=float(np.percentile(np.maximum(height[outer] - outside_level, 0), 95)),
        outside_negative_p95_mm=float(np.percentile(np.maximum(outside_level - height[outer], 0), 95)),
    )
    field = np.concatenate((output, baseline, height[..., None]), axis=-1)
    edges = []
    for index in range(6):
        side = _side_profiles(vertices[index], vertices[(index + 1) % 6], vertices.mean(axis=0),
                              baseline, field, support, levels, profiles_per_side, profile_half_width_mm)
        side['side_index'] = index
        edges.append(side)
    metrics['edges'] = edges
    straightness = [edge['straightness_rms_mm'] for edge in edges if edge['straightness_rms_mm'] is not None]
    widths = [edge['transition_width_median_mm'] for edge in edges if edge['transition_width_median_mm'] is not None]
    surface_lengths = [edge['surface_transition_length_median_mm'] for edge in edges
                       if edge['surface_transition_length_median_mm'] is not None]
    metrics['edge_straightness_mean_mm'] = float(np.mean(straightness)) if straightness else None
    metrics['transition_width_median_of_edges_mm'] = float(np.median(widths)) if widths else None
    metrics['surface_transition_length_median_of_edges_mm'] = (
        float(np.median(surface_lengths)) if surface_lengths else None
    )
    complete = all(side['status'] == 'ok' for side in edges)
    if target_coverage < .9 or metrics['contour_touches_roi_boundary'] or not complete:
        metrics['status'] = 'partial'
        metrics['reason'] = 'inspect_coverage_boundary_and_edge_status'
    else:
        metrics['status'] = 'ok'
    normalized = np.where(support, (height - outside_level) / contrast, np.nan)
    return HexagonEvaluation(metrics, target, predicted, lower, upper, normalized)


def _fit_hexagon_vertices(baseline: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fit six ordered convex vertices to a binary contour in its tangent plane."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('The 50 percent component has no contour')
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 6:
        raise ValueError('The 50 percent contour has fewer than six samples')
    rc = contour[:, ::-1]
    xyz = baseline[rc[:, 0], rc[:, 1]]
    center = xyz.mean(axis=0)
    _, singular, axes = np.linalg.svd(xyz - center, full_matrices=False)
    if singular[1] <= 1e-9:
        raise ValueError('The 50 percent contour is geometrically degenerate')
    plane = np.asarray((xyz - center) @ axes[:2].T, np.float32)
    hull = cv2.convexHull(plane, returnPoints=False).reshape(-1)
    if len(hull) < 6:
        raise ValueError('The 50 percent contour hull has fewer than six vertices')
    # Remove the smallest convex-hull triangles. This keeps six free vertices;
    # no regular-hexagon side, angle, rotation or scale is imposed.
    while len(hull) > 6:
        points = plane[hull]
        before = np.roll(points, 1, axis=0) - points
        after = np.roll(points, -1, axis=0) - points
        area = np.abs(before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0])
        hull = np.delete(hull, int(np.argmin(area)))
    vertices = rc[hull].astype(np.float64)
    _hexagon_mask(vertices, mask.shape)
    return vertices


def evaluate_fitted_hexagon(
    baseline_xyz: np.ndarray,
    xyz_out: np.ndarray,
    valid_mask: np.ndarray,
    seed_vertices_rc: np.ndarray,
    *,
    roi_mask: np.ndarray,
    contact_sign: int = 1,
    profiles_per_side: int = 15,
    profile_half_width_mm: float | None = None,
) -> HexagonEvaluation:
    """Fit a free convex hexagon to each surface's 50% contour in a shared ROI.

    The common seed is used only to estimate inner and outer response levels;
    Fitted vertices and the IoU describe each surface's six-sidedness.
    Thresholds remain fixed throughout seed and fitted-contour evaluation.
    """
    initial = evaluate_hexagon(
        baseline_xyz, xyz_out, valid_mask, seed_vertices_rc, roi_mask=roi_mask,
        contact_sign=contact_sign, profiles_per_side=profiles_per_side,
        profile_half_width_mm=profile_half_width_mm,
    )
    if initial.metrics['status'] == 'failed':
        return initial
    try:
        vertices = _fit_hexagon_vertices(np.asarray(baseline_xyz, np.float64), initial.predicted_mask)
    except ValueError as exc:
        metrics = dict(initial.metrics, status='failed', reason='six_vertex_fit_failed', fit_error=str(exc))
        return HexagonEvaluation(metrics, initial.target_mask, initial.predicted_mask,
                                 initial.low_mask, initial.high_mask, initial.normalized_response)
    result = evaluate_hexagon(
        baseline_xyz, xyz_out, valid_mask, vertices, roi_mask=roi_mask,
        contact_sign=contact_sign, profiles_per_side=profiles_per_side,
        profile_half_width_mm=profile_half_width_mm,
        response_levels_mm=(initial.metrics['outer_median_mm'], initial.metrics['inner_median_mm']),
    )
    result.metrics.update(
        fitted_vertices_rc=vertices.tolist(),
        fit_kind='free_convex_hexagon_to_each_50_percent_contour',
        seed_target_iou=initial.metrics['target_iou'],
        fitted_inner_sample_count=result.metrics['inner_sample_count'],
        fitted_outer_sample_count=result.metrics['outer_sample_count'],
        inner_sample_count=initial.metrics['inner_sample_count'],
        outer_sample_count=initial.metrics['outer_sample_count'],
    )
    return result
