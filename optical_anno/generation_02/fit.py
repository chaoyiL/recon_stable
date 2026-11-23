"""Fit image evidence with the same sphere/surface contact as manual annotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from calibrate.utils.norm_pipeline import _finalize_contact_geometry, initial_calibration_surface
from utils.region_annotations import RegionAnnotation
from utils.sphere_contact import SphereSurfaceContact
from utils.surface_projection import invert_surface_projection


_MAXIMUM_FOOTPRINT_TOLERANCE_PX = 0.5


@dataclass(frozen=True)
class _Candidate:
    center_px: tuple[float, float]
    depth_mm: float
    contour_px: np.ndarray
    score: float
    depth_coordinate: float


def _initial_depth(frame, center_px: np.ndarray, radius_hint_px: float | None, radius_mm: float) -> float:
    """Estimate indentation from image-scale evidence using 3D surface distances."""
    if radius_hint_px is None:
        return 0.1 * radius_mm
    angles = np.arange(8) * (np.pi / 4)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    pixels = np.vstack((center_px, center_px + radius_hint_px * directions))
    inverse = invert_surface_projection(pixels, frame.surface, frame.camera, frame.systems.geometry)
    if not inverse.valid[0] or not inverse.valid[1:].any():
        return 0.1 * radius_mm
    distances = np.linalg.norm(inverse.xyz[1:][inverse.valid[1:]] - inverse.xyz[0], axis=1)
    rho = min(float(np.median(distances)), radius_mm)
    # The flat-surface sphere formula supplies only a search seed; every
    # candidate boundary is computed against the actual parametric surface.
    depth = rho * rho / (radius_mm + np.sqrt(max(radius_mm * radius_mm - rho * rho, 0.0)))
    return float(np.clip(depth, radius_mm * 1e-4, radius_mm))


def _inside_rubber(center_px: np.ndarray, contour_px: np.ndarray, rubber: np.ndarray) -> bool:
    """Check the complete rasterized footprint in original image coordinates."""
    height, width = rubber.shape
    if (contour_px.ndim != 2 or contour_px.shape[1] != 2 or len(contour_px) < 3
            or not np.isfinite(contour_px).all()):
        return False
    rounded = np.rint(contour_px)
    if (np.any(rounded[:, 0] < 0) or np.any(rounded[:, 0] >= width)
            or np.any(rounded[:, 1] < 0) or np.any(rounded[:, 1] >= height)):
        return False
    if cv2.pointPolygonTest(contour_px.astype(np.float32), tuple(map(float, center_px)), False) <= 0:
        return False
    footprint = np.zeros((height, width), np.uint8)
    cv2.fillPoly(footprint, [rounded.astype(np.int32)], 1)
    return bool(footprint.any() and not np.any(footprint.astype(bool) & ~rubber))


def _annotation(frame, candidate: _Candidate) -> RegionAnnotation:
    half_extent = np.maximum(np.ptp(candidate.contour_px, axis=0) / 2, 1e-6)
    return RegionAnnotation(
        frame.image_key,
        frame.linear_rgb.shape[1::-1],
        status='draft',
        center_px=candidate.center_px,
        shape='sphere_contact',
        semi_axes_px=tuple(map(float, half_extent)),
        contour_px=candidate.contour_px.tolist(),
        semantics='contact_boundary',
        indentation_depth_mm=candidate.depth_mm,
    )


def _diameter_mm(points: np.ndarray) -> float:
    """Largest straight-line distance between the sampled 3D boundary points."""
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Contact boundary must contain finite 3D points')
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return float(np.max(distances))


def _contour_overrun_px(contour: np.ndarray, maximum: np.ndarray) -> float:
    """Measure how far the actual polygon extends beyond the maximum polygon."""
    edges = np.roll(contour, -1, axis=0) - contour
    lengths = np.linalg.norm(edges, axis=1)
    # Sampling each edge, including its start vertex, also checks excursions
    # between vertices when the maximum polygon has a concave section.
    steps = np.maximum(1, np.ceil(lengths / 0.2).astype(int))
    reference = maximum.astype(np.float32)
    largest = 0.0
    for start, edge, count in zip(contour, edges, steps, strict=True):
        for fraction in np.arange(count) / count:
            point = start + fraction * edge
            distance = cv2.pointPolygonTest(reference, tuple(map(float, point)), True)
            largest = max(largest, -distance)
    return float(largest)


def _footprint_limit(frame, contact, annotation, radius_mm: float, side: int) -> dict:
    """Describe the final intersection and the manual editor's maximum-depth outline."""
    actual = np.asarray(contact.boundary_px, np.float64)
    info = dict(
        sphere_diameter_mm=float(2 * radius_mm),
        maximum_depth_mm=radius_mm,
        maximum_boundary_px=None,
        maximum_span_px=None,
        actual_span_px=np.ptp(actual, axis=0).astype(float).tolist(),
        actual_diameter_mm=_diameter_mm(contact.boundary_xyz),
        maximum_diameter_mm=None,
        maximum_contour_overrun_px=None,
        within_maximum_footprint=None,
        tolerance_px=_MAXIMUM_FOOTPRINT_TOLERANCE_PX,
        maximum_unavailable_reason=None,
    )
    try:
        if annotation.indentation_depth_mm == radius_mm:
            maximum_contact = contact
        else:
            maximum_contact = SphereSurfaceContact(frame.surface, frame.camera, frame.systems.geometry).compute(
                annotation.center_px, radius_mm, radius_mm, side)
        maximum = np.asarray(maximum_contact.boundary_px, np.float64)
        if maximum.ndim != 2 or maximum.shape[1] != 2 or len(maximum) < 3 or not np.isfinite(maximum).all():
            raise ValueError('Maximum-depth boundary must contain finite image points')
        maximum_diameter = _diameter_mm(maximum_contact.boundary_xyz)
    except (ValueError, RuntimeError, cv2.error) as exc:
        info['maximum_unavailable_reason'] = str(exc)
        return info
    overrun = 0.0 if maximum_contact is contact else _contour_overrun_px(actual, maximum)
    info.update(
        maximum_boundary_px=maximum.tolist(),
        maximum_span_px=np.ptp(maximum, axis=0).astype(float).tolist(),
        maximum_diameter_mm=maximum_diameter,
        maximum_contour_overrun_px=overrun,
        within_maximum_footprint=bool(overrun <= _MAXIMUM_FOOTPRINT_TOLERANCE_PX),
    )
    return info


def fit_contact(
    frame,
    anchor,
    settings,
    seed_px: np.ndarray | tuple[float, float],
    score_boundary: Callable[[np.ndarray, np.ndarray], float],
    *,
    radius_hint_px: float | None = None,
) -> tuple[RegionAnnotation | None, dict]:
    """Maximize image evidence over projected sphere/surface contacts.

    ``seed_px`` and the ordered boundary passed to ``score_boundary`` are
    distorted image (x, y) coordinates in pixels. The optional radius is an
    image-space search hint; sphere radius and indentation are in millimetres.
    Returned drafts use the converged marker-exclusion surface and its actual
    projected intersection. Diagnostics contain only JSON-native values.
    """
    settings.require_sphere()
    radius_mm = float(settings.sphere_radius_mm)
    side = settings.side
    seed = np.asarray(seed_px, np.float64)
    height, width = frame.linear_rgb.shape[:2]
    rubber = np.asarray(frame.surface_mask, bool)
    if seed.shape != (2,) or not np.isfinite(seed).all():
        raise ValueError('seed_px must be a finite (x, y) pixel')
    if not (0 <= seed[0] < width and 0 <= seed[1] < height):
        raise ValueError('seed_px is outside the image')
    if rubber.shape != (height, width) or tuple(frame.camera.image_shape) != (height, width):
        raise ValueError('Frame image, rubber mask, and camera dimensions must match')
    if radius_hint_px is not None:
        radius_hint_px = float(radius_hint_px)
        if not np.isfinite(radius_hint_px) or radius_hint_px <= 0:
            raise ValueError('radius_hint_px must be finite and positive')

    frame = initial_calibration_surface(frame, anchor)
    projector = SphereSurfaceContact(frame.surface, frame.camera, frame.systems.geometry)
    initial_depth = _initial_depth(frame, seed, radius_hint_px, radius_mm)
    # sqrt(depth / radius) changes roughly in proportion to projected footprint
    # radius near zero indentation and spans the whole physically allowed range.
    initial_q = float(np.sqrt(initial_depth / radius_mm))
    scale_px = radius_hint_px if radius_hint_px is not None else 0.1 * min(height, width)
    move_px = float(np.clip(0.5 * scale_px, 4.0, 0.25 * min(height, width)))
    lower = np.array([max(0.0, seed[0] - move_px), max(0.0, seed[1] - move_px), 0.01])
    upper = np.array([min(width - 1.0, seed[0] + move_px),
                      min(height - 1.0, seed[1] + move_px), 1.0])
    counts = dict(evaluations=0, valid=0, invalid_geometry=0, outside_rubber=0, nonfinite_score=0)
    cache: dict[tuple[float, float, float], _Candidate | None] = {}
    candidates: list[_Candidate] = []

    def evaluate(values) -> _Candidate | None:
        x, y, q = np.clip(np.asarray(values, np.float64), lower, upper)
        key = tuple(np.round([x, y, q], 10))
        if key in cache:
            return cache[key]
        counts['evaluations'] += 1
        center = np.array([x, y])
        depth = float(radius_mm * q * q)
        try:
            contour = np.asarray(projector.compute(center, depth, radius_mm, side).boundary_px, np.float64)
        except (ValueError, RuntimeError, cv2.error):
            counts['invalid_geometry'] += 1
            cache[key] = None
            return None
        if not _inside_rubber(center, contour, rubber):
            counts['outside_rubber'] += 1
            cache[key] = None
            return None
        score = float(score_boundary(center, contour))
        if not np.isfinite(score):
            counts['nonfinite_score'] += 1
            cache[key] = None
            return None
        candidate = _Candidate((float(x), float(y)), depth, contour, score, float(q))
        cache[key] = candidate
        counts['valid'] += 1
        candidates.append(candidate)
        return candidate

    # Sample depth broadly at the evidence seed, then search the center at the
    # two best separated depths. The fixed budget keeps interactive use bounded.
    q_samples = sorted(set(float(np.clip(q, lower[2], upper[2])) for q in (
        0.1, 0.35, 0.7, 1.0, *(initial_q * factor for factor in (0.35, 0.5, 0.75, 1.0, 1.4, 2.0))
    )))
    at_seed = [candidate for q in q_samples if (candidate := evaluate((*seed, q))) is not None]
    at_seed.sort(key=lambda candidate: candidate.score, reverse=True)
    depths = []
    for candidate in at_seed:
        if all(abs(candidate.depth_coordinate - q) >= max(0.08, 0.2 * q) for q in depths):
            depths.append(candidate.depth_coordinate)
        if len(depths) == 2:
            break
    if not depths:
        depths = [initial_q, 0.1]
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1),
               (0.7071, 0.7071), (0.7071, -0.7071),
               (-0.7071, 0.7071), (-0.7071, -0.7071)]
    for q in depths:
        for dx, dy in offsets:
            evaluate((seed[0] + dx * move_px, seed[1] + dy * move_px, q))

    if candidates:
        best = max(candidates, key=lambda candidate: candidate.score)
        q_step = max(0.08, best.depth_coordinate * 0.3)
        for factor in (1.0, 0.5, 0.25, 0.125):
            steps = (move_px * 0.5 * factor, move_px * 0.5 * factor, q_step * factor)
            for axis, step in enumerate(steps):
                origin = np.array([*best.center_px, best.depth_coordinate])
                for sign in (-1.0, 1.0):
                    trial = origin.copy()
                    trial[axis] += sign * step
                    candidate = evaluate(trial)
                    if candidate is not None and candidate.score > best.score:
                        best = candidate

    diagnostics = dict(status='abstain', search=counts, initial_depth_mm=float(initial_depth))
    if not candidates:
        diagnostics['reason'] = 'no_valid_contact'
        return None, diagnostics

    # If the strongest candidate fails the marker-exclusion iteration, try the
    # next highest scoring candidate. Each attempt still uses the final surface.
    failure = None
    last_limit = None
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)[:3]:
        annotation = _annotation(frame, candidate)
        try:
            final, contact, annotation, masks = _finalize_contact_geometry(frame, anchor, annotation, settings)
            final_contour = np.asarray(contact.boundary_px, np.float64)
            final_rubber = np.asarray(final.surface_mask, bool)
            if np.any(masks['contact'] & ~final_rubber):
                failure = 'final_contact_outside_rubber'
                continue
            final_score = float(score_boundary(np.asarray(annotation.center_px), final_contour))
            if not np.isfinite(final_score):
                failure = 'nonfinite_final_score'
                continue
            last_limit = _footprint_limit(final, contact, annotation, radius_mm, side)
            if last_limit['within_maximum_footprint'] is False:
                failure = 'final_contact_exceeds_maximum_footprint'
                continue
        except (ValueError, RuntimeError, cv2.error) as exc:
            failure = str(exc)
            continue
        annotation.status = 'draft'
        annotation.contour_px = final_contour.tolist()
        annotation.semi_axes_px = tuple(map(float, np.maximum(np.ptp(final_contour, axis=0) / 2, 1e-6)))
        diagnostics.update(status='proposed', score=final_score, initial_surface_score=candidate.score,
                           final_surface_score=final_score, center_px=list(annotation.center_px),
                           indentation_depth_mm=float(annotation.indentation_depth_mm),
                           depth_near_maximum=bool(annotation.indentation_depth_mm >= 0.98 * radius_mm),
                           footprint_limit=last_limit)
        return annotation, diagnostics
    diagnostics.update(reason='final_geometry_failed', detail=failure or 'Contact refinement failed')
    if last_limit is not None:
        diagnostics['footprint_limit'] = last_limit
    return None, diagnostics
