"""First-generation contact ellipse detector for fixed-pose linear RGB batches."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from reconstruction.surface_reconstruction.markers import (
    detect_cross_centers,
    filter_centers_by_mask,
    find_tracking_region,
)
from utils.image_io import linear_rgb_to_tracking_bgr8, validate_linear_rgb


ContactProposal = tuple[tuple[float, float], tuple[float, float], float]


@dataclass(frozen=True)
class ContactBackground:
    """Batch median and conservative image masks in original pixel coordinates."""

    linear_rgb: np.ndarray
    region_mask: np.ndarray
    interior_mask: np.ndarray
    overview_crop: tuple[int, int, int, int]
    scale: float
    training_frames: int


def _kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _prepare(image: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove detected marker strokes before spatially smoothing linear RGB."""
    gray, region = find_tracking_region(linear_rgb_to_tracking_bgr8(image))
    centers, _, diagnostics = detect_cross_centers(gray, region.roi)
    _, diagnostics = filter_centers_by_mask(centers, diagnostics, region.marker_mask)
    marker_mask = np.zeros(gray.shape, np.uint8)
    padding = max(1, round(2 * scale))
    for candidate in diagnostics:
        x, y, width, height = candidate['bbox']
        cv2.rectangle(marker_mask, (x - padding, y - padding),
                      (x + width - 1 + padding, y + height - 1 + padding), 1, -1)
    interpolated = np.empty_like(image)
    radius = max(1.0, 3.0 * scale)
    for channel in range(3):
        interpolated[..., channel] = cv2.inpaint(
            image[..., channel], marker_mask, radius, cv2.INPAINT_TELEA)
    smoothed = cv2.GaussianBlur(interpolated, (0, 0), max(0.8, 4.0 * scale))
    return smoothed, region.mask, len(diagnostics)


def _overlap(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def fit_contact_background(images: list[np.ndarray]) -> ContactBackground:
    """Fit a batch median using at least five fixed-pose pressed frames.

    Each pixel needs a non-contact appearance in a majority of the selected
    frames. The caller should sample frames with varying contact positions;
    this median is a local batch baseline rather than a reference photograph.
    """
    if len(images) < 5:
        raise ValueError('At least five frames are required for a batch background')
    validated = [validate_linear_rgb(image) for image in images]
    shape = validated[0].shape
    if any(image.shape != shape for image in validated):
        raise ValueError('All batch images must have the same size')
    scale = min(shape[:2]) / 480.0
    prepared = []
    for image in validated:
        try:
            prepared.append(_prepare(image, scale))
        except (RuntimeError, cv2.error):
            continue
    if len(prepared) < 5:
        raise ValueError('At least five frames need a detectable rubber region')

    regions = np.stack([item[1] for item in prepared])
    consensus = regions.mean(axis=0) >= 0.85
    kept = [item for item in prepared if _overlap(item[1], consensus) >= 0.85]
    if len(kept) < 5:
        raise ValueError('At least five frames need a consistent rubber pose')
    region_mask = np.mean(np.stack([item[1] for item in kept]), axis=0) >= 0.85
    border = max(2, round(5 * scale))
    interior = cv2.erode(region_mask.astype(np.uint8), _kernel(border)).astype(bool)
    if np.count_nonzero(interior) < 500 * scale * scale:
        raise ValueError('The common rubber interior is too small')
    median = np.median(np.stack([item[0] for item in kept]), axis=0).astype(np.float32)
    x, y, width, height = cv2.boundingRect(region_mask.astype(np.uint8))
    margin = max(4, round(12 * scale))
    crop = (max(0, x - margin), max(0, y - margin),
            min(shape[1], x + width + margin), min(shape[0], y + height + margin))
    return ContactBackground(median, region_mask, interior, crop, scale, len(kept))


def _candidate_ellipses(score: np.ndarray, mask: np.ndarray, threshold: float, scale: float) -> list[dict]:
    active = ((score > threshold) & mask).astype(np.uint8)
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, _kernel(max(1, round(5 * scale))))
    active &= mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    minimum_area = max(25 * scale * scale, 0.001 * np.count_nonzero(mask))
    maximum_area = 0.2 * np.count_nonzero(mask)
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not minimum_area <= area <= maximum_area:
            continue
        component = labels == label
        # Rank by the broad response, but fit to the stronger inner footprint.
        component_peak = float(np.percentile(score[component], 98))
        core_threshold = max(threshold, 0.5 * component_peak)
        points = cv2.findNonZero((component & (score >= core_threshold)).astype(np.uint8))
        if points is None or len(points) < 5:
            continue
        hull = cv2.convexHull(points)
        if len(hull) < 5:
            continue
        (cx, cy), (diameter_x, diameter_y), angle = cv2.fitEllipse(hull)
        if min(diameter_x, diameter_y) < 10 * scale or max(diameter_x, diameter_y) > 3 * min(diameter_x, diameter_y):
            continue
        mass = float(np.maximum(score[component] - threshold, 0).sum())
        if mass <= 0:
            continue
        candidates.append(dict(
            center=(float(cx), float(cy)), axes=(float(diameter_x / 2), float(diameter_y / 2)),
            angle=float(angle), area=area, mass=mass, core_threshold=core_threshold,
        ))
    return sorted(candidates, key=lambda item: item['mass'], reverse=True)


def _ellipse_inside_fraction(candidate: dict, mask: np.ndarray) -> float:
    center = tuple(int(round(v)) for v in candidate['center'])
    axes = tuple(max(1, int(round(v))) for v in candidate['axes'])
    points = cv2.ellipse2Poly(center, axes, int(round(candidate['angle'])), 0, 360, 3)
    x, y = points[:, 0], points[:, 1]
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    on_rubber = np.zeros(len(points), bool)
    on_rubber[inside] = mask[y[inside], x[inside]]
    return float(on_rubber.mean())


def propose_contact(image: np.ndarray, background: ContactBackground) -> tuple[ContactProposal | None, dict]:
    """Propose one ellipse and JSON-safe diagnostics from a fitted batch baseline.

    The ellipse describes a thresholded color-change footprint, with center and
    semi-axes in image pixels and angle in degrees. Weak, multiple or strongly
    clipped responses are reported as abstentions.
    """
    image = validate_linear_rgb(image)
    if image.shape != background.linear_rgb.shape:
        raise ValueError('Image shape differs from the batch background')
    diagnostics = dict(status='abstain', training_frames=background.training_frames)
    try:
        current, region, markers = _prepare(image, background.scale)
    except (RuntimeError, cv2.error):
        diagnostics['reason'] = 'rubber_region_unavailable'
        return None, diagnostics
    overlap = _overlap(region, background.region_mask)
    diagnostics.update(region_overlap=overlap, detected_markers=markers)
    if overlap < 0.85:
        diagnostics['reason'] = 'pose_changed'
        return None, diagnostics
    border = max(2, round(5 * background.scale))
    interior = cv2.erode(region.astype(np.uint8), _kernel(border)).astype(bool)
    mask = background.interior_mask & interior
    if not mask.any():
        diagnostics['reason'] = 'rubber_region_unavailable'
        return None, diagnostics

    difference = current - background.linear_rgb
    offset = np.median(difference[mask], axis=0)
    difference -= offset
    energy = np.sqrt(np.mean(difference ** 2, axis=-1))
    energy = cv2.GaussianBlur(energy, (0, 0), max(0.8, 4 * background.scale))
    values = energy[mask]
    median = float(np.median(values))
    noise = float(1.4826 * np.median(np.abs(values - median)))
    peak = float(np.percentile(values, 99.5))
    threshold = max(0.012, median + 3 * noise, median + 0.3 * (peak - median))
    diagnostics.update(background_energy=median, noise_energy=noise, peak_energy=peak,
                       threshold=threshold, rgb_offset=offset.astype(float).tolist())
    if peak <= threshold:
        diagnostics['reason'] = 'weak_signal'
        return None, diagnostics

    candidates = _candidate_ellipses(energy, mask, threshold, background.scale)
    diagnostics['candidate_count'] = len(candidates)
    if not candidates:
        diagnostics['reason'] = 'no_compact_candidate'
        return None, diagnostics
    best = candidates[0]
    diagnostics.update(component_area_px=best['area'], score=best['mass'],
                       core_threshold=best['core_threshold'])
    if len(candidates) > 1 and candidates[1]['mass'] >= 0.5 * best['mass']:
        diagnostics['reason'] = 'ambiguous_candidates'
        diagnostics['second_score'] = candidates[1]['mass']
        return None, diagnostics
    inside_fraction = _ellipse_inside_fraction(best, mask)
    diagnostics['ellipse_inside_fraction'] = inside_fraction
    if inside_fraction < 0.8:
        diagnostics['reason'] = 'edge_truncated'
        return None, diagnostics
    diagnostics['status'] = 'proposed'
    return (best['center'], best['axes'], best['angle']), diagnostics


def propose_contacts(images: list[np.ndarray]) -> tuple[
    list[ContactProposal | None], list[dict], tuple[int, int, int, int]
]:
    """Fit a batch background and return proposals, diagnostics and overview crop."""
    background = fit_contact_background(images)
    results = [propose_contact(image, background) for image in images]
    return [item[0] for item in results], [item[1] for item in results], background.overview_crop
