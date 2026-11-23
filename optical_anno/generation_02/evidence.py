"""Photometric evidence for projected sphere-contact boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from optical_anno.generation_01.detector import ContactBackground, _overlap, _prepare
from utils.image_io import validate_linear_rgb


_BAD_SCORE = -np.inf
_ANGLES = np.arange(64, dtype=np.float32) * (2 * np.pi / 64)
_DIRECTIONS = np.column_stack((np.cos(_ANGLES), np.sin(_ANGLES)))
RayScore = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return cv2.remap(image, points[..., 0], points[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _radial_scores(difference: np.ndarray, mask: np.ndarray,
                   centers: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Score the change in signed radial RGB gradient at each boundary."""
    samples, valid = [], []
    for fraction in (0.5, 0.75, 1.0, 1.25):
        points = centers[:, None, :] + fraction * vectors
        samples.append(_sample(difference, points))
        valid.append(_sample(mask, points) > 0.99)
    direction = (samples[0] - np.roll(samples[0], 32, axis=1)) * 0.5
    curvature = 2 * samples[2] - samples[1] - samples[3]
    values = np.sum(curvature * direction, axis=-1)
    good = valid[0] & np.roll(valid[0], 32, axis=1) & valid[1] & valid[2] & valid[3]
    count = good.sum(axis=1)
    ordered = np.sort(np.where(good, values, np.inf), axis=1)
    ranks = np.arange(values.shape[1])[None, :]
    trim = count // 10
    use = (ranks >= trim[:, None]) & (ranks < (count - trim)[:, None])
    total = np.where(use, ordered, 0).sum(axis=1)
    result = total / np.maximum(count - 2 * trim, 1)
    return np.where(count >= 0.6 * values.shape[1], result * (count / values.shape[1]), _BAD_SCORE)


def _boundary_vectors(center: np.ndarray, boundary: np.ndarray) -> np.ndarray | None:
    """Intersect each image-space ray with the supplied polygon edges."""
    polygon = boundary.astype(np.float32).reshape(-1, 1, 2)
    if cv2.pointPolygonTest(polygon, tuple(center.astype(float)), False) < 0:
        return None
    start = boundary - center
    edge = np.roll(start, -1, axis=0) - start
    directions = _DIRECTIONS[:, None, :]
    denominator = directions[..., 0] * edge[None, :, 1] - directions[..., 1] * edge[None, :, 0]
    safe = np.where(np.abs(denominator) > 1e-8, denominator, 1.0)
    radius = (start[:, 0] * edge[:, 1] - start[:, 1] * edge[:, 0])[None, :] / safe
    along = (start[None, :, 0] * directions[..., 1] - start[None, :, 1] * directions[..., 0]) / safe
    hit = (np.abs(denominator) > 1e-8) & (radius > 0) & (along >= 0) & (along <= 1)
    radius = np.min(np.where(hit, radius, np.inf), axis=1)
    if not np.all(np.isfinite(radius)):
        return None
    return radius[:, None] * _DIRECTIONS


@dataclass(frozen=True)
class ContactEvidence:
    """A coarse pixel seed and scoring function for physical projected polygons."""

    seed_px: tuple[float, float]
    radius_hint_px: float
    diagnostics: dict
    _difference: np.ndarray = field(repr=False)
    _mask: np.ndarray = field(repr=False)
    _score_rays: RayScore = field(default=_radial_scores, repr=False)
    _center_sigma_px: float | None = field(default=None, repr=False)

    def score_boundary(self, center_px: np.ndarray, boundary_px: np.ndarray) -> float:
        center = np.asarray(center_px, dtype=np.float64)
        boundary = np.asarray(boundary_px, dtype=np.float64)
        if (center.shape != (2,) or boundary.ndim != 2 or boundary.shape[1] != 2
                or len(boundary) < 3 or not np.all(np.isfinite(center))
                or not np.all(np.isfinite(boundary))):
            return _BAD_SCORE
        vectors = _boundary_vectors(center, boundary)
        if vectors is None:
            return _BAD_SCORE
        score = float(self._score_rays(self._difference, self._mask,
                                       center[None, :], vectors[None, :, :])[0])
        if not score > 0:
            return _BAD_SCORE
        if self._center_sigma_px is not None:
            displacement = (center - self.seed_px) / self._center_sigma_px
            score *= float(np.exp(-0.5 * np.dot(displacement, displacement)))
        return score


def contact_evidence(image: np.ndarray, background: ContactBackground, *,
                     score_rays: RayScore = _radial_scores,
                     maximum_seed_radius_px: float | None = None,
                     fixed_background_pose: bool = True) -> ContactEvidence:
    """Prepare a photometric seed and score from a pressed linear-RGB frame.

    Coordinates and radius are in original image pixels. The score samples the
    supplied polygon along image rays and measures the change in signed color
    gradients across its boundary. It is maximized by the physical solver.
    """
    image = validate_linear_rgb(image)
    if image.shape != background.linear_rgb.shape:
        raise ValueError('Image shape differs from the batch background')
    try:
        current, region, markers = _prepare(image, background.scale)
    except (RuntimeError, cv2.error) as error:
        raise ValueError('Rubber region unavailable') from error
    overlap = _overlap(region, background.region_mask)
    if fixed_background_pose and overlap < 0.85:
        raise ValueError('Rubber pose differs from the batch background')
    border = max(2, round(5 * background.scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
    mask = background.interior_mask & cv2.erode(region.astype(np.uint8), kernel).astype(bool)
    if not mask.any():
        raise ValueError('Rubber interior unavailable')
    difference = current - background.linear_rgb
    difference -= np.median(difference[mask], axis=0)
    energy = np.sqrt(np.mean(difference ** 2, axis=2))
    mask_float = mask.astype(np.float32)
    sigma = max(0.8, 4 * background.scale)
    if maximum_seed_radius_px is None:
        energy = cv2.GaussianBlur(energy, (0, 0), sigma)
    else:
        support = cv2.GaussianBlur(mask_float, (0, 0), sigma)
        energy = cv2.GaussianBlur(energy * mask_float, (0, 0), sigma) / np.maximum(support, 1e-6)
    values = energy[mask]
    median = float(np.median(values))
    noise = float(1.4826 * np.median(np.abs(values - median)))
    peak = float(np.percentile(values, 99.5))
    threshold = max(0.012, median + 3 * noise, median + 0.3 * (peak - median))
    if peak <= threshold:
        raise ValueError('Weak contact signal')
    active = ((energy > threshold) & mask).astype(np.uint8)
    radius = max(1, round(5 * background.scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, kernel)
    active &= mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    minimum_area = max(25 * background.scale ** 2, 0.001 * np.count_nonzero(mask))
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if minimum_area <= area <= 0.3 * np.count_nonzero(mask):
            selected = labels == label
            mass = float(np.maximum(energy[selected] - threshold, 0).sum())
            candidates.append((mass, area, label))
    if not candidates:
        raise ValueError('No compact contact signal')
    candidates.sort(reverse=True)
    _, area, label = candidates[0]
    selected = labels == label
    ys, xs = np.nonzero(selected)
    weights = np.maximum(energy[selected] - threshold, 0)
    center = np.array([np.average(xs, weights=weights), np.average(ys, weights=weights)])
    extent = np.sqrt(area / np.pi)
    if maximum_seed_radius_px is None:
        offsets = np.linspace(-0.6, 0.6, 9) * extent
        radii = extent * np.linspace(0.35, 1.05, 7)
    else:
        offsets = np.linspace(-1.0, 1.0, 17) * max(extent, maximum_seed_radius_px)
        minimum = min(max(2.0, 0.35 * extent), 0.5 * maximum_seed_radius_px)
        radii = np.linspace(minimum, maximum_seed_radius_px, 10)
    xx, yy = np.meshgrid(center[0] + offsets, center[1] + offsets)
    seed_centers = np.repeat(np.column_stack((xx.ravel(), yy.ravel())), len(radii), axis=0)
    radii = np.tile(radii, xx.size)
    vectors = radii[:, None, None] * _DIRECTIONS[None, :, :]
    scores = score_rays(difference, mask_float, seed_centers, vectors)
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        raise ValueError('Weak signed boundary signal')
    return ContactEvidence(
        tuple(float(value) for value in seed_centers[best]), float(radii[best]),
        dict(region_overlap=overlap, detected_markers=markers, threshold=threshold,
             boundary_score=float(scores[best]), component_area_px=area,
             second_component_ratio=(candidates[1][0] / candidates[0][0] if len(candidates) > 1 else 0.0)),
        difference, mask_float, score_rays,
    )
