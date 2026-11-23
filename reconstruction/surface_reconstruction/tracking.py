"""Frame-local grid initialization and persistent video IDs, independent of 3D pose."""

from dataclasses import dataclass
from itertools import combinations
import cv2
import numpy as np

from utils.image_io import linear_rgb_to_tracking_bgr8, validate_linear_rgb
from scipy.optimize import linear_sum_assignment
try:
    from . import markers as detector
except ImportError:  # Direct execution from this directory.
    import markers as detector


def detect_candidates(gray):
    region = detector.find_inner_region(gray)
    centers, _, diagnostics = detector.detect_cross_centers(gray, region.roi)
    centers, _ = detector.filter_centers_by_mask(
        centers, diagnostics, region.marker_mask
    )
    return centers, region.roi


def select_complete_grid(centers, rows, cols, direction="auto"):
    """Reject up to two outliers only if one complete grid is clearly supported.

    This is initialization/acquisition, not a per-video-frame ID sorter.
    Rejects missing/ambiguous grids instead of fabricating points.
    """
    centers = np.asarray(centers, float)
    expected = rows * cols
    extra = len(centers) - expected
    if extra < 0 or extra > 2:
        raise ValueError(
            f"Need a complete {expected}-point grid; observed {len(centers)} candidates"
        )
    template = np.array([(c, r) for r in range(rows) for c in range(cols)], float)
    centered = (template - template.mean(0)) / np.ptp(template, axis=0)
    basis = np.column_stack(
        [centered[:, 0] ** i * centered[:, 1] ** j for i in range(4) for j in range(4)]
    )
    inverse = np.linalg.pinv(basis)
    leverage = np.sum(basis * inverse.T, axis=1)
    hypotheses = []
    for dropped in combinations(range(len(centers)), extra):
        kept = np.delete(np.arange(len(centers)), dropped)
        try:
            ordered = detector.sort_grid_points(centers[kept], rows, cols, direction)[:, 1:]
        except RuntimeError:
            continue
        h, _ = cv2.findHomography(template, ordered, 0)
        if h is None:
            continue
        distances = np.linalg.norm(ordered[:, None] - ordered[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        pitch = float(np.median(distances.min(axis=1)))
        # A smooth image warp allows a bent first frame; homography is only a degeneracy check.
        warp_error = np.linalg.norm(ordered - basis @ (inverse @ ordered), axis=1) / np.maximum(
            1 - leverage, 0.1
        )
        warp_rms = float(np.sqrt(np.mean(warp_error**2)))
        if warp_rms < 0.15 * pitch and warp_error.max() < 0.6 * pitch:
            hypotheses.append((warp_rms, ordered.copy(), list(dropped)))
    hypotheses.sort(key=lambda v: v[0])
    if not hypotheses:
        raise ValueError("No geometrically consistent complete grid; inspect the marker detections")
    if len(hypotheses) > 1 and hypotheses[1][0] - hypotheses[0][0] < 0.15:
        raise ValueError("Ambiguous outlier removal: competing grid assignments")
    rms, ordered, dropped = hypotheses[0]
    return np.ascontiguousarray(ordered), {
        "candidate_count": len(centers),
        "rejected_candidates": centers[dropped].tolist(),
        "grid_warp_crossvalidation_rms_px": rms,
    }


def require_regular_template(ids, xy):
    rows, cols = len(np.unique(xy[:, 1])), len(np.unique(xy[:, 0]))
    expected = np.array([(x, y) for y in np.unique(xy[:, 1]) for x in np.unique(xy[:, 0])])
    if (
        rows < 2
        or cols < 2
        or not np.array_equal(ids, np.arange(rows * cols))
        or not np.array_equal(xy, expected)
    ):
        raise ValueError("Grid acquisition requires a complete row-major rectangular template")
    return rows, cols


@dataclass
class TrackingSeed:
    ids: np.ndarray
    pixels: np.ndarray
    gray: np.ndarray

    @classmethod
    def build(cls, frame_linear_rgb, ids, xy, direction="auto", region=None,
              *, tracking_bgr=None, tracking_gray=None):
        frame_linear_rgb = validate_linear_rgb(frame_linear_rgb)
        if tracking_bgr is None:
            tracking_bgr = linear_rgb_to_tracking_bgr8(frame_linear_rgb)
        if region is None:
            gray, region = detector.find_tracking_region(tracking_bgr)
        else:
            gray = cv2.cvtColor(tracking_bgr, cv2.COLOR_BGR2GRAY) if tracking_gray is None else tracking_gray
        rows, cols = require_regular_template(ids, xy)
        roi = region.roi
        centers, _, diagnostics = detector.detect_cross_centers(gray, roi)
        centers, _ = detector.filter_centers_by_mask(
            centers, diagnostics, region.marker_mask
        )
        pixels, meta = select_complete_grid(centers, rows, cols, direction)
        return cls(ids.copy(), pixels, gray), roi, meta


def _predict_supported_motion(previous_positions, pixels, matched):
    """Predict all ID positions from spatially supported candidate matches.

    The RANSAC threshold identifies a stable reference motion, while later
    association uses the wider marker-pitch gate to preserve local deformation.
    Return None when the evidence cannot safely identify a frame-wide motion.
    """
    trusted = np.flatnonzero(matched)
    if len(trusted) < 8:
        return None
    h, inliers = cv2.findHomography(
        previous_positions[trusted], pixels[trusted], cv2.RANSAC, 2.0
    )
    if h is None or not np.isfinite(h).all() or inliers is None:
        return None
    support = trusted[inliers.ravel().astype(bool)]
    if len(support) < max(8, int(np.ceil(len(trusted) / 2))):
        return None
    support_hull = cv2.convexHull(previous_positions[support].astype(np.float32))
    all_hull = cv2.convexHull(previous_positions.astype(np.float32))
    support_area = cv2.contourArea(support_hull)
    total_area = cv2.contourArea(all_hull)
    denominator = previous_positions @ h[2, :2] + h[2, 2]
    if (total_area <= 0 or support_area < 0.1 * total_area
            or np.any(np.abs(denominator) <= 1e-6 * np.max(np.abs(denominator)))
            or not (np.all(denominator > 0) or np.all(denominator < 0))):
        return None
    inferred = cv2.perspectiveTransform(previous_positions[:, None, :], h).reshape(-1, 2)
    return inferred if np.isfinite(inferred).all() else None


def _candidate_support(predicted, candidates, gate):
    """Count distinct detections explained by a full set of predicted IDs."""
    costs = np.linalg.norm(predicted[:, None] - candidates[None], axis=2)
    rows, columns = linear_sum_assignment(
        np.column_stack((costs, np.full((len(predicted), len(predicted)), gate)))
    )
    observed = columns < len(candidates)
    return int(np.count_nonzero(costs[rows[observed], columns[observed]] < gate))


def _select_motion_prediction(previous, inferred, candidates, candidate_ids, matched, gate):
    """Resolve coherent grid-period LK aliases using detected boundary evidence.

    Compare the LK consensus with the previous grid and translations suggested
    by unused candidates. Similar-scoring distinct motions remain ambiguous.
    Predictions guide association only and are never returned as observations.
    """
    if not len(candidates):
        return None, False
    hypotheses = [inferred]
    if np.median(np.linalg.norm(previous - inferred, axis=1)) >= gate:
        hypotheses.append(previous)
    used = set(candidate_ids[matched])
    for index, point in enumerate(candidates):
        if index in used:
            continue
        nearest = np.argmin(np.linalg.norm(inferred - point, axis=1))
        shift = point - inferred[nearest]
        if np.linalg.norm(shift) >= gate:
            hypotheses.append(inferred + shift)
    scores = [_candidate_support(hypothesis, candidates, gate) for hypothesis in hypotheses]
    best = int(np.argmax(scores))
    for index, hypothesis in enumerate(hypotheses):
        if (index != best and scores[index] >= scores[best] - 1
                and np.median(np.linalg.norm(hypothesis - hypotheses[best], axis=1)) >= gate):
            return None, False
    return hypotheses[best], best != 0


def _grid_period_shifts(positions, pitch):
    """Estimate repeated-grid offsets from neighboring previous positions."""
    offsets = (positions[:, None] - positions[None, :]).reshape(-1, 2)
    lengths = np.linalg.norm(offsets, axis=1)
    offsets = offsets[(lengths > 0.7 * pitch) & (lengths < 2.5 * pitch)]
    offsets = offsets[np.argsort(np.linalg.norm(offsets, axis=1))]
    centers, counts = [], []
    for offset in offsets:
        if centers:
            distances = np.linalg.norm(np.asarray(centers) - offset, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] < 0.2 * pitch:
                counts[nearest] += 1
                centers[nearest] += (offset - centers[nearest]) / counts[nearest]
                continue
        centers.append(offset.copy())
        counts.append(1)
    return [center for center, count in zip(centers, counts) if count >= 3]


def _match_partial_grid(anchor, candidates, pitch, grid_steps):
    """Associate distinct observations near a frame-aligned grid."""
    gate = 0.5 * pitch

    def assign(positions):
        costs = np.linalg.norm(positions[:, None] - candidates[None], axis=2)
        rows, columns = linear_sum_assignment(
            np.column_stack((costs, np.full((len(positions), len(positions)), gate))))
        observed = columns < len(candidates)
        rows, columns = rows[observed], columns[observed]
        distances = costs[rows, columns]
        keep = distances < gate
        if len(candidates) > 1:
            nearest = np.sort(costs[rows], axis=1)
            keep &= nearest[:, 1] - distances > gate * 0.25
        score = np.sum(1 - (distances[keep] / gate) ** 2)
        return rows[keep], columns[keep], score

    rows, columns, support = assign(anchor)
    # The frame anchors ties when an entire edge is missing. A substantially
    # stronger shifted grid is evidence that these IDs are misplaced.
    if len(rows) and any(assign(anchor + step)[2] > support + 1 for step in grid_steps):
        rows = columns = np.empty(0, dtype=int)
    pixels = np.full(anchor.shape, np.nan)
    valid = np.zeros(len(anchor), bool)
    pixels[rows] = candidates[columns]
    valid[rows] = True
    return pixels, valid


def _still_candidates(tracking_bgr, roi, pitch):
    """Supplement gray detections with distinct maximum-channel detections."""
    gray = cv2.cvtColor(tracking_bgr, cv2.COLOR_BGR2GRAY)
    centers, _, _ = detector.detect_cross_centers(gray, roi)
    additional, _, _ = detector.detect_cross_centers(tracking_bgr.max(axis=2), roi)
    merged = list(centers)
    for center in additional:
        if not merged or np.min(np.linalg.norm(np.asarray(merged) - center, axis=1)) > 0.2 * pitch:
            merged.append(center)
    return np.asarray(merged, dtype=float).reshape(-1, 2)


def _grid_axes(pixels, rows, cols):
    grid = pixels.reshape(rows, cols, 2)
    return np.stack(((grid[-1] - grid[0]).mean(axis=0), (grid[:, -1] - grid[:, 0]).mean(axis=0)))


def match_still_markers(frame_linear_rgb, reference_pixels, ids, *, reference_surface_mask,
                        tracking_bgr=None, template_xy=None, row_direction="auto", region=None):
    """Match a still image to reference IDs using detected marker centers.

    ``frame_linear_rgb`` is H x W x 3 linear RGB, ``reference_pixels`` is
    N x 2 image coordinates and ``ids`` is the corresponding N-vector.
    ``template_xy``, when provided, is the N x 2 row-major material grid in mm;
    it enables complete-grid validation. A reference surface mask aligns the
    ID search with the current rubber frame, which can be supplied as `region`.
    Invalid returned pixels are NaN.
    """
    frame_linear_rgb = validate_linear_rgb(frame_linear_rgb)
    reference_pixels = np.asarray(reference_pixels, dtype=float)
    ids = np.asarray(ids)
    if (ids.ndim != 1 or reference_pixels.shape != (len(ids), 2)
            or len(ids) < 2 or len(np.unique(ids)) != len(ids)
            or not np.isfinite(reference_pixels).all()):
        raise ValueError("Reference markers require distinct IDs and finite N x 2 pixels")
    if tracking_bgr is None:
        tracking_bgr = linear_rgb_to_tracking_bgr8(frame_linear_rgb)
    gray = cv2.cvtColor(tracking_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape != frame_linear_rgb.shape[:2]:
        raise ValueError("Tracking image dimensions must match the frame")
    reference_surface_mask = np.asarray(reference_surface_mask, dtype=bool)
    if reference_surface_mask.shape != gray.shape:
        raise ValueError("Reference rubber mask must match the image size")
    if region is None:
        _, region = detector.find_tracking_region(tracking_bgr)
    rx, ry, rw, rh = cv2.boundingRect(reference_surface_mask.astype(np.uint8))
    x, y, w, h = cv2.boundingRect(region.mask.astype(np.uint8))
    if min(rw, rh, w, h) < 2:
        raise ValueError("Both images require a visible rubber interior")
    # The black frame anchors grid identities independently of the crosses.
    scale = np.array([(w - 1) / (rw - 1), (h - 1) / (rh - 1)])
    anchor = (reference_pixels - [rx, ry]) * scale + [x, y]
    roi = region.roi
    distances = np.linalg.norm(anchor[:, None] - anchor[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    pitch = float(np.median(distances.min(axis=1)))
    if pitch <= 0 or not np.isfinite(pitch):
        raise ValueError("Reference marker positions must be distinct")
    candidates = _still_candidates(tracking_bgr, roi, pitch)
    x, y = np.rint(candidates).astype(int).T
    candidates = candidates[region.marker_mask[y, x]]
    grid_steps = _grid_period_shifts(anchor, pitch)
    ordered = None
    if template_xy is not None:
        template_xy = np.asarray(template_xy, dtype=float)
        if template_xy.shape != reference_pixels.shape or not np.isfinite(template_xy).all():
            raise ValueError("Template must contain one finite coordinate pair per marker")
        rows, cols = require_regular_template(ids, template_xy)
        if len(ids) <= len(candidates) <= len(ids) + 2:
            try:
                ordered, _ = select_complete_grid(candidates, rows, cols, row_direction)
            except (ValueError, RuntimeError):
                pass  # Use unambiguous individual observations from the frame anchor.
    if ordered is not None:
        # Crosses are symmetric under a half turn. Use the ordering whose
        # row and column directions agree with the reference orientation.
        alignment = np.sum(_grid_axes(ordered, rows, cols) * _grid_axes(reference_pixels, rows, cols), axis=1)
        if np.all(alignment < 0):
            ordered = ordered[::-1].copy()
        elif np.any(alignment <= 0):
            ordered = None
    if ordered is not None:
        # Local bending can move an individual mark past a cell midpoint.
        # Only a grid-wide shift toward adjacent IDs questions the ordering.
        offset = np.median(ordered - anchor, axis=0)
        if any(offset @ step >= 0.5 * (step @ step) for step in grid_steps):
            ordered = None
    if ordered is None:
        pixels, valid = _match_partial_grid(anchor, candidates, pitch, grid_steps)
    else:
        pixels, valid = ordered, np.ones(len(ids), bool)
    return dict(ids=ids.copy(), pixels=pixels, valid=valid,
                confidence=np.where(valid, 0.35, 0.0),
                status="ok" if valid.all() else ("partial" if valid.any() else "lost"),
                candidate_count=len(candidates), roi=roi)


class MarkerTracker:
    """Persistent IDs using forward/backward LK plus one-to-one marker matching.

    Missing points remain invalid. Nearby reacquisition uses a motion prediction
    supported by already tracked IDs; never renumbers a newly detected grid.
    """

    def __init__(self, seed, fb_threshold=1.0, gate_fraction=0.35):
        self.seed = seed
        self.previous_gray = seed.gray.copy()
        self.positions = seed.pixels.copy()
        self.valid = np.ones(len(self.positions), bool)
        self.fb_threshold = fb_threshold
        self.gate_fraction = gate_fraction
        self.failure_count = 0

    def update(self, image_linear_rgb, *, tracking_bgr=None, tracking_gray=None):
        """Track IDs from one linear-RGB frame using BT.709 detector values."""
        image_linear_rgb = validate_linear_rgb(image_linear_rgb)
        if tracking_bgr is None:
            tracking_bgr = linear_rgb_to_tracking_bgr8(image_linear_rgb)
        gray = cv2.cvtColor(tracking_bgr, cv2.COLOR_BGR2GRAY) if tracking_gray is None else tracking_gray
        if gray.shape != self.seed.gray.shape:
            raise ValueError("Frame resolution changed; reinitialize with matching calibration")
        n = len(self.positions)
        valid = np.zeros(n, bool)
        confidence = np.zeros(n, float)
        pixels = np.full((n, 2), np.nan)
        fb = np.full(n, np.inf)
        # Established IDs give a moving search window; do not rediscover the border every frame.
        lo = np.maximum(np.floor(self.positions.min(0) - 30), 0).astype(int)
        hi = np.minimum(np.ceil(self.positions.max(0) + 30), [gray.shape[1], gray.shape[0]]).astype(
            int
        )
        roi = (*lo, *(hi - lo))
        candidates, _, _ = detector.detect_cross_centers(gray, roi)
        active = np.flatnonzero(self.valid)
        predicted = self.positions.copy()
        reliable = np.zeros(n, bool)
        distances = np.linalg.norm(self.positions[:, None] - self.positions[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        pitch = float(np.median(distances.min(axis=1)))
        if len(active):
            p = np.ascontiguousarray(self.positions[active, None, :], dtype=np.float32)
            q, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                p,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if q is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.previous_gray, q, None, winSize=(21, 21), maxLevel=3
                )
                if back is not None:
                    fb[active] = np.linalg.norm(back.reshape(-1, 2) - p.reshape(-1, 2), axis=1)
                    finite = np.isfinite(q.reshape(-1, 2)).all(1)
                    predicted[active] = np.where(
                        finite[:, None], q.reshape(-1, 2), self.positions[active]
                    )
                    reliable[active] = (
                        (status.ravel() > 0)
                        & (back_status.ravel() > 0)
                        & (fb[active] < self.fb_threshold)
                        & finite
                    )
        gate = self.gate_fraction * pitch
        candidate_ids = np.full(n, -1, int)
        good = np.flatnonzero(reliable)
        if len(good) and len(candidates):
            costs = np.linalg.norm(predicted[good, None] - candidates[None], axis=2)
            # Dummy columns permit unmatched IDs, rather than forcing bad matches.
            rr, cc = linear_sum_assignment(
                np.column_stack((costs, np.full((len(good), len(good)), gate)))
            )
            for r, c in zip(rr, cc):
                if c < len(candidates) and costs[r, c] < gate:
                    idx = good[r]
                    pixels[idx] = candidates[c]
                    valid[idx] = True
                    candidate_ids[idx] = c
                    confidence[idx] = max(0.1, 1 - fb[idx] / self.fb_threshold) * max(
                        0.1, 1 - costs[r, c] / gate
                    )

        # Forward/backward LK can agree on the wrong repeated cross. Establish a
        # widely supported motion model before treating any provisional match as
        # an ID anchor; local deviations up to the matching gate remain allowed.
        inferred = _predict_supported_motion(self.positions, pixels, valid)
        if inferred is not None:
            inferred, changed_motion = _select_motion_prediction(
                self.positions, inferred, candidates, candidate_ids, valid, gate)
            if changed_motion:
                # Reassociate from the winning motion, never from predicted points.
                valid[:] = False
                pixels[:] = np.nan
                confidence[:] = 0
        if inferred is None:
            # Without a spatially supported model, the apparent LK matches may
            # all belong to another grid column. Keep the previous frame state.
            valid[:] = False
            pixels[:] = np.nan
            confidence[:] = 0
        else:
            inconsistent = valid & (np.linalg.norm(pixels - inferred, axis=1) >= gate)
            valid[inconsistent] = False
            pixels[inconsistent] = np.nan
            confidence[inconsistent] = 0
            used = set(candidate_ids[valid])
            lost = np.flatnonzero(~valid)
            available = np.array([j for j in range(len(candidates)) if j not in used], int)
            if len(lost) and len(available):
                costs = np.linalg.norm(
                    inferred[lost, None] - candidates[available][None], axis=2
                )
                rr, cc = linear_sum_assignment(
                    np.column_stack((costs, np.full((len(lost), len(lost)), gate * 0.7)))
                )
                for r, c in zip(rr, cc):
                    alternatives = np.sort(costs[r])
                    if (
                        c < len(available)
                        and costs[r, c] < gate * 0.7
                        and (
                            len(alternatives) < 2
                            or alternatives[1] - alternatives[0] > gate * 0.25
                        )
                    ):
                        idx = lost[r]
                        pixels[idx] = candidates[available[c]]
                        valid[idx] = True
                        confidence[idx] = 0.35
            predicted[~valid] = inferred[~valid]
        # Advance only with enough evidence; a lost frame does not silently reset IDs.
        if valid.sum() >= 8:
            self.positions = np.where(valid[:, None], pixels, predicted)
            self.previous_gray = gray.copy()
            self.valid = valid.copy()
            self.failure_count = 0
        else:
            self.failure_count += 1
        return {
            "ids": self.seed.ids.copy(),
            "pixels": pixels,
            "valid": valid,
            "confidence": confidence,
            "fb_error_px": fb,
            "candidate_count": len(candidates),
            "roi": roi,
            "status": "ok" if valid.all() else ("partial" if valid.sum() >= 8 else "lost"),
            "consecutive_lost_frames": self.failure_count,
        }
