"""Cross detection, candidate visualization, and conservative grid acquisition."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    from utils.color_difference import bgr8_to_linear_rgb
    from utils.image_io import (
        linear_rgb_to_bgr8, linear_rgb_to_tracking_bgr8, load_linear_image,
        validate_linear_rgb,
    )
except ModuleNotFoundError:  # Direct script execution puts this directory first.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.color_difference import bgr8_to_linear_rgb
    from utils.image_io import (
        linear_rgb_to_bgr8, linear_rgb_to_tracking_bgr8, load_linear_image,
        validate_linear_rgb,
    )

LOCAL_BLOCK_SIZE = 15
LOCAL_THRESHOLD_C = 15
MIN_PEAK_CONTRAST = 60
MIN_AREA = 3
MAX_AREA = 300
MIN_WIDTH = 2
MAX_WIDTH = 30
MIN_HEIGHT = 2
MAX_HEIGHT = 30


def _has_four_cross_arms(points, center):
    """Require four supported arms for a recovered fragmented cross."""
    dx, dy = (points - center).T
    for angle in range(0, 90, 15):
        theta = np.deg2rad(angle)
        along = dx * np.cos(theta) + dy * np.sin(theta)
        across = -dx * np.sin(theta) + dy * np.cos(theta)
        # Four 45-degree sectors represent a pair of perpendicular strokes.
        limit = np.tan(np.pi / 8)
        arms = (
            np.count_nonzero((along >= 2) & (np.abs(across) <= limit * along)),
            np.count_nonzero((along <= -2) & (np.abs(across) <= -limit * along)),
            np.count_nonzero((across >= 2) & (np.abs(along) <= limit * across)),
            np.count_nonzero((across <= -2) & (np.abs(along) <= -limit * across)),
        )
        if min(arms) >= 3:
            return True
    return False


def _has_cross_indents(component):
    """Require two concave notches, tolerating a missing arm and rotation."""
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours[0]) < 4:
        return False
    hull = cv2.convexHull(contours[0], returnPoints=False)
    if hull is None or len(hull) < 3:
        return False
    try:
        defects = cv2.convexityDefects(contours[0], hull)
    except cv2.error:
        return False  # Self-intersecting contours do not define reliable notches.
    # OpenCV stores notch depth in fixed-point pixels with 8 fractional bits.
    return defects is not None and np.count_nonzero(defects.reshape(-1, 4)[:, 3] >= 256) >= 2


def _component_candidates(binary, contrast, roi_rect):
    x, y, w, h = map(int, roi_rect)
    crop = binary[y:y+h, x:x+w].copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    centers, debug, components = [], [], []
    for label in range(1, n):
        left, top, width, height, area = map(int, stats[label])
        if not (
            MIN_AREA <= area <= MAX_AREA
            and MIN_WIDTH <= width <= MAX_WIDTH
            and MIN_HEIGHT <= height <= MAX_HEIGHT
        ):
            continue
        # Partial border fragments can look like small crosses after ROI clipping.
        if left == 0 or top == 0 or left + width >= w or top + height >= h:
            continue
        if not 0.15 <= width / height <= 6.0:
            continue
        # Solid occluders are not cross-shaped even when their size and
        # contrast match a marker. Leave room for thick or distorted arms.
        if area / (width * height) > 0.8:
            continue
        component = labels[top:top+height, left:left+width] == label
        patch = contrast[y+top:y+top+height, x+left:x+left+width]
        peak = float(patch[component].max())
        if peak < MIN_PEAK_CONTRAST or not _has_cross_indents(component):
            continue
        yy, xx = np.nonzero(component)
        # Local darkness weights reduce the effect of nonuniform illumination.
        weights = patch[component].astype(float)
        cx = x + left + float(np.average(xx, weights=weights))
        cy = y + top + float(np.average(yy, weights=weights))
        points = np.column_stack((xx+x+left, yy+y+top))
        centers.append([cx, cy])
        components.append(points)
        debug.append(
            dict(area=area, width=width, height=height, peak_contrast=peak,
                 center=(cx, cy), bbox=(x+left, y+top, width, height))
        )
    if len(debug) >= 8:
        sizes = np.asarray([(item["width"], item["height"]) for item in debug], dtype=float)
        min_size = np.maximum((MIN_WIDTH, MIN_HEIGHT), 0.5 * np.median(sizes, axis=0))
        keep = np.all(sizes >= min_size, axis=1)
        centers = np.asarray(centers, dtype=float)[keep].tolist()
        debug = [item for item, accepted in zip(debug, keep) if accepted]
        components = [item for item, accepted in zip(components, keep) if accepted]
    return np.asarray(centers, dtype=float).reshape(-1, 2), crop, debug, components


def _fragmented_cross(component, binary, contrast):
    """Recover only short gaps between four supported arms of a dark cross."""
    points = component[binary[component[:, 1], component[:, 0]] != 0]
    if len(points) < MIN_AREA:
        return None
    lo, hi = points.min(axis=0), points.max(axis=0)
    width, height = (hi - lo + 1).tolist()
    if not 0.5 <= width / height <= 2:
        return None
    mask = np.zeros((height, width), np.uint8)
    mask[points[:, 1]-lo[1], points[:, 0]-lo[0]] = 1
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 3 or np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] >= 2) < 2:
        return None
    weights = contrast[points[:, 1], points[:, 0]].astype(float)
    if weights.sum() == 0:
        return None
    center = np.average(points, axis=0, weights=weights)
    if not _has_four_cross_arms(points, center):
        return None
    info = dict(area=len(points), width=width, height=height,
                peak_contrast=float(weights.max()), center=tuple(center),
                bbox=(int(lo[0]), int(lo[1]), width, height))
    return center, info, points


def _analyze_components(gray, roi_rect):
    """Find crosses, retaining original foreground pixels across small gaps."""
    x, y, w, h = map(int, roi_rect)
    if x < 0 or y < 0 or w < 3 or h < 3 or x + w > gray.shape[1] or y + h > gray.shape[0]:
        raise ValueError("ROI must be a positive rectangle inside the image")
    # Compute on the full image so the ROI boundary cannot affect thresholding.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        LOCAL_BLOCK_SIZE, LOCAL_THRESHOLD_C,
    )
    contrast = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    centers, crop, debug, components = _component_candidates(binary, contrast, roi_rect)

    # Closing proposes connections only. Existing detections and returned marker
    # pixels use the original binary so a nearby edge cannot change their masks.
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, _, proposals = _component_candidates(closed, contrast, roi_rect)
    accepted_pixels = np.zeros(gray.shape, np.bool_)
    for component in components:
        accepted_pixels[component[:, 1], component[:, 0]] = True
    for component in proposals:
        if accepted_pixels[component[:, 1], component[:, 0]].any():
            continue
        recovered = _fragmented_cross(component, binary, contrast)
        if recovered is None:
            continue
        center, info, points = recovered
        if len(centers) and np.any(np.linalg.norm(centers-center, axis=1)
                                   < max(3., .5 * min(info["width"], info["height"]))):
            continue
        centers = np.vstack((centers, center))
        debug.append(info)
        components.append(points)
        accepted_pixels[points[:, 1], points[:, 0]] = True
    return centers, crop, debug, components


def _detect_components(gray, roi_rect):
    return _analyze_components(gray, roi_rect)[:3]


@dataclass(frozen=True)
class InnerRegion:
    roi: tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray
    marker_mask: np.ndarray


def _marker_edge_margin(diagnostics):
    """Return one typical marker dimension as a frame-edge safety margin."""
    if len(diagnostics) < 8:
        return None
    sizes = np.asarray(
        [(item["width"], item["height"]) for item in diagnostics], dtype=float
    )
    return int(np.ceil(np.median(sizes, axis=0).max()))


def filter_centers_by_mask(centers, diagnostics, allowed_mask):
    """Keep candidates whose centers lie inside an allowed image mask."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    mask = np.asarray(allowed_mask, dtype=bool)
    if len(diagnostics) != len(centers):
        raise ValueError("Candidate diagnostics must match candidate centers")
    pixels = np.rint(centers).astype(int)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 0] < mask.shape[1])
        & (pixels[:, 1] < mask.shape[0])
    )
    keep = np.zeros(len(centers), dtype=bool)
    indices = np.flatnonzero(inside)
    keep[indices] = mask[pixels[indices, 1], pixels[indices, 0]]
    return centers[keep], [item for item, accepted in zip(diagnostics, keep) if accepted]


def _marker_search_mask(frame_mask, diagnostics):
    """Keep marker centers one typical marker width inside the frame."""
    margin = _marker_edge_margin(diagnostics)
    if margin is None:
        return None
    distance = cv2.distanceTransform(
        np.asarray(frame_mask, dtype=np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return distance > margin


def _matching_frame_mask(gray, threshold, seed, bounds):
    """Return a closed dark-frame hole matching the seed, if one exists."""
    binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    seed_area = int(seed.sum())
    x, y, w, h = bounds
    dx, dy = max(8, 0.1 * w), max(8, 0.1 * h)
    matches = []
    for index, candidate in enumerate(contours):
        if hierarchy[0, index, 3] < 0:
            continue
        area = cv2.contourArea(candidate)
        if not 0.96 * seed_area <= area <= 1.35 * seed_area:
            continue
        cx, cy, cw, ch = cv2.boundingRect(candidate)
        if (cx < x - dx or cy < y - dy
                or cx + cw > x + w + dx or cy + ch > y + h + dy):
            continue
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [candidate], -1, 1, cv2.FILLED)
        overlap = np.count_nonzero(seed & mask)
        # Reject a frame break that leaks into the bright surroundings.
        if overlap >= 0.98 * seed_area:
            matches.append((overlap / int(mask.sum()), mask))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def _refine_inner_contour(gray, contour):
    """Trace the dark frame around a candidate without filling real concavities.

    The candidate may exclude dark paper connected to the frame in an Otsu
    threshold. Estimate the frame and paper levels near that candidate and
    search for a lower threshold that still forms a closed, matching frame.
    """
    seed = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(seed, [contour], -1, 1, cv2.FILLED)
    near = np.ones((7, 7), np.uint8)
    far = np.ones((21, 21), np.uint8)
    outside = (cv2.dilate(seed, near) != 0) & (seed == 0)
    inside = (cv2.erode(seed, near) != 0) & (cv2.erode(seed, far) == 0)
    if outside.any() and inside.any():
        frame_level = float(np.median(gray[outside]))
        paper_level = float(np.percentile(gray[inside], 10))
        if paper_level - frame_level >= 8:
            for fraction in (0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.95):
                threshold = frame_level + fraction * (paper_level - frame_level)
                match = _matching_frame_mask(gray, threshold, seed, cv2.boundingRect(contour))
                if match is not None:
                    seed = match
                    break

    # OpenCV's hole contour includes the adjacent foreground pixel. Removing
    # that pixel keeps the returned interior on the paper side of the frame.
    seed = cv2.erode(seed, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)
    return contour, mask.astype(bool)


def find_inner_region(gray):
    """Find the framed paper interior and an inset marker search region.

    Thresholding and closing locate the frame only; marker localization uses
    the original grayscale image. A darker threshold recovers shaded paper
    while preserving the shape of a closed black frame.
    """
    local = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7
    )
    global_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    candidates = {}
    for binary in (local, global_mask):
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for idx, contour in enumerate(contours):
            parent = hierarchy[0, idx, 3]
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            rect = (x, y, w, h)
            if parent < 0 or not 1000 < area < 0.8 * gray.size or min(w, h) < 35:
                continue
            if cv2.contourArea(contours[parent]) > 8 * area:
                continue
            if rect not in candidates:
                _, _, diagnostics, _ = _analyze_components(gray, rect)
                refined_contour, mask = _refine_inner_contour(gray, contour)
                marker_mask = _marker_search_mask(mask, diagnostics)
                if marker_mask is None:
                    continue
                search_roi = cv2.boundingRect(marker_mask.astype(np.uint8))
                if search_roi[2] < 3 or search_roi[3] < 3:
                    continue
                centers, _, local_diagnostics = _detect_components(gray, search_roi)
                centers, local_diagnostics = filter_centers_by_mask(
                    centers, local_diagnostics, marker_mask
                )
                if len(centers) >= 8:
                    candidates[rect] = (
                        (len(centers), -area), search_roi, refined_contour, mask, marker_mask
                    )
    if not candidates:
        raise RuntimeError(
            "Paper border not found; show the complete marked region for initialization"
        )
    _, roi, contour, mask, marker_mask = max(
        candidates.values(), key=lambda value: value[0]
    )
    return InnerRegion(roi, contour, mask, marker_mask)


def find_tracking_region(tracking_bgr):
    """Find the framed region in luminance or the strongest color channel."""
    gray = cv2.cvtColor(tracking_bgr, cv2.COLOR_BGR2GRAY)
    try:
        return gray, find_inner_region(gray)
    except RuntimeError:
        strongest_channel = tracking_bgr.max(axis=2)
        return strongest_channel, find_inner_region(strongest_channel)


def find_inner_roi(gray):
    """Compatibility wrapper returning the marker search rectangle."""
    return find_inner_region(gray).roi


def marker_reference_sizes(frame_linear_rgb, marker_pixels, *, tracking_bgr=None):
    frame_linear_rgb = validate_linear_rgb(frame_linear_rgb)
    if tracking_bgr is None:
        tracking_bgr = linear_rgb_to_tracking_bgr8(frame_linear_rgb)
    gray = cv2.cvtColor(tracking_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    centers, _, diagnostics, _ = _analyze_components(gray, (0, 0, w, h))
    result = np.full((len(marker_pixels), 2), 8., np.float64)
    for i, pixel in enumerate(marker_pixels):
        if len(centers) and np.isfinite(pixel).all():
            k = np.argmin(np.linalg.norm(centers-pixel, axis=1))
            if np.linalg.norm(centers[k]-pixel) < 3:
                result[i] = diagnostics[k]['width'], diagnostics[k]['height']
    return result


def build_marker_image_mask(frame_linear_rgb, marker_pixels, marker_valid,
                            reference_marker_sizes, *, dilation_px=1,
                            tracking_bgr=None):
    frame_linear_rgb = validate_linear_rgb(frame_linear_rgb)
    if tracking_bgr is None:
        tracking_bgr = linear_rgb_to_tracking_bgr8(frame_linear_rgb)
    gray = cv2.cvtColor(tracking_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    pixels = np.asarray(marker_pixels, np.float64)
    valid = np.asarray(marker_valid, bool)
    sizes = np.asarray(reference_marker_sizes, np.float64)
    if pixels.shape != (len(valid), 2) or sizes.shape != pixels.shape:
        raise ValueError('Marker slots and reference sizes must match')
    finite = np.isfinite(pixels).all(1)
    points = pixels[finite]
    pitch = 0.
    if len(points) > 1:
        distances = np.linalg.norm(points[:, None]-points[None], axis=2)
        np.fill_diagonal(distances, np.inf)
        pitch = float(np.median(distances.min(1)))
    centers, _, _, components = _analyze_components(gray, (0, 0, w, h))
    radii = np.maximum(np.linalg.norm(sizes, axis=1)/2+2, np.ceil(.35*pitch))
    costs = np.linalg.norm(pixels[:, None]-centers[None], axis=-1)
    nearby = finite[:, None] & (costs <= radii[:, None])
    mask = np.zeros((h, w), np.uint8)
    for i in np.flatnonzero(finite):
        candidates = np.flatnonzero(nearby[i])
        if valid[i] and len(candidates) == 1 and nearby[:, candidates[0]].sum() == 1:
            component = components[candidates[0]]
            mask[component[:, 1], component[:, 0]] = 1
        else:
            cv2.circle(mask, tuple(np.rint(pixels[i]).astype(int)), int(np.ceil(radii[i])), 1, -1)
    if dilation_px:
        mask = cv2.dilate(mask, np.ones((2*dilation_px+1, 2*dilation_px+1), np.uint8))
    return mask.astype(bool)


def detect_cross_centers(gray, roi_rect):
    """Return observed centres, binary ROI and component diagnostics."""
    return _detect_components(gray, roi_rect)


def annotate_cross_candidates(frame, roi_rect, diagnostics, message=""):
    """Draw every raw detector candidate before grid selection or numbering."""
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        canvas = frame.copy()
    else:
        raise ValueError("frame must be a grayscale or BGR image")

    if roi_rect is not None:
        x, y, w, h = map(int, roi_rect)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 170, 0), 2, cv2.LINE_AA)
    for index, item in enumerate(diagnostics):
        cx, cy = np.rint(item["center"]).astype(int)
        left, top, width, height = map(int, item["bbox"])
        cv2.rectangle(
            canvas,
            (left, top),
            (left + width, top + height),
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.drawMarker(
            canvas,
            (cx, cy),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            9,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(index),
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 80, 230),
            1,
            cv2.LINE_AA,
        )

    count = len(diagnostics)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 50), (245, 245, 245), -1)
    cv2.putText(
        canvas,
        f"raw cross candidates: {count}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    detail = message or "blue: ROI | yellow: component | red: weighted center"
    cv2.putText(
        canvas,
        detail[:92],
        (10, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (30, 30, 190) if message else (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return canvas


def inspect_cross_candidates(frame_linear_rgb, message=""):
    """Detect from BT.709 tracking values and draw over direct linear display values."""
    frame_linear_rgb = validate_linear_rgb(frame_linear_rgb)
    if frame_linear_rgb.size == 0:
        raise ValueError("empty input image")
    tracking_bgr = linear_rgb_to_tracking_bgr8(frame_linear_rgb)
    gray = cv2.cvtColor(tracking_bgr, cv2.COLOR_BGR2GRAY)
    try:
        region = find_inner_region(gray)
        roi = region.roi
        centers, binary, diagnostics = detect_cross_centers(gray, roi)
        centers, diagnostics = filter_centers_by_mask(
            centers, diagnostics, region.marker_mask
        )
        error = ""
    except (ValueError, RuntimeError, cv2.error) as exc:
        roi = None
        centers = np.empty((0, 2), dtype=float)
        binary = np.empty((0, 0), dtype=np.uint8)
        diagnostics = []
        error = str(exc)
    display_bgr = linear_rgb_to_bgr8(frame_linear_rgb)
    overlay = annotate_cross_candidates(display_bgr, roi, diagnostics, message or error)
    return overlay, binary, centers, diagnostics, roi, error


def sort_grid_points(centers, rows, cols, row_direction="auto"):
    """Assign a complete grid without assuming that the paper is upright.

    Tests both PCA axes using row separation and cell orientation. Auto uses
    positive image x/y as the direction of increasing template rows; identical
    crosses cannot resolve a physical 180-degree rotation. No points are invented.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (rows * cols, 2) or not np.isfinite(centers).all():
        raise RuntimeError(
            f"检测到 {len(centers)} 个 mark，模板要求 {rows*cols} 个；请查看候选点图"
        )
    if len(np.unique(centers, axis=0)) != len(centers):
        raise RuntimeError("检测点存在重复坐标")
    directions = {
        "down": np.array([0.0, 1.0]),
        "up": np.array([0.0, -1.0]),
        "right": np.array([1.0, 0.0]),
        "left": np.array([-1.0, 0.0]),
    }
    if row_direction != "auto" and row_direction not in directions:
        raise ValueError("row_direction must be auto/down/up/right/left")
    _, _, axes = np.linalg.svd(centers - centers.mean(0), full_matrices=False)
    candidates = []
    for vertical in axes:
        vertical = vertical.copy()
        if row_direction == "auto":
            vertical *= 1 if vertical[np.argmax(np.abs(vertical))] >= 0 else -1
        else:
            alignment = vertical @ directions[row_direction]
            if abs(alignment) < np.cos(np.pi / 4):
                continue
            vertical *= 1 if alignment >= 0 else -1
        horizontal = np.array([vertical[1], -vertical[0]])
        order = np.argsort(centers @ vertical).reshape(rows, cols)
        groups = centers[order] @ vertical
        pitches = np.diff(groups.mean(axis=1))
        if len(pitches) and (
            np.min(pitches) <= 0 or np.max(np.ptp(groups, axis=1) / np.median(pitches)) > 0.65
        ):
            continue
        for r in range(rows):
            order[r] = order[r, np.argsort(centers[order[r]] @ horizontal)]
        grid = centers[order]
        column_edges = np.diff(grid, axis=1)
        row_edges = np.diff(grid, axis=0)
        if rows > 1 and cols > 1:
            # Both triangles of each cell must retain a consistent orientation.
            cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            if np.any(cross(column_edges[:-1], row_edges[:, :-1]) <= 0) or np.any(
                cross(column_edges[1:], row_edges[:, 1:]) <= 0
            ):
                continue
        # Compare dispersion of each row across the candidate row axis.
        score = float(np.mean(np.ptp(groups, axis=1)) / (np.median(pitches) if len(pitches) else 1))
        candidates.append((score, order.ravel()))
    if not candidates:
        raise RuntimeError(
            "网格编号不可靠：请检查漏检、误检并重新拍摄清晰网格；强折叠不可直接按位置编号"
        )
    _, order = min(candidates, key=lambda item: item[0])
    return np.column_stack((np.arange(len(centers)), centers[order]))


def _print_detection(centers, diagnostics, roi, error):
    summary = f"candidates={len(centers)} roi={roi}"
    if diagnostics:
        areas = [item["area"] for item in diagnostics]
        contrasts = [item["peak_contrast"] for item in diagnostics]
        summary += (
            f" area={min(areas)}..{max(areas)}"
            f" peak_contrast={min(contrasts):.0f}..{max(contrasts):.0f}"
        )
    if error:
        summary += f" error={error}"
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Display raw cross candidates before the 48-point grid check."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Analyze one image")
    source.add_argument("--video", help="Video file or camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")

    if args.image is not None:
        try:
            frame = load_linear_image(args.image)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        overlay, binary, centers, diagnostics, roi, error = inspect_cross_candidates(frame)
        _print_detection(centers, diagnostics, roi, error)
        cv2.imshow("Raw cross candidates | Esc/Q: exit", overlay)
        if binary.size:
            cv2.imshow("Marker threshold inside ROI", binary)
        try:
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()
        return

    source_value = int(args.video) if args.video.isdigit() else args.video
    print(f"Opening video source {args.video}...", flush=True)
    capture = cv2.VideoCapture(source_value)
    if not capture.isOpened():
        parser.error(f"Cannot open video or camera: {args.video}")
    if isinstance(source_value, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    backend = capture.getBackendName()
    print(f"Opened with {backend}; waiting for frames...", flush=True)
    frame_number = 0
    try:
        while True:
            ok, encoded_bgr = capture.read()
            if not ok:
                print(
                    "No frame received" if frame_number == 0 else "Video input ended",
                    flush=True,
                )
                break
            frame_number += 1
            frame_linear_rgb = bgr8_to_linear_rgb(encoded_bgr)
            overlay, binary, centers, diagnostics, roi, error = inspect_cross_candidates(
                frame_linear_rgb)
            if frame_number == 1 or frame_number % 30 == 0:
                _print_detection(centers, diagnostics, roi, error)
            cv2.imshow("Raw cross candidates | Esc/Q: exit", overlay)
            if binary.size:
                cv2.imshow("Marker threshold inside ROI", binary)
            key = cv2.waitKey(1) & 255
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
