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
# Search limits must include complete cross arms, not only their centers.
# Exclude the reflective right frame from the current camera view.
DETECT_X_MAX = 485
DETECT_X_MIN = 140


def _analyze_components(gray, roi_rect):
    """Local threshold; no fixed inset and no erosion of thin cross arms."""
    x, y, w, h = map(int, roi_rect)
    if x < 0 or y < 0 or w < 3 or h < 3 or x + w > gray.shape[1] or y + h > gray.shape[0]:
        raise ValueError("ROI must be a positive rectangle inside the image")
    # Clip the search in source-image pixels; both X limits are inclusive.
    right = min(x + w, DETECT_X_MAX + 1)
    x = max(x, DETECT_X_MIN)
    w = right - x
    if w < 3:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((h, 0), dtype=np.uint8),
            [],
            [],
        )
    # Compute on the full image so the ROI boundary cannot affect thresholding.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        LOCAL_BLOCK_SIZE,
        LOCAL_THRESHOLD_C,
    )
    contrast = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    crop = binary[y : y + h, x : x + w].copy()
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
        component = labels[top : top + height, left : left + width] == label
        patch = contrast[y + top : y + top + height, x + left : x + left + width]
        peak = float(patch[component].max())
        if peak < MIN_PEAK_CONTRAST:
            continue
        yy, xx = np.nonzero(component)
        # Local darkness weights reduce the effect of nonuniform illumination.
        weights = patch[component].astype(float)
        cx = x + left + float(np.average(xx, weights=weights))
        cy = y + top + float(np.average(yy, weights=weights))
        centers.append([cx, cy])
        components.append(np.column_stack((xx+x+left, yy+y+top)))
        debug.append(
            dict(
                area=area,
                width=width,
                height=height,
                peak_contrast=peak,
                center=(cx, cy),
                bbox=(x + left, y + top, width, height),
            )
        )
    return np.asarray(centers, dtype=float).reshape(-1, 2), crop, debug, components


def _detect_components(gray, roi_rect):
    return _analyze_components(gray, roi_rect)[:3]


@dataclass(frozen=True)
class InnerRegion:
    roi: tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray


def find_inner_region(gray):
    """Find the framed paper interior even when glare breaks a global mask.

    Closing is applied only to the frame search, never to mark localization.
    A bounding rectangle retains marks in shadows that indent the white contour.
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
            # Keep ROI discovery consistent with component detection.
            search_x = max(x, DETECT_X_MIN)
            search_w = min(x + w, DETECT_X_MAX + 1) - search_x
            if search_w < 3:
                continue
            rect = (search_x, y, search_w, h)
            if parent < 0 or not 1000 < area < 0.8 * gray.size or min(w, h) < 35:
                continue
            if cv2.contourArea(contours[parent]) > 8 * area:
                continue
            if rect not in candidates:
                centers, _, _ = _detect_components(gray, rect)
                if len(centers) >= 8:
                    candidates[rect] = ((len(centers), -area), contour.copy())
    if not candidates:
        raise RuntimeError(
            "Paper border not found; show the complete marked region for initialization"
        )
    roi = max(candidates, key=lambda rect: candidates[rect][0])
    contour = candidates[roi][1]
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)
    return InnerRegion(roi, contour, mask.astype(bool))



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
        roi = find_inner_roi(gray)
        centers, binary, diagnostics = detect_cross_centers(gray, roi)
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
