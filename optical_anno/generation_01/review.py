"""Generate PNG overlays for reviewing contact annotations."""

from pathlib import Path

import cv2
import numpy as np

from utils.region_annotations import RegionAnnotation, region_boundary_points


REVIEW_COLORS = ((40, 220, 255), (85, 255, 85))  # BGR: visual, automatic.


def contact_overlay(
    image: np.ndarray, annotations: list[tuple[str, RegionAnnotation | None, tuple[int, int, int]]]
) -> np.ndarray:
    """Draw contact footprints on a BGR8 preview at original pixel coordinates."""
    preview = image.copy()
    for index, (label, region, color) in enumerate(annotations):
        text = label
        if region is None:
            text += '  NO PROPOSAL'
        else:
            points = np.rint(region_boundary_points(region)).astype(np.int32)
            center = tuple(np.rint(region.center_px).astype(int))
            cv2.polylines(preview, [points], True, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.polylines(preview, [points], True, color, 1, cv2.LINE_AA)
            cv2.drawMarker(preview, center, color, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
        cv2.putText(preview, text, (10, 24 + 21 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, color, 1, cv2.LINE_AA)
    return preview


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'写入 PNG 失败：{path}')


def contact_sheet(
    images: list[tuple[str, np.ndarray]],
    *,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    columns: int = 4,
) -> np.ndarray:
    """Make a labeled grid of equally sized previews, optionally cropped to the rubber."""
    tiles = []
    for name, image in images:
        if crop_xyxy is not None:
            x0, y0, x1, y1 = crop_xyxy
            image = image[y0:y1, x0:x1]
        scale = min(1., 360 / image.shape[1])
        thumb = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(thumb, 23, 5, 4, 4, cv2.BORDER_CONSTANT)
        cv2.putText(tile, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .47,
                    (230, 230, 230), 1, cv2.LINE_AA)
        tiles.append(tile)
    tiles.extend([np.zeros_like(tiles[0])] * ((-len(tiles)) % columns))
    return np.vstack([
        np.hstack(tiles[start:start + columns]) for start in range(0, len(tiles), columns)
    ])
