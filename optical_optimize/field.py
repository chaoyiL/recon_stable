"""Register the rubber by its crosses and measure its linear RGB field."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from reconstruction.surface_reconstruction.markers import (
    detect_cross_centers,
    filter_centers_by_mask,
    find_inner_region,
)
from reconstruction.surface_reconstruction.tracking import select_complete_grid
from utils.color_difference import erode_valid_mask
from utils.image_io import linear_rgb_to_tracking_bgr8


@dataclass
class Surface:
    pixels: np.ndarray  # Ordered marker centers, (x, y) in camera pixels.
    mask: np.ndarray  # Interior of the black frame, H×W.


def detect_surface(image: np.ndarray) -> Surface:
    """Localize the full 6×8 marker grid under strongly colored illumination."""
    bgr = linear_rgb_to_tracking_bgr8(image)
    for gray in (bgr.max(axis=2), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)):
        try:
            region = find_inner_region(gray)
            centers, _, diagnostics = detect_cross_centers(gray, region.roi)
            centers, _ = filter_centers_by_mask(centers, diagnostics, region.marker_mask)
            pixels, _ = select_complete_grid(centers, 6, 8, "auto")
            return Surface(pixels, region.mask)
        except (ValueError, RuntimeError) as error:
            detection_error = str(error)
    raise ValueError(f"需要完整的 6×8 胶皮十字网格，检测失败：{detection_error}")


def marker_mask(surface: Surface) -> np.ndarray:
    """Exclude the black crosses and their immediate colored fringes."""
    region = erode_valid_mask(surface.mask, 8).astype(np.uint8)
    for point in np.rint(surface.pixels).astype(int):
        cv2.circle(region, tuple(point), 7, 0, -1)
    return region.astype(bool)


@dataclass
class Comparison:
    reference: np.ndarray
    transform: np.ndarray  # Camera pixels to reference pixels.
    mask: np.ndarray  # Common, marker-free region in reference pixel coordinates.
    grid_indices: np.ndarray
    grid_counts: np.ndarray
    grid_rows: np.ndarray
    grid_columns: np.ndarray
    registration: dict

    @classmethod
    def build(
        cls,
        reference: np.ndarray,
        observation: np.ndarray,
        observed_surface: Surface | None = None,
    ) -> "Comparison":
        source = observed_surface if observed_surface is not None else detect_surface(observation)
        target = detect_surface(reference)
        transform, _ = cv2.findHomography(source.pixels, target.pixels, 0)
        if transform is None:
            raise ValueError("胶皮十字网格无法配准")
        projected = cv2.perspectiveTransform(source.pixels[:, None, :], transform)[:, 0]
        errors = np.linalg.norm(projected - target.pixels, axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        if not np.isfinite(rms) or rms > 4 or float(errors.max()) > 10:
            raise ValueError(f"胶皮网格配准误差过大，RMS={rms:.2f} px")

        height, width = reference.shape[:2]
        observed_mask = cv2.warpPerspective(
            marker_mask(source).astype(np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        reference_mask = marker_mask(target)
        mask = reference_mask & observed_mask.astype(bool)
        coverage = float(mask.sum() / max(1, reference_mask.sum()))
        if coverage < 0.75:
            raise ValueError(f"配准后的胶皮有效重叠过小，覆盖率={coverage:.1%}")
        rows, columns = np.nonzero(mask)
        grid_rows = ((rows - rows.min()) * 12 // (np.ptp(rows) + 1)).astype(int)
        grid_columns = ((columns - columns.min()) * 28 // (np.ptp(columns) + 1)).astype(int)
        grid_indices = grid_rows * 28 + grid_columns
        grid_counts = np.bincount(grid_indices, minlength=12 * 28)
        return cls(reference, transform, mask, grid_indices, grid_counts, grid_rows, grid_columns, {
            "markers": len(errors),
            "rms_px": rms,
            "max_px": float(errors.max()),
            "reference_coverage": coverage,
            "reference_surface_bbox_xywh": cv2.boundingRect(target.mask.astype(np.uint8)),
            "observed_surface_bbox_xywh": cv2.boundingRect(source.mask.astype(np.uint8)),
        })

    def align(self, observation: np.ndarray) -> np.ndarray:
        """Resample linear RGB into the reference's camera pixels."""
        height, width = self.reference.shape[:2]
        return cv2.warpPerspective(observation, self.transform, (width, height))

    def features(self, aligned: np.ndarray) -> np.ndarray:
        """RGB means of nonempty spatial cells, in row/column/channel order."""
        values = aligned[self.mask]
        occupied = self.grid_counts > 0
        sums = np.column_stack([
            np.bincount(self.grid_indices, weights=channel, minlength=len(self.grid_counts))
            for channel in values.T
        ])
        return (sums[occupied] / self.grid_counts[occupied, None]).ravel()

    def metrics(self, aligned: np.ndarray) -> dict:
        """Report pixelwise color error, signed regional bias and camera clipping."""
        values = aligned[self.mask]
        target = self.reference[self.mask]
        residual = values - target
        return {
            "pixels": len(values),
            "rgb_rmse": float(np.sqrt(np.mean(residual**2))),
            "field_rmse": float(np.sqrt(np.mean(
                (self.features(aligned) - self.features(self.reference))**2
            ))),
            "rgb_rmse_by_channel": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
            "mean_rgb": values.mean(axis=0).tolist(),
            "reference_mean_rgb": target.mean(axis=0).tolist(),
            "mean_error_rgb": residual.mean(axis=0).tolist(),
            "clipped_high_fraction_rgb": (values >= .99).mean(axis=0).tolist(),
            "reference_clipped_high_fraction_rgb": (target >= .99).mean(axis=0).tolist(),
            "regional_error_rgb": {
                label: residual[selection].mean(axis=0).tolist()
                for label, selection in (
                    ("left", self.grid_columns < 9),
                    ("center", (self.grid_columns >= 9) & (self.grid_columns < 19)),
                    ("right", self.grid_columns >= 19),
                    ("top", self.grid_rows < 6),
                    ("bottom", self.grid_rows >= 6),
                )
            },
        }

    def write_preview(self, path: Path, observations: list[tuple[str, np.ndarray]]) -> None:
        """Write aligned images and signed residuals at a fixed display scale."""
        x, y, width, height = cv2.boundingRect(self.mask.astype(np.uint8))
        region = np.s_[y:y + height, x:x + width]
        panes = []
        for label, aligned in [("Reference", self.reference), *observations]:
            panel = linear_rgb_to_tracking_bgr8(aligned[region])
            panel[~self.mask[region]] = 0
            residual = np.clip(.5 + 2 * (aligned[region] - self.reference[region]), 0, 1)
            delta = np.rint(255 * residual[..., ::-1]).astype(np.uint8)
            delta[~self.mask[region]] = 0
            row = np.zeros((height + 46, 2 * width + 36, 3), np.uint8)
            row[34:34 + height, 12:12 + width] = panel
            row[34:34 + height, width + 24:2 * width + 24] = delta
            cv2.putText(row, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .66, (255, 255, 255), 2)
            cv2.putText(
                row, "RGB residual (gray = 0)", (width + 24, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2,
            )
            panes.append(row)
        if not cv2.imwrite(str(path), np.vstack(panes)):
            raise OSError(f"预览文件写入失败：{path}")
