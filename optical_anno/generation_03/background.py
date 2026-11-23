"""Geometry-conditioned batch backgrounds on a shared material grid."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calibrate.utils.norm_pipeline import FrameCalibrationData
from optical_anno.generation_01.detector import ContactBackground, _prepare
from utils.image_io import validate_linear_rgb
from utils.surface_geometry import compute_surface_geometry
from utils.surface_sampling import rasterize_surface_grid, sample_linear_rgb_to_surface


# Normal differences are chord lengths between unit vectors. A bounded weight
# ratio keeps a few geometrically close contacts from dominating the median.
_NORMAL_BANDWIDTH = 2 * np.sin(np.deg2rad(5) / 2)
_MAX_NORMAL_DIFFERENCE = 2 * np.sin(np.deg2rad(15) / 2)
_MIN_BACKGROUND_OBSERVATIONS = 4


def _check_projection(uv: np.ndarray) -> None:
    """Require a locally regular image projection for the inverse rasterizer."""
    uv = np.asarray(uv, np.float64)
    finite = np.isfinite(uv).all(axis=-1)
    cells = finite[:-1, :-1] & finite[:-1, 1:] & finite[1:, :-1] & finite[1:, 1:]
    if not cells.any():
        raise ValueError('Material grid has no finite projected cells')
    p, q = uv[:-1, :-1][cells], uv[:-1, 1:][cells]
    r, s = uv[1:, :-1][cells], uv[1:, 1:][cells]
    horizontal = np.stack((q - p, q - p, s - r, s - r))
    vertical = np.stack((r - p, s - q, r - p, s - q))
    area = horizontal[..., 0] * vertical[..., 1] - horizontal[..., 1] * vertical[..., 0]
    scale = np.linalg.norm(horizontal, axis=-1) * np.linalg.norm(vertical, axis=-1)
    orientation = np.sign(area[0, 0])
    if np.any(scale <= 1e-12) or np.any(area * orientation <= 1e-8 * scale):
        raise ValueError('Material grid projection is folded or degenerate')


def _normal_grid(frame: FrameCalibrationData) -> tuple[np.ndarray, np.ndarray]:
    grid = frame.systems.lightfield
    _check_projection(grid.uv)
    geometry = compute_surface_geometry(grid.xyz, grid.valid_mask)
    return geometry['normal'].astype(np.float32), geometry['valid']


def _crop(mask: np.ndarray, scale: float) -> tuple[int, int, int, int]:
    x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
    margin = max(4, round(12 * scale))
    return (max(0, x - margin), max(0, y - margin),
            min(mask.shape[1], x + width + margin), min(mask.shape[0], y + height + margin))


def _weighted_median(colors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Median over frames, independently for each linear RGB channel."""
    result = np.empty(colors.shape[1:], np.float32)
    half_weight = weights.sum(axis=0) * 0.5
    for channel in range(3):
        values = colors[..., channel]
        order = np.argsort(values, axis=0)
        sorted_weights = np.take_along_axis(weights, order, axis=0)
        sorted_values = np.take_along_axis(values, order, axis=0)
        selected = np.argmax(np.cumsum(sorted_weights, axis=0) >= half_weight, axis=0)
        result[..., channel] = np.take_along_axis(sorted_values, selected[None], axis=0)[0]
    return result


@dataclass
class MaterialBackground:
    """Pressed-frame samples tied to one frozen material-coordinate chart.

    The output of ``for_frame`` is an image-sized background compatible with
    generation 2 evidence. Its interior mask admits only projected grid points
    with enough geometrically similar observations. Input frames can include
    contacts, provided a majority of the weighted observations at each
    material point show the unpressed appearance.

    ``overview_crop`` grows to include regions queried with ``for_frame``.
    Read it after processing all target frames when making a batch overview.
    """

    overview_crop: tuple[int, int, int, int]
    _chart_id: str = field(repr=False)
    _coordinates: np.ndarray = field(repr=False)
    _st: np.ndarray = field(repr=False)
    _image_shape: tuple[int, int, int] = field(repr=False)
    image_keys: tuple[str, ...] = field(repr=False)
    _colors: np.ndarray = field(repr=False)
    _normals: np.ndarray = field(repr=False)
    _valid: np.ndarray = field(repr=False)
    _scale: float = field(repr=False)

    def _check_frame(self, frame: FrameCalibrationData) -> None:
        grid = frame.systems.lightfield
        if validate_linear_rgb(frame.linear_rgb).shape != self._image_shape:
            raise ValueError('Background frame image sizes differ')
        if (frame.chart.chart_id != self._chart_id or grid.chart_id != self._chart_id
                or grid.st is None or grid.st.shape != self._st.shape
                or grid.coordinates.shape != self._coordinates.shape
                or not np.allclose(grid.st, self._st, rtol=0, atol=1e-8)
                or not np.allclose(grid.coordinates, self._coordinates, rtol=0, atol=1e-8)):
            raise ValueError('Background frames require the same frozen material grid')
        region = np.asarray(frame.surface_mask, bool)
        if region.shape != self._image_shape[:2] or not region.any():
            raise ValueError('Background target requires a rubber region')

    def for_frame(self, frame: FrameCalibrationData) -> ContactBackground:
        """Predict the target's background, excluding every sample with its key."""
        self._check_frame(frame)
        grid = frame.systems.lightfield
        target_normal, target_valid = _normal_grid(frame)
        selected = np.asarray([key != frame.image_key for key in self.image_keys])
        count_frames = int(selected.sum())
        colors, normals, valid = self._colors[selected], self._normals[selected], self._valid[selected]
        difference = np.linalg.norm(normals - target_normal[None], axis=-1)
        valid = valid & target_valid[None] & (difference <= _MAX_NORMAL_DIFFERENCE)
        weight = np.where(valid, 0.5 + 0.5 * np.exp(-0.5 * (difference / _NORMAL_BANDWIDTH) ** 2), 0.0)
        support = valid.sum(axis=0) >= _MIN_BACKGROUND_OBSERVATIONS
        median = _weighted_median(colors, weight)
        projection = rasterize_surface_grid(median, grid.uv, support, self._image_shape[:2])
        region = np.asarray(frame.surface_mask, bool)
        border = max(2, round(5 * self._scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
        interior = projection.valid & cv2.erode(region.astype(np.uint8), kernel).astype(bool)
        if not interior.any():
            raise ValueError('Insufficient matching material background support')

        crop = _crop(region, self._scale)
        old = self.overview_crop
        self.overview_crop = (min(old[0], crop[0]), min(old[1], crop[1]),
                              max(old[2], crop[2]), max(old[3], crop[3]))
        return ContactBackground(projection.values.astype(np.float32), region, interior,
                                 self.overview_crop, self._scale, count_frames)


def fit_material_background(frames: list[FrameCalibrationData]) -> MaterialBackground:
    """Sample smoothed pressed frames and local normals at shared material indices."""
    if len(frames) <= _MIN_BACKGROUND_OBSERVATIONS:
        raise ValueError('At least five frames are required for a batch background')
    first = frames[0]
    image = validate_linear_rgb(first.linear_rgb)
    grid = first.systems.lightfield
    if grid.st is None or grid.chart_id != first.chart.chart_id:
        raise ValueError('Background requires a frozen material grid')
    background = MaterialBackground(
        (image.shape[1], image.shape[0], 0, 0), first.chart.chart_id,
        np.array(grid.coordinates, copy=True), np.array(grid.st, copy=True), image.shape,
        (), np.empty(0), np.empty(0), np.empty(0), min(image.shape[:2]) / 480.0,
    )
    keys, colors, normals, valid = [], [], [], []
    for frame in frames:
        background._check_frame(frame)
        if frame.image_key in keys:
            raise ValueError('Background image keys must be unique')
        try:
            normal, normal_valid = _normal_grid(frame)
            smoothed, region, _ = _prepare(frame.linear_rgb, background._scale)
        except (ValueError, RuntimeError, cv2.error):
            continue
        grid = frame.systems.lightfield
        sampled = sample_linear_rgb_to_surface(smoothed, grid.uv, region)
        sample_valid = sampled.valid & normal_valid
        if not sample_valid.any():
            continue
        keys.append(frame.image_key)
        colors.append(sampled.values)
        normals.append(normal)
        valid.append(sample_valid)
        crop = _crop(region, background._scale)
        old = background.overview_crop
        background.overview_crop = (min(old[0], crop[0]), min(old[1], crop[1]),
                                    max(old[2], crop[2]), max(old[3], crop[3]))
    if len(keys) <= _MIN_BACKGROUND_OBSERVATIONS:
        raise ValueError('At least five frames need usable geometry and a detectable rubber region')
    background.image_keys = tuple(keys)
    background._colors = np.stack(colors).astype(np.float32)
    background._normals = np.stack(normals).astype(np.float32)
    background._valid = np.stack(valid)
    return background
