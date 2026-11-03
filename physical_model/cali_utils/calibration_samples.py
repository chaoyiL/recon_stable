"""Paired linear-RGB observations and tangent-slope labels for LUT fitting."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np

from physical_model.model_utils.config import validate_config
from physical_model.model_utils.curve_geometry import base_surface, deformed_surface, sphere_pose
from physical_model.model_utils.geometry import (
    camera_frame,
    contact_radius,
    height_normal,
    intersect_surface,
)
from physical_model.model_utils.renderer import encode_linear_rgb, render
from physical_model.model_utils.surface_coordinates import (
    bending_radius,
    center_material_mm,
    normal_direction,
    surface_size,
)
from .calibration_scan import ScanPose, closed_axis


COORDINATE_KIND = 'reference_arclength'
FRAME_CONVENTION = 'st_grid_columns_rows'


@dataclass(frozen=True)
class MaterialGrid:
    """Regular coordinates (s,t) in millimeters on the undeformed chart."""

    x_mm: np.ndarray
    y_mm: np.ndarray
    xy_mm: np.ndarray
    st_mm: np.ndarray
    chart_id: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.xy_mm.shape[:2]


@dataclass
class CalibrationReference:
    config: dict
    sensor: np.ndarray
    linear_rgb: np.ndarray
    image_valid: np.ndarray
    grid_color: np.ndarray
    grid_valid: np.ndarray
    grid_uv: np.ndarray
    base_points: np.ndarray
    base_normals: np.ndarray


@dataclass
class CalibrationFrame:
    pose: ScanPose
    config: dict
    sensor: np.ndarray
    linear_rgb: np.ndarray
    image_valid: np.ndarray
    gt: dict
    uv_current: np.ndarray
    color_residual_grid: np.ndarray
    photometric_valid_mask: np.ndarray
    slope_grid: np.ndarray
    sample_valid_mask: np.ndarray
    contact_mask: np.ndarray
    material_grid_index: np.ndarray
    render_seconds: float
    sampling_seconds: float

    @property
    def sample_count(self) -> int:
        return int(np.count_nonzero(self.sample_valid_mask))


def build_material_grid(config: dict) -> MaterialGrid:
    """Use the same reference-arclength axes for flat and curved surfaces."""
    width, height = map(float, surface_size(config))
    step_x, step_y = map(float, config['calibration']['material_grid']['step_xy_mm'])
    if not all(math.isfinite(value) and value > 0 for value in (step_x, step_y)):
        raise ValueError('Calibration material-grid steps must be positive and finite')
    xs = np.asarray(closed_axis(-width / 2, width / 2, step_x), np.float64)
    ys = np.asarray(closed_axis(-height / 2, height / 2, step_y), np.float64)
    s, t = np.meshgrid(xs, ys)
    xy = np.stack((s, t), axis=-1)
    st = xy + np.asarray([width / 2, height / 2])
    digest = hashlib.sha256()
    digest.update(np.asarray([width, height, step_x, step_y], '<f8').tobytes())
    digest.update(xs.astype('<f8').tobytes())
    digest.update(ys.astype('<f8').tobytes())
    return MaterialGrid(xs, ys, xy, st, 'flat-' + digest.hexdigest()[:20])


def bilinear_sample(field: np.ndarray, uv: np.ndarray,
                    source_valid: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Sample an HxWxC image with four valid neighbors at pixel coordinates."""
    image = np.asarray(field)
    coordinates = np.asarray(uv, np.float64)
    if image.ndim != 3 or coordinates.shape[-1:] != (2,):
        raise ValueError('Bilinear sampling expects HxWxC image and (...,2) pixels')
    height, width = image.shape[:2]
    finite = np.isfinite(coordinates).all(axis=-1)
    safe = np.where(finite[..., None], coordinates, 0.)
    x, y = safe[..., 0], safe[..., 1]
    valid = finite & (x >= 0) & (y >= 0) & (x < width - 1) & (y < height - 1)
    ix = np.clip(np.floor(x).astype(np.int64), 0, width - 2)
    iy = np.clip(np.floor(y).astype(np.int64), 0, height - 2)
    fx = (x - ix)[..., None]
    fy = (y - iy)[..., None]
    a, b = image[iy, ix], image[iy, ix + 1]
    c, d = image[iy + 1, ix], image[iy + 1, ix + 1]
    valid &= (np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
              & np.isfinite(c).all(axis=-1) & np.isfinite(d).all(axis=-1))
    if source_valid is not None:
        mask = np.asarray(source_valid, bool)
        if mask.shape != image.shape[:2]:
            raise ValueError('Bilinear source-valid mask must match image height and width')
        valid &= (mask[iy, ix] & mask[iy, ix + 1]
                  & mask[iy + 1, ix] & mask[iy + 1, ix + 1])
    values = ((1 - fy) * ((1 - fx) * a + fx * b)
              + fy * ((1 - fx) * c + fx * d))
    return np.where(valid[..., None], values, 0.), valid


def project_world(points: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Project world millimeters to undistorted image pixels using OpenCV axes."""
    camera = config['camera']
    points = np.asarray(points, np.float64)
    vector = points - np.asarray(camera['position_mm'], np.float64)
    camera_points = vector @ camera_frame(camera)
    z = camera_points[..., 2]
    front = np.isfinite(camera_points).all(axis=-1) & (z > 0)
    safe_z = np.where(front, z, 1.)
    uv = np.stack((camera['fx_px'] * camera_points[..., 0] / safe_z + camera['cx_px'],
                   camera['fy_px'] * camera_points[..., 1] / safe_z + camera['cy_px']), -1)
    uv = np.where(front[..., None], uv, np.nan)
    return uv, front


def base_geometry(config: dict, grid: MaterialGrid):
    """Return base points, normals and tangent frame for each (s,t) grid node."""
    s, t = grid.xy_mm[..., 0], grid.xy_mm[..., 1]
    radius = bending_radius(config)
    if radius is None:
        point = np.stack((s, t, np.zeros_like(s)), axis=-1)
        normal = np.zeros_like(point)
        normal[..., 2] = 1.
        tangent = np.zeros((*s.shape, 3, 2), np.float64)
        tangent[..., 0, 0] = 1.
        tangent[..., 1, 1] = 1.
        return point, normal, tangent
    point, normal = base_surface(s, t, radius)
    angle = s / radius
    tangent = np.zeros((*s.shape, 3, 2), np.float64)
    tangent[..., 0, 0] = np.cos(angle)
    tangent[..., 2, 0] = -np.sin(angle)
    tangent[..., 1, 1] = 1.
    return point, normal, tangent


def sample_geometry(config: dict, grid: MaterialGrid, pose: ScanPose):
    """Evaluate exact sphere-contact points and slopes in the base tangent frame."""
    s, t = grid.xy_mm[..., 0], grid.xy_mm[..., 1]
    base, normal, tangent = base_geometry(config, grid)
    curved = bending_radius(config) is not None
    if curved:
        point, deformed_normal, displacement = deformed_surface(s, t, config)
        offset = np.asarray(sphere_pose(config)[2]) - base
        sphere_inside = (np.sum(offset * offset, axis=-1)
                         < float(config['indentation']['radius_mm']) ** 2)
        contact = sphere_inside & (pose.depth_mm > 0)
    else:
        z, _ = height_normal(s, t, config)
        point = np.stack((s, t, z), axis=-1)
        depth = float(config['indentation']['depth_mm'])
        contact_r = float(contact_radius(config))
        center = center_material_mm(config)
        distance2 = (s - center[0]) ** 2 + (t - center[1]) ** 2
        margin = float(config['calibration']['sampling'].get('contact_margin_mm', 0.))
        contact = (distance2 < max(contact_r - margin, 0.) ** 2) & (depth > 0)
        sphere_center = np.array([
            center[0], center[1],
            normal_direction(config) * (depth - float(config['indentation']['radius_mm']))
        ])
        radius = float(config['indentation']['radius_mm'])
        cap_normal = normal_direction(config) * (point - sphere_center) / radius
        deformed_normal = np.where(contact[..., None], cap_normal, normal)
        displacement = z * float(normal_direction(config))

    dot = np.sum(normal * deformed_normal, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = -np.einsum('...ij,...i->...j', tangent, deformed_normal) / dot[..., None]
    return point, normal, deformed_normal, tangent, slopes, contact, displacement


def _point_visibility(points: np.ndarray, normals: np.ndarray, config: dict) -> np.ndarray:
    camera_origin = np.asarray(config['camera']['position_mm'], np.float64)
    vector = points - camera_origin
    distance = np.linalg.norm(vector, axis=-1)
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(distance) & (distance > 0)
    safe_distance = np.where(finite, distance, 1.)
    direction = vector / safe_distance[..., None]
    first_hit = intersect_surface(camera_origin, direction, config)
    facing = np.sum(normals * -direction, axis=-1) > 0
    return finite & facing & np.isclose(first_hit, distance, rtol=1e-7, atol=1e-6)


def render_observation(config: dict) -> dict:
    started = time.perf_counter()
    sensor, _, gt = render(config, device=config['render'].get('device', 'cpu'))
    linear = encode_linear_rgb(sensor, config)
    valid = np.asarray(gt['mask'], bool) & np.isfinite(linear).all(axis=-1)
    return dict(sensor=sensor, linear_rgb=linear, gt=gt, image_valid=valid,
                elapsed_seconds=time.perf_counter() - started)


def render_reference(config: dict, grid: MaterialGrid) -> CalibrationReference:
    """Render once at zero indentation and sample the undeformed material chart."""
    reference_config = {**config, 'indentation': dict(config['indentation'])}
    reference_config['indentation']['depth_mm'] = 0.
    validate_config(reference_config, allow_curve=True, allow_contact_tangent=True)
    frame = render_observation(reference_config)
    base_points, base_normals, _ = base_geometry(reference_config, grid)
    uv, front = project_world(base_points, reference_config)
    color, sampled = bilinear_sample(frame['linear_rgb'], uv, frame['image_valid'])
    visible = _point_visibility(base_points, base_normals *
                                float(reference_config['surface']['view_side_z']),
                                reference_config)
    valid = front & sampled & visible
    return CalibrationReference(reference_config, frame['sensor'], frame['linear_rgb'],
                                frame['image_valid'], color, valid, uv,
                                base_points, base_normals)


def _pose_config(config: dict, pose: ScanPose) -> dict:
    result = {**config, 'indentation': dict(config['indentation'])}
    result['indentation']['center_uv'] = list(pose.center_uv)
    result['indentation'].pop('center_xy_mm', None)
    result['indentation'].pop('direction_z', None)
    if 'direction_normal' not in result['indentation']:
        result['indentation']['direction_normal'] = normal_direction(config)
    result['indentation']['depth_mm'] = float(pose.depth_mm)
    return result


def sample_pose(config: dict, reference: CalibrationReference,
                grid: MaterialGrid, pose: ScanPose) -> CalibrationFrame:
    current_config = _pose_config(config, pose)
    validate_config(current_config, allow_curve=True, allow_contact_tangent=True)
    frame = render_observation(current_config)
    start = time.perf_counter()
    point, base_normal, normal, _, slopes, contact, _ = sample_geometry(
        current_config, grid, pose)
    uv, front = project_world(point, current_config)
    observed, sampled = bilinear_sample(frame['linear_rgb'], uv, frame['image_valid'])
    visible = _point_visibility(point, normal *
                                float(current_config['surface']['view_side_z']), current_config)
    residual = observed - reference.grid_color
    normal_valid = ((np.sum(base_normal * normal, axis=-1) > .1)
                    & np.isfinite(slopes).all(axis=-1))
    photo_valid = (reference.grid_valid & sampled & front & visible
                   & np.isfinite(residual).all(axis=-1))
    valid = contact & photo_valid & normal_valid
    residual = np.where(photo_valid[..., None], residual, 0.)
    rows, columns = np.indices(grid.shape, dtype=np.int32)
    grid_index = np.stack((rows, columns), axis=-1)
    return CalibrationFrame(
        pose=pose,
        config=current_config,
        sensor=frame['sensor'],
        linear_rgb=frame['linear_rgb'],
        image_valid=frame['image_valid'],
        gt=frame['gt'],
        uv_current=uv,
        color_residual_grid=residual,
        photometric_valid_mask=photo_valid,
        slope_grid=slopes,
        sample_valid_mask=valid,
        contact_mask=contact,
        material_grid_index=grid_index,
        render_seconds=frame['elapsed_seconds'],
        sampling_seconds=time.perf_counter() - start,
    )


def compact_frame_samples(frame: CalibrationFrame, grid: MaterialGrid) -> dict[str, np.ndarray]:
    valid = frame.sample_valid_mask
    selected = np.argwhere(valid).astype(np.int32)
    rows, columns = selected[:, 0], selected[:, 1]
    return {
        'color_residual_samples': frame.color_residual_grid[valid].astype(np.float32),
        'slope_samples': frame.slope_grid[valid].astype(np.float32),
        'material_coordinates': grid.xy_mm[rows, columns].astype(np.float64),
        'st_samples': grid.st_mm[rows, columns].astype(np.float32),
        'material_grid_index': selected,
    }


def image_space_diagnostics(reference: CalibrationReference,
                            frame: CalibrationFrame):
    """Warp the fixed zero-depth reference into the current camera image."""
    positions = np.asarray(frame.gt['position_mm'], np.float64)
    current_valid = frame.image_valid & np.isfinite(positions).all(axis=-1)
    if 'base_position_mm' in frame.gt:
        base = np.asarray(frame.gt['base_position_mm'], np.float64)
    else:
        base = positions.copy()
        base[..., 2] = 0.
    uv_reference, front = project_world(base, reference.config)
    background, sampled = bilinear_sample(reference.linear_rgb, uv_reference,
                                          reference.image_valid)
    residual = frame.linear_rgb - background
    valid = current_valid & front & sampled & np.isfinite(residual).all(axis=-1)
    return np.where(valid[..., None], residual, 0.), valid


def calibration_declarations(grid: MaterialGrid, config: dict) -> dict:
    return {
        'coordinate_kind': COORDINATE_KIND,
        'coordinate_chart_id': grid.chart_id,
        'coordinate_chart_storage': 'embedded_flat_axes',
        'frame_convention': FRAME_CONVENTION,
        'residual_domain': 'material_grid',
        'color_space': 'linear_rgb',
        'image_encoding': 'linear_rgb_float',
        'surface_bending_radius_mm': bending_radius(config),
    }
