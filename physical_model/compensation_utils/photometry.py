"""Pointwise signed linear-RGB response used by compensation solvers.

Geometry inputs are reference material coordinates in millimetres. Optical
normals follow the renderer's view-side convention and light visibility may be
frozen at the undeformed reference position, as required by GN3.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any

import numpy as np

from physical_model.model_utils.geometry import camera_frame
from physical_model.model_utils.light_distribution import analytic_light_samples
from physical_model.model_utils.surface_coordinates import analytic_parameter, bending_radius


@dataclass(frozen=True)
class LightModel:
    """Light and camera parameters for the analytic linear-RGB renderer."""

    config: dict[str, Any]
    points: np.ndarray
    emission_normals: np.ndarray
    powers: np.ndarray
    exponents: np.ndarray
    normalization: np.ndarray
    camera_rgb: np.ndarray
    camera_position: np.ndarray
    camera_forward: np.ndarray
    camera_aperture: float
    camera_post_scale: float
    point_chunk_size: int


@dataclass(frozen=True)
class DepthResponseField:
    """Per-material-point reference state and its frozen shadow visibility."""

    material_coordinates: np.ndarray
    position: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    background: np.ndarray
    reference_visible: np.ndarray
    reference_active: np.ndarray
    light_model: LightModel

    def take_rows(self, indices: np.ndarray) -> 'DepthResponseField':
        """Return a point subset while retaining the shared light model."""
        index = np.asarray(indices)
        if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
            raise ValueError('indices must be a one-dimensional integer array')
        return DepthResponseField(
            self.material_coordinates[index], self.position[index], self.normal[index],
            self.basis[index], self.background[index], self.reference_visible[index],
            self.reference_active[index], self.light_model)


def _validate_points(value: Any, name: str, width: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape (N,{width})')
    return np.ascontiguousarray(result)


def _light_model(config: dict[str, Any], point_chunk_size: int | None = None) -> LightModel:
    if not isinstance(config, dict):
        raise TypeError('config must be a physical configuration dictionary')
    if config.get('backend', 'analytic') != 'analytic':
        raise ValueError('compensation requires the analytic physical model')
    if config.get('render', {}).get('tone_map') != 'none':
        raise ValueError('compensation requires render.tone_map="none"')

    point_parts, normal_parts, power_parts, exponent_parts = [], [], [], []
    for strip in config.get('lights', []):
        points, emission_normal, powers = analytic_light_samples(strip, config)
        point_parts.append(np.asarray(points, dtype=np.float64))
        normal_parts.append(np.broadcast_to(emission_normal, points.shape).astype(np.float64, copy=True))
        power_parts.append(np.asarray(powers, dtype=np.float64))
        exponent_parts.append(np.full(len(points), strip['cosine_exponent'], dtype=np.float64))
    if not point_parts:
        raise ValueError('config.lights must contain at least one light sample')

    points = np.concatenate(point_parts)
    emission_normals = np.concatenate(normal_parts)
    powers = np.concatenate(power_parts)
    exponents = np.concatenate(exponent_parts)
    camera = config['camera']
    response = np.asarray(camera['response_rgb_3x3'], dtype=np.float64)
    reflectance = np.asarray(config['surface']['reflectance_rgb'], dtype=np.float64)
    if response.shape != (3, 3) or reflectance.shape != (3,) \
            or not np.isfinite(response).all() or not np.isfinite(reflectance).all():
        raise ValueError('camera response and surface reflectance must be finite RGB arrays')
    exposure = float(config['render']['exposure_scale'])
    focal = analytic_parameter(config, 'camera', 'focal_length_mm')
    pinhole = analytic_parameter(config, 'camera', 'pinhole_half_width_mm')
    if not np.isfinite([exposure, focal, pinhole]).all() or exposure <= 0 or focal <= 0 or pinhole <= 0:
        raise ValueError('exposure_scale, focal_length_mm, and pinhole_half_width_mm must be positive')
    if point_chunk_size is None:
        point_chunk_size = max(1, min(1024, 262144 // max(len(points), 1)))
    if isinstance(point_chunk_size, bool) or not isinstance(point_chunk_size, int) \
            or point_chunk_size < 1:
        raise ValueError('point_chunk_size must be a positive integer')
    camera_position = np.asarray(camera['position_mm'], dtype=np.float64)
    camera_forward = camera_frame(camera)[:, 2].astype(np.float64, copy=False)
    return LightModel(
        config=copy.deepcopy(config), points=points, emission_normals=emission_normals,
        powers=powers, exponents=exponents, normalization=(exponents + 1.0) / (2.0 * math.pi),
        camera_rgb=(powers * reflectance[None, :]) @ response.T,
        camera_position=camera_position, camera_forward=camera_forward,
        camera_aperture=(2.0 * pinhole / focal) ** 2,
        camera_post_scale=exposure * 1e6 / math.pi,
        point_chunk_size=point_chunk_size)


def _per_light_factors(points: np.ndarray, optical_normals: np.ndarray,
                       model: LightModel, frozen_visibility: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from physical_model.model_utils.light_response_cpu import per_light_factors

    return per_light_factors(
        points, optical_normals, model.config,
        frozen_visibility=frozen_visibility,
        light_chunk_size=model.config.get('render', {}).get('light_chunk_size'))


def _camera_post(points: np.ndarray, model: LightModel) -> np.ndarray:
    ray = points - model.camera_position
    distance = np.linalg.norm(ray, axis=-1)
    direction = ray / np.maximum(distance[:, None], np.finfo(np.float64).tiny)
    cosine = np.maximum(direction @ model.camera_forward, 0.0)
    return model.camera_aperture * cosine ** 4 * model.camera_post_scale


def _response_matrix(points: np.ndarray, optical_normal: np.ndarray,
                     model: LightModel, *, frozen_visibility: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build B and return masks/directions for the active receiver branch."""
    source_factor, emitter_active, visible, direction = _per_light_factors(
        points, optical_normal, model, frozen_visibility)
    receiver_active = np.sum(direction * optical_normal[:, None, :], axis=-1) > 0.0
    active = emitter_active & visible & receiver_active
    gain = source_factor * active
    gain *= _camera_post(points, model)[:, None]
    matrix = np.einsum('nl,lc,nlj->ncj', gain, model.camera_rgb, direction,
                       optimize=True)
    return matrix, active, emitter_active & visible, direction


def camera_linear_rgb(points: np.ndarray, optical_normals: np.ndarray,
                      config: dict[str, Any], *,
                      frozen_visibility: np.ndarray | None = None,
                      point_chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate signed floating-point camera-linear RGB at material points.

    ``points`` are world positions in millimetres and ``optical_normals`` are
    unit normals facing the camera. The returned validity mask requires finite
    RGB and no tone mapping, clipping, gamma conversion, or integer quantizing
    is applied.
    """
    point_array = _validate_points(points, 'points', 3)
    normal_array = _validate_points(optical_normals, 'optical_normals', 3)
    if normal_array.shape != point_array.shape:
        raise ValueError('optical_normals must have the same shape as points')
    lengths = np.linalg.norm(normal_array, axis=-1)
    if not np.allclose(lengths, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError('optical_normals must be unit length')
    if isinstance(point_chunk_size, bool) or not isinstance(point_chunk_size, int) \
            or point_chunk_size < 1:
        raise ValueError('point_chunk_size must be a positive integer')
    model = _light_model(config, point_chunk_size)
    if frozen_visibility is not None:
        mask = np.asarray(frozen_visibility)
        if mask.dtype.kind != 'b' or mask.shape != (len(point_array), len(model.points)):
            raise ValueError('frozen_visibility must be boolean with shape (N,L)')
    else:
        mask = None

    output = np.empty((len(point_array), 3), dtype=np.float64)
    for start in range(0, len(point_array), point_chunk_size):
        stop = min(start + point_chunk_size, len(point_array))
        visible_part = None if mask is None else mask[start:stop]
        matrix, _, _, _ = _response_matrix(
            point_array[start:stop], normal_array[start:stop], model,
            frozen_visibility=visible_part)
        output[start:stop] = np.einsum('nij,nj->ni', matrix, normal_array[start:stop])
    return output, np.isfinite(output).all(axis=-1)


def reference_geometry(coordinates: np.ndarray, radius_mm: float | None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reference position, normal, and tangent frame for (s,t) mm."""
    s, t = coordinates.T
    if radius_mm is None:
        position = np.column_stack((s, t, np.zeros_like(s)))
        normal = np.zeros_like(position)
        normal[:, 2] = 1.0
        e1 = np.zeros_like(position)
        e1[:, 0] = 1.0
    else:
        from physical_model.model_utils.curve_geometry import base_surface
        position, normal = base_surface(s, t, radius_mm)
        angle = s / radius_mm
        e1 = np.column_stack((np.cos(angle), np.zeros_like(s), -np.sin(angle)))
    e2 = np.zeros_like(position)
    e2[:, 1] = 1.0
    basis = np.stack((e1, e2), axis=-1)
    return position, normal, basis


def build_response_field(material_coordinates: np.ndarray, config: dict[str, Any], *,
                         point_chunk_size: int | None = None) -> DepthResponseField:
    """Build reference geometry, background, and exact reference-face shadows.

    ``material_coordinates`` are reference arclength ``(s,t)`` in millimetres.
    The config's indentation pose is ignored for this field: shadow visibility
    and background are always evaluated at zero depth for the selected surface
    curvature. The returned ``reference_visible`` mask has shape ``(N,L)``.
    """
    coordinates = _validate_points(material_coordinates, 'material_coordinates', 2)
    model = _light_model(config, point_chunk_size)
    radius_mm = bending_radius(config)
    position, normal, basis = reference_geometry(coordinates, radius_mm)
    reference_config = copy.deepcopy(model.config)
    reference_config['indentation']['depth_mm'] = 0.0
    model = LightModel(
        config=reference_config, points=model.points, emission_normals=model.emission_normals,
        powers=model.powers, exponents=model.exponents, normalization=model.normalization,
        camera_rgb=model.camera_rgb, camera_position=model.camera_position,
        camera_forward=model.camera_forward, camera_aperture=model.camera_aperture,
        camera_post_scale=model.camera_post_scale, point_chunk_size=model.point_chunk_size)

    count = len(position)
    light_count = len(model.points)
    reference_visible = np.empty((count, light_count), dtype=bool)
    reference_active = np.empty((count, light_count), dtype=bool)
    background = np.empty((count, 3), dtype=np.float64)
    optical_reference = -normal
    for start in range(0, count, model.point_chunk_size):
        stop = min(start + model.point_chunk_size, count)
        source_factor, emitter_active, visible, direction = _per_light_factors(
            position[start:stop], optical_reference[start:stop], model)
        reference_visible[start:stop] = visible
        active = emitter_active & visible & (
            np.sum(direction * optical_reference[start:stop, None, :], axis=-1) > 0.0)
        reference_active[start:stop] = active
        gain = source_factor * active
        gain *= _camera_post(position[start:stop], model)[:, None]
        matrix = np.einsum('nl,lc,nlj->ncj', gain, model.camera_rgb, direction,
                           optimize=True)
        background[start:stop] = np.einsum(
            'nij,nj->ni', matrix, optical_reference[start:stop])
    return DepthResponseField(coordinates, position, normal, basis, background,
                              reference_visible, reference_active, model)
