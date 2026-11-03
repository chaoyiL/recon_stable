"""Shared JSON configuration loading and validation."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .geometry import camera_frame, contact_radius, strip_geometry
from .light_distribution import validate_emission
from .surface_coordinates import (bending_radius, center_material_mm, normal_direction,
                                  analytic_parameter, require_flat)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    _require(result.shape == (size,) and np.all(np.isfinite(result)),
             f'{name}: expected a finite vector of length {size}')
    return result


def _positive(value, name: str, *, zero: bool = False) -> float:
    _require(not isinstance(value, bool) and np.isscalar(value) and np.isfinite(value)
             and (value >= 0 if zero else value > 0), f'{name}: invalid number')
    return float(value)


def _positive_int(value, name: str, *, minimum: int = 1) -> int:
    _require(type(value) is int and value >= minimum, f'{name}: expected integer >= {minimum}')
    return value


def validate_config(config: dict, *, allow_contact_tangent: bool = False,
                    config_path: str | Path | None = None, allow_curve: bool = False) -> dict:
    """Validate a simulation config and return the same mapping.

    ``allow_contact_tangent`` is used by the calibration scanner, whose first and
    last poses deliberately place the analytic contact circle on the rectangle edge.
    """
    _require(isinstance(config, dict), 'Configuration root must be an object')
    _require(config.get('backend', 'analytic') == 'analytic',
             'This package supports only the analytic backend')
    _require(isinstance(config.get('surface'), dict), 'Missing configuration section: surface')
    _require(isinstance(config.get('indentation'), dict), 'Missing configuration section: indentation')
    curved = bending_radius(config) is not None
    if curved and not allow_curve:
        require_flat(config, 'This entry/backend')
    try:
        surface, indentation, camera = (
            config[key] for key in ('surface', 'indentation', 'camera'))
    except KeyError as exc:
        raise ValueError(f'Missing configuration section: {exc.args[0]}') from exc
    render = config.get('render')
    _require(isinstance(render, dict), 'Missing configuration section: render')
    if allow_curve:
        allowed = {
            'surface': {'size_mm', 'view_side_z', 'reflectance_rgb', 'bending'},
            'camera': {'width_px', 'height_px', 'fx_px', 'fy_px', 'cx_px', 'cy_px',
                       'position_mm', 'look_at_mm', 'up_world', 'rotation_camera_to_world',
                       'focal_length_mm', 'response_rgb_3x3', 'distortion_opencv', 'pinhole_half_width_mm'},
            'render': {'shadows', 'exposure_scale', 'tone_map', 'device', 'geometry_dtype',
                       'radiometry_dtype', 'pixel_chunk_size', 'light_chunk_size',
                       'cuda_graphs', 'compile_lighting', 'mesh_step_mm'},
        }
        _require(not set(config) - {'backend', 'surface', 'indentation', 'camera', 'render',
                                    'lights', 'zemax', 'calibration'}, 'Unknown curve config section')
        # Historical analytic snapshots stored two geometry/export parameters in
        # a top-level ``zemax`` block. Keep that inert block readable; there is no
        # Zemax backend and its rays, wavelengths and installation settings are
        # neither validated nor used by analytic rendering.
        for section, keys in allowed.items():
            _require(not set(config[section]) - keys, f'{section}: unknown curve parameters')

    size = _vector(surface['size_mm'], 2, 'surface.size_mm')
    _require(np.all(size > 0), 'surface.size_mm must be positive')
    reflectance = _vector(surface['reflectance_rgb'], 3, 'surface.reflectance_rgb')
    _require(np.all((reflectance >= 0) & (reflectance <= 1)),
             'surface.reflectance_rgb must be in [0,1]')
    _require(surface['view_side_z'] in (-1, 1), 'surface.view_side_z must be +/-1')
    direction = normal_direction(config)
    _require(not set(indentation) - {'center_uv', 'center_xy_mm', 'radius_mm', 'depth_mm',
                                     'direction_normal', 'direction_z'},
             'indentation contains unknown parameters')
    _positive(indentation['radius_mm'], 'indentation.radius_mm')
    _positive(indentation['depth_mm'], 'indentation.depth_mm', zero=True)
    _require(indentation['depth_mm'] < indentation['radius_mm'],
             'Require 0 <= depth_mm < radius_mm (single-valued cap)')
    center = center_material_mm(config)
    radius = float(contact_radius(config))
    tolerance = 1e-10 * max(1., *size.tolist(), radius)
    if curved:
        pass  # Exact cylindrical contact bounds are checked after common fields.
    elif allow_contact_tangent:
        _require(np.all(np.abs(center) + radius <= size / 2 + tolerance),
                 'Contact disc must lie inside or tangent to the surface')
    else:
        _require(np.all(np.abs(center) + radius < size / 2),
                 'Contact disc must lie strictly inside the surface')

    for key in ('width_px', 'height_px'):
        _positive_int(camera[key], f'camera.{key}')
    for key in ('fx_px', 'fy_px', 'focal_length_mm'):
        _positive(camera[key], f'camera.{key}')
    for key in ('cx_px', 'cy_px'):
        _require(np.isfinite(camera[key]), f'camera.{key} must be finite')
    position = _vector(camera['position_mm'], 3, 'camera.position_mm')
    _require(position[2] * surface['view_side_z']
             > max(0, direction * surface['view_side_z']
                   * indentation['depth_mm']) + .1,
             'Camera must be outside the entire surface, on view_side_z')
    if 'rotation_camera_to_world' not in camera:
        _vector(camera['look_at_mm'], 3, 'camera.look_at_mm')
        _vector(camera['up_world'], 3, 'camera.up_world')
    rotation = camera_frame(camera)
    _require(rotation.shape == (3, 3)
             and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
             and np.isclose(np.linalg.det(rotation), 1),
             'camera rotation must be a proper orthonormal matrix')
    response = np.asarray(camera['response_rgb_3x3'])
    _require(response.shape == (3, 3) and np.all(np.isfinite(response)),
             'Invalid camera response matrix')
    if 'distortion_opencv' in camera:
        _require(np.all(np.asarray(camera['distortion_opencv']) == 0),
                 'Only undistorted pinhole intrinsics are supported')

    _require(isinstance(config.get('lights'), list) and len(config['lights']) > 0,
             'At least one LED strip is required')
    total_power = 0.
    for index, light in enumerate(config['lights']):
        prefix = f'lights[{index}]'
        if allow_curve:
            _require(not set(light) - {'side', 'offset_mm', 'z_mm', 'length_mm', 'width_mm',
                'cosine_exponent', 'samples_width', 'samples_length', 'power_rgb_w',
                'along_offset_mm', 'normal_world', 'emission_model', 'point_spacing_mm',
                'point_power_rgb_w'}, f'{prefix}: unknown curve parameters')
        _require(light['side'] in ('left', 'right', 'top', 'bottom'),
                 f'{prefix}.side is unknown')
        for key in ('offset_mm', 'length_mm', 'width_mm'):
            _positive(light[key], f'{prefix}.{key}', zero=key == 'offset_mm')
        _require(np.isfinite(light['z_mm'])
                 and light['z_mm'] * surface['view_side_z'] > 0,
                 f'{prefix}.z_mm must lie on the viewing side')
        total_power += validate_emission(light, backend=config.get('backend', 'analytic'),
                                         prefix=prefix)
        exponent = _positive(light['cosine_exponent'], f'{prefix}.cosine_exponent')
        _require(exponent <= 100, f'{prefix}.cosine_exponent must be <= 100')
        _positive_int(light['samples_width'], f'{prefix}.samples_width')
        if 'normal_world' in light:
            _vector(light['normal_world'], 3, f'{prefix}.normal_world')
        _require(np.isfinite(light.get('along_offset_mm', 0)),
                 f'{prefix}.along_offset_mm must be finite')
        if not curved:
            strip_geometry(light, config)
    _require(total_power > 0,
             'At least one light must have positive power')

    analytic_parameter(config, 'render', 'mesh_step_mm')
    analytic_parameter(config, 'camera', 'pinhole_half_width_mm')

    _positive(render['exposure_scale'], 'render.exposure_scale')
    _require(render['tone_map'] == 'none',
             'render.tone_map must be none for analytic linear-RGB observations; '
             'clip/reinhard configurations require a new scan')
    _require(type(render['shadows']) is bool, 'render.shadows must be boolean')
    device = render.get('device', 'cuda:0')
    _require(isinstance(device, str) and re.fullmatch(r'cpu|cuda(?::[0-9]+)?',
                                                       device.lower()) is not None,
             'render.device must be cpu or cuda[:index]')
    _require(render.get('geometry_dtype', 'float64') in ('float32', 'float64'),
             'render.geometry_dtype must be float32 or float64')
    _require(render.get('radiometry_dtype', 'float32') in ('float32', 'float64'),
             'render.radiometry_dtype must be float32 or float64')
    _positive_int(render.get('pixel_chunk_size', 16384), 'render.pixel_chunk_size')
    _positive_int(render.get('light_chunk_size', 16), 'render.light_chunk_size')
    _require(type(render.get('cuda_graphs', True)) is bool,
             'render.cuda_graphs must be boolean')
    _require(render.get('compile_lighting', 'off') in ('off', 'auto', 'required'),
             'render.compile_lighting must be off, auto or required')
    if curved:
        from .curve_geometry import validate_curve_geometry
        validate_curve_geometry(config, allow_contact_tangent=allow_contact_tangent)
    return config


def load_config(path: str | Path, *, allow_contact_tangent: bool = False,
                allow_curve: bool = False) -> dict:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    try:
        return validate_config(config, allow_contact_tangent=allow_contact_tangent,
                               config_path=path, allow_curve=allow_curve)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'{path}: {exc}') from exc
