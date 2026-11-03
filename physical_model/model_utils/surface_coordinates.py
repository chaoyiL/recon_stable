"""Signed reference-arclength coordinates shared by all simulator inputs.

Legacy XY inputs remain readable on planes. Derived millimetres are never added
as a second editable position field to a canonical configuration.
"""
from __future__ import annotations

import copy
import math
import numpy as np

COORDINATE_MODEL = 'surface_uv_signed_v1'

# Parameters used by analytic rendering that appeared under ``zemax`` in
# historical analytic snapshots. This mapping is read-only migration support.
_LEGACY_ANALYTIC_PARAMETERS = {
    ('camera', 'pinhole_half_width_mm'),
    ('render', 'mesh_step_mm'),
}


def _pair(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (2,) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError(f'{name}: expected two finite numbers (not booleans)')
    return raw.astype(float)


def surface_size(config: dict) -> np.ndarray:
    size = _pair(config['surface']['size_mm'], 'surface.size_mm')
    if np.any(size <= 0):
        raise ValueError('surface.size_mm must be positive')
    return size


def bending_radius(config: dict) -> float | None:
    """Validate the bending schema without importing a renderer or GPU runtime."""
    bending = config['surface'].get('bending', {})
    if not isinstance(bending, dict) or set(bending) - {'axis', 'radius_mm', 'direction_z'}:
        raise ValueError('surface.bending: expected axis, radius_mm, direction_z only')
    if bending.get('axis', 'x') != 'x' or bending.get('direction_z', -1) != -1 \
            or isinstance(bending.get('direction_z', -1), bool):
        raise ValueError('surface.bending supports axis=x and direction_z=-1 only')
    radius = bending.get('radius_mm')
    if radius is not None and (type(radius) not in (int, float)
                               or not math.isfinite(radius) or radius <= 0):
        raise ValueError('surface.bending.radius_mm must be null or finite and positive')
    return None if radius is None else float(radius)


def require_flat(config: dict, operation: str) -> None:
    if bending_radius(config) is not None:
        raise ValueError(f'{operation} does not support curved surfaces; use simulate_curve.py '
                         'for analytic rendering or set surface.bending.radius_mm=null')


def uv_to_material(uv, size_mm) -> np.ndarray:
    values = _pair(uv, 'indentation.center_uv')
    size = _pair(size_mm, 'surface.size_mm')
    if np.any(np.abs(values) > 1) or np.any(size <= 0):
        raise ValueError('center_uv must be in [-1,1]^2 and size_mm must be positive')
    return values * (size / 2)


def material_to_uv(xy, size_mm) -> np.ndarray:
    values = _pair(xy, 'material position')
    size = _pair(size_mm, 'surface.size_mm')
    if np.any(size <= 0) or np.any(np.abs(values) > size / 2):
        raise ValueError('material position must be inside the positive-sized surface')
    return values / (size / 2)


def center_material_mm(config: dict) -> np.ndarray:
    """Return the reference chart's signed millimetres, not curved world XY."""
    indentation = config['indentation']
    if ('center_uv' in indentation) == ('center_xy_mm' in indentation):
        raise ValueError('indentation requires exactly one of center_uv and legacy center_xy_mm')
    size = surface_size(config)
    if 'center_uv' in indentation:
        return uv_to_material(indentation['center_uv'], size)
    require_flat(config, 'Legacy indentation.center_xy_mm')
    center = _pair(indentation['center_xy_mm'], 'indentation.center_xy_mm')
    material_to_uv(center, size)
    return center


def center_uv(config: dict) -> np.ndarray:
    return material_to_uv(center_material_mm(config), surface_size(config))


def normal_direction(config: dict) -> int:
    indentation = config['indentation']
    keys = [key for key in ('direction_normal', 'direction_z') if key in indentation]
    if len(keys) != 1:
        raise ValueError('indentation requires exactly one of direction_normal and legacy direction_z')
    value = indentation[keys[0]]
    if type(value) not in (int, float) or value not in (-1, 1):
        raise ValueError(f'indentation.{keys[0]} must be +/-1')
    if keys[0] == 'direction_z':
        require_flat(config, 'Legacy indentation.direction_z')
    return int(value)


def set_center_material_mm(config: dict, xy) -> None:
    """Update a planar pose, preserving its input schema and exact legacy values."""
    require_flat(config, 'Planar pose update')
    uv = material_to_uv(xy, surface_size(config))
    indentation = config['indentation']
    if 'center_uv' in indentation:
        indentation['center_uv'] = uv.tolist()
    else:
        indentation['center_xy_mm'] = list(xy)


def legacy_flat_config(config: dict) -> dict:
    """Adapt only at a legacy backend boundary, without changing saved input."""
    require_flat(config, 'Legacy backend')
    center, direction = center_material_mm(config), normal_direction(config)
    result = copy.deepcopy(config)
    result['surface'].pop('bending', None)
    indentation = result['indentation']
    indentation.pop('center_uv', None)
    indentation.pop('direction_normal', None)
    indentation.update(center_xy_mm=center.tolist(), direction_z=direction)
    return result


def analytic_parameter(config: dict, section: str, key: str) -> float:
    """Read an analytic parameter, including its historical snapshot location.

    Old analytic manifests placed pinhole and mesh values in a top-level
    ``zemax`` object. Read those values without rewriting the saved config so
    analytic rendering and signatures retain their original meaning.
    """
    current = config.get(section, {}).get(key)
    legacy_section = config.get('zemax')
    legacy = (legacy_section.get(key)
              if (section, key) in _LEGACY_ANALYTIC_PARAMETERS
              and isinstance(legacy_section, dict) else None)
    if current is not None and legacy is not None and current != legacy:
        raise ValueError(f'{section}.{key} conflicts with legacy zemax.{key}')
    value = current if current is not None else legacy
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{section}.{key} must be a finite positive number')
    return float(value)
