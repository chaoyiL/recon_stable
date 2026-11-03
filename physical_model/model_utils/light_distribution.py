"""Analytic strip emission; powers are absolute RGB watts, never colour codes."""
from __future__ import annotations

import math

import numpy as np

from .geometry import strip_geometry, strip_samples
from .surface_coordinates import bending_radius


DISCRETE_EMISSION_MODEL = 'discrete_rgb_points_v1'


def _point_count(strip: dict) -> int:
    length, spacing = strip['length_mm'], strip['point_spacing_mm']
    for name, value in (('length_mm', length), ('point_spacing_mm', spacing)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name}: expected a finite positive number')
    ratio = length / spacing
    if not math.isfinite(ratio):
        raise ValueError('point_spacing_mm: emitter count is not finite')
    return math.floor(ratio) + 1


def point_offsets_mm(strip: dict) -> np.ndarray:
    """Maximal centred row within the existing strip, with exact LED pitch."""
    count = _point_count(strip)
    return (np.arange(count, dtype=float) - (count - 1) / 2) * strip['point_spacing_mm']


def validate_emission(strip: dict, *, backend: str, prefix: str) -> float:
    """Validate one emission definition and return its total radiated watts."""
    model = strip.get('emission_model', 'uniform_strip_v1')
    if model == DISCRETE_EMISSION_MODEL:
        if backend != 'analytic':
            raise ValueError(f'{prefix}.emission_model: {model} requires analytic; '
                             f'{backend} does not implement per-point emission')
        for key in ('power_rgb_w', 'samples_length'):
            if key in strip:
                raise ValueError(f'{prefix}.{key}: incompatible with {model}; '
                                 'use point_power_rgb_w and point_spacing_mm')
        if 'point_spacing_mm' not in strip or 'point_power_rgb_w' not in strip:
            raise ValueError(f'{prefix}: {model} requires point_spacing_mm and point_power_rgb_w')
        try:
            # Check the supplied power-array shape before allocating positions.
            count = _point_count(strip)
        except ValueError as exc:
            raise ValueError(f'{prefix}.{exc}') from exc
        expected_shape = (count, 3)
        key = 'point_power_rgb_w'
    elif model == 'uniform_strip_v1':
        if 'point_spacing_mm' in strip or 'point_power_rgb_w' in strip:
            raise ValueError(f'{prefix}: point fields require emission_model={DISCRETE_EMISSION_MODEL}')
        count = strip.get('samples_length')
        if type(count) is not int or count < 1:
            raise ValueError(f'{prefix}.samples_length: expected integer >= 1')
        expected_shape = (3,)
        key = 'power_rgb_w'
    else:
        raise ValueError(f'{prefix}.emission_model: unknown model {model!r}')
    try:
        raw = np.asarray(strip[key])
        if raw.dtype.kind not in 'fiu':
            raise ValueError('powers must be numbers')
        power = raw.astype(float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{prefix}.{key}: expected nonnegative finite RGB watts, '
                         f'shape {expected_shape}') from exc
    if power.shape != expected_shape or not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError(f'{prefix}.{key}: expected nonnegative finite RGB watts, '
                         f'shape {expected_shape}')
    return float(power.sum())


def analytic_light_samples(strip: dict, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, fixed inward normal and RGB watts per quadrature point.

    Each discrete longitudinal emitter owns an independently editable RGB power.
    Its power is integrated uniformly across the unchanged strip width. Width
    samples are quadrature nodes, not additional independently powered LEDs.
    Legacy uniform strips retain their original positions and power division.
    """
    validate_emission(strip, backend=config.get('backend', 'analytic'), prefix='light')
    if bending_radius(config) is not None:
        from .curve_geometry import curve_light_samples
        return curve_light_samples(strip, config)
    if strip.get('emission_model') != DISCRETE_EMISSION_MODEL:
        points, normal = strip_samples(strip, config)
        powers = np.broadcast_to(np.asarray(strip['power_rgb_w'], float) / len(points),
                                 (len(points), 3)).copy()
        return points, normal, powers
    center, u, v, normal = strip_geometry(strip, config)
    along = point_offsets_mm(strip)
    nv = strip['samples_width']
    across = ((np.arange(nv) + .5) / nv - .5) * strip['width_mm']
    aa, bb = np.meshgrid(along, across)
    points = center + aa.ravel()[:, None] * u + bb.ravel()[:, None] * v
    powers = np.tile(np.asarray(strip['point_power_rgb_w'], float) / nv, (nv, 1))
    return points, normal, powers
