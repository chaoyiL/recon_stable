"""NumPy per-light geometry factors shared by rendering and compensation."""
from __future__ import annotations

import numpy as np

from .geometry import intersect_surface
from .light_distribution import analytic_light_samples


def _light_table(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points, normals, powers, exponents = [], [], [], []
    for strip in config['lights']:
        samples, emission_normal, sample_powers = analytic_light_samples(strip, config)
        count = len(samples)
        points.append(samples)
        normals.append(np.broadcast_to(emission_normal, (count, 3)))
        powers.append(sample_powers)
        exponents.append(np.full(count, strip['cosine_exponent'], dtype=float))
    return (np.concatenate(points), np.concatenate(normals), np.concatenate(powers),
            np.concatenate(exponents))


def _as_points(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f'{name} must have shape (N,3) and contain finite values')
    return result


def _iter_light_factors(points: np.ndarray, normals: np.ndarray, config: dict, *,
                        frozen_visibility: np.ndarray | None,
                        light_chunk_size: int | None):
    light_points, light_normals, powers, exponents = _light_table(config)
    point_count, light_count = len(points), len(light_points)
    if frozen_visibility is not None:
        visibility = np.asarray(frozen_visibility)
        if visibility.dtype.kind != 'b' or visibility.shape != (point_count, light_count):
            raise ValueError(f'frozen_visibility must be boolean with shape {(point_count, light_count)}')
    else:
        visibility = None
    if light_chunk_size is None:
        light_chunk_size = int(config.get('render', {}).get('light_chunk_size', 16))
    if type(light_chunk_size) is not int or light_chunk_size < 1:
        raise ValueError('light_chunk_size must be a positive integer')

    shadows = config.get('render', {}).get('shadows', True)
    for start in range(0, light_count, light_chunk_size):
        stop = min(start + light_chunk_size, light_count)
        vector = light_points[None, start:stop, :] - points[:, None, :]
        distance2 = np.sum(vector * vector, axis=-1)
        distance = np.sqrt(distance2)
        direction = vector / distance[..., None]
        cosine_emit = np.stack([
            np.maximum(-direction[:, index] @ light_normals[start + index], 0.0)
            for index in range(stop - start)
        ], axis=1)
        active = cosine_emit > 0.0
        normalization = (exponents[None, start:stop] + 1.0) / (2.0 * np.pi)
        source_factor = normalization * cosine_emit ** exponents[None, start:stop] / distance2

        if visibility is not None:
            visible = visibility[:, start:stop]
        elif shadows:
            shadow_origin = points[:, None, :] + normals[:, None, :] * 1e-6
            hit_distance = intersect_surface(shadow_origin, direction, config)
            visible = hit_distance >= distance - 1e-5
        else:
            visible = np.ones((point_count, stop - start), dtype=bool)

        yield (start, stop, source_factor, active, visible, direction,
               powers[start:stop], cosine_emit, distance2, exponents[start:stop])


def per_light_factors(
    points: np.ndarray,
    normals: np.ndarray,
    config: dict,
    *,
    frozen_visibility: np.ndarray | None = None,
    light_chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source factors, emitter activity, occlusion, and receiver directions.

    ``source_factor`` has shape ``(N,L)`` and units of ``mm^-2``. It excludes emitted RGB watts,
    receiver cosine/activity, and visibility. ``emitter_active`` and
    ``visible`` are boolean ``(N,L)`` arrays. ``direction`` has shape
    ``(N,L,3)`` and points from each receiver sample toward each light sample.
    Light order follows ``config['lights']`` and each strip's analytic sample
    order. A supplied visibility mask freezes only occlusion and avoids tracing
    candidate rays again.
    """
    point_array = _as_points(points, 'points')
    normal_array = _as_points(normals, 'normals')
    if normal_array.shape != point_array.shape:
        raise ValueError('normals must have the same shape as points')
    table = _light_table(config)
    light_count = len(table[0])
    source_factor = np.empty((len(point_array), light_count), dtype=float)
    emitter_active = np.empty((len(point_array), light_count), dtype=bool)
    visible = np.empty((len(point_array), light_count), dtype=bool)
    direction = np.empty((len(point_array), light_count, 3), dtype=float)
    for start, stop, factor, active, mask, vector, _, _, _, _ in _iter_light_factors(
            point_array, normal_array, config, frozen_visibility=frozen_visibility,
            light_chunk_size=light_chunk_size):
        source_factor[:, start:stop] = factor
        emitter_active[:, start:stop] = active
        visible[:, start:stop] = mask
        direction[:, start:stop] = vector
    return source_factor, emitter_active, visible, direction


def irradiance_at_points(
    points: np.ndarray,
    normals: np.ndarray,
    config: dict,
    *,
    frozen_visibility: np.ndarray | None = None,
    light_chunk_size: int | None = None,
) -> np.ndarray:
    """Integrate incoming RGB irradiance at points; units are W/mm^2 per band."""
    point_array = _as_points(points, 'points')
    normal_array = _as_points(normals, 'normals')
    if normal_array.shape != point_array.shape:
        raise ValueError('normals must have the same shape as points')
    irradiance = np.zeros((len(point_array), 3), dtype=float)
    for (start, stop, source_factor, active, visible, direction, powers,
         cosine_emit, distance2, exponents) in _iter_light_factors(
            point_array, normal_array, config, frozen_visibility=frozen_visibility,
            light_chunk_size=light_chunk_size):
        for light_index in range(stop - start):
            cosine_in = np.maximum(np.sum(normal_array * direction[:, light_index], axis=-1), 0.0)
            emission = cosine_emit[:, light_index]
            exponent = exponents[light_index]
            factor = ((exponent + 1.0) / (2.0 * np.pi) * emission ** exponent)
            factor *= ((emission > 0.0) * cosine_in * visible[:, light_index]
                       / distance2[:, light_index])
            irradiance += factor[:, None] * powers[light_index]
    return irradiance
