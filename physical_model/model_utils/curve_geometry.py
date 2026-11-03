"""Exact cylindrical reference surface, rigid-sphere contact and attached LEDs.

All distances are millimetres. Only the exposed cap replaces the reference
surface: the probe is not an additional optical object.
"""
from __future__ import annotations

import math
import numpy as np

from .surface_coordinates import bending_radius, center_material_mm, normal_direction

GEOMETRY_MODEL = 'cylinder_x_sphere_contact_v1'
LIGHT_MODEL = 'edge_attached_local_frame_v1'
CONTACT_MODEL = 'local_normal_sphere_exit_v1'
HIT_EPS_MM = 1e-7
BOUND_EPS_MM = 1e-9


def base_surface(s, t, radius: float) -> tuple[np.ndarray, np.ndarray]:
    s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
    angle = s / radius
    normal = np.stack((np.sin(angle), np.zeros_like(s), np.cos(angle)), -1)
    point = np.stack((radius * np.sin(angle), t,
                      -2 * radius * np.sin(angle / 2) ** 2), -1)
    return point, normal


def sphere_pose(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the undeformed anchor, inward direction and actual world ball centre."""
    s, t = center_material_mm(config)
    base, normal = base_surface(s, t, bending_radius(config))
    direction = -normal
    ball = config['indentation']
    return base, direction, base + (ball['depth_mm'] - ball['radius_mm']) * direction


def contact_bounds(config: dict) -> tuple[float, float]:
    """Exact half extents of the complete contact patch in reference arclength."""
    R = bending_radius(config)
    r, d = config['indentation']['radius_mm'], config['indentation']['depth_mm']
    a2 = d * (2 * r - d)
    return 2 * R * math.asin(math.sqrt(a2 / (4 * R * (R + r - d)))), math.sqrt(a2)


def validate_curve_geometry(config: dict, *, allow_contact_tangent: bool = False) -> None:
    """Analytic bounds certify the whole contact domain, not sampled edge points."""
    from .geometry import camera_frame
    R = bending_radius(config)
    W, H = config['surface']['size_mm']
    if R is None or not W / (2 * R) < math.pi / 2:
        raise ValueError('surface.bending.radius_mm must exceed surface.size_mm[0]/pi')
    if config['surface']['view_side_z'] != -1 or normal_direction(config) != -1:
        raise ValueError('Curved geometry requires view_side_z=-1 and direction_normal=-1')
    r, d = config['indentation']['radius_mm'], config['indentation']['depth_mm']
    if d >= R or -(r - d) + d * (2 * r - d) / (2 * R) >= -BOUND_EPS_MM:
        raise ValueError('indentation.depth_mm/radius_mm and bending.radius_mm produce '
                         'a folded or discontinuous contact branch')
    half_s, half_t = contact_bounds(config)
    s, t = center_material_mm(config)
    tolerance = 1e-10 * max(1., W, H, half_s, half_t) if allow_contact_tangent else 0.
    bounds = np.array([abs(s) + half_s, abs(t) + half_t])
    outside = (bounds > np.array([W, H]) / 2 + tolerance if allow_contact_tangent
               else bounds >= np.array([W, H]) / 2)
    if np.any(outside):
        raise ValueError('Curved sphere contact must lie strictly inside the reference surface')
    camera = config['camera']
    position = np.asarray(camera['position_mm'], float)
    if not np.allclose(position[:2], 0, rtol=0, atol=1e-10) or position[2] >= 0 \
            or not np.allclose(camera_frame(camera)[:, 2], [0, 0, 1], rtol=0, atol=1e-10):
        raise ValueError('Curved camera must be centred on (0,0,-distance), looking along +Z')
    if 'look_at_mm' in camera and not np.allclose(camera['look_at_mm'], 0, rtol=0, atol=1e-10):
        raise ValueError('Curved camera.look_at_mm must be [0,0,0]')
    sag = 2 * R * math.sin(W / (4 * R)) ** 2
    if -position[2] <= sag + d + .1:
        raise ValueError('camera.position_mm: distance must exceed edge sag + depth_mm + 0.1 mm')
    for index, light in enumerate(config['lights']):
        low = light['z_mm'] - light['width_mm'] / 2
        high = light['z_mm'] + light['width_mm'] / 2
        if R + low <= 0:
            raise ValueError(f'lights[{index}]: band reaches the cylinder axis (R+z<=0)')
        if light['side'] in ('top', 'bottom'):
            extreme_s = abs(light.get('along_offset_mm', 0)) \
                + light['length_mm'] / 2 * R / (R + light['z_mm'])
            if extreme_s / R >= math.pi / 2:
                raise ValueError(f'lights[{index}]: extended strip wraps beyond the supported arc')
        if light['offset_mm'] == 0 and low <= 0 <= high:
            raise ValueError(f'lights[{index}]: emitting band intersects the surface edge')
        if 'normal_world' in light:
            _, normals, _ = curve_light_samples(light, config)
            requested = np.asarray(light['normal_world'], float)
            length = np.linalg.norm(requested)
            if length == 0 or not np.allclose(normals, requested / length, rtol=0, atol=1e-12):
                raise ValueError(f'lights[{index}].normal_world conflicts with attached inward frame')


def deformed_surface(s, t, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = bending_radius(config)
    base, normal = base_surface(s, t, R)
    _, _, center = sphere_pose(config)
    r, d = config['indentation']['radius_mm'], config['indentation']['depth_mm']
    offset = center - base
    a = np.sum(offset * -normal, -1)
    b = np.sum(offset * offset, -1) - r * r
    inside = (b < 0) & (d > 0)
    root = np.sqrt(np.maximum(a * a - b, 0))
    # Rationalized positive root avoids loss of very shallow indentation.
    delta = np.where(inside, -b / np.maximum(root - a, 1e-300), 0.)
    point = base - delta[..., None] * normal
    geometric = np.where(inside[..., None], -(point - center) / r, normal)
    return point, geometric, delta


def project_to_base(point: np.ndarray, radius: float):
    """Inverse normal chart with a stable radial displacement at large R."""
    x, y, z = np.moveaxis(point, -1, 0)
    scaled_radius = np.hypot(x / radius, 1 + z / radius)
    angle = np.arctan2(x / radius, 1 + z / radius)
    s = radius * angle
    base, normal = base_surface(s, y, radius)
    delta = -(2 * z + (x * x + z * z) / radius) / np.maximum(scaled_radius + 1, 1e-300)
    return s, y, base, normal, delta, scaled_radius


def _roots(a, b, c):
    """Stable quadratic roots; a=0 (axis-parallel ray) has no cylindrical hit."""
    discriminant = b * b - 4 * a * c
    root = np.sqrt(np.maximum(discriminant, 0))
    q = -.5 * (b + np.copysign(root, b))
    valid = (a > 0) & (discriminant >= 0)
    safe_a = np.where(a > 0, a, 1.)
    first = q / safe_a
    second = np.where(q != 0, c / np.where(q != 0, q, 1.), -b / (2 * safe_a))
    return first, second, valid


def intersect_curve(origin, direction, config: dict, t_min: float = HIT_EPS_MM):
    """Nearest positive active cylinder/cap hit, with 0=miss, 1=base, 2=cap."""
    origin, direction = np.broadcast_arrays(np.asarray(origin, float), np.asarray(direction, float))
    R = bending_radius(config)
    W, H = config['surface']['size_mm']
    _, _, center = sphere_pose(config)
    r, d = config['indentation']['radius_mm'], config['indentation']['depth_mm']
    ox, oy, oz = np.moveaxis(origin, -1, 0)
    dx, dy, dz = np.moveaxis(direction, -1, 0)
    a = (dx * dx + dz * dz) / (2 * R)
    b = dz + (ox * dx + oz * dz) / R
    c = oz + (ox * ox + oz * oz) / (2 * R)
    first, second, exists = _roots(a, b, c)
    distance = np.full_like(first, np.inf)
    part = np.zeros(first.shape, np.int8)
    for candidate in (first, second):
        p = origin + candidate[..., None] * direction
        s, t, base, _, _, radial = project_to_base(p, R)
        in_ball = (np.sum((base - center) ** 2, -1) < r * r) & (d > 0)
        valid = exists & (candidate > t_min) & (np.abs(s) <= W / 2 + BOUND_EPS_MM)
        valid &= (np.abs(t) <= H / 2 + BOUND_EPS_MM) & (radial > 0) & ~in_ball
        replace = valid & (candidate < distance)
        distance = np.where(replace, candidate, distance)
        part = np.where(replace, 1, part)
    if d > 0:
        oc = origin - center
        first, second, exists = _roots(np.sum(direction * direction, -1),
                                       2 * np.sum(oc * direction, -1),
                                       np.sum(oc * oc, -1) - r * r)
        for candidate in (first, second):
            p = origin + candidate[..., None] * direction
            s, t, base, normal, delta, radial = project_to_base(p, R)
            valid = exists & (candidate > t_min) & (np.abs(s) <= W / 2 + BOUND_EPS_MM)
            valid &= (np.abs(t) <= H / 2 + BOUND_EPS_MM) & (radial > 0)
            valid &= (delta >= -BOUND_EPS_MM) & (delta < R)
            valid &= np.sum((base - center) ** 2, -1) <= r * r + BOUND_EPS_MM
            valid &= np.sum((p - center) * -normal, -1) >= 0
            replace = valid & (candidate < distance)
            distance = np.where(replace, candidate, distance)
            part = np.where(replace, 2, part)
    return distance, part


def camera_ground_truth_curve(config: dict) -> dict:
    from .renderer_cpu import camera_rays
    from .geometry import camera_frame
    rays = camera_rays(config)
    origin = np.asarray(config['camera']['position_mm'], float)
    distance, part = intersect_curve(origin, rays, config)
    point = origin + np.where(np.isfinite(distance), distance, 0)[..., None] * rays
    s, t, base, normal, delta, _ = project_to_base(point, bending_radius(config))
    _, _, center = sphere_pose(config)
    geometric = np.where((part == 2)[..., None],
                         -(point - center) / config['indentation']['radius_mm'], normal)
    optical = geometric * config['surface']['view_side_z']
    hit = np.isfinite(distance) & (np.sum(optical * -rays, -1) > 0)
    W, H = config['surface']['size_mm']
    fields = {'position_mm': point, 'normal_world': optical,
              'normal_geometric_world': geometric,
              'surface_uv': np.stack((2 * s / W, 2 * t / H), -1),
              'base_position_mm': base, 'base_normal_world': normal,
              'normal_displacement_mm': np.where(part == 2, delta, 0.),
              'depth_mm': (point - origin) @ camera_frame(config['camera'])[:, 2]}
    result = {key: np.where(hit[..., None] if value.ndim == 3 else hit, value, np.nan)
              for key, value in fields.items()}
    result.update(mask=hit, contact_mask=hit & (part == 2), surface_part=np.where(hit, part, 0))
    return result


def curve_light_samples(strip: dict, config: dict):
    from .light_distribution import point_offsets_mm, DISCRETE_EMISSION_MODEL
    R = bending_radius(config)
    W, H = config['surface']['size_mm']
    discrete = strip.get('emission_model') == DISCRETE_EMISSION_MODEL
    along = point_offsets_mm(strip) if discrete else \
        ((np.arange(strip['samples_length']) + .5) / strip['samples_length'] - .5) * strip['length_mm']
    nv = strip['samples_width']
    across = ((np.arange(nv) + .5) / nv - .5) * strip['width_mm']
    ell, width = np.meshgrid(along, across)
    side = strip['side']
    outward = -1 if side in ('left', 'top') else 1
    # Keep the historical side-specific width ordering, including discrete LEDs.
    width_sign = -outward if side in ('left', 'right') else outward
    z = strip['z_mm'] + width_sign * width.ravel()
    if side in ('left', 'right'):
        s = np.full(ell.size, outward * W / 2)
        t = strip.get('along_offset_mm', 0.) + ell.ravel()
        base, normal = base_surface(s, t, R)
        tangent = np.stack((normal[:, 2], np.zeros_like(s), -normal[:, 0]), -1)
        outer = outward * tangent
        weights = np.ones(ell.size)
    else:
        s = strip.get('along_offset_mm', 0.) + ell.ravel() * R / (R + strip['z_mm'])
        t = np.full_like(s, outward * H / 2)
        base, normal = base_surface(s, t, R)
        outer = np.broadcast_to([0., float(outward), 0.], base.shape)
        weights = (R + z) / (R + strip['z_mm'])
    points = base + strip['offset_mm'] * outer + z[:, None] * normal
    if discrete:
        powers = np.tile(np.asarray(strip['point_power_rgb_w'], float) / nv, (nv, 1))
    else:
        powers = weights[:, None] / weights.sum() * np.asarray(strip['power_rgb_w'], float)
    return points, -outer, powers
