"""Torch counterpart of curve_geometry; no tensor-to-host reads in kernels."""
from __future__ import annotations

from .curve_geometry import BOUND_EPS_MM, HIT_EPS_MM, sphere_pose


def pack_curve_pose(config: dict) -> list[float]:
    return [*sphere_pose(config)[2].tolist(), float(config['indentation']['depth_mm'] > 0)]


def _curve_projection_terms_torch(point, R, torch):
    x, y, z = point.unbind(-1)
    radial = torch.sqrt((x / R) ** 2 + (1 + z / R) ** 2)
    angle = torch.atan2(x / R, 1 + z / R)
    return x, y, z, radial, angle


def project_to_base_torch(point, R, torch):
    x, y, z, radial, angle = _curve_projection_terms_torch(point, R, torch)
    normal = torch.stack((torch.sin(angle), torch.zeros_like(x), torch.cos(angle)), -1)
    base = torch.stack((R * torch.sin(angle), y, -2 * R * torch.sin(angle / 2) ** 2), -1)
    delta = -(2 * z + (x * x + z * z) / R) / (radial + 1)
    return R * angle, y, base, normal, delta, radial


def _project_curve_coordinates_torch(point, R, torch):
    """Return only the cylinder coordinates needed for boundary tests."""
    _, y, _, radial, angle = _curve_projection_terms_torch(point, R, torch)
    return R * angle, y, radial


def _roots(a, b, c, torch):
    disc = b * b - 4 * a * c
    root = torch.sqrt(torch.clamp(disc, min=0))
    q = -.5 * (b + torch.copysign(root, b))
    safe_a = torch.where(a > 0, a, torch.ones_like(a))
    first = q / safe_a
    second = torch.where(q != 0, c / torch.where(q != 0, q, torch.ones_like(q)),
                         -b / (2 * safe_a))
    return first, second, (a > 0) & (disc >= 0)


def intersect_curve_torch(origin, direction, pose, *, width, height, radius,
                          curve_radius, torch, t_min=HIT_EPS_MM,
                          reference_only=False):
    """Intersect the finite cylinder and, unless reference-only, active sphere cap.

    ``reference_only`` is an explicit host scalar for the no-contact shadow path.
    It skips sphere-cap candidates and the cylinder's indentation-ball exclusion.
    """
    origin, direction = torch.broadcast_tensors(origin, direction)
    R = curve_radius
    if reference_only:
        center, active = None, False
    else:
        center, active = pose[:3], pose[3] > 0
    ox, oy, oz = origin.unbind(-1)
    dx, dy, dz = direction.unbind(-1)
    first, second, exists = _roots((dx * dx + dz * dz) / (2 * R),
                                   dz + (ox * dx + oz * dz) / R,
                                   oz + (ox * ox + oz * oz) / (2 * R), torch)
    distance = torch.full_like(first, float('inf'))
    part = torch.zeros_like(first, dtype=torch.int8)
    for candidate in (first, second):
        point = origin + candidate[..., None] * direction
        if reference_only:
            s, t, radial = _project_curve_coordinates_torch(point, R, torch)
            good = exists & (candidate > t_min) & (torch.abs(s) <= width / 2 + BOUND_EPS_MM)
            good &= (torch.abs(t) <= height / 2 + BOUND_EPS_MM) & (radial > 0)
        else:
            s, t, base, _, _, radial = project_to_base_torch(point, R, torch)
            in_ball = (torch.sum((base - center) ** 2, -1) < radius * radius) & active
            good = exists & (candidate > t_min) & (torch.abs(s) <= width / 2 + BOUND_EPS_MM)
            good &= (torch.abs(t) <= height / 2 + BOUND_EPS_MM) & (radial > 0) & ~in_ball
        replace = good & (candidate < distance)
        distance = torch.where(replace, candidate, distance)
        part = torch.where(replace, 1, part)
    if not reference_only:
        oc = origin - center
        first, second, exists = _roots(torch.sum(direction * direction, -1),
                                       2 * torch.sum(oc * direction, -1),
                                       torch.sum(oc * oc, -1) - radius * radius, torch)
        for candidate in (first, second):
            point = origin + candidate[..., None] * direction
            s, t, base, normal, delta, radial = project_to_base_torch(point, R, torch)
            good = exists & active & (candidate > t_min) & (torch.abs(s) <= width / 2 + BOUND_EPS_MM)
            good &= (torch.abs(t) <= height / 2 + BOUND_EPS_MM) & (radial > 0)
            good &= (delta >= -BOUND_EPS_MM) & (delta < R)
            good &= torch.sum((base - center) ** 2, -1) <= radius * radius + BOUND_EPS_MM
            good &= torch.sum((point - center) * -normal, -1) >= 0
            replace = good & (candidate < distance)
            distance = torch.where(replace, candidate, distance)
            part = torch.where(replace, 2, part)
    return distance, part


def curve_ground_truth_torch(origin, rays, forward, pose, config, torch):
    R = config['surface']['bending']['radius_mm']
    W, H = config['surface']['size_mm']
    r = config['indentation']['radius_mm']
    distance, part = intersect_curve_torch(origin, rays, pose, width=W, height=H,
                                           radius=r, curve_radius=R, torch=torch)
    point = origin + torch.where(torch.isfinite(distance), distance, 0)[..., None] * rays
    s, t, base, normal, delta, _ = project_to_base_torch(point, R, torch)
    geometric = torch.where((part == 2)[..., None], -(point - pose[:3]) / r, normal)
    optical = geometric * config['surface']['view_side_z']
    hit = torch.isfinite(distance) & (torch.sum(optical * -rays, -1) > 0)
    fields = {'position_mm': point, 'normal_world': optical,
              'normal_geometric_world': geometric,
              'surface_uv': torch.stack((2 * s / W, 2 * t / H), -1),
              'base_position_mm': base, 'base_normal_world': normal,
              'normal_displacement_mm': torch.where(part == 2, delta, 0.),
              'depth_mm': (point - origin) @ forward}
    result = {key: torch.where(hit[..., None] if value.ndim == 3 else hit, value, float('nan'))
              for key, value in fields.items()}
    result.update(mask=hit, contact_mask=hit & (part == 2), surface_part=torch.where(hit, part, 0))
    return result
