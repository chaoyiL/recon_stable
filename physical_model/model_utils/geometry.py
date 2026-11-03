"""Analytic spherical-cap height field. All geometry uses millimeters."""
from pathlib import Path
import struct
import numpy as np
from .surface_coordinates import (center_material_mm, normal_direction,
                                  analytic_parameter, require_flat, bending_radius)


LIGHT_GEOMETRY_MODEL = 'vertical_edge_strips_v1'


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError('Cannot normalize a zero vector')
    return v / n


def camera_frame(camera):
    """Columns: OpenCV camera x-right, y-down, z-forward in world coordinates."""
    if 'rotation_camera_to_world' in camera:
        return np.asarray(camera['rotation_camera_to_world'], float)
    forward = unit(np.asarray(camera['look_at_mm']) - camera['position_mm'])
    right = unit(np.cross(forward, camera['up_world']))
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def contact_radius(config):
    b = config['indentation']
    return np.sqrt(2 * b['radius_mm'] * b['depth_mm'] - b['depth_mm'] ** 2)


def height_normal(x, y, config):
    require_flat(config, 'height_normal(x,y)')
    s, b = config['surface'], config['indentation']
    x, y = np.broadcast_arrays(x, y)
    center = center_material_mm(config)
    dx, dy = x - center[0], y - center[1]
    rr = dx * dx + dy * dy
    inside = (rr < contact_radius(config) ** 2) & (b['depth_mm'] > 0)
    root = np.sqrt(np.maximum(b['radius_mm'] ** 2 - rr, 1e-16))
    sign = normal_direction(config)
    z = np.where(inside, sign * (b['depth_mm'] - b['radius_mm'] + root), 0.)
    hx = np.where(inside, -sign * dx / root, 0.)
    hy = np.where(inside, -sign * dy / root, 0.)
    normal = unit(np.stack((-hx, -hy, np.ones_like(hx)), -1)) * s['view_side_z']
    return z, normal


def intersect_surface(origin, direction, config, t_min=1e-7):
    """Nearest positive hit with the finite plane outside the cap, or with the cap.

    Supports arbitrary ray origins, including occlusion rays; returns t=inf on miss.
    The small sphere used to prescribe deformation is not an extra optical object.
    """
    if bending_radius(config) is not None:
        from .curve_geometry import intersect_curve
        return intersect_curve(origin, direction, config, t_min=t_min)[0]
    origin, direction = np.broadcast_arrays(np.asarray(origin, float), np.asarray(direction, float))
    s, b = config['surface'], config['indentation']
    w, h = s['size_mm']
    a2 = contact_radius(config) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        tp = -origin[..., 2] / direction[..., 2]
    tp = np.where(np.isfinite(tp), tp, -1.)
    pp = origin + tp[..., None] * direction
    xy = center_material_mm(config)
    sign = normal_direction(config)
    radial = (pp[..., 0] - xy[0]) ** 2 + (pp[..., 1] - xy[1]) ** 2
    valid = (tp > t_min) & (radial >= a2 - 1e-10)
    valid &= (np.abs(pp[..., 0]) <= w / 2) & (np.abs(pp[..., 1]) <= h / 2)
    t = np.where(valid, tp, np.inf)
    if b['depth_mm'] > 0:
        center = np.array([*xy, sign * (b['depth_mm'] - b['radius_mm'])])
        oc = origin - center
        qa = np.sum(direction * direction, -1)
        qb = np.sum(oc * direction, -1)
        qc = np.sum(oc * oc, -1) - b['radius_mm'] ** 2
        disc = qb * qb - qa * qc
        root = np.sqrt(np.maximum(disc, 0))
        for ts in ((-qb - root) / qa, (-qb + root) / qa):
            ps = origin + ts[..., None] * direction
            rr = (ps[..., 0] - center[0]) ** 2 + (ps[..., 1] - center[1]) ** 2
            good = (disc >= 0) & (ts > t_min) & (rr <= a2 + 1e-10)
            good &= sign * (ps[..., 2] - center[2]) >= 0
            good &= (np.abs(ps[..., 0]) <= w / 2) & (np.abs(ps[..., 1]) <= h / 2)
            t = np.minimum(t, np.where(good, ts, np.inf))
    return t


def strip_geometry(strip, config):
    """Fixed inward-facing vertical rectangle beside a contact-surface edge.

    offset_mm is the perpendicular, horizontal edge-to-emitting-plane gap.
    z_mm is the signed centre height; width_mm runs only along world Z.
    Moving or resizing a strip never rotates its emitting plane.
    """
    require_flat(config, 'Planar strip_geometry')
    w, h = config['surface']['size_mm']
    side, offset, z = strip['side'], strip['offset_mm'], strip['z_mm']
    along = strip.get('along_offset_mm', 0.)
    if side in ('left', 'right'):
        outward = -1 if side == 'left' else 1
        center = np.array([outward * (w / 2 + offset), along, z])
        u = np.array([0., 1., 0.])
        normal = np.array([-outward, 0., 0.])
    elif side in ('top', 'bottom'):
        outward = -1 if side == 'top' else 1
        center = np.array([along, outward * (h / 2 + offset), z])
        u = np.array([1., 0., 0.])
        normal = np.array([0., -outward, 0.])
    else:
        raise ValueError(f'Unknown light side: {side!r}')
    # Accept an explicit matching normal for compatibility, but never silently
    # tilt the source or reintroduce automatic aiming at the world origin.
    if 'normal_world' in strip:
        requested = np.asarray(strip['normal_world'], dtype=float)
        if requested.shape != (3,) or not np.isfinite(requested).all() \
                or np.linalg.norm(requested) < 1e-12 \
                or not np.allclose(unit(requested), normal, rtol=0, atol=1e-12):
            raise ValueError(f'{side} light normal_world must match the fixed inward normal '
                             f'{normal.tolist()}; tilted emitting planes are not supported')
    # The sign of v varies by side to preserve a right-handed (u, v, normal)
    # frame. Its physical axis is always world Z.
    v = np.cross(normal, u)
    return center, u, v, normal


def strip_samples(strip, config):
    c, u, v, n = strip_geometry(strip, config)
    nu, nv = strip['samples_length'], strip['samples_width']
    a = ((np.arange(nu) + .5) / nu - .5) * strip['length_mm']
    b = ((np.arange(nv) + .5) / nv - .5) * strip['width_mm']
    aa, bb = np.meshgrid(a, b)
    points = c + aa.ravel()[:, None] * u + bb.ravel()[:, None] * v
    return points, n


def export_stl(path, config):
    """Binary open-surface STL, triangle normals point towards the optical side."""
    w, h = config['surface']['size_mm']
    step = analytic_parameter(config, 'render', 'mesh_step_mm')
    nx, ny = int(np.ceil(w / step)) + 1, int(np.ceil(h / step)) + 1
    if 2 * (nx - 1) * (ny - 1) > 1_500_000:
        raise ValueError('STL mesh is too large; increase mesh_step_mm')
    x, y = np.meshgrid(np.linspace(-w / 2, w / 2, nx), np.linspace(-h / 2, h / 2, ny))
    if bending_radius(config) is not None:
        from .curve_geometry import deformed_surface
        p = deformed_surface(x, y, config)[0]
    else:
        z, _ = height_normal(x, y, config)
        p = np.stack((x, y, z), -1)
    p00, p10, p11, p01 = p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]
    tris = np.concatenate((np.stack((p00, p10, p11), -2).reshape(-1, 3, 3),
                           np.stack((p00, p11, p01), -2).reshape(-1, 3, 3)))
    if config['surface']['view_side_z'] < 0:
        tris = tris[:, [0, 2, 1]]
    normals = unit(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]))
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    rows = np.zeros(len(tris), dtype=dtype)
    rows['normal'], rows['vertices'] = normals, tris
    with Path(path).open('wb') as f:
        f.write(b'Tactile spherical-cap surface; millimeters'.ljust(80, b' '))
        f.write(struct.pack('<I', len(rows)))
        f.write(rows.tobytes())
    return len(rows)
