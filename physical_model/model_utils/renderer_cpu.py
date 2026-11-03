"""NumPy analytic renderer retained as the numerical CPU reference."""
from __future__ import annotations

import numpy as np

from .geometry import camera_frame, height_normal, intersect_surface, unit
from .light_response_cpu import irradiance_at_points
from .surface_coordinates import bending_radius, analytic_parameter


def camera_rays(config):
    camera = config['camera']
    x, y = np.meshgrid(np.arange(camera['width_px']),
                       np.arange(camera['height_px']))
    local = np.stack(((x - camera['cx_px']) / camera['fx_px'],
                      (y - camera['cy_px']) / camera['fy_px'],
                      np.ones_like(x)), -1)
    return unit(local @ camera_frame(camera).T)


def camera_ground_truth(config):
    if bending_radius(config) is not None:
        from .curve_geometry import camera_ground_truth_curve
        return camera_ground_truth_curve(config)
    camera = config['camera']
    rays = camera_rays(config)
    origin = np.asarray(camera['position_mm'])
    distance = intersect_surface(origin, rays, config)
    hit = np.isfinite(distance)
    positions = origin + np.where(hit, distance, 0)[..., None] * rays
    _, normals = height_normal(positions[..., 0], positions[..., 1], config)
    hit &= np.sum(normals * -rays, -1) > 0
    forward = camera_frame(camera)[:, 2]
    depth = np.where(hit, (positions - origin) @ forward, np.nan)
    return {
        'mask': hit,
        'position_mm': np.where(hit[..., None], positions, np.nan),
        'normal_world': np.where(hit[..., None], normals, np.nan),
        'depth_mm': depth,
    }


def render_cpu(config):
    gt = camera_ground_truth(config)
    mask = gt['mask']
    points, normals = gt['position_mm'][mask], gt['normal_world'][mask]
    irradiance = irradiance_at_points(points, normals, config, light_chunk_size=1)
    radiance = (irradiance * np.asarray(config['surface']['reflectance_rgb'])
                / np.pi)
    radiance_image = np.zeros((*mask.shape, 3))
    radiance_image[mask] = radiance * 1e6
    camera = config['camera']
    rays = camera_rays(config)
    cosine_fourth = (rays @ camera_frame(camera)[:, 2]) ** 4
    area = (2 * analytic_parameter(config, 'camera', 'pinhole_half_width_mm')) ** 2
    sensor = (radiance_image * (area / camera['focal_length_mm'] ** 2)
              * cosine_fourth[..., None])
    return sensor, radiance_image, gt
