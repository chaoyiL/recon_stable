"""Camera projection and unambiguous inversion of a visible parametric patch."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CameraModel:
    matrix: np.ndarray
    distortion: np.ndarray
    image_shape: tuple[int, int]  # height, width

    def __post_init__(self):
        matrix = np.array(self.matrix, np.float64, copy=True)
        distortion = np.array(self.distortion, np.float64, copy=True)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or min(matrix[0, 0], matrix[1, 1]) <= 0
                or not np.isfinite(distortion).all() or min(self.image_shape) <= 0):
            raise ValueError('Invalid camera model')
        object.__setattr__(self, 'matrix', matrix)
        object.__setattr__(self, 'distortion', distortion)
        object.__setattr__(self, 'image_shape', tuple(self.image_shape))


def project_points(xyz: np.ndarray, camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, np.float64)
    shape = xyz.shape[:-1]
    finite = np.isfinite(xyz).all(axis=-1)
    points = np.where(finite[..., None], xyz, [0, 0, 1]).reshape(-1, 3)
    uv = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera.matrix,
                           camera.distortion)[0].reshape(*shape, 2)
    h, w = camera.image_shape
    valid = (finite & np.isfinite(uv).all(axis=-1) & (xyz[..., 2] > 0)
             & (uv[..., 0] >= 0) & (uv[..., 0] < w)
             & (uv[..., 1] >= 0) & (uv[..., 1] < h))
    return uv, valid


def unproject_rays(pixels: np.ndarray, camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels, np.float64)
    finite = np.isfinite(pixels).all(axis=-1)
    safe = np.where(finite[..., None], pixels, 0)
    xy = cv2.undistortPoints(safe.reshape(-1, 1, 2), camera.matrix,
                            camera.distortion).reshape(*pixels.shape[:-1], 2)
    rays = np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    valid = finite & np.isfinite(rays).all(axis=-1)
    return np.where(valid[..., None], rays, 0), valid


def projected_circle_radius_px(
    center_px: tuple[float, float],
    radius_mm: float,
    depth_mm: float,
    camera: CameraModel,
) -> float:
    """Fit an image-space circle inside a projected camera-parallel physical disk.

    center_px uses original, distorted image pixels; depth_mm is camera Z.
    The physical disk has radius_mm and its center projects to center_px.
    Return the smallest projected radius over 128 boundary directions, including
    lens distortion. Without distortion this is min(fx, fy) * radius_mm / depth_mm.
    The disk may extend beyond the image; image clipping does not change its size.
    """
    if not np.isfinite([radius_mm, depth_mm]).all() or min(radius_mm, depth_mm) <= 0:
        raise ValueError('Physical radius and camera depth must be positive and finite')
    rays, valid = unproject_rays(np.asarray(center_px)[None], camera)
    if not valid[0] or rays[0, 2] <= 0:
        raise ValueError('Circle center must have a finite forward camera ray')
    center = rays[0] * (depth_mm / rays[0, 2])
    angles = np.arange(128) * (2 * np.pi / 128)
    boundary = center + radius_mm * np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    pixels, _ = project_points(boundary, camera)
    radii = np.linalg.norm(pixels - center_px, axis=-1)
    if not np.isfinite(radii).all() or radii.min() <= 0:
        raise ValueError('Physical disk must have a finite positive projected radius')
    return float(radii.min())


@dataclass(frozen=True)
class ProjectionInverse:
    parameters: np.ndarray
    xyz: np.ndarray
    valid: np.ndarray
    sigma_mm: np.ndarray
    reprojection_error_px: np.ndarray


def _grid_field(grid, name):
    if isinstance(grid, dict):
        return grid[name]
    if name == 'parameters' and hasattr(grid, 'coordinates'):
        return grid.coordinates
    return getattr(grid, name)


def invert_surface_projection(pixels, surface, camera: CameraModel, seed_grid, annotation_sigma_px=1.0) -> ProjectionInverse:
    if not np.isfinite(annotation_sigma_px) or annotation_sigma_px <= 0:
        raise ValueError('annotation_sigma_px must be positive')
    pixels = np.asarray(pixels, np.float64)
    shape = pixels.shape[:-1]
    targets = pixels.reshape(-1, 2)
    uv = np.asarray(_grid_field(seed_grid, 'uv'), np.float64)
    parameters = np.asarray(_grid_field(seed_grid, 'parameters'), np.float64)
    if isinstance(seed_grid, dict):
        mask = np.asarray(seed_grid.get('geometry_valid_mask', seed_grid.get('valid_mask')), bool)
    else:
        mask = np.asarray(seed_grid.valid_mask, bool)
    mask = mask & np.isfinite(uv).all(axis=-1) & np.isfinite(parameters).all(axis=-1)
    quads = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    rr, cc = np.nonzero(quads)
    triangles_uv, triangles_ab = [], []
    for offsets in (((0, 0), (0, 1), (1, 1)), ((0, 0), (1, 1), (1, 0))):
        triangles_uv.append(np.stack([uv[rr+i, cc+j] for i, j in offsets], axis=1))
        triangles_ab.append(np.stack([parameters[rr+i, cc+j] for i, j in offsets], axis=1))
    triangles_uv, triangles_ab = np.concatenate(triangles_uv), np.concatenate(triangles_ab)
    edge1 = triangles_uv[:, 1]-triangles_uv[:, 0]
    edge2 = triangles_uv[:, 2]-triangles_uv[:, 0]
    determinants = edge1[:, 0]*edge2[:, 1]-edge1[:, 1]*edge2[:, 0]
    good = np.abs(determinants) >= 1e-8
    triangles_uv, triangles_ab = triangles_uv[good], triangles_ab[good]
    edge1, edge2, determinants = edge1[good], edge2[good], determinants[good]
    lower, upper = triangles_uv.min(axis=1), triangles_uv.max(axis=1)
    bounds = np.asarray(surface.parameter_bounds)
    lo, span = bounds[[0, 2]], bounds[[1, 3]]-bounds[[0, 2]]
    count = len(targets)
    output_ab, output_xyz = np.full((count, 2), np.nan), np.full((count, 3), np.nan)
    valid, sigma, errors = np.zeros(count, bool), np.full(count, np.nan), np.full(count, np.inf)

    def evaluate(x):
        ab = lo+x*span
        xyz = np.asarray(surface.evaluate(ab[None]))[0]
        uv, _ = project_points(xyz[None], camera)
        return uv[0]

    for i, target in enumerate(targets):
        if not np.isfinite(target).all():
            continue
        ids = np.flatnonzero(((target >= lower-1e-6) & (target <= upper+1e-6)).all(1))
        if not len(ids):
            continue
        delta = target-triangles_uv[ids, 0]
        u = (delta[:, 0]*edge2[ids, 1]-delta[:, 1]*edge2[ids, 0])/determinants[ids]
        v = (edge1[ids, 0]*delta[:, 1]-edge1[ids, 1]*delta[:, 0])/determinants[ids]
        weights = np.column_stack((1-u-v, u, v))
        inside = (weights >= -1e-6).all(1)
        seeds = np.sum(triangles_ab[ids[inside]]*weights[inside, :, None], axis=1)
        solutions = []
        for seed in seeds:
            initial = np.clip((seed-lo)/span, 0, 1)
            result = least_squares(lambda x: evaluate(x)-target, initial, bounds=(0, 1),
                                   max_nfev=50, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            error = np.linalg.norm(evaluate(result.x)-target)
            if error > 1 or not np.isfinite(result.x).all():
                continue
            if not any(np.max(np.abs(result.x-old[0])) <= 1e-4 for old in solutions):
                solutions.append((result.x, error))
        if len(solutions) != 1:
            continue
        x, error = solutions[0]
        ab = lo+x*span
        da, db = surface.first_derivatives(ab[None])
        jq = np.column_stack((da[0]*span[0], db[0]*span[1]))
        jp = np.empty((2, 2), np.float64)
        for axis in (0, 1):
            left, right = x.copy(), x.copy()
            left[axis] = max(0, left[axis]-1e-5)
            right[axis] = min(1, right[axis]+1e-5)
            jp[:, axis] = (evaluate(right)-evaluate(left))/(right[axis]-left[axis])
        try:
            scale = annotation_sigma_px*np.linalg.norm(jq @ np.linalg.inv(jp), ord=2)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(scale):
            continue
        output_ab[i], output_xyz[i] = ab, surface.evaluate(ab[None])[0]
        valid[i], sigma[i], errors[i] = True, max(scale, 1e-6), error
    return ProjectionInverse(output_ab.reshape(*shape, 2), output_xyz.reshape(*shape, 3),
                             valid.reshape(shape), sigma.reshape(shape), errors.reshape(shape))
