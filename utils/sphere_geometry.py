"""Camera-ray sphere intersections and cap selection, in millimetres."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius_mm: float

    def __post_init__(self):
        center = np.array(self.center, np.float64, copy=True)
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(self.radius_mm) or self.radius_mm <= 0:
            raise ValueError('A sphere requires a finite center and positive radius')
        object.__setattr__(self, 'center', center)
        object.__setattr__(self, 'radius_mm', float(self.radius_mm))


def intersect_sphere_rays(ray_origins, ray_directions, sphere):
    directions = np.asarray(ray_directions, np.float64)
    origins = np.broadcast_to(np.asarray(ray_origins, np.float64), directions.shape)
    norm = np.linalg.norm(directions, axis=-1)
    unit = directions/np.maximum(norm[..., None], 1e-30)
    oc = origins-sphere.center
    b = -np.sum(unit*oc, axis=-1)
    discriminant = b*b-(np.sum(oc*oc, axis=-1)-sphere.radius_mm**2)
    valid = (np.isfinite(directions).all(axis=-1) & np.isfinite(origins).all(axis=-1)
             & (norm > 1e-12) & (discriminant >= -1e-9))
    root = np.sqrt(np.maximum(discriminant, 0))
    distances = np.stack((b-root, b+root), axis=-1)
    points = origins[..., None, :]+distances[..., :, None]*unit[..., None, :]
    valid = valid[..., None] & (distances > 0) & np.isfinite(points).all(axis=-1)
    return distances, points, valid


def sphere_normals(points: np.ndarray, sphere: Sphere) -> np.ndarray:
    return (np.asarray(points, np.float64)-sphere.center)/sphere.radius_mm


def select_sphere_cap(candidate_points, reference_point, reference_normal,
                      side, position_tolerance_mm):
    """Return -1 (minus root) or +1 (plus root); reject tied branches."""
    if side not in (-1, 1) or not np.isfinite(position_tolerance_mm) or position_tolerance_mm <= 0:
        raise ValueError('Invalid sphere cap side or position tolerance')
    points = np.asarray(candidate_points, np.float64)
    if points.shape != (2, 3):
        raise ValueError('Sphere cap needs candidates in minus/plus order')
    a, n0 = np.asarray(reference_point), np.asarray(reference_normal)
    # The caller masks invalid roots and normals facing away from the local patch.
    # Position and branch uniqueness are independent of the sphere representation.
    delta = points-a
    valid = np.isfinite(points).all(axis=-1) & (side*(delta @ n0) <= position_tolerance_mm)
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError('No sphere cap on the requested normal side')
    distance = np.linalg.norm(delta, axis=-1)
    order = ids[np.argsort(distance[ids])]
    if len(order) > 1 and distance[order[1]]-distance[order[0]] <= position_tolerance_mm:
        raise ValueError('Sphere cap branch is ambiguous')
    return -1 if order[0] == 0 else 1
