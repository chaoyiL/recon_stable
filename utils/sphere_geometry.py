"""Fixed-radius sphere fitting and camera-ray cap selection, in millimetres."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


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


def fit_sphere_center(boundary_xyz, sigma_mm, radius_mm, center_frame, side) -> tuple[Sphere, float]:
    points, scales = np.asarray(boundary_xyz, np.float64), np.asarray(sigma_mm, np.float64)
    if (side not in (-1, 1) or not np.isfinite(radius_mm) or radius_mm <= 0
            or points.ndim != 2 or points.shape[1] != 3 or scales.shape != (len(points),)
            or len(points) < 3 or not np.isfinite(points).all()
            or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError('Invalid sphere fitting inputs')
    frame = np.asarray(center_frame, np.float64)
    if (frame.shape != (4, 3) or not np.isfinite(frame).all()
            or not np.allclose(frame[1:] @ frame[1:].T, np.eye(3), atol=1e-6, rtol=0)):
        raise ValueError('Sphere fitting requires a finite orthonormal center frame')
    a, e1, e2, n0 = frame
    x, y = (points-a) @ e1, (points-a) @ e2
    design = np.column_stack((x, y, np.ones(len(x))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, -(x*x+y*y), rcond=None)
    if rank < 3:
        raise ValueError('Contact boundary circle is degenerate')
    cx, cy = -coefficients[:2]/2
    contact_radius2 = cx*cx+cy*cy-coefficients[2]
    if not 0 < contact_radius2 < radius_mm**2:
        raise ValueError('Contact radius must be positive and smaller than sphere radius')
    initial = np.array([cx, cy, np.sqrt(radius_mm**2-contact_radius2)])
    lower, upper = np.array([-2*radius_mm, -2*radius_mm, 1e-6]), np.full(3, 2*radius_mm)
    if np.any(initial <= lower) or np.any(initial >= upper):
        raise ValueError('Circle initialization is outside sphere fit bounds')

    def center(v):
        return a+v[0]*e1+v[1]*e2+side*v[2]*n0

    def residual(v):
        return (np.linalg.norm(points-center(v), axis=1)-radius_mm)/scales

    result = least_squares(residual, initial, bounds=(lower, upper), loss='soft_l1',
                           f_scale=1, max_nfev=200)
    rms = float(np.sqrt(np.mean(residual(result.x)**2)))
    singular = np.linalg.svd(result.jac, compute_uv=False)
    if (not result.success or not np.isfinite(result.x).all() or rms > 2
            or singular[-1]/max(singular[0], 1e-30) < 1e-6):
        raise ValueError('Sphere fit rejected: boundary RMS or identifiability')
    return Sphere(center(result.x), radius_mm), rms


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
