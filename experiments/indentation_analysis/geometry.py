"""Sphere fitting, sampled-surface projection, and spatial line analysis.

XYZ positions and distances share one Cartesian frame and use millimetres.
Grid coordinates are floating-point (row, column) indices, not physical lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from typing import Sequence

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SphereFit:
    center_xyz_mm: np.ndarray
    radius_mm: float
    radial_rms_mm: float
    jacobian_condition: float


@dataclass(frozen=True)
class SurfaceProjection:
    xyz_mm: np.ndarray
    grid_rc: np.ndarray
    unit_direction: np.ndarray
    distance_mm: float


@dataclass(frozen=True)
class PairAngles:
    indices: tuple[int, int]
    angle_3d_degrees: float
    projected_angle_degrees: float
    extended_line_distance_mm: float | None


@dataclass(frozen=True)
class LineAnalysis:
    endpoint_plane_point: np.ndarray
    endpoint_plane_normal: np.ndarray
    endpoint_plane_rms_mm: float
    endpoint_plane_max_mm: float
    line_angles_to_endpoint_plane_degrees: np.ndarray
    direction_plane_normal: np.ndarray
    direction_angles_to_plane_degrees: np.ndarray
    pairs: list[PairAngles]


def sphere_indentation_depth(center_xyz_mm: np.ndarray, radius_mm: float,
                             projection_xyz_mm: np.ndarray,
                             positive_normal: np.ndarray) -> tuple[float, float]:
    """Return signed center height and sphere indentation depth in millimetres.

    The XYZ inputs share a Cartesian frame. ``projection_xyz_mm`` must be the
    nearest normal projection of the fitted sphere center onto the baseline
    surface. ``positive_normal`` establishes the positive side and may differ
    from the surface normal at the projection. The signed height is the full
    3D center-to-projection distance; depth is ``radius_mm - height`` and may
    have either sign. A nonzero displacement with an alignment within 1e-12 of
    orthogonal has an ambiguous side and raises ValueError.
    """
    if not isinstance(radius_mm, Real) or isinstance(radius_mm, (bool, np.bool_)):
        raise ValueError('radius_mm must be a finite positive real number')
    try:
        radius = float(radius_mm)
    except (OverflowError, ValueError) as error:
        raise ValueError('radius_mm must be a finite positive real number') from error
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius_mm must be a finite positive real number')

    vectors = []
    for name, value in (('center_xyz_mm', center_xyz_mm),
                        ('projection_xyz_mm', projection_xyz_mm),
                        ('positive_normal', positive_normal)):
        try:
            if np.iscomplexobj(value):
                raise ValueError(f'{name} must be a finite real XYZ vector')
            vector = np.asarray(value, np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'{name} must be a finite XYZ vector') from error
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f'{name} must be a finite XYZ vector')
        vectors.append(vector)
    center, projection, normal = vectors

    normal_scale = np.max(np.abs(normal))
    if normal_scale == 0:
        raise ValueError('positive_normal must be nonzero')
    normal = normal / normal_scale
    normal /= np.linalg.norm(normal)

    with np.errstate(over='ignore', invalid='ignore'):
        displacement = center - projection
    if not np.isfinite(displacement).all():
        raise ValueError('center-to-projection distance must be finite')
    scale = np.max(np.abs(displacement))
    if scale == 0:
        return 0., radius

    direction = displacement / scale
    direction_length = np.linalg.norm(direction)
    alignment = float((direction / direction_length) @ normal)
    if abs(alignment) <= 1e-12:
        raise ValueError('positive_normal gives an ambiguous center side')
    with np.errstate(over='ignore'):
        distance = float(scale * direction_length)
    height = distance if alignment > 0 else -distance
    depth = radius - height
    if not np.isfinite(height) or not np.isfinite(depth):
        raise ValueError('center height and indentation depth must be finite')
    return height, depth


def fit_sphere(points: np.ndarray, radius_mm: float, *,
               side_normal: np.ndarray | None = None) -> SphereFit:
    """Fit a known-radius sphere by ordinary least squares on radial distances.

    ``points`` is finite N×3 XYZ in millimetres, N >= 4; ``radius_mm`` is a
    finite, strictly positive physical radius, held fixed during fitting. If
    supplied, ``side_normal`` selects centers on its positive side of the point
    centroid. Otherwise both sides are searched and an indistinguishable
    mirrored solution is rejected. Finite multi-start search is not a proof of
    a global optimum for arbitrary non-spherical point clouds. The condition
    number belongs to the three-center-parameter Jacobian, not an uncertainty.
    """
    xyz = np.asarray(points, np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 4 or not np.isfinite(xyz).all():
        raise ValueError('Sphere fit requires at least four finite Nx3 points')
    if (not isinstance(radius_mm, Real) or isinstance(radius_mm, (bool, np.bool_))
            or not np.isfinite(radius_mm) or radius_mm <= 0):
        raise ValueError('radius_mm must be a finite positive real number')
    radius = float(radius_mm)
    origin = xyz.mean(axis=0)
    delta = xyz - origin
    _, spread, axes = np.linalg.svd(delta, full_matrices=False)
    if spread[0] <= 0 or spread[1] <= 1e-10 * spread[0]:
        raise ValueError('Sphere fit is degenerate: points are collinear')
    normal = axes[-1]
    tangential = delta - (delta @ normal)[:, None] * normal
    height = np.sqrt(max(radius ** 2 - np.mean(np.sum(tangential ** 2, axis=1)),
                         (.1 * radius) ** 2))
    seeds = [height * normal, -height * normal]

    # The free algebraic center is only an initializer, never an output or a
    # reason to choose a side; include its reflection for shallow noisy caps.
    scale = spread[0] / np.sqrt(len(delta))
    normalized = delta / scale
    design = np.column_stack((2 * normalized, np.ones(len(delta))))
    solution, _, rank, _ = np.linalg.lstsq(design, np.sum(normalized ** 2, axis=1), rcond=None)
    if rank == 4 and np.isfinite(solution).all():
        initial = solution[:3] * scale
        seeds.extend((initial, initial - 2 * np.dot(initial, normal) * normal))

    basis = np.eye(3)
    bounds = (-np.inf, np.inf)
    if side_normal is not None:
        positive = np.asarray(side_normal, np.float64)
        if positive.shape != (3,) or not np.isfinite(positive).all() or np.linalg.norm(positive) <= 1e-12:
            raise ValueError('side_normal must be a finite nonzero XYZ direction')
        positive = positive / np.linalg.norm(positive)
        axis = np.eye(3)[np.argmin(np.abs(positive))]
        tangent = np.cross(positive, axis)
        tangent /= np.linalg.norm(tangent)
        basis = np.column_stack((tangent, np.cross(positive, tangent), positive))
        bounds = ([-np.inf, -np.inf, 0.], np.inf)

    def residual(value):
        return np.linalg.norm(delta - basis @ value, axis=1) - radius

    def jacobian(value):
        difference = basis @ value - delta
        length = np.linalg.norm(difference, axis=1)
        return (difference / np.maximum(length[:, None], 1e-15)) @ basis

    candidates = []
    for seed in seeds:
        initial = basis.T @ seed
        if side_normal is not None:
            initial[2] = max(abs(initial[2]), 1e-8 * max(1., radius))
        try:
            # Parameter/gradient stopping resolves candidate centers to the
            # same precision even when the residual remains large.
            result = least_squares(residual, initial, jac=jacobian, bounds=bounds, x_scale='jac',
                                   ftol=None, xtol=1e-11, gtol=1e-11, max_nfev=400)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        if result.success and np.isfinite(result.x).all() and np.isfinite(result.fun).all():
            center = basis @ result.x
            rms = float(np.sqrt(np.mean(result.fun ** 2)))
            candidates.append((rms, center, result))
    if not candidates:
        raise ValueError('Sphere fit did not converge to a finite solution')
    candidates.sort(key=lambda candidate: candidate[0])
    rms, center, result = candidates[0]
    for other_rms, other_center, _ in candidates[1:]:
        if np.linalg.norm(other_center - center) > 1e-6 * max(1., radius):
            if other_rms - rms <= max(1e-9 * max(1., radius), .01 * other_rms):
                raise ValueError('Sphere center is ambiguous between distinct solutions')
            break
    if side_normal is not None and result.x[2] <= 1e-8 * max(1., radius):
        raise ValueError('Sphere center is ambiguous at the requested side boundary')
    singular = np.linalg.svd(result.jac, compute_uv=False)
    if singular[-1] <= 1e-12 * singular[0]:
        raise ValueError('Sphere fit is not identifiable')
    return SphereFit(origin + center, radius, rms, float(singular[0] / singular[-1]))


class SurfaceProjector:
    """Find a nearest point on a sampled surface using its grid interpolant.

    The grid must have at least 4×4 XYZ points. Complete grids use one cubic
    spline per coordinate; grids with holes use 4×4 all-valid local splines,
    so invalid values never enter interpolation. A projection constrained by
    an invalid-support boundary is rejected. The search uses finitely many
    nearest seeds; a folded surface may have unexamined distant minima, so this
    is not a proof of global uniqueness for arbitrary surfaces.
    """

    def __init__(self, baseline_xyz: np.ndarray, valid_mask: np.ndarray):
        points = np.asarray(baseline_xyz, np.float64)
        valid = np.asarray(valid_mask, bool)
        if (points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 4
                or valid.shape != points.shape[:2]):
            raise ValueError('Surface requires matching XYZ and mask grids of at least 4x4')
        if np.any(valid & ~np.isfinite(points).all(axis=-1)):
            raise ValueError('Valid surface points must be finite')
        self.points = points.copy()
        self.valid = valid.copy()
        self.rows, self.columns = points.shape[:2]
        self.complete = bool(valid.all())
        self._safe_cells = None
        self._cell_indices = None
        self._splines = None
        if self.complete:
            self._splines = self._make_splines(0, 0, self.rows, self.columns)
            self._tree = cKDTree(self.points.reshape(-1, 3))
        else:
            # Each cell uses the four surrounding rows/columns, shifted at
            # physical boundaries; every point in that support must be valid.
            count = np.pad(valid.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            sums = count[4:, 4:] - count[:-4, 4:] - count[4:, :-4] + count[:-4, :-4]
            row_start = np.clip(np.arange(self.rows - 1) - 1, 0, self.rows - 4)
            col_start = np.clip(np.arange(self.columns - 1) - 1, 0, self.columns - 4)
            self._safe_cells = sums[row_start[:, None], col_start[None, :]] == 16
            rr, cc = np.nonzero(self._safe_cells)
            if not len(rr):
                raise ValueError('Surface has no all-valid cubic interpolation support')
            self._cell_indices = np.column_stack((rr, cc))
            centers = (self.points[rr, cc] + self.points[rr + 1, cc]
                       + self.points[rr, cc + 1] + self.points[rr + 1, cc + 1]) / 4
            self._tree = cKDTree(centers)

    def _make_splines(self, row: int, column: int, rows: int, columns: int):
        r = np.arange(row, row + rows)
        c = np.arange(column, column + columns)
        return tuple(RectBivariateSpline(r, c, self.points[row:row + rows, column:column + columns, axis],
                                        kx=3, ky=3, s=0) for axis in range(3))

    @staticmethod
    def _evaluate(splines, rc):
        xyz = np.array([s.ev(rc[0], rc[1]).item() for s in splines])
        jacobian = np.array([[s.ev(rc[0], rc[1], dx=1).item(),
                              s.ev(rc[0], rc[1], dy=1).item()] for s in splines])
        return xyz, jacobian

    def _candidates(self, center):
        count = min(8 if self.complete else 32, self._tree.n)
        indices = np.atleast_1d(self._tree.query(center, k=count)[1])
        for index in indices:
            if self.complete:
                initial = np.array(np.unravel_index(index, (self.rows, self.columns)), float)
                bounds = ([0., 0.], [self.rows - 1., self.columns - 1.])
                yield self._splines, initial, bounds
            else:
                row, column = self._cell_indices[index]
                start_row = int(np.clip(row - 1, 0, self.rows - 4))
                start_col = int(np.clip(column - 1, 0, self.columns - 4))
                splines = self._make_splines(start_row, start_col, 4, 4)
                yield splines, np.array([row + .5, column + .5]), ([row, column], [row + 1, column + 1])

    def _has_valid_neighborhood(self, rc):
        if self.complete:
            return True
        # At an interior edge of the supported domain the closest point may
        # actually lie in the missing region, so do not report it as known.
        for step in ((1e-6, 0), (-1e-6, 0), (0, 1e-6), (0, -1e-6)):
            candidate = rc + step
            if (candidate < 0).any() or candidate[0] > self.rows - 1 or candidate[1] > self.columns - 1:
                continue
            row = min(int(candidate[0]), self.rows - 2)
            col = min(int(candidate[1]), self.columns - 2)
            if not self._safe_cells[row, col]:
                return False
        return True

    def project(self, center_xyz_mm: np.ndarray) -> SurfaceProjection:
        """Project a finite XYZ center to the supported sampled surface.

        The returned unit vector points from the projected point to the center;
        a center coincident with the surface has no defined direction.
        """
        center = np.asarray(center_xyz_mm, np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Projection center must be a finite XYZ vector')
        candidates = []
        for splines, initial, bounds in self._candidates(center):
            result = least_squares(
                lambda rc: self._evaluate(splines, rc)[0] - center, initial,
                jac=lambda rc: self._evaluate(splines, rc)[1], bounds=bounds,
                ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=100,
            )
            if not result.success or not np.isfinite(result.x).all():
                continue
            xyz, jac = self._evaluate(splines, result.x)
            cross = np.linalg.norm(np.cross(jac[:, 0], jac[:, 1]))
            scale = np.linalg.norm(jac[:, 0]) * np.linalg.norm(jac[:, 1])
            if (not np.isfinite(xyz).all() or not np.isfinite(jac).all()
                    or scale <= 0 or cross <= 1e-12 * scale):
                continue
            distance = float(np.linalg.norm(center - xyz))
            normality_error = np.linalg.norm(jac.T @ (center - xyz)) / max(
                np.linalg.norm(jac) * distance, 1e-30)
            candidates.append((distance, result.x.copy(), xyz, normality_error))
        if not candidates:
            raise ValueError('Surface projection has no identifiable solution')
        candidates.sort(key=lambda item: item[0])
        distance, rc, xyz, normality_error = candidates[0]
        if not self._has_valid_neighborhood(rc):
            raise ValueError('Nearest surface projection touches invalid interpolation support')
        if distance <= 1e-10:
            raise ValueError('Center lies on the surface; direction is undefined')
        if normality_error > 1e-6:
            raise ValueError('Nearest surface point is boundary-constrained and not a normal projection')
        for other_distance, _, other_xyz, _ in candidates[1:]:
            if other_distance - distance > 1e-7 * max(1., distance):
                break
            if np.linalg.norm(other_xyz - xyz) > 1e-5 * max(1., distance):
                raise ValueError('Surface projection has multiple equally near solutions')
        return SurfaceProjection(xyz, rc, (center - xyz) / distance, distance)


def _plane_normal(points: np.ndarray, description: str) -> np.ndarray:
    _, singular, vectors = np.linalg.svd(points, full_matrices=False)
    if len(singular) < 2 or singular[0] <= 0 or singular[1] <= 1e-12 * singular[0]:
        raise ValueError(f'{description} plane is not uniquely identifiable')
    normal = vectors[-1]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    return normal


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(first, second)),
                                      np.clip(first @ second, -1., 1.))))


def analyze_lines(projections: Sequence[SurfaceProjection], centers: np.ndarray) -> LineAnalysis:
    """Analyze Q→center directions and the corresponding finite line segments.

    Endpoint-plane residuals concern the 2N finite endpoints. The separate
    direction plane passes through the origin. Pair angles are directed 0–180°;
    projected angles use the endpoint plane. Extended-line distances concern
    infinite lines, not the finite segments, and are None for near-parallel
    directions. At least three nondegenerate, identifiable lines are required.
    """
    centers = np.asarray(centers, np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 3:
        raise ValueError('Line analysis requires at least three Nx3 centers')
    if len(projections) != len(centers) or not np.isfinite(centers).all():
        raise ValueError('Projection and center arrays must correspond and be finite')
    anchors = np.asarray([p.xyz_mm for p in projections], np.float64)
    declared_directions = np.asarray([p.unit_direction for p in projections], np.float64)
    declared_distances = np.asarray([p.distance_mm for p in projections], np.float64)
    if (anchors.shape != centers.shape or declared_directions.shape != centers.shape
            or not np.isfinite(anchors).all() or not np.isfinite(declared_directions).all()
            or not np.isfinite(declared_distances).all()):
        raise ValueError('Projections require finite XYZ, direction, and distance')
    vectors = centers - anchors
    distances = np.linalg.norm(vectors, axis=1)
    if np.any(distances <= 1e-10):
        raise ValueError('Coincident endpoints do not define a line direction')
    directions = vectors / distances[:, None]
    if (not np.allclose(declared_directions, directions, rtol=1e-6, atol=1e-7)
            or not np.allclose(declared_distances, distances, rtol=1e-6, atol=1e-7)):
        raise ValueError('Projection direction and distance must match its center')

    endpoints = np.vstack((anchors, centers))
    point = endpoints.mean(axis=0)
    normal = _plane_normal(endpoints - point, 'Endpoint')
    signed = (endpoints - point) @ normal
    line_angles = np.degrees(np.arcsin(np.clip(np.abs(directions @ normal), 0, 1)))
    projected = directions - (directions @ normal)[:, None] * normal
    projected_lengths = np.linalg.norm(projected, axis=1)
    if np.any(projected_lengths <= 1e-10):
        raise ValueError('A direction perpendicular to the endpoint plane has no projected angle')
    projected /= projected_lengths[:, None]
    direction_normal = _plane_normal(directions, 'Direction')
    direction_angles = np.degrees(np.arcsin(np.clip(np.abs(directions @ direction_normal), 0, 1)))
    pairs = []
    for i, j in combinations(range(len(centers)), 2):
        cross = np.cross(directions[i], directions[j])
        cross_length = np.linalg.norm(cross)
        distance = (None if cross_length <= 1e-8 else
                    float(abs((anchors[j] - anchors[i]) @ cross) / cross_length))
        pairs.append(PairAngles((i, j), _angle(directions[i], directions[j]),
                                _angle(projected[i], projected[j]), distance))
    return LineAnalysis(point, normal, float(np.sqrt(np.mean(signed ** 2))),
                        float(np.max(np.abs(signed))), line_angles,
                        direction_normal, direction_angles, pairs)
