"""Closed intersections of a parametric surface with a positioned sphere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from utils.sphere_geometry import Sphere
from utils.surface_geometry import ParametricSurface
from utils.surface_projection import (
    CameraModel, build_projection_triangles, invert_surface_projection, project_points,
)


_DIRECTIONS = 128
_RADIAL_INTERVALS = 96


@dataclass(frozen=True)
class SphereContact:
    """Sphere and its ordered, implicitly closed intersection with the surface.

    Positions and radius are in camera coordinates and millimetres, pixels are
    distorted (x, y), and parameters are native (a, b). ``center_frame`` has
    rows (A, e1, e2, n), with n along X_a cross X_b. Boundary arrays contain
    128 corresponding samples; their first point is not repeated at the end.
    ``center_sigma_mm`` is the inverse projection's one-pixel uncertainty.
    """

    sphere: Sphere
    center_parameters: np.ndarray
    center_frame: np.ndarray
    center_sigma_mm: float
    boundary_parameters: np.ndarray
    boundary_xyz: np.ndarray
    boundary_px: np.ndarray


class SphereSurfaceContact:
    """Compute a star-shaped surface/sphere intersection for interactive use.

    ``seed_grid`` follows ``invert_surface_projection``'s grid contract and
    should describe the visible patch. It and the surface must stay fixed for
    this instance. A finite 128-direction, 96-interval scan rejects detected
    multiple crossings; features smaller than that sampling can be missed.
    """

    def __init__(self, surface: ParametricSurface, camera: CameraModel, seed_grid):
        bounds = np.asarray(surface.parameter_bounds, np.float64)
        if (bounds.shape != (4,) or not np.isfinite(bounds).all()
                or bounds[1] <= bounds[0] or bounds[3] <= bounds[2]):
            raise ValueError('Surface parameter_bounds must contain two finite intervals')
        self.surface = surface
        self.camera = camera
        self.seed_grid = seed_grid
        self._lower = bounds[[0, 2]]
        self._upper = bounds[[1, 3]]
        self._span = self._upper - self._lower
        if not np.isfinite(self._span).all():
            raise ValueError('Surface parameter spans must be finite')
        angles = np.arange(_DIRECTIONS) * (2 * np.pi / _DIRECTIONS)
        self._directions = np.column_stack((np.cos(angles), np.sin(angles)))
        self._prepare_visibility_grid(seed_grid)

    def compute(
        self, center_px: np.ndarray, depth_mm: float, radius_mm: float, side: int,
    ) -> SphereContact:
        """Position a sphere and return its complete visible boundary.

        ``center_px`` is an (x, y) distorted image point; depth and radius are
        in millimetres with 0 < depth <= radius. ``side`` is +1 or -1 along
        the surface normal. Failure to identify a closed visible boundary raises
        ValueError. Parameter directions are only search coordinates, not arc
        lengths or physical distances.
        """
        point = np.asarray(center_px, np.float64)
        depth = np.asarray(depth_mm, np.float64)
        radius = np.asarray(radius_mm, np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('center_px must be a finite (x, y) pixel')
        h, w = self.camera.image_shape
        if not (0 <= point[0] < w and 0 <= point[1] < h):
            raise ValueError('Contact center is outside the image')
        if (depth.shape != () or radius.shape != () or not np.isfinite(depth)
                or not np.isfinite(radius) or depth <= 0 or radius <= 0 or depth > radius):
            raise ValueError('Require finite 0 < depth_mm <= radius_mm')
        side_value = np.asarray(side)
        if (side_value.shape != () or side_value.dtype.kind not in 'iuf'
                or side_value.item() not in (-1, 1)):
            raise ValueError('side must be +1 or -1')
        depth, radius, side = float(depth), float(radius), int(side_value)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            contact_radius_squared = np.float64(depth)*(np.float64(radius)+np.float64(radius-depth))
        if not np.isfinite(contact_radius_squared) or contact_radius_squared <= 0:
            raise ValueError('depth_mm and radius_mm are outside the numerical range')

        inverse = invert_surface_projection(point[None], self.surface, self.camera, self.seed_grid,
                                            triangles=self._projection_triangles)
        if not inverse.valid[0]:
            raise ValueError('Contact center has no unique surface inverse')
        center_parameters = inverse.parameters[0]
        origin = inverse.xyz[0]
        _, origin_visible = project_points(origin[None], self.camera)
        if not origin_visible[0]:
            raise ValueError('Contact center is not visible')
        da, db, normal = self._surface_frame(center_parameters[None])
        e1 = da[0] / np.linalg.norm(da[0])
        n0 = normal[0]
        frame = np.stack((origin, e1, np.cross(n0, e1), n0))
        sphere = Sphere(origin + side * (radius-depth) * n0, radius)

        normalized_center = (center_parameters-self._lower) / self._span
        if np.any(normalized_center <= 0) or np.any(normalized_center >= 1):
            raise ValueError('Contact center is on the surface boundary')
        directions = self._directions
        with np.errstate(divide='ignore'):
            limits = np.where(directions > 0, (1-normalized_center)/directions,
                              np.where(directions < 0, -normalized_center/directions, np.inf))
        edge = np.min(limits, axis=1)
        fractions = np.linspace(0, 1, _RADIAL_INTERVALS + 1)
        positions = edge[:, None] * fractions[None, :]
        parameters = self._parameters_on_rays(normalized_center, positions)
        samples = self._evaluate(parameters.reshape(-1, 2)).reshape(_DIRECTIONS, -1, 3)
        values = self._residual(samples, origin, sphere.center, contact_radius_squared)
        inside = values < 0
        changes = inside[:, :-1] != inside[:, 1:]
        if (not inside[:, 0].all() or inside[:, -1].any()
                or not np.all(changes.sum(axis=1) == 1)
                or np.any(values[:, -1] <= 0)):
            raise ValueError('Sphere intersection is truncated or has multiple crossings')
        first = np.argmax(changes, axis=1)
        row = np.arange(_DIRECTIONS)
        left, right = positions[row, first], positions[row, first+1]
        # A fixed number of vectorized steps gives sub-micron accuracy for
        # ordinary patch dimensions without approximating the surface itself.
        for _ in range(40):
            middle = (left+right) / 2
            ab = self._parameters_on_rays(normalized_center, middle)
            value = self._residual(self._evaluate(ab), origin, sphere.center, contact_radius_squared)
            left = np.where(value < 0, middle, left)
            right = np.where(value < 0, right, middle)
        boundary_parameters = self._parameters_on_rays(normalized_center, (left+right)/2)
        boundary_xyz = self._evaluate(boundary_parameters)
        error_mm = np.abs(np.linalg.norm(boundary_xyz-sphere.center, axis=1)-radius)
        if not np.isfinite(error_mm).all() or np.any(error_mm > 1e-9*max(1, radius)):
            raise ValueError('Sphere intersection did not converge')
        boundary_px, visible = project_points(boundary_xyz, self.camera)
        if not visible.all():
            raise ValueError('Sphere intersection leaves the visible image')
        da, db, normals = self._surface_frame(boundary_parameters)
        radial_tangent = (da * (directions[:, 0]*self._span[0])[:, None]
                          + db * (directions[:, 1]*self._span[1])[:, None])
        transverse = np.einsum('ij,ij->i', boundary_xyz-sphere.center, radial_tangent)
        tangent_scale = radius * np.linalg.norm(radial_tangent, axis=1)
        facing = np.einsum('ij,ij->i', normals, boundary_xyz)
        origin_facing = np.dot(n0, origin)
        if (np.any(transverse <= 1e-10*tangent_scale)
                or abs(origin_facing) <= 1e-10*np.linalg.norm(origin)
                or np.any(facing*origin_facing <= 1e-10*np.linalg.norm(boundary_xyz, axis=1)
                          * abs(origin_facing))):
            raise ValueError('Sphere intersection or its projection is degenerate')
        self._check_image_boundary(boundary_px, boundary_parameters)
        return SphereContact(sphere, center_parameters.copy(), frame,
                             float(inverse.sigma_mm[0]), boundary_parameters,
                             boundary_xyz, boundary_px)

    def _parameters_on_rays(self, center, distance):
        direction = self._directions
        if distance.ndim == 2:
            direction = direction[:, None, :]
        normalized = center + distance[..., None]*direction
        return np.clip(self._lower + normalized*self._span, self._lower, self._upper)

    def _evaluate(self, parameters):
        points = np.asarray(self.surface.evaluate(parameters), np.float64)
        if points.shape != (len(parameters), 3) or not np.isfinite(points).all():
            raise ValueError('Surface evaluation must return finite Nx3 points')
        return points

    @staticmethod
    def _residual(points, origin, center, contact_radius_squared):
        # Subtract at A first to avoid cancellation of large camera coordinates.
        delta = points-origin
        with np.errstate(over='ignore', invalid='ignore'):
            value = (np.sum(delta*delta, axis=-1) - 2*np.sum(delta*(center-origin), axis=-1)
                     - contact_radius_squared)
        if not np.isfinite(value).all():
            raise ValueError('Sphere intersection is outside the numerical range')
        return value

    def _surface_frame(self, parameters):
        derivatives = self.surface.first_derivatives(parameters)
        if not isinstance(derivatives, (tuple, list)) or len(derivatives) != 2:
            raise ValueError('Surface first_derivatives must return (X_a, X_b)')
        da, db = (np.asarray(d, np.float64) for d in derivatives)
        if (da.shape != (len(parameters), 3) or db.shape != da.shape
                or not np.isfinite(da).all() or not np.isfinite(db).all()):
            raise ValueError('Surface derivatives must be finite Nx3 arrays')
        cross = np.cross(da, db)
        norm = np.linalg.norm(cross, axis=1)
        scale = np.linalg.norm(da, axis=1)*np.linalg.norm(db, axis=1)
        if (not np.isfinite(norm).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0) or np.any(norm <= 1e-12*scale)):
            raise ValueError('Surface normal is degenerate')
        return da, db, cross/norm[:, None]

    def _prepare_visibility_grid(self, grid):
        triangles = build_projection_triangles(grid)
        if not len(triangles.uv):
            raise ValueError('seed_grid has no valid projected triangles')
        self._projection_triangles = triangles
        centers = triangles.uv.mean(axis=1)
        self._triangle_tree = cKDTree(centers)
        radius = np.max(np.linalg.norm(triangles.uv-centers[:, None], axis=2))
        # Barycentric containment allows a small tolerance beyond triangle edges.
        self._triangle_radius = float(radius*(1+3e-6) + 1e-6)

    def _check_image_boundary(self, pixels, parameters):
        # A simple, nonzero-area polygon is required for an identifiable region.
        ends = np.roll(pixels, -1, axis=0)
        edges = ends-pixels
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(lengths <= 1e-8):
            raise ValueError('Sphere intersection projection is degenerate')

        def cross(a, b):
            return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]

        if abs(np.sum(cross(pixels, ends))) <= 1e-10*np.sum(lengths**2):
            raise ValueError('Sphere intersection projection is degenerate')
        delta = pixels[None, :, :]-pixels[:, None, :]
        denom = cross(edges[:, None, :], edges[None, :, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            t = cross(delta, edges[None, :, :])/denom
            u = cross(delta, edges[:, None, :])/denom
        indices = np.arange(len(pixels))
        nonadjacent = ((indices[:, None] != indices[None, :])
                       & ((indices[:, None]+1) % len(pixels) != indices[None, :])
                       & ((indices[None, :]+1) % len(pixels) != indices[:, None]))
        crossing = ((np.abs(denom) > 1e-12*lengths[:, None]*lengths[None, :])
                    & (t >= -1e-10) & (t <= 1+1e-10) & (u >= -1e-10) & (u <= 1+1e-10))
        parallel = np.abs(denom) <= 1e-12*lengths[:, None]*lengths[None, :]
        collinear = np.abs(cross(delta, edges[:, None, :])) <= 1e-10*np.maximum(
            1, lengths[:, None]*np.linalg.norm(delta, axis=2))
        low = np.minimum(pixels, ends)
        high = np.maximum(pixels, ends)
        overlap = ((low[:, None, :] <= high[None, :, :]+1e-10)
                   & (low[None, :, :] <= high[:, None, :]+1e-10)).all(axis=2)
        if np.any(nonadjacent & (crossing | (parallel & collinear & overlap))):
            raise ValueError('Sphere intersection projection crosses itself')

        candidates = self._triangle_tree.query_ball_point(pixels, self._triangle_radius)
        triangles = self._projection_triangles
        for pixel, parameter, ids in zip(pixels, parameters, candidates):
            ids, weights = triangles.containing(pixel, ids)
            if not len(ids):
                raise ValueError('Sphere intersection is outside the visible surface grid')
            seeds = np.sum(triangles.parameters[ids]*weights[:, :, None], axis=1)
            separation = np.max(np.abs((seeds-parameter)/self._span), axis=1)
            cell_size = np.max(np.ptp(triangles.parameters[ids]/self._span, axis=1), axis=1)
            for seed in seeds[separation > np.maximum(1e-4, 0.1*cell_size)]:
                def residual(normalized):
                    xyz = self._evaluate((self._lower+normalized*self._span)[None])
                    return project_points(xyz, self.camera)[0][0]-pixel

                initial = np.clip((seed-self._lower)/self._span, 0, 1)
                result = least_squares(residual, initial, bounds=(0, 1), max_nfev=50,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10)
                if (np.linalg.norm(residual(result.x)) <= 1
                        and np.max(np.abs(result.x-(parameter-self._lower)/self._span)) > 1e-4):
                    raise ValueError('Sphere intersection has an ambiguous surface projection')
