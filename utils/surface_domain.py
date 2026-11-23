"""Extend a marker surface to a reference image's visible contact domain."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from utils.surface_geometry import OffsetSurface, ParametricSurface
from utils.surface_projection import CameraModel, project_points
from utils.surface_sampling import sample_mask_to_surface


DOMAIN_KIND = 'black_frame_tangent_v1'


def _bounds(value) -> np.ndarray:
    bounds = np.asarray(value, np.float64)
    if (bounds.shape != (4,) or not np.isfinite(bounds).all()
            or np.any(bounds[[1, 3]] <= bounds[[0, 2]])):
        raise ValueError('Surface parameter bounds must be two finite intervals')
    return bounds


class _TangentExtendedSurface:
    """Continue the boundary tangents in native parameters without stretching.

    At a corner the mixed derivative supplies the bilinear cross term. The
    original surface and its first derivatives are unchanged inside its domain.
    """

    def __init__(self, surface: ParametricSurface, parameter_bounds):
        original = _bounds(surface.parameter_bounds)
        extended = _bounds(parameter_bounds)
        if (np.any(extended[[0, 2]] > original[[0, 2]])
                or np.any(extended[[1, 3]] < original[[1, 3]])):
            raise ValueError('Extended bounds must contain the marker surface bounds')
        self.surface = surface
        self.parameter_bounds = tuple(extended)
        self._original = original
        self.integration_breakpoints = tuple(
            tuple(x for x in original[axis:axis+2]
                  if extended[axis] < x < extended[axis+1])
            for axis in (0, 2)
        )

    def _prepare(self, parameters):
        values = np.asarray(parameters, np.float64)
        if values.shape[-1:] != (2,) or not np.isfinite(values).all():
            raise ValueError('Surface parameters must be finite and end in two coordinates')
        shape = values.shape[:-1]
        points = values.reshape(-1, 2)
        lower = np.asarray(self.parameter_bounds)[[0, 2]]
        upper = np.asarray(self.parameter_bounds)[[1, 3]]
        if np.any(points < lower) or np.any(points > upper):
            raise ValueError('Parameters lie outside extended surface bounds')
        clipped = np.clip(points, self._original[[0, 2]], self._original[[1, 3]])
        return shape, clipped, points - clipped

    def _mixed_derivative(self, parameters):
        second = getattr(self.surface, 'second_derivatives', None)
        if callable(second):
            try:
                derivatives = second(parameters)
            except NotImplementedError:
                derivatives = None
            if derivatives is not None:
                return np.asarray(derivatives[1], np.float64)
        # Generic surfaces may only provide first derivatives. Difference them
        # inside the original bounds, with one-sided steps at the boundary.
        step = np.cbrt(np.finfo(np.float64).eps) * (self._original[3] - self._original[2])
        left, right = parameters.copy(), parameters.copy()
        left[:, 1] = np.maximum(left[:, 1] - step, self._original[2])
        right[:, 1] = np.minimum(right[:, 1] + step, self._original[3])
        da_left = np.asarray(self.surface.first_derivatives(left)[0], np.float64)
        da_right = np.asarray(self.surface.first_derivatives(right)[0], np.float64)
        return (da_right - da_left) / (right[:, 1] - left[:, 1])[:, None]

    def evaluate(self, parameters):
        shape, clipped, delta = self._prepare(parameters)
        points = np.asarray(self.surface.evaluate(clipped), np.float64).copy()
        outside = np.any(delta != 0, axis=1)
        if outside.any():
            da, db = self.surface.first_derivatives(clipped[outside])
            d = delta[outside]
            points[outside] += d[:, :1] * da + d[:, 1:] * db
            corners = (d[:, 0] != 0) & (d[:, 1] != 0)
            if corners.any():
                indices = np.flatnonzero(outside)[corners]
                mixed = self._mixed_derivative(clipped[indices])
                points[indices] += (delta[indices, 0] * delta[indices, 1])[:, None] * mixed
        return points.reshape(*shape, 3)

    def first_derivatives(self, parameters):
        shape, clipped, delta = self._prepare(parameters)
        da, db = (np.array(x, np.float64, copy=True)
                  for x in self.surface.first_derivatives(clipped))
        outside = np.any(delta != 0, axis=1)
        if outside.any():
            mixed = self._mixed_derivative(clipped[outside])
            da[outside] += delta[outside, 1, None] * mixed
            db[outside] += delta[outside, 0, None] * mixed
        return da.reshape(*shape, 3), db.reshape(*shape, 3)


def extend_surface(surface: ParametricSurface, parameter_bounds) -> ParametricSurface:
    """Return a tangent continuation over bounds containing the marker domain."""
    return _TangentExtendedSurface(surface, parameter_bounds)


def _project(surface, parameters, camera):
    points = np.asarray(surface.evaluate(parameters), np.float64)
    uv, _ = project_points(points, camera)
    return uv, points


def _projection_jacobian(surface, parameters, camera, lower, upper):
    columns = []
    for axis in (0, 1):
        step = 1e-5 * (upper[axis] - lower[axis])
        left, right = parameters.copy(), parameters.copy()
        left[:, axis] = np.maximum(left[:, axis] - step, lower[axis])
        right[:, axis] = np.minimum(right[:, axis] + step, upper[axis])
        uv_left, _ = _project(surface, left, camera)
        uv_right, _ = _project(surface, right, camera)
        columns.append((uv_right - uv_left) / (right[:, axis] - left[:, axis])[:, None])
    return np.stack(columns, axis=-1)


def _invert_boundary(surface, pixels, camera, lower, upper):
    # One reference-frame search, seeded by a projected parameter grid. Every
    # contour pixel is subsequently refined against the distorted camera model.
    aa, bb = np.meshgrid(np.linspace(lower[0], upper[0], 65),
                         np.linspace(lower[1], upper[1], 129))
    grid = np.stack((aa, bb), axis=-1).reshape(-1, 2)
    uv, xyz = _project(surface, grid, camera)
    usable = np.isfinite(uv).all(axis=1) & (xyz[:, 2] > 0)
    if not usable.any():
        raise ValueError('Extended reference surface has no visible projection')
    points = grid[usable][cKDTree(uv[usable]).query(pixels)[1]]
    for _ in range(20):
        projected, xyz = _project(surface, points, camera)
        residual = projected - pixels
        error = np.linalg.norm(residual, axis=1)
        active = (error > 1e-4) & (xyz[:, 2] > 0)
        if not active.any():
            break
        jac = _projection_jacobian(surface, points, camera, lower, upper)
        determinant = np.linalg.det(jac)
        active &= np.isfinite(jac).all(axis=(1, 2)) & (np.abs(determinant) > 1e-8)
        if not active.any():
            break
        step = np.zeros_like(points)
        step[active] = np.linalg.solve(jac[active], residual[active, :, None])[..., 0]
        step = np.clip(step, -.15 * (upper - lower), .15 * (upper - lower))
        updated = points.copy()
        accepted = ~active
        for backtrack in range(8):
            candidate = np.clip(points - step * (.5 ** backtrack), lower, upper)
            trial, _ = _project(surface, candidate, camera)
            improve = (np.linalg.norm(trial - pixels, axis=1) < error) & ~accepted
            updated[improve] = candidate[improve]
            accepted |= improve
            if accepted.all():
                break
        points = updated
    projected, xyz = _project(surface, points, camera)
    error = np.linalg.norm(projected - pixels, axis=1)
    if np.any(~np.isfinite(error) | (error > .05) | (xyz[:, 2] <= 0)):
        raise ValueError('Reference surface cannot cover the complete black-frame boundary')
    return points


def estimate_reference_bounds(
    marker_surface: ParametricSurface,
    offset_mm: float,
    camera: CameraModel,
    reference_surface_mask: np.ndarray,
) -> tuple[float, float, float, float]:
    """Find frozen native bounds covering the reference black-frame mask.

    Marker-space tangents are extended before the signed normal offset. The
    search is limited to half a marker-domain span per side; the rectangular
    result has a small margin and must still be clipped by each image mask.
    """
    mask = np.asarray(reference_surface_mask, bool)
    if mask.shape != camera.image_shape or not mask.any():
        raise ValueError('Reference surface mask must be a nonempty camera-sized array')
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    pixels = np.concatenate(contours).reshape(-1, 2).astype(np.float64)
    original = _bounds(marker_surface.parameter_bounds)
    lower, upper = original[[0, 2]], original[[1, 3]]
    span = upper - lower
    search_lower, search_upper = lower - .5 * span, upper + .5 * span
    search_bounds = (search_lower[0], search_upper[0], search_lower[1], search_upper[1])
    surface = OffsetSurface(extend_surface(marker_surface, search_bounds), offset_mm)
    points = _invert_boundary(surface, pixels, camera, search_lower, search_upper)
    # Check the solved boundary against the orientation at the marker center.
    center = ((lower + upper) / 2)[None]
    reference_det = np.linalg.det(_projection_jacobian(surface, center, camera,
                                                       search_lower, search_upper))[0]
    determinant = np.linalg.det(_projection_jacobian(surface, points, camera,
                                                       search_lower, search_upper))
    if (not np.isfinite(reference_det) or abs(reference_det) <= 1e-8
            or np.any(~np.isfinite(determinant))
            or np.any(determinant * np.sign(reference_det) <= 1e-8)):
        raise ValueError('Reference black-frame boundary has a folded projection')
    # Allow modest changes in the current frame while keeping one frozen domain.
    margin = .05 * span
    result_lower = np.minimum(points.min(axis=0) - margin, lower)
    result_upper = np.maximum(points.max(axis=0) + margin, upper)
    if np.any(result_lower < search_lower) or np.any(result_upper > search_upper):
        raise ValueError('Reference black frame exceeds the supported extrapolation range')
    aa, bb = np.meshgrid(np.linspace(result_lower[0], result_upper[0], 33),
                         np.linspace(result_lower[1], result_upper[1], 65))
    grid = np.stack((aa, bb), axis=-1).reshape(-1, 2)
    uv, xyz = _project(surface, grid, camera)
    inside = (xyz[:, 2] > 0) & sample_mask_to_surface(mask, uv)
    if not inside.any():
        raise ValueError('Reference black-frame interior has no valid projected samples')
    interior_det = np.linalg.det(_projection_jacobian(surface, grid[inside], camera,
                                                       search_lower, search_upper))
    if (np.any(~np.isfinite(interior_det))
            or np.any(interior_det * np.sign(reference_det) <= 1e-8)):
        raise ValueError('Reference black-frame interior has a folded projection')
    return (float(result_lower[0]), float(result_upper[0]),
            float(result_lower[1]), float(result_upper[1]))
