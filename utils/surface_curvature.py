"""Local overall-surface curvature and transitions at the marker fit boundary."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from utils.surface_geometry import compute_surface_geometry


@dataclass(frozen=True)
class SurfaceCurvature:
    values: np.ndarray  # rows x columns x 3, k11/k12/k22 in the local tangent basis, mm^-1
    valid: np.ndarray
    smoothing_half_width_mm: float

    def npz_fields(self) -> dict[str, np.ndarray]:
        return dict(curvature_grid_per_mm=self.values, curvature_valid_mask=self.valid,
                    curvature_smoothing_half_width_mm=np.asarray(self.smoothing_half_width_mm))


def compute_surface_curvature(
    xyz: np.ndarray,
    valid_mask: np.ndarray,
    native_coordinates: np.ndarray,
    template_xy_mm: np.ndarray,
    *,
    smoothing_half_width_mm: float = 0.0,
) -> SurfaceCurvature:
    """Compute curvature on the regular st grid; find seams in native (a,b) coordinates.

    XYZ and the half-width are in mm. The curvature mask describes geometric
    support, independent of the image's marker, saturation and contact masks.
    """
    points = np.asarray(xyz, np.float64)
    valid = np.asarray(valid_mask, bool)
    native = np.asarray(native_coordinates, np.float64)
    template = np.asarray(template_xy_mm, np.float64)
    shape = points.shape[:2]
    if (points.ndim != 3 or points.shape[-1] != 3 or valid.shape != shape
            or native.shape != (*shape, 2)):
        raise ValueError('XYZ, geometry mask and native coordinates must share one material grid')
    if template.ndim != 2 or template.shape[1] != 2 or not len(template) or not np.isfinite(template).all():
        raise ValueError('The marker template must be a finite N×2 array')
    if not np.isfinite(smoothing_half_width_mm) or smoothing_half_width_mm < 0:
        raise ValueError('Curvature smoothing half-width must be finite and nonnegative')
    if min(shape) < 3:
        return SurfaceCurvature(np.zeros((*shape, 3), np.float32), np.zeros(shape, bool),
                                smoothing_half_width_mm)

    geometry = compute_surface_geometry(points, valid)
    tensor = geometry['curvature_tensor']
    curvature = np.stack((tensor[..., 0, 0], tensor[..., 0, 1], tensor[..., 1, 1]), axis=-1).astype(np.float32)
    support = geometry['valid']
    if smoothing_half_width_mm > 0 and support.any():
        curvature = _smooth_seams(curvature, support, points, valid, native, template, smoothing_half_width_mm)
    return SurfaceCurvature(curvature, support, smoothing_half_width_mm)


def _smooth_seams(curvature, support, xyz, valid, native, template, half_width):
    lower, upper = template.min(axis=0), template.max(axis=0)
    seam = np.zeros_like(support)
    spacings = []
    for axis in (0, 1):
        coordinates = np.moveaxis(native, axis, 0)
        geometry_valid = np.moveaxis(valid, axis, 0)
        supported = np.moveaxis(support, axis, 0)
        steps = np.linalg.norm(np.diff(xyz, axis=axis), axis=-1)
        steps = np.moveaxis(steps, axis, 0)
        measurable = (steps > 0) & np.isfinite(steps)
        local_steps = steps[measurable & supported[:-1] & supported[1:]]
        if not len(local_steps):
            local_steps = steps[measurable & geometry_valid[:-1] & geometry_valid[1:]]
        if not len(local_steps):
            return curvature
        spacings.append(float(np.median(local_steps)))

        left, right = coordinates[:-1], coordinates[1:]
        # Include both native directions because a regular st grid can shear.
        crosses = ((left < lower) != (right < lower)) | ((left > upper) != (right > upper))
        pairs = supported[:-1] & supported[1:] & crosses.any(axis=-1)
        endpoints = np.moveaxis(seam, axis, 0)
        endpoints[:-1] |= pairs
        endpoints[1:] |= pairs
    if not seam.any():
        return curvature

    distance = distance_transform_edt(~seam, sampling=spacings)
    sigma = tuple(half_width / (4 * spacing) for spacing in spacings)
    denominator = gaussian_filter(support.astype(np.float64), sigma, mode='constant')
    numerator = gaussian_filter(curvature.astype(np.float64), (*sigma, 0), mode='constant')
    band = support & (distance < half_width)
    weight = 0.5 * (1 + np.cos(np.pi * distance[band] / half_width))
    averaged = numerator[band] / denominator[band, None]
    result = curvature.copy()
    result[band] += (weight[:, None] * (averaged - curvature[band])).astype(result.dtype)
    return result
