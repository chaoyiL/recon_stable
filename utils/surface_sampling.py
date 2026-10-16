"""Bilinear sampling and image previews with conservative grid support."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError


@dataclass(frozen=True)
class SampledField:
    values: np.ndarray
    valid: np.ndarray
    jacobian: np.ndarray | None = None


def bilinear_sample(field: np.ndarray, xy: np.ndarray, source_valid_mask=None, *, allow_boundary=False,
                    with_jacobian=False) -> SampledField:
    field = np.asarray(field)
    if field.ndim == 2:
        field = field[..., None]
    if field.ndim != 3 or min(field.shape[:2]) < 2:
        raise ValueError("field must be HxWxC, at least 2x2")
    dtype = np.result_type(field.dtype, np.float32)
    field = field.astype(dtype, copy=False)
    xy = np.asarray(xy, np.float64)
    if xy.shape[-1:] != (2,):
        raise ValueError("xy must end in two coordinates")
    h, w = field.shape[:2]
    finite = np.isfinite(xy).all(axis=-1)
    safe = np.where(finite[..., None], xy, 0)
    valid = finite & (safe[..., 0] >= 0) & (safe[..., 1] >= 0)
    if allow_boundary:
        valid &= (safe[..., 0] <= w - 1) & (safe[..., 1] <= h - 1)
    else:
        valid &= (safe[..., 0] < w - 1) & (safe[..., 1] < h - 1)
    safe = np.clip(safe, [0, 0], [w - 1, h - 1])
    lower = np.minimum(np.floor(safe).astype(np.int64), [w - 2, h - 2])
    x, y = lower[..., 0], lower[..., 1]
    fx, fy = np.moveaxis((safe - lower).astype(dtype), -1, 0)
    a, b, c, d = field[y, x], field[y, x + 1], field[y + 1, x], field[y + 1, x + 1]
    for value in (a, b, c, d):
        valid &= np.isfinite(value).all(axis=-1)
    if source_valid_mask is not None:
        source = np.asarray(source_valid_mask, bool)
        if source.shape != (h, w):
            raise ValueError("source_valid_mask shape mismatch")
        valid &= source[y, x] & source[y, x + 1] & source[y + 1, x] & source[y + 1, x + 1]
    fx, fy = fx[..., None], fy[..., None]
    values = (1-fy)*((1-fx)*a+fx*b) + fy*((1-fx)*c+fx*d)
    jac = None
    if with_jacobian:
        jac = np.stack(((1-fy)*(b-a)+fy*(d-c), (1-fx)*(c-a)+fx*(d-b)), axis=-1)
        jac = np.where(valid[..., None, None], jac, 0)
    return SampledField(np.where(valid[..., None], values, 0), valid, jac)


def sample_linear_rgb_to_surface(linear_rgb: np.ndarray, uv: np.ndarray, source_valid_mask) -> SampledField:
    if np.asarray(linear_rgb).ndim != 3 or np.asarray(linear_rgb).shape[-1] != 3:
        raise ValueError("linear_rgb must be HxWx3")
    return bilinear_sample(linear_rgb, uv, source_valid_mask)


def sample_mask_to_surface(image_mask: np.ndarray, uv: np.ndarray) -> np.ndarray:
    mask = np.asarray(image_mask, bool)
    return bilinear_sample(mask.astype(np.float32), uv, mask).valid


def rasterize_surface_grid(field: np.ndarray, uv: np.ndarray, valid_mask: np.ndarray,
                           image_shape: tuple[int, int]) -> SampledField:
    """Render a visible, nonfolded surface field in original image pixels.

    field is rows x columns [x channels]; uv is the corresponding rows x columns
    x 2 array of image (x, y) coordinates in pixels, including lens distortion.
    valid_mask marks usable grid nodes; image_shape is (height, width).
    Returns image-sized values and validity, with invalid values zeroed.
    A piecewise-linear inverse UV map supplies
    grid coordinates for conservative four-neighbour bilinear sampling. Invalid
    supports, pixels outside the projection and degenerate projections are invalid.
    This interpolation is for display, not geometric inversion or training.
    """
    field, uv, valid = np.asarray(field), np.asarray(uv, np.float64), np.asarray(valid_mask, bool)
    if field.ndim == 2:
        field = field[..., None]
    if field.ndim != 3 or min(field.shape[:2]) < 2:
        raise ValueError('field must be rows x columns x channels, at least 2x2')
    if uv.shape != (*field.shape[:2], 2) or valid.shape != field.shape[:2]:
        raise ValueError('uv and valid_mask must match the surface grid')
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError('image_shape must be a positive (height, width)')
    rows, columns = field.shape[:2]
    yy, xx = np.indices((rows, columns), dtype=float)
    grid_xy = np.stack((xx, yy), axis=-1)
    finite = np.isfinite(uv).all(axis=-1)
    image_xy = np.full((*image_shape, 2), np.nan)
    if finite.sum() >= 3:
        try:
            inverse = LinearNDInterpolator(uv[finite], grid_xy[finite])
            image_y, image_x = np.indices(image_shape)
            image_xy = inverse(image_x, image_y)
        except QhullError:
            pass  # Collinear or coincident projections have no image area.
    image_xy = np.clip(image_xy, [0, 0], [columns-1, rows-1])
    return bilinear_sample(field, image_xy, valid & finite, allow_boundary=True)
