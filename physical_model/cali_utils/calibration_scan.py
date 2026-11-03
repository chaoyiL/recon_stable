"""Deterministic planar and cylindrical material-position scan generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from physical_model.model_utils.curve_geometry import contact_bounds
from physical_model.model_utils.geometry import contact_radius
from physical_model.model_utils.surface_coordinates import (
    bending_radius,
    material_to_uv,
    surface_size,
)


@dataclass(frozen=True)
class ScanPose:
    pose_id: int
    image_key: str
    depth_index: int
    row_index: int
    column_index: int
    center_xy_mm: tuple[float, float]
    depth_mm: float
    contact_radius_mm: float
    center_uv: tuple[float, float]
    contact_half_extent_mm: tuple[float, float]

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ('center_xy_mm', 'center_uv', 'contact_half_extent_mm'):
            value[key] = list(value[key])
        return value


def closed_axis(start: float, stop: float, step: float, *, include_endpoint: bool = True) -> list[float]:
    """Return stable closed interval samples, optionally appending the far endpoint."""
    values = (float(start), float(stop), float(step))
    if not all(math.isfinite(value) for value in values) or step <= 0 or stop < start:
        raise ValueError('Axis bounds must be finite, ordered, and use a positive step')
    tolerance = 1e-10 * max(1., abs(start), abs(stop), abs(step))
    if abs(stop - start) <= tolerance:
        return [float(stop)]
    count = math.floor((stop - start) / step + 1e-10)
    result = [float(start + index * step) for index in range(count + 1)]
    result = [value for value in result if value <= stop + tolerance]
    if result and abs(result[-1] - stop) <= tolerance:
        result[-1] = float(stop)
    elif include_endpoint:
        result.append(float(stop))
    return result


def build_scan_poses(config: dict) -> list[ScanPose]:
    """Scan centers over the domain where the whole sphere-contact patch fits."""
    calibration = config['calibration']
    scan = calibration['scan']
    radius = float(config['indentation']['radius_mm'])
    width, height = map(float, surface_size(config))
    step_x, step_y = map(float, scan['xy_step_mm'])
    include = scan.get('include_endpoints', True)
    if type(include) is not bool:
        raise ValueError('calibration.scan.include_endpoints must be boolean')

    depths = closed_axis(float(scan['depth_min_mm']), float(scan['depth_max_mm']),
                         float(scan['depth_step_mm']), include_endpoint=include)
    if not 0 < radius or any(not 0 < depth < radius for depth in depths):
        raise ValueError('Every calibration depth must satisfy 0 < depth < sphere radius')
    if any(not math.isfinite(value) or value <= 0 for value in (width, height, step_x, step_y)):
        raise ValueError('Surface size and calibration XY steps must be positive and finite')

    curved = bending_radius(config) is not None
    temporary = []
    for depth_index, depth in enumerate(depths):
        contact_r = math.sqrt(2 * radius * depth - depth * depth)
        if curved:
            center_half_s, center_half_t = contact_bounds({
                **config,
                'indentation': {**config['indentation'], 'depth_mm': depth},
            })
        else:
            center_half_s = center_half_t = contact_r
        if 2 * center_half_s > width + 1e-10 * max(width, 1.) \
                or 2 * center_half_t > height + 1e-10 * max(height, 1.):
            raise ValueError(f'Contact patch at depth {depth:g} mm does not fit on the surface')
        xs = closed_axis(-width / 2 + center_half_s,
                         width / 2 - center_half_s, step_x,
                         include_endpoint=include)
        ys = closed_axis(-height / 2 + center_half_t,
                         height / 2 - center_half_t, step_y,
                         include_endpoint=include)
        for row_index, y in enumerate(ys):
            for column_index, x in enumerate(xs):
                uv = material_to_uv((x, y), (width, height))
                temporary.append((depth_index, row_index, column_index, x, y, depth,
                                  contact_r, center_half_s, center_half_t, uv))

    if not temporary:
        raise ValueError('Calibration scan generated no poses')
    key_width = max(6, len(str(len(temporary) - 1)))
    return [
        ScanPose(
            pose_id=index,
            image_key=f'{index:0{key_width}d}',
            depth_index=depth_index,
            row_index=row_index,
            column_index=column_index,
            center_xy_mm=(float(x), float(y)),
            depth_mm=float(depth),
            contact_radius_mm=float(contact_r),
            center_uv=tuple(map(float, uv)),
            contact_half_extent_mm=(float(half_s), float(half_t)),
        )
        for index, (depth_index, row_index, column_index, x, y, depth, contact_r,
                    half_s, half_t, uv) in enumerate(temporary)
    ]
