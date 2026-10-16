"""Signed linear-RGB dead zones and smooth slope attenuation for NumPy/JAX."""
from __future__ import annotations

from typing import Any

import numpy as np


def resolve_zero_color_radii(
    inner_radius: float, outer_radius: float,
    negative_inner_radius: float | None = None,
    negative_outer_radius: float | None = None,
) -> tuple[float, float, float, float]:
    """Validate positive/negative magnitudes; omitted negative values inherit positive ones."""
    inner, outer = float(inner_radius), float(outer_radius)
    negative_inner = inner if negative_inner_radius is None else float(negative_inner_radius)
    negative_outer = outer if negative_outer_radius is None else float(negative_outer_radius)
    for sign, start, end in (('positive', inner, outer), ('negative', negative_inner, negative_outer)):
        if (not np.isfinite([start, end]).all() or start < 0 or end < 0
                or (end == 0 and start != 0) or (end > 0 and start >= end)):
            raise ValueError(f'zero_color_protection {sign}: require 0 <= inner_radius < outer_radius; '
                             'both zero disables that direction; radii are nonnegative magnitudes')
    return inner, outer, negative_inner, negative_outer


def signed_color_protection_factors(
    colors: Any, inner_radius: float, outer_radius: float,
    negative_inner_radius: float, negative_outer_radius: float,
    *, array_module: Any = np,
) -> tuple[Any, Any]:
    """Return slope multiplier and zero-prior mask for RGB arrays ending in 3.

    Each sign uses its largest channel magnitude and its own smoothstep. The
    larger response activates the two LUT slopes together, preserving direction.
    A disabled sign bypasses protection for its nonzero components; zero-valued
    channels do not bypass protection for the other sign. All four radii zero
    disable the whole policy, including its zero-slope confidence prior.
    Radii must already be resolved; JAX callers pass static configuration scalars.
    """
    xp = array_module
    positive = xp.max(xp.maximum(colors, 0), axis=-1)
    negative = xp.max(xp.maximum(-colors, 0), axis=-1)
    if outer_radius == 0 and negative_outer_radius == 0:
        return xp.ones_like(positive), xp.zeros_like(positive, dtype=bool)

    def response(magnitude, inner, outer):
        if outer == 0:
            return xp.where(magnitude > 0, 1., 0.)
        transition = xp.clip((magnitude-inner)/(outer-inner), 0, 1)
        return transition**2*(3-2*transition)

    attenuation = xp.maximum(response(positive, inner_radius, outer_radius),
                             response(negative, negative_inner_radius, negative_outer_radius))
    return attenuation, attenuation == 0
