"""Curve-aware lighting factor compiled and captured independently of the plane."""
from __future__ import annotations

import torch
from .curve_geometry_gpu import intersect_curve_torch
from .light_response import response_from_geometry


def curve_light_block_response(point: torch.Tensor, normal: torch.Tensor,
                               shadow_origin: torch.Tensor | None, light: torch.Tensor,
                               emission: torch.Tensor, powers: torch.Tensor,
                               exponent: torch.Tensor, normalization: torch.Tensor,
                               pose_parameters: torch.Tensor, *, width: float, height: float,
                               radius: float, curve_radius: float, shadows: bool,
                               unit_exponents: bool, reference_only: bool = False
                               ) -> tuple[torch.Tensor, ...]:
    """Return the curved light factor, visibility, and reusable pair geometry."""
    vector = light[None, :, :] - point[:, None, :]
    distance2 = torch.sum(vector * vector, -1)
    distance = torch.sqrt(torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
    direction = vector / distance[..., None]
    cosine_emit = torch.clamp(torch.sum(-direction * emission[None, :, :], -1), min=0)
    visible = torch.ones_like(cosine_emit, dtype=torch.bool)
    if shadows:
        hit, _ = intersect_curve_torch(shadow_origin, direction, pose_parameters,
                                       width=width, height=height, radius=radius,
                                       curve_radius=curve_radius, torch=torch,
                                       reference_only=reference_only)
        visible = hit >= distance - 1e-5
    return response_from_geometry(normal, powers, exponent, normalization, direction,
                                  distance2, cosine_emit, visible,
                                  unit_exponents=unit_exponents)


def curve_light_block_factor(point, normal, shadow_origin, light, emission, powers,
                             exponent, normalization, pose_parameters, *,
                             width, height, radius, curve_radius, shadows,
                             unit_exponents, reference_only=False):
    """Compatibility wrapper returning only ``(factor, visible)``."""
    response = curve_light_block_response(
        point, normal, shadow_origin, light, emission, powers, exponent,
        normalization, pose_parameters, width=width, height=height,
        radius=radius, curve_radius=curve_radius, shadows=shadows,
        unit_exponents=unit_exponents, reference_only=reference_only)
    return response[0], response[1]


def shade_curve_light_block(*args, **meta):
    factor, visible = curve_light_block_factor(*args, **meta)
    return torch.sum(factor[..., None] * args[5][None, :, :], dim=1), visible
