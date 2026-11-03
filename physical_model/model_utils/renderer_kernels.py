"""Pure light-block tensor kernel, imported only by the CUDA renderer."""
from __future__ import annotations

import torch

from .geometry_gpu import intersect_surface_parameters_torch
from .light_response import response_from_geometry


def light_block_response(point: torch.Tensor, normal: torch.Tensor,
                         shadow_origin: torch.Tensor | None, light: torch.Tensor,
                         emission: torch.Tensor, powers: torch.Tensor,
                         exponent: torch.Tensor, normalization: torch.Tensor,
                         pose_parameters: torch.Tensor, *, width: float, height: float,
                         radius: float, sign: float, shadows: bool, unit_exponents: bool,
                         reference_only: bool = False) -> tuple[torch.Tensor, ...]:
    """Return the light factor, visibility, and reusable pair geometry."""
    vector = light[None, :, :] - point[:, None, :]
    distance2 = torch.sum(vector * vector, dim=-1)
    distance = torch.sqrt(torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
    direction = vector / distance[..., None]
    cosine_emit = torch.clamp(torch.sum(-direction * emission[None, :, :], dim=-1), min=0)
    visible = torch.ones_like(cosine_emit, dtype=torch.bool)
    if shadows:
        hit_t = intersect_surface_parameters_torch(
            shadow_origin, direction, pose_parameters, width=width, height=height,
            radius=radius, sign=sign, torch=torch, include_sphere=not reference_only)
        visible = hit_t >= distance - 1e-5
    return response_from_geometry(normal, powers, exponent, normalization, direction,
                                  distance2, cosine_emit, visible,
                                  unit_exponents=unit_exponents)


def light_block_factor(point, normal, shadow_origin, light, emission, powers,
                       exponent, normalization, pose_parameters, *,
                       width, height, radius, sign, shadows, unit_exponents,
                       reference_only=False):
    """Compatibility wrapper returning only ``(factor, visible)``."""
    response = light_block_response(
        point, normal, shadow_origin, light, emission, powers, exponent,
        normalization, pose_parameters, width=width, height=height,
        radius=radius, sign=sign, shadows=shadows,
        unit_exponents=unit_exponents, reference_only=reference_only)
    return response[0], response[1]


def shade_light_block(*args, **meta):
    factor, visible = light_block_factor(*args, **meta)
    powers = args[5]
    contribution = torch.sum(factor[..., None] * powers[None, :, :], dim=1)
    return contribution, visible
