"""Shared light-response coefficient construction for renderer kernels."""
from __future__ import annotations

import torch


def emission_power_coefficient(cosine_emit: torch.Tensor, exponent: torch.Tensor,
                               normalization: torch.Tensor, *,
                               unit_exponents: bool) -> torch.Tensor:
    """Return per-pair source power before receiver, visibility, and distance terms."""
    emission_power = cosine_emit if unit_exponents else cosine_emit ** exponent[None, :]
    return normalization[None, :] * emission_power


def response_from_geometry(normal: torch.Tensor, powers: torch.Tensor,
                           exponent: torch.Tensor, normalization: torch.Tensor,
                           direction: torch.Tensor, distance2: torch.Tensor,
                           cosine_emit: torch.Tensor, visible: torch.Tensor, *,
                           unit_exponents: bool) -> tuple[torch.Tensor, ...]:
    """Build the renderer's light factor from already-computed pair geometry."""
    cosine_in = torch.clamp(torch.sum(normal[:, None, :] * direction, dim=-1), min=0)
    factor = emission_power_coefficient(cosine_emit, exponent, normalization,
                                        unit_exponents=unit_exponents)
    factor *= ((cosine_emit > 0) * cosine_in * visible
               / torch.clamp(distance2, min=torch.finfo(distance2.dtype).tiny))
    return factor.to(powers.dtype), visible, direction, distance2, cosine_emit
