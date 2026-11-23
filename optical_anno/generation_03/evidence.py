"""Opposing RGB responses at projected sphere boundaries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from calibrate.utils.norm_pipeline import CalibrationSettings, FrameCalibrationData
from optical_anno.generation_01.detector import ContactBackground
from optical_anno.generation_02.evidence import _DIRECTIONS, ContactEvidence, _radial_scores, _sample, contact_evidence


def _opposing_scores(difference: np.ndarray, mask: np.ndarray,
                     centers: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Measure whether the RGB transitions on opposite sides support one center."""
    samples, valid = [], []
    for fraction in (0.60, 0.82, 1.25, 1.40):
        points = centers[:, None, :] + fraction * vectors
        samples.append(_sample(difference, points))
        valid.append(_sample(mask, points) > 0.99)
    support = np.logical_and.reduce(valid)
    pairs = support & np.roll(support, len(_DIRECTIONS) // 2, axis=1)
    count = pairs.sum(axis=1)
    inner = 0.5 * (samples[0] + samples[1])
    outer = 0.5 * (samples[2] + samples[3])
    contrast = inner - outer
    opposite = 0.5 * (contrast - np.roll(contrast, len(_DIRECTIONS) // 2, axis=1))
    # For opposing surface tilts the first angular harmonic captures the
    # common direction of the RGB changes, independently in each channel.
    normalizer = 2.0 / np.maximum(count, 1)
    tilt = np.einsum('nac,ad->ncd', opposite * pairs[..., None], _DIRECTIONS) * normalizer[:, None, None]
    boundary_change = 2 * samples[1] - samples[0] - samples[2]
    change = np.einsum('nac,ad->ncd', boundary_change * pairs[..., None], _DIRECTIONS) * normalizer[:, None, None]
    explained = 0.5 * np.sum(tilt ** 2, axis=(1, 2))
    energy = np.sum(opposite ** 2 * pairs[..., None], axis=(1, 2)) / np.maximum(count, 1)
    coherence = np.minimum(explained / np.maximum(energy, 1e-10), 1.0)
    aligned = 0.5 * np.sum(tilt * change, axis=(1, 2))
    coverage = count / len(_DIRECTIONS)
    return np.where(coverage >= 0.75, aligned * coherence * coverage, -np.inf)


def _radius_limit_px(frame: FrameCalibrationData, sphere_radius_mm: float) -> float:
    """Set the scale of coarse circle hypotheses from the current projection."""
    grid = frame.systems.lightfield
    scales = []
    for axis in (0, 1):
        valid = np.moveaxis(grid.valid_mask, axis, 0)
        distance_mm = np.linalg.norm(np.diff(grid.xyz, axis=axis), axis=-1)
        distance_px = np.linalg.norm(np.diff(grid.uv, axis=axis), axis=-1)
        usable = (np.moveaxis(valid[:-1] & valid[1:], 0, axis)
                  & (distance_mm > 1e-8) & (distance_px > 0)
                  & np.isfinite(distance_px) & np.isfinite(distance_mm))
        scales.append(distance_px[usable] / distance_mm[usable])
    pixels_per_mm = np.concatenate(scales)
    if not pixels_per_mm.size:
        raise ValueError('Cannot estimate the projected ball scale')
    return float(sphere_radius_mm * np.percentile(pixels_per_mm, 95))


def sphere_evidence(frame: FrameCalibrationData, background: ContactBackground,
                    settings: CalibrationSettings) -> ContactEvidence:
    limit_px = _radius_limit_px(frame, settings.sphere_radius_mm)
    evidence = contact_evidence(frame.linear_rgb, background, score_rays=_opposing_scores,
                                maximum_seed_radius_px=limit_px, fixed_background_pose=False)
    # The opposing responses locate the center even when their extent includes
    # shading beyond contact. Radius and depth follow the sharper boundary
    # transition, with center movement limited by the localization scale.
    center_sigma_px = 0.2 * limit_px
    diagnostics = dict(evidence.diagnostics, seed_score='opposing_rgb',
                       maximum_seed_radius_px=limit_px, center_sigma_px=center_sigma_px)
    return replace(evidence, diagnostics=diagnostics, _score_rays=_radial_scores,
                   _center_sigma_px=center_sigma_px)
