"""Measure the rectangular 3b1 contact face on the shared reconstruction grid."""

import numpy as np

from reconstruction.local_recon import LocalReconstructionResult


# Trace the visible contact outline on the original image in (u, v) pixels.
PLATEAU_UV = ((174, 166), (452, 162), (455, 253), (174, 263))
FACE_INSET_PX = 8.
EXTERIOR_GUARD_PX = 12.


def plateau_masks(uv: np.ndarray, photo: np.ndarray, outline: np.ndarray,
                  away_from_caps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use a convex image-space outline ordered from its top-left corner clockwise."""
    edges = np.roll(outline, -1, axis=0) - outline
    relative = uv[..., None, :] - outline
    signed_distance = (edges[:, 0] * relative[..., 1] - edges[:, 1] * relative[..., 0]) / np.linalg.norm(edges, axis=1)
    nearest_edge = signed_distance.min(axis=-1)
    return photo & (nearest_edge >= FACE_INSET_PX) & away_from_caps, photo & (nearest_edge <= -EXTERIOR_GUARD_PX)


def face_metrics(result: LocalReconstructionResult, face: np.ndarray, threshold_mm: float) -> dict:
    """Measure face indentation coverage and deviation from its own best-fit plane."""
    displacement = result.displacement[face].astype(np.float64)
    xyz = result.xyz_out[face].astype(np.float64)
    centered = xyz - xyz.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    plane_residual = centered @ directions[-1]
    return dict(displacement_p10_median_p90_mm=np.percentile(displacement, [10, 50, 90]).tolist(),
                indentation_fraction=float(np.mean(displacement < -threshold_mm)),
                elevation_fraction=float(np.mean(displacement > threshold_mm)),
                best_fit_plane_rms_mm=float(np.sqrt(np.mean(plane_residual**2))))
