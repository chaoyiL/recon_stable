"""Analyze contact depths and sphere-derived normals on a reconstructed grid."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from .geometry import SphereFit, SurfaceProjection, SurfaceProjector, fit_sphere


@dataclass(frozen=True)
class Reconstruction:
    """Reconstructed XYZ and signed baseline-normal displacement, in millimetres."""

    xyz: np.ndarray
    displacement: np.ndarray
    baseline_xyz: np.ndarray
    valid: np.ndarray
    photometric_valid: np.ndarray


@dataclass(frozen=True)
class Contact:
    """One connected depression and its partition in material-grid coordinates."""

    index: int
    peak_rc: tuple[int, int]
    max_depth_mm: float
    partition: np.ndarray


@dataclass(frozen=True)
class FitStep:
    fraction: float
    threshold_mm: float
    sample_count: int
    fit: SphereFit | None
    error: str | None
    region_mask: np.ndarray


@dataclass(frozen=True)
class ContactResult:
    contact: Contact
    steps: tuple[FitStep, ...]
    stable_index: int | None
    projection: SurfaceProjection | None
    error: str | None

    @property
    def selected_step(self) -> FitStep | None:
        return None if self.stable_index is None else self.steps[self.stable_index]


def load_reconstruction(path: str | Path) -> Reconstruction:
    """Read a local-reconstruction NPZ, preserving geometric and photometric masks."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != '.npz':
        raise ValueError('Point cloud must be a local reconstruction NPZ file')
    required = ('xyz_out', 'displacement', 'displacement_vectors', 'valid_mask',
                'photometric_valid_mask')
    with np.load(source, allow_pickle=False) as data:
        missing = sorted(set(required) - set(data.files))
        if missing:
            raise ValueError(f'{source}: missing reconstruction fields: {", ".join(missing)}')
        xyz = np.asarray(data['xyz_out'], np.float64)
        displacement = np.asarray(data['displacement'], np.float64)
        vectors = np.asarray(data['displacement_vectors'], np.float64)
        valid = np.asarray(data['valid_mask'], bool)
        photometric = np.asarray(data['photometric_valid_mask'], bool)
    if (xyz.ndim != 3 or xyz.shape[-1] != 3 or min(xyz.shape[:2]) < 4
            or vectors.shape != xyz.shape or displacement.shape != xyz.shape[:2]
            or valid.shape != xyz.shape[:2] or photometric.shape != xyz.shape[:2]):
        raise ValueError('XYZ, displacements and masks must describe the same grid (at least 4x4)')
    if not valid.any():
        raise ValueError('Reconstruction has no valid geometry')
    if (not np.isfinite(xyz[valid]).all() or not np.isfinite(vectors[valid]).all()
            or not np.isfinite(displacement[valid]).all()):
        raise ValueError('Valid reconstruction geometry and displacement must be finite')
    return Reconstruction(xyz, displacement, xyz - vectors, valid, photometric & valid)


def find_contacts(reconstruction: Reconstruction, *, count: int, threshold_mm: float,
                  min_region_points: int = 100) -> list[Contact]:
    """Separate negative-displacement depressions; coordinates are zero-based row/column.

    Significant eight-connected regions must be disjoint at ``threshold_mm``. The
    deeper scan is partitioned by nearest peak in grid coordinates, so an expanding
    region cannot absorb another peak even when shallow depressions merge.
    """
    if count < 2 or min_region_points < 4 or not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError('Require at least two contacts, a positive threshold and at least four region points')
    depth = -reconstruction.displacement
    groups, number = label(reconstruction.valid & (depth > threshold_mm), structure=np.ones((3, 3)))
    peaks = []
    for region_id in range(1, number + 1):
        rows, columns = np.nonzero(groups == region_id)
        if len(rows) < min_region_points:
            continue
        i = int(np.argmax(depth[rows, columns]))
        peaks.append((int(rows[i]), int(columns[i]), float(depth[rows[i], columns[i]])))
    if len(peaks) != count:
        details = ', '.join(f'({r},{c}):{value:.3f}mm' for r, c, value in peaks) or 'none'
        raise ValueError(f'Expected {count} separate contacts at {threshold_mm:g} mm; '
                         f'found {len(peaks)} ({details}). Adjust --threshold-mm or --count.')
    peaks.sort(key=lambda item: (item[0], item[1]))
    rr, cc = np.indices(depth.shape)
    distances = np.stack([(rr-row)**2 + (cc-column)**2 for row, column, _ in peaks])
    partitions = np.argmin(distances, axis=0)
    return [Contact(i+1, (row, column), peak, partitions == i)
            for i, (row, column, peak) in enumerate(peaks)]


def scan_contact(reconstruction: Reconstruction, contact: Contact,
                 fractions: np.ndarray) -> tuple[FitStep, ...]:
    """Grow a cap from its deepest point and fit each photometrically valid subset."""
    depth = -reconstruction.displacement
    steps = []
    for fraction in fractions:
        threshold = float(fraction * contact.max_depth_mm)
        candidate = reconstruction.valid & contact.partition & (depth > threshold)
        groups, _ = label(candidate, structure=np.ones((3, 3)))
        peak_group = groups[contact.peak_rc]
        if peak_group == 0:
            raise ValueError(f'Contact {contact.index}: deepest point was lost during segmentation')
        region = groups == peak_group
        mask = region & reconstruction.photometric_valid
        count = int(mask.sum())
        fit, error = None, None
        if count < 4:
            error = 'fewer than four photometrically valid points'
        else:
            try:
                fit = fit_sphere(reconstruction.xyz[mask])
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                error = str(exc)
        steps.append(FitStep(float(fraction), threshold, count, fit, error, region))
    return tuple(steps)


def first_stable_step(steps: tuple[FitStep, ...], *, min_points: int = 100,
                      max_condition: float = 100., diameter_tolerance: float = .02,
                      center_tolerance_mm: float = .1) -> int | None:
    """Return first locally stable three-step window; no stability is fabricated."""
    for end in range(2, len(steps)):
        window = steps[end-2:end+1]
        if any(step.fit is None or step.sample_count < min_points
               or step.fit.jacobian_condition >= max_condition for step in window):
            continue
        diameters = np.array([2*step.fit.radius_mm for step in window])
        centers = np.array([step.fit.center_xyz_mm for step in window])
        center_spread = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max()
        if np.ptp(diameters) / diameters.mean() <= diameter_tolerance and center_spread <= center_tolerance_mm:
            return end
    return None


def analyze_contacts(reconstruction: Reconstruction, contacts: list[Contact]) -> list[ContactResult]:
    """Scan every region and project stable fitted sphere centers onto the baseline."""
    try:
        projector = SurfaceProjector(reconstruction.baseline_xyz, reconstruction.valid)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        projector, projector_error = None, str(exc)
    else:
        projector_error = None
    fractions = np.round(np.arange(.95, .099, -.05), 2)
    results = []
    for contact in contacts:
        steps = scan_contact(reconstruction, contact, fractions)
        stable = first_stable_step(steps)
        if stable is None:
            results.append(ContactResult(contact, steps, None, None, 'No stable three-step sphere fit'))
            continue
        if projector is None:
            results.append(ContactResult(contact, steps, stable, None,
                                         f'Surface projection failed: {projector_error}'))
            continue
        try:
            projection = projector.project(steps[stable].fit.center_xyz_mm)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            results.append(ContactResult(contact, steps, stable, None, f'Surface projection failed: {exc}'))
        else:
            results.append(ContactResult(contact, steps, stable, projection, None))
    return results
