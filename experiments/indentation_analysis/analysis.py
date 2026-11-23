"""Fit contact regions and analyze sphere-derived normals on a reconstructed grid."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from .depth_fitting import DepthSample, DepthScanOptions, scan_until_converged
from .geometry import SphereFit, SurfaceProjection, SurfaceProjector


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
    """One thresholded connected region and its material-grid centroid."""

    index: int
    center_rc: tuple[float, float]
    region_mask: np.ndarray


@dataclass(frozen=True)
class ContactResult:
    contact: Contact
    radius_mm: float
    sample_count: int
    fit: SphereFit | None
    projection: SurfaceProjection | None
    error: str | None
    fit_mask: np.ndarray
    selection: dict[str, object]
    warnings: tuple[str, ...]
    depth_samples: list[DepthSample] = field(default_factory=list)


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


def find_contacts(reconstruction: Reconstruction, *, count: int | None, threshold_mm: float,
                  min_region_points: int = 100) -> list[Contact]:
    """Find eight-connected regions, ordered by centroid; ``None`` accepts any positive count."""
    if ((count is not None and count < 1) or min_region_points < 4
            or not np.isfinite(threshold_mm) or threshold_mm <= 0):
        raise ValueError('Require at least one contact, a positive threshold and at least four region points')
    groups, number = label(reconstruction.valid & (reconstruction.displacement < -threshold_mm),
                           structure=np.ones((3, 3)))
    regions = []
    for region_id in range(1, number + 1):
        region = groups == region_id
        rows, columns = np.nonzero(region)
        if len(rows) < min_region_points:
            continue
        center = (float(rows.mean()), float(columns.mean()))
        regions.append((center, region))
    if count is None and not regions:
        raise ValueError(f'No separate contacts at {threshold_mm:g} mm. '
                         'Adjust --threshold-mm or --min-region-points.')
    if count is not None and len(regions) != count:
        details = ', '.join(f'({row:.1f},{column:.1f})' for (row, column), _ in regions) or 'none'
        raise ValueError(f'Expected {count} separate contacts at {threshold_mm:g} mm; '
                         f'found {len(regions)} ({details}). Adjust --threshold-mm or --count.')
    regions.sort(key=lambda item: item[0])
    return [Contact(index, center, region)
            for index, (center, region) in enumerate(regions, start=1)]


def _baseline_direction(reconstruction: Reconstruction, mask: np.ndarray) -> np.ndarray:
    vectors = reconstruction.xyz[mask] - reconstruction.baseline_xyz[mask]
    normals = vectors / reconstruction.displacement[mask, None]
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
        raise ValueError('Cannot recover baseline-normal direction for sphere fitting')
    direction = np.mean(normals / lengths[:, None], axis=0)
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length <= 1e-6:
        raise ValueError('Baseline normals do not define a consistent sphere side')
    return direction / length



def analyze_contacts(reconstruction: Reconstruction, contacts: list[Contact],
                     sphere_radii_mm: tuple[float, ...], *,
                     scan_options: DepthScanOptions | None = None) -> list[ContactResult]:
    """Fit the first converged deepest-first subset of each contact.

    Each candidate is a photometrically valid point in the contact mask. The
    selected mask contains exactly the candidates whose displacement is within
    the converged depth of the deepest candidate; on failure it is empty.
    Sphere radii, displacements, depths, and Cartesian positions use mm.
    """
    if (len(sphere_radii_mm) != len(contacts) or any(isinstance(radius, (bool, np.bool_))
            or not isinstance(radius, (int, float, np.integer, np.floating))
            or not np.isfinite(radius) or radius <= 0 for radius in sphere_radii_mm)):
        raise ValueError('Each contact requires one finite positive sphere radius in mm')
    options = DepthScanOptions() if scan_options is None else scan_options
    if not isinstance(options, DepthScanOptions):
        raise ValueError('scan_options must be a DepthScanOptions instance')
    try:
        projector = SurfaceProjector(reconstruction.baseline_xyz, reconstruction.valid)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        projector, projector_error = None, str(exc)
    else:
        projector_error = None

    results = []
    for contact, radius in zip(contacts, sphere_radii_mm, strict=True):
        radius = float(radius)
        candidates = contact.region_mask & reconstruction.valid & reconstruction.photometric_valid
        candidate_count = int(candidates.sum())
        fit_mask = np.zeros(reconstruction.valid.shape, dtype=bool)
        fit, projection, error = None, None, None
        samples: list[DepthSample] = []
        selection: dict[str, object] = dict(
            method='depth_not_converged', candidate_count=candidate_count,
            converged_depth_mm=None, converged_rmse_mm=None, converged_mse_mm2=None,
            scanned_levels=0, convergence_options=asdict(options),
            last_depth_mm=None, last_rmse_mm=None, last_fit_error=None,
            center_baseline_signed_distance_mm=None,
        )
        if candidate_count == 0:
            error = 'Depth scan did not converge: no photometrically valid candidates'
        else:
            points = reconstruction.xyz[candidates]
            displacement = reconstruction.displacement[candidates]
            with np.errstate(divide='ignore', invalid='ignore'):
                normals = (points - reconstruction.baseline_xyz[candidates]) / displacement[:, None]
            try:
                scan = scan_until_converged(points, displacement, normals, radius, options=options)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                error = f'Depth scan did not converge: {exc}'
            else:
                samples = scan.samples
                selection['scanned_levels'] = len(samples)
                if samples:
                    last = samples[-1]
                    selection.update(last_depth_mm=last.depth_mm,
                                     last_rmse_mm=None if last.fit is None else last.fit.radial_rms_mm,
                                     last_fit_error=last.error)
                if scan.sample is None:
                    error = f'Depth scan did not converge: {scan.status}'
                else:
                    sample = scan.sample
                    fit = sample.fit
                    fit_mask = candidates & (reconstruction.displacement - displacement.min()
                                             <= sample.depth_mm)
                    assert fit is not None and int(fit_mask.sum()) == sample.sample_count
                    selection.update(method='depth_convergence',
                                     converged_depth_mm=sample.depth_mm,
                                     converged_rmse_mm=fit.radial_rms_mm,
                                     converged_mse_mm2=fit.radial_rms_mm**2)
        warnings = []
        if fit is not None:
            if projector is None:
                error = f'Surface projection failed: {projector_error}'
            else:
                try:
                    projection = projector.project(fit.center_xyz_mm)
                    direction = _baseline_direction(reconstruction, fit_mask)
                    signed = float((fit.center_xyz_mm - projection.xyz_mm) @ direction)
                    selection['center_baseline_signed_distance_mm'] = signed
                    if signed < 0:
                        warnings.append('center_negative_side')
                    if abs(signed) <= .05 * radius:
                        warnings.append('center_near_baseline')
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    projection = None
                    error = f'Surface projection failed: {exc}'
        results.append(ContactResult(contact, radius, int(fit_mask.sum()), fit, projection, error,
                                     fit_mask, selection, tuple(warnings), samples))
    return results
