"""Reconstruct thickness counterfactuals using frozen calibration models.

The marker fits, learned models, reference chart and material-grid identities
are held fixed. Each requested offset resamples both images, recomputes local
geometry and solves the full local reconstruction. Model calibration offsets
and experimental offsets remain separate provenance values.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from reconstruction.get_surface import build_reconstruction_provider
from reconstruction.local_recon import (
    LocalReconstructionResult,
    LocalReconstructionSettings,
    load_local_reconstruction_settings,
    load_reconstruction_arrays,
    reconstruct_local_surface,
)
from reconstruction.pipeline import (
    MaterialSamplingGrid,
    _fit_frame,
    load_sampling_grid,
    make_reconstruction_input,
    sample_material_map,
)
from reconstruction.recon_batch import load_batch_calibrations
from reconstruction.surface_reconstruction.markers import build_marker_image_mask, marker_reference_sizes
from reconstruction.surface_reconstruction.surface import BicubicSurfaceAdapter
from utils.calibration_store import METHODS
from utils.image_io import load_linear_image, linear_rgb_to_tracking_bgr8
from utils.normal_models import load_normal_model
from utils.surface_domain import extend_surface
from utils.surface_geometry import OffsetSurface
from utils.surface_projection import CameraModel


@dataclass(frozen=True)
class FrozenCalibration:
    model: Any
    grid: MaterialSamplingGrid
    settings: LocalReconstructionSettings
    calibration_offset_mm: float


def _source_record(path: Path, root: Path) -> dict:
    path = path.expanduser().resolve()
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        name = str(path)
    return dict(path=name, sha256=digest.hexdigest(), bytes=path.stat().st_size)


def _check_input_declarations(data: dict, metadata: dict) -> None:
    """Keep the production model identity checks while allowing an experimental offset."""
    for key in ('coordinate_kind', 'coordinate_chart_id', 'domain_kind', 'frame_convention',
                'residual_domain', 'color_space', 'rgb_transfer_function'):
        if key not in data or key not in metadata or data[key] != metadata[key]:
            raise ValueError(f'Normal model/input declaration mismatch: {key}')


@dataclass
class Study:
    """Frozen calibration and marker geometry for a single target image.

    ``reconstruct`` returns the standard local result and updates
    ``last_current_uv`` and ``last_reference_uv`` with the current run's H×W×2
    image coordinates in pixels. All offsets and XYZ positions use millimetres.
    """

    models: dict[str, FrozenCalibration]
    provenance: dict
    reference_image: np.ndarray = field(repr=False)
    current_image: np.ndarray = field(repr=False)
    reference_marker: BicubicSurfaceAdapter = field(repr=False)
    current_marker: BicubicSurfaceAdapter = field(repr=False)
    camera: CameraModel = field(repr=False)
    reference_surface_mask: np.ndarray = field(repr=False)
    current_surface_mask: np.ndarray = field(repr=False)
    reference_marker_mask: np.ndarray = field(repr=False)
    current_marker_mask: np.ndarray = field(repr=False)
    template_xy_mm: np.ndarray = field(repr=False)
    last_current_uv: np.ndarray | None = field(default=None, init=False, repr=False)
    last_reference_uv: np.ndarray | None = field(default=None, init=False, repr=False)
    last_method: str | None = field(default=None, init=False)
    last_offset_mm: float | None = field(default=None, init=False)

    @property
    def methods(self) -> tuple[str, ...]:
        """Return the five frozen model names in their saved batch order."""
        return tuple(self.models)

    def reconstruct(self, method: str, offset_mm: float) -> LocalReconstructionResult:
        """Run image sampling, normal prediction and Poisson solve for one offset."""
        if method not in self.models:
            raise ValueError(f'Unknown method: {method}')
        if isinstance(offset_mm, (bool, np.bool_)) or not np.isscalar(offset_mm):
            raise ValueError('offset_mm must be a finite nonnegative number')
        try:
            offset = float(offset_mm)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('offset_mm must be a finite nonnegative number') from exc
        if not np.isfinite(offset) or offset < 0:
            raise ValueError('offset_mm must be a finite nonnegative number')
        self.last_current_uv = self.last_reference_uv = None
        self.last_method = self.last_offset_mm = None
        frozen = self.models[method]
        grid, calibration, settings = frozen.grid, frozen.model, frozen.settings
        reference_surface = OffsetSurface(extend_surface(self.reference_marker, grid.parameter_bounds), offset)
        current_surface = OffsetSurface(extend_surface(self.current_marker, grid.parameter_bounds), offset)
        reference_map = sample_material_map(reference_surface, self.camera, grid, self.reference_surface_mask)
        current_map = sample_material_map(current_surface, self.camera, grid, self.current_surface_mask)
        data = make_reconstruction_input(
            current_map, reference_map, self.current_image, self.reference_image,
            self.current_surface_mask, self.reference_surface_mask,
            self.current_marker_mask, self.reference_marker_mask, offset)
        _check_input_declarations(data, calibration.metadata)
        xyz, colors, valid, photo, trusted, st = load_reconstruction_arrays(
            data, expected_metadata=None, include_st=True)
        result = reconstruct_local_surface(
            xyz, colors, calibration, st=st, valid_mask=valid, photometric_valid_mask=photo,
            zero_color_inner_radius=settings.zero_color_inner_radius,
            zero_color_outer_radius=settings.zero_color_outer_radius,
            zero_color_negative_inner_radius=settings.zero_color_negative_inner_radius,
            zero_color_negative_outer_radius=settings.zero_color_negative_outer_radius,
            trusted_no_contact_mask=trusted if settings.no_contact_constraints_enabled else None,
            trusted_no_contact_confidence=settings.trusted_no_contact_confidence,
            displacement_zero_lambda_per_mm2=(settings.displacement_zero_lambda_per_mm2
                                             if settings.no_contact_constraints_enabled else 0.),
            lsmr_atol=settings.lsmr_atol, lsmr_btol=settings.lsmr_btol,
            lsmr_max_iterations=settings.lsmr_max_iterations)
        lo, hi = self.template_xy_mm.min(axis=0), self.template_xy_mm.max(axis=0)
        extrapolated = np.any((grid.parameters < lo) | (grid.parameters > hi), axis=-1)
        result = replace(result, contact_offset_mm=offset,
                         extrapolated_mask=extrapolated & result.valid_mask,
                         domain_kind=calibration.metadata.get('domain_kind'))
        self.last_current_uv = current_map.uv.copy()
        self.last_reference_uv = reference_map.uv.copy()
        self.last_method, self.last_offset_mm = method, offset
        return result


def prepare_study(config_path: str | Path, image_path: str | Path, reference_path: str | Path,
                  baseline_dir: str | Path) -> Study:
    """Freeze the five existing models, their grids, and both tracked marker fits.

    ``baseline_dir`` supplies each method's archived grid shape; the model's
    original calibration offset is used only to validate and load its grid.
    The same material identities and sampling resolution are retained for
    every experimental offset.
    """
    config = Path(config_path).expanduser().resolve()
    image_file, reference_file = Path(image_path).expanduser().resolve(), Path(reference_path).expanduser().resolve()
    baseline = Path(baseline_dir).expanduser().resolve()
    root = config.parent
    image, reference = load_linear_image(image_file), load_linear_image(reference_file)
    provider = build_reconstruction_provider(config)
    session = provider.session
    reference_fit = _fit_frame(session, reference, 'reference image')
    reference_marker = BicubicSurfaceAdapter(session.model, reference_fit['coefficients'])
    reference_bgr = linear_rgb_to_tracking_bgr8(reference)
    sizes = marker_reference_sizes(reference, reference_fit['tracking']['pixels'], tracking_bgr=reference_bgr)
    reference_marker_mask = build_marker_image_mask(
        reference, reference_fit['tracking']['pixels'], reference_fit['tracking']['valid'], sizes,
        tracking_bgr=reference_bgr)
    session.model.reset()  # Retain marker IDs seeded from the reference image.
    current_fit = _fit_frame(session, image, 'current image')
    current_marker = BicubicSurfaceAdapter(session.model, current_fit['coefficients'])
    current_bgr = linear_rgb_to_tracking_bgr8(image)
    current_marker_mask = build_marker_image_mask(
        image, current_fit['tracking']['pixels'], current_fit['tracking']['valid'], sizes,
        tracking_bgr=current_bgr)
    camera = CameraModel(session.camera, session.distortion, reference.shape[:2])
    artifacts = load_batch_calibrations(config)
    models: dict[str, FrozenCalibration] = {}
    model_provenance = {}
    for method in METHODS:
        artifact = artifacts[method]
        if artifact.grid_file is None:
            raise ValueError(f'{method}: calibration has no frozen material grid')
        archived = baseline/f'{method}.npz'
        with np.load(archived, allow_pickle=False) as data:
            shape = tuple(data['xyz_out'].shape[:2])
            if data['xyz_out'].ndim != 3 or data['xyz_out'].shape[-1] != 3:
                raise ValueError(f'{archived}: invalid archived point cloud')
            archived_offset = float(data['contact_offset_mm'])
        calibration = load_normal_model(artifact.model_file)
        trained_offset = float(calibration.metadata['contact_offset_mm'])
        if not np.isfinite(trained_offset) or not np.isclose(archived_offset, trained_offset, atol=1e-9, rtol=0):
            raise ValueError(f'{method}: archived and calibration offsets differ')
        trained_reference = calibration.metadata.get('reference_image')
        if trained_reference is None or Path(trained_reference).expanduser().resolve() != reference_file:
            raise ValueError(f'{method}: calibration and study reference images differ')
        grid = load_sampling_grid(artifact.grid_file, calibration, camera, shape,
                                  template_xy_mm=session.xy, contact_offset_mm=trained_offset)
        settings = load_local_reconstruction_settings(
            config, input_override=str(image_file), calibration_override=str(artifact.model_file))
        models[method] = FrozenCalibration(calibration, grid, settings, trained_offset)
        chart_file = artifact.grid_file.parent/'reference_coordinates.npz'
        with np.load(chart_file, allow_pickle=False) as chart:
            if str(chart['chart_id']) != grid.chart_id:
                raise ValueError(f'{method}: reference chart and material grid differ')
        model_provenance[method] = dict(
            calibration_offset_mm=trained_offset, grid_shape=list(shape), chart_id=grid.chart_id,
            model=_source_record(artifact.model_file, root),
            material_grid=_source_record(artifact.grid_file, root),
            reference_chart=_source_record(chart_file, root),
            archived_reconstruction=_source_record(archived, root),
            reconstruction_settings={name: getattr(settings, name) for name in (
                'lsmr_atol', 'lsmr_btol', 'lsmr_max_iterations',
                'zero_color_inner_radius', 'zero_color_outer_radius',
                'zero_color_negative_inner_radius', 'zero_color_negative_outer_radius',
                'no_contact_constraints_enabled', 'trusted_no_contact_confidence',
                'displacement_zero_lambda_per_mm2')},
            model_metadata=deepcopy(calibration.metadata),
        )
    shapes = {item.grid.parameters.shape[:2] for item in models.values()}
    if len(shapes) != 1:
        raise ValueError(f'Archived method grids differ in shape: {sorted(shapes)}')
    provenance = dict(
        kind='frozen_calibration_thickness_sensitivity',
        description='Both images are resampled and the local reconstruction is solved at each experimental '
                    'offset using the original fixed normal models and material identities.',
        interpretation='Offsets other than the calibration offset apply the frozen models across thicknesses '
                       'and are sensitivity experiments, rather than separately calibrated reconstructions.',
        config=_source_record(config, root), image=_source_record(image_file, root),
        reference_image=_source_record(reference_file, root),
        surface_config=_source_record(session.config_path, root),
        template=_source_record(session.config_path.parent/session.config['template_file'], root),
        camera_calibration=_source_record(session.config_path.parent/session.config['camera_file'], root),
        reference_marker_fit=deepcopy(reference_fit['info']),
        current_marker_fit=deepcopy(current_fit['info']),
        reference_marker_coefficients=reference_fit['coefficients'].tolist(),
        current_marker_coefficients=current_fit['coefficients'].tolist(),
        marker_template_ids=session.ids.tolist(), marker_template_xy_mm=session.xy.tolist(),
        reference_marker_pixels=reference_fit['tracking']['pixels'].tolist(),
        current_marker_pixels=current_fit['tracking']['pixels'].tolist(),
        reference_tracked_count=int(reference_fit['tracking']['valid'].sum()),
        current_tracked_count=int(current_fit['tracking']['valid'].sum()),
        models=model_provenance,
    )
    return Study(models, provenance, reference, image, reference_marker, current_marker, camera,
                 reference_fit['inner_region'].mask, current_fit['inner_region'].mask,
                 reference_marker_mask, current_marker_mask, np.asarray(session.xy, np.float64))
