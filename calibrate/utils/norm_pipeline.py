"""Image calibration stages using the shared geometry and sampling utilities."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
import csv
import filecmp
import glob
import hashlib
import json
import math
import os
import shutil
import cv2
import numpy as np
from utils.config import load_config, load_contact_surface_offset_mm
from utils.calibration_store import CalibrationRun, activate_run, create_run, lut_directory, stage_reference_chart, validate_method
from utils.image_io import load_linear_image, linear_rgb_to_tracking_bgr8, validate_linear_rgb
from utils.surface_coordinates import (ArcLengthChart, COORDINATE_KIND, FRAME_CONVENTION, load_arclength_chart,
                                       save_arclength_chart, build_material_sampling_grid)
from utils.surface_geometry import (OffsetSurface, ParametricSurface, compute_surface_geometry, normals_to_local_slopes,
                                    local_slopes_to_normals)
from utils.surface_domain import DOMAIN_KIND, extend_surface
from utils.surface_projection import CameraModel, project_points, unproject_rays
from utils.surface_sampling import (sample_mask_to_surface,
                                    sample_linear_rgb_to_surface)
from utils.sphere_geometry import (intersect_sphere_rays, select_sphere_cap,
                                   sphere_normals)
from utils.sphere_contact import SphereSurfaceContact
from utils.color_difference import (CAMERA_RGB_TRANSFER_FUNCTION,
                                    require_camera_rgb_transfer, signed_color_difference)
from utils.region_annotations import dilate_contact_mask, load_regions, region_masks
from utils.reconstruction_interface import (load_material_coordinate_specs, MaterialCoordinateMap,
                                            MaterialCoordinateSystems)
from utils.normal_mapping import build_normal_lut
from utils.normal_models import predict_normal_model
from reconstruction.surface_reconstruction.surface import SurfaceModel, BicubicSurfaceAdapter
from reconstruction.surface_reconstruction.run_reconstruction import SurfaceReconstructionProvider, SurfaceReconstructionSession
from reconstruction.surface_reconstruction.tracking import match_still_markers
from reconstruction.surface_reconstruction.markers import (InnerRegion, find_tracking_region,
    marker_reference_sizes, build_marker_image_mask)


CONTACT_OUTER_MARGIN_PX = 20
MAXIMUM_CONTACT_FIT_ITERATIONS = 6


class FrameCalibrationError(ValueError):
    def __init__(self, code, message, metrics=None, *, preview=None):
        self.code = code
        self.metrics = metrics or {}
        self.preview = preview
        super().__init__(f'{code}: {message}')
        self.display_message = str(self)


def _path(value, base, name):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f'normal_calibration.{name} must be an explicit path')
    value = Path(value).expanduser()
    return (base/value).resolve() if not value.is_absolute() else value.resolve()


@dataclass(frozen=True)
class CalibrationSettings:
    work_dir: Path
    input_file: Path
    output: Path
    annotations_file: Path
    contact_offset_mm: float
    images: tuple[str, ...] = ()
    reference_image: Path | None = None
    image_root: Path | None = None
    sphere_radius_mm: float | None = None
    sphere_side: str | None = None
    boundary_model: str | None = None
    lut_size: int = 64
    maximum_rms_angle_degrees: float = 5.
    background_max_abs: float = .01
    background_min_fraction: float = .1
    validation_fraction: float = .2
    validation_rms_degrees: float = 5.
    maximum_out_of_range_fraction: float = .05
    maximum_unsupported_fraction: float = .05
    method: str = 'table'
    lut_dir: Path | None = None
    export_output: bool = False
    mlp_options: dict = field(default_factory=dict)
    background_contact_exclusion_margin_px: int = CONTACT_OUTER_MARGIN_PX
    single_frame_markers: bool = False

    def __post_init__(self):
        validate_method(self.method)
        margin = self.background_contact_exclusion_margin_px
        if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
            raise ValueError('normal_calibration.background_contact_exclusion_margin_px 必须是非负整数')
        if not isinstance(self.mlp_options, dict):
            raise ValueError('normal_calibration.mlp must be a mapping')
        if not np.isfinite(self.contact_offset_mm):
            raise ValueError('contact_surface.offset_mm must be finite')
        positive = (self.maximum_rms_angle_degrees, self.validation_rms_degrees)
        if not np.isfinite(positive).all() or min(positive) <= 0 or self.lut_size < 2:
            raise ValueError('Invalid LUT settings')
        if (not 0 < self.validation_fraction < 1 or not 0 <= self.background_max_abs
                or not np.isfinite(self.background_max_abs) or not 0 < self.background_min_fraction <= 1
                or not 0 <= self.maximum_out_of_range_fraction <= 1
                or not 0 <= self.maximum_unsupported_fraction <= 1):
            raise ValueError('Invalid calibration checks')

    def require_sphere(self):
        if self.sphere_radius_mm is None or not np.isfinite(self.sphere_radius_mm) or self.sphere_radius_mm <= 0:
            raise ValueError('sphere_radius_mm must specify the real positive radius in mm')
        if self.sphere_side not in {'positive_normal', 'negative_normal'}:
            raise ValueError('sphere_side must be positive_normal or negative_normal')
        if self.boundary_model != 'zero_boundary_displacement':
            raise ValueError('Only explicit zero_boundary_displacement is supported')

    @property
    def side(self):
        self.require_sphere()
        return 1 if self.sphere_side == 'positive_normal' else -1

    def lut_options(self):
        return dict(size=self.lut_size, maximum_rms_angle_degrees=self.maximum_rms_angle_degrees)


def parse_settings(config_path, overrides=None):
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    raw = dict(config.get('normal_calibration', {}))
    raw.update({key: value for key, value in (overrides or {}).items() if value is not None})
    base = config_path.parent
    work = _path(raw.get('work_dir', 'assets/normal_calibration'), base, 'work_dir')
    images = raw.get('images', [])
    if isinstance(images, str):
        images = [images]
    patterns = tuple(str(_path(pattern, base, 'images')) for pattern in images)
    reference = raw.get('reference_image')
    checks = raw.get('checks') or {}
    return CalibrationSettings(work_dir=work,
        input_file=_path(raw.get('input_file') or work/'normal_samples.npz', base, 'input_file'),
        output=_path(raw.get('output') or work/'normal_lut.npz', base, 'output'),
        annotations_file=_path(raw.get('annotations_file') or work/'manual_regions.yaml', base, 'annotations_file'),
        contact_offset_mm=load_contact_surface_offset_mm(config_path),
        images=patterns, reference_image=_path(reference, base, 'reference_image') if reference else None,
        sphere_radius_mm=raw.get('sphere_radius_mm'), sphere_side=raw.get('sphere_side'),
        boundary_model=raw.get('boundary_model'),
        method=validate_method(raw.get('method', 'table')),
        lut_dir=lut_directory(config_path, dict(config, normal_calibration=raw)),
        export_output=bool((overrides or {}).get('output') or raw.get('output')),
        mlp_options=raw.get('mlp') or {},
        background_contact_exclusion_margin_px=raw.get('background_contact_exclusion_margin_px',
                                                       CONTACT_OUTER_MARGIN_PX),
        lut_size=int(raw.get('lut_size', 64)),
        maximum_rms_angle_degrees=float(raw.get('maximum_rms_angle_degrees', 5)),
        **{key: float(checks[key]) for key in ('background_max_abs', 'background_min_fraction',
           'validation_fraction', 'validation_rms_degrees', 'maximum_out_of_range_fraction',
           'maximum_unsupported_fraction') if key in checks})


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_value), encoding='utf-8')
    temporary.replace(path)


def _save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {key: np.asarray(value) for key, value in values.items()}
    if any(value.dtype.hasobject for value in values.values()):
        raise ValueError('Object arrays are forbidden in calibration caches')
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)
    return path


@dataclass
class FrameCalibrationData:
    image_key: str
    image_index: int
    source_image: Path
    linear_rgb: np.ndarray
    surface_mask: np.ndarray
    tracking: dict
    template_xy_mm: np.ndarray
    reference_marker_sizes: np.ndarray
    marker_image_mask: np.ndarray
    coefficients: np.ndarray
    surface: ParametricSurface
    contact_offset_mm: float
    camera: CameraModel
    chart: ArcLengthChart
    chart_file: Path
    systems: MaterialCoordinateSystems
    geometry_stage: str
    fit_info: dict
    solver_options: dict
    marker_fit_valid: np.ndarray | None = None


def _provider_from_frame(frame):
    """Rehydrate only cached calibration arrays; do not reopen configuration or images."""
    session = SurfaceReconstructionSession.__new__(SurfaceReconstructionSession)
    session.model = SurfaceModel(frame.template_xy_mm, frame.camera.matrix, frame.camera.distortion, frame.solver_options)
    session.xy = frame.template_xy_mm
    session.ids = np.arange(len(session.xy))
    session.camera, session.distortion = frame.camera.matrix, frame.camera.distortion
    session.expected_size = frame.camera.image_shape[::-1]
    session.row_direction = 'auto'
    session.shape_failed_frames = session.lost_frames = session.segment = 0
    session.tracker = None
    provider = SurfaceReconstructionProvider.__new__(SurfaceReconstructionProvider)
    provider.session = session
    provider.contact_offset_mm = frame.contact_offset_mm
    provider.coordinate_chart = frame.chart
    provider.coordinate_chart_file = frame.chart_file
    provider.last_frame_result = dict(tracking=frame.tracking,
        inner_region=InnerRegion((0, 0, *session.expected_size), np.empty((0, 1, 2), int),
                                 frame.surface_mask, frame.surface_mask),
        coefficients=frame.coefficients, info=frame.fit_info)
    return provider


def _specs(frame):
    from utils.reconstruction_interface import CoordinateGridSpec, MaterialCoordinateSpecs
    return MaterialCoordinateSpecs(**{name: CoordinateGridSpec(*getattr(frame.systems, name).xyz.shape[:2])
                                     for name in ('geometry', 'lightfield', 'observation')})


def _save_frame(path, frame):
    from reconstruction.get_surface import save_material_coordinates
    provider = _provider_from_frame(frame)
    if frame.marker_fit_valid is not None:
        provider.last_frame_result['marker_fit_valid'] = frame.marker_fit_valid
    save_material_coordinates(path, frame.systems, frame.surface_mask, frame.linear_rgb, provider,
        surface_coefficients=frame.coefficients, geometry_stage=frame.geometry_stage,
        metadata=dict(image_key=frame.image_key, image_index=frame.image_index, source_image=str(frame.source_image)))
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    grid = frame.systems.observation
    geometry = compute_surface_geometry(grid.xyz, grid.valid_mask)
    values.update(principal_curvatures_per_mm=geometry['principal_curvatures'].astype(np.float32),
                  curvature_grid_per_mm=_curvature_components(geometry), curvature_valid_mask=geometry['valid'])
    values.update(reference_marker_sizes=np.asarray(frame.reference_marker_sizes, np.float64),
                  marker_image_mask=np.asarray(frame.marker_image_mask, bool),
                  solver_options=np.asarray(json.dumps(frame.solver_options)),
                  format_version=np.asarray(5, np.int32),
                  contact_offset_mm=np.asarray(frame.contact_offset_mm, np.float64))
    return _save_npz(path, **values)


def _curvature_components(geometry):
    """Return the symmetric curvature tensor in the same tangent basis as p/q, in 1/mm."""
    tensor = np.asarray(geometry['curvature_tensor'])
    return np.stack((tensor[..., 0, 0], tensor[..., 0, 1], tensor[..., 1, 1]), axis=-1).astype(np.float32)


def _require_contact_offset(actual, expected, artifact_name):
    try:
        actual = float(actual)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{artifact_name} has no valid contact_offset_mm; rerun prepare and calibration') from exc
    if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-9, rtol=0):
        raise ValueError(
            f'{artifact_name} contact_offset_mm={actual!r} differs from configured '
            f'{expected!r}; rerun prepare and calibration')


def _frame_surface(model, coefficients, contact_offset_mm, parameter_bounds):
    marker_surface = BicubicSurfaceAdapter(model, coefficients)
    return OffsetSurface(extend_surface(marker_surface, parameter_bounds), contact_offset_mm)


def load_frame_calibration_data(path: str | Path, settings: CalibrationSettings) -> FrameCalibrationData:
    """Replay frozen geometry, refreshing frame and marker masks from cached RGB."""
    path = Path(path).resolve()
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(data['format_version'])
            if version == 5:
                if str(data['color_space']) != 'linear_rgb':
                    raise ValueError('Geometry cache requires linear RGB')
                require_camera_rgb_transfer({'rgb_transfer_function': str(data['rgb_transfer_function'])}, '几何缓存')
                linear_rgb = validate_linear_rgb(data['observed_linear_rgb'])
            else:
                raise ValueError('Geometry cache predates black-frame domain; rerun prepare')
            if 'contact_offset_mm' not in data:
                raise ValueError('Geometry cache has no contact_offset_mm; rerun prepare')
            contact_offset_mm = float(data['contact_offset_mm'])
            _require_contact_offset(contact_offset_mm, settings.contact_offset_mm, 'Geometry cache')
            if 'domain_kind' not in data:
                raise ValueError('Geometry cache lacks black-frame domain; rerun prepare')
            chart_file = (path.parent/str(data['coordinate_chart_file'])).resolve()
            chart = load_arclength_chart(chart_file)
            if (str(data['coordinate_chart_id']) != chart.chart_id
                    or str(data['coordinate_kind']) != COORDINATE_KIND
                    or str(data['frame_convention']) != FRAME_CONVENTION
                    or str(data['domain_kind']) != DOMAIN_KIND or chart.domain_kind != DOMAIN_KIND):
                raise ValueError('Coordinate chart or black-frame domain mismatch; rerun prepare')
            camera = CameraModel(data['camera_matrix'], data['dist_coeffs'], tuple(data['image_shape']))
            template = np.asarray(data['template_xy_mm'], np.float64)
            options = json.loads(str(data['solver_options']))
            model = SurfaceModel(template, camera.matrix, camera.distortion, options)
            coefficients = np.asarray(data['surface_coefficients'], np.float64)
            surface = _frame_surface(model, coefficients, contact_offset_mm, chart.parameter_bounds)
            try:
                _, region = find_tracking_region(linear_rgb_to_tracking_bgr8(linear_rgb))
            except (RuntimeError, cv2.error) as exc:
                raise FrameCalibrationError('region_not_found', str(exc)) from exc
            maps = {}
            for name in ('geometry', 'lightfield', 'observation'):
                mask_key = ('geometry_grid_valid_mask' if name == 'geometry' and 'geometry_grid_valid_mask' in data
                            else f'{name}_valid_mask')
                maps[name] = MaterialCoordinateMap(data[f'{name}_xyz'], data[f'{name}_uv'],
                    data[f'{name}_coordinates'], data[mask_key], data[f'{name}_st'], COORDINATE_KIND, chart.chart_id)
                expected_st, coordinate_valid = chart.to_st(maps[name].coordinates)
                if not coordinate_valid.all() or not np.allclose(expected_st, maps[name].st, atol=1e-6, rtol=0):
                    raise ValueError('Cached st/material coordinates differ from reference chart')
                # An old segmentation may have removed dark gel connected to
                # the frame. Rebuild validity instead of intersecting that mask;
                # the XYZ, UV and frozen material identities remain unchanged.
                _, projected = project_points(maps[name].xyz, camera)
                maps[name] = replace(maps[name], valid_mask=(coordinate_valid & projected
                    & sample_mask_to_surface(region.mask, maps[name].uv)))
            marker_mask = build_marker_image_mask(linear_rgb, data['marker_pixels'],
                                                 data['marker_valid'], data['reference_marker_sizes'])
            return FrameCalibrationData(str(data['image_key']), int(data['image_index']),
                Path(str(data['source_image'])), linear_rgb, region.mask,
                dict(pixels=data['marker_pixels'], valid=data['marker_valid'], confidence=data['marker_confidence']),
                template, data['reference_marker_sizes'], marker_mask, coefficients, surface, contact_offset_mm,
                camera, chart, chart_file, MaterialCoordinateSystems(**maps), str(data['geometry_stage']),
                json.loads(str(data['fit_info'])), options,
                data['marker_fit_valid'] if 'marker_fit_valid' in data else None)
    except FrameCalibrationError:
        raise
    except (KeyError, ValueError) as exc:
        raise FrameCalibrationError('coordinate_chart_invalid', str(exc)) from exc


def _frame_from_provider(provider, systems, image, source, key, index, sizes, coefficients, stage):
    result = provider.last_frame_result
    session = provider.session
    contact_offset_mm = float(provider.contact_offset_mm)
    return FrameCalibrationData(key, index, Path(source), image, result['inner_region'].mask,
        result['tracking'], session.xy.copy(), sizes,
        build_marker_image_mask(image, result['tracking']['pixels'], result['tracking']['valid'], sizes),
        coefficients.copy(), _frame_surface(session.model, coefficients, contact_offset_mm,
                                            provider.coordinate_chart.parameter_bounds), contact_offset_mm,
        CameraModel(session.camera, session.distortion, image.shape[:2]), provider.coordinate_chart,
        provider.coordinate_chart_file, systems, stage, dict(result['info']), dict(session.model.options),
        result.get('marker_fit_valid'))


def prepare_reference(config_path: str | Path, settings: CalibrationSettings) -> FrameCalibrationData:
    from reconstruction.get_surface import build_reconstruction_provider
    if settings.reference_image is None:
        raise ValueError('reference_image must name the fixed background for this batch')
    image = load_linear_image(settings.reference_image)
    provider = build_reconstruction_provider(config_path, contact_surface=True)
    _require_contact_offset(provider.contact_offset_mm, settings.contact_offset_mm, 'Reconstruction provider')
    try:
        systems = provider.reconstruct(image, load_material_coordinate_specs(config_path))
    except (ValueError, RuntimeError) as exc:
        result = provider.last_frame_result
        code = result['info']['status'] if result and result['info']['status'] in {'region_not_found', 'tracking_failed'} else 'geometry_rejected'
        if result and result.get('coefficients') is not None and result['info']['status'] != 'ambiguous':
            code = 'coordinate_chart_invalid'
        raise FrameCalibrationError(code, str(exc)) from exc
    provider.coordinate_chart_file = (settings.work_dir/'reference_coordinates.npz').resolve()
    save_arclength_chart(provider.coordinate_chart_file, provider.coordinate_chart)
    sizes = marker_reference_sizes(image, provider.last_frame_result['tracking']['pixels'])
    frame = _frame_from_provider(provider, systems, image, settings.reference_image, '__reference__', -1,
                                 sizes, provider.last_frame_result['coefficients'], 'final')
    _save_frame(settings.work_dir/'reference_surface.npz', frame)
    return frame


def prepare_image(config_path, settings: CalibrationSettings, reference: FrameCalibrationData, image_path) -> FrameCalibrationData:
    from reconstruction.get_surface import build_reconstruction_provider
    source = Path(image_path).resolve()
    try:
        image = load_linear_image(source)
    except (OSError, ValueError) as exc:
        raise FrameCalibrationError('tracking_failed', f'Cannot read {source}: {exc}') from exc
    _require_contact_offset(reference.contact_offset_mm, settings.contact_offset_mm, 'Reference cache')
    provider = build_reconstruction_provider(config_path, contact_surface=True)
    _require_contact_offset(provider.contact_offset_mm, settings.contact_offset_mm, 'Reconstruction provider')
    if (not np.array_equal(provider.session.xy, reference.template_xy_mm)
            or not np.array_equal(provider.session.camera, reference.camera.matrix)
            or not np.array_equal(provider.session.distortion, reference.camera.distortion)):
        raise FrameCalibrationError('coordinate_chart_invalid', 'Camera or physical template changed since reference preparation')
    provider.bind_coordinate_chart(reference.chart)
    provider.coordinate_chart_file = reference.chart_file
    if settings.single_frame_markers:
        tracking_bgr = linear_rgb_to_tracking_bgr8(image)
        try:
            _, region = find_tracking_region(tracking_bgr)
        except (ValueError, RuntimeError, cv2.error) as exc:
            raise FrameCalibrationError('region_not_found', str(exc)) from exc
        try:
            tracking = match_still_markers(
                image, reference.tracking['pixels'], provider.session.ids,
                tracking_bgr=tracking_bgr, template_xy=provider.session.xy,
                row_direction=provider.session.row_direction, region=region,
                reference_surface_mask=reference.surface_mask,
            )
        except (ValueError, RuntimeError, cv2.error) as exc:
            raise FrameCalibrationError('tracking_failed', str(exc)) from exc
        result = dict(tracking=tracking, inner_region=region)
        provider.last_frame_result = result
    else:
        seed = provider.track_frame(reference.linear_rgb)
        if seed['error'] or not np.allclose(seed['tracking']['pixels'], reference.tracking['pixels'], atol=1e-5):
            raise FrameCalibrationError('tracking_failed', 'Reference marker identity changed; rerun prepare')
        result = provider.track_frame(image)
        if result['error']:
            raise FrameCalibrationError(result['info']['status'], result['error'])
    try:
        systems = provider.fit_tracked_frame(load_material_coordinate_specs(config_path),
            result['inner_region'].mask, prior_coefficients=reference.coefficients)
    except (ValueError, RuntimeError) as exc:
        info = (provider.last_frame_result or {}).get('info', {})
        if info.get('status') == 'insufficient_observations':
            tracking = result['tracking']
            matched = int(tracking['valid'].sum())
            candidates = tracking.get('candidate_count')
            detail = f' {candidates} cross candidates detected.' if candidates is not None else ''
            needed = provider.session.model.minimum_observations
            raise FrameCalibrationError('tracking_failed', f'Matched {matched} marker IDs, need {needed}.{detail}') from exc
        raise FrameCalibrationError('geometry_rejected', str(exc)) from exc
    root = settings.image_root or source.parent
    key = Path(os.path.relpath(source, root)).as_posix()
    return _frame_from_provider(provider, systems, image, source, key, -1, reference.reference_marker_sizes,
                                 provider.last_frame_result['coefficients'], 'initial')


def _coordinate_metadata(frame, output_path):
    if frame.chart.domain_kind != DOMAIN_KIND:
        raise FrameCalibrationError('coordinate_chart_invalid', 'Black-frame domain required; rerun prepare')
    return dict(coordinate_kind=COORDINATE_KIND, coordinate_chart_id=frame.chart.chart_id,
                domain_kind=frame.chart.domain_kind,
                coordinate_chart_file=os.path.relpath(frame.chart_file, Path(output_path).resolve().parent),
                frame_convention=FRAME_CONVENTION, residual_domain='material_grid', color_space='linear_rgb',
                rgb_transfer_function=CAMERA_RGB_TRANSFER_FUNCTION,
                contact_offset_mm=frame.contact_offset_mm)


def _observed_frame_colors(frame):
    grid = frame.systems.observation
    source = frame.surface_mask & ~frame.marker_image_mask & np.all(frame.linear_rgb < 1, axis=-1)
    observed = sample_linear_rgb_to_surface(frame.linear_rgb, grid.uv, source)
    geometry_valid = grid.valid_mask.copy()
    return dict(observed_color_grid=observed.values.astype(np.float32),
                geometry_valid_mask=geometry_valid,
                photometric_valid_mask=geometry_valid & observed.valid)


def compute_frame_colors(final_frame, reference, settings):
    current, ref = final_frame.systems.observation, reference.systems.observation
    if (current.chart_id != ref.chart_id or current.coordinate_kind != COORDINATE_KIND
            or not np.array_equal(current.coordinates, ref.coordinates) or not np.allclose(current.st, ref.st, atol=1e-7, rtol=0)):
        raise FrameCalibrationError('coordinate_chart_invalid', 'Reference/current material identities differ')
    observed = _observed_frame_colors(final_frame)
    source_ref = reference.surface_mask & ~reference.marker_image_mask & np.all(reference.linear_rgb < 1, axis=-1)
    background = sample_linear_rgb_to_surface(reference.linear_rgb, ref.uv, source_ref)
    photo = observed['photometric_valid_mask'] & ref.valid_mask & background.valid
    delta, photo = signed_color_difference(observed['observed_color_grid'], background.values, photo)
    return dict(reference_color_grid=background.values.astype(np.float32),
                observed_color_grid=observed['observed_color_grid'], color_residual_grid=delta,
                geometry_valid_mask=observed['geometry_valid_mask'], photometric_valid_mask=photo)


def sphere_contact_for_annotation(frame, annotation, settings):
    """Construct the prescribed ball; center_px is the surface anchor projection."""
    if annotation.shape != 'sphere_contact' or annotation.indentation_depth_mm is None:
        raise FrameCalibrationError('annotation_missing',
            'Contact position and indentation depth are required; rerun annotate --edit')
    try:
        contact = SphereSurfaceContact(frame.surface, frame.camera, frame.systems.geometry).compute(
            annotation.center_px, annotation.indentation_depth_mm, settings.sphere_radius_mm, settings.side)
    except ValueError as exc:
        raise FrameCalibrationError('sphere_contact_failed', str(exc)) from exc
    return contact


def _contact_annotation(annotation, contact):
    return replace(annotation, shape='sphere_contact', contour_px=contact.boundary_px.tolist())


def _excluded_marker_ids(frame, exclusion):
    pixels = np.asarray(frame.tracking['pixels'])
    valid = np.asarray(frame.tracking['valid'], bool) & np.isfinite(pixels).all(axis=1)
    ij = np.rint(np.nan_to_num(pixels)).astype(int)
    h, w = exclusion.shape
    inside = valid & (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    excluded = np.zeros(len(pixels), bool)
    excluded[inside] = exclusion[ij[inside, 1], ij[inside, 0]]
    return excluded


def _refit_frame(prepared, reference, provider, exclusion=None, *, stage='initial'):
    try:
        systems = provider.fit_tracked_frame(_specs(prepared), prepared.surface_mask,
                                             exclusion, reference.coefficients)
    except (ValueError, RuntimeError) as exc:
        raise FrameCalibrationError('geometry_rejected', str(exc)) from exc
    result = provider.last_frame_result
    return replace(prepared, coefficients=result['coefficients'],
        surface=_frame_surface(provider.session.model, result['coefficients'], prepared.contact_offset_mm,
                               prepared.chart.parameter_bounds),
        systems=systems, geometry_stage=stage, fit_info=result['info'],
        marker_fit_valid=result['marker_fit_valid'])


def initial_calibration_surface(frame, reference):
    """Upgrade reference-only previews to an actual fit of the current observations."""
    if frame.geometry_stage == 'preview_reference':
        return _refit_frame(frame, reference, _provider_from_frame(frame))
    return frame


def _fit_contact_surface(prepared, reference, annotation, settings):
    """Keep the image anchor fixed and iterate until the excluded markers agree."""
    provider = _provider_from_frame(prepared)
    contact = sphere_contact_for_annotation(prepared, annotation, settings)
    for _ in range(MAXIMUM_CONTACT_FIT_ITERATIONS):
        region = _contact_annotation(annotation, contact)
        masks = region_masks(region, prepared.linear_rgb.shape, outer_margin_px=CONTACT_OUTER_MARGIN_PX)
        excluded = _excluded_marker_ids(prepared, masks['exclusion'])
        final = _refit_frame(prepared, reference, provider, masks['exclusion'], stage='final')
        contact = sphere_contact_for_annotation(final, annotation, settings)
        region = _contact_annotation(annotation, contact)
        masks = region_masks(region, final.linear_rgb.shape, outer_margin_px=CONTACT_OUTER_MARGIN_PX)
        if np.array_equal(excluded, _excluded_marker_ids(prepared, masks['exclusion'])):
            return final, contact, region, masks
    raise FrameCalibrationError('geometry_rejected',
        'Contact boundary / excluded-marker iteration did not converge; adjust position or depth')


def make_sphere_labels(frame, annotation, sphere, settings, *, center_frame=None,
                       center_sigma_mm=None, geometry=None):
    if center_frame is None:
        contact = sphere_contact_for_annotation(frame, annotation, settings)
        center_frame, center_sigma_mm = contact.center_frame, contact.center_sigma_mm
        geometry = compute_surface_geometry(frame.systems.observation.xyz, frame.systems.observation.valid_mask)
    a, _, _, n0 = center_frame
    center_rays, center_ray_valid = unproject_rays(np.asarray(annotation.center_px)[None], frame.camera)
    _, candidates, root_valid = intersect_sphere_rays(np.zeros(3), center_rays, sphere)
    center_normals = -settings.side*sphere_normals(candidates[0], sphere)
    eligible = root_valid[0] & center_ray_valid[0] & ((center_normals @ n0) > .1)
    filtered = np.where(eligible[:, None], candidates[0], np.nan)
    try:
        root_sign = select_sphere_cap(filtered, a, n0, settings.side, center_sigma_mm)
    except ValueError as exc:
        raise FrameCalibrationError('sphere_branch_ambiguous', str(exc)) from exc
    root_index = 0 if root_sign == -1 else 1
    grid = frame.systems.observation
    rays, ray_valid = unproject_rays(grid.uv, frame.camera)
    _, points, roots_valid = intersect_sphere_rays(np.zeros(3), rays, sphere)
    normals = -settings.side*sphere_normals(points[..., root_index, :], sphere)
    slopes, finite = normals_to_local_slopes(normals, geometry['normal'], geometry['frame'])
    masks = region_masks(annotation, frame.linear_rgb.shape, outer_margin_px=CONTACT_OUTER_MARGIN_PX)
    contact = sample_mask_to_surface(masks['contact'], grid.uv)
    valid = (ray_valid & roots_valid[..., root_index] & finite & geometry['valid']
             & ((normals @ n0) > .1) & contact)
    return dict(slope_grid=slopes.astype(np.float32), normal_gt_grid=normals.astype(np.float32),
                sphere_label_valid=valid, contact_mask=contact,
                sphere_center_xyz=sphere.center, sphere_radius_mm=sphere.radius_mm,
                sphere_root_sign=root_sign, sphere_side=settings.sphere_side,
                boundary_model=settings.boundary_model)


def _finalize_contact_geometry(prepared, reference, annotation, settings):
    settings.require_sphere()
    if annotation is None or annotation.semantics != 'contact_boundary':
        raise FrameCalibrationError('annotation_missing', 'An explicit contact_boundary annotation is required')
    _require_contact_offset(prepared.contact_offset_mm, settings.contact_offset_mm, 'Prepared frame')
    _require_contact_offset(reference.contact_offset_mm, settings.contact_offset_mm, 'Reference frame')
    if prepared.chart.chart_id != reference.chart.chart_id or not np.array_equal(prepared.template_xy_mm, reference.template_xy_mm):
        raise FrameCalibrationError('coordinate_chart_invalid', 'Prepared/reference chart or template mismatch')
    if (not np.array_equal(prepared.camera.matrix, reference.camera.matrix)
            or not np.array_equal(prepared.camera.distortion, reference.camera.distortion)):
        raise FrameCalibrationError('coordinate_chart_invalid', 'Prepared/reference camera mismatch')
    prepared = initial_calibration_surface(prepared, reference)
    try:
        final, contact, annotation, masks = _fit_contact_surface(prepared, reference, annotation, settings)
    except FrameCalibrationError:
        raise
    except ValueError as exc:
        raise FrameCalibrationError('sphere_contact_failed', str(exc)) from exc
    return final, contact, annotation, masks


def _geometry_record(final, contact, annotation, colors, metrics):
    geometry = compute_surface_geometry(final.systems.observation.xyz, final.systems.observation.valid_mask)
    record = dict(frame=final, **colors, metrics=metrics,
                  contact_boundary_px=contact.boundary_px,
                  contact_center_parameters=contact.center_parameters,
                  indentation_depth_mm=annotation.indentation_depth_mm,
                  principal_curvatures_per_mm=geometry['principal_curvatures'].astype(np.float32),
                  curvature_grid_per_mm=_curvature_components(geometry),
                  curvature_valid_mask=geometry['valid'])
    return record, geometry


def _fit_metrics(frame):
    return dict(sample_count=0, reprojection_rms_px=frame.fit_info.get('reprojection_rms_px'),
                metric_max_error=frame.fit_info.get('metric_max_error'),
                marker_fit_count=int(frame.marker_fit_valid.sum()))


def _add_sphere_labels(record, contact, annotation, settings, geometry):
    final = record['frame']
    metrics = record['metrics']
    residual_mm = np.linalg.norm(contact.boundary_xyz-contact.sphere.center, axis=1)-contact.sphere.radius_mm
    metrics.update(boundary_valid_count=len(contact.boundary_px), boundary_max_missing=0,
                   sphere_boundary_rms=float(np.sqrt(np.mean(residual_mm**2)) / contact.center_sigma_mm),
                   indentation_depth_mm=annotation.indentation_depth_mm)
    labels = make_sphere_labels(final, annotation, contact.sphere, settings,
                                center_frame=contact.center_frame,
                                center_sigma_mm=contact.center_sigma_mm, geometry=geometry)
    valid = (record['geometry_valid_mask'] & record['photometric_valid_mask']
             & labels['contact_mask'] & labels['sphere_label_valid'])
    metrics['sample_count'] = int(valid.sum())
    record.update(**labels, sample_valid_mask=valid,
                  reference_normals=geometry['normal'].astype(np.float32),
                  tangent_frames=geometry['frame'].astype(np.float32))
    if not valid.any():
        raise FrameCalibrationError('no_valid_samples', 'No paired colors and sphere slopes')


def _raise_preview_errors(record, errors):
    if errors:
        error = errors[0]
        error.preview = dict(record, preview_errors=[str(item) for item in errors])
        error.display_message = '\n'.join(str(item) for item in errors)
        raise error


def finalize_image(prepared: FrameCalibrationData, reference: FrameCalibrationData, annotation,
                   settings: CalibrationSettings, *, diagnostic_preview=False) -> dict:
    """Return accepted calibration; optional diagnostics are attached to raised errors only."""
    final, contact, annotation, masks = _finalize_contact_geometry(prepared, reference, annotation, settings)
    colors = compute_frame_colors(final, reference, settings)
    margin = settings.background_contact_exclusion_margin_px
    background_exclusion = dilate_contact_mask(masks['contact'], margin)
    outside = sample_mask_to_surface(~background_exclusion, final.systems.observation.uv)
    background_valid = colors['photometric_valid_mask'] & outside
    denominator = np.count_nonzero(colors['photometric_valid_mask'])
    fraction = float(background_valid.sum()/max(denominator, 1))
    median = (float(np.median(np.max(np.abs(colors['color_residual_grid'][background_valid]), axis=-1)))
              if background_valid.any() else None)
    metrics = dict(background_fraction=fraction, background_median_max_abs=median,
                   background_contact_exclusion_margin_px=margin, **_fit_metrics(final))
    record, geometry = _geometry_record(final, contact, annotation, colors, metrics)
    record['reference_image'] = str(reference.source_image)
    errors = []
    if median is None or fraction < settings.background_min_fraction or median > settings.background_max_abs:
        error = FrameCalibrationError('background_mismatch',
            f'fraction={fraction:.5g} (min={settings.background_min_fraction:.5g}), '
            f'median={median} (max={settings.background_max_abs:.5g})',
            dict(background_fraction=fraction, background_median_max_abs=median))
        if not diagnostic_preview:
            raise error
        errors.append(error)
    try:
        _add_sphere_labels(record, contact, annotation, settings, geometry)
    except FrameCalibrationError as error:
        if not diagnostic_preview:
            raise
        errors.append(error)
    _raise_preview_errors(record, errors)
    return record


def finalize_observed_image(prepared: FrameCalibrationData, anchor: FrameCalibrationData, annotation,
                            settings: CalibrationSettings, *, diagnostic_preview=False) -> dict:
    """Fit the prescribed contact and sample current linear RGB on its material grid.

    The anchor supplies the surface-fit prior and chart identity. Photometric
    validity uses current-frame surface, marker and saturation masks. Returned
    sphere normals, slopes, curvature and masks have observation-grid shape.
    """
    final, contact, annotation, _ = _finalize_contact_geometry(prepared, anchor, annotation, settings)
    colors = _observed_frame_colors(final)
    record, geometry = _geometry_record(final, contact, annotation, colors, _fit_metrics(final))
    try:
        _add_sphere_labels(record, contact, annotation, settings, geometry)
    except FrameCalibrationError as error:
        if not diagnostic_preview:
            raise
        _raise_preview_errors(record, [error])
    return record


def _write_final_record(record, reference, settings):
    frame = record['frame']
    if frame.geometry_stage != 'final' or record.get('preview_errors'):
        raise FrameCalibrationError('geometry_rejected', 'Rejected previews cannot supply training samples')
    stem = f'{frame.image_index:06d}'
    _save_frame(settings.work_dir/'overall'/f'{stem}.npz', frame)
    grid = frame.systems.observation
    path = settings.work_dir/'frames'/f'{stem}.npz'
    values = {key: value for key, value in record.items() if key not in {'frame', 'metrics'}}
    values.update(format_version=1, xyz=grid.xyz, uv=grid.uv, st=grid.st.astype(np.float32),
                  material_coordinates=grid.coordinates, geometry_stage='final',
                  image_key=frame.image_key, image_index=frame.image_index,
                  reference_image=str(reference.source_image), fit_info=json.dumps(record['metrics']),
                  **_coordinate_metadata(frame, path))
    _save_npz(path, **values)


def export_normal_samples(frame_records, settings, output_path):
    records = list(frame_records)
    if not records:
        raise FrameCalibrationError('no_valid_samples', 'No accepted image has valid samples')
    keys = [record['frame'].image_key for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Image keys must be unique')
    fields = {key: [] for key in ('color_residual_samples', 'slope_samples', 'image_index',
                                  'material_coordinates', 'st_samples', 'reference_normal_samples',
                                  'tangent_frame_samples', 'curvature_samples', 'curvature_valid_mask')}
    first = records[0]['frame']
    for index, record in enumerate(records):
        if record.get('preview_errors'):
            raise FrameCalibrationError('geometry_rejected', 'Rejected previews cannot supply training samples')
        frame = record['frame']
        _require_contact_offset(frame.contact_offset_mm, settings.contact_offset_mm, 'Sample frame')
        if (frame.geometry_stage != 'final' or frame.chart.chart_id != first.chart.chart_id
                or record['reference_image'] != records[0]['reference_image']):
            raise FrameCalibrationError('coordinate_chart_invalid', 'Samples need final geometry and one frozen chart')
        valid = np.asarray(record['sample_valid_mask'], bool)
        grid = frame.systems.observation
        fields['color_residual_samples'].append(record['color_residual_grid'][valid])
        fields['slope_samples'].append(record['slope_grid'][valid])
        fields['image_index'].append(np.full(valid.sum(), index, np.int32))
        fields['material_coordinates'].append(grid.coordinates[valid])
        fields['st_samples'].append(grid.st[valid].astype(np.float32))
        fields['reference_normal_samples'].append(record['reference_normals'][valid])
        fields['tangent_frame_samples'].append(record['tangent_frames'][valid])
        fields['curvature_samples'].append(record['curvature_grid_per_mm'][valid])
        fields['curvature_valid_mask'].append(record['curvature_valid_mask'][valid])
    values = {key: np.concatenate(value) for key, value in fields.items()}
    values.update(format_version=1, image_paths=np.asarray(keys),
                  reference_image=records[0]['reference_image'],
                  sphere_radius_mm=settings.sphere_radius_mm, sphere_side=settings.sphere_side,
                  boundary_model=settings.boundary_model, **_coordinate_metadata(first, output_path))
    return _save_npz(output_path, **values)


def sample_curvatures_from_caches(samples, work_dir):
    """Reconstruct old sample curvatures from matching final geometry, without editing caches.

    The returned Nx3 tensor components use the same signed orthonormal tangent
    basis as the cached p/q labels. Every image's sample order and coordinate
    chart must agree with both its final frame and overall geometry files.
    """
    colors = np.asarray(samples['color_residual_samples'])
    slopes = np.asarray(samples['slope_samples'])
    groups = np.asarray(samples['image_index'])
    keys = np.asarray(samples['image_paths']).astype(str)
    st = np.asarray(samples['st_samples'])
    n = len(colors)
    if (colors.shape != (n, 3) or slopes.shape != (n, 2) or groups.shape != (n,)
            or st.shape != (n, 2) or groups.dtype.kind not in 'iu'
            or (groups < 0).any() or (groups >= len(keys)).any() or len(set(keys)) != len(keys)):
        raise ValueError('Invalid grouped sample identities for curvature recovery')
    work = Path(work_dir)
    manifest = json.loads((work/'manifest.json').read_text(encoding='utf-8'))
    chart_id = str(np.asarray(samples['coordinate_chart_id']).item())
    if manifest.get('coordinate_chart_id') != chart_id:
        raise ValueError('Manifest/sample coordinate chart mismatch; rerun samples')
    entries = {row['key']: row for row in manifest['images']}
    if len(entries) != len(manifest['images']):
        raise ValueError('Duplicate image keys in calibration manifest')
    curvature = np.zeros((n, 3), np.float32)
    validity = np.zeros(n, bool)
    for group in np.unique(groups):
        key = keys[group]
        if key not in entries:
            raise ValueError(f'Missing manifest entry for sample image {key}')
        stem = f"{entries[key]['index']:06d}.npz"
        try:
            with np.load(work/'frames'/stem, allow_pickle=False) as frame, \
                    np.load(work/'overall'/stem, allow_pickle=False) as overall:
                for artifact in (frame, overall):
                    if (str(artifact['image_key']) != key or str(artifact['coordinate_chart_id']) != chart_id
                            or str(artifact['geometry_stage']) != 'final'):
                        raise ValueError(f'Final frame/overall identity or chart mismatch for {key}')
                    for field in ('coordinate_kind', 'domain_kind', 'frame_convention', 'contact_offset_mm'):
                        if str(artifact[field]) != str(samples[field]):
                            raise ValueError(f'{field} differs for {key}')
                if str(frame['reference_image']) != str(samples['reference_image']):
                    raise ValueError(f'Reference image differs for {key}')
                selected = np.asarray(frame['sample_valid_mask'], bool)
                sample_group = groups == group
                for frame_field, sample_field in (('color_residual_grid', 'color_residual_samples'),
                        ('slope_grid', 'slope_samples'), ('st', 'st_samples'),
                        ('material_coordinates', 'material_coordinates'),
                        ('reference_normals', 'reference_normal_samples'),
                        ('tangent_frames', 'tangent_frame_samples')):
                    if not np.array_equal(frame[frame_field][selected], np.asarray(samples[sample_field])[sample_group]):
                        raise ValueError(f'{frame_field} samples differ for {key}')
                for frame_field, overall_field in (('xyz', 'observation_xyz'), ('uv', 'observation_uv'),
                        ('st', 'observation_st'), ('material_coordinates', 'observation_coordinates'),
                        ('geometry_valid_mask', 'observation_valid_mask')):
                    if not np.array_equal(frame[frame_field], overall[overall_field]):
                        raise ValueError(f'Final frame/overall {frame_field} differs for {key}')
                geometry = compute_surface_geometry(frame['xyz'], frame['geometry_valid_mask'])
                curvature[sample_group] = _curvature_components(geometry)[selected]
                validity[sample_group] = geometry['valid'][selected]
        except (FileNotFoundError, KeyError) as exc:
            raise ValueError(f'Missing final geometry for {key}: {exc}') from exc
    return curvature, validity


def _reference_sampling_grid_values(settings, metadata, *, output_dir=None):
    reference = load_frame_calibration_data(settings.work_dir/'reference_surface.npz', settings)
    expected = dict(coordinate_kind=COORDINATE_KIND, coordinate_chart_id=reference.chart.chart_id,
                    domain_kind=DOMAIN_KIND,
                    frame_convention=FRAME_CONVENTION, residual_domain='material_grid', color_space='linear_rgb',
                    contact_offset_mm=settings.contact_offset_mm)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'LUT/reference {key} mismatch; regenerate samples from this reference')
    if Path(metadata['reference_image']).resolve() != reference.source_image.resolve():
        raise ValueError('LUT/reference image mismatch; regenerate samples from this reference')
    shape = reference.systems.observation.xyz.shape[:2]
    values = build_material_sampling_grid(reference.surface, reference.chart, shape,
                                          reference.template_xy_mm, reference.camera)
    chart_directory = settings.work_dir if output_dir is None else Path(output_dir)
    chart_path = reference.chart_file if output_dir is None else chart_directory/'reference_coordinates.npz'
    values.update(reference_image=np.asarray(str(reference.source_image)),
                  coordinate_chart_file=np.asarray(os.path.relpath(chart_path, chart_directory)),
                  contact_offset_mm=np.asarray(settings.contact_offset_mm, np.float64))
    return values


@dataclass(frozen=True)
class _FitFiles:
    directory: Path
    model_file: Path
    training_file: Path
    grid_file: Path
    chart_file: Path
    report_file: Path
    run: CalibrationRun | None


def _fit_files(settings):
    if settings.lut_dir is not None:
        run = create_run(settings.lut_dir, settings.method)
        return _FitFiles(run.directory, run.model_file, run.training_file, run.grid_file,
                         run.chart_file, run.report_file, run)
    output = settings.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return _FitFiles(output.parent, output, output.with_name(output.stem+'_train.npz'),
                     output.parent/'lut_material_grid.npz', output.parent/'reference_coordinates.npz',
                     output.with_name(output.stem+'_validation.json'), None)


def _split_image_groups(image_index, keys, fraction):
    active = sorted(set(image_index.tolist()), key=lambda index: keys[index])
    if len(active) < 2:
        raise ValueError('Whole-image validation requires at least two images with valid samples')
    shuffled = np.array(active)
    np.random.default_rng(42).shuffle(shuffled)
    count = min(len(active)-1, max(1, math.ceil(fraction*len(active))))
    return shuffled[count:], shuffled[:count]


def _grouped_sample_metadata(samples, sample_path, settings):
    required = ('coordinate_kind', 'coordinate_chart_id', 'coordinate_chart_file', 'frame_convention',
                'residual_domain', 'color_space', 'rgb_transfer_function', 'reference_image',
                'sphere_radius_mm', 'sphere_side', 'boundary_model', 'contact_offset_mm', 'domain_kind')
    missing = set(required)-set(samples)
    if missing:
        raise ValueError(f'Image samples lack {sorted(missing)}; rerun prepare and samples')
    if int(samples['format_version']) != 1:
        raise ValueError('Grouped image samples require format_version=1')
    metadata = {key: samples[key].item() for key in required}
    require_camera_rgb_transfer(metadata, '图像标定样本')
    _require_contact_offset(metadata['contact_offset_mm'], settings.contact_offset_mm, 'Training samples')
    for key, expected in dict(coordinate_kind=COORDINATE_KIND, frame_convention=FRAME_CONVENTION,
                              residual_domain='material_grid', color_space='linear_rgb',
                              boundary_model='zero_boundary_displacement', domain_kind=DOMAIN_KIND).items():
        if metadata[key] != expected:
            raise ValueError(f'Invalid sample declaration: {key}; rerun prepare and samples')
    chart_path = (Path(sample_path).resolve().parent/metadata['coordinate_chart_file']).resolve()
    chart = load_arclength_chart(chart_path)
    if chart.chart_id != metadata['coordinate_chart_id'] or chart.domain_kind != DOMAIN_KIND:
        raise ValueError('Image samples and black-frame chart differ; rerun prepare and samples')
    return metadata, chart_path


def _model_training_data(samples, settings):
    colors = np.asarray(samples['color_residual_samples'], np.float32)
    slopes = np.asarray(samples['slope_samples'], np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3 or slopes.shape != (len(colors), 2) or not len(colors):
        raise ValueError('Expected nonempty paired Nx3 colors and Nx2 slopes')
    if not np.isfinite(colors).all() or not np.isfinite(slopes).all():
        raise ValueError('Training samples must be finite')
    if 'contact_offset_mm' not in samples:
        raise ValueError('Training samples predate contact-surface calibration; rerun prepare and samples')
    _require_contact_offset(samples['contact_offset_mm'], settings.contact_offset_mm, 'Training samples')
    return colors, slopes


def _train_model(method, colors, slopes, groups, st, curvature, curvature_valid, settings):
    if method == 'table':
        return build_normal_lut(colors, slopes, **settings.lut_options())
    from calibrate.utils.train_normal_mlp import train_normal_mlp
    return train_normal_mlp(colors, slopes, groups, method=method, st=st,
                            curvature_per_mm=curvature, curvature_valid_mask=curvature_valid,
                            options=settings.mlp_options)


def _copy_atomically(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if Path(source).resolve() == destination.resolve():
        return
    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            with Path(source).open('rb') as source_stream:
                shutil.copyfileobj(source_stream, stream)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _export_bundle(files, settings, report, sample_path):
    if not settings.export_output or files.run is None:
        return
    destination = settings.output.resolve()
    if destination.is_relative_to(settings.lut_dir.resolve()):
        raise ValueError('--output must not overwrite a versioned calibration run')
    chart_target = destination.parent/'reference_coordinates.npz'
    grid_target = destination.parent/'lut_material_grid.npz'
    protected_names = {'reference_coordinates.npz', 'lut_material_grid.npz', 'validation.json', 'current.json'}
    protected_paths = {Path(sample_path).resolve(), settings.annotations_file.resolve(), files.chart_file.resolve(),
                       (settings.work_dir/'reference_coordinates.npz').resolve(),
                       (settings.work_dir/'reference_surface.npz').resolve()}
    if settings.reference_image is not None:
        protected_paths.add(settings.reference_image.resolve())
    if destination.name in protected_names or destination in protected_paths:
        raise ValueError('--output must not overwrite calibration input data')
    for source, target in ((files.chart_file, chart_target), (files.grid_file, grid_target)):
        if target.exists() and not filecmp.cmp(source, target, shallow=False):
            raise ValueError(f'Existing calibration companion differs: {target}; choose a separate output directory')
    # Stage the bundle and replace the user-requested model only after its
    # companion chart and grid are complete.
    _copy_atomically(files.chart_file, chart_target)
    _copy_atomically(files.grid_file, grid_target)
    _copy_atomically(files.model_file, destination)
    exported_report = destination.with_name(destination.stem+'_validation.json')
    _save_json(exported_report, dict(report, model_file=str(destination)))
    report['exported_model_file'] = str(destination)
    report['exported_report_file'] = str(exported_report)


def _fit_and_validate_samples(sample_path, settings, files):
    sample_digest = _sha256(sample_path)
    with np.load(sample_path, allow_pickle=False) as data:
        samples = {key: data[key] for key in data.files}
    if _sha256(sample_path) != sample_digest:
        raise ValueError('Sample file changed while loading; retry calibration')
    colors, slopes = _model_training_data(samples, settings)
    base_report = dict(method=settings.method, run_dir=str(files.directory),
                       training_file=str(files.training_file), report_file=str(files.report_file),
                       sample_file=str(Path(sample_path).resolve()), sample_sha256=sample_digest)
    if 'image_index' not in samples:
        if settings.method != 'table':
            raise ValueError('MLP training requires grouped images for independent validation')
        model = build_normal_lut(colors, slopes, **settings.lut_options())
        model = replace(model, metadata=dict(method='table', validation_status='unvalidated_external_samples',
                                             contact_offset_mm=settings.contact_offset_mm))
        model.save(files.model_file, total_sample_count=len(colors))
        report = dict(base_report, status='unvalidated_external_samples', model_file=str(files.model_file),
                      model_sha256=_sha256(files.model_file), total_sample_count=len(colors))
        _save_json(files.report_file, report)
        return report

    metadata, chart_path = _grouped_sample_metadata(samples, sample_path, settings)
    groups = np.asarray(samples['image_index'])
    keys = np.asarray(samples['image_paths']).astype(str)
    if (groups.shape != (len(colors),) or groups.dtype.kind not in 'iu' or keys.ndim != 1
            or (groups < 0).any() or (groups >= len(keys)).any() or len(set(keys)) != len(keys)):
        raise ValueError('Invalid image grouping')
    st = np.asarray(samples['st_samples'], np.float32)
    normals = np.asarray(samples['reference_normal_samples'])
    frames = np.asarray(samples['tangent_frame_samples'])
    if (st.shape != (len(colors), 2) or not np.isfinite(st).all()
            or (st < 0).any() or (st > 1).any()):
        raise ValueError('Image samples require finite Nx2 material coordinates in [0, 1]')
    if normals.shape != (len(colors), 3) or frames.shape != (len(colors), 3, 2):
        raise ValueError('Invalid validation frames')
    curvature = curvature_valid = None
    if settings.method in ('mlp6', 'mlp8'):
        if ('curvature_samples' in samples) != ('curvature_valid_mask' in samples):
            raise ValueError('Curvature samples and validity must be supplied together')
        if 'curvature_samples' in samples:
            curvature = np.asarray(samples['curvature_samples'], np.float32)
            curvature_valid = np.asarray(samples['curvature_valid_mask'], bool)
        else:
            curvature, curvature_valid = sample_curvatures_from_caches(samples, settings.work_dir)
        if curvature.shape != (len(colors), 3) or curvature_valid.shape != (len(colors),):
            raise ValueError('Curvature samples must be Nx3 with an N-element validity mask')
        curvature_valid &= np.isfinite(curvature).all(axis=-1)
        if not curvature_valid.all():
            raise ValueError(f'Every paired sample needs valid curvature for {settings.method}; '
                             f'{int((~curvature_valid).sum())}/{len(colors)} are invalid')

    if files.run is not None:
        stage_reference_chart(files.run, chart_path)
    else:
        _copy_atomically(chart_path, files.chart_file)
    metadata['coordinate_chart_file'] = files.chart_file.name
    grid_values = _reference_sampling_grid_values(settings, metadata, output_dir=files.directory)
    _save_npz(files.grid_file, **grid_values)
    base_report.update(chart_file=str(files.chart_file), chart_sha256=_sha256(files.chart_file),
                       grid_file=str(files.grid_file), grid_sha256=_sha256(files.grid_file))
    training_groups, held_out = _split_image_groups(groups, keys, settings.validation_fraction)
    held_mask = np.isin(groups, held_out)
    training_mask = ~held_mask
    common = dict(metadata, method=settings.method, validation_status='training_only',
                  training_images=keys[training_groups].tolist(), validation_images=keys[held_out].tolist())
    model = _train_model(settings.method, colors[training_mask], slopes[training_mask], groups[training_mask],
                         st[training_mask], None if curvature is None else curvature[training_mask],
                         None if curvature_valid is None else curvature_valid[training_mask], settings)
    model = replace(model, metadata=common)
    model.save(files.training_file, total_sample_count=int(training_mask.sum()))
    training_digest = _sha256(files.training_file)
    predicted = predict_normal_model(colors[held_mask], model, st=st[held_mask],
        curvature_per_mm=None if curvature is None else curvature[held_mask],
        curvature_valid_mask=None if curvature_valid is None else curvature_valid[held_mask])
    # Support estimates are diagnostics only. Every held-out point contributes.
    if not np.isfinite(predicted.slopes).all():
        raise ValueError('Model returned nonfinite held-out slopes')
    predicted_normals, vp = local_slopes_to_normals(predicted.slopes, normals[held_mask], frames[held_mask])
    true_normals, vt = local_slopes_to_normals(slopes[held_mask], normals[held_mask], frames[held_mask])
    if not (vp & vt).all():
        raise ValueError('Invalid validation normals')
    angles = np.rad2deg(np.arccos(np.clip(np.sum(predicted_normals*true_normals, axis=-1), -1, 1)))
    out_of_range = ((colors[held_mask] < model.color_min) | (colors[held_mask] > model.color_max)).any(axis=-1)
    validation_index = groups[held_mask]
    images = []
    for group in held_out:
        selected = validation_index == group
        rms = float(np.sqrt(np.mean(angles[selected]**2)))
        fraction = float(np.mean(out_of_range[selected]))
        unsupported = float(np.mean(~predicted.supported_mask[selected]))
        images.append(dict(image_key=keys[group], sample_count=int(selected.sum()), rms_degrees=rms,
                           out_of_range_fraction=fraction, median_degrees=float(np.median(angles[selected])),
                           p95_degrees=float(np.percentile(angles[selected], 95)),
                           unsupported_fraction=unsupported,
                           passed=(rms <= settings.validation_rms_degrees
                                   and fraction <= settings.maximum_out_of_range_fraction
                                   and unsupported <= settings.maximum_unsupported_fraction)))
    passed = all(row['passed'] for row in images)
    report = dict(base_report, status='passed' if passed else 'failed', model_file=str(files.training_file),
                  training_images=common['training_images'], validation_images=common['validation_images'],
                  split_role='historical_development_holdout_not_independent_test',
                  training_model_sha256=training_digest,
                  images=images, sample_count=int(held_mask.sum()), all_held_out_points_evaluated=True,
                  rms_degrees=float(np.sqrt(np.mean(angles**2))), median_degrees=float(np.median(angles)),
                  p95_degrees=float(np.percentile(angles, 95)), out_of_range_fraction=float(out_of_range.mean()),
                  unsupported_fraction=float(np.mean(~predicted.supported_mask)),
                  thresholds=dict(rms_degrees=settings.validation_rms_degrees,
                                  maximum_out_of_range_fraction=settings.maximum_out_of_range_fraction,
                                  maximum_unsupported_fraction=settings.maximum_unsupported_fraction))
    if not passed:
        _copy_atomically(files.training_file, files.model_file)
        report.update(model_file=str(files.model_file), model_sha256=_sha256(files.model_file),
                      model_role='outer_training_only_candidate')
        _save_json(files.report_file, report)
        return report

    final = _train_model(settings.method, colors, slopes, groups, st, curvature, curvature_valid, settings)
    report['model_file'] = str(files.model_file)
    final_metadata = dict(metadata, method=settings.method, validation_status='passed', validation=report)
    final = replace(final, metadata=final_metadata)
    final.save(files.model_file, total_sample_count=len(colors))
    report.update(model_sha256=_sha256(files.model_file), model_role='all_samples_after_validation')
    _export_bundle(files, settings, report, sample_path)
    _save_json(files.report_file, report)
    if files.run is not None:
        report['current_file'] = str(activate_run(files.run))
        _save_json(files.report_file, report)
    return report


class _LUTValidationFailure(ValueError):
    def __init__(self, message, report):
        self.report = report
        super().__init__(message)


def fit_and_validate_samples(sample_path, settings, *, raise_on_failure=True):
    """Train one model in an isolated run and activate only a passed image calibration."""
    files = _fit_files(settings)
    try:
        report = _fit_and_validate_samples(sample_path, settings, files)
    except (OSError, KeyError, ValueError, RuntimeError, ImportError) as exc:
        report = dict(status='failed', reason='input_or_fit_failed', message=str(exc), method=settings.method,
                      run_dir=str(files.directory), report_file=str(files.report_file),
                      training_file=str(files.training_file),
                      model_file=str(files.training_file) if files.training_file.is_file() else None, images=[])
        _save_json(files.report_file, report)
        if raise_on_failure:
            raise _LUTValidationFailure(f'{exc}; see {files.report_file}', report) from exc
        return report
    if report['status'] == 'failed' and raise_on_failure:
        raise _LUTValidationFailure(f'Whole-image validation failed; see {files.report_file}', report)
    return report


def load_manifest(settings):
    return json.loads((settings.work_dir/'manifest.json').read_text(encoding='utf-8'))


def _write_report(settings, reports):
    path = settings.work_dir/'verification'/'report.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['image_key', 'stage', 'status', 'reason', 'message', 'sample_count', 'marker_fit_count',
               'reprojection_rms_px', 'metric_max_error', 'background_contact_exclusion_margin_px',
               'background_fraction', 'background_median_max_abs',
               'boundary_valid_count', 'boundary_max_missing', 'sphere_boundary_rms', 'indentation_depth_mm']
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(reports)
    temporary.replace(path)


def run_prepare(config_path, settings):
    work_dir = settings.work_dir.resolve()
    output_files = {settings.input_file.resolve(), settings.output.resolve(),
                    settings.output.with_name(settings.output.stem+'_train.npz').resolve()}
    paths = {Path(path).resolve() for pattern in settings.images
             for path in glob.iglob(pattern, recursive=True) if Path(path).is_file()}
    # Exclude calibration outputs even when an image pattern matches them.
    lut_dir = settings.lut_dir.resolve() if settings.lut_dir is not None else None
    paths = sorted(str(path) for path in paths if not path.is_relative_to(work_dir)
                   and (lut_dir is None or not path.is_relative_to(lut_dir)) and path not in output_files)
    if not paths:
        raise ValueError('No calibration images matched images/--images')
    root = Path(os.path.commonpath([str(Path(path).parent) for path in paths]))
    settings = replace(settings, image_root=root)
    reference = prepare_reference(config_path, settings)
    manifest = dict(version=1, image_root=str(root), reference_image=str(reference.source_image),
                    coordinate_chart_id=reference.chart.chart_id, images=[])
    if settings.single_frame_markers:
        manifest['marker_matching'] = 'single_frame'
    reports = []
    for index, source in enumerate(paths):
        key = Path(os.path.relpath(source, root)).as_posix()
        row = dict(index=index, key=key, source_image=source, reference_image=str(reference.source_image))
        try:
            frame = prepare_image(config_path, settings, reference, source)
            frame.image_index = index
            _save_frame(settings.work_dir/'prepared'/f'{index:06d}.npz', frame)
            row.update(status='prepared', reason='')
        except FrameCalibrationError as exc:
            row.update(status='skip', reason=exc.code, message=str(exc))
        manifest['images'].append(row)
        reports.append(dict(image_key=key, stage='prepare', **{k: v for k, v in row.items() if k not in {'index', 'key'}}))
        outcome = ' | '.join(row[field] for field in ('status', 'message') if row.get(field))
        print(f'prepare {index+1}/{len(paths)} {key}: {outcome}', flush=True)
    _save_json(settings.work_dir/'manifest.json', manifest)
    _write_report(settings, reports)
    return manifest


def run_samples(settings):
    settings.require_sphere()
    manifest = load_manifest(settings)
    reference = load_frame_calibration_data(settings.work_dir/'reference_surface.npz', settings)
    regions = load_regions(settings.annotations_file, image_root=manifest['image_root'])
    records, reports = [], []
    for entry in manifest['images']:
        key = entry['key']
        row = dict(image_key=key, stage='samples', sample_count=0)
        annotation = regions.get(key)
        if entry['status'] == 'skip':
            row.update(status='skip', reason=entry['reason'])
        elif key in regions and regions[key].status == 'skip':
            row.update(status='skip', reason='', message='Annotation marked skip')
        else:
            try:
                frame = load_frame_calibration_data(settings.work_dir/'prepared'/f"{entry['index']:06d}.npz", settings)
                if annotation is None or annotation.status != 'accepted':
                    raise FrameCalibrationError('annotation_missing', 'No accepted contact_boundary')
                record = finalize_image(frame, reference, annotation, settings)
                _write_final_record(record, reference, settings)
                records.append(record)
                row.update(status='accepted', reason='', **record['metrics'])
            except FrameCalibrationError as exc:
                row.update(status='failed', reason=exc.code, message=str(exc), **exc.metrics)
        reports.append(row)
        print(f"samples {key}: {row['status']} {row.get('reason', '')}")
    _write_report(settings, reports)
    export_normal_samples(records, settings, settings.input_file)
    return records
