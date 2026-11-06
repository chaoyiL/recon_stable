"""Image calibration stages using the shared geometry and sampling utilities."""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import glob
import json
import math
import os
import numpy as np
from utils.config import load_config, load_contact_surface_offset_mm
from utils.image_io import load_linear_image, validate_linear_rgb
from utils.surface_coordinates import (ArcLengthChart, COORDINATE_KIND, FRAME_CONVENTION, load_arclength_chart,
                                       save_arclength_chart, build_material_sampling_grid)
from utils.surface_geometry import (OffsetSurface, ParametricSurface, compute_surface_geometry, normals_to_local_slopes,
                                    local_slopes_to_normals)
from utils.surface_projection import CameraModel, unproject_rays
from utils.surface_sampling import (sample_mask_to_surface,
                                    sample_linear_rgb_to_surface)
from utils.sphere_geometry import (intersect_sphere_rays, select_sphere_cap,
                                   sphere_normals)
from utils.sphere_contact import SphereSurfaceContact
from utils.color_difference import (CAMERA_RGB_TRANSFER_FUNCTION,
                                    require_camera_rgb_transfer, signed_color_difference)
from utils.region_annotations import load_regions, region_masks
from utils.reconstruction_interface import (load_material_coordinate_specs, MaterialCoordinateMap,
                                            MaterialCoordinateSystems)
from utils.normal_mapping import NormalCalibration, build_normal_lut, lookup_slopes_raw
from reconstruction.surface_reconstruction.surface import SurfaceModel, BicubicSurfaceAdapter
from reconstruction.surface_reconstruction.run_reconstruction import SurfaceReconstructionProvider, SurfaceReconstructionSession
from reconstruction.surface_reconstruction.markers import InnerRegion, marker_reference_sizes, build_marker_image_mask


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

    def __post_init__(self):
        if not np.isfinite(self.contact_offset_mm):
            raise ValueError('contact_surface.offset_mm must be finite')
        positive = (self.maximum_rms_angle_degrees, self.validation_rms_degrees)
        if not np.isfinite(positive).all() or min(positive) <= 0 or self.lut_size < 2:
            raise ValueError('Invalid LUT settings')
        if (not 0 < self.validation_fraction < 1 or not 0 <= self.background_max_abs
                or not np.isfinite(self.background_max_abs) or not 0 < self.background_min_fraction <= 1
                or not 0 <= self.maximum_out_of_range_fraction <= 1):
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
    raw = dict(load_config(config_path).get('normal_calibration', {}))
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
        lut_size=int(raw.get('lut_size', 64)),
        maximum_rms_angle_degrees=float(raw.get('maximum_rms_angle_degrees', 5)),
        **{key: float(checks[key]) for key in ('background_max_abs', 'background_min_fraction',
           'validation_fraction', 'validation_rms_degrees', 'maximum_out_of_range_fraction') if key in checks})


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
                  curvature_valid_mask=geometry['valid'])
    values.update(reference_marker_sizes=np.asarray(frame.reference_marker_sizes, np.float64),
                  marker_image_mask=np.asarray(frame.marker_image_mask, bool),
                  solver_options=np.asarray(json.dumps(frame.solver_options)),
                  format_version=np.asarray(4, np.int32),
                  contact_offset_mm=np.asarray(frame.contact_offset_mm, np.float64))
    return _save_npz(path, **values)


def _require_contact_offset(actual, expected, artifact_name):
    try:
        actual = float(actual)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{artifact_name} has no valid contact_offset_mm; rerun prepare and calibration') from exc
    if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-9, rtol=0):
        raise ValueError(
            f'{artifact_name} contact_offset_mm={actual!r} differs from configured '
            f'{expected!r}; rerun prepare and calibration')


def _frame_surface(model, coefficients, contact_offset_mm):
    return OffsetSurface(BicubicSurfaceAdapter(model, coefficients), contact_offset_mm)


def load_frame_calibration_data(path: str | Path, settings: CalibrationSettings) -> FrameCalibrationData:
    """Replay cached geometry; refresh marker photometry masks with current padding."""
    path = Path(path).resolve()
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(data['format_version'])
            if version == 4:
                if str(data['color_space']) != 'linear_rgb':
                    raise ValueError('Geometry cache requires linear RGB')
                require_camera_rgb_transfer({'rgb_transfer_function': str(data['rgb_transfer_function'])}, '几何缓存')
                linear_rgb = validate_linear_rgb(data['observed_linear_rgb'])
            else:
                raise ValueError('Geometry cache predates contact-surface support; rerun prepare')
            if 'contact_offset_mm' not in data:
                raise ValueError('Geometry cache has no contact_offset_mm; rerun prepare')
            contact_offset_mm = float(data['contact_offset_mm'])
            _require_contact_offset(contact_offset_mm, settings.contact_offset_mm, 'Geometry cache')
            chart_file = (path.parent/str(data['coordinate_chart_file'])).resolve()
            chart = load_arclength_chart(chart_file)
            if (str(data['coordinate_chart_id']) != chart.chart_id
                    or str(data['coordinate_kind']) != COORDINATE_KIND
                    or str(data['frame_convention']) != FRAME_CONVENTION):
                raise ValueError('Coordinate chart identity mismatch; rerun prepare')
            camera = CameraModel(data['camera_matrix'], data['dist_coeffs'], tuple(data['image_shape']))
            template = np.asarray(data['template_xy_mm'], np.float64)
            options = json.loads(str(data['solver_options']))
            model = SurfaceModel(template, camera.matrix, camera.distortion, options)
            coefficients = np.asarray(data['surface_coefficients'], np.float64)
            surface = _frame_surface(model, coefficients, contact_offset_mm)
            maps = {}
            for name in ('geometry', 'lightfield', 'observation'):
                mask_key = ('geometry_grid_valid_mask' if name == 'geometry' and 'geometry_grid_valid_mask' in data
                            else f'{name}_valid_mask')
                maps[name] = MaterialCoordinateMap(data[f'{name}_xyz'], data[f'{name}_uv'],
                    data[f'{name}_coordinates'], data[mask_key], data[f'{name}_st'], COORDINATE_KIND, chart.chart_id)
                expected_st, coordinate_valid = chart.to_st(maps[name].coordinates)
                if not coordinate_valid.all() or not np.allclose(expected_st, maps[name].st, atol=1e-6, rtol=0):
                    raise ValueError('Cached st/material coordinates differ from reference chart')
            marker_mask = build_marker_image_mask(linear_rgb, data['marker_pixels'],
                                                 data['marker_valid'], data['reference_marker_sizes'])
            return FrameCalibrationData(str(data['image_key']), int(data['image_index']),
                Path(str(data['source_image'])), linear_rgb, data['surface_mask'],
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
        coefficients.copy(), _frame_surface(session.model, coefficients, contact_offset_mm), contact_offset_mm,
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
    seed = provider.track_frame(reference.linear_rgb)
    if seed['error'] or not np.allclose(seed['tracking']['pixels'], reference.tracking['pixels'], atol=1e-5):
        raise FrameCalibrationError('tracking_failed', 'Reference marker identity changed; rerun prepare')
    result = provider.track_frame(image)
    if result['error']:
        raise FrameCalibrationError(result['info']['status'], result['error'])
    if result['tracking']['valid'].sum() < 32:
        raise FrameCalibrationError('tracking_failed', 'Too few tracked marker observations')
    try:
        systems = provider.fit_tracked_frame(load_material_coordinate_specs(config_path),
            result['inner_region'].mask, prior_coefficients=reference.coefficients)
    except (ValueError, RuntimeError) as exc:
        raise FrameCalibrationError('geometry_rejected', str(exc)) from exc
    root = settings.image_root or source.parent
    key = Path(os.path.relpath(source, root)).as_posix()
    return _frame_from_provider(provider, systems, image, source, key, -1, reference.reference_marker_sizes,
                                 provider.last_frame_result['coefficients'], 'initial')


def _coordinate_metadata(frame, output_path):
    return dict(coordinate_kind=COORDINATE_KIND, coordinate_chart_id=frame.chart.chart_id,
                coordinate_chart_file=os.path.relpath(frame.chart_file, Path(output_path).resolve().parent),
                frame_convention=FRAME_CONVENTION, residual_domain='material_grid', color_space='linear_rgb',
                rgb_transfer_function=CAMERA_RGB_TRANSFER_FUNCTION,
                contact_offset_mm=frame.contact_offset_mm)


def compute_frame_colors(final_frame, reference, settings):
    current, ref = final_frame.systems.observation, reference.systems.observation
    if (current.chart_id != ref.chart_id or current.coordinate_kind != COORDINATE_KIND
            or not np.array_equal(current.coordinates, ref.coordinates) or not np.allclose(current.st, ref.st, atol=1e-7, rtol=0)):
        raise FrameCalibrationError('coordinate_chart_invalid', 'Reference/current material identities differ')
    source_current = final_frame.surface_mask & ~final_frame.marker_image_mask & np.all(final_frame.linear_rgb < 1, axis=-1)
    source_ref = reference.surface_mask & ~reference.marker_image_mask & np.all(reference.linear_rgb < 1, axis=-1)
    observed = sample_linear_rgb_to_surface(final_frame.linear_rgb, current.uv, source_current)
    background = sample_linear_rgb_to_surface(reference.linear_rgb, ref.uv, source_ref)
    geometry_valid = current.valid_mask.copy()
    photo = geometry_valid & ref.valid_mask & observed.valid & background.valid
    delta, photo = signed_color_difference(observed.values, background.values, photo)
    return dict(reference_color_grid=background.values.astype(np.float32),
                observed_color_grid=observed.values.astype(np.float32), color_residual_grid=delta,
                geometry_valid_mask=geometry_valid, photometric_valid_mask=photo)


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
        surface=_frame_surface(provider.session.model, result['coefficients'], prepared.contact_offset_mm),
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


def finalize_image(prepared: FrameCalibrationData, reference: FrameCalibrationData, annotation,
                   settings: CalibrationSettings, *, diagnostic_preview=False) -> dict:
    """Return accepted calibration; optional diagnostics are attached to raised errors only."""
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
    colors = compute_frame_colors(final, reference, settings)
    outside = sample_mask_to_surface(~masks['exclusion'], final.systems.observation.uv)
    background_valid = colors['photometric_valid_mask'] & outside
    denominator = np.count_nonzero(colors['photometric_valid_mask'])
    fraction = float(background_valid.sum()/max(denominator, 1))
    median = (float(np.median(np.max(np.abs(colors['color_residual_grid'][background_valid]), axis=-1)))
              if background_valid.any() else None)
    metrics = dict(background_fraction=fraction, background_median_max_abs=median,
                   sample_count=0, reprojection_rms_px=final.fit_info.get('reprojection_rms_px'),
                   metric_max_error=final.fit_info.get('metric_max_error'),
                   marker_fit_count=int(final.marker_fit_valid.sum()))
    geometry = compute_surface_geometry(final.systems.observation.xyz, final.systems.observation.valid_mask)
    record = dict(frame=final, reference_image=str(reference.source_image), **colors, metrics=metrics,
                  contact_boundary_px=contact.boundary_px,
                  contact_center_parameters=contact.center_parameters,
                  indentation_depth_mm=annotation.indentation_depth_mm,
                  principal_curvatures_per_mm=geometry['principal_curvatures'].astype(np.float32),
                  curvature_valid_mask=geometry['valid'])
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
        residual_mm = np.linalg.norm(contact.boundary_xyz-contact.sphere.center, axis=1)-contact.sphere.radius_mm
        metrics.update(boundary_valid_count=len(contact.boundary_px), boundary_max_missing=0,
                       sphere_boundary_rms=float(np.sqrt(np.mean(residual_mm**2)) / contact.center_sigma_mm),
                       indentation_depth_mm=annotation.indentation_depth_mm)
        labels = make_sphere_labels(final, annotation, contact.sphere, settings,
                                    center_frame=contact.center_frame,
                                    center_sigma_mm=contact.center_sigma_mm, geometry=geometry)
        valid = (colors['geometry_valid_mask'] & colors['photometric_valid_mask']
                 & labels['contact_mask'] & labels['sphere_label_valid'])
        metrics['sample_count'] = int(valid.sum())
        record.update(**labels, sample_valid_mask=valid,
                      reference_normals=geometry['normal'].astype(np.float32),
                      tangent_frames=geometry['frame'].astype(np.float32))
        if not valid.any():
            raise FrameCalibrationError('no_valid_samples', 'No paired colors and sphere slopes')
    except FrameCalibrationError as error:
        if not diagnostic_preview:
            raise
        errors.append(error)
    if errors:
        error = errors[0]
        error.preview = dict(record, preview_errors=[str(item) for item in errors])
        error.display_message = '\n'.join(str(item) for item in errors)
        raise error
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
                                  'material_coordinates', 'st_samples', 'reference_normal_samples', 'tangent_frame_samples')}
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
    values = {key: np.concatenate(value) for key, value in fields.items()}
    values.update(format_version=1, image_paths=np.asarray(keys),
                  reference_image=records[0]['reference_image'],
                  sphere_radius_mm=settings.sphere_radius_mm, sphere_side=settings.sphere_side,
                  boundary_model=settings.boundary_model, **_coordinate_metadata(first, output_path))
    return _save_npz(output_path, **values)


def _reference_sampling_grid_values(settings, metadata):
    reference = load_frame_calibration_data(settings.work_dir/'reference_surface.npz', settings)
    expected = dict(coordinate_kind=COORDINATE_KIND, coordinate_chart_id=reference.chart.chart_id,
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
    values.update(reference_image=np.asarray(str(reference.source_image)),
                  coordinate_chart_file=np.asarray(os.path.relpath(reference.chart_file, settings.work_dir)),
                  contact_offset_mm=np.asarray(settings.contact_offset_mm, np.float64))
    return values


def update_material_sampling_grid(settings: CalibrationSettings) -> Path:
    """Regenerate the published image LUT's grid from its matching reference cache."""
    calibration = NormalCalibration.load(settings.output)
    if calibration.metadata.get('validation_status') != 'passed':
        raise ValueError('Sampling grid regeneration requires a validated image LUT')
    values = _reference_sampling_grid_values(settings, calibration.metadata)
    return _save_npz(settings.work_dir/'lut_material_grid.npz', **values)


def _publish_image_calibration(settings, calibration, total_sample_count):
    values = _reference_sampling_grid_values(settings, calibration.metadata)
    grid_path = settings.work_dir/'lut_material_grid.npz'
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    # Finish both files before replacing either published artifact. Each staged
    # file stays on its destination filesystem so replacement uses rename.
    with TemporaryDirectory(prefix='.normal-grid-', dir=settings.work_dir) as grid_dir, \
         TemporaryDirectory(prefix='.normal-lut-', dir=settings.output.parent) as lut_dir:
        staged_grid = _save_npz(Path(grid_dir)/grid_path.name, **values)
        staged_lut = calibration.save(Path(lut_dir)/settings.output.name, total_sample_count=total_sample_count)
        staged_grid.replace(grid_path)
        staged_lut.replace(settings.output)


def _fit_and_validate_samples(sample_path, settings):
    with np.load(sample_path, allow_pickle=False) as data:

        # 颜色，坡度加载以及维度检查
        colors = np.asarray(data['color_residual_samples'], np.float32)
        slopes = np.asarray(data['slope_samples'], np.float32)
        if colors.ndim != 2 or colors.shape[1] != 3 or slopes.shape != (len(colors), 2) or not len(colors):
            raise ValueError('Expected nonempty paired Nx3 colors and Nx2 slopes')
        if not np.isfinite(colors).all() or not np.isfinite(slopes).all():
            raise ValueError('Training samples must be finite')
        if 'contact_offset_mm' not in data:
            raise ValueError('Training samples predate contact-surface calibration; rerun prepare and samples')
        _require_contact_offset(data['contact_offset_mm'], settings.contact_offset_mm, 'Training samples')
        if 'image_index' not in data:
            calibration = build_normal_lut(colors, slopes, **settings.lut_options())
            calibration = replace(calibration, metadata=dict(
                validation_status='unvalidated_external_samples',
                contact_offset_mm=settings.contact_offset_mm))
            calibration.save(settings.output, total_sample_count=len(colors))
            report = dict(status='unvalidated_external_samples', total_sample_count=len(colors))
            _save_json(settings.work_dir/'verification'/'validation.json', report)
            return report
        if int(data['format_version']) != 1:
            raise ValueError('Grouped image samples require format_version=1')
        require_camera_rgb_transfer(
            {'rgb_transfer_function': str(data['rgb_transfer_function']) if 'rgb_transfer_function' in data else None},
            '图像标定样本',
        )
        
        # 超参数检查
        required = ('coordinate_kind', 'coordinate_chart_id', 'coordinate_chart_file', 'frame_convention',
                    'residual_domain', 'color_space', 'rgb_transfer_function', 'reference_image',
                    'sphere_radius_mm', 'sphere_side', 'boundary_model', 'contact_offset_mm')
        metadata = {key: data[key].item() for key in required}
        _require_contact_offset(metadata['contact_offset_mm'], settings.contact_offset_mm, 'Training samples')
        for key, expected in dict(coordinate_kind=COORDINATE_KIND, frame_convention=FRAME_CONVENTION,
                                  residual_domain='material_grid', color_space='linear_rgb',
                                  boundary_model='zero_boundary_displacement').items():
            if metadata[key] != expected:
                raise ValueError(f'Invalid sample declaration: {key}')
        
        # 选取验证样本与标定样本？
        image_index = np.asarray(data['image_index'])
        keys = np.asarray(data['image_paths']).astype(str)
        if (image_index.shape != (len(colors),) or image_index.dtype.kind not in 'iu'
                or (image_index < 0).any() or (image_index >= len(keys)).any() or len(set(keys)) != len(keys)):
            raise ValueError('Invalid image grouping')
        active = sorted(set(image_index.tolist()), key=lambda i: keys[i])
        if len(active) < 2:
            raise ValueError('Whole-image validation requires at least two images with valid samples')
        shuffled = np.array(active)
        np.random.default_rng(42).shuffle(shuffled)
        count = min(len(active)-1, max(1, math.ceil(settings.validation_fraction*len(active))))
        held_out = shuffled[:count]
        validation_mask = np.isin(image_index, held_out)
        training_mask = ~validation_mask

        # 标定
        calibration = build_normal_lut(colors[training_mask], slopes[training_mask], **settings.lut_options())
        common = dict(metadata, validation_status='training_only',
                      training_images=keys[shuffled[count:]].tolist(), validation_images=keys[held_out].tolist())
        chart_path = (Path(sample_path).resolve().parent/metadata['coordinate_chart_file']).resolve()
        train_path = settings.output.with_name(settings.output.stem+'_train.npz')
        common['coordinate_chart_file'] = os.path.relpath(chart_path, train_path.resolve().parent)
        calibration = replace(calibration, metadata=common)
        calibration.save(train_path, total_sample_count=int(training_mask.sum()))
        predicted, _, _ = lookup_slopes_raw(colors[validation_mask], calibration)
        if 'reference_normal_samples' not in data or 'tangent_frame_samples' not in data:
            raise ValueError('Image sample files require shared reference normals and tangent frames')
        normals = np.asarray(data['reference_normal_samples'])
        frames = np.asarray(data['tangent_frame_samples'])
        if normals.shape != (len(colors), 3) or frames.shape != (len(colors), 3, 2):
            raise ValueError('Invalid validation frames')
        predicted_normals, vp = local_slopes_to_normals(predicted, normals[validation_mask], frames[validation_mask])
        true_normals, vt = local_slopes_to_normals(slopes[validation_mask], normals[validation_mask], frames[validation_mask])
        if not (vp & vt).all():
            raise ValueError('Invalid validation normals')
        angles = np.rad2deg(np.arccos(np.clip(np.sum(predicted_normals*true_normals, axis=-1), -1, 1)))
        out_of_range = ((colors[validation_mask] < calibration.color_min) | (colors[validation_mask] > calibration.color_max)).any(axis=-1)
        validation_index = image_index[validation_mask]
        images = []
        for i in held_out:
            selected = validation_index == i
            rms = float(np.sqrt(np.mean(angles[selected]**2)))
            fraction = float(np.mean(out_of_range[selected]))
            images.append(dict(image_key=keys[i], rms_degrees=rms, out_of_range_fraction=fraction,
                               median_degrees=float(np.median(angles[selected])), p95_degrees=float(np.percentile(angles[selected], 95)),
                               passed=rms <= settings.validation_rms_degrees and fraction <= settings.maximum_out_of_range_fraction))
        passed = all(row['passed'] for row in images)
        report = dict(status='passed' if passed else 'failed', training_images=common['training_images'],
                      validation_images=common['validation_images'], images=images,
                      median_degrees=float(np.median(angles)), p95_degrees=float(np.percentile(angles, 95)),
                      out_of_range_fraction=float(out_of_range.mean()),
                      thresholds=dict(rms_degrees=settings.validation_rms_degrees,
                                      maximum_out_of_range_fraction=settings.maximum_out_of_range_fraction))
        _save_json(settings.work_dir/'verification'/'validation.json', report)
        if not passed:
            raise _LUTValidationFailure(f'Whole-image LUT validation failed; see {settings.work_dir / "verification/validation.json"}', report)
        final = build_normal_lut(colors, slopes, **settings.lut_options())
        metadata.update(validation_status='passed', validation=report,
                        coordinate_chart_file=os.path.relpath(chart_path, settings.output.resolve().parent))
        _publish_image_calibration(settings, replace(final, metadata=metadata), len(colors))
        return report



class _LUTValidationFailure(ValueError):
    def __init__(self, message, report):
        self.report = report
        super().__init__(message)


def fit_and_validate_samples(sample_path, settings):
    try:
        report = _fit_and_validate_samples(sample_path, settings)
    except (OSError, KeyError, ValueError) as exc:
        if not isinstance(exc, _LUTValidationFailure):
            _save_json(settings.work_dir/'verification'/'validation.json',
                       dict(status='failed', reason='input_or_fit_failed', message=str(exc)))
        raise
    return report


def load_manifest(settings):
    return json.loads((settings.work_dir/'manifest.json').read_text(encoding='utf-8'))


def _write_report(settings, reports):
    path = settings.work_dir/'verification'/'report.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['image_key', 'stage', 'status', 'reason', 'message', 'sample_count', 'marker_fit_count',
               'reprojection_rms_px', 'metric_max_error', 'background_fraction', 'background_median_max_abs',
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
    paths = sorted(str(path) for path in paths
                   if not path.is_relative_to(work_dir) and path not in output_files)
    if not paths:
        raise ValueError('No calibration images matched images/--images')
    root = Path(os.path.commonpath([str(Path(path).parent) for path in paths]))
    settings = replace(settings, image_root=root)
    reference = prepare_reference(config_path, settings)
    manifest = dict(version=1, image_root=str(root), reference_image=str(reference.source_image),
                    coordinate_chart_id=reference.chart.chart_id, images=[])
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
        print(f"prepare {index+1}/{len(paths)} {key}: {row['status']}")
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
