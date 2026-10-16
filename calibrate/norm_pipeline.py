"""Image calibration stages. Geometry and sampling live in reusable utils modules."""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import csv
import glob
import json
import math
import os
import cv2
import numpy as np
from utils.config import load_config
from utils.surface_coordinates import (ArcLengthChart, COORDINATE_KIND, FRAME_CONVENTION, load_arclength_chart,
                                       save_arclength_chart)
from utils.surface_geometry import (ParametricSurface, compute_surface_geometry, normals_to_local_slopes,
                                    local_slopes_to_normals)
from utils.surface_projection import CameraModel, invert_surface_projection, unproject_rays
from utils.surface_sampling import (bilinear_sample, sample_mask_to_surface,
                                    sample_linear_rgb_to_surface)
from utils.sphere_geometry import (fit_sphere_center, intersect_sphere_rays, select_sphere_cap,
                                   sphere_normals)
from utils.color_difference import bgr8_to_linear_rgb, signed_color_difference
from utils.region_annotations import BOUNDARY_SAMPLE_COUNT, load_regions, region_boundary_points, region_masks
from utils.reconstruction_interface import (load_material_coordinate_specs, MaterialCoordinateMap,
                                            MaterialCoordinateSystems)
from utils.normal_mapping import NormalCalibration, build_normal_lut, lookup_slopes_raw
from reconstruction.surface_reconstruction.surface import SurfaceModel, BicubicSurfaceAdapter
from reconstruction.surface_reconstruction.run_reconstruction import SurfaceReconstructionProvider, SurfaceReconstructionSession
from reconstruction.surface_reconstruction.markers import InnerRegion, marker_reference_sizes, build_marker_image_mask


CONTACT_INNER_MARGIN_PX = 3
CONTACT_OUTER_MARGIN_PX = 20
MINIMUM_BOUNDARY_VALID_COUNT = 96
MAXIMUM_BOUNDARY_MISSING_RUN = 32


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
    images: tuple[str, ...] = ()
    reference_image: Path | None = None
    image_root: Path | None = None
    sphere_radius_mm: float | None = None
    sphere_side: str | None = None
    boundary_model: str | None = None
    lut_size: int = 64
    minimum_samples_per_node: int = 5
    maximum_rms_angle_degrees: float = 5.
    background_max_abs: float = .01
    background_min_fraction: float = .1
    validation_fraction: float = .2
    validation_rms_degrees: float = 5.
    maximum_out_of_range_fraction: float = .05

    def __post_init__(self):
        positive = (self.maximum_rms_angle_degrees, self.validation_rms_degrees)
        if not np.isfinite(positive).all() or min(positive) <= 0 or self.lut_size < 2 or self.minimum_samples_per_node < 1:
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
        return dict(size=self.lut_size, minimum_samples_per_node=self.minimum_samples_per_node,
                    maximum_rms_angle_degrees=self.maximum_rms_angle_degrees)


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
        images=patterns, reference_image=_path(reference, base, 'reference_image') if reference else None,
        sphere_radius_mm=raw.get('sphere_radius_mm'), sphere_side=raw.get('sphere_side'),
        boundary_model=raw.get('boundary_model'),
        lut_size=int(raw.get('lut_size', 64)), minimum_samples_per_node=int(raw.get('minimum_samples_per_node', 5)),
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
    original_bgr: np.ndarray
    surface_mask: np.ndarray
    tracking: dict
    template_xy_mm: np.ndarray
    reference_marker_sizes: np.ndarray
    marker_image_mask: np.ndarray
    coefficients: np.ndarray
    surface: ParametricSurface
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
    provider.coordinate_chart = frame.chart
    provider.coordinate_chart_file = frame.chart_file
    provider.last_frame_result = dict(tracking=frame.tracking,
        inner_region=InnerRegion((0, 0, *session.expected_size), np.empty((0, 1, 2), int), frame.surface_mask),
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
    save_material_coordinates(path, frame.systems, frame.surface_mask, frame.original_bgr, provider,
        surface_coefficients=frame.coefficients, geometry_stage=frame.geometry_stage,
        metadata=dict(image_key=frame.image_key, image_index=frame.image_index, source_image=str(frame.source_image)))
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    values.update(reference_marker_sizes=np.asarray(frame.reference_marker_sizes, np.float64),
                  marker_image_mask=np.asarray(frame.marker_image_mask, bool),
                  solver_options=np.asarray(json.dumps(frame.solver_options)))
    return _save_npz(path, **values)


def load_frame_calibration_data(path: str | Path) -> FrameCalibrationData:
    """Replay cached geometry; refresh marker photometry masks with current padding."""
    path = Path(path).resolve()
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data['format_version']) != 2:
                raise ValueError('Geometry cache requires version 2; rerun prepare')
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
            surface = BicubicSurfaceAdapter(model, coefficients)
            maps = {}
            for name in ('geometry', 'lightfield', 'observation'):
                mask_key = ('geometry_grid_valid_mask' if name == 'geometry' and 'geometry_grid_valid_mask' in data
                            else f'{name}_valid_mask')
                maps[name] = MaterialCoordinateMap(data[f'{name}_xyz'], data[f'{name}_uv'],
                    data[f'{name}_coordinates'], data[mask_key], data[f'{name}_st'], COORDINATE_KIND, chart.chart_id)
                expected_st, coordinate_valid = chart.to_st(maps[name].coordinates)
                if not coordinate_valid.all() or not np.allclose(expected_st, maps[name].st, atol=1e-6, rtol=0):
                    raise ValueError('Cached st/material coordinates differ from reference chart')
            marker_mask = build_marker_image_mask(data['observed_bgr'], data['marker_pixels'],
                                                 data['marker_valid'], data['reference_marker_sizes'])
            return FrameCalibrationData(str(data['image_key']), int(data['image_index']),
                Path(str(data['source_image'])), data['observed_bgr'], data['surface_mask'],
                dict(pixels=data['marker_pixels'], valid=data['marker_valid'], confidence=data['marker_confidence']),
                template, data['reference_marker_sizes'], marker_mask, coefficients, surface,
                camera, chart, chart_file, MaterialCoordinateSystems(**maps), str(data['geometry_stage']),
                json.loads(str(data['fit_info'])), options,
                data['marker_fit_valid'] if 'marker_fit_valid' in data else None)
    except (KeyError, ValueError) as exc:
        raise FrameCalibrationError('coordinate_chart_invalid', str(exc)) from exc


def _frame_from_provider(provider, systems, image, source, key, index, sizes, coefficients, stage):
    result = provider.last_frame_result
    session = provider.session
    return FrameCalibrationData(key, index, Path(source), image, result['inner_region'].mask,
        result['tracking'], session.xy.copy(), sizes,
        build_marker_image_mask(image, result['tracking']['pixels'], result['tracking']['valid'], sizes),
        coefficients.copy(), BicubicSurfaceAdapter(session.model, coefficients),
        CameraModel(session.camera, session.distortion, image.shape[:2]), provider.coordinate_chart,
        provider.coordinate_chart_file, systems, stage, dict(result['info']), dict(session.model.options),
        result.get('marker_fit_valid'))


def prepare_reference(config_path: str | Path, settings: CalibrationSettings) -> FrameCalibrationData:
    from reconstruction.get_surface import build_reconstruction_provider
    if settings.reference_image is None:
        raise ValueError('reference_image must name the fixed background for this batch')
    image = cv2.imread(str(settings.reference_image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read reference_image: {settings.reference_image}')
    provider = build_reconstruction_provider(config_path)
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
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FrameCalibrationError('tracking_failed', f'Cannot read {source}')
    provider = build_reconstruction_provider(config_path)
    if (not np.array_equal(provider.session.xy, reference.template_xy_mm)
            or not np.array_equal(provider.session.camera, reference.camera.matrix)
            or not np.array_equal(provider.session.distortion, reference.camera.distortion)):
        raise FrameCalibrationError('coordinate_chart_invalid', 'Camera or physical template changed since reference preparation')
    provider.bind_coordinate_chart(reference.chart)
    provider.coordinate_chart_file = reference.chart_file
    seed = provider.track_frame(reference.original_bgr)
    if seed['error'] or not np.allclose(seed['tracking']['pixels'], reference.tracking['pixels'], atol=1e-5):
        raise FrameCalibrationError('tracking_failed', 'Reference marker identity changed; rerun prepare')
    result = provider.track_frame(image)
    if result['error']:
        raise FrameCalibrationError(result['info']['status'], result['error'])
    if result['tracking']['valid'].sum() < 32:
        raise FrameCalibrationError('tracking_failed', 'Too few tracked marker observations')
    try:
        systems = provider.sample_coefficients(reference.coefficients, load_material_coordinate_specs(config_path))
    except ValueError as exc:
        raise FrameCalibrationError('coordinate_chart_invalid', str(exc)) from exc
    root = settings.image_root or source.parent
    key = Path(os.path.relpath(source, root)).as_posix()
    return _frame_from_provider(provider, systems, image, source, key, -1, reference.reference_marker_sizes,
                                 reference.coefficients, 'preview_reference')


def _coordinate_metadata(frame, output_path):
    return dict(coordinate_kind=COORDINATE_KIND, coordinate_chart_id=frame.chart.chart_id,
                coordinate_chart_file=os.path.relpath(frame.chart_file, Path(output_path).resolve().parent),
                frame_convention=FRAME_CONVENTION, residual_domain='material_grid', color_space='linear_rgb')


def compute_frame_colors(final_frame, reference, settings):
    current, ref = final_frame.systems.observation, reference.systems.observation
    if (current.chart_id != ref.chart_id or current.coordinate_kind != COORDINATE_KIND
            or not np.array_equal(current.coordinates, ref.coordinates) or not np.allclose(current.st, ref.st, atol=1e-7, rtol=0)):
        raise FrameCalibrationError('coordinate_chart_invalid', 'Reference/current material identities differ')
    source_current = final_frame.surface_mask & ~final_frame.marker_image_mask & np.all(final_frame.original_bgr < 255, axis=-1)
    source_ref = reference.surface_mask & ~reference.marker_image_mask & np.all(reference.original_bgr < 255, axis=-1)
    observed = sample_linear_rgb_to_surface(bgr8_to_linear_rgb(final_frame.original_bgr), current.uv, source_current)
    background = sample_linear_rgb_to_surface(bgr8_to_linear_rgb(reference.original_bgr), ref.uv, source_ref)
    geometry_valid = current.valid_mask.copy()
    photo = geometry_valid & ref.valid_mask & observed.valid & background.valid
    delta, photo = signed_color_difference(observed.values, background.values, photo)
    return dict(reference_color_grid=background.values.astype(np.float32),
                observed_color_grid=observed.values.astype(np.float32), color_residual_grid=delta,
                geometry_valid_mask=geometry_valid, photometric_valid_mask=photo)


def _max_cyclic_missing(valid):
    valid = np.asarray(valid, bool)
    if not valid.any():
        return len(valid)
    missing = np.tile(~valid, 2)
    maximum = run = 0
    for absent in missing:
        run = run+1 if absent else 0
        maximum = max(maximum, run)
    return min(maximum, len(valid))


def fit_ball_center(frame, annotation, settings):
    grid = frame.systems.observation
    boundary = invert_surface_projection(region_boundary_points(annotation, BOUNDARY_SAMPLE_COUNT), frame.surface, frame.camera, grid)
    if boundary.valid.sum() < MINIMUM_BOUNDARY_VALID_COUNT or _max_cyclic_missing(boundary.valid) > MAXIMUM_BOUNDARY_MISSING_RUN:
        raise FrameCalibrationError('sphere_fit_failed', 'Contact boundary has insufficient or discontinuous inverse projections')
    center = invert_surface_projection(np.asarray(annotation.center_px)[None], frame.surface, frame.camera, grid)
    if not center.valid[0]:
        raise FrameCalibrationError('sphere_fit_failed', 'Contact center has no unique surface inverse')
    geometry = compute_surface_geometry(grid.xyz, grid.valid_mask)
    center_st, ok = frame.chart.to_st(center.parameters)
    if not ok[0]:
        raise FrameCalibrationError('coordinate_chart_invalid', 'Contact center outside reference chart')
    h, w = grid.valid_mask.shape
    xy = center_st[0]*[w-1, h-1]
    e1_sample = bilinear_sample(geometry['frame'][..., :, 0], xy, geometry['valid'])
    n_sample = bilinear_sample(geometry['normal'], xy, geometry['valid'])
    if not e1_sample.valid or not n_sample.valid:
        raise FrameCalibrationError('sphere_fit_failed', 'Contact center has no valid local frame')
    n0 = n_sample.values/np.linalg.norm(n_sample.values)
    e1 = e1_sample.values-n0*np.dot(e1_sample.values, n0)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n0, e1)
    center_frame = (center.xyz[0], e1, e2, n0)
    try:
        sphere, rms = fit_sphere_center(boundary.xyz[boundary.valid], boundary.sigma_mm[boundary.valid],
                                        settings.sphere_radius_mm, center_frame, settings.side)
    except ValueError as exc:
        raise FrameCalibrationError('sphere_fit_failed', str(exc)) from exc
    metrics = dict(boundary_valid_count=int(boundary.valid.sum()),
                   boundary_max_missing=_max_cyclic_missing(boundary.valid), sphere_boundary_rms=rms)
    return sphere, center_frame, float(center.sigma_mm[0]), geometry, metrics


def make_sphere_labels(frame, annotation, sphere, settings, *, center_frame=None,
                       center_sigma_mm=None, geometry=None):
    if center_frame is None:
        _, center_frame, center_sigma_mm, geometry, _ = fit_ball_center(frame, annotation, settings)
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
    masks = region_masks(annotation, frame.original_bgr.shape, inner_margin_px=CONTACT_INNER_MARGIN_PX, outer_margin_px=CONTACT_OUTER_MARGIN_PX)
    interior = sample_mask_to_surface(masks['interior'], grid.uv)
    valid = (ray_valid & roots_valid[..., root_index] & finite & geometry['valid']
             & ((normals @ n0) > .1) & interior)
    return dict(slope_grid=slopes.astype(np.float32), normal_gt_grid=normals.astype(np.float32),
                sphere_label_valid=valid, contact_interior=interior,
                sphere_center_xyz=sphere.center, sphere_radius_mm=sphere.radius_mm,
                sphere_root_sign=root_sign, sphere_side=settings.sphere_side,
                boundary_model=settings.boundary_model)


def finalize_image(prepared: FrameCalibrationData, reference: FrameCalibrationData, annotation,
                   settings: CalibrationSettings, *, diagnostic_preview=False) -> dict:
    """Return accepted calibration; optional diagnostics are attached to raised errors only."""
    settings.require_sphere()
    if annotation is None or annotation.semantics != 'contact_boundary':
        raise FrameCalibrationError('annotation_missing', 'An explicit contact_boundary annotation is required')
    if prepared.chart.chart_id != reference.chart.chart_id or not np.array_equal(prepared.template_xy_mm, reference.template_xy_mm):
        raise FrameCalibrationError('coordinate_chart_invalid', 'Prepared/reference chart or template mismatch')
    if (not np.array_equal(prepared.camera.matrix, reference.camera.matrix)
            or not np.array_equal(prepared.camera.distortion, reference.camera.distortion)):
        raise FrameCalibrationError('coordinate_chart_invalid', 'Prepared/reference camera mismatch')
    try:
        masks = region_masks(annotation, prepared.original_bgr.shape, inner_margin_px=CONTACT_INNER_MARGIN_PX, outer_margin_px=CONTACT_OUTER_MARGIN_PX)
    except ValueError as exc:
        raise FrameCalibrationError('annotation_missing', str(exc)) from exc
    provider = _provider_from_frame(prepared)
    try:
        systems = provider.fit_tracked_frame(_specs(prepared), prepared.surface_mask, masks['exclusion'], reference.coefficients)
    except (ValueError, RuntimeError) as exc:
        raise FrameCalibrationError('geometry_rejected', str(exc)) from exc
    result = provider.last_frame_result
    final = replace(prepared, coefficients=result['coefficients'],
                    surface=BicubicSurfaceAdapter(provider.session.model, result['coefficients']),
                    systems=systems, geometry_stage='final', fit_info=result['info'], marker_fit_valid=result['marker_fit_valid'])
    colors = compute_frame_colors(final, reference, settings)
    outside = sample_mask_to_surface(~masks['exclusion'], systems.observation.uv)
    background_valid = colors['photometric_valid_mask'] & outside
    denominator = np.count_nonzero(colors['photometric_valid_mask'])
    fraction = float(background_valid.sum()/max(denominator, 1))
    median = (float(np.median(np.max(np.abs(colors['color_residual_grid'][background_valid]), axis=-1)))
              if background_valid.any() else None)
    metrics = dict(background_fraction=fraction, background_median_max_abs=median,
                   sample_count=0, reprojection_rms_px=final.fit_info.get('reprojection_rms_px'),
                   metric_max_error=final.fit_info.get('metric_max_error'),
                   marker_fit_count=int(final.marker_fit_valid.sum()))
    record = dict(frame=final, reference_image=str(reference.source_image), **colors, metrics=metrics)
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
        sphere, center_frame, sigma, geometry, sphere_metrics = fit_ball_center(final, annotation, settings)
        metrics.update(sphere_metrics)
        labels = make_sphere_labels(final, annotation, sphere, settings, center_frame=center_frame,
                                    center_sigma_mm=sigma, geometry=geometry)
        valid = (colors['geometry_valid_mask'] & colors['photometric_valid_mask']
                 & labels['contact_interior'] & labels['sphere_label_valid'])
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


def _fit_and_validate_samples(sample_path, settings):
    with np.load(sample_path, allow_pickle=False) as data:
        colors = np.asarray(data['color_residual_samples'], np.float32)
        slopes = np.asarray(data['slope_samples'], np.float32)
        if colors.ndim != 2 or colors.shape[1] != 3 or slopes.shape != (len(colors), 2) or not len(colors):
            raise ValueError('Expected nonempty paired Nx3 colors and Nx2 slopes')
        if not np.isfinite(colors).all() or not np.isfinite(slopes).all():
            raise ValueError('Training samples must be finite')
        if 'image_index' not in data:
            calibration = build_normal_lut(colors, slopes, **settings.lut_options())
            calibration = replace(calibration, metadata=dict(validation_status='unvalidated_external_samples'))
            calibration.save(settings.output, total_sample_count=len(colors))
            report = dict(status='unvalidated_external_samples', total_sample_count=len(colors))
            _save_json(settings.work_dir/'verification'/'validation.json', report)
            return report
        if int(data['format_version']) != 1:
            raise ValueError('Grouped image samples require format_version=1')
        required = ('coordinate_kind', 'coordinate_chart_id', 'coordinate_chart_file', 'frame_convention',
                    'residual_domain', 'color_space', 'reference_image', 'sphere_radius_mm', 'sphere_side', 'boundary_model')
        metadata = {key: data[key].item() for key in required}
        for key, expected in dict(coordinate_kind=COORDINATE_KIND, frame_convention=FRAME_CONVENTION,
                                  residual_domain='material_grid', color_space='linear_rgb',
                                  boundary_model='zero_boundary_displacement').items():
            if metadata[key] != expected:
                raise ValueError(f'Invalid sample declaration: {key}')
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
        replace(final, metadata=metadata).save(settings.output, total_sample_count=len(colors))
        return report



class _LUTValidationFailure(ValueError):
    def __init__(self, message, report):
        self.report = report
        super().__init__(message)


def fit_and_validate_samples(sample_path, settings):
    try:
        return _fit_and_validate_samples(sample_path, settings)
    except (OSError, KeyError, ValueError) as exc:
        if not isinstance(exc, _LUTValidationFailure):
            _save_json(settings.work_dir/'verification'/'validation.json',
                       dict(status='failed', reason='input_or_fit_failed', message=str(exc)))
        raise


def load_manifest(settings):
    return json.loads((settings.work_dir/'manifest.json').read_text(encoding='utf-8'))


def _write_report(settings, reports):
    path = settings.work_dir/'verification'/'report.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['image_key', 'stage', 'status', 'reason', 'message', 'sample_count', 'marker_fit_count',
               'reprojection_rms_px', 'metric_max_error', 'background_fraction', 'background_median_max_abs',
               'boundary_valid_count', 'boundary_max_missing', 'sphere_boundary_rms']
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(reports)
    temporary.replace(path)


def run_prepare(config_path, settings):
    paths = sorted({str(Path(path).resolve()) for pattern in settings.images for path in glob.glob(pattern) if Path(path).is_file()})
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
        frame = None
        try:
            frame = prepare_image(config_path, settings, reference, source)
            frame.image_index = index
            _save_frame(settings.work_dir/'prepared'/f'{index:06d}.npz', frame)
            row.update(status='prepared', reason='')
        except FrameCalibrationError as exc:
            row.update(status='skip', reason=exc.code, message=str(exc))
        from calibrate.manual_norm_regions import save_status_preview
        save_status_preview(frame, source, None, settings.work_dir/'verification'/f'{index:06d}.png',
                            row.get('message', 'preview_reference'))
        manifest['images'].append(row)
        reports.append(dict(image_key=key, stage='prepare', **{k: v for k, v in row.items() if k not in {'index', 'key'}}))
        print(f"prepare {index+1}/{len(paths)} {key}: {row['status']}")
    _save_json(settings.work_dir/'manifest.json', manifest)
    _write_report(settings, reports)
    return manifest


def run_samples(settings):
    settings.require_sphere()
    manifest = load_manifest(settings)
    reference = load_frame_calibration_data(settings.work_dir/'reference_surface.npz')
    regions = load_regions(settings.annotations_file, image_root=manifest['image_root'])
    records, reports = [], []
    for entry in manifest['images']:
        key = entry['key']
        row = dict(image_key=key, stage='samples', sample_count=0)
        frame = None
        annotation = regions.get(key)
        if entry['status'] == 'skip':
            row.update(status='skip', reason=entry['reason'])
        elif key in regions and regions[key].status == 'skip':
            row.update(status='skip', reason='', message='Annotation marked skip')
        else:
            try:
                frame = load_frame_calibration_data(settings.work_dir/'prepared'/f"{entry['index']:06d}.npz")
                if annotation is None or annotation.status != 'accepted':
                    raise FrameCalibrationError('annotation_missing', 'No accepted contact_boundary')
                record = finalize_image(frame, reference, annotation, settings)
                _write_final_record(record, reference, settings)
                from calibrate.manual_norm_regions import save_verification_preview
                save_verification_preview(record, annotation, settings.work_dir/'verification'/f"{entry['index']:06d}.png")
                records.append(record)
                row.update(status='accepted', reason='', **record['metrics'])
            except FrameCalibrationError as exc:
                row.update(status='failed', reason=exc.code, message=str(exc), **exc.metrics)
        if row['status'] != 'accepted':
            from calibrate.manual_norm_regions import save_status_preview
            save_status_preview(frame, entry['source_image'], annotation,
                                settings.work_dir/'verification'/f"{entry['index']:06d}.png",
                                row.get('message', row.get('reason', 'skip')))
        reports.append(row)
        print(f"samples {key}: {row['status']} {row.get('reason', '')}")
    _write_report(settings, reports)
    export_normal_samples(records, settings, settings.input_file)
    return records
