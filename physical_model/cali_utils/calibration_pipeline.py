"""Analytic sphere scan, signed RGB-to-slope LUT fitting, and pose validation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import copy
import csv
import hashlib
import json
import math
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from physical_model.model_utils.config import load_config, validate_config
from physical_model.model_utils.curve_geometry import GEOMETRY_MODEL, LIGHT_MODEL
from physical_model.model_utils.geometry import LIGHT_GEOMETRY_MODEL
from physical_model.model_utils.observation import IMAGE_ENCODING, OBSERVATION_MODEL_ID
from physical_model.model_utils.surface_coordinates import bending_radius, normal_direction
from utils.normal_mapping import NormalCalibration, build_normal_lut, lookup_slopes_raw
from .calibration_samples import (
    MaterialGrid,
    base_geometry,
    build_material_grid,
    calibration_declarations,
    compact_frame_samples,
    image_space_diagnostics,
    render_reference,
    sample_pose,
)
from .calibration_scan import ScanPose, build_scan_poses


@dataclass(frozen=True)
class CalibrationSettings:
    output_dir: Path
    lut_size: int
    maximum_rms_angle_degrees: float
    validation_fraction: float
    validation_rms_degrees: float
    maximum_out_of_range_fraction: float
    minimum_samples_per_pose: int
    validation_seed: int
    difference_gain: float
    visualization_enabled: bool
    contact_margin_mm: float


def _positive(value, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) \
            or (value < 0 if allow_zero else value <= 0):
        comparator = 'nonnegative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be finite and {comparator}')
    return float(value)


def _unique_run_dir() -> Path:
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    return Path(__file__).resolve().parents[1] / 'lut' / run


def parse_calibration_settings(config: dict, config_path: Path,
                               output_override: Path | None = None) -> CalibrationSettings:
    raw = config.get('calibration')
    if not isinstance(raw, dict):
        raise ValueError('Configuration requires a calibration object')
    allowed = {
        'calibration': {'scan', 'material_grid', 'sampling', 'lut', 'validation',
                        'visualization', 'output_dir'},
        'scan': {'xy_step_mm', 'depth_min_mm', 'depth_max_mm',
                 'depth_step_mm', 'include_endpoints'},
        'material_grid': {'step_xy_mm'},
        'sampling': {'contact_margin_mm', 'minimum_reference_dot',
                     'min_samples_per_pose'},
        'lut': {'size', 'maximum_rms_angle_degrees'},
        'validation': {'fraction', 'seed', 'rms_degrees',
                       'maximum_out_of_range_fraction'},
        'visualization': {'enabled', 'difference_gain'},
    }
    unknown = set(raw) - allowed['calibration']
    if unknown:
        raise ValueError(f'Unknown calibration settings: {sorted(unknown)}')
    required = ('scan', 'material_grid', 'sampling', 'lut', 'validation', 'visualization')
    if any(not isinstance(raw.get(key), dict) for key in required):
        raise ValueError('Calibration configuration is incomplete')
    for section in required:
        unknown = set(raw[section]) - allowed[section]
        if unknown:
            raise ValueError(f'Unknown calibration.{section} settings: {sorted(unknown)}')
    if 'output_dir' in raw and not isinstance(raw['output_dir'], str):
        raise ValueError('calibration.output_dir must be a path string')

    radius = _positive(config['indentation']['radius_mm'], 'indentation.radius_mm')
    scan = raw['scan']
    for key in ('depth_min_mm', 'depth_max_mm', 'depth_step_mm'):
        _positive(scan.get(key), f'calibration.scan.{key}')
    if not 0 < scan['depth_min_mm'] <= scan['depth_max_mm'] < radius:
        raise ValueError('Calibration depths must satisfy 0 < min <= max < sphere radius')
    xy_steps = np.asarray(scan.get('xy_step_mm'), np.float64)
    grid_steps = np.asarray(raw['material_grid'].get('step_xy_mm'), np.float64)
    if xy_steps.shape != (2,) or not np.isfinite(xy_steps).all() or np.any(xy_steps <= 0):
        raise ValueError('calibration.scan.xy_step_mm must contain two positive values')
    if type(scan.get('include_endpoints', True)) is not bool:
        raise ValueError('calibration.scan.include_endpoints must be boolean')
    if grid_steps.shape != (2,) or not np.isfinite(grid_steps).all() or np.any(grid_steps <= 0):
        raise ValueError('calibration.material_grid.step_xy_mm must contain two positive values')

    sampling = raw['sampling']
    margin = _positive(sampling.get('contact_margin_mm', 0.),
                       'calibration.sampling.contact_margin_mm', allow_zero=True)
    if float(sampling.get('minimum_reference_dot', .1)) != .1:
        raise ValueError('calibration.sampling.minimum_reference_dot must be 0.1')
    minimum_samples = sampling.get('min_samples_per_pose', 16)
    if type(minimum_samples) is not int or minimum_samples < 1:
        raise ValueError('calibration.sampling.min_samples_per_pose must be a positive integer')
    shallowest_radius = math.sqrt(2 * radius * scan['depth_min_mm'] - scan['depth_min_mm'] ** 2)
    if margin >= shallowest_radius:
        raise ValueError('contact_margin_mm must be smaller than the shallowest contact radius')
    if bending_radius(config) is not None and margin != 0:
        raise ValueError('Curved calibration requires contact_margin_mm=0')

    lut = raw['lut']
    size = lut.get('size', 64)
    if type(size) is not int or size < 2:
        raise ValueError('calibration.lut.size must be an integer >= 2')
    maximum_rms = _positive(lut.get('maximum_rms_angle_degrees', 5.),
                            'calibration.lut.maximum_rms_angle_degrees')
    validation = raw['validation']
    fraction = float(validation.get('fraction', .2))
    maximum_out = float(validation.get('maximum_out_of_range_fraction', .05))
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('calibration.validation.fraction must be in (0,1)')
    if not np.isfinite(maximum_out) or not 0 <= maximum_out <= 1:
        raise ValueError('calibration.validation.maximum_out_of_range_fraction must be in [0,1]')
    validation_rms = _positive(validation.get('rms_degrees', 20.),
                               'calibration.validation.rms_degrees')
    seed = validation.get('seed', 42)
    if type(seed) is not int or seed < 0:
        raise ValueError('calibration.validation.seed must be a nonnegative integer')

    visualization = raw['visualization']
    enabled = visualization.get('enabled', True)
    if type(enabled) is not bool:
        raise ValueError('calibration.visualization.enabled must be boolean')
    gain = _positive(visualization.get('difference_gain', 2.),
                     'calibration.visualization.difference_gain')

    if output_override is not None:
        output = output_override.expanduser().resolve()
    elif raw.get('output_dir'):
        configured = Path(raw['output_dir']).expanduser()
        output = (config_path.parent / configured).resolve() if not configured.is_absolute() \
            else configured.resolve()
    else:
        output = _unique_run_dir()
    return CalibrationSettings(output, size, maximum_rms, fraction, validation_rms,
                               maximum_out, minimum_samples, seed, gain, enabled, margin)


def load_calibration_config(config_path: str | Path,
                            calibration_config_path: str | Path | None = None) -> dict:
    """Load the analytic simulator config and a separate LUT workflow config."""
    config_path = Path(config_path).expanduser().resolve()
    source_config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(source_config, dict):
        raise ValueError('Simulation config root must be an object')
    if source_config.get('backend', 'analytic') != 'analytic':
        raise ValueError('LUT calibration currently supports the analytic renderer only')
    simulation = load_config(config_path, allow_curve=True, allow_contact_tangent=True)
    if 'calibration' in simulation and calibration_config_path is None:
        config = copy.deepcopy(simulation)
    else:
        source = (Path(calibration_config_path).expanduser().resolve()
                  if calibration_config_path is not None else
                  Path(__file__).resolve().parents[1] / 'configs' / 'calibration.json')
        document = json.loads(source.read_text(encoding='utf-8'))
        if (not isinstance(document, dict) or set(document) != {'format_version', 'calibration'}
                or document.get('format_version') != 1
                or not isinstance(document.get('calibration'), dict)):
            raise ValueError(f'Invalid LUT calibration config: {source}')
        if 'calibration' in simulation:
            raise ValueError('Simulation config already embeds calibration; omit --calibration-config')
        config = dict(simulation, calibration=copy.deepcopy(document['calibration']))
    validate_config(config, allow_curve=True, allow_contact_tangent=True)
    return config


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=_json_value), encoding='utf-8')
    temporary.replace(path)


def _atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _json_value(value):
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, ScanPose):
        return value.to_dict()
    raise TypeError(f'Not JSON serializable: {type(value)}')


def _config_signature(config: dict) -> str:
    value = copy.deepcopy(config)
    value.get('calibration', {}).pop('output_dir', None)
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def _ensure_new_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f'Calibration output directory is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    (path / 'verification').mkdir(exist_ok=True)


def _pose_split(poses: list[ScanPose], settings: CalibrationSettings) -> dict[str, list[str]]:
    if len(poses) < 2:
        raise ValueError('Whole-pose validation requires at least two scan poses')
    keys = np.asarray([pose.image_key for pose in poses])
    order = np.arange(len(keys))
    np.random.default_rng(settings.validation_seed).shuffle(order)
    validation_count = min(len(keys) - 1,
                           max(1, math.ceil(settings.validation_fraction * len(keys))))
    return {
        'training_images': keys[order[validation_count:]].tolist(),
        'validation_images': keys[order[:validation_count]].tolist(),
    }


def _write_report(path: Path, rows: list[dict]) -> None:
    columns = ('pose_id', 'image_key', 'depth_index', 'row_index', 'column_index',
               'cx_mm', 'cy_mm', 'center_u', 'center_v', 'depth_mm',
               'contact_radius_mm', 'status', 'sample_count', 'rejected_reason',
               'render_ms', 'sampling_ms', 'preview_file')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_preview(reference, frame, path: Path, settings: CalibrationSettings,
                  status: str) -> None:
    residual, valid = image_space_diagnostics(reference, frame)
    rgb = np.rint(255 * np.clip(.5 + settings.difference_gain * residual, 0, 1)).astype(np.uint8)
    rgb[~valid] = 0
    image = Image.fromarray(rgb, 'RGB')
    draw = ImageDraw.Draw(image)
    pose = frame.pose
    text = (f'{pose.image_key} center=({pose.center_xy_mm[0]:.2f},'
            f'{pose.center_xy_mm[1]:.2f}) mm d={pose.depth_mm:.2f} mm '
            f'samples={frame.sample_count} {status}')
    bounds = draw.textbbox((5, 5), text)
    draw.rectangle((bounds[0] - 2, bounds[1] - 2, bounds[2] + 2, bounds[3] + 2),
                   fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _save_samples(path: Path, pieces: list[dict], accepted_poses: list[ScanPose],
                  grid: MaterialGrid, reference, metadata: dict) -> dict[str, np.ndarray]:
    fields = ('color_residual_samples', 'slope_samples', 'material_coordinates',
              'st_samples', 'material_grid_index')
    empty_values = {
        'color_residual_samples': np.empty((0, 3), np.float32),
        'slope_samples': np.empty((0, 2), np.float32),
        'material_coordinates': np.empty((0, 2), np.float64),
        'st_samples': np.empty((0, 2), np.float32),
        'material_grid_index': np.empty((0, 2), np.int32),
    }
    values = {key: (np.concatenate([piece[key] for piece in pieces]) if pieces
                    else empty_values[key]) for key in fields}
    counts = [len(piece['slope_samples']) for piece in pieces]
    values['image_index'] = np.concatenate([
        np.full(count, index, np.int32) for index, count in enumerate(counts)
    ]) if counts else np.empty((0,), np.int32)
    values['image_paths'] = np.asarray([pose.image_key for pose in accepted_poses], dtype='<U32')
    values['pose_ids'] = np.asarray([pose.pose_id for pose in accepted_poses], np.int32)
    total = len(values['slope_samples'])
    base_normal = reference.base_normals
    _, _, tangent = base_geometry(reference.config, grid)
    indices = values['material_grid_index']
    rows, columns = indices[:, 0], indices[:, 1]
    values['reference_normal_samples'] = base_normal[rows, columns].astype(np.float32)
    values['tangent_frame_samples'] = tangent[rows, columns].astype(np.float32)
    values.update(
        format_version=np.asarray(1, np.int32),
        material_grid_x_mm=grid.x_mm.astype(np.float64),
        material_grid_y_mm=grid.y_mm.astype(np.float64),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, default=_json_value)),
        reference_image=np.asarray('analytic_reference:' + metadata['run_signature']),
        reference_storage=np.asarray('rendered_once_in_memory'),
        sphere_radius_mm=np.asarray(metadata['sphere_radius_mm'], np.float64),
        sphere_side=np.asarray(metadata['sphere_side']),
        boundary_model=np.asarray('zero_boundary_displacement'),
    )
    for key, value in calibration_declarations(grid, reference.config).items():
        values[key] = np.asarray(value)
    _atomic_npz(path, **values)
    if len(values['color_residual_samples']) != total:
        raise RuntimeError('Saved sample arrays have inconsistent lengths')
    return values


def _normal_vectors(slopes: np.ndarray) -> np.ndarray:
    vectors = np.concatenate((-np.asarray(slopes, np.float64),
                              np.ones((len(slopes), 1), np.float64)), axis=-1)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-15)


def _angle_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    first, second = _normal_vectors(predicted), _normal_vectors(truth)
    return np.degrees(np.arccos(np.clip(np.sum(first * second, axis=-1), -1, 1)))


def fit_and_validate(samples: dict[str, np.ndarray], settings: CalibrationSettings,
                     output_dir: Path, metadata: dict,
                     pose_split: dict[str, list[str]]) -> tuple[dict, NormalCalibration | None,
                                                                NormalCalibration | None,
                                                                NormalCalibration | None]:
    colors = np.asarray(samples['color_residual_samples'], np.float32)
    slopes = np.asarray(samples['slope_samples'], np.float32)
    image_index = np.asarray(samples['image_index'], np.int32)
    keys = np.asarray(samples['image_paths']).astype(str)
    training_keys = list(pose_split['training_images'])
    validation_keys = list(pose_split['validation_images'])
    train_indices = np.asarray([i for i, key in enumerate(keys) if key in set(training_keys)], int)
    valid_indices = np.asarray([i for i, key in enumerate(keys) if key in set(validation_keys)], int)
    training_mask = np.isin(image_index, train_indices)
    validation_mask = np.isin(image_index, valid_indices)
    options = dict(size=settings.lut_size,
                   maximum_rms_angle_degrees=settings.maximum_rms_angle_degrees)
    report_path = output_dir / 'verification' / 'validation.json'
    training_lut = final_lut = diagnostic_lut = None
    fit_error = None

    if not np.any(training_mask):
        fit_error = 'No accepted training-pose samples are available'
    else:
        try:
            training_lut = build_normal_lut(colors[training_mask], slopes[training_mask], **options)
        except Exception as exc:
            fit_error = f'{type(exc).__name__}: {exc}'
    if training_lut is not None:
        training_metadata = dict(
            metadata,
            validation_status='training_only',
            training_images=training_keys,
            validation_images=validation_keys,
            total_sample_count=int(training_mask.sum()),
            original_supported_node_count=int(np.count_nonzero(training_lut.original_valid)),
            filled_node_fraction=float(1 - np.mean(training_lut.original_valid)),
        )
        training_lut = replace(training_lut, metadata=training_metadata)
        training_lut.save(output_dir / 'normal_lut_train.npz',
                          total_sample_count=int(training_mask.sum()))

    pose_rows = []
    all_angles = []
    all_out_of_range = []
    accepted_validation = set(keys[valid_indices].tolist())
    missing = sorted(set(validation_keys) - accepted_validation)
    if training_lut is not None and np.any(validation_mask):
        prediction, _, _ = lookup_slopes_raw(colors[validation_mask], training_lut)
        truth = slopes[validation_mask]
        angles = _angle_errors(prediction, truth)
        out_of_range = ((colors[validation_mask] < training_lut.color_min)
                        | (colors[validation_mask] > training_lut.color_max)).any(axis=-1)
        all_angles.append(angles)
        all_out_of_range.append(out_of_range)
        validation_sample_indices = image_index[validation_mask]
        for key in validation_keys:
            try:
                local_index = int(np.flatnonzero(keys == key)[0])
            except IndexError:
                pose_rows.append({
                    'image_key': key, 'sample_count': 0, 'rms_degrees': None,
                    'median_degrees': None, 'p95_degrees': None,
                    'out_of_range_fraction': None, 'passed': False,
                    'failure_reason': 'pose rejected or below minimum sample count',
                })
                continue
            selected = validation_sample_indices == local_index
            if not np.any(selected):
                pose_rows.append({
                    'image_key': key, 'sample_count': 0, 'rms_degrees': None,
                    'median_degrees': None, 'p95_degrees': None,
                    'out_of_range_fraction': None, 'passed': False,
                    'failure_reason': 'no held-out samples',
                })
                continue
            local_angles, local_out = angles[selected], out_of_range[selected]
            rms = float(np.sqrt(np.mean(local_angles ** 2)))
            out_fraction = float(np.mean(local_out))
            passed = (rms <= settings.validation_rms_degrees
                      and out_fraction <= settings.maximum_out_of_range_fraction)
            pose_rows.append({
                'image_key': key,
                'sample_count': int(selected.sum()),
                'rms_degrees': rms,
                'median_degrees': float(np.median(local_angles)),
                'p95_degrees': float(np.percentile(local_angles, 95)),
                'out_of_range_fraction': out_fraction,
                'passed': bool(passed),
                'failure_reason': '' if passed else 'held-out threshold exceeded',
            })
    else:
        for key in validation_keys:
            pose_rows.append({
                'image_key': key, 'sample_count': 0, 'rms_degrees': None,
                'median_degrees': None, 'p95_degrees': None,
                'out_of_range_fraction': None, 'passed': False,
                'failure_reason': fit_error or 'no held-out samples',
            })

    angle_values = np.concatenate(all_angles) if all_angles else np.empty(0, np.float64)
    out_values = (np.concatenate(all_out_of_range) if all_out_of_range
                  else np.empty(0, bool))
    passed = (training_lut is not None and bool(pose_rows) and not missing
              and all(row['passed'] for row in pose_rows))
    report = {
        'status': 'passed' if passed else 'failed',
        'training_images': training_keys,
        'validation_images': validation_keys,
        'missing_required_poses': missing,
        'images': pose_rows,
        'rms_degrees': (float(np.sqrt(np.mean(angle_values ** 2)))
                        if angle_values.size else None),
        'median_degrees': float(np.median(angle_values)) if angle_values.size else None,
        'p95_degrees': float(np.percentile(angle_values, 95)) if angle_values.size else None,
        'out_of_range_fraction': float(np.mean(out_values)) if out_values.size else None,
        'thresholds': {
            'rms_degrees': settings.validation_rms_degrees,
            'maximum_out_of_range_fraction': settings.maximum_out_of_range_fraction,
        },
        'sample_counts': {
            'total': int(len(colors)),
            'training': int(training_mask.sum()),
            'validation': int(validation_mask.sum()),
        },
        'fit_error': fit_error,
        'report_file': str(report_path),
    }

    if len(colors):
        try:
            fitted = build_normal_lut(colors, slopes, **options)
            lut_metadata = dict(
                metadata,
                validation_status='passed' if passed else 'failed',
                validation=report,
                training_images=training_keys,
                validation_images=validation_keys,
                total_sample_count=len(colors),
                original_supported_node_count=int(np.count_nonzero(fitted.original_valid)),
                filled_node_fraction=float(1 - np.mean(fitted.original_valid)),
            )
            fitted = replace(fitted, metadata=lut_metadata)
            if passed:
                fitted.save(output_dir / 'normal_lut.npz', total_sample_count=len(colors))
                final_lut = fitted
                report['publication_status'] = 'published'
            else:
                fitted.save(output_dir / 'normal_lut_diagnostic.npz',
                            total_sample_count=len(colors))
                diagnostic_lut = fitted
                report['publication_status'] = 'diagnostic_only'
        except Exception as exc:
            if passed:
                report['publication_status'] = 'failed'
                report['publication_error'] = f'{type(exc).__name__}: {exc}'
                _atomic_json(report_path, report)
                raise RuntimeError('Validation passed, but the formal LUT could not be built or saved') from exc
            report['publication_status'] = 'diagnostic_fit_failed'
            report['diagnostic_fit_error'] = f'{type(exc).__name__}: {exc}'
    else:
        report['publication_status'] = 'no_samples'
    _atomic_json(report_path, report)
    return report, training_lut, final_lut, diagnostic_lut


def _dry_run_summary(config: dict, poses: list[ScanPose], grid: MaterialGrid,
                     settings: CalibrationSettings) -> dict:
    by_depth = {}
    for pose in poses:
        row = by_depth.setdefault(str(pose.depth_mm), {
            'depth_mm': pose.depth_mm,
            'contact_radius_mm': pose.contact_radius_mm,
            'pose_count': 0,
        })
        row['pose_count'] += 1
    camera = config['camera']
    return {
        'status': 'dry-run',
        'backend': 'analytic',
        'surface_bending_radius_mm': bending_radius(config),
        'pose_count': len(poses),
        'render_count_sampling_pass': len(poses) + 1,
        'render_count_with_visualization': len(poses) + 1,
        'material_grid_shape': list(grid.shape),
        'material_grid_points': int(np.prod(grid.shape)),
        'image_pixels': int(camera['width_px'] * camera['height_px']),
        'depths': list(by_depth.values()),
        'first_pose': poses[0].to_dict(),
        'last_pose': poses[-1].to_dict(),
        'output_dir': str(settings.output_dir),
    }


def _save_sample_archives(output_dir: Path, pieces: list[dict], accepted_poses: list[ScanPose],
                          grid: MaterialGrid, reference, metadata: dict) -> dict:
    samples = _save_samples(output_dir / 'normal_samples.npz', pieces, accepted_poses,
                            grid, reference, metadata)
    ambiguity_fields = ('color_residual_samples', 'slope_samples', 'image_index', 'image_paths')
    _atomic_npz(output_dir / 'ambiguity_inputs.npz', format_version=np.asarray(1, np.int32),
                **{key: samples[key] for key in ambiguity_fields},
                run_signature=np.asarray(metadata['run_signature']),
        metadata=samples['metadata'],
                source_samples=np.asarray(str((output_dir / 'normal_samples.npz').resolve())))
    return samples


def run_calibration(config: dict, config_path: str | Path, *,
                    output_override: str | Path | None = None,
                    dry_run: bool = False) -> dict:
    """Run a fresh analytic calibration and persist its complete provenance."""
    config_path = Path(config_path).expanduser().resolve()
    config = copy.deepcopy(config)
    validate_config(config, allow_curve=True, allow_contact_tangent=True)
    if config.get('backend', 'analytic') != 'analytic':
        raise ValueError('LUT calibration supports only the analytic backend')
    settings = parse_calibration_settings(
        config, config_path,
        Path(output_override).expanduser() if output_override is not None else None)
    poses = build_scan_poses(config)
    grid = build_material_grid(config)
    split = _pose_split(poses, settings)
    summary = _dry_run_summary(config, poses, grid, settings)
    summary.update(
        pose_split={
            'training_pose_count': len(split['training_images']),
            'validation_pose_count': len(split['validation_images']),
            'validation_pose_keys_preview': split['validation_images'][:10],
        },
        visualization_enabled=settings.visualization_enabled,
    )
    if dry_run:
        return summary

    _ensure_new_output(settings.output_dir)
    run_signature = _config_signature(config)
    radius_curve = bending_radius(config)
    metadata = dict(
        calibration_kind='analytic_sphere_scan',
        backend='analytic',
        observation_model_id=OBSERVATION_MODEL_ID,
        geometry_model_id=(GEOMETRY_MODEL if radius_curve is not None else 'planar_sphere_cap_v1'),
        light_geometry_model=(LIGHT_MODEL if radius_curve is not None else LIGHT_GEOMETRY_MODEL),
        run_signature=run_signature,
        simulation_config_signature=run_signature,
        sphere_radius_mm=float(config['indentation']['radius_mm']),
        sphere_direction_z=int(normal_direction(config)),
        sphere_side=('positive_normal' if normal_direction(config) == -1
                     else 'negative_normal'),
        boundary_model='zero_boundary_displacement',
        scan=copy.deepcopy(config['calibration']['scan']),
        reference_color_source='analytic_zero_depth_linear_rgb_float',
        fixed_reference_normal='per-material-base-surface-normal',
        fixed_tangent_frame='per-material-base-surface-frame',
        slope_frame_id='reference_local_tangent_b_v1',
        sampling_version='material_paired_linear_rgb_curve_v2',
        **calibration_declarations(grid, config),
    )
    manifest = {
        'format_version': 1,
        'status': 'running',
        'run_status': 'running',
        'stage': 'sampling',
        'backend': 'analytic',
        'output_dir': str(settings.output_dir),
        'run_signature': run_signature,
        'config_path': str(config_path),
        'calibration_config': copy.deepcopy(config['calibration']),
        'config': config,
        'observation_model_id': OBSERVATION_MODEL_ID,
        'image_encoding': IMAGE_ENCODING,
        'reference': {'storage': 'rendered_once_in_memory', 'depth_mm': 0.0},
        'pose_split': split,
        'poses': [],
        'artifacts': {},
    }
    _atomic_json(settings.output_dir / 'config.json', config)
    _atomic_json(settings.output_dir / 'manifest.json', manifest)
    rows: list[dict] = []
    pieces: list[dict] = []
    accepted_poses: list[ScanPose] = []
    start_total = time.perf_counter()
    try:
        reference = render_reference(config, grid)
        for sequence, pose in enumerate(poses, 1):
            frame = sample_pose(config, reference, grid, pose)
            compact = compact_frame_samples(frame, grid)
            accepted = frame.sample_count >= settings.minimum_samples_per_pose
            status = 'accepted' if accepted else 'rejected_insufficient_samples'
            reason = '' if accepted else (
                f'{frame.sample_count} < {settings.minimum_samples_per_pose}')
            preview_path = settings.output_dir / 'verification' / f'{pose.image_key}.png'
            if settings.visualization_enabled:
                _save_preview(reference, frame, preview_path, settings, status)
                preview_file = str(preview_path.relative_to(settings.output_dir))
            else:
                preview_file = ''
            if accepted:
                pieces.append(compact)
                accepted_poses.append(pose)
            row = {
                'pose_id': pose.pose_id,
                'image_key': pose.image_key,
                'depth_index': pose.depth_index,
                'row_index': pose.row_index,
                'column_index': pose.column_index,
                'cx_mm': pose.center_xy_mm[0],
                'cy_mm': pose.center_xy_mm[1],
                'center_u': pose.center_uv[0],
                'center_v': pose.center_uv[1],
                'depth_mm': pose.depth_mm,
                'contact_radius_mm': pose.contact_radius_mm,
                'status': status,
                'sample_count': frame.sample_count,
                'rejected_reason': reason,
                'render_ms': 1000 * frame.render_seconds,
                'sampling_ms': 1000 * frame.sampling_seconds,
                'preview_file': preview_file,
            }
            rows.append(row)
            manifest['poses'].append({**pose.to_dict(), 'status': status,
                                      'sample_count': frame.sample_count,
                                      'rejected_reason': reason,
                                      'source_image': preview_file,
                                      'verification_image': preview_file})
            print(f'[{sequence}/{len(poses)}] {pose.image_key}: {status}, '
                  f'{frame.sample_count} samples, {frame.render_seconds:.3f}s', flush=True)

        accepted_keys = [pose.image_key for pose in accepted_poses]
        sample_metadata = dict(metadata,
                               accepted_pose_count=len(accepted_poses),
                               rejected_pose_count=len(poses) - len(accepted_poses),
                               rejected_pose_reasons=dict(Counter(
                                   row['rejected_reason'] for row in rows
                                   if row['rejected_reason'])),
                               pose_key_by_image_index=accepted_keys,
                               pose_id_by_image_index=[pose.pose_id for pose in accepted_poses],
                               material_grid_shape=list(grid.shape))
        samples = _save_sample_archives(settings.output_dir, pieces, accepted_poses,
                                        grid, reference, sample_metadata)
        _write_report(settings.output_dir / 'verification' / 'report.csv', rows)
        manifest['stage'] = 'fit_and_validate'
        manifest['artifacts'].update(samples='normal_samples.npz',
                                     ambiguity_inputs='ambiguity_inputs.npz')
        _atomic_json(settings.output_dir / 'manifest.json', manifest)
        report, training_lut, final_lut, diagnostic_lut = fit_and_validate(
            samples, settings, settings.output_dir, sample_metadata, split)

        visualization_error = None
        visualization_path = None
        lut_for_visualization = final_lut or diagnostic_lut
        if settings.visualization_enabled and lut_for_visualization is not None:
            try:
                from .calibration_visualization import generate_calibration_visualizations

                visualization_path = generate_calibration_visualizations(
                    samples, lut_for_visualization, report,
                    settings.output_dir / 'visualization')
            except Exception as exc:
                visualization_error = f'{type(exc).__name__}: {exc}'

        manifest['status'] = report['status']
        manifest['run_status'] = 'completed'
        manifest['stage'] = 'completed'
        manifest['validation'] = report
        manifest['artifacts'] = {
            'samples': 'normal_samples.npz',
            'ambiguity_inputs': 'ambiguity_inputs.npz',
            'training_lut': ('normal_lut_train.npz' if training_lut is not None else None),
            'formal_lut': 'normal_lut.npz' if final_lut is not None else None,
            'diagnostic_lut': ('normal_lut_diagnostic.npz'
                               if diagnostic_lut is not None else None),
            'verification_report': 'verification/report.csv',
            'validation': 'verification/validation.json',
            'visualization': (str(visualization_path.relative_to(settings.output_dir))
                              if visualization_path else None),
        }
        manifest['performance'] = {
            'total_seconds': time.perf_counter() - start_total,
            'render_seconds_median': float(np.median([row['render_ms'] for row in rows])) / 1000,
            'render_seconds_p95': float(np.percentile([row['render_ms'] for row in rows], 95)) / 1000,
            'device': config['render'].get('device', 'cpu'),
            'sampling_render_count': len(poses) + 1,
            'visualization_render_count': 0,
        }
        manifest['visualization_status'] = (
            'disabled' if not settings.visualization_enabled else
            'failed' if visualization_error else 'passed')
        if visualization_error:
            manifest['visualization_error'] = visualization_error
        _atomic_json(settings.output_dir / 'manifest.json', manifest)
        return manifest
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['run_status'] = 'failed'
        manifest['stage'] = 'failed'
        manifest['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        validation_path = settings.output_dir / 'verification' / 'validation.json'
        if validation_path.is_file():
            manifest['validation'] = json.loads(validation_path.read_text(encoding='utf-8'))
        manifest['artifacts'].update({
            key: filename for key, filename in (
                ('training_lut', 'normal_lut_train.npz'),
                ('formal_lut', 'normal_lut.npz'),
                ('diagnostic_lut', 'normal_lut_diagnostic.npz'),
                ('samples', 'normal_samples.npz'),
                ('ambiguity_inputs', 'ambiguity_inputs.npz'),
                ('verification_report', 'verification/report.csv'),
                ('validation', 'verification/validation.json'),
            ) if (settings.output_dir / filename).is_file()
        })
        manifest['performance'] = {'total_seconds': time.perf_counter() - start_total}
        _atomic_json(settings.output_dir / 'manifest.json', manifest)
        raise
