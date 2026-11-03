"""Compare direct LUT lookup, fixed-position GN3, and joint slope-depth GN3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid
from types import SimpleNamespace
from typing import Any

import numpy as np

from physical_model.compensation_utils.depth import (
    _validate_depth_settings,
    _validate_solver_settings,
    predict_active_gauss_newton,
    predict_depth_gauss_newton,
)
from physical_model.compensation_utils.common_domain import score_common_domain
from physical_model.compensation_utils.experiment import (
    depth_rmse,
    lut_support,
    pose_config,
    pose_metric,
    reference_area_weights,
    reference_geometry,
)
from physical_model.compensation_utils.photometry import (
    build_response_field,
)
from physical_model.cali_utils.calibration_samples import (
    MaterialGrid,
    render_reference,
    sample_geometry,
    sample_pose,
)
from physical_model.model_utils.config import validate_config
from physical_model.model_utils.observation import IMAGE_ENCODING, OBSERVATION_MODEL_ID
from physical_model.model_utils.surface_coordinates import (bending_radius,
                                                             normal_direction)
from utils.normal_mapping import NormalCalibration, lookup_slopes_raw


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'compensation.json'
METHODS = ('naive_lut', 'fixed_position_slope_gn3', 'depth_slope_gn3')


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read JSON configuration {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{path}: JSON root must be an object')
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_value)
                         + '\n', encoding='utf-8')
    temporary.replace(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _radius_id(radius_mm: float | None) -> str:
    return ('flat' if radius_mm is None else
            'radius_' + format(float(radius_mm), '.12g').replace('.', 'p'))


def _validate_scanner(settings: dict, base_config: dict) -> tuple[list[float | None], list[dict], dict]:
    if set(settings) - {'radii_mm', 'poses', 'grid_stride', 'point_chunk_size',
                        'solver', 'metrics', 'output_dir'}:
        raise ValueError('compensation config contains unknown keys')
    radii_raw = settings.get('radii_mm')
    if not isinstance(radii_raw, list) or not radii_raw:
        raise ValueError('compensation.radii_mm must be a nonempty list')
    radii: list[float | None] = []
    for radius in radii_raw:
        if radius is None:
            radii.append(None)
        elif isinstance(radius, bool) or not isinstance(radius, (int, float)) \
                or not np.isfinite(radius) or radius <= 0:
            raise ValueError('Each radius must be null or a finite positive number')
        else:
            radii.append(float(radius))
    if len({_radius_id(radius) for radius in radii}) != len(radii):
        raise ValueError('compensation.radii_mm contains duplicate curvatures')

    pose_values = settings.get('poses')
    if not isinstance(pose_values, list) or not pose_values:
        raise ValueError('compensation.poses must be a nonempty list')
    poses: list[dict] = []
    for index, value in enumerate(pose_values):
        if not isinstance(value, dict) or set(value) != {'pose_id', 'center_uv', 'depth_mm'}:
            raise ValueError(f'compensation.poses[{index}] must contain pose_id, center_uv, depth_mm')
        pose_id = value['pose_id']
        uv = np.asarray(value['center_uv'], dtype=np.float64)
        depth = value['depth_mm']
        if not isinstance(pose_id, str) or not pose_id.strip() \
                or any(char in pose_id for char in ('/', '\\', '\0')):
            raise ValueError(f'compensation.poses[{index}].pose_id must be a path-safe string')
        if uv.shape != (2,) or not np.isfinite(uv).all() or np.any(np.abs(uv) > 1.0):
            raise ValueError(f'compensation.poses[{index}].center_uv must be in [-1,1]^2')
        if isinstance(depth, bool) or not isinstance(depth, (int, float)) \
                or not np.isfinite(depth) or depth <= 0.0 \
                or depth >= float(base_config['indentation']['radius_mm']):
            raise ValueError(f'compensation.poses[{index}].depth_mm must lie inside the sphere radius')
        poses.append({'pose_id': pose_id, 'center_uv': uv.tolist(), 'depth_mm': float(depth)})
    if len({pose['pose_id'] for pose in poses}) != len(poses):
        raise ValueError('compensation.poses contains duplicate pose_id values')

    stride = settings.get('grid_stride', 1)
    point_chunk_size = settings.get('point_chunk_size', 512)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError('compensation.grid_stride must be a positive integer')
    if isinstance(point_chunk_size, bool) or not isinstance(point_chunk_size, int) \
            or point_chunk_size < 1:
        raise ValueError('compensation.point_chunk_size must be a positive integer')
    solver = settings.get('solver')
    if not isinstance(solver, dict):
        raise ValueError('compensation.solver must be an object')
    required_solver = {
        'steps', 'damping', 'prior_weight', 'max_step', 'initial_depth_mm',
        'depth_scale_mm', 'max_depth_mm', 'max_depth_step_mm', 'depth_damping',
        'depth_effect_threshold', 'rank_rtol', 'line_search_steps', 'color_weight',
    }
    if set(solver) != required_solver:
        missing = sorted(required_solver - set(solver))
        extra = sorted(set(solver) - required_solver)
        raise ValueError(f'compensation.solver keys differ; missing={missing}, unknown={extra}')
    if solver['steps'] != 3:
        raise ValueError('Depth GN3 uses exactly three iterations')
    initial_depth = solver['initial_depth_mm']
    maximum_depth = solver['max_depth_mm']
    if isinstance(initial_depth, bool) or not isinstance(initial_depth, (int, float)) \
            or not np.isfinite(initial_depth) or initial_depth < 0.0:
        raise ValueError('solver.initial_depth_mm must be finite and nonnegative')
    if isinstance(maximum_depth, bool) or not isinstance(maximum_depth, (int, float)) \
            or not np.isfinite(maximum_depth) or maximum_depth <= 0.0 \
            or initial_depth > maximum_depth:
        raise ValueError('solver.max_depth_mm must be finite, positive, and contain the initial depth')
    solver['color_weight'] = np.asarray(solver['color_weight'], dtype=np.float64).tolist()
    if len(solver['color_weight']) != 3 or not np.isfinite(solver['color_weight']).all() \
            or np.any(np.asarray(solver['color_weight']) < 0.0):
        raise ValueError('solver.color_weight must be three finite nonnegative values')
    _validate_solver_settings(
        steps=solver['steps'], damping=solver['damping'],
        prior_weight=solver['prior_weight'], max_step=solver['max_step'])
    _validate_depth_settings(
        depth_scale_mm=solver['depth_scale_mm'], max_depth_mm=solver['max_depth_mm'],
        max_depth_step_mm=solver['max_depth_step_mm'], depth_damping=solver['depth_damping'],
        depth_effect_threshold=solver['depth_effect_threshold'], rank_rtol=solver['rank_rtol'],
        line_search_steps=solver['line_search_steps'])
    metrics = settings.get('metrics')
    if not isinstance(metrics, dict) or set(metrics) != {'slope_tolerances'}:
        raise ValueError('compensation.metrics must contain only slope_tolerances')
    tolerances = np.asarray(metrics['slope_tolerances'], dtype=np.float64)
    if tolerances.ndim != 1 or len(tolerances) == 0 \
            or not np.isfinite(tolerances).all() or np.any(tolerances < 0.0):
        raise ValueError('metrics.slope_tolerances must be a nonempty nonnegative list')
    output_dir = settings.get('output_dir', '../compensate_results')
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError('compensation.output_dir must be a nonempty path string')
    return radii, poses, {
        'grid_stride': stride,
        'point_chunk_size': point_chunk_size,
        'solver': solver,
        'slope_tolerances': tolerances.tolist(),
        'output_dir': output_dir,
    }


def _load_pose_grid(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Material grid axes are required: {path}')
    with np.load(path, allow_pickle=False) as archive:
        required = ('material_grid_x_mm', 'material_grid_y_mm')
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f'{path} is missing material grid axes: {missing}')
        x_axis = np.asarray(archive['material_grid_x_mm'], dtype=np.float64)
        y_axis = np.asarray(archive['material_grid_y_mm'], dtype=np.float64)
        metadata = {}
        if 'metadata' in archive:
            metadata = json.loads(str(archive['metadata']))
        if 'coordinate_chart_id' in archive:
            metadata['coordinate_chart_id'] = str(archive['coordinate_chart_id'])
    for name, axis in (('material_grid_x_mm', x_axis), ('material_grid_y_mm', y_axis)):
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() \
                or np.any(np.diff(axis) <= 0.0):
            raise ValueError(f'{path}: {name} must be finite, increasing, and have at least two entries')
    return x_axis, y_axis, metadata


def _artifact_paths(lut_results: Path | None, lut_file: Path | None
                    ) -> tuple[Path, Path, Path, Path]:
    if (lut_results is None) == (lut_file is None):
        raise ValueError('Select exactly one calibration artifact with --lut-results or --lut')
    if lut_results is not None:
        root = lut_results.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f'--lut-results must be an artifact directory: {root}')
        training = root / 'normal_lut_train.npz'
        final = root / 'normal_lut.npz'
        if training.is_file():
            selected = training
        elif final.is_file():
            selected = final
        else:
            raise FileNotFoundError(f'{root} contains no usable normal_lut_train.npz or normal_lut.npz')
    else:
        selected = lut_file.expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f'--lut file does not exist: {selected}')
        root = selected.parent
    manifest = root / 'manifest.json'
    physical_config = root / 'config.json'
    return root, selected, manifest, physical_config


def _load_artifact(lut_results: Path | None, lut_file: Path | None, *,
                   allow_unpublished: bool = False) -> dict:
    root, lut_path, manifest_path, config_path = _artifact_paths(lut_results, lut_file)
    lut = NormalCalibration.load(lut_path)
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    if config_path.is_file():
        base_config = _read_json(config_path)
    elif isinstance(manifest.get('config'), dict):
        base_config = manifest['config']
    else:
        raise FileNotFoundError(f'Physical config is missing beside the LUT: {config_path}')
    samples_path = root / 'normal_samples.npz'
    x_axis, y_axis, sample_metadata = _load_pose_grid(samples_path)
    metadata = lut.metadata
    for key, expected in (
        ('color_space', 'linear_rgb'),
        ('image_encoding', IMAGE_ENCODING),
        ('observation_model_id', OBSERVATION_MODEL_ID),
        ('coordinate_kind', 'reference_arclength'),
        ('slope_frame_id', 'reference_local_tangent_b_v1'),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f'{lut_path}: incompatible LUT metadata {key}={metadata.get(key)!r}; '
                             f'expected {expected!r}')
    status = metadata.get('validation_status')
    if status not in ('training_only', 'passed'):
        raise ValueError(f'{lut_path}: LUT validation_status must be training_only or passed, got {status!r}')
    if lut_path.name == 'normal_lut_diagnostic.npz':
        raise ValueError('Diagnostic LUTs made from the full data set cannot be used for evaluation')
    calibration_status = manifest.get('status')
    if status == 'training_only' and calibration_status == 'failed' and not allow_unpublished:
        raise ValueError('The source calibration did not pass validation. Use --allow-unpublished '
                         'to run an explicit training-only experiment; the result will retain '
                         'the failed source status.')
    if not isinstance(manifest, dict):
        raise ValueError(f'{manifest_path}: manifest must be an object')
    validate_config(base_config, allow_curve=True, allow_contact_tangent=True)
    if bending_radius(base_config) is not None:
        raise ValueError('The source LUT must come from a flat reference surface')
    if base_config['surface']['view_side_z'] != -1 or normal_direction(base_config) != -1:
        raise ValueError('Compensation requires the +Z geometric / -Z optical normal convention')
    signatures = {
        'manifest': manifest.get('run_signature'),
        'LUT': metadata.get('run_signature'),
        'samples': sample_metadata.get('run_signature'),
    }
    if not all(signatures.values()) or len(set(signatures.values())) != 1:
        raise ValueError(f'Calibration run signatures differ across LUT, manifest, and samples: {signatures}')
    for key, expected in (
        ('color_space', 'linear_rgb'),
        ('image_encoding', IMAGE_ENCODING),
        ('observation_model_id', OBSERVATION_MODEL_ID),
        ('coordinate_kind', 'reference_arclength'),
        ('slope_frame_id', 'reference_local_tangent_b_v1'),
    ):
        if sample_metadata.get(key) != expected:
            raise ValueError(f'{samples_path}: incompatible sample metadata {key}='
                             f'{sample_metadata.get(key)!r}; expected {expected!r}')
    return {
        'root': root,
        'lut_path': lut_path,
        'lut_hash': _digest(lut_path),
        'lut': lut,
        'lut_metadata': metadata,
        'calibration_status': calibration_status,
        'manifest': manifest,
        'base_config': base_config,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'sample_metadata': sample_metadata,
        'samples_path': samples_path,
    }


def _source_pose_entries(manifest: dict) -> tuple[dict[str, dict], set[str], set[str]]:
    split = manifest.get('pose_split', {})
    training = set(map(str, split.get('training_images', [])))
    validation = set(map(str, split.get('validation_images', [])))
    entries: dict[str, dict] = {}
    for value in manifest.get('poses', []):
        if not isinstance(value, dict):
            continue
        key = str(value.get('image_key', ''))
        center = value.get('center_uv')
        if center is None and 'center_u' in value and 'center_v' in value:
            center = [value['center_u'], value['center_v']]
        depth = value.get('depth_mm')
        if key and center is not None and depth is not None:
            entries[key] = {'center_uv': np.asarray(center, dtype=np.float64),
                            'depth_mm': float(depth)}
    return entries, training, validation


def _check_pose_independence(poses: list[dict], artifact: dict) -> dict:
    status = artifact['lut_metadata']['validation_status']
    entries, training, validation = _source_pose_entries(artifact['manifest'])
    if not entries:
        if status == 'passed':
            raise ValueError('Cannot establish independent test poses: calibration manifest has no pose geometry')
        return {'status': status, 'matched_training_poses': [], 'matched_validation_poses': [],
                'matched_source_poses': []}
    matched_training, matched_validation, matched_source = [], [], []
    for pose in poses:
        keys = [key for key, entry in entries.items()
                if np.array_equal(entry['center_uv'].shape, (2,))
                and np.allclose(entry['center_uv'], pose['center_uv'], rtol=0.0, atol=1e-12)
                and np.isclose(entry['depth_mm'], pose['depth_mm'], rtol=0.0, atol=1e-12)]
        matched_source.extend(keys)
        matched_training.extend(key for key in keys if key in training)
        matched_validation.extend(key for key in keys if key in validation)
    if matched_training:
        raise ValueError('Configured evaluation poses overlap LUT training poses: '
                         + ', '.join(sorted(set(matched_training))[:8]))
    if status == 'passed' and matched_source:
        raise ValueError('The full-data LUT cannot evaluate calibration poses; select the training LUT '
                         'or configure independent sphere poses: '
                         + ', '.join(sorted(set(matched_source))[:8]))
    if status == 'training_only' and not training:
        raise ValueError('The training-only LUT metadata does not declare its source training pose split')
    return {
        'status': status,
        'matched_training_poses': sorted(set(matched_training)),
        'matched_validation_poses': sorted(set(matched_validation)),
        'matched_source_poses': sorted(set(matched_source)),
        'source_training_pose_count': len(training),
        'source_validation_pose_count': len(validation),
    }


def _sample_grid(artifact: dict, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    x_axis = artifact['x_axis'][::stride]
    y_axis = artifact['y_axis'][::stride]
    if x_axis[-1] != artifact['x_axis'][-1]:
        x_axis = np.r_[x_axis, artifact['x_axis'][-1]]
    if y_axis[-1] != artifact['y_axis'][-1]:
        y_axis = np.r_[y_axis, artifact['y_axis'][-1]]
    if len(x_axis) < 2 or len(y_axis) < 2:
        raise ValueError('grid_stride leaves fewer than two points on a material axis')
    s, t = np.meshgrid(x_axis, y_axis, indexing='xy')
    coordinates = np.column_stack((s.ravel(), t.ravel()))
    area = reference_area_weights(x_axis, y_axis).ravel()
    row, column = np.indices((len(y_axis), len(x_axis)))
    grid_index = np.column_stack((row.ravel(), column.ravel())).astype(np.int32)
    return x_axis, y_axis, coordinates, area, grid_index


def _plan(artifact: dict, config_path: Path, config: dict, args: argparse.Namespace) -> dict:
    scanner = config.get('compensation')
    if not isinstance(scanner, dict):
        raise ValueError('Config must contain a compensation object')
    radii, poses, resolved = _validate_scanner(scanner, artifact['base_config'])
    output_root = Path(resolved['output_dir']).expanduser()
    resolved['output_dir'] = str((config_path.parent / output_root).resolve()
                                 if not output_root.is_absolute() else output_root.resolve())
    pose_safety = _check_pose_independence(poses, artifact)
    x_axis, y_axis, coordinates, _, _ = _sample_grid(artifact, resolved['grid_stride'])
    return {
        'config_path': config_path,
        'config': config,
        'artifact': artifact,
        'radii': radii,
        'poses': poses,
        'resolved': resolved,
        'pose_safety': pose_safety,
        'grid': {
            'x_axis': x_axis,
            'y_axis': y_axis,
            'coordinates': coordinates,
        },
        'device': args.device,
    }


def _plot_results(output: Path, radii: list[float | None],
                  metrics: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['flat' if radius is None else f'R={radius:g} mm' for radius in radii]
    positions = np.arange(len(radii))
    styles = {'naive_lut': 'o-', 'fixed_position_slope_gn3': 's-', 'depth_slope_gn3': '^-'}
    colors = {'naive_lut': '#b5523b', 'fixed_position_slope_gn3': '#4c78a8',
              'depth_slope_gn3': '#3b8c69'}
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for method in METHODS:
        values = [metrics['by_radius'][_radius_id(radius)]['methods'][method].get(
            'pose_equal_slope_rmse') for radius in radii]
        axis.plot(positions, values, styles[method], color=colors[method], label=method)
    axis.set_xticks(positions, labels)
    axis.set_ylabel('pose-equal area-weighted slope RMSE')
    axis.set_xlabel('reference curvature')
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    slope_path = output / 'slope_rmse_by_radius.png'
    figure.savefig(slope_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    values = [metrics['by_radius'][_radius_id(radius)]['methods']['depth_slope_gn3'].get(
        'pose_equal_depth_rmse_mm') for radius in radii]
    axis.plot(positions, values, 'o-', color=colors['depth_slope_gn3'])
    axis.set_xticks(positions, labels)
    axis.set_ylabel('pose-equal area-weighted depth RMSE (mm)')
    axis.set_xlabel('reference curvature')
    axis.grid(True, alpha=0.25)
    depth_path = output / 'depth_rmse_by_radius.png'
    figure.savefig(depth_path, dpi=160)
    plt.close(figure)
    return [slope_path.name, depth_path.name]


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_raw(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays,
                            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False,
                                                           default=_json_value)))
    temporary.replace(path)


def _solver_diagnostic_summary(values: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in values.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim == 0:
            result[key] = value.item()
        elif value.dtype.kind == 'b':
            result[key + '_count'] = int(np.count_nonzero(value))
        elif key.endswith('_count'):
            result[key] = int(np.sum(value))
    for key in ('active_switch_total', 'source_eligibility_switch_total'):
        if key in values:
            result[key] = int(values[key])
    return result


def _run_radius(plan: dict, radius: float | None, output: Path,
                per_pose_rows: list[dict]) -> dict:
    artifact = plan['artifact']
    coordinates = plan['grid']['coordinates']
    base = artifact['base_config']
    resolved = plan['resolved']
    radius_key = _radius_id(radius)
    reference_config = pose_config(base, radius)
    reference_config['render']['device'] = plan['device']
    validate_config(reference_config, allow_curve=True, allow_contact_tangent=True)
    x_axis = plan['grid']['x_axis']
    y_axis = plan['grid']['y_axis']
    s, t = np.meshgrid(x_axis, y_axis, indexing='xy')
    xy = np.stack((s, t), axis=-1)
    size = np.asarray(base['surface']['size_mm'], dtype=np.float64)
    st = xy + size[None, None, :] / 2.0
    grid = MaterialGrid(x_axis, y_axis, xy, st,
                        str(artifact['sample_metadata'].get('coordinate_chart_id',
                                                            'reference_arclength_grid')))
    reference = render_reference(reference_config, grid)
    x0, n0, _ = reference_geometry(coordinates, radius)

    geometry_by_pose: dict[str, dict] = {}
    candidate_masks = []
    for pose_index, pose in enumerate(plan['poses']):
        scan_pose = SimpleNamespace(
            pose_id=pose_index, image_key=pose['pose_id'], depth_mm=pose['depth_mm'],
            center_uv=tuple(pose['center_uv']))
        frame = sample_pose(reference_config, reference, grid, scan_pose)
        geometry = sample_geometry(frame.config, grid, scan_pose)
        valid = np.asarray(frame.sample_valid_mask, dtype=bool).reshape(-1)
        if not np.any(valid):
            raise ValueError(f'{radius_key}/{pose["pose_id"]}: no valid material samples remain')
        geometry_by_pose[pose['pose_id']] = {
            'config': frame.config,
            'frame': frame,
            'geometry': geometry,
            'scan_pose': scan_pose,
            'valid': valid,
        }
        candidate_masks.append(valid)

    union = np.flatnonzero(np.logical_or.reduce(candidate_masks))
    field_config = pose_config(base, radius)
    field_config['render']['device'] = plan['device']
    response_field = build_response_field(
        coordinates[union], field_config,
        point_chunk_size=resolved['point_chunk_size'])
    source_lut = artifact['lut']
    solver = resolved['solver']
    pose_metrics: dict[str, dict] = {}

    for pose in plan['poses']:
        item = geometry_by_pose[pose['pose_id']]
        valid_global = np.flatnonzero(item['valid'])
        row_indices = np.searchsorted(union, valid_global)
        field = response_field.take_rows(row_indices)
        geometry = item['geometry']
        frame = item['frame']
        state_config = item['config']
        ground_truth_slope = frame.slope_grid.reshape(-1, 2)[valid_global].astype(np.float64)
        ground_truth_depth = geometry[6].reshape(-1)[valid_global].astype(np.float64)
        contact_position = geometry[0].reshape(-1, 3)[valid_global]
        contact_geometric_normal = geometry[2].reshape(-1, 3)[valid_global]
        observed_reference_rgb = reference.grid_color.reshape(-1, 3)[valid_global]
        color_delta = frame.color_residual_grid.reshape(-1, 3)[valid_global].astype(np.float64)
        observed_contact_rgb = observed_reference_rgb + color_delta
        if not np.isfinite(color_delta).all() or not np.isfinite(observed_contact_rgb).all():
            raise ValueError(f'{radius_key}/{pose["pose_id"]}: image-paired linear RGB is non-finite')
        beta_lut, lut_variance, lut_confidence = lookup_slopes_raw(color_delta, source_lut)
        beta_lut = np.asarray(beta_lut, dtype=np.float64)
        beta_fixed, fixed_diag = predict_active_gauss_newton(
            field, color_delta, beta_lut, steps=3,
            damping=float(solver['damping']), prior_weight=float(solver['prior_weight']),
            max_step=float(solver['max_step']),
            color_weight=np.asarray(solver['color_weight'], dtype=np.float64),
            diagnostics=True)
        (beta_depth, depth_estimate), depth_diag = predict_depth_gauss_newton(
            field, color_delta, beta_lut, float(solver['initial_depth_mm']), steps=3,
            damping=float(solver['damping']), prior_weight=float(solver['prior_weight']),
            max_step=float(solver['max_step']),
            color_weight=np.asarray(solver['color_weight'], dtype=np.float64),
            depth_scale_mm=float(solver['depth_scale_mm']),
            max_depth_mm=float(solver['max_depth_mm']),
            max_depth_step_mm=float(solver['max_depth_step_mm']),
            depth_damping=float(solver['depth_damping']),
            depth_effect_threshold=float(solver['depth_effect_threshold']),
            rank_rtol=float(solver['rank_rtol']),
            line_search_steps=int(solver['line_search_steps']), diagnostics=True)

        area = plan['grid_area'][valid_global]
        grid_index = plan['grid_index'][valid_global]
        color_inside, original_support = lut_support(color_delta, source_lut)
        method_predictions = {
            'naive_lut': beta_lut,
            'fixed_position_slope_gn3': beta_fixed,
            'depth_slope_gn3': beta_depth,
        }
        method_rows = {}
        for method, predicted in method_predictions.items():
            metric = pose_metric(predicted, ground_truth_slope, area,
                                 resolved['slope_tolerances'])
            row = {
                'condition': radius_key,
                'radius_mm': radius,
                'pose_id': pose['pose_id'],
                'center_u': pose['center_uv'][0],
                'center_v': pose['center_uv'][1],
                'depth_truth_mm': pose['depth_mm'],
                'method': method,
                'sample_count': int(len(valid_global)),
                'lut_out_of_range_fraction': float(np.sum(area * ~color_inside) / np.sum(area)),
                'lut_original_support_fraction': float(
                    np.sum(area * original_support) / np.sum(area)),
                **metric,
            }
            method_rows[method] = row
            per_pose_rows.append(row)
        depth_method_row = method_rows['depth_slope_gn3']
        depth_method_row['depth_rmse_mm'] = depth_rmse(depth_estimate, ground_truth_depth, area)
        depth_method_row['depth_bias_mm'] = float(
            np.sum(area * (depth_estimate - ground_truth_depth)) / np.sum(area))
        depth_method_row['depth_estimate_mean_mm'] = float(np.sum(area * depth_estimate) / np.sum(area))
        depth_method_row['depth_truth_mean_mm'] = float(np.sum(area * ground_truth_depth) / np.sum(area))
        depth_method_row.update({f'depth_diag_{key}': value for key, value in
                                 _solver_diagnostic_summary(depth_diag).items()})
        method_rows['fixed_position_slope_gn3'].update({
            f'slope_diag_{key}': value for key, value in _solver_diagnostic_summary(fixed_diag).items()})

        raw_path = output / 'raw' / radius_key / f'{pose["pose_id"]}.npz'
        arrays = {
            'grid_index': grid_index.astype(np.int32),
            'material_coordinates_mm': coordinates[valid_global].astype(np.float64),
            'reference_position_mm': x0[valid_global].astype(np.float64),
            'reference_normal': n0[valid_global].astype(np.float64),
            'tangent_frame': geometry[3].reshape(-1, 3, 2)[valid_global].astype(np.float64),
            'contact_position_mm': contact_position.astype(np.float64),
            'contact_geometric_normal': contact_geometric_normal.astype(np.float64),
            'color_delta_linear_rgb': color_delta.astype(np.float64),
            'model_background_linear_rgb': field.background.astype(np.float64),
            'observed_reference_linear_rgb': observed_reference_rgb.astype(np.float64),
            'observed_contact_linear_rgb': observed_contact_rgb.astype(np.float64),
            'slope_truth': ground_truth_slope.astype(np.float64),
            'depth_truth_mm': ground_truth_depth.astype(np.float64),
            'slope_naive_lut': beta_lut.astype(np.float64),
            'slope_fixed_position_gn3': beta_fixed.astype(np.float64),
            'slope_depth_gn3': beta_depth.astype(np.float64),
            'depth_estimate_gn3_mm': depth_estimate.astype(np.float64),
            'area_mm2': area.astype(np.float64),
            'lut_variance': np.asarray(lut_variance),
            'lut_confidence': np.asarray(lut_confidence),
            'lut_color_in_range': color_inside,
            'lut_original_node_support': original_support,
        }
        for name, values in depth_diag.items():
            if isinstance(values, np.ndarray) and values.ndim > 0 and len(values) == len(valid_global):
                arrays['depth_diagnostic_' + name] = values
        _save_raw(raw_path, arrays, {
            'condition': radius_key,
            'radius_mm': radius,
            'pose': pose,
            'methods': list(METHODS),
            'observation_model_id': OBSERVATION_MODEL_ID,
            'image_encoding': IMAGE_ENCODING,
            'color_delta_definition': 'bilinear sampled contact linear RGB minus separately sampled reference linear RGB',
            'reference_visibility': 'exact_at_X0_frozen_for_slope_depth_solver',
            'test_truth_used_only_for_observation_generation_and_metrics': True,
            'source_lut': str(artifact['lut_path']),
            'source_lut_sha256': artifact['lut_hash'],
        })
        pose_metrics[pose['pose_id']] = {
            'pose': pose,
            'sample_count': int(len(valid_global)),
            'raw_file': str(raw_path.relative_to(output)),
            'methods': method_rows,
        }
        print(f'{radius_key}/{pose["pose_id"]}: {len(valid_global)} valid points', flush=True)

    return {'radius_mm': radius, 'pose_metrics': pose_metrics}


def _aggregate_metrics(by_radius: dict[str, Any], radii: list[float | None]) -> dict[str, Any]:
    result = {'by_radius': {}, 'overall': {}}
    all_pose_methods: dict[str, dict[str, list[dict]]] = {method: {} for method in METHODS}
    for radius in radii:
        radius_key = _radius_id(radius)
        poses = by_radius[radius_key]['pose_metrics']
        methods = {}
        for method in METHODS:
            rows = [value['methods'][method] for value in poses.values()]
            methods[method] = {
                'pose_equal_slope_rmse': float(np.sqrt(np.mean(
                    [row['slope_rmse'] ** 2 for row in rows]))),
                'mean_pose_slope_error_p95': float(np.mean(
                    [row['slope_error_p95'] for row in rows])),
                'mean_pose_lut_out_of_range_fraction': float(np.mean(
                    [row['lut_out_of_range_fraction'] for row in rows])),
                'mean_pose_lut_original_support_fraction': float(np.mean(
                    [row['lut_original_support_fraction'] for row in rows])),
                'pose_count': len(rows),
                'sample_count_total': int(sum(row['sample_count'] for row in rows)),
            }
            all_pose_methods[method][radius_key] = rows
            if method == 'depth_slope_gn3':
                methods[method]['pose_equal_depth_rmse_mm'] = float(np.sqrt(np.mean(
                    [row['depth_rmse_mm'] ** 2 for row in rows])))
                methods[method]['mean_pose_depth_bias_mm'] = float(np.mean(
                    [row['depth_bias_mm'] for row in rows]))
                methods[method]['mean_rank_deficient_count'] = float(np.mean(
                    [row.get('depth_diag_rank_deficient_count', 0) for row in rows]))
                methods[method]['mean_depth_weak_count'] = float(np.mean(
                    [row.get('depth_diag_depth_effect_weak_count', 0) for row in rows]))
        result['by_radius'][radius_key] = {
            'radius_mm': radius,
            'pose_count': len(poses),
            'sample_count_total': int(sum(value['sample_count'] for value in poses.values())),
            'methods': methods,
        }

    all_rows = {method: [row for poses in all_pose_methods[method].values() for row in poses]
                for method in METHODS}
    for method, rows in all_rows.items():
        result['overall'][method] = {
            'pose_equal_slope_rmse': float(np.sqrt(np.mean([row['slope_rmse'] ** 2 for row in rows]))),
            'mean_pose_slope_error_p95': float(np.mean([row['slope_error_p95'] for row in rows])),
            'pose_count': len(rows),
            'sample_count_total': int(sum(row['sample_count'] for row in rows)),
        }
        if method == 'depth_slope_gn3':
            result['overall'][method]['pose_equal_depth_rmse_mm'] = float(np.sqrt(np.mean(
                [row['depth_rmse_mm'] ** 2 for row in rows])))
            result['overall'][method]['mean_pose_depth_bias_mm'] = float(np.mean(
                [row['depth_bias_mm'] for row in rows]))
    return result


def _dry_run_report(plan: dict) -> dict:
    return {
        'status': 'dry-run',
        'source_lut': str(plan['artifact']['lut_path']),
        'source_lut_sha256': plan['artifact']['lut_hash'],
        'source_lut_validation_status': plan['artifact']['lut_metadata']['validation_status'],
        'source_calibration_status': plan['artifact']['calibration_status'],
        'pose_independence': plan['pose_safety'],
        'device': plan['device'],
        'observation_model_id': OBSERVATION_MODEL_ID,
        'radii_mm': plan['radii'],
        'poses': plan['poses'],
        'material_grid_shape_after_stride': [len(plan['grid']['y_axis']),
                                             len(plan['grid']['x_axis'])],
        'material_candidate_count': len(plan['grid']['coordinates']),
        'point_chunk_size': plan['resolved']['point_chunk_size'],
        'solver': plan['resolved']['solver'],
        'output_root': plan['resolved']['output_dir'],
        'writes_output': False,
    }


def run_experiment(plan: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f'Results directory already exists: {output}')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    radii = plan['radii']
    x_axis, y_axis, coordinates, area, grid_index = _sample_grid(
        plan['artifact'], plan['resolved']['grid_stride'])
    plan['grid_area'] = area
    plan['grid_index'] = grid_index
    snapshot = {
        'format_version': 1,
        'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'device': plan['device'],
        'compensation_config': plan['config'],
        'source_physical_config': plan['artifact']['base_config'],
        'source_lut': str(plan['artifact']['lut_path']),
        'source_lut_sha256': plan['artifact']['lut_hash'],
        'source_lut_metadata': plan['artifact']['lut_metadata'],
        'source_manifest': plan['artifact']['manifest'],
        'pose_independence': plan['pose_safety'],
        'observation_model_id': OBSERVATION_MODEL_ID,
        'image_encoding': IMAGE_ENCODING,
        'material_grid': {
            'source_samples': str(plan['artifact']['samples_path']),
            'coordinate_kind': 'reference_arclength',
            'chart_axes_mm': {'s': x_axis.tolist(), 't': y_axis.tolist()},
            'sampling_stride': plan['resolved']['grid_stride'],
        },
        'methods': list(METHODS),
        'metric_definition': 'sqrt(pose-equal mean of reference-area-weighted squared local tangent slope vector errors)',
        'truth_usage': 'test sphere geometry is used only to synthesize observation and score predictions',
    }
    _write_json(output / 'config_snapshot.json', snapshot)
    rows: list[dict] = []
    by_radius = {}
    try:
        for radius in radii:
            result = _run_radius(plan, radius, output, rows)
            by_radius[_radius_id(radius)] = result
        metrics = _aggregate_metrics(by_radius, radii)
        common_scores, coverage = score_common_domain(
            output, [_radius_id(radius) for radius in radii],
            [pose['pose_id'] for pose in plan['poses']],
            plan['resolved']['slope_tolerances'])
        common_complete = all(row['common_point_count'] > 0 for row in coverage.values())
        common_report = {
            'status': 'completed' if common_complete else 'empty_common_domain',
            'domain': 'per-pose material grid intersection across all configured curvatures',
            'coverage': coverage,
            'metrics': _aggregate_metrics(common_scores, radii) if common_complete else None,
        }
        _write_json(output / 'common_domain_metrics.json', common_report)
        _write_csv(output / 'per_pose_metrics.csv', rows)
        figure_files = _plot_results(output, radii, metrics)
        report = {
            'status': 'completed',
            'source_lut': str(plan['artifact']['lut_path']),
            'source_lut_sha256': plan['artifact']['lut_hash'],
            'source_lut_validation_status': plan['artifact']['lut_metadata']['validation_status'],
            'source_calibration_status': plan['artifact']['calibration_status'],
            'pose_independence': plan['pose_safety'],
            'observation_model_id': OBSERVATION_MODEL_ID,
            'image_encoding': IMAGE_ENCODING,
            'methods': list(METHODS),
            'radii_mm': radii,
            'poses': plan['poses'],
            'metrics': metrics,
            'artifacts': {
                'per_pose_metrics': 'per_pose_metrics.csv',
                'common_domain_metrics': 'common_domain_metrics.json',
                'raw_arrays': 'raw/<condition>/<pose_id>.npz',
                'figures': figure_files,
                'config_snapshot': 'config_snapshot.json',
            },
            'elapsed_seconds': time.perf_counter() - started,
        }
        _write_json(output / 'metrics.json', report)
        snapshot['status'] = 'completed'
        snapshot['completed_utc'] = datetime.now(timezone.utc).isoformat()
        snapshot['elapsed_seconds'] = report['elapsed_seconds']
        _write_json(output / 'config_snapshot.json', snapshot)
        return report
    except Exception as exc:
        snapshot['status'] = 'failed'
        snapshot['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        snapshot['elapsed_seconds'] = time.perf_counter() - started
        _write_json(output / 'config_snapshot.json', snapshot)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lut-results', type=Path,
                       help='Calibration run directory; normal_lut_train.npz is preferred')
    group.add_argument('--lut', type=Path,
                       help='Explicit signed linear-RGB NormalCalibration .npz file')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help=f'Radii, sphere poses, sampling, and solver settings (default: {DEFAULT_CONFIG})')
    parser.add_argument('--out', type=Path,
                        help='New result directory (must not already exist)')
    parser.add_argument('--device', choices=('cpu',), default='cpu',
                        help='Solver device; current NumPy implementation supports CPU')
    parser.add_argument('--allow-unpublished', action='store_true',
                        help='Allow a training-only LUT when source calibration validation failed')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate inputs and print the scan plan without writing results')
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = _read_json(config_path)
    if set(config) != {'format_version', 'compensation'} or config['format_version'] != 1:
        raise ValueError('Compensation config must have format_version=1 and a compensation object')
    artifact = _load_artifact(args.lut_results, args.lut,
                              allow_unpublished=args.allow_unpublished)
    plan = _plan(artifact, config_path, config, args)
    if args.dry_run:
        print(json.dumps(_dry_run_report(plan), ensure_ascii=False, indent=2, default=_json_value))
        return
    if args.out is None:
        root = Path(plan['resolved']['output_dir'])
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
        output = root / run_id
    else:
        output = args.out.expanduser().resolve()
    report = run_experiment(plan, output)
    print(f'Completed: {output / "metrics.json"}', flush=True)
    print(json.dumps(report['metrics']['overall'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
