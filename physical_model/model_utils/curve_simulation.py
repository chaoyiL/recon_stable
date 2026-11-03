"""CLI orchestration and reproducible artifacts for analytic curved surfaces."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np

from .backend_layout import PROJECT_ROOT, simulation_config_path, new_render_output_dir
from .config import validate_config
from .curve_geometry import (GEOMETRY_MODEL, LIGHT_MODEL, CONTACT_MODEL,
                             BOUND_EPS_MM, HIT_EPS_MM, sphere_pose, contact_bounds)
from .geometry import LIGHT_GEOMETRY_MODEL, export_stl, contact_radius
from .light_distribution import analytic_light_samples
from .renderer import render, save_result
from .surface_coordinates import (COORDINATE_MODEL, bending_radius, center_uv,
                                  center_material_mm, normal_direction)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding='utf-8')
    temporary.replace(path)


def _code_record() -> dict:
    files = [PROJECT_ROOT / 'simulate_curve.py',
             *sorted((PROJECT_ROOT / 'model_utils').glob('*.py'))]
    hashes = {str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'):
              hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    result = {'source_sha256': hashes}
    for name, args in (('git_commit', ['rev-parse', 'HEAD']),
                       ('git_status', ['status', '--porcelain'])):
        try:
            process = subprocess.run(['git', *args], cwd=PROJECT_ROOT,
                                     capture_output=True, text=True, encoding='utf-8', timeout=10)
            result[name] = process.stdout.strip() if process.returncode == 0 else 'unavailable'
        except (OSError, subprocess.TimeoutExpired):
            result[name] = 'unavailable'
    return result


def describe_scene(config: dict) -> dict:
    R = bending_radius(config)
    center = center_material_mm(config)
    W, H = config['surface']['size_mm']
    if R is None:
        anchor = np.array([*center, 0.])
        direction = np.array([0., 0., normal_direction(config)])
        ball = config['indentation']
        sphere = anchor + (ball['depth_mm'] - ball['radius_mm']) * direction
        half_extents = [float(contact_radius(config))] * 2
    else:
        anchor, direction, sphere = sphere_pose(config)
        half_extents = list(contact_bounds(config))
    lights = []
    for strip in config['lights']:
        points, normals, powers = analytic_light_samples(strip, config)
        lights.append({'side': strip['side'], 'sample_count': len(points),
                       'position_mm': points.tolist(),
                       'normal_world': np.broadcast_to(normals, points.shape).tolist(),
                       'power_rgb_w': powers.tolist(), 'total_power_rgb_w': powers.sum(0).tolist()})
    return {'geometry_model_id': GEOMETRY_MODEL if R is not None else 'planar_sphere_cap_v1',
            'coordinate_model_id': COORDINATE_MODEL,
            'contact_model_id': CONTACT_MODEL if R is not None else 'planar_sphere_cap_v1',
            'light_geometry_model': LIGHT_MODEL if R is not None else LIGHT_GEOMETRY_MODEL,
            'center_uv': center_uv(config).tolist(), 'center_material_mm': center.tolist(),
            'anchor_world_mm': anchor.tolist(), 'sphere_center_world_mm': sphere.tolist(),
            'indentation_direction_world': direction.tolist(),
            'contact_half_extents_material_mm': half_extents,
            'curve_radius_mm': R, 'edge_angle_degrees': 0. if R is None else np.degrees(W / (2 * R)),
            'edge_sag_mm': 0. if R is None else 2 * R * np.sin(W / (4 * R)) ** 2,
            'camera_center_distance_mm': float(np.linalg.norm(config['camera']['position_mm'])),
            'image_pixels': config['camera']['width_px'] * config['camera']['height_px'],
            'light_sample_count': sum(row['sample_count'] for row in lights),
            'light_samples': lights, 'surface_uv_range': [-1, 1],
            'surface_uv_reference': 'bent_unindented_reference_surface',
            'length_unit': 'mm', 'hit_epsilon_mm': HIT_EPS_MM,
            'boundary_epsilon_mm': BOUND_EPS_MM,
            'geometry_dtype': config['render'].get('geometry_dtype', 'float64'),
            'radiometry_dtype': config['render'].get('radiometry_dtype', 'float32')}


def _flat_truth(gt: dict, config: dict) -> dict:
    mask, point = gt['mask'], gt['position_mm']
    base = point.copy()
    base[..., 2] = np.where(mask, 0., np.nan)
    normal = np.zeros_like(point)
    normal[..., 2] = 1
    normal[~mask] = np.nan
    center = center_material_mm(config)
    inside = np.sum((point[..., :2] - center) ** 2, -1) < contact_radius(config) ** 2
    gt.update(normal_geometric_world=gt['normal_world'] * config['surface']['view_side_z'],
              surface_uv=point[..., :2] / (np.asarray(config['surface']['size_mm']) / 2),
              base_position_mm=base, base_normal_world=normal,
              normal_displacement_mm=point[..., 2] * normal_direction(config),
              contact_mask=mask & inside, surface_part=np.where(mask, np.where(inside, 2, 1), 0))
    return gt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Render analytic cylindrical bending with local-normal sphere indentation')
    parser.add_argument('--config', type=Path, default=simulation_config_path('analytic'))
    parser.add_argument('--calibration-config', type=Path,
                        help='Calibration scan settings used with --calibrate-lut')
    parser.add_argument('--out', type=Path,
                        help='New/empty output directory (default: physical_model/render_results/<unique run>)')
    parser.add_argument('--calibrate-lut', action='store_true',
                        help='Run the virtual scan/LUT calibration with the same surface overrides')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--radius-mm', type=float, help='Positive X bending radius in mm')
    mode.add_argument('--flat', action='store_true', help='Zero curvature; independent of --reference')
    parser.add_argument('--center-uv', type=float, nargs=2, metavar=('U', 'V'), help='Reference arclength coordinates in [-1,1]^2')
    parser.add_argument('--depth-mm', type=float, help='Displacement along the anchor normal in mm')
    parser.add_argument('--device', help='cpu or cuda device; never silently falls back to CPU')
    parser.add_argument('--reference', action='store_true', help='Also render zero indentation with identical curvature and optics')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print scene/workload without GPU initialization or writes')
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.calibrate_lut:
        if args.reference:
            parser.error('--reference is not used for LUT calibration; calibration renders its own zero-depth baseline')
        if args.center_uv is not None:
            parser.error('--center-uv does not apply to a full LUT scan; scan locations come from calibration settings')
        forwarded = ['--config', str(args.config)]
        if args.calibration_config is not None:
            forwarded.extend(['--calibration-config', str(args.calibration_config)])
        if args.out is not None:
            forwarded.extend(['--out', str(args.out)])
        if args.dry_run:
            forwarded.append('--dry-run')
        if args.device is not None:
            forwarded.extend(['--device', args.device])
        if args.radius_mm is not None:
            forwarded.extend(['--curve-radius-mm', str(args.radius_mm)])
        if args.flat:
            forwarded.append('--flat')
        if args.depth_mm is not None:
            forwarded.extend(['--depth-mm', str(args.depth_mm)])
        from physical_model.cali_utils.cli import main as calibration_main
        exit_code = calibration_main(forwarded)
        if exit_code:
            raise SystemExit(exit_code)
        return
    if args.calibration_config is not None:
        parser.error('--calibration-config requires --calibrate-lut')
    config_path = args.config.expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding='utf-8-sig'))
        config = copy.deepcopy(raw)
        if config.get('backend', 'analytic') != 'analytic':
            raise ValueError('simulate_curve.py requires backend=analytic')
        if args.radius_mm is not None or args.flat:
            config['surface'].setdefault('bending', {'axis': 'x', 'direction_z': -1})['radius_mm'] = \
                None if args.flat else args.radius_mm
        if args.center_uv is not None:
            config['indentation'].pop('center_xy_mm', None)
            config['indentation']['center_uv'] = args.center_uv
        if args.depth_mm is not None:
            config['indentation']['depth_mm'] = args.depth_mm
        if args.device is not None:
            config['render']['device'] = args.device
        validate_config(config, allow_curve=True, config_path=config_path)
        output = (args.out or new_render_output_dir(
            'analytic', curved=bending_radius(config) is not None)).expanduser().resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f'Output directory is not empty: {output}')
        scene = describe_scene(config)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(f'{config_path}: {exc}')
    summary = {key: value for key, value in scene.items() if key != 'light_samples'}
    summary.update(config_path=str(config_path), output_dir=str(output),
                   device=config['render']['device'], reference=args.reference,
                   frame_count=2 if args.reference else 1, effective_config=config)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    # An exclusive claim also prevents simultaneous writers to an initially empty directory.
    with (output / '.curve_run').open('x', encoding='utf-8') as claim:
        claim.write(uuid.uuid4().hex)
    overrides = {key: value for key, value in vars(args).items()
                 if key not in ('config', 'out') and value is not None}
    signature_data = json.dumps({'scene': scene, 'config': config}, sort_keys=True, separators=(',', ':'))
    run = dict(status='running', backend='analytic', config_path=str(config_path),
               input_config=raw, config=config, cli_overrides=overrides,
               scene=scene, code=_code_record(), reference_same_curvature=True,
               config_signature=hashlib.sha256(signature_data.encode('utf-8')).hexdigest(), frames=[])
    _atomic_json(output / 'run.json', run)
    context = None
    started = time.perf_counter()
    try:
        if str(config['render']['device']).lower() != 'cpu':
            if bending_radius(config) is None:
                from .renderer_gpu import AnalyticGpuContext
                context = AnalyticGpuContext(config)
            else:
                from .curve_renderer_gpu import CurveGpuContext
                context = CurveGpuContext(config)
        for name in ['indented'] + (['reference'] if args.reference else []):
            cfg = copy.deepcopy(config)
            if name == 'reference':
                cfg['indentation']['depth_mm'] = 0.
            folder = output / name if args.reference else output
            folder.mkdir(exist_ok=True)
            frame_started = time.perf_counter()
            if context is None:
                sensor, radiance, gt = render(cfg, device='cpu')
                execution = {'device': 'cpu', 'mode': 'numpy',
                             'geometry_dtype': 'float64', 'radiometry_dtype': 'float64'}
            else:
                with context.torch.inference_mode():
                    sensor, radiance, gt, _ = context.render_tensors(config=cfg)
                    sensor, radiance = sensor.cpu().numpy(), radiance.cpu().numpy()
                    gt = {key: value.cpu().numpy() for key, value in gt.items()}
                execution = {'device': str(context.device), 'compile': context.compile_metrics(),
                             'cuda_graph_capture_count': context.cuda_graph_capture_count,
                             'cuda_graph_replay_count': context.cuda_graph_replay_count,
                             'cuda_graph_fallback_reason': context.cuda_graph_fallback_reason,
                             'oom_retry_count': context.oom_retry_count}
            if bending_radius(cfg) is None:
                gt = _flat_truth(gt, cfg)
            export_stl(folder / 'surface.stl', cfg)
            frame_scene = describe_scene(cfg)
            frame_signature = hashlib.sha256(json.dumps({'scene': frame_scene, 'config': cfg},
                sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
            extra = dict(frame_scene, execution=execution,
                         config_signature=frame_signature, run_signature=run['config_signature'], input_config=raw,
                         cli_overrides=overrides, code=run['code'], reference_same_curvature=True,
                         frame_kind=name, elapsed_seconds=time.perf_counter() - frame_started)
            save_result(folder, sensor, cfg, 'analytic', gt=gt, radiance=radiance, extra=extra)
            run['frames'].append({'kind': name, 'path': str(folder.relative_to(output)),
                                  'elapsed_seconds': extra['elapsed_seconds'],
                                  'valid_pixels': int(gt['mask'].sum()), 'execution': execution})
            _atomic_json(output / 'run.json', run)
            print(f'{name}: analytic -> {folder} ({extra["elapsed_seconds"]:.2f}s)', flush=True)
        run['status'] = 'complete'
    except Exception as exc:
        run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        run['elapsed_seconds'] = time.perf_counter() - started
        _atomic_json(output / 'run.json', run)
