"""Render a planar spherical indentation with the analytic tactile model."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import uuid

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physical_model.model_utils.backend_layout import new_render_output_dir, simulation_config_path
from physical_model.model_utils.config import load_config, validate_config
from physical_model.model_utils.geometry import export_stl
from physical_model.model_utils.light_distribution import analytic_light_samples
from physical_model.model_utils.renderer import render, save_result
from physical_model.model_utils.surface_coordinates import bending_radius


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=simulation_config_path())
    parser.add_argument('--backend', choices=['analytic'], default='analytic')
    parser.add_argument('--out', type=Path, help='Output directory; default is a unique physical_model/render_results run')
    parser.add_argument('--device', help='cpu or cuda[:index]; no automatic device fallback')
    parser.add_argument('--reference', action='store_true', help='Also render zero indentation with the same optics')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print effective settings without rendering or writing')
    return parser


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _scene_summary(config: dict) -> dict:
    light_count = sum(len(analytic_light_samples(light, config)[0]) for light in config['lights'])
    return {
        'geometry': 'planar_spherical_cap',
        'image_pixels': config['camera']['width_px'] * config['camera']['height_px'],
        'light_sample_count': light_count,
        'device': config['render']['device'],
        'reference': False,
        'frame_count': 1,
        'effective_config': config,
    }


def _render_frame(config: dict, folder: Path, context=None) -> dict:
    started = time.perf_counter()
    if context is None:
        sensor, radiance, gt = render(config, device='cpu')
        execution = {'device': 'cpu', 'mode': 'numpy'}
    else:
        with context.torch.inference_mode():
            sensor, radiance, gt, _ = context.render_tensors(config=config)
            sensor, radiance = sensor.detach().cpu().numpy(), radiance.detach().cpu().numpy()
            gt = {key: value.detach().cpu().numpy() for key, value in gt.items()}
        execution = {
            'device': str(context.device),
            'compile': context.compile_metrics(),
            'cuda_graph_capture_count': context.cuda_graph_capture_count,
            'cuda_graph_replay_count': context.cuda_graph_replay_count,
            'cuda_graph_fallback_reason': context.cuda_graph_fallback_reason,
            'oom_retry_count': context.oom_retry_count,
        }
    export_stl(folder / 'surface.stl', config)
    save_result(folder, sensor, config, 'analytic', gt=gt, radiance=radiance,
                extra={'execution': execution, 'elapsed_seconds': time.perf_counter() - started})
    return {'valid_pixels': int(gt['mask'].sum()), 'execution': execution,
            'elapsed_seconds': time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    try:
        config = load_config(config_path)
        if config.get('backend', 'analytic') != args.backend:
            raise ValueError(f'--backend {args.backend} disagrees with config backend {config.get("backend")}')
        if args.device is not None:
            config['render']['device'] = args.device
        validate_config(config)
        if bending_radius(config) is not None:
            raise ValueError('simulate.py renders planar surfaces; use simulate_curve.py for bending')
        output = (args.out or new_render_output_dir('analytic')).expanduser().resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f'Output directory is not empty: {output}')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(f'{config_path}: {exc}')

    scene = _scene_summary(config)
    scene['reference'] = args.reference
    scene['frame_count'] = 2 if args.reference else 1
    summary = {**scene, 'config_path': str(config_path), 'output_dir': str(output)}
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    output.mkdir(parents=True, exist_ok=True)
    claim = output / '.analytic_run'
    try:
        with claim.open('x', encoding='utf-8') as stream:
            stream.write(uuid.uuid4().hex)
    except FileExistsError as exc:
        raise ValueError(f'Output directory is already claimed: {output}') from exc

    run = {'status': 'running', 'backend': 'analytic', 'config_path': str(config_path),
           'config': config, 'reference': args.reference, 'frames': []}
    _atomic_json(output / 'run.json', run)
    started = time.perf_counter()
    context = None
    try:
        if config['render']['device'].lower() != 'cpu':
            from physical_model.model_utils.renderer_gpu import AnalyticGpuContext
            context = AnalyticGpuContext(config, config['render']['device'])
        for kind in (['indented', 'reference'] if args.reference else ['indented']):
            frame_config = copy.deepcopy(config)
            if kind == 'reference':
                frame_config['indentation']['depth_mm'] = 0.0
            folder = output / kind if args.reference else output
            folder.mkdir(exist_ok=True)
            frame = _render_frame(frame_config, folder, context)
            frame['kind'] = kind
            frame['path'] = str(folder.relative_to(output)) if args.reference else '.'
            run['frames'].append(frame)
            _atomic_json(output / 'run.json', run)
            print(f"{kind}: analytic -> {folder} ({frame['elapsed_seconds']:.2f}s)", flush=True)
        run['status'] = 'complete'
    except Exception as exc:
        run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        run['elapsed_seconds'] = time.perf_counter() - started
        _atomic_json(output / 'run.json', run)


if __name__ == '__main__':
    main()
