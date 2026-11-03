"""Command-line entry point for analytic signed normal-LUT calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from physical_model.model_utils.backend_layout import simulation_config_path
from physical_model.model_utils.config import validate_config
from .calibration_pipeline import load_calibration_config, run_calibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Scan analytic sphere poses and fit a signed linear-RGB to slope LUT')
    parser.add_argument('--config', '--simulation-config', dest='config', type=Path,
                        help='Analytic simulation config (default: configs/analytic.json)')
    parser.add_argument('--calibration-config', type=Path,
                        help='Shared scan, fit and validation settings')
    parser.add_argument('--out', type=Path,
                        help='Output run directory (must not already contain files)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate settings and print scan workload without rendering')
    parser.add_argument('--device',
                        help='Renderer device: cpu or cuda[:index] (default: config value)')
    parser.add_argument('--radius-mm', type=float,
                        help='Override indentation sphere radius')
    curvature = parser.add_mutually_exclusive_group()
    curvature.add_argument('--flat', action='store_true',
                           help='Override the surface to a plane')
    curvature.add_argument('--curve-radius-mm', type=float,
                           help='Override the cylindrical surface bending radius')
    parser.add_argument('--depth-mm', type=float,
                        help='Calibrate only this indentation depth')
    return parser


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.radius_mm is not None:
        config['indentation']['radius_mm'] = args.radius_mm
    if args.flat:
        config.setdefault('surface', {}).setdefault('bending', {})['radius_mm'] = None
    if args.curve_radius_mm is not None:
        config.setdefault('surface', {}).setdefault('bending', {})[
            'radius_mm'] = args.curve_radius_mm
    if args.depth_mm is not None:
        config['indentation']['depth_mm'] = args.depth_mm
        scan = config['calibration']['scan']
        scan['depth_min_mm'] = args.depth_mm
        scan['depth_max_mm'] = args.depth_mm
    if args.device is not None:
        if re.fullmatch(r'cpu|cuda(?::[0-9]+)?', args.device.lower()) is None:
            raise ValueError('--device must be cpu or cuda[:index]')
        config['render']['device'] = args.device.lower()
    return config


def main(argv: list[str] | None = None) -> None:
    """Run calibration; accept argv so simulate_curve can delegate directly."""
    args = _parser().parse_args(argv)
    config_path = (args.config or simulation_config_path('analytic')).expanduser().resolve()
    config = load_calibration_config(config_path, args.calibration_config)
    config = _apply_overrides(config, args)
    validate_config(config, allow_curve=True, allow_contact_tangent=True)
    result = run_calibration(config, config_path, output_override=args.out,
                             dry_run=args.dry_run)
    print(json.dumps(result if args.dry_run else {
        'status': result['status'],
        'run_status': result.get('run_status'),
        'output_dir': result.get('output_dir'),
        'run_signature': result.get('run_signature'),
        'validation': result.get('validation'),
        'artifacts': result.get('artifacts'),
    }, indent=2, ensure_ascii=False))
    if not args.dry_run and result['status'] != 'passed':
        raise SystemExit(2)
