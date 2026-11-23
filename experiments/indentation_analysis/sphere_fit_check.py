"""Render saved spherical-cap fits from each LUT through a dedicated CLI."""
from __future__ import annotations

import argparse
import json
from numbers import Real
from pathlib import Path
from typing import Sequence

import numpy as np

from .analysis import _baseline_direction, load_reconstruction
from .geometry import SphereFit, sphere_indentation_depth
from .visualization import SphereFitView, write_sphere_fit_overview


def _load_sphere_fit_views(report_paths: Sequence[Path]) -> list[SphereFitView]:
    """Read saved fitted regions, sphere parameters and surface projections.

    Masks follow the contact order and reconstruction grid saved by the
    analysis runner. All returned XYZ coordinates use camera-frame mm.
    """
    views = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        method = report_path.parent.name.removesuffix('_analysis')
        reconstruction = load_reconstruction(report_path.parent.parent / f'{method}.npz')
        selection_path = report_path.parent / 'fit_selection.npz'
        with np.load(selection_path, allow_pickle=False) as data:
            candidates = data['candidate_masks']
            fit_masks = data['fit_masks']
            indices = data['contact_index']
        contacts = report['contacts']
        expected_shape = (len(contacts), *reconstruction.valid.shape)
        if (candidates.shape != expected_shape or fit_masks.shape != expected_shape
                or candidates.dtype != np.bool_ or fit_masks.dtype != np.bool_):
            raise ValueError(f'{selection_path}: contact masks must be boolean with shape {expected_shape}')
        if (indices.shape != (len(contacts),)
                or not np.array_equal(indices, [contact['index'] for contact in contacts])):
            raise ValueError(f'{selection_path}: contact indices disagree with analysis.json')

        for candidate, fit_mask, contact in zip(candidates, fit_masks, contacts, strict=True):
            label = f"{method} | Contact {contact['index']}"
            if np.any(candidate & ~reconstruction.photometric_valid):
                raise ValueError(f'{label}: candidates must be photometrically valid')
            if np.any(fit_mask & ~candidate):
                raise ValueError(f'{label}: fitting mask must be a subset of candidates')
            if (int(candidate.sum()) != contact['selection']['candidate_count']
                    or int(fit_mask.sum()) != contact['sample_count']):
                raise ValueError(f'{label}: saved mask counts disagree with analysis.json')

            points = reconstruction.xyz[fit_mask]
            fit_data = contact['fit']
            fit = None
            if fit_data is not None:
                center = np.asarray(fit_data['center_xyz_mm'], dtype=np.float64)
                radius = fit_data['radius_mm']
                reported_radius = contact['sphere_radius_mm']
                rms = fit_data['radial_rms_mm']
                condition = fit_data['jacobian_condition']
                scalars = (radius, reported_radius, rms, condition)
                if (center.shape != (3,) or not np.isfinite(center).all()
                        or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                               or not np.isfinite(value) for value in scalars)
                        or radius <= 0 or reported_radius <= 0 or rms < 0 or condition <= 0
                        or not np.isclose(radius, reported_radius, rtol=1e-12, atol=1e-12)
                        or len(points) < 4):
                    raise ValueError(f'{label}: invalid saved sphere fit')
                actual_rms = np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius)**2))
                if not np.isclose(actual_rms, rms, rtol=1e-5, atol=1e-7):
                    raise ValueError(f'{label}: radial RMS disagrees with saved fitting points')
                fit = SphereFit(center, float(radius), float(rms), float(condition))

            projection_data = contact['projection']
            projection = None
            center_height, indentation_depth = None, None
            if projection_data is not None:
                projection = np.asarray(projection_data['xyz_mm'], dtype=np.float64)
                if fit is None or projection.shape != (3,) or not np.isfinite(projection).all():
                    raise ValueError(f'{label}: invalid saved surface projection')
                center_height, indentation_depth = sphere_indentation_depth(
                    fit.center_xyz_mm, fit.radius_mm, projection,
                    _baseline_direction(reconstruction, fit_mask))
            views.append(SphereFitView(
                label=label, candidate_xyz_mm=reconstruction.xyz[candidate],
                fit_xyz_mm=points, excluded_xyz_mm=reconstruction.xyz[candidate & ~fit_mask],
                fit=fit, projection_xyz_mm=projection,
                center_height_mm=center_height, indentation_depth_mm=indentation_depth,
                status=contact['selection'].get('method')))
    return views


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='读取各 LUT 已有的球冠拟合结果，生成三维拟合总图')
    parser.add_argument('batch_dir', type=Path, help='含重建 NPZ 和各方法 *_analysis 的 batch 目录')
    parser.add_argument('--output', type=Path, help='输出 PNG 路径，默认 batch_dir/analysis/sphere_fits.png')
    args = parser.parse_args(argv)
    batch = args.batch_dir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else batch/'analysis/sphere_fits.png'
    if output.suffix.lower() != '.png':
        parser.error('--output 需要 PNG 文件路径')
    reports = sorted(batch.glob('*_analysis/analysis.json'),
                     key=lambda path: (path.parent.name != 'table_analysis', path.parent.name))
    if not reports:
        parser.error(f'{batch}: 需要各方法的 *_analysis/analysis.json 和 fit_selection.npz')
    try:
        views = _load_sphere_fit_views(reports)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_sphere_fit_overview(output, views)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f'球冠拟合图生成失败：{exc}\n')
    print(f'{len(views)} 个拟合面板已保存：{output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
