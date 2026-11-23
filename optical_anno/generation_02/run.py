"""用球面实时求交器批量标注压入区，并生成原图叠加预览。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from calibrate.draw_contact_area import DEFAULT_GEOMETRY_IMAGE
from calibrate.utils.norm_pipeline import (
    CalibrationSettings,
    FrameCalibrationData,
    initial_calibration_surface,
    load_frame_calibration_data,
    load_manifest,
    parse_settings,
    run_prepare,
)
from optical_anno.generation_01.detector import ContactBackground, fit_contact_background
from optical_anno.generation_01.review import REVIEW_COLORS, contact_overlay, save_png
from optical_anno.generation_01.run import (
    ASSETS,
    MAX_BACKGROUND_FRAMES,
    annotation_metrics,
    find_images,
    write_overviews,
)
from optical_anno.generation_02.evidence import ContactEvidence, contact_evidence
from optical_anno.generation_02.fit import fit_contact
from utils.image_io import linear_rgb_to_tracking_bgr8, load_linear_image
from utils.region_annotations import RegionAnnotation, load_regions, save_regions


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = ROOT / 'config.yaml'
MAXIMUM_COLOR = (0, 160, 255)
COMPARISON_COLOR = (255, 200, 60)
EvidenceFactory = Callable[[FrameCalibrationData, ContactBackground, CalibrationSettings], ContactEvidence]


class FrameBackground(Protocol):
    overview_crop: tuple[int, int, int, int]
    image_keys: tuple[str, ...]

    def for_frame(self, frame: FrameCalibrationData) -> ContactBackground: ...


def solver_overlay(
    image: np.ndarray,
    region: RegionAnnotation | None,
    limit: dict | None,
    comparison: RegionAnnotation | None = None,
) -> np.ndarray:
    """Draw the same orange maximum as the manual editor, then the draft."""
    preview = image.copy()
    if limit is not None and limit['maximum_boundary_px'] is not None:
        points = np.rint(limit['maximum_boundary_px']).astype(np.int32)
        cv2.polylines(preview, [points], True, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.polylines(preview, [points], True, MAXIMUM_COLOR, 5, cv2.LINE_AA)
    label = 'SPHERE SOLVER' if region is None else f'CONTACT  h={region.indentation_depth_mm:.2f} mm'
    annotations = [(label, region, REVIEW_COLORS[1])]
    if comparison is not None:
        annotations.insert(0, ('COMPARISON DRAFT', comparison, COMPARISON_COLOR))
    preview = contact_overlay(preview, annotations)
    if limit is not None:
        maximum_label = 'MAX FOOTPRINT' if limit['maximum_boundary_px'] is not None else 'MAX OUTLINE UNAVAILABLE'
        text = f"{maximum_label}  ball D={limit['sphere_diameter_mm']:g} mm  h={limit['maximum_depth_mm']:g} mm"
        cv2.putText(preview, text, (10, 24 + 21 * len(annotations)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, MAXIMUM_COLOR, 1, cv2.LINE_AA)
    return preview


def geometry_settings(
    config: Path,
    output: Path,
    *,
    geometry_dir: Path | None,
    geometry_image: Path | None,
    images: list[str] | None,
) -> tuple[CalibrationSettings, dict]:
    paths = None if images is None else [str(path) for path in find_images(images)]
    settings = parse_settings(config, dict(images=paths, reference_image=geometry_image or DEFAULT_GEOMETRY_IMAGE))
    settings.require_sphere()
    settings = replace(
        settings,
        work_dir=geometry_dir if geometry_dir is not None else output / 'geometry',
        annotations_file=output / 'regions.yaml',
        single_frame_markers=True,
    )
    manifest = run_prepare(config, settings) if geometry_dir is None else load_manifest(settings)
    return settings, manifest


def fit_frame(
    entry: dict, image: np.ndarray, background_for_frame: Callable[[FrameCalibrationData], ContactBackground],
    anchor: FrameCalibrationData, settings: CalibrationSettings,
    *, evidence_factory: EvidenceFactory | None = None,
) -> tuple[RegionAnnotation | None, dict]:
    if entry['status'] != 'prepared':
        return None, {'status': 'abstain', 'reason': 'geometry_unavailable',
                      'detail': entry.get('message', entry.get('reason', ''))}
    path = settings.work_dir / 'prepared' / f"{entry['index']:06d}.npz"
    try:
        prepared = load_frame_calibration_data(path, settings)
        if (prepared.image_key != entry['key'] or prepared.image_index != entry['index']
                or prepared.chart.chart_id != anchor.chart.chart_id
                or not np.array_equal(prepared.linear_rgb, image)):
            raise ValueError('图像或几何缓存与清单不一致，请重新准备几何')
        frame = initial_calibration_surface(prepared, anchor)
        background = background_for_frame(frame)
        evidence = (contact_evidence(image, background) if evidence_factory is None
                    else evidence_factory(frame, background, settings))
        region, diagnostics = fit_contact(
            frame, anchor, settings, evidence.seed_px, evidence.score_boundary,
            radius_hint_px=evidence.radius_hint_px,
        )
        diagnostics.update(matched_markers=int(prepared.tracking['valid'].sum()),
                           seed_px=list(evidence.seed_px), radius_hint_px=float(evidence.radius_hint_px),
                           photometric_evidence=evidence.diagnostics)
        return region, diagnostics
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        return None, {'status': 'abstain', 'reason': 'fit_unavailable', 'detail': str(error)}


def annotate_batch(
    output: Path, settings: CalibrationSettings, manifest: dict, comparison_file: Path | None,
    *, material_background_factory: Callable[[list[FrameCalibrationData]], FrameBackground] | None = None,
    evidence_factory: EvidenceFactory | None = None,
    generation: str = 'generation_02',
) -> list[RegionAnnotation]:
    root = Path(manifest['image_root'])
    entries = manifest['images']
    paths = [root / row['key'] for row in entries]
    if not paths:
        raise ValueError('几何清单中需要压入图')
    anchor = load_frame_calibration_data(settings.work_dir / 'reference_surface.npz', settings)
    if anchor.chart.chart_id != manifest['coordinate_chart_id']:
        raise ValueError('几何锚点与清单使用了不同的材料坐标')
    candidates = list(range(len(entries)))
    if material_background_factory is not None:
        candidates = [i for i in candidates if entries[i]['status'] == 'prepared']
        if not candidates:
            raise ValueError('底色估计需要已准备的曲面几何')
    positions = np.linspace(0, len(candidates) - 1, min(len(candidates), MAX_BACKGROUND_FRAMES), dtype=int)
    indices = [candidates[i] for i in positions]
    frame_background = None
    if material_background_factory is None:
        background = fit_contact_background([load_linear_image(paths[i]) for i in indices])
        overview_crop = background.overview_crop
        background_method = 'pixel_median'
        background_keys = [entries[i]['key'] for i in indices]

        def background_for_frame(_frame: FrameCalibrationData) -> ContactBackground:
            return background
    else:
        samples = []
        for i in indices:
            entry = entries[i]
            path = settings.work_dir / 'prepared' / f"{entry['index']:06d}.npz"
            prepared = load_frame_calibration_data(path, settings)
            if prepared.image_key != entry['key'] or prepared.image_index != entry['index']:
                raise ValueError('底色样本的几何缓存与清单不一致，请重新准备几何')
            samples.append(initial_calibration_surface(prepared, anchor))
        frame_background = material_background_factory(samples)
        del samples
        background_method = 'material_geometry_leave_one_out'
        background_for_frame = frame_background.for_frame
        background_keys = list(frame_background.image_keys)
    comparison = {} if comparison_file is None else load_regions(comparison_file, image_root=root)
    regions, rows = [], []
    previews, comparisons = [], []
    for entry, path in zip(entries, paths, strict=True):
        index, key = entry['index'], entry['key']
        image = load_linear_image(path)
        region, diagnostics = fit_frame(entry, image, background_for_frame, anchor, settings,
                                        evidence_factory=evidence_factory)
        row = dict(index=index, image=key, diagnostics=diagnostics)
        title = f'{index:02d} {path.stem.removeprefix("capture_")}'
        if region is not None:
            regions.append(region)
            row.update(center_px=region.center_px, indentation_depth_mm=region.indentation_depth_mm)
            title += f'  h={region.indentation_depth_mm:.2f} mm'
        else:
            title += '  REVIEW'
        filename = f'{index:02d}_{path.stem}.png'
        image_bgr = linear_rgb_to_tracking_bgr8(image)
        limit = diagnostics.get('footprint_limit')
        preview_path = output / 'frames' / filename
        save_png(preview_path, solver_overlay(image_bgr, region, limit))
        previews.append((title, preview_path))
        earlier = comparison.get(key)
        if earlier is not None:
            comparison_path = output / 'comparison' / 'frames' / filename
            save_png(comparison_path, solver_overlay(image_bgr, region, limit, comparison=earlier))
            comparisons.append((title, comparison_path))
            if region is not None:
                metrics = annotation_metrics(earlier, region)
                metrics['output_to_comparison_area_ratio'] = metrics.pop('auto_to_visual_area_ratio')
                row['comparison'] = metrics
        rows.append(row)
        outcome = ' | '.join(diagnostics[key] for key in ('status', 'reason', 'detail') if diagnostics.get(key))
        print(f'{index:02d} {path.stem} {outcome}', flush=True)

    save_regions(output / 'regions.yaml', regions, image_root=root)
    if frame_background is not None:
        overview_crop = frame_background.overview_crop
    write_overviews(output, previews, overview_crop)
    if comparisons:
        write_overviews(output / 'comparison', comparisons, overview_crop)
    report = dict(
        generation=generation, geometry_dir=str(settings.work_dir), background_method=background_method,
        marker_matching=manifest.get('marker_matching', 'reference_tracking'),
        coordinate_chart_id=anchor.chart.chart_id, images=len(entries),
        prepared_geometry=sum(entry['status'] == 'prepared' for entry in entries),
        automatic_drafts=len(regions), sphere_radius_mm=settings.sphere_radius_mm,
        sphere_diameter_mm=2 * settings.sphere_radius_mm, maximum_depth_mm=settings.sphere_radius_mm,
        sphere_side=settings.sphere_side, background_images=background_keys,
        per_image=rows,
    )
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                                       encoding='utf-8')
    return regions


def main(
    *, material_background_factory: Callable[[list[FrameCalibrationData]], FrameBackground] | None = None,
    evidence_factory: EvidenceFactory | None = None,
    generation: str = HERE.name,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='胶皮和压球参数的 YAML 配置')
    parser.add_argument('--geometry-dir', type=Path, help='已准备的几何缓存目录')
    parser.add_argument('--images', nargs='+', help='新一批压入图路径、目录或 glob')
    parser.add_argument('--geometry-image', type=Path, help='准备新几何时的几何锚点图')
    parser.add_argument('--output', type=Path,
                        help=f'新建输出目录，默认 optical_anno/assets/{generation}/runs/<时间戳>')
    parser.add_argument('--compare-with', type=Path, help='生成与另一份标注草稿的叠加图')
    parser.add_argument('--review', action='store_true', help='生成草稿后打开原标注窗口实时复核')
    args = parser.parse_args()
    if args.geometry_dir is not None and (args.images or args.geometry_image):
        parser.error('--geometry-dir 读取已有几何。用 --images 或 --geometry-image 准备新几何。')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output or ASSETS / generation / 'runs' / timestamp).expanduser().resolve()
    geometry_dir = args.geometry_dir
    try:
        output.mkdir(parents=True, exist_ok=False)
        settings, manifest = geometry_settings(
            (args.config or CONFIG).expanduser().resolve(), output,
            geometry_dir=geometry_dir.expanduser().resolve() if geometry_dir is not None else None,
            geometry_image=args.geometry_image, images=args.images,
        )
        regions = annotate_batch(output, settings, manifest, args.compare_with,
                                 material_background_factory=material_background_factory,
                                 evidence_factory=evidence_factory,
                                 generation=generation)
        print(f'已生成 {len(regions)}/{len(manifest["images"])} 张球面草稿，保存到 {output}', flush=True)
        if args.review:
            from calibrate.utils.manual_norm_regions import annotate_images

            reviewed = output / 'reviewed_regions.yaml'
            save_regions(reviewed, regions, image_root=manifest['image_root'])
            annotate_images(replace(settings, annotations_file=reviewed), observed_only=True)
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        parser.exit(1, f'球面压入区标注：{error}\n')


if __name__ == '__main__':
    main()
