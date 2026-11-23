"""从同一构型的压入图批量圈选接触区，输出可复核的标注草稿和 PNG。"""

from __future__ import annotations

import argparse
from datetime import datetime
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

from optical_anno.generation_01.detector import fit_contact_background, propose_contact
from optical_anno.generation_01.review import REVIEW_COLORS, contact_overlay, contact_sheet, save_png
from utils.image_io import IMAGE_SUFFIXES, linear_rgb_to_tracking_bgr8, load_linear_image
from utils.region_annotations import RegionAnnotation, load_regions, region_masks, save_regions


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE.parent / 'assets'
MAX_BACKGROUND_FRAMES = 48
OVERVIEW_PAGE_SIZE = 24


def find_images(patterns: list[str]) -> list[Path]:
    """Resolve file, direct directory, or glob inputs in stable order."""
    suffixes = IMAGE_SUFFIXES | {'.npz'}
    paths = set()
    for pattern in patterns:
        source = Path(pattern).expanduser()
        matched = (source,) if source.exists() else (Path(name) for name in glob.iglob(str(source), recursive=True))
        for path in matched:
            candidates = path.iterdir() if path.is_dir() else (path,)
            paths.update(file.resolve() for file in candidates
                         if file.is_file() and file.suffix.lower() in suffixes)
    if not paths:
        raise ValueError('请提供压入图路径、目录或 glob')
    return sorted(paths)


def annotation_metrics(manual: RegionAnnotation, automatic: RegionAnnotation) -> dict:
    """Compare image-space proposals with a separate visual review."""
    width, height = manual.image_size
    masks = [region_masks(region, (height, width), outer_margin_px=0)['contact']
             for region in (manual, automatic)]
    return {
        'center_distance_px': float(np.linalg.norm(np.subtract(manual.center_px, automatic.center_px))),
        'iou': float(np.count_nonzero(masks[0] & masks[1]) / np.count_nonzero(masks[0] | masks[1])),
        'auto_to_visual_area_ratio': float(np.count_nonzero(masks[1]) / np.count_nonzero(masks[0])),
    }


def write_overviews(
    directory: Path, previews: list[tuple[str, Path]], crop: tuple[int, int, int, int]
) -> None:
    for start in range(0, len(previews), OVERVIEW_PAGE_SIZE):
        page = previews[start:start + OVERVIEW_PAGE_SIZE]
        panels = [(name, cv2.imread(str(path))) for name, path in page]
        name = 'overview.png' if len(previews) <= OVERVIEW_PAGE_SIZE else f'overview_{start // OVERVIEW_PAGE_SIZE + 1:03d}.png'
        save_png(directory / name, contact_sheet(panels, crop_xyxy=crop))


def suggest_batch(paths: list[Path], output: Path, manual_path: Path | None = None) -> dict:
    """Estimate from press-only observations, then optionally compare exported drafts."""
    root = Path(os.path.commonpath([path.parent for path in paths]))
    keys = [path.relative_to(root).as_posix() for path in paths]
    anchor_indices = np.linspace(0, len(paths) - 1, min(len(paths), MAX_BACKGROUND_FRAMES), dtype=int)
    background_images = [load_linear_image(paths[index]) for index in anchor_indices]
    background = fit_contact_background(background_images)
    del background_images
    output.mkdir(parents=True, exist_ok=False)
    manual = {} if manual_path is None else load_regions(manual_path, image_root=root)
    regions, rows = [], []
    automatic_previews, comparison_previews = [], []
    for index, (key, path) in enumerate(zip(keys, paths, strict=True)):
        image = load_linear_image(path)
        guess, diagnostic = propose_contact(image, background)
        region = None
        if guess is not None:
            center, axes, angle = guess
            region = RegionAnnotation(
                key, image.shape[1::-1], center_px=center, shape='ellipse',
                semi_axes_px=axes, angle_degrees=angle, semantics='contact_boundary'
            )
            regions.append(region)
        row = {'image': key, 'diagnostics': diagnostic}
        title = f'{index:02d} {path.stem.removeprefix("capture_")}'
        filename = f'{index:02d}_{path.stem}.png'
        image_bgr = linear_rgb_to_tracking_bgr8(image)
        auto_overlay = contact_overlay(image_bgr, [('AUTOMATIC', region, REVIEW_COLORS[1])])
        preview_path = output / 'frames' / filename
        save_png(preview_path, auto_overlay)
        automatic_previews.append((title, preview_path))

        visual_region = manual.get(key)
        if visual_region is not None:
            comparison_path = output / 'comparison' / 'frames' / filename
            comparison = contact_overlay(image_bgr, [
                ('VISUAL', visual_region, REVIEW_COLORS[0]),
                ('AUTOMATIC', region, REVIEW_COLORS[1]),
            ])
            if region is not None:
                row['comparison'] = annotation_metrics(visual_region, region)
                title += f"  IoU {row['comparison']['iou']:.2f}"
            save_png(comparison_path, comparison)
            comparison_previews.append((title, comparison_path))
        rows.append(row)

    save_regions(output / 'regions.yaml', regions, image_root=root)
    write_overviews(output, automatic_previews, background.overview_crop)
    if comparison_previews:
        write_overviews(output / 'comparison', comparison_previews, background.overview_crop)
    scores = [row['comparison'] for row in rows if 'comparison' in row]
    report = {
        'images': len(paths), 'automatic_drafts': len(regions),
        'background_images': [keys[i] for i in anchor_indices],
        'comparison_count': len(scores), 'per_image': rows,
    }
    if scores:
        report['comparison_medians'] = {
            key: float(np.median([score[key] for score in scores])) for key in scores[0]
        }
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                                       encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', default=[str(ROOT / 'assets' / 'cali_norm_pics')],
                        help='同一拍摄构型的压入图路径、目录或 glob')
    parser.add_argument('--output', type=Path,
                        help=f'新建输出目录，默认 optical_anno/assets/{HERE.name}/runs/<时间戳>')
    parser.add_argument('--compare-with', type=Path, help='独立标注 YAML，用于输出叠加对照和误差')
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = args.output or ASSETS / HERE.name / 'runs' / timestamp
    try:
        report = suggest_batch(find_images(args.images), output.expanduser().resolve(), args.compare_with)
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        parser.exit(1, f'自动圈选压入区：{error}\n')
    print(f"已生成 {report['automatic_drafts']}/{report['images']} 张标注草稿，保存到 {output}")


if __name__ == '__main__':
    main()
