"""标注压球区域并训练光场预测 MLP 与 MLP8 法向模型。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from calibrate.utils.norm_pipeline import (
    CalibrationSettings,
    FrameCalibrationData,
    FrameCalibrationError,
    _coordinate_metadata,
    _reference_sampling_grid_values,
    _save_npz,
    finalize_observed_image,
    load_frame_calibration_data,
    load_manifest,
    parse_settings,
    run_prepare,
)
from utils.calibration_store import create_run, stage_reference_chart
from utils.config import get_curvature_smoothing_half_width_mm, load_config
from utils.region_annotations import load_regions, save_regions

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'optical_learning' / 'assets'
DEFAULT_GEOMETRY_IMAGE = ROOT / 'assets' / 'reference_flat.npz'


def import_contact_suggestions(settings: CalibrationSettings, manifest: dict, path: Path) -> int:
    """Use independent image-space drafts to initialize the sphere editor."""
    root = manifest['image_root']
    suggestions = load_regions(path, image_root=root)
    regions = load_regions(settings.annotations_file, image_root=root)
    count = 0
    for entry in manifest['images']:
        key = entry['key']
        suggestion = suggestions.get(key)
        if (entry['status'] == 'prepared' and key not in regions and suggestion is not None
                and suggestion.status == 'draft' and suggestion.semantics == 'contact_boundary'):
            regions[key] = suggestion
            count += 1
    if count:
        save_regions(settings.annotations_file, regions, image_root=root)
    return count


def collect_contact_records(
    settings: CalibrationSettings, manifest: dict, anchor: FrameCalibrationData
) -> list[dict]:
    """Recompute the final base and sphere labels for each accepted annotation."""
    regions = load_regions(settings.annotations_file, image_root=manifest['image_root'])
    accepted = [
        entry for entry in manifest['images'] if entry['status'] == 'prepared'
        and entry['key'] in regions and regions[entry['key']].status == 'accepted'
    ]
    if len(accepted) < 3:
        raise ValueError(
            f'当前有 {len(accepted)} 张已确认标注，按整图划分训练和验证至少需 3 张。'
            '请加 --edit 重开标注，圈出压入区并用 Enter 确认。'
        )
    records = []
    for entry in accepted:
        key = entry['key']
        path = settings.work_dir / 'prepared' / f"{entry['index']:06d}.npz"
        prepared = load_frame_calibration_data(path, settings)
        if prepared.image_key != key or prepared.image_index != entry['index']:
            raise ValueError(f'{key} 的几何缓存与清单不一致，请用 --images 指定本批图像重新准备')
        try:
            records.append(finalize_observed_image(prepared, anchor, regions[key], settings))
        except FrameCalibrationError as error:
            raise ValueError(f'{key} 的标注需复核：{error}。可用 --edit 重开标注。') from error
        print(f"标注 {key}，球面有效样本 {records[-1]['metrics']['sample_count']}", flush=True)
    return records


def train(
    config: Path, settings: CalibrationSettings, lut_workspace: Path, background_workspace: Path
) -> dict:
    from optical_learning.calibration import train_contact_models

    manifest = load_manifest(settings)
    anchor = load_frame_calibration_data(settings.work_dir / 'reference_surface.npz', settings)
    if anchor.chart.chart_id != manifest['coordinate_chart_id']:
        raise ValueError('几何锚点与清单的材料坐标不一致，请用 --images 指定本批图像重新准备')
    settings = replace(settings, reference_image=anchor.source_image)
    records = collect_contact_records(settings, manifest, anchor)
    run = create_run(lut_workspace, 'mlp8')
    stage_reference_chart(run, anchor.chart_file)
    metadata = _coordinate_metadata(anchor, run.model_file)
    metadata['reference_image'] = str(anchor.source_image)
    _save_npz(run.grid_file, **_reference_sampling_grid_values(settings, metadata, output_dir=run.directory))
    smoothing = get_curvature_smoothing_half_width_mm(load_config(config))
    return train_contact_models(
        records, settings, run.directory,
        background_directory=background_workspace / run.directory.name,
        curvature_smoothing_half_width_mm=smoothing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--images', nargs='+', help='压球图路径或 glob，默认读取 normal_calibration.images')
    parser.add_argument('--geometry-image', type=Path, help='材料网格的几何锚点，默认 assets/reference_flat.npz')
    parser.add_argument('--run-name', help='可选批次名，在两个输出目录下使用同名子目录')
    parser.add_argument('--edit', action='store_true', help='重新打开已确认和跳过的标注')
    parser.add_argument('--suggestions', type=Path, help='将标注建议 YAML 用于初始化尚未标注的图')
    args = parser.parse_args()
    if args.run_name is not None and (
        args.run_name in ('', '.', '..') or Path(args.run_name).name != args.run_name
    ):
        parser.error('--run-name 须为一个目录名')

    config = args.config.expanduser().resolve()
    batch = args.run_name or '.'
    lut_workspace = ASSETS / 'lut_cali' / batch
    background_workspace = ASSETS / 'background_cali' / batch
    work = lut_workspace / 'geometry'
    manifest_path = work / 'manifest.json'
    try:
        settings = replace(
            parse_settings(config, dict(
                images=args.images,
                reference_image=args.geometry_image or DEFAULT_GEOMETRY_IMAGE,
            )),
            work_dir=work,
            input_file=work / 'normal_samples.npz',
            output=work / 'normal_lut.npz',
            annotations_file=lut_workspace / 'contact_regions.yaml',
            lut_dir=lut_workspace,
            method='mlp8',
            single_frame_markers=True,
            export_output=False,
        )
        settings.require_sphere()
        if args.images or args.geometry_image or not manifest_path.is_file():
            manifest = run_prepare(config, settings)
        else:
            manifest = load_manifest(settings)
        prepared = sum(entry['status'] == 'prepared' for entry in manifest['images'])
        print(f"几何准备通过 {prepared}/{len(manifest['images'])} 张，清单 {manifest_path}", flush=True)
        if not prepared:
            raise ValueError(f'本批图像需要能跟踪的几何锚点，逐图原因见 {manifest_path}')
        if args.suggestions:
            count = import_contact_suggestions(settings, manifest, args.suggestions)
            print(f'已导入 {count} 张压入区草稿，供球面编辑器复核', flush=True)

        from calibrate.utils.manual_norm_regions import annotate_images

        if not annotate_images(settings, edit=args.edit, observed_only=True):
            print(f'标注草稿已保存到 {settings.annotations_file}')
            return
        report = train(config, settings, lut_workspace, background_workspace)
        print(f"光场 MLP 已保存到 {report['background']['model_file']}")
        print(f"MLP8 已保存到 {report['normal']['model_file']}（验证状态 {report['status']}）")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'压入区标定：{error}\n')


if __name__ == '__main__':
    main()
