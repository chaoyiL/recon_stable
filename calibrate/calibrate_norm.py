"""Image contact calibration and compatible external signed-color sample fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
from utils.config import load_config
from calibrate.utils.norm_pipeline import (
    _LUTValidationFailure, fit_and_validate_samples, parse_settings, run_prepare, run_samples,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent/'config.yaml'


def main() -> None:
    parser = argparse.ArgumentParser(description='压球图片色差—法向标定，或外部色差/坡度样本 LUT 拟合')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input', help='外部样本 NPZ；覆盖 normal_calibration.input_file，选择 samples 模式')
    source.add_argument('--images', nargs='+', help='压球线性 NPZ 或旧相机图片/glob，选择 images 模式')
    parser.add_argument('--output', help='通过验证后额外导出模型及其配套网格')
    parser.add_argument('--stage', choices=['prepare', 'annotate', 'samples', 'fit', 'compare', 'all'])
    parser.add_argument('--reference-image', help='本批次固定底色图，支持线性 NPZ 或旧相机图片')
    parser.add_argument('--sphere-radius-mm', type=float)
    parser.add_argument('--sphere-side', choices=['positive_normal', 'negative_normal'])
    parser.add_argument('--annotations', help='标注 YAML')
    parser.add_argument('--edit', action='store_true', help='重新编辑 accepted/skip 标注')
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    raw = load_config(config_path).get('normal_calibration')
    
    if not isinstance(raw, dict):
        parser.error('config.yaml 缺少 normal_calibration 配置段')
    mode = 'samples' if args.input else ('images' if args.images else raw.get('input_mode', 'samples'))
    if mode not in {'images', 'samples'}:
        parser.error('normal_calibration.input_mode must be images or samples')

    stage = args.stage or ('all' if mode == 'images' else 'fit')
    if mode == 'samples' and stage not in {'fit', 'compare'}:
        parser.error('外部样本模式只支持 --stage fit 或 compare')
    if mode == 'samples' and not (args.input or raw.get('input_file')):
        parser.error('样本模式需要 --input 或 normal_calibration.input_file')
    settings = parse_settings(config_path, dict(input_file=args.input, output=args.output, images=args.images,
        reference_image=args.reference_image, sphere_radius_mm=args.sphere_radius_mm,
        sphere_side=args.sphere_side, annotations_file=args.annotations))
    if stage in {'annotate', 'samples', 'all'}:
        settings.require_sphere()
    if stage in {'prepare', 'all'}:
        run_prepare(config_path, settings)
    if stage in {'annotate', 'all'}:
        from calibrate.utils.manual_norm_regions import annotate_images
        if not annotate_images(settings, edit=args.edit):
            print('标注草稿已保存。重新运行 annotate 可继续。')
            return
    if stage in {'samples', 'all'}:
        run_samples(settings)

    if stage in {'fit', 'all'}:
        try:
            report = fit_and_validate_samples(settings.input_file, settings)
        except _LUTValidationFailure as error:
            print(f"法向模型未通过验证，候选模型: {error.report.get('model_file')}")
            print(f"验证报告: {error.report['report_file']}")
            raise SystemExit(str(error)) from error
        print(f"法向模型已保存: {report['model_file']}；validation_status={report['status']}")
        if report['status'] == 'passed':
            print(f"验证报告: {report['report_file']}")
            if report.get('exported_model_file'):
                print(f"额外导出: {report['exported_model_file']}")
    if stage == 'compare':
        from calibrate.utils.compare_normal_models import compare_normal_models
        comparison = compare_normal_models(settings)
        print(f"对照结果: {comparison['summary_file']}")


if __name__ == '__main__':
    main()
