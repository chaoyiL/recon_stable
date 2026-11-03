"""CLI for calibration visualizations."""
from __future__ import annotations

import argparse
from pathlib import Path

from calibrate.utils.norm_pipeline import parse_settings
from calibrate.utils.visualize_norm import generate_calibration_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description='生成本批法向标定的色差图、p/q 映射和结果汇总')
    parser.add_argument('--config', default=Path(__file__).resolve().parents[1]/'config.yaml')
    parser.add_argument('--calibration', help='覆盖已通过标定或验证失败的诊断 LUT 路径')
    parser.add_argument('--samples', help='覆盖配套 normal_samples.npz 路径')
    parser.add_argument('--work-dir', help='覆盖配套 frames/overall 和参考缓存目录')
    parser.add_argument('--output-dir', help='默认输出到 work_dir/visualization')
    parser.add_argument('--color-distance-threshold', type=float, default=.01,
                        help='不良映射的有符号线性 RGB 欧氏距离上限，默认 0.01')
    parser.add_argument('--normal-angle-threshold-degrees', type=float, default=20.,
                        help='不良映射的局部法向夹角下限（严格大于），默认 20 度')
    args = parser.parse_args()
    settings = parse_settings(args.config)
    output = generate_calibration_visualizations(args.calibration or settings.output,
                                                args.samples or settings.input_file,
                                                args.work_dir or settings.work_dir, args.output_dir,
                                                color_distance_threshold=args.color_distance_threshold,
                                                normal_angle_threshold_degrees=args.normal_angle_threshold_degrees)
    print(f'法向标定可视化已保存: {output}')


if __name__ == '__main__':
    main()
