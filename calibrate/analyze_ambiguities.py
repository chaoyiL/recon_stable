"""CLI for calibration ambiguity analysis."""
from __future__ import annotations

import argparse

from calibrate.utils.analyze_ambiguities import analyze_ambiguities


def main() -> None:
    parser = argparse.ArgumentParser(
        description='分析色差/法向歧义的规律、局部分支及改进实验，输出 PNG、JSON、CSV 和 NPZ')
    parser.add_argument('--work-dir', required=True, help='含本批LUT、样本、标注及frames/overall缓存的目录')
    parser.add_argument('--output-dir', help='默认输出到work-dir/ambiguity_analysis')
    parser.add_argument('--color-distance-threshold', type=float, default=.01)
    parser.add_argument('--normal-angle-threshold-degrees', type=float, default=20.)
    parser.add_argument('--dominance-fraction', type=float, default=.8,
                        help='外围主支撑占两支总支撑的最低比例，默认0.8')
    args = parser.parse_args()
    output = analyze_ambiguities(args.work_dir, args.output_dir,
        color_distance_threshold=args.color_distance_threshold,
        normal_angle_threshold_degrees=args.normal_angle_threshold_degrees,
        dominance_fraction=args.dominance_fraction)
    print(f'不良映射分析已保存: {output}')


if __name__ == '__main__':
    main()
