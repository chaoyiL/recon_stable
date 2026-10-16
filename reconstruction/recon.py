"""单张图片的整体曲面与局部形变重建入口。"""
from __future__ import annotations

import argparse

from reconstruction.local_recon import save_reconstruction_result
from reconstruction.pipeline import DEFAULT_CONFIG_PATH, reconstruct_image


def main() -> None:
    parser = argparse.ArgumentParser(description='单张图片 → marker 整体曲面 → LUT 局部形变')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--image', required=True, help='待重建图片')
    parser.add_argument('--reference-image', help='覆盖配置中的无接触参考图')
    parser.add_argument('--material-grid', help='覆盖与 LUT 配套的材料网格或完整参考弧长图')
    parser.add_argument('--calibration', help='覆盖 local_reconstruction.calibration_file')
    parser.add_argument('--output', help='结果 NPZ；默认保存到 reconstruction.output_dir')
    parser.add_argument('--preview-output', help='预览图；默认与结果 NPZ 同目录、同名')
    parser.add_argument('--show', action='store_true', help='保存后显示预览窗口')
    parser.add_argument('--lsmr-atol', type=float)
    parser.add_argument('--lsmr-btol', type=float)
    parser.add_argument('--lsmr-max-iterations', type=int)
    args = parser.parse_args()
    result, output = reconstruct_image(
        args.image, args.config, reference_image=args.reference_image,
        material_grid_file=args.material_grid, calibration_file=args.calibration,
        output=args.output, preview_output=args.preview_output,
        lsmr_atol=args.lsmr_atol, lsmr_btol=args.lsmr_btol,
        lsmr_max_iterations=args.lsmr_max_iterations)
    save_reconstruction_result(result, output, preview_output=args.preview_output, show=args.show)


if __name__ == '__main__':
    main()
