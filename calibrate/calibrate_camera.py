"""相机标定命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from calibrate.utils.camera_calibration import run_camera_calibration

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="棋盘格或黑底白圆点标定板单目相机标定"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML 配置文件，默认 {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_camera_calibration(args.config)


if __name__ == "__main__":
    main()
