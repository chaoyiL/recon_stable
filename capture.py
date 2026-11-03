"""固定曝光采集相机帧，并保存为线性 RGB NPZ。"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from utils.camera import open_camera
from utils.color_difference import (
    CAMERA_RGB_TRANSFER_FUNCTION,
    bgr8_to_linear_rgb,
)
from utils.config import (
    ConfigError,
    load_config_sections,
    parse_camera_config,
    require_keys,
)
from utils.image_io import linear_rgb_to_bgr8, save_linear_image

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def capture_output_path(configured_path: Path) -> Path:
    """Use an explicit NPZ path or create a timestamped NPZ in a directory."""
    suffix = configured_path.suffix.lower()
    if not configured_path.is_dir():
        if suffix == ".npz":
            return configured_path
        if suffix in IMAGE_SUFFIXES:
            raise ValueError(
                f"静态图片必须保存为线性 RGB NPZ，请将 capture.save 改为 .npz: "
                f"{configured_path}"
            )
    configured_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = configured_path / f"capture_{timestamp}.npz"
    counter = 1
    while candidate.exists():
        candidate = configured_path / f"capture_{timestamp}_{counter:02d}.npz"
        counter += 1
    return candidate


def video_output_path(configured_path: Path) -> Path:
    """Create a timestamped frame directory under the configured directory."""
    if not configured_path.is_dir() \
            and configured_path.suffix.lower() in VIDEO_SUFFIXES:
        raise ValueError(
            f"录像现保存为逐帧 NPZ 目录，请将 capture.video_save 配置为目录: "
            f"{configured_path}"
        )
    configured_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = configured_path / f"capture_{timestamp}"
    counter = 1
    while candidate.exists():
        candidate = configured_path / f"capture_{timestamp}_{counter:02d}"
        counter += 1
    candidate.mkdir()
    return candidate


def recording_fps(camera_fps: float, fallback: float = 30.0) -> float:
    """过滤部分相机后端返回的 0、NaN 等无效帧率。"""
    if math.isfinite(camera_fps) and 1.0 <= camera_fps <= 240.0:
        return float(camera_fps)
    return float(fallback)


def save_video_manifest(
    output_dir: Path,
    fps: float,
    frames: list[dict[str, int | float | str]],
) -> bool:
    """Write the frame index and nominal time for a completed recording."""
    manifest = {
        "format_version": 1,
        "color_space": "linear_rgb",
        "rgb_transfer_function": CAMERA_RGB_TRANSFER_FUNCTION,
        "fps": fps,
        "frames": frames,
    }
    try:
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except OSError as error:
        print(f"录像清单保存失败: {error}", file=sys.stderr)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="固定曝光采集相机帧并保存线性 RGB NPZ"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML 配置文件，默认 {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args()

    try:
        config_path = Path(args.config).expanduser().resolve()
        camera_section, capture_section = load_config_sections(
            config_path,
            "camera",
            "capture",
        )
        camera = parse_camera_config(camera_section)
        require_keys(capture_section, "capture", "save", "video_save")
        raw_save_path = capture_section["save"]
        if raw_save_path is not None and not isinstance(raw_save_path, str):
            raise ConfigError("capture.save 必须是字符串或 null")
        save_path = Path(raw_save_path).expanduser() if raw_save_path else None
        if save_path is not None and not save_path.is_absolute():
            save_path = config_path.parent/save_path
        raw_video_save_path = capture_section["video_save"]
        if not isinstance(raw_video_save_path, str) or not raw_video_save_path.strip():
            raise ConfigError("capture.video_save 必须是非空字符串")
        video_save_path = Path(raw_video_save_path).expanduser()
        if not video_save_path.is_absolute():
            video_save_path = config_path.parent/video_save_path
    except ConfigError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        sys.exit(2)

    cap = open_camera(
        camera.device,
        camera.exposure,
        camera.white_balance_temperature,
        camera.width,
        camera.height,
        fourcc=camera.fourcc,
        backlight_compensation=camera.backlight_compensation,
    )

    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    actual_auto_wb = cap.get(cv2.CAP_PROP_AUTO_WB)
    actual_wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"相机已打开: device={camera.device}, 分辨率={w}x{h}")
    print(f"自动曝光={actual_auto}, 曝光值={actual_exposure}")
    print(f"自动白平衡={actual_auto_wb}, 白平衡色温={actual_wb} K")
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_name = actual_fourcc.to_bytes(4, "little").decode("ascii", errors="replace")
    print(f"采集格式={fourcc_name}, 背光补偿={cap.get(cv2.CAP_PROP_BACKLIGHT)}")
    print("预览直接显示线性 RGB 值")
    print("静态图片和录像均保存为线性 RGB float32 NPZ")
    print("按 v 开始录像；按 s 停止录像；按 p 保存当前帧；按 q 或 Esc 退出")

    window = "camera preview"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    active_video_dir: Path | None = None
    video_fps = 0.0
    video_start_time: float | None = None
    frame_records: list[dict[str, int | float | str]] = []
    recorded_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("读取帧失败", file=sys.stderr)
                break

            linear_rgb = bgr8_to_linear_rgb(frame)
            if active_video_dir is not None:
                frame_time = time.monotonic()
                if video_start_time is None:
                    video_start_time = frame_time
                timestamp_s = frame_time - video_start_time
                frame_name = f"frame_{recorded_frames:06d}.npz"
                frame_path = active_video_dir / frame_name
                try:
                    save_linear_image(frame_path, linear_rgb)
                except OSError as error:
                    print(f"录像帧保存失败: {error}", file=sys.stderr)
                    save_video_manifest(active_video_dir, video_fps, frame_records)
                    active_video_dir = None
                    video_start_time = None
                    frame_records = []
                    recorded_frames = 0
                else:
                    frame_records.append({
                        "index": recorded_frames,
                        "timestamp_s": timestamp_s,
                        "file": frame_name,
                    })
                    recorded_frames += 1

            display_frame = linear_rgb_to_bgr8(linear_rgb)
            cv2.putText(
                display_frame,
                "LINEAR RGB",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if active_video_dir is not None:
                cv2.putText(
                    display_frame,
                    f"REC  {recorded_frames}",
                    (16, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q 或 Esc
                break
            if key == ord("v"):
                if active_video_dir is not None:
                    print(f"正在录像: {active_video_dir}")
                    continue
                try:
                    active_video_dir = video_output_path(video_save_path)
                except (OSError, ValueError) as error:
                    print(f"开始录像失败: {error}", file=sys.stderr)
                    continue
                video_fps = recording_fps(float(cap.get(cv2.CAP_PROP_FPS)))
                video_start_time = None
                frame_records = []
                recorded_frames = 0
                print(
                    f"开始录像: {active_video_dir} "
                    f"({frame.shape[1]}x{frame.shape[0]}, {video_fps:.2f} FPS)"
                )
            elif key == ord("s"):
                if active_video_dir is None:
                    print("当前没有正在进行的录像。按 v 开始录像。")
                    continue
                manifest_saved = save_video_manifest(
                    active_video_dir, video_fps, frame_records
                )
                result = "录像已保存" if manifest_saved else "录像帧已写入，清单保存失败"
                print(f"{result}: {active_video_dir}（{recorded_frames} 帧）")
                active_video_dir = None
                video_start_time = None
                frame_records = []
                recorded_frames = 0
            elif key == ord("p") and save_path:
                try:
                    output_path = capture_output_path(save_path)
                    save_linear_image(output_path, linear_rgb)
                    print(f"已保存到 {output_path}")
                except (OSError, ValueError) as error:
                    print(f"保存失败: {error}", file=sys.stderr)
    finally:
        if active_video_dir is not None:
            manifest_saved = save_video_manifest(
                active_video_dir, video_fps, frame_records
            )
            result = "录像已保存" if manifest_saved else "录像帧已写入，清单保存失败"
            print(f"{result}: {active_video_dir}（{recorded_frames} 帧）")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
