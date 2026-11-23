"""Capture, compare and match a tactile sensor's lighting to a reference."""

import argparse
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from hardware.led_protocol import LedLayout, LedSettings
from optical_optimize.devices import (
    camera_readback,
    camera_session,
    capture_mean,
    configured_pwm,
    open_strip,
    send_pwm,
)
from optical_optimize.field import Comparison, detect_surface
from optical_optimize.solver import bounded_pwm, propose_pwm
from utils.config import CameraConfig, load_config_sections, parse_camera_config
from utils.image_io import linear_rgb_to_tracking_bgr8, load_linear_image, save_linear_image

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
DEFAULT_REFERENCE = ROOT / "assets" / "reference_flat.npz"
Measure = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_preset(path: Path, layout: LedLayout, pwm: np.ndarray) -> None:
    write_json(path, {"format_version": 1, "layout": asdict(layout), "pwm_rgb": pwm.tolist()})


def load_preset(path: Path, layout: LedLayout) -> np.ndarray:
    preset = json.loads(path.read_text(encoding="utf-8"))
    if preset.get("format_version") != 1 or preset.get("layout") != asdict(layout):
        raise ValueError("灯光预设的版本或灯珠布局与当前配置不一致")
    pwm = np.asarray(preset["pwm_rgb"])
    if pwm.shape != (layout.active_count, 3) or pwm.dtype.kind not in "iu":
        raise ValueError("灯光预设需要为每颗有效灯珠保存三个整数 PWM 值")
    if np.any(pwm < 0) or np.any(pwm > 255):
        raise ValueError("预设 PWM 超出 0–255")
    return pwm.astype(int)


def archive_frame(out: Path, name: str, image: np.ndarray) -> None:
    save_linear_image(out / f"{name}.npz", image)
    path = out / f"{name}.png"
    if not cv2.imwrite(str(path), linear_rgb_to_tracking_bgr8(image)):
        raise OSError(f"预览文件写入失败：{path}")


def load_camera_settings(config: Path, device: int | None) -> CameraConfig:
    (section,) = load_config_sections(config, "camera")
    settings = parse_camera_config(section)
    return settings if device is None else replace(settings, device=device)


def create_run(output: Path | None) -> Path:
    out = output or Path(__file__).with_name("runs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    return out


def capture_current(args) -> None:
    camera = load_camera_settings(args.config, args.camera)
    out = create_run(args.output)
    with camera_session(camera) as cap:
        frame = capture_mean(cap, args.frames, .8)
        readback = camera_readback(cap)
    archive_frame(out, "current", frame)
    write_json(out / "camera.json", {"requested": asdict(camera), "readback": readback})
    print(f"当前胶皮画面已保存到 {out / 'current.npz'}")


def rmse(features: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((features - target)**2)))


def probe_channels(
    measure: Measure, start: np.ndarray, first: np.ndarray, cap: int, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure each PWM channel and interpolate repeated baseline readings for drift."""
    size = start.size
    steps = np.where(cap - start.ravel() >= step, step, -step).astype(int)
    samples = np.empty((size, first.size), dtype=float)
    previous = first
    for group_start in range(0, size, 13):
        group_stop = min(group_start + 13, size)
        for index in range(group_start, group_stop):
            pwm = start.copy()
            pwm.flat[index] += steps[index]
            _, samples[index] = measure(pwm)
        _, following = measure(start)
        for index in range(group_start, group_stop):
            fraction = (index - group_start + 1) / (group_stop - group_start + 1)
            drift_baseline = (1 - fraction) * previous + fraction * following
            samples[index] -= drift_baseline
        previous = following
        print(f"已测 {group_stop}/{size} 个灯珠颜色通道", flush=True)
    return samples + previous, steps, previous


def measure_update(
    measure: Measure,
    current: np.ndarray,
    target: np.ndarray,
    proposal: np.ndarray,
    best_error: float,
    round_number: int,
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    for fraction in (1., .5, .25):
        pwm = np.rint(current + fraction * (proposal - current)).astype(int)
        if np.array_equal(pwm, current):
            break
        _, features = measure(pwm)
        error = rmse(features, target)
        accepted = error < best_error - .00015
        records.append({
            "round": round_number, "fraction": fraction, "field_rmse": error,
            "accepted": accepted, "pwm_rgb": pwm.tolist(),
        })
        print(f"第 {round_number} 轮，步长 {fraction:g}，实拍光场 RMSE {error:.5f}", flush=True)
        if accepted:
            return pwm, features, error
    return None


def optimize_live(args, config: Path, reference_file: Path) -> Path:
    camera = load_camera_settings(config, args.camera)
    (led_section,) = load_config_sections(config, "led_controller")
    settings = LedSettings.from_mapping(led_section)
    layout = settings.layout
    start = load_preset(args.start_preset, layout) if args.start_preset else configured_pwm(settings)
    if start.max() > args.max_pwm:
        raise ValueError("起始灯光超过 --max-pwm；请提高上限或选择其他起始预设")
    if args.probe_step * 2 > args.max_pwm:
        raise ValueError("--probe-step 的两倍必须小于或等于 --max-pwm")
    reference = load_linear_image(reference_file)
    out = create_run(args.output)

    with camera_session(camera) as cap:
        before = capture_mean(cap, args.frames, .8)
        archive_frame(out, "before", before)
        readback = camera_readback(cap)
        surface = detect_surface(before)
        comparison = Comparison.build(reference, before, surface)
        target = comparison.features(reference)
        aligned_before = comparison.align(before)
        report = {
            "reference": str(reference_file.relative_to(ROOT)) if reference_file.is_relative_to(ROOT)
            else str(reference_file),
            "camera_requested": asdict(camera),
            "camera_readback": readback,
            "registration": comparison.registration,
            "before": comparison.metrics(aligned_before),
            "max_pwm": args.max_pwm,
            "probe_step": args.probe_step,
            "settle_seconds": args.settle,
            "frames_per_capture": args.frames,
            "iterations": [],
        }
        write_json(out / "report.json", report)
        print(f"当前光场 RMSE {report['before']['field_rmse']:.5f}，记录写入 {out}", flush=True)

        with open_strip(args.port) as device:
            def measure(pwm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                send_pwm(device, layout, pwm)
                frame = capture_mean(cap, args.frames, args.settle)
                if frame.shape != before.shape:
                    raise ValueError("采集过程中相机分辨率发生变化")
                return frame, comparison.features(comparison.align(frame))

            best_pwm = start
            try:
                initial, initial_features = measure(start)
                archive_frame(out, "start", initial)
                save_preset(out / "start_preset.json", layout, start)
                report["start"] = comparison.metrics(comparison.align(initial))
                print(f"起始预设光场 RMSE {report['start']['field_rmse']:.5f}", flush=True)
                samples, steps, best_features = probe_channels(
                    measure, start, initial_features, args.max_pwm, args.probe_step
                )
                report["probe_baseline_drift_rmse"] = rmse(best_features, initial_features)
                proposal, jacobian = propose_pwm(
                    start, best_features, samples, target, steps, args.max_pwm
                )
                np.savez_compressed(
                    out / "response.npz", baseline_pwm=start, baseline_features=best_features,
                    probe_steps=steps, probe_features=samples, target_features=target,
                    jacobian=jacobian, transform=comparison.transform, mask=comparison.mask,
                )
                best_error = rmse(best_features, target)
                for round_number in range(1, args.iterations + 1):
                    update = measure_update(
                        measure, best_pwm, target, proposal, best_error, round_number, report["iterations"]
                    )
                    if update is None:
                        break
                    pwm, features, error = update
                    delta = (pwm - best_pwm).ravel()
                    correction = features - best_features - jacobian @ delta
                    jacobian += np.outer(correction, delta) / (delta @ delta)
                    best_pwm, best_features, best_error = pwm, features, error
                    if round_number == args.iterations:
                        break
                    proposal = bounded_pwm(
                        best_pwm, best_features, target, jacobian, args.max_pwm
                    )
                final, verification = measure(best_pwm)
                archive_frame(out, "after", final)
                motion = detect_surface(final).pixels - surface.pixels
                motion_rms = float(np.sqrt(np.mean(np.sum(motion**2, axis=1))))
                if motion_rms > 2:
                    raise RuntimeError(f"采集过程中胶皮十字移动了 {motion_rms:.2f} px，请固定构型后重测")
                report["before_to_after_marker_rms_px"] = motion_rms
                report["final_repeatability_rmse"] = rmse(best_features, verification)
            except BaseException:
                try:
                    send_pwm(device, layout, best_pwm)
                    save_preset(out / "recovery_preset.json", layout, best_pwm)
                    write_json(out / "report.json", report)
                except (RuntimeError, OSError) as error:
                    print(f"恢复已测灯光时发生错误：{error}", file=sys.stderr)
                raise

    save_preset(out / "preset.json", layout, best_pwm)
    aligned_final = comparison.align(final)
    report["after"] = comparison.metrics(aligned_final)
    report["preset"] = "preset.json"
    write_json(out / "report.json", report)
    comparison.write_preview(out / "comparison.png", [
        ("Before", aligned_before), ("Start preset", comparison.align(initial)),
        ("After", aligned_final),
    ])
    print(f"优化后实拍光场 RMSE {report['after']['field_rmse']:.5f}，灯光已应用", flush=True)
    return out


def compare_saved(args, reference_file: Path) -> None:
    reference = load_linear_image(reference_file)
    images = [(path.stem, load_linear_image(path)) for path in args.images]
    comparison = Comparison.build(reference, images[0][1])
    aligned = [(label, comparison.align(image)) for label, image in images]
    write_json(args.output / "report.json", {
        "reference": str(reference_file),
        "registration": comparison.registration,
        "images": [
            {"path": str(path), **comparison.metrics(image)}
            for path, (_, image) in zip(args.images, aligned)
        ],
    })
    comparison.write_preview(args.output / "comparison.png", aligned)


def add_capture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=4)


def add_reference_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference", type=Path, default=DEFAULT_REFERENCE,
        help="历史参考图，默认 assets/reference_flat.npz",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m optical_optimize", description="采集胶皮光场并按历史参考细调灯带 RGB PWM"
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="采集当前灯光下的胶皮图像")
    add_capture_arguments(capture)
    compare = commands.add_parser("compare", help="比较同一构型下采集的图像")
    compare.add_argument("images", type=Path, nargs="+")
    add_reference_argument(compare)
    compare.add_argument("--output", type=Path, required=True)
    optimize = commands.add_parser("optimize", help="逐灯测量颜色响应并应用实拍得分最低的设置")
    add_capture_arguments(optimize)
    add_reference_argument(optimize)
    optimize.add_argument("--port", default="/dev/ttyACM0")
    optimize.add_argument("--start-preset", type=Path)
    optimize.add_argument("--max-pwm", type=int, default=112)
    optimize.add_argument("--probe-step", type=int, default=36)
    optimize.add_argument("--settle", type=float, default=.15)
    optimize.add_argument("--iterations", type=int, default=8)
    apply = commands.add_parser("apply", help="把已验证的 PWM 预设重新发送到灯带")
    apply.add_argument("preset", type=Path)
    apply.add_argument("--port", default="/dev/ttyACM0")
    args = parser.parse_args()

    try:
        if args.command == "apply":
            (led_section,) = load_config_sections(args.config, "led_controller")
            layout = LedSettings.from_mapping(led_section).layout
            pwm = load_preset(args.preset, layout)
            with open_strip(args.port) as device:
                send_pwm(device, layout, pwm)
            print(f"已应用 {args.preset}，灯带确认 {len(pwm)} 颗有效灯珠")
            return
        if args.command == "capture":
            if args.frames <= 0:
                raise ValueError("帧数须为正")
            capture_current(args)
            return
        reference_file = args.reference.expanduser().resolve()
        if args.command == "compare":
            compare_saved(args, reference_file)
            return
        if not (
            0 < args.max_pwm <= 255 and args.probe_step > 0 and args.frames > 0
            and args.iterations > 0 and math.isfinite(args.settle) and args.settle >= 0
        ):
            raise ValueError("PWM 上限须在 1–255，步长、帧数和迭代数须为正，等待时间须为非负有限值")
        optimize_live(args, args.config, reference_file)
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as error:
        parser.exit(1, f"光场处理失败：{error}\n")


if __name__ == "__main__":
    main()
