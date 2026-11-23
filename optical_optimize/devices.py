"""Headless camera acquisition and acknowledged LED PWM writes."""

from contextlib import contextmanager
import time

import cv2
import numpy as np
import serial

from hardware.led_protocol import BAUD, LedLayout, LedSettings, default_colors, encode_frame
from utils.camera import open_camera
from utils.color_difference import bgr8_to_linear_rgb
from utils.config import CameraConfig


def configured_pwm(settings: LedSettings) -> np.ndarray:
    """Read the actual transmitted RGB bytes from the configured default frame."""
    layout = settings.layout
    frame = encode_frame(
        layout, default_colors(layout), settings.brightness, rgb_scale=settings.rgb_scale
    )
    return np.frombuffer(frame[4 + 3 * layout.skip:-1], dtype=np.uint8).reshape(-1, 3).astype(int)


@contextmanager
def open_strip(port: str):
    """Wait for Arduino startup before sending the first frame."""
    with serial.Serial(port, BAUD, timeout=1, write_timeout=1, exclusive=True) as device:
        time.sleep(2)
        yield device


def send_pwm(device: serial.Serial, layout: LedLayout, pwm: np.ndarray) -> None:
    """Send an active-LED N×3 integer PWM table and require a firmware ACK."""
    colors = [(*rgb, 100) for rgb in pwm.tolist()]
    packet = encode_frame(layout, colors)
    device.reset_input_buffer()
    if device.write(packet) != len(packet):
        raise RuntimeError("灯带串口发送了不完整的数据帧")
    ack = device.read(1)
    if ack != b"K":
        raise RuntimeError(f"灯带未确认刷新，串口应答 {ack!r}")


@contextmanager
def camera_session(camera: CameraConfig):
    cap = open_camera(
        camera.device, camera.exposure, camera.white_balance_temperature,
        camera.width, camera.height, fourcc=camera.fourcc,
        backlight_compensation=camera.backlight_compensation,
    )
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        yield cap
    finally:
        cap.release()


def camera_readback(cap: cv2.VideoCapture) -> dict:
    properties = (
        "FRAME_WIDTH", "FRAME_HEIGHT", "FOURCC", "FPS", "BUFFERSIZE",
        "AUTO_EXPOSURE", "EXPOSURE", "AUTO_WB", "WB_TEMPERATURE", "BACKLIGHT",
        "GAIN", "GAMMA", "BRIGHTNESS", "CONTRAST", "SATURATION", "SHARPNESS",
    )
    values = {name.lower(): cap.get(getattr(cv2, f"CAP_PROP_{name}")) for name in properties}
    values["fourcc"] = int(values["fourcc"]).to_bytes(4, "little").decode("ascii", errors="replace")
    return values


def capture_mean(cap: cv2.VideoCapture, samples: int, settle_seconds: float) -> np.ndarray:
    """Discard queued/settling frames and average fresh samples in linear RGB."""
    deadline = time.monotonic() + settle_seconds
    discarded = 0
    while discarded < 3 or time.monotonic() < deadline:
        if not cap.grab():
            raise RuntimeError("清空相机采集队列时读取失败")
        discarded += 1
    frames = []
    for _ in range(samples):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError("相机读取失败")
        frames.append(bgr8_to_linear_rgb(bgr))
    return np.mean(frames, axis=0)
