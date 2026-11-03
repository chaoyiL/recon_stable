"""Linear RGB image input and output helpers."""

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from utils.color_difference import (
    CAMERA_RGB_TRANSFER_FUNCTION,
    bgr8_to_linear_rgb,
)


NPZ_KEYS = {
    "linear_rgb",
    "color_space",
    "rgb_transfer_function",
    "format_version",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def validate_linear_rgb(image: np.ndarray) -> np.ndarray:
    """Return an HxWx3 float32 RGB image with finite values in [0, 1]."""
    try:
        source = np.asarray(image)
    except (TypeError, ValueError) as error:
        raise ValueError("linear_rgb 必须是数值数组") from error
    if not np.issubdtype(source.dtype, np.floating):
        raise ValueError("linear_rgb 必须是浮点数组")
    values = source.astype(np.float32, copy=False)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("linear_rgb 必须是 HxWx3 RGB 数组")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("linear_rgb 的图像宽高必须大于 0")
    if not np.all(np.isfinite(values)):
        raise ValueError("linear_rgb 必须包含有限值")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("linear_rgb 数值必须在 [0, 1] 范围内")
    return values


def cached_linear_rgb(
    data: Mapping[str, object],
    prefix: str = "observed",
) -> np.ndarray:
    """Read linear RGB from a cache, decoding its legacy BGR8 field if needed."""
    linear_key = f"{prefix}_linear_rgb"
    bgr_key = f"{prefix}_bgr"
    if linear_key in data:
        try:
            return validate_linear_rgb(np.asarray(data[linear_key]))
        except ValueError as error:
            raise ValueError(f"缓存字段 {linear_key!r} 无效: {error}") from error
    if bgr_key in data:
        try:
            return bgr8_to_linear_rgb(np.asarray(data[bgr_key]))
        except ValueError as error:
            raise ValueError(f"缓存字段 {bgr_key!r} 无效: {error}") from error
    raise ValueError(
        f"缓存缺少 {linear_key!r} 和兼容字段 {bgr_key!r}"
    )


def load_linear_image(path: str | Path) -> np.ndarray:
    """Load a linear RGB NPZ image or decode a legacy encoded image once."""
    image_path = Path(path).expanduser()
    if image_path.suffix.lower() == ".npz":
        try:
            with np.load(image_path, allow_pickle=False) as data:
                missing_keys = NPZ_KEYS.difference(data.files)
                if missing_keys:
                    raise ValueError(
                        f"{image_path}: NPZ 缺少必须键 {sorted(missing_keys)}"
                    )
                if data["color_space"].shape != () \
                        or data["color_space"].item() != "linear_rgb":
                    raise ValueError(f"{image_path}: color_space 必须为 'linear_rgb'")
                if data["rgb_transfer_function"].shape != () \
                        or data["rgb_transfer_function"].item() \
                        != CAMERA_RGB_TRANSFER_FUNCTION:
                    raise ValueError(
                        f"{image_path}: rgb_transfer_function 必须为 "
                        f"{CAMERA_RGB_TRANSFER_FUNCTION!r}"
                    )
                version = data["format_version"]
                if version.shape != () or version.dtype.kind not in "iu" \
                        or int(version.item()) != 1:
                    raise ValueError(f"{image_path}: format_version 必须为整数 1")
                linear_rgb = data["linear_rgb"]
                if linear_rgb.dtype != np.float32:
                    raise ValueError(f"{image_path}: linear_rgb 必须是 float32")
                return validate_linear_rgb(linear_rgb)
        except FileNotFoundError:
            raise
        except (OSError, KeyError) as error:
            raise ValueError(f"无法读取线性 RGB NPZ 图像 {image_path}: {error}") from error

    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图像格式: {image_path.suffix or '(无扩展名)'}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return bgr8_to_linear_rgb(image_bgr)


def save_linear_image(path: str | Path, linear_rgb: np.ndarray) -> Path:
    """Save a validated linear RGB image and its format metadata as NPZ."""
    image_path = Path(path).expanduser()
    if image_path.suffix.lower() != ".npz":
        raise ValueError("线性 RGB 图像必须保存为 .npz")
    values = validate_linear_rgb(linear_rgb)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=image_path.parent,
            prefix=f".{image_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(
                stream,
                linear_rgb=values,
                color_space=np.asarray("linear_rgb"),
                rgb_transfer_function=np.asarray(CAMERA_RGB_TRANSFER_FUNCTION),
                format_version=np.asarray(1, dtype=np.uint8),
            )
        os.replace(temporary_path, image_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return image_path


def linear_rgb_to_bgr8(linear_rgb: np.ndarray) -> np.ndarray:
    """Quantize linear RGB directly to BGR8 for display, without encoding."""
    values = validate_linear_rgb(linear_rgb)
    return np.rint(values[..., ::-1] * 255.0).astype(np.uint8)


def linear_rgb_to_tracking_bgr8(linear_rgb: np.ndarray) -> np.ndarray:
    """Apply the BT.709 transfer curve for marker tracking input only."""
    values = validate_linear_rgb(linear_rgb)
    encoded = np.where(
        values < 0.018,
        4.5 * values,
        1.099 * np.power(values, 0.45) - 0.099,
    )
    return np.rint(encoded[..., ::-1] * 255.0).astype(np.uint8)
