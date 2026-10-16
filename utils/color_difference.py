"""从已给定参考图开始计算有符号色差；不估计或拟合参考背景。"""

from __future__ import annotations

import cv2
import jax.numpy as jnp
import numpy as np


def bgr8_to_linear_rgb(image_bgr: np.ndarray) -> np.ndarray:
    values = np.asarray(image_bgr)
    if values.ndim != 3 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError("image_bgr 必须是 HxWx3 uint8")
    srgb = values[..., ::-1].astype(np.float32) / 255.0
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def signed_color_difference(
    observed_linear_rgb: np.ndarray,
    reference_linear_rgb: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 ``observed-reference``，保留三个通道的正负符号。"""
    observed = np.asarray(observed_linear_rgb, np.float32)
    reference = np.asarray(reference_linear_rgb, np.float32)
    if observed.ndim != 3 or observed.shape[-1] != 3 \
            or reference.shape != observed.shape:
        raise ValueError("观测和参考必须同为 HxWx3 线性 RGB")
    valid = np.all(np.isfinite(observed), axis=-1) \
        & np.all(np.isfinite(reference), axis=-1)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, np.bool_)
        if supplied.shape != observed.shape[:2]:
            raise ValueError("valid_mask 尺寸必须与图像一致")
        valid &= supplied
    difference = np.where(valid[..., None], observed - reference, 0)
    return difference.astype(np.float32), valid


def signed_color_difference_jax(
    observed_linear_rgb: jnp.ndarray,
    reference_linear_rgb: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """JAX 版有符号色差，供后续 GPU 局部重建接线。"""
    return jnp.where(
        jnp.asarray(valid_mask, jnp.bool_)[..., None],
        jnp.asarray(observed_linear_rgb, jnp.float32)
        - jnp.asarray(reference_linear_rgb, jnp.float32),
        0,
    )


def erode_valid_mask(valid_mask: np.ndarray, pixels: int) -> np.ndarray:
    valid = np.asarray(valid_mask, np.bool_)
    if valid.ndim != 2 or not isinstance(pixels, int) or pixels < 0:
        raise ValueError("valid_mask 必须是二维数组，pixels 必须是非负整数")
    if pixels == 0:
        return valid.copy()
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
    return cv2.erode(
        valid.astype(np.uint8), kernel, iterations=1,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.bool_)


def signed_difference_preview(
    residual_linear_rgb: np.ndarray,
    valid_mask: np.ndarray,
    gain: float = 2.0,
) -> np.ndarray:
    """仅供诊断：零差值显示为中性灰，正负方向保留。"""
    residual = np.asarray(residual_linear_rgb, np.float32)
    valid = np.asarray(valid_mask, np.bool_)
    if residual.ndim != 3 or residual.shape[-1] != 3 \
            or valid.shape != residual.shape[:2] or gain <= 0:
        raise ValueError("色差预览输入无效")
    rgb = np.clip(0.5 + gain * residual, 0, 1)
    bgr = np.rint(rgb[..., ::-1] * 255).astype(np.uint8)
    return np.where(valid[..., None], bgr, 0)
