"""共享的相机初始化。"""

from __future__ import annotations

import math

import cv2


def open_camera(
    device: int,
    exposure: float,
    white_balance_temperature: float,
    width: int | None,
    height: int | None,
    *,
    fourcc: str | None = None,
    backlight_compensation: float | None = None,
) -> cv2.VideoCapture:
    """打开 V4L2 相机并设置请求的采集格式、曝光和白平衡。"""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机 device={device}")

    try:
        if fourcc is not None:
            requested_fourcc = cv2.VideoWriter_fourcc(*fourcc)
            if not cap.set(cv2.CAP_PROP_FOURCC, requested_fourcc):
                raise RuntimeError(f"相机驱动拒绝设置 FOURCC 格式 {fourcc}")
            actual_fourcc = float(cap.get(cv2.CAP_PROP_FOURCC))
            if not math.isfinite(actual_fourcc) \
                    or round(actual_fourcc) != requested_fourcc:
                raise RuntimeError(
                    f"FOURCC 格式设置未生效：请求={fourcc}，"
                    f"回读编码={actual_fourcc}")

        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if backlight_compensation is not None:
            if not cap.set(cv2.CAP_PROP_BACKLIGHT, backlight_compensation):
                raise RuntimeError(
                    "相机驱动拒绝设置背光补偿值 "
                    f"{backlight_compensation}")
            actual_backlight = float(cap.get(cv2.CAP_PROP_BACKLIGHT))
            backlight_tolerance = max(.5, .1 * abs(backlight_compensation))
            if not math.isfinite(actual_backlight) \
                    or abs(actual_backlight - backlight_compensation) > backlight_tolerance:
                raise RuntimeError(
                    "背光补偿设置未生效："
                    f"请求={backlight_compensation}，回读={actual_backlight}")

        # V4L2: 1=手动曝光，3=光圈优先自动曝光。
        if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1):
            raise RuntimeError("相机驱动拒绝切换到手动曝光")
        actual_auto_exposure = float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        if not math.isfinite(actual_auto_exposure) or actual_auto_exposure != 1:
            raise RuntimeError(
                "手动曝光模式设置未生效："
                f"请求=1，回读={actual_auto_exposure}")
        if not cap.set(cv2.CAP_PROP_EXPOSURE, exposure):
            raise RuntimeError(f"相机驱动拒绝设置曝光值 {exposure}")

        # 固定焦相机通常返回 false；支持自动对焦的相机则必须成功关闭并回读验证。
        autofocus_controlled = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        actual_autofocus = float(cap.get(cv2.CAP_PROP_AUTOFOCUS))
        if autofocus_controlled and math.isfinite(actual_autofocus) \
                and actual_autofocus > .5:
            raise RuntimeError(
                f"自动对焦未成功关闭，驱动回读值={actual_autofocus}")

        if not cap.set(cv2.CAP_PROP_AUTO_WB, 0):
            raise RuntimeError("相机驱动不支持或拒绝关闭自动白平衡")
        if not cap.set(cv2.CAP_PROP_WB_TEMPERATURE, white_balance_temperature):
            raise RuntimeError(
                f"相机驱动不支持或拒绝设置白平衡色温 {white_balance_temperature} K")
        actual_auto_wb=float(cap.get(cv2.CAP_PROP_AUTO_WB))
        actual_temperature=float(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
        actual_exposure=float(cap.get(cv2.CAP_PROP_EXPOSURE))
        exposure_tolerance=max(1.,.1*abs(float(exposure)))
        if not math.isfinite(actual_exposure) \
                or abs(actual_exposure-float(exposure))>exposure_tolerance:
            raise RuntimeError(
                "手动曝光设置未生效："
                f"请求={exposure}，回读={actual_exposure}")
        if not math.isfinite(actual_auto_wb) or actual_auto_wb < 0 or actual_auto_wb > .5:
            raise RuntimeError(
                f"自动白平衡未成功关闭，驱动回读值={actual_auto_wb}")
        tolerance=max(250.,.1*float(white_balance_temperature))
        if (
            not math.isfinite(actual_temperature)
            or actual_temperature<=0
            or abs(actual_temperature-white_balance_temperature)>tolerance
        ):
            raise RuntimeError(
                "手动白平衡设置未生效："
                f"请求={white_balance_temperature} K，回读={actual_temperature} K")
    except Exception:
        cap.release()
        raise
    return cap
