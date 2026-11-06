"""仓库公共配置读取与相机参数校验。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """配置内容无效。"""


@dataclass(frozen=True)
class CameraConfig:
    device: int
    exposure: float
    white_balance_temperature: float
    width: int | None
    height: int | None
    fourcc: str | None = None
    backlight_compensation: float | None = None


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML；顶层 ``extends`` 可继承另一个 YAML。"""

    def load_one(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        resolved = path.expanduser().resolve()
        if resolved in stack:
            chain = " -> ".join(str(item) for item in (*stack, resolved))
            raise ConfigError(f"配置 extends 存在循环: {chain}")
        try:
            value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ConfigError(f"配置文件不存在: {resolved}") from error
        except yaml.YAMLError as error:
            raise ConfigError(f"配置文件格式错误: {error}") from error
        if not isinstance(value, Mapping):
            raise ConfigError("配置文件根节点必须是字典")
        parent = value.get("extends")
        if parent is None:
            return deepcopy(dict(value))
        if not isinstance(parent, str) or not parent.strip():
            raise ConfigError("顶层 extends 必须是非空路径字符串")
        parent_path = Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        return _merge(load_one(parent_path, (*stack, resolved)), value)

    return load_one(Path(config_path), ())


def load_config_sections(
    config_path: str | Path, *section_names: str,
) -> tuple[dict[str, Any], ...]:
    config = load_config(config_path)
    sections: list[dict[str, Any]] = []
    for name in section_names:
        value = config.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"缺少字典配置段: {name}")
        sections.append(dict(value))
    return tuple(sections)


def load_contact_surface_offset_mm(config_path: str | Path) -> float:
    """Read the signed contact-surface offset along the marker surface normal, in mm."""
    section = load_config(config_path).get("contact_surface")
    if not isinstance(section, Mapping) or "offset_mm" not in section:
        raise ConfigError("contact_surface.offset_mm 必须明确配置")
    unknown = set(section) - {"offset_mm"}
    if unknown:
        raise ConfigError(f"contact_surface 包含未知字段: {sorted(unknown)}")
    value = section["offset_mm"]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError("contact_surface.offset_mm 必须是有限数字，单位 mm")
    try:
        offset = float(value)
    except OverflowError as error:
        raise ConfigError("contact_surface.offset_mm 必须是有限数字，单位 mm") from error
    if not math.isfinite(offset):
        raise ConfigError("contact_surface.offset_mm 必须是有限数字，单位 mm")
    return offset


def require_keys(section: Mapping[str, Any], section_name: str, *keys: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(f"配置段 {section_name} 缺少字段: {', '.join(missing)}")


def parse_camera_config(section: Mapping[str, Any] | None) -> CameraConfig:
    if not isinstance(section, Mapping):
        raise ConfigError("camera 必须是字典")
    require_keys(
        section, "camera", "device", "exposure",
        "white_balance_temperature", "width", "height")
    unknown = set(section) - {
        "device", "exposure", "white_balance_temperature", "width", "height",
        "fourcc", "backlight_compensation",
    }
    if unknown:
        raise ConfigError(f"camera 包含未知字段: {sorted(unknown)}")
    device = section["device"]
    exposure = section["exposure"]
    temperature = section["white_balance_temperature"]
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise ConfigError("camera.device 必须是非负整数")
    if not isinstance(exposure, (int, float)) or isinstance(exposure, bool):
        raise ConfigError("camera.exposure 必须是数字")
    if not isinstance(temperature, (int, float)) \
            or isinstance(temperature, bool) or not 1000 <= temperature <= 20000:
        raise ConfigError(
            "camera.white_balance_temperature 必须是 1000..20000 K 的数字")
    dimensions: list[int | None] = []
    for name in ("width", "height"):
        value = section[name]
        if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ConfigError(f"camera.{name} 必须是正整数或 null")
        dimensions.append(value)
    fourcc = section.get("fourcc")
    if fourcc is not None and (
            not isinstance(fourcc, str) or len(fourcc) != 4 or not fourcc.isascii()):
        raise ConfigError("camera.fourcc 必须是 4 个 ASCII 字符或 null")
    backlight_compensation = section.get("backlight_compensation")
    if backlight_compensation is not None:
        if not isinstance(backlight_compensation, (int, float)) \
                or isinstance(backlight_compensation, bool):
            raise ConfigError(
                "camera.backlight_compensation 必须是有限非负数或 null")
        try:
            backlight_compensation = float(backlight_compensation)
        except OverflowError as error:
            raise ConfigError(
                "camera.backlight_compensation 必须是有限非负数或 null") from error
        if not math.isfinite(backlight_compensation) or backlight_compensation < 0:
            raise ConfigError(
                "camera.backlight_compensation 必须是有限非负数或 null")
    return CameraConfig(
        device=device,
        exposure=float(exposure),
        white_balance_temperature=float(temperature),
        width=dimensions[0],
        height=dimensions[1],
        fourcc=fourcc,
        backlight_compensation=backlight_compensation,
    )
