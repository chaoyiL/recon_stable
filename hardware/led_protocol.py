"""LED layout, default colours and variable-length RGB serial frames."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real

BAUD = 115200
HEADER = b"\xaa\x55"
# Includes the dark prefix; keep in sync with the Uno firmware's memory limit.
MAX_LED_COUNT = 128
SIDES = ("top", "right", "bottom", "left")
LAYOUT_FIELDS = (*SIDES, "skip")
LedColor = tuple[int, int, int, int]  # R, G, B (0..255), brightness (0..100).


@dataclass(frozen=True)
class LedLayout:
    """Wire order: dark prefix, top L→R, right T→B, bottom R→L, left B→T."""

    top: int = 9
    right: int = 4
    bottom: int = 9
    left: int = 4
    skip: int = 3

    def __post_init__(self) -> None:
        for name in LAYOUT_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.active_count == 0:
            raise ValueError("四边至少需要一颗灯珠")
        if self.total_count > MAX_LED_COUNT:
            raise ValueError(f"四边数量加空出数量不能超过 {MAX_LED_COUNT} 颗")

    @classmethod
    def from_mapping(cls, values: Mapping) -> "LedLayout":
        if not isinstance(values, Mapping):
            raise ValueError("led_controller 必须是字典")
        unknown = values.keys() - set(LAYOUT_FIELDS)
        if unknown:
            raise ValueError(f"led_controller 包含未知字段: {sorted(unknown)}")
        return cls(**values)

    @property
    def edge_counts(self) -> tuple[int, int, int, int]:
        return self.top, self.right, self.bottom, self.left

    @property
    def active_count(self) -> int:
        return sum(self.edge_counts)

    @property
    def total_count(self) -> int:
        return self.skip + self.active_count

    def grid_positions(self) -> list[tuple[int, int]]:
        """Return one non-overlapping GUI cell per controllable LED, in wire order."""
        columns = max(self.top, self.bottom, 1)
        rows = max(self.right, self.left, 1)

        def spread(count: int, span: int) -> list[int]:
            if count == 1:
                return [1 + (span - 1) // 2]
            return [1 + round(i * (span - 1) / (count - 1)) for i in range(count)]

        return (
            [(0, col) for col in spread(self.top, columns)]
            + [(row, columns + 1) for row in spread(self.right, rows)]
            + [(rows + 1, col) for col in reversed(spread(self.bottom, columns))]
            + [(row, 0) for row in reversed(spread(self.left, rows))]
        )


def _validated_rgb_scale(values: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 3:
        raise ValueError("rgb_scale 必须包含三个 RGB 校正系数")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values
    ):
        raise ValueError("rgb_scale 的三个系数必须是 0..1 范围内的有限数字")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class LedSettings:
    """Persistent LED output settings loaded from the led_controller mapping."""

    layout: LedLayout = LedLayout()
    brightness: int = 100
    rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, LedLayout):
            raise ValueError("layout 必须是 LedLayout")
        if type(self.brightness) is not int or not 0 <= self.brightness <= 100:
            raise ValueError("brightness 必须是 0..100 的整数")
        object.__setattr__(self, "rgb_scale", _validated_rgb_scale(self.rgb_scale))

    @classmethod
    def from_mapping(cls, values: Mapping) -> "LedSettings":
        if not isinstance(values, Mapping):
            raise ValueError("led_controller 必须是字典")
        fields = {*LAYOUT_FIELDS, "brightness", "rgb_scale"}
        unknown = values.keys() - fields
        if unknown:
            raise ValueError(f"led_controller 包含未知字段: {sorted(unknown)}")
        layout_values = {name: values[name] for name in LAYOUT_FIELDS if name in values}
        return cls(
            layout=LedLayout.from_mapping(layout_values),
            brightness=values.get("brightness", 100),
            rgb_scale=values.get("rgb_scale", (1.0, 1.0, 1.0)),
        )


def default_colors(layout: LedLayout) -> list[LedColor]:
    """Keep the original four-side pattern, stretching gradients to each edge."""
    colors = []
    for side, count in zip(SIDES, layout.edge_counts):
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            if side == "top":
                rgb = (191, round(64 * t), round(64 * (1 - t)))
            elif side == "right":
                rgb = (0, 255, 0)
            elif side == "bottom":
                rgb = (0, round(255 * (1 - t)), round(255 * t))
            else:
                rgb = (0, 0, 255)
            if side in ("top", "bottom") and i in (0, count - 1):
                rgb = tuple(min(255, round(value * 1.3)) for value in rgb)
            colors.append((*rgb, 100))
    return colors


def encode_frame(
    layout: LedLayout,
    colors: Sequence[LedColor],
    brightness: int = 100,
    *,
    rgb_scale: Sequence[float] = (1.0, 1.0, 1.0),
) -> bytes:
    """AA 55 + uint16 little-endian total count + RGB + XOR(count bytes and RGB).

    ``colors`` contains only the four edges. The dark prefix is always zero;
    per-LED and global brightness plus ``rgb_scale`` are applied before uint8
    transmission. The scale adjusts logical R, G and B output independently.
    """
    if len(colors) != layout.active_count:
        raise ValueError("颜色数量必须与四边灯珠总数一致")
    if type(brightness) is not int or not 0 <= brightness <= 100:
        raise ValueError("整体亮度必须是 0..100 的整数")
    rgb_scale = _validated_rgb_scale(rgb_scale)
    data = bytearray(layout.total_count.to_bytes(2, "little"))
    data.extend(bytes(3 * layout.skip))
    for color in colors:
        if len(color) != 4 or any(type(value) is not int for value in color):
            raise ValueError("每颗灯珠颜色必须为四个整数 R、G、B、亮度")
        r, g, b, level = color
        if any(not 0 <= value <= 255 for value in (r, g, b)) or not 0 <= level <= 100:
            raise ValueError("RGB 必须在 0..255，灯珠亮度必须在 0..100")
        brightness_scale = level / 100 * brightness / 100
        data.extend(
            int(value * brightness_scale * channel_scale)
            for value, channel_scale in zip((r, g, b), rgb_scale)
        )
    checksum = 0
    for value in data:
        checksum ^= value
    return HEADER + data + bytes([checksum])
