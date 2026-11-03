"""整体重建实现与三套材料坐标系之间的稳定数据接口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from utils.config import ConfigError, load_config


COORDINATE_SYSTEM_NAMES = ("geometry", "lightfield", "observation")


@dataclass(frozen=True)
class CoordinateGridSpec:
    """一套材料坐标网格的目标尺寸，不规定如何生成它。"""

    rows: int
    columns: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 2
            for value in (self.rows, self.columns)
        ):
            raise ValueError("材料坐标网格 rows/columns 必须是至少为 2 的整数")


@dataclass(frozen=True)
class MaterialCoordinateSpecs:
    geometry: CoordinateGridSpec
    lightfield: CoordinateGridSpec
    observation: CoordinateGridSpec


@dataclass(frozen=True)
class MaterialCoordinateMap:
    """一套材料坐标的几何和成像对应；不包含模板或归一化状态。"""

    xyz: np.ndarray
    uv: np.ndarray
    coordinates: np.ndarray
    valid_mask: np.ndarray
    st: np.ndarray | None = None
    coordinate_kind: str = "template_parameter"
    chart_id: str | None = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, np.float32)
        uv = np.asarray(self.uv, np.float32)
        coordinates = np.asarray(self.coordinates, np.float32)
        valid = np.asarray(self.valid_mask, np.bool_)
        shape = xyz.shape[:2]
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("xyz 必须是 rows x columns x 3")
        if uv.shape != (*shape, 2) or coordinates.shape != (*shape, 2):
            raise ValueError("uv/material coordinates 必须是 rows x columns x 2")
        if valid.shape != shape:
            raise ValueError("valid_mask 尺寸必须与材料坐标网格一致")
        finite = (
            np.all(np.isfinite(xyz), axis=-1)
            & np.all(np.isfinite(uv), axis=-1)
            & np.all(np.isfinite(coordinates), axis=-1)
        )
        if np.any(valid & ~finite):
            raise ValueError("有效材料坐标点必须全部为有限数")
        if self.st is not None:
            st = np.asarray(self.st, np.float64)
            if st.shape != (*shape, 2) or not np.isfinite(st).all():
                raise ValueError("st must be a finite rows x columns x 2 grid")
            object.__setattr__(self, "st", st)
        if self.coordinate_kind == "reference_arclength" and (self.st is None or not self.chart_id):
            raise ValueError("Reference arc-length coordinates require st and chart_id")
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "valid_mask", valid)


@dataclass(frozen=True)
class MaterialCoordinateSystems:
    """整体重建 provider 必须同时返回的三套材料坐标。"""

    geometry: MaterialCoordinateMap
    lightfield: MaterialCoordinateMap
    observation: MaterialCoordinateMap

    def validate_against(self, specs: MaterialCoordinateSpecs) -> None:
        for name in COORDINATE_SYSTEM_NAMES:
            value = getattr(self, name)
            spec = getattr(specs, name)
            if value.xyz.shape[:2] != (spec.rows, spec.columns):
                raise ValueError(
                    f"{name} 坐标尺寸 {value.xyz.shape[:2]} 与接口配置 "
                    f"{(spec.rows, spec.columns)} 不一致")


@runtime_checkable
class OverallReconstructionProvider(Protocol):
    """整体重建实现；frame 参数为 HxWx3 float32 [0, 1] 线性 RGB。"""

    def reconstruct(
        self,
        frame_linear_rgb: np.ndarray,
        specs: MaterialCoordinateSpecs,
        surface_mask: np.ndarray | None = None,
    ) -> MaterialCoordinateSystems:
        """从 HxWx3 float32 [0, 1] 线性 RGB 图像返回坐标；mask 限定可选有效域。"""


def _grid_spec(raw: Any, name: str) -> CoordinateGridSpec:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"material_coordinates.{name} 必须是字典")
    unknown = set(raw) - {"rows", "columns"}
    if unknown:
        raise ConfigError(
            f"material_coordinates.{name} 包含未知字段: {sorted(unknown)}")
    try:
        return CoordinateGridSpec(rows=raw["rows"], columns=raw["columns"])
    except KeyError as error:
        raise ConfigError(
            f"material_coordinates.{name} 缺少 rows/columns") from error
    except ValueError as error:
        raise ConfigError(f"material_coordinates.{name}: {error}") from error


def parse_material_coordinate_specs(raw: Any) -> MaterialCoordinateSpecs:
    if not isinstance(raw, Mapping):
        raise ConfigError("material_coordinates 必须是字典")
    unknown = set(raw) - set(COORDINATE_SYSTEM_NAMES)
    if unknown:
        raise ConfigError(f"material_coordinates 包含未知字段: {sorted(unknown)}")
    return MaterialCoordinateSpecs(**{
        name: _grid_spec(raw.get(name), name)
        for name in COORDINATE_SYSTEM_NAMES
    })


def load_material_coordinate_specs(
    config_path: str | Path,
) -> MaterialCoordinateSpecs:
    return parse_material_coordinate_specs(
        load_config(config_path).get("material_coordinates"))
