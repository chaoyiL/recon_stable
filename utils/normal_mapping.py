"""Signed RGB residual LUT data, robust construction and raw interpolation."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import numpy as np
from scipy.spatial import cKDTree
from utils.surface_geometry import local_slopes_to_normals


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')


@dataclass(frozen=True)
class NormalCalibration:
    """颜色残差到局部坡度的完整查找表。"""

    slopes: np.ndarray
    variances: np.ndarray
    color_min: np.ndarray
    color_max: np.ndarray
    sigma_ref2: float
    original_valid: np.ndarray | None = None
    sample_counts: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        slopes=np.asarray(self.slopes,np.float32)
        variances=np.asarray(self.variances,np.float32)
        color_min=np.asarray(self.color_min,np.float32)
        color_max=np.asarray(self.color_max,np.float32)
        if slopes.ndim!=4 or slopes.shape[0]<2 or slopes.shape[-1]!=2 \
                or slopes.shape[:3]!=(slopes.shape[0],)*3:
            raise ValueError("slopes 必须是 NxNxNx2 立方查找表")
        if variances.shape!=slopes.shape[:3]:
            raise ValueError("variances 尺寸必须与 slopes 的前三维一致")
        if color_min.shape!=(3,) or color_max.shape!=(3,) \
                or not np.isfinite(color_min).all() or not np.isfinite(color_max).all() \
                or np.any(color_max<=color_min) or np.any(color_min>0) \
                or np.any(color_max<0):
            raise ValueError(
                "color_min/color_max 必须是包含零点的有效 RGB 三通道范围")
        if not np.isfinite(slopes).all() or not np.isfinite(variances).all() \
                or np.any(variances<0) or not np.isfinite(self.sigma_ref2) \
                or self.sigma_ref2<=0:
            raise ValueError("法向标定模型包含无效数值")
        if self.original_valid is not None \
                and np.asarray(self.original_valid).shape!=variances.shape:
            raise ValueError("original_valid 尺寸无效")
        if self.sample_counts is not None \
                and np.asarray(self.sample_counts).shape!=variances.shape:
            raise ValueError("sample_counts 尺寸无效")
        if not isinstance(self.metadata, dict):
            raise ValueError("LUT metadata must be a mapping")
        object.__setattr__(self,"metadata",dict(self.metadata))
        object.__setattr__(self,"slopes",slopes)
        object.__setattr__(self,"variances",variances)
        object.__setattr__(self,"color_min",color_min)
        object.__setattr__(self,"color_max",color_max)
        object.__setattr__(self,"sigma_ref2",float(self.sigma_ref2))

    @property
    def size(self) -> int:
        return int(self.slopes.shape[0])

    def save(self,path: str | Path,**metadata: Any) -> Path:
        output=Path(path).expanduser()
        output.parent.mkdir(parents=True,exist_ok=True)
        values: dict[str,Any]={
            "format_version":np.asarray(5,np.int32),
            "color_residual_mode":np.asarray("signed"),
            "slopes":self.slopes,
            "variances":self.variances,
            "color_min":self.color_min,
            "color_max":self.color_max,
            "sigma_ref2":np.asarray(self.sigma_ref2,np.float32),
        }
        if self.original_valid is not None:
            values["original_valid"]=np.asarray(self.original_valid,np.bool_)
        if self.sample_counts is not None:
            values["sample_counts"]=np.asarray(self.sample_counts,np.int32)
        combined = dict(self.metadata)
        combined.update(metadata)
        for key, value in metadata.items():
            if key in values or key == "metadata":
                raise ValueError(f"Metadata must not overwrite LUT field {key}")
            array = np.asarray(value) if not isinstance(value, dict) else None
            values[key] = (array if array is not None and not array.dtype.hasobject else
                           np.asarray(json.dumps(value, ensure_ascii=False, default=_json_value)))
        values["metadata"] = np.asarray(json.dumps(combined, ensure_ascii=False, default=_json_value))
        temporary=output.with_suffix(output.suffix+".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,**values)
        temporary.replace(output)
        return output

    @classmethod
    def load(cls,path: str | Path) -> "NormalCalibration":
        source=Path(path).expanduser()
        with np.load(source,allow_pickle=False) as data:
            version=int(data["format_version"]) if "format_version" in data else 0
            if version not in {4,5}:
                raise ValueError(f"不支持的法向标定模型版本: {version}")
            if "color_residual_mode" not in data \
                    or str(data["color_residual_mode"])!="signed":
                raise ValueError("法向 LUT 缺少有符号色差语义；请重新运行 calibrate-norm")
            metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
            return cls(
                metadata=metadata,
                slopes=data["slopes"],variances=data["variances"],
                color_min=data["color_min"],color_max=data["color_max"],
                sigma_ref2=float(data["sigma_ref2"]),
                original_valid=(data["original_valid"]
                                if "original_valid" in data else None),
                sample_counts=(data["sample_counts"]
                               if "sample_counts" in data else None),
            )


def build_normal_lut(
    colors: np.ndarray,
    slopes: np.ndarray,
    *,
    size: int = 64,
    maximum_rms_angle_degrees: float = 5.0,
) -> NormalCalibration:
    """建立有符号 RGB→局部坡度 LUT；保留 MAD 后满足角度阈值的非空节点。"""
    color = np.asarray(colors, np.float64).reshape(-1, 3) # 颜色
    slope = np.asarray(slopes, np.float64).reshape(-1, 2) # 坡度
    if color.shape[0] != slope.shape[0]:
        raise ValueError("Color and slope sample counts differ")
    finite = np.all(np.isfinite(color), axis=1) & np.all(np.isfinite(slope), axis=1)
    color = color[finite]
    slope = slope[finite]
    if color.shape[0] != slope.shape[0] or color.shape[0] < 1:
        raise ValueError("颜色与坡度样本必须非空且逐点对应")
    if size < 2 or maximum_rms_angle_degrees <= 0:
        raise ValueError("LUT 尺寸和角度阈值必须为正")

    # 计算归一化颜色坐标
    color_min = np.minimum(np.min(color, axis=0), 0)
    color_max = np.maximum(np.max(color, axis=0), 0)
    span = color_max - color_min
    if np.any(span <= 0):
        raise ValueError("每个色差通道必须覆盖非零范围")
    coordinates = np.rint(
        np.clip((color - color_min) / span, 0, 1) * (size - 1)
    ).astype(np.int32)
    flat_nodes = np.ravel_multi_index(coordinates.T, (size, size, size))
    order = np.argsort(flat_nodes)
    flat_nodes = flat_nodes[order]
    slope = slope[order]

    lut = np.zeros((size, size, size, 2), np.float64)
    variance = np.zeros((size, size, size), np.float64)
    sample_counts = np.zeros((size, size, size), np.int32)
    original_valid = np.zeros((size, size, size), np.bool_)

    maximum_rms = np.deg2rad(maximum_rms_angle_degrees) # 传入阈值
    starts = np.r_[0, np.flatnonzero(np.diff(flat_nodes)) + 1]
    ends = np.r_[starts[1:], flat_nodes.size]
    for start, end in zip(starts, ends, strict=True):
        values = slope[start:end]
        normals, _ = local_slopes_to_normals(values, np.array([0., 0., 1.]), np.eye(3)[:, :2])
        mean = np.sum(normals, axis=0)
        mean /= np.maximum(np.linalg.norm(mean), 1e-12)
        angles = np.arccos(np.clip(normals @ mean, -1, 1))
        median = np.median(angles)
        mad = 1.4826 * np.median(np.abs(angles - median))
        kept = angles <= median + 3 * mad + 1e-12
        values = values[kept]
        normals = normals[kept]
        mean_normal = np.sum(normals, axis=0)
        mean_normal /= np.maximum(np.linalg.norm(mean_normal), 1e-12)
        rms = np.sqrt(np.mean(np.arccos(
            np.clip(normals @ mean_normal, -1, 1)) ** 2))
        if rms > maximum_rms:
            continue
        node = np.unravel_index(int(flat_nodes[start]), (size, size, size))
        node_mean = np.mean(values, axis=0)
        lut[node] = node_mean
        variance[node] = np.mean(np.sum((values - node_mean) ** 2, axis=1))
        sample_counts[node] = values.shape[0]
        original_valid[node] = True

    valid_coordinates = np.argwhere(original_valid)
    if not valid_coordinates.size:
        raise ValueError("没有满足角度误差要求的 LUT 节点")
    sigma_ref2 = max(float(np.median(variance[original_valid])), 1e-8)
    missing = np.argwhere(~original_valid)
    tree = cKDTree(valid_coordinates / (size - 1))
    neighbor_count = min(8, valid_coordinates.shape[0])
    for start in range(0, missing.shape[0], 65536):
        target = missing[start:start + 65536]
        distances, neighbors = tree.query(
            target / (size - 1), k=neighbor_count)
        if neighbor_count == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        weights = 1 / np.maximum(distances, 1e-12) ** 2
        weights /= np.sum(weights, axis=1, keepdims=True)
        source = valid_coordinates[neighbors]
        source_slopes = lut[source[..., 0], source[..., 1], source[..., 2]]
        filled = np.sum(weights[..., None] * source_slopes, axis=1)
        source_variance = variance[source[..., 0], source[..., 1], source[..., 2]]
        filled_variance = np.sum(
            weights * (source_variance + np.sum(
                (source_slopes - filled[:, None]) ** 2, axis=-1)), axis=1)
        lut[target[:, 0], target[:, 1], target[:, 2]] = filled
        variance[target[:, 0], target[:, 1], target[:, 2]] = filled_variance

    zero = (size - 1) * np.clip((0 - color_min) / span, 0, 1)
    lower = np.floor(zero).astype(int)
    upper = np.minimum(lower + 1, size - 1)
    for red in {lower[0], upper[0]}:
        for green in {lower[1], upper[1]}:
            for blue in {lower[2], upper[2]}:
                lut[red, green, blue] = 0
                variance[red, green, blue] = sigma_ref2
    return NormalCalibration(
        slopes=lut.astype(np.float32),
        variances=variance.astype(np.float32),
        color_min=color_min.astype(np.float32),
        color_max=color_max.astype(np.float32),
        sigma_ref2=sigma_ref2,
        original_valid=original_valid,
        sample_counts=sample_counts,
    )



def find_ambiguous_color_pairs(
    colors: np.ndarray,
    slopes: np.ndarray,
    *,
    color_distance_threshold: float = .01,
    normal_angle_threshold_degrees: float = 20.,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return conflicting unordered sample pairs, RGB distances and angles.

    colors is Nx3 signed linear RGB; slopes is Nx2 dimensionless local p/q.
    A pair conflicts when RGB L2 distance <= the color threshold and the angle
    between normalize([-p, -q, 1]) exceeds the degree threshold. Both samples
    share the LUT's local frame convention, regardless of their image or position.
    Every pair is considered once (i < j), including pairs within one image.
    """
    colors = np.asarray(colors, np.float64)
    slopes = np.asarray(slopes, np.float64)
    if colors.ndim != 2 or colors.shape[1] != 3 or slopes.shape != (len(colors), 2):
        raise ValueError('Expected paired Nx3 colors and Nx2 slopes')
    if not np.isfinite(colors).all() or not np.isfinite(slopes).all():
        raise ValueError('Ambiguity samples must be finite')
    if (not np.isfinite([color_distance_threshold, normal_angle_threshold_degrees]).all()
            or color_distance_threshold < 0 or not 0 <= normal_angle_threshold_degrees <= 180):
        raise ValueError('Color threshold must be nonnegative; normal angle must be in [0, 180] degrees')
    pairs = cKDTree(colors).query_pairs(color_distance_threshold, output_type='ndarray')
    normals, _ = local_slopes_to_normals(slopes, np.array([0., 0., 1.]), np.eye(3)[:, :2])
    first, second = normals[pairs[:, 0]], normals[pairs[:, 1]]
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(first, second), axis=-1),
                                   np.sum(first * second, axis=-1)))
    selected = angles > normal_angle_threshold_degrees
    pairs = pairs[selected]
    distances = np.linalg.norm(colors[pairs[:, 0]] - colors[pairs[:, 1]], axis=-1)
    return pairs, distances, angles[selected]


def lookup_slopes_raw(color_residual, calibration, valid_mask=None):
    colors=np.asarray(color_residual,np.float32)
    if colors.shape[-1:] != (3,):
        raise ValueError('Color residual must end in RGB')
    valid=np.isfinite(colors).all(axis=-1)
    if valid_mask is not None:
        mask=np.asarray(valid_mask,bool)
        if mask.shape != valid.shape:
            raise ValueError('Color valid mask shape mismatch')
        valid &= mask
    scale=calibration.color_max-calibration.color_min
    colors=np.where(valid[...,None],colors,0)
    normalized=np.clip((colors-calibration.color_min)/scale,0,1)
    coordinate=normalized*(calibration.size-1)
    lower=np.floor(coordinate).astype(np.int32)
    upper=np.minimum(lower+1,calibration.size-1)
    fraction=coordinate-lower
    slopes=np.zeros((*colors.shape[:-1],2),np.float32)
    variance=np.zeros(colors.shape[:-1],np.float32)
    corner_values=[]
    corner_variances=[]
    corner_weights=[]
    for ar in (0,1):
        ir=np.where(ar,upper[...,0],lower[...,0])
        wr=np.where(ar,fraction[...,0],1-fraction[...,0])
        for ag in (0,1):
            ig=np.where(ag,upper[...,1],lower[...,1])
            wg=np.where(ag,fraction[...,1],1-fraction[...,1])
            for ab in (0,1):
                ib=np.where(ab,upper[...,2],lower[...,2])
                wb=np.where(ab,fraction[...,2],1-fraction[...,2])
                weight=(wr*wg*wb).astype(np.float32)
                value=calibration.slopes[ir,ig,ib]
                corner_values.append(value)
                corner_variances.append(calibration.variances[ir,ig,ib])
                corner_weights.append(weight)
                slopes += weight[...,None]*value
    for weight,value,node_variance in zip(
            corner_weights,corner_values,corner_variances,strict=True):
        variance += weight*(node_variance+np.sum((value-slopes)**2,axis=-1))
    confidence=1/(1+variance/calibration.sigma_ref2)

    return (np.where(valid[...,None],slopes,0).astype(np.float32),
            np.where(valid,variance,0).astype(np.float32),
            np.where(valid,confidence,0).astype(np.float32))
