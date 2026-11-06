"""曲率感知的视触觉局部形变重建。

在冻结参考弧长网格上求解位移；原图颜色按两侧各自 UV 采样后相减。
几何未知量覆盖 marker 位置，光度方程仅覆盖有效颜色。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from utils.color_difference import (CAMERA_RGB_TRANSFER_FUNCTION,
                                    require_camera_rgb_transfer, signed_color_difference)
from utils.config import load_config, load_contact_surface_offset_mm
from utils.color_protection import resolve_zero_color_radii, signed_color_protection_factors
from utils.image_io import cached_linear_rgb


from utils.normal_mapping import NormalCalibration, lookup_slopes_raw
from utils.surface_geometry import compute_surface_geometry as _surface_geometry
from utils.surface_geometry import local_slopes_to_normals


@dataclass(frozen=True)
class LocalReconstructionResult:
    xyz_out: np.ndarray
    displacement: np.ndarray
    displacement_vectors: np.ndarray
    reference_normals: np.ndarray
    slopes: np.ndarray
    observed_normals: np.ndarray
    curvature_correction: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    boundary_mask: np.ndarray
    solver_istop: int
    solver_iterations: int
    solver_residual_norm: float
    trusted_no_contact_mask: np.ndarray | None = None
    photometric_valid_mask: np.ndarray | None = None
    inferred_mask: np.ndarray | None = None
    contact_offset_mm: float | None = None

    def save(self,path: str | Path) -> Path:
        output=Path(path).expanduser()
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix(output.suffix+".tmp")
        values={
            "xyz_out":self.xyz_out,"displacement":self.displacement,
            "displacement_vectors":self.displacement_vectors,
            "reference_normals":self.reference_normals,"slopes":self.slopes,
            "observed_normals":self.observed_normals,
            "curvature_correction":self.curvature_correction,
            "confidence":self.confidence,"valid_mask":self.valid_mask,
            "boundary_mask":self.boundary_mask,
            "solver_istop":np.asarray(self.solver_istop,np.int32),
            "solver_iterations":np.asarray(self.solver_iterations,np.int32),
            "solver_residual_norm":np.asarray(
                self.solver_residual_norm,np.float64),
        }
        if self.photometric_valid_mask is not None:
            values["photometric_valid_mask"] = self.photometric_valid_mask
        if self.inferred_mask is not None:
            values["inferred_mask"] = self.inferred_mask
        if self.trusted_no_contact_mask is not None:
            values["trusted_no_contact_mask"]=np.asarray(
                self.trusted_no_contact_mask,np.bool_)
        if self.contact_offset_mm is not None:
            values["contact_offset_mm"] = np.asarray(self.contact_offset_mm, np.float64)
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,**values)
        temporary.replace(output)
        return output


@dataclass(frozen=True)
class LocalReconstructionSettings:
    input_file: Path
    calibration_file: Path
    output_file: Path
    lsmr_atol: float
    lsmr_btol: float
    lsmr_max_iterations: int | None
    zero_color_inner_radius: float
    zero_color_outer_radius: float
    no_contact_constraints_enabled: bool
    trusted_no_contact_confidence: float
    displacement_zero_lambda_per_mm2: float
    zero_color_negative_inner_radius: float | None = None
    zero_color_negative_outer_radius: float | None = None


@dataclass(frozen=True)
class NoContactResidualModel:
    """启动无触碰帧得到的位置相关残差中心和鲁棒通道尺度。"""

    center: np.ndarray
    channel_scale: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        center=np.asarray(self.center,np.float32)
        scale=np.asarray(self.channel_scale,np.float32)
        valid=np.asarray(self.valid_mask,np.bool_)
        if center.ndim!=3 or center.shape[-1]!=3 \
                or valid.shape!=center.shape[:2] or scale.shape!=(3,) \
                or not np.isfinite(center).all() or not np.isfinite(scale).all() \
                or np.any(scale<=0):
            raise ValueError("无接触残差模型尺寸或数值无效")
        object.__setattr__(self,"center",center)
        object.__setattr__(self,"channel_scale",scale)
        object.__setattr__(self,"valid_mask",valid)


@dataclass(frozen=True)
class NoContactConstraintSettings:
    enabled: bool
    minimum_startup_valid_fraction: float
    minimum_channel_scale: float
    trusted_score_threshold: float
    contact_guard_score_threshold: float
    contact_guard_radius_pixels: int
    surface_edge_margin_pixels: int
    slope_confidence: float
    displacement_zero_lambda_per_mm2: float


def parse_no_contact_constraints(config: object) -> NoContactConstraintSettings:
    """解析可信无接触门控和绝对零位移软约束。"""
    if not isinstance(config,dict):
        raise ValueError("local_reconstruction 必须是映射")
    raw=config.get("no_contact_constraints",{})
    if raw is None:
        raw={}
    if not isinstance(raw,dict):
        raise ValueError(
            "local_reconstruction.no_contact_constraints 必须是映射")
    fields={
        "enabled","minimum_startup_valid_fraction","minimum_channel_scale",
        "trusted_score_threshold","contact_guard_score_threshold",
        "contact_guard_radius_pixels","surface_edge_margin_pixels",
        "slope_confidence","displacement_zero_lambda_per_mm2",
    }
    unknown=set(raw)-fields
    if unknown:
        raise ValueError(
            "local_reconstruction.no_contact_constraints 包含未知字段: "+
            ", ".join(sorted(unknown)))
    enabled=raw.get("enabled",False)
    minimum_fraction=float(raw.get("minimum_startup_valid_fraction",.8))
    minimum_scale=float(raw.get("minimum_channel_scale",.005))
    trusted=float(raw.get("trusted_score_threshold",2.5))
    guard=float(raw.get("contact_guard_score_threshold",5.))
    guard_radius=raw.get("contact_guard_radius_pixels",20)
    edge_margin=raw.get("surface_edge_margin_pixels",4)
    slope_confidence=float(raw.get("slope_confidence",1.))
    zero_lambda=float(raw.get("displacement_zero_lambda_per_mm2",.25))
    if not isinstance(enabled,bool) \
            or not 0<minimum_fraction<=1 or minimum_scale<=0 \
            or trusted<=0 or guard<=trusted \
            or not isinstance(guard_radius,int) or isinstance(guard_radius,bool) \
            or guard_radius<0 \
            or not isinstance(edge_margin,int) or isinstance(edge_margin,bool) \
            or edge_margin<0 or not 0<slope_confidence<=1 or zero_lambda<0 \
            or not np.isfinite([
                minimum_fraction,minimum_scale,trusted,guard,
                slope_confidence,zero_lambda]).all():
        raise ValueError("no_contact_constraints 参数无效")
    return NoContactConstraintSettings(
        enabled=enabled,
        minimum_startup_valid_fraction=minimum_fraction,
        minimum_channel_scale=minimum_scale,
        trusted_score_threshold=trusted,
        contact_guard_score_threshold=guard,
        contact_guard_radius_pixels=guard_radius,
        surface_edge_margin_pixels=edge_margin,
        slope_confidence=slope_confidence,
        displacement_zero_lambda_per_mm2=zero_lambda)


def fit_no_contact_residual_model(
    residual_samples: np.ndarray,
    valid_masks: np.ndarray,
    *,
    minimum_valid_fraction: float = .8,
    minimum_channel_scale: float = .005,
) -> NoContactResidualModel:
    """从启动无触碰帧拟合规范曲面残差中心及全局鲁棒通道尺度。"""
    samples=np.asarray(residual_samples,np.float64)
    masks=np.asarray(valid_masks,np.bool_)
    if samples.ndim!=4 or samples.shape[-1]!=3 \
            or masks.shape!=samples.shape[:3] or samples.shape[0]<2 \
            or not np.isfinite(samples).all() \
            or not 0<minimum_valid_fraction<=1 \
            or not np.isfinite(minimum_channel_scale) \
            or minimum_channel_scale<=0:
        raise ValueError("无接触残差样本、有效域或拟合参数无效")
    counts=np.sum(masks,axis=0)
    required=int(np.ceil(minimum_valid_fraction*samples.shape[0]))
    model_valid=counts>=required
    masked=np.ma.array(
        samples,mask=np.broadcast_to(~masks[...,None],samples.shape))
    center=np.ma.median(masked,axis=0).filled(0.)
    deviation=samples-center[None]
    scale=np.asarray([
        1.4826*np.median(np.abs(deviation[...,channel][masks]))
        if np.any(masks) else 0.
        for channel in range(3)],np.float64)
    scale=np.maximum(scale,float(minimum_channel_scale))
    center=np.where(model_valid[...,None],center,0.)
    return NoContactResidualModel(
        center.astype(np.float32),scale.astype(np.float32),model_valid)


def classify_trusted_no_contact(
    color_residual: np.ndarray,
    valid_mask: np.ndarray,
    model: NoContactResidualModel,
    *,
    trusted_score_threshold: float,
    contact_guard_score_threshold: float,
    contact_guard_radius_pixels: int = 0,
    surface_edge_margin_pixels: int = 0,
) -> tuple[np.ndarray,np.ndarray]:
    """以无接触统计分数和强接触邻域保护生成保守可信无接触掩膜。"""
    colors=np.asarray(color_residual,np.float32)
    valid=np.asarray(valid_mask,np.bool_)
    if colors.shape!=model.center.shape or valid.shape!=colors.shape[:2] \
            or trusted_score_threshold<=0 \
            or contact_guard_score_threshold<=trusted_score_threshold \
            or contact_guard_radius_pixels<0 or surface_edge_margin_pixels<0:
        raise ValueError("无接触分类输入尺寸或阈值无效")
    score=np.sqrt(np.sum(
        ((colors-model.center)/model.channel_scale)**2,axis=-1))
    input_valid=valid&np.isfinite(score)
    usable=input_valid&model.valid_mask
    if surface_edge_margin_pixels:
        radius=surface_edge_margin_pixels
        kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
        usable=cv2.erode(
            usable.astype(np.uint8),kernel,iterations=1,
            borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    guarded=input_valid&(score>=contact_guard_score_threshold)
    if contact_guard_radius_pixels:
        radius=contact_guard_radius_pixels
        kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
        guarded=cv2.dilate(
            guarded.astype(np.uint8),kernel,iterations=1,
            borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    trusted=usable&(score<=trusted_score_threshold)&~guarded
    return trusted,np.where(input_valid,score,0).astype(np.float32)


def parse_zero_color_protection(config: object) -> tuple[float, float, float, float]:
    """Return positive inner/outer and negative inner/outer magnitudes."""
    if not isinstance(config,dict):
        raise ValueError("local_reconstruction 必须是映射")
    raw=config.get("zero_color_protection",{})
    if raw is None:
        raw={}
    if not isinstance(raw,dict):
        raise ValueError(
            "local_reconstruction.zero_color_protection 必须是映射")
    unknown=set(raw)-{"inner_radius","outer_radius","negative_inner_radius","negative_outer_radius"}
    if unknown:
        raise ValueError(
            "local_reconstruction.zero_color_protection 包含未知字段: "+
            ", ".join(sorted(unknown)))
    return resolve_zero_color_radii(raw.get("inner_radius",0.), raw.get("outer_radius",0.),
                                    raw.get("negative_inner_radius"), raw.get("negative_outer_radius"))


def _configured_path(
    cli_value: str | None,
    configured_value: object,
    *,
    config_base: Path,
    name: str,
) -> Path:
    if cli_value is not None:
        if not cli_value:
            raise ValueError(f"{name} 命令行路径不能为空")
        return Path(cli_value).expanduser()
    if not isinstance(configured_value,str) or not configured_value:
        raise ValueError(
            f"缺少 local_reconstruction.{name}；请在 config.yaml 配置或用命令行覆盖")
    path=Path(configured_value).expanduser()
    return path if path.is_absolute() else config_base/path


def _optional_configured_path(
    cli_value: str | None,
    configured_value: object,
    *,
    config_base: Path,
    name: str,
) -> Path | None:
    """解析可选路径；显式传入空字符串仍视为配置错误。"""
    if cli_value is not None:
        if not cli_value:
            raise ValueError(f"{name} 命令行路径不能为空")
        return Path(cli_value).expanduser()
    if configured_value is None:
        return None
    if not isinstance(configured_value,str) or not configured_value:
        raise ValueError(f"local_reconstruction.{name} 必须是非空路径或 null")
    path=Path(configured_value).expanduser()
    return path if path.is_absolute() else config_base/path


def default_reconstruction_output(
    config_path: str | Path, output_dir: object, filename: str,
) -> Path:
    """Resolve a default result NPZ under the configured output directory's pc/ subfolder."""
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError('reconstruction.output_dir 必须是非空路径')
    directory = Path(output_dir).expanduser()
    if not directory.is_absolute():
        directory = Path(config_path).expanduser().parent / directory
    return directory / 'pc' / filename


def load_local_reconstruction_settings(
    config_path: str | Path,
    *,
    input_override: str | None = None,
    calibration_override: str | None = None,
    output_override: str | None = None,
    lsmr_atol_override: float | None = None,
    lsmr_btol_override: float | None = None,
    lsmr_max_iterations_override: int | None = None,
) -> LocalReconstructionSettings:
    """读取独立局部重建配置；命令行值优先于 YAML。"""
    source=Path(config_path).expanduser()
    all_config=load_config(source)
    if not isinstance(all_config,dict):
        raise ValueError("config.yaml 顶层必须是映射")
    raw=all_config.get("local_reconstruction")
    if not isinstance(raw,dict):
        raise ValueError("config.yaml 缺少 local_reconstruction 配置段")
    input_file=_optional_configured_path(
        input_override,raw.get("input_file"),config_base=source.parent,
        name="input_file")
    if input_file is None:
        raise ValueError(
            "local-recon 必须用 --input 指定 NPZ；单张图片重建请使用 recon --image")
    if calibration_override is not None:
        calibration_file=_configured_path(
            calibration_override,None,config_base=source.parent,
            name="calibration_file")
    else:
        calibration_file=_configured_path(
            None,raw.get("calibration_file"),config_base=source.parent,
            name="calibration_file")
    output_file=_optional_configured_path(
        output_override,raw.get("output_file"),config_base=source.parent,
        name="output_file")
    if output_file is None:
        reconstruction=all_config.get('reconstruction', {})
        if not isinstance(reconstruction, dict):
            raise ValueError('reconstruction 必须是映射')
        output_file=default_reconstruction_output(
            source, reconstruction.get('output_dir', 'assets/reconstruction'),
            input_file.stem+"_local_reconstruction.npz")
    atol=(float(raw.get("lsmr_atol",1e-7)) if lsmr_atol_override is None
          else float(lsmr_atol_override))
    btol=(float(raw.get("lsmr_btol",1e-7)) if lsmr_btol_override is None
          else float(lsmr_btol_override))
    maximum=(raw.get("lsmr_max_iterations")
             if lsmr_max_iterations_override is None
             else lsmr_max_iterations_override)
    if atol<=0 or btol<=0 or not np.isfinite(atol) or not np.isfinite(btol) \
            or (maximum is not None and
                (not isinstance(maximum,int) or isinstance(maximum,bool)
                 or maximum<1)):
        raise ValueError("local_reconstruction 的 LSMR 参数无效")
    zero_inner,zero_outer,negative_inner,negative_outer=parse_zero_color_protection(raw)
    no_contact=parse_no_contact_constraints(raw)
    return LocalReconstructionSettings(
        input_file=input_file,calibration_file=calibration_file,
        output_file=output_file,lsmr_atol=atol,lsmr_btol=btol,
        lsmr_max_iterations=maximum,
        zero_color_inner_radius=zero_inner,
        zero_color_outer_radius=zero_outer,
        zero_color_negative_inner_radius=negative_inner,
        zero_color_negative_outer_radius=negative_outer,
        no_contact_constraints_enabled=no_contact.enabled,
        trusted_no_contact_confidence=no_contact.slope_confidence,
        displacement_zero_lambda_per_mm2=(
            no_contact.displacement_zero_lambda_per_mm2))


def sample_full_resolution_residual(
    residual_linear_rgb: np.ndarray,
    uv: np.ndarray,
    valid_image: np.ndarray | None = None,
) -> tuple[np.ndarray,np.ndarray]:
    """按原生曲面 UV 从全分辨率线性 RGB 色差图双线性采样。"""
    from utils.surface_sampling import sample_linear_rgb_to_surface
    sampled=sample_linear_rgb_to_surface(residual_linear_rgb,uv,valid_image)
    return sampled.values,sampled.valid


def lookup_slopes(
    color_residual: np.ndarray,
    calibration: NormalCalibration,
    valid_mask: np.ndarray | None = None,
    *,
    zero_color_inner_radius: float = 0.,
    zero_color_outer_radius: float = 0.,
    zero_color_negative_inner_radius: float | None = None,
    zero_color_negative_outer_radius: float | None = None,
    trusted_no_contact_mask: np.ndarray | None = None,
    trusted_no_contact_confidence: float = 1.,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """查询完整 LUT；零坡度保护内区按无接触先验赋予零坡度和置信度 1。"""
    colors=np.asarray(color_residual,np.float32)
    if colors.ndim<1 or colors.shape[-1]!=3:
        raise ValueError("color_residual 最后一维必须是 RGB")
    radii=resolve_zero_color_radii(zero_color_inner_radius, zero_color_outer_radius,
                                    zero_color_negative_inner_radius, zero_color_negative_outer_radius)
    if not np.isfinite(trusted_no_contact_confidence) \
            or not 0<trusted_no_contact_confidence<=1:
        raise ValueError('trusted_no_contact_confidence must be in (0, 1]')
    valid=np.all(np.isfinite(colors),axis=-1)
    if valid_mask is not None:
        supplied=np.asarray(valid_mask,dtype=np.bool_)
        if supplied.shape!=colors.shape[:-1]:
            raise ValueError("valid_mask 尺寸与颜色残差不一致")
        valid &= supplied
    if trusted_no_contact_mask is None:
        trusted=np.zeros(colors.shape[:-1],np.bool_)
    else:
        trusted=np.asarray(trusted_no_contact_mask,np.bool_)
        if trusted.shape!=colors.shape[:-1]:
            raise ValueError("trusted_no_contact_mask 尺寸与颜色残差不一致")
        trusted &= valid
    slopes,variance,confidence=lookup_slopes_raw(colors,calibration,valid)
    attenuation,zero_slope=signed_color_protection_factors(colors, *radii)
    slopes *= attenuation[...,None]
    # LUT variance stays diagnostic; invalid photometry never receives the prior.
    confidence=np.where(valid & zero_slope,1.,confidence)
    slopes=np.where(trusted[...,None],0,slopes)
    confidence=np.where(
        trusted,np.maximum(confidence,trusted_no_contact_confidence),confidence)
    slopes=np.where(valid[...,None],slopes,0).astype(np.float32)
    variance=np.where(valid,variance,0).astype(np.float32)
    confidence=np.where(valid,confidence,0).astype(np.float32)
    return slopes,variance,confidence


def reconstruct_local_surface(
    xyz: np.ndarray,
    color_residual_grid: np.ndarray,
    calibration: NormalCalibration,
    *,
    valid_mask: np.ndarray | None = None,
    geometry_valid_mask: np.ndarray | None = None,
    photometric_valid_mask: np.ndarray | None = None,
    zero_color_inner_radius: float = 0.,
    zero_color_outer_radius: float = 0.,
    zero_color_negative_inner_radius: float | None = None,
    zero_color_negative_outer_radius: float | None = None,
    trusted_no_contact_mask: np.ndarray | None = None,
    trusted_no_contact_confidence: float = 1.,
    displacement_zero_lambda_per_mm2: float = 0.,
    lsmr_atol: float = 1e-7,
    lsmr_btol: float = 1e-7,
    lsmr_max_iterations: int | None = None,
) -> LocalReconstructionResult:
    """以等权颜色方程求解法向局部位移；置信度仅供输出和诊断。"""
    points=np.asarray(xyz,np.float64)
    colors=np.asarray(color_residual_grid,np.float32)
    if colors.shape!=points.shape:
        raise ValueError("color_residual_grid 必须与 xyz 同为 rows x columns x 3")
    valid=np.all(np.isfinite(points),axis=-1)
    if valid_mask is not None:
        supplied=np.asarray(valid_mask,dtype=np.bool_)
        if supplied.shape!=points.shape[:2]:
            raise ValueError("valid_mask 尺寸必须与 xyz 前两维一致")
        valid &= supplied
    if geometry_valid_mask is not None:
        supplied=np.asarray(geometry_valid_mask,bool)
        if supplied.shape != valid.shape:
            raise ValueError("geometry_valid_mask shape mismatch")
        valid &= supplied
    photo=valid & np.isfinite(colors).all(axis=-1)
    if photometric_valid_mask is not None:
        supplied=np.asarray(photometric_valid_mask,bool)
        if supplied.shape != valid.shape:
            raise ValueError("photometric_valid_mask shape mismatch")
        photo &= supplied
    if not photo.any():
        raise ValueError("局部重建没有有效颜色方程")
    if trusted_no_contact_mask is None:
        trusted_no_contact=np.zeros(points.shape[:2],np.bool_)
    else:
        trusted_no_contact=np.asarray(trusted_no_contact_mask,np.bool_)
        if trusted_no_contact.shape!=points.shape[:2]:
            raise ValueError("trusted_no_contact_mask 尺寸与 xyz 前两维不一致")
        trusted_no_contact = trusted_no_contact & photo
    if not np.isfinite(displacement_zero_lambda_per_mm2) \
            or displacement_zero_lambda_per_mm2<0:
        raise ValueError("displacement_zero_lambda_per_mm2 必须为非负有限数")
    slopes,_,confidence=lookup_slopes(
        colors,calibration,photo,
        zero_color_inner_radius=zero_color_inner_radius,
        zero_color_outer_radius=zero_color_outer_radius,
        zero_color_negative_inner_radius=zero_color_negative_inner_radius,
        zero_color_negative_outer_radius=zero_color_negative_outer_radius,
        trusted_no_contact_mask=trusted_no_contact,
        trusted_no_contact_confidence=trusted_no_contact_confidence)
    geometry=_surface_geometry(points,valid)
    equation_mask=geometry["valid"]&photo
    unknown_mask=geometry["interior"]
    unknown_index=np.full(points.shape[:2],-1,np.int32)
    unknown_index[unknown_mask]=np.arange(np.count_nonzero(unknown_mask),dtype=np.int32)
    if not np.any(equation_mask & unknown_mask):
        raise ValueError("局部重建没有有效内部颜色方程")
    frame=geometry["frame"]
    tangent=np.einsum("...ca,...a->...c",frame,slopes,optimize=True)
    jacobian=geometry["jacobian"]
    g_inverse=geometry["metric_inverse"]
    second=geometry["second"]
    parameter=np.einsum(
        "...ab,...cb,...c->...a",g_inverse,jacobian,tangent,optimize=True)
    shape_parameter=np.einsum(
        "...ab,...b->...a",second,parameter,optimize=True)
    curvature=np.einsum(
        "...ca,...ab,...b->...c",jacobian,g_inverse,shape_parameter,
        optimize=True)
    gamma=np.einsum("...ca,...c->...a",frame,curvature,optimize=True)
    k_matrix=geometry["k"]

    matrix_rows=[]; matrix_columns=[]; matrix_values=[]; targets=[]
    row_number=0
    rows,columns=points.shape[:2]
    for i,j in np.argwhere(equation_mask):
        center=unknown_index[i,j]
        if center<0:
            continue
        for component in range(2):
            entries=((i,j,float(gamma[i,j,component])),
                     (i,j+1,float(.5*k_matrix[i,j,component,0])),
                     (i,j-1,float(-.5*k_matrix[i,j,component,0])),
                     (i+1,j,float(.5*k_matrix[i,j,component,1])),
                     (i-1,j,float(-.5*k_matrix[i,j,component,1])))
            for ni,nj,value in entries:
                index=unknown_index[ni,nj]
                if index>=0 and value!=0:
                    matrix_rows.append(row_number)
                    matrix_columns.append(int(index))
                    matrix_values.append(value)
            targets.append(float(slopes[i,j,component]))
            row_number+=1
    if row_number==0:
        raise ValueError("局部重建没有可用的颜色方程")
    # Photometric holes remain displacement unknowns, connected only where colors are missing.
    for i,j in np.argwhere(unknown_mask & ~photo):
        neighbours=[(ni,nj) for ni,nj in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)) if valid[ni,nj]]
        if not neighbours:
            continue
        distances2=np.asarray([np.sum((points[ni,nj]-points[i,j])**2) for ni,nj in neighbours])
        weights=1/np.maximum(distances2,1e-12)
        weights/=weights.sum()
        h=np.sqrt(np.mean(distances2))
        if h <= 1e-12:
            continue
        matrix_rows.append(row_number)
        matrix_columns.append(int(unknown_index[i,j]))
        matrix_values.append(1/h)
        for (ni,nj),weight in zip(neighbours,weights):
            index=unknown_index[ni,nj]
            if index>=0:
                matrix_rows.append(row_number)
                matrix_columns.append(int(index))
                matrix_values.append(float(-weight/h))
        targets.append(0.)
        row_number+=1
    # Dprior0：可信无接触点只增加 lambda 控制的软 d=0 行。它不改变未知量域，
    # 因而不会把统计掩膜的方形形态硬刻进曲面。
    if displacement_zero_lambda_per_mm2>0:
        prior_scale=float(np.sqrt(displacement_zero_lambda_per_mm2))
        for i,j in np.argwhere(trusted_no_contact&unknown_mask):
            matrix_rows.append(row_number)
            matrix_columns.append(int(unknown_index[i,j]))
            matrix_values.append(prior_scale)
            targets.append(0.)
            row_number+=1
    # 中心差分在规则网格上存在四个奇偶子格常量零空间。连续问题的零 Dirichlet
    # 边界本应消除它们，但若边界点直接从未知量中删去，离散矩阵仍会保留这些
    # 棋盘格自由度。对四个子格分别用最靠近左上边界的 2x2 同奇偶点外推 d(0,0)=0；
    # 这只是离散 Dirichlet 闭合，不是平滑或拉普拉斯正则项。
    grid_i,grid_j=np.indices(points.shape[:2])
    for parity_i in (0,1):
        for parity_j in (0,1):
            submask=(unknown_mask&(grid_i%2==parity_i)&(grid_j%2==parity_j))
            coordinates=np.argwhere(submask)
            if coordinates.size==0:
                continue
            unique_i=np.unique(coordinates[:,0]); unique_j=np.unique(coordinates[:,1])
            added=False
            if unique_i.size>=2 and unique_j.size>=2:
                i0,i1=unique_i[:2]; j0,j1=unique_j[:2]
                corners=((i0,j0),(i0,j1),(i1,j0),(i1,j1))
                if all(unknown_index[i,j]>=0 for i,j in corners):
                    wi=np.asarray([i1,-i0],np.float64)/(i1-i0)
                    wj=np.asarray([j1,-j0],np.float64)/(j1-j0)
                    for ai,i in enumerate((i0,i1)):
                        for aj,j in enumerate((j0,j1)):
                            matrix_rows.append(row_number)
                            matrix_columns.append(int(unknown_index[i,j]))
                            matrix_values.append(float(wi[ai]*wj[aj]))
                    added=True
            if not added:
                i,j=coordinates[0]
                matrix_rows.append(row_number)
                matrix_columns.append(int(unknown_index[i,j]))
                matrix_values.append(1.)
            targets.append(0.)
            row_number+=1
    matrix=coo_matrix(
        (matrix_values,(matrix_rows,matrix_columns)),
        shape=(row_number,int(np.count_nonzero(unknown_mask)))).tocsr()
    target=np.asarray(targets,np.float64)
    maximum=(max(200,4*matrix.shape[1])
             if lsmr_max_iterations is None else int(lsmr_max_iterations))
    solution=lsmr(
        matrix,target,atol=float(lsmr_atol),
        btol=float(lsmr_btol),maxiter=maximum)
    displacement=np.zeros(points.shape[:2],np.float64)
    displacement[unknown_mask]=solution[0]
    normal=geometry["normal"]
    displacement_vectors=displacement[...,None]*normal
    observed,_=local_slopes_to_normals(slopes,normal,frame)
    output=points+displacement_vectors
    inferred=geometry["valid"] & ~photo
    inferred_geometry=_surface_geometry(output,valid)
    observed=np.where(inferred[...,None],inferred_geometry["normal"],observed)
    return LocalReconstructionResult(
        xyz_out=output.astype(np.float32),
        displacement=displacement.astype(np.float32),
        displacement_vectors=displacement_vectors.astype(np.float32),
        reference_normals=normal.astype(np.float32),
        slopes=slopes.astype(np.float32),
        observed_normals=observed.astype(np.float32),
        curvature_correction=curvature.astype(np.float32),
        confidence=confidence.astype(np.float32),valid_mask=valid,
        boundary_mask=valid&~unknown_mask,solver_istop=int(solution[1]),
        solver_iterations=int(solution[2]),solver_residual_norm=float(solution[3]),
        trusted_no_contact_mask=trusted_no_contact,
        photometric_valid_mask=photo,inferred_mask=inferred)


def _load_reconstruction_input(path: Path, expected_metadata=None):
    """Load an NPZ and validate its geometry, paired photometry and declarations."""
    with np.load(path, allow_pickle=False) as data:
        return load_reconstruction_arrays(data, expected_metadata)


def load_reconstruction_arrays(data, expected_metadata=None):
    """Read an array mapping using the same contract as an offline recon input."""
    import json
    from utils.surface_sampling import sample_linear_rgb_to_surface
    from utils.surface_coordinates import COORDINATE_KIND, FRAME_CONVENTION
    metadata=json.loads(str(data['metadata'])) if 'metadata' in data else {}
    transfer = str(data['rgb_transfer_function']) if 'rgb_transfer_function' in data else metadata.get('rgb_transfer_function')
    declarations=('coordinate_kind','coordinate_chart_id','frame_convention','residual_domain','color_space')
    actual={key: str(data[key]) if key in data else metadata.get(key) for key in declarations}
    new_path=actual['coordinate_kind']==COORDINATE_KIND
    if expected_metadata is not None:
        expected=dict(expected_metadata)
        if new_path or expected.get('coordinate_kind')==COORDINATE_KIND:
            for key in declarations:
                if actual[key] is None or expected.get(key) is None or actual[key]!=expected[key]:
                    raise ValueError(f'LUT/input declaration mismatch: {key}')
        if (new_path or expected.get('coordinate_kind') == COORDINATE_KIND
                or 'contact_offset_mm' in expected or 'contact_offset_mm' in data):
            actual_offset = data.get('contact_offset_mm', metadata.get('contact_offset_mm'))
            expected_offset = expected.get('contact_offset_mm')
            try:
                offsets = np.asarray([actual_offset, expected_offset], np.float64)
            except (TypeError, ValueError):
                raise ValueError('LUT/input requires contact_offset_mm; rerun calibration') from None
            if (offsets.shape != (2,) or not np.isfinite(offsets).all()
                    or not np.isclose(offsets[0], offsets[1], rtol=0, atol=1e-9)):
                raise ValueError('LUT/input contact_offset_mm mismatch; rerun calibration')
    if new_path and (actual['frame_convention']!=FRAME_CONVENTION
                     or actual['residual_domain']!='material_grid' or actual['color_space']!='linear_rgb'):
        raise ValueError('New coordinate input requires shared frames and material-grid linear RGB')
    xyz=np.asarray(data['xyz'],np.float64)
    shape=xyz.shape[:2]
    if xyz.ndim!=3 or xyz.shape[-1]!=3:
        raise ValueError('xyz must be rows x columns x 3')
    geometry=np.isfinite(xyz).all(axis=-1)
    for key in ('geometry_valid_mask','valid_mask'):
        if key in data:
            mask=np.asarray(data[key],bool)
            if mask.shape!=shape:
                raise ValueError(f'{key} shape mismatch')
            geometry &= mask
    if 'geometry_stage' in data and str(data['geometry_stage'])!='final':
        raise ValueError('Preview geometry cannot be used for local reconstruction')
    if new_path and ('color_residual_grid' in data or 'observed_color_grid' in data) and 'photometric_valid_mask' not in data:
        raise ValueError('Material-grid colors require photometric_valid_mask')
    if 'color_residual_grid' in data:
        colors=np.asarray(data['color_residual_grid'],np.float32)
        photo=np.isfinite(colors).all(axis=-1)
    elif 'observed_color_grid' in data and 'reference_color_grid' in data:
        colors,photo=signed_color_difference(data['observed_color_grid'],data['reference_color_grid'])
    elif 'residual_linear_rgb' in data:
        if new_path:
            raise ValueError('Precomputed residuals for reference_arclength must be material-grid color_residual_grid')
        sampled=sample_linear_rgb_to_surface(data['residual_linear_rgb'],data['uv'],
                    data['residual_valid'] if 'residual_valid' in data else None)
        colors,photo=sampled.values,sampled.valid
    else:
        if 'uv' not in data:
            raise ValueError('Image colors require current uv')
        if new_path:
            if 'reference_uv' not in data or 'reference_st' not in data or 'st' not in data:
                raise ValueError('Independent reference sampling requires reference_uv, reference_st and st')
            native_key='material_coordinates' if 'material_coordinates' in data else 'coordinates'
            reference_key='reference_material_coordinates' if 'reference_material_coordinates' in data else 'reference_coordinates'
            if (native_key not in data or reference_key not in data
                    or data[native_key].shape!=data[reference_key].shape
                    or not np.allclose(data[native_key],data[reference_key],rtol=0,atol=1e-6)
                    or data['st'].shape!=data['reference_st'].shape
                    or not np.allclose(data['st'],data['reference_st'],rtol=0,atol=1e-7)):
                raise ValueError('Reference/current material identities differ')
        ref_uv=data['reference_uv'] if 'reference_uv' in data else data['uv']

        def source_valid(prefix, image):
            valid=np.isfinite(image).all(axis=-1)
            def mask_field(key):
                mask=np.asarray(data[key],bool)
                if mask.shape!=valid.shape:
                    raise ValueError(f'{key} shape differs from source image')
                return mask
            # Explicit source masks can be provided for already-linear images.
            source_key=prefix+'source_valid_mask'
            surface_key='surface_mask' if prefix=='observed_' else 'reference_surface_mask'
            marker_key='marker_image_mask' if prefix=='observed_' else 'reference_marker_image_mask'
            if source_key in data:
                valid &= mask_field(source_key)
            elif new_path:
                if surface_key not in data or marker_key not in data:
                    raise ValueError(f'Image sampling requires {surface_key} and {marker_key}')
                valid &= mask_field(surface_key) & ~mask_field(marker_key)
            valid &= np.all(image<1,axis=-1)
            if prefix=='observed_' and 'residual_valid' in data:
                valid &= mask_field('residual_valid')
            return valid

        if any(key in data for key in (
                'observed_linear_rgb', 'reference_linear_rgb',
                'observed_bgr', 'reference_bgr')):
            if 'observed_bgr' in data or 'reference_bgr' in data:
                if transfer is not None:
                    require_camera_rgb_transfer({'rgb_transfer_function': transfer}, '输入图像')
                if expected_metadata is not None:
                    require_camera_rgb_transfer(expected_metadata, 'LUT')
                transfer = CAMERA_RGB_TRANSFER_FUNCTION
            observed = cached_linear_rgb(data, 'observed')
            reference = cached_linear_rgb(data, 'reference')
            current_valid=source_valid('observed_',observed)
            ref_valid=source_valid('reference_',reference)
        else:
            raise ValueError('Input needs material-grid colors/residuals or both observed/reference images')
        current=sample_linear_rgb_to_surface(observed,data['uv'],current_valid)
        background=sample_linear_rgb_to_surface(reference,ref_uv,ref_valid)
        colors,photo=signed_color_difference(current.values,background.values,current.valid & background.valid)
    if expected_metadata is not None and transfer != expected_metadata.get('rgb_transfer_function'):
        raise ValueError('LUT/input declaration mismatch: rgb_transfer_function')
    if colors.shape!=xyz.shape or photo.shape!=shape:
        raise ValueError('Material-grid colors and geometry shapes differ')
    if 'photometric_valid_mask' in data:
        mask=np.asarray(data['photometric_valid_mask'],bool)
        if mask.shape!=shape:
            raise ValueError('photometric_valid_mask shape mismatch')
        photo &= mask
    photo &= geometry
    trusted=np.asarray(data['trusted_no_contact_mask'],bool) if 'trusted_no_contact_mask' in data else None
    if trusted is not None:
        if trusted.shape!=shape:
            raise ValueError('trusted_no_contact_mask shape mismatch')
        trusted &= photo
    return xyz,colors,geometry,photo,trusted


def reconstruct_from_input(data, calibration, settings) -> LocalReconstructionResult:
    """Shared inference policy for image pipelines and preassembled NPZ inputs."""
    xyz, colors, valid, photo, trusted = load_reconstruction_arrays(data, calibration.metadata)
    result = reconstruct_local_surface(
        xyz, colors, calibration, valid_mask=valid, photometric_valid_mask=photo,
        zero_color_inner_radius=settings.zero_color_inner_radius,
        zero_color_outer_radius=settings.zero_color_outer_radius,
        zero_color_negative_inner_radius=settings.zero_color_negative_inner_radius,
        zero_color_negative_outer_radius=settings.zero_color_negative_outer_radius,
        trusted_no_contact_mask=trusted if settings.no_contact_constraints_enabled else None,
        trusted_no_contact_confidence=settings.trusted_no_contact_confidence,
        displacement_zero_lambda_per_mm2=(settings.displacement_zero_lambda_per_mm2
                                         if settings.no_contact_constraints_enabled else 0.),
        lsmr_atol=settings.lsmr_atol, lsmr_btol=settings.lsmr_btol,
        lsmr_max_iterations=settings.lsmr_max_iterations)
    if 'contact_offset_mm' in calibration.metadata:
        result = replace(result, contact_offset_mm=float(calibration.metadata['contact_offset_mm']))
    return result


def save_reconstruction_result(
    result: LocalReconstructionResult, output: str | Path, *,
    preview_output: str | Path | None = None, show: bool = False,
) -> Path:
    """Save a local-result NPZ and optionally export or display its scientific figure."""
    output = result.save(output)
    print(f"局部重建完成: {output}；unknown={int(np.count_nonzero(~result.boundary_mask & result.valid_mask))} "
          f"iterations={result.solver_iterations} residual={result.solver_residual_norm:.6g}")
    if preview_output is not None or show:
        from reconstruction.viewer import view_reconstruction
        view_reconstruction(output, image_output=preview_output, show=show)
    return output


def local_main() -> None:
    """独立处理外部整体重建和参考图模块导出的 NPZ。"""
    parser=argparse.ArgumentParser(description="离线 NPZ 局部接触形变重建")
    parser.add_argument(
        "--config",default=Path(__file__).resolve().parents[1]/"config.yaml",
        help="配置文件；默认读取仓库根目录的 config.yaml")
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",help="整体坐标、参考 RGB/色差和有效域 NPZ")
    source.add_argument(
        "--view",metavar="RESULT_NPZ",help="显示已有重建结果，不重新求解")
    parser.add_argument(
        "--calibration",help="覆盖 local_reconstruction.calibration_file")
    parser.add_argument(
        "--output",help="输出 NPZ；默认保存到 reconstruction.output_dir/pc")
    parser.add_argument("--lsmr-atol",type=float,help="覆盖 LSMR atol")
    parser.add_argument("--lsmr-btol",type=float,help="覆盖 LSMR btol")
    parser.add_argument(
        "--lsmr-max-iterations",type=int,help="覆盖 LSMR 最大迭代次数")
    parser.add_argument("--show",action="store_true",help="重建完成后显示结果窗口")
    parser.add_argument(
        "--preview-output",help="显式导出三维曲面及位移图；默认只保存结果 NPZ")
    args=parser.parse_args()
    if args.view is not None:
        if args.output is not None:
            parser.error("--view 模式请用 --preview-output 指定预览图片路径")
        from reconstruction.viewer import view_reconstruction
        view_reconstruction(args.view,image_output=args.preview_output,
                            show=args.show or args.preview_output is None)
        return
    settings=load_local_reconstruction_settings(
        args.config,input_override=args.input,
        calibration_override=args.calibration,output_override=args.output,
        lsmr_atol_override=args.lsmr_atol,lsmr_btol_override=args.lsmr_btol,
        lsmr_max_iterations_override=args.lsmr_max_iterations)
    calibration=NormalCalibration.load(settings.calibration_file)
    if 'contact_offset_mm' in calibration.metadata:
        configured_offset = load_contact_surface_offset_mm(args.config)
        if not np.isclose(float(calibration.metadata['contact_offset_mm']), configured_offset, rtol=0, atol=1e-9):
            raise ValueError('LUT contact_offset_mm differs from configuration; rerun calibration')
    with np.load(settings.input_file, allow_pickle=False) as data:
        result = reconstruct_from_input(data, calibration, settings)
    save_reconstruction_result(result, settings.output_file,
                               preview_output=args.preview_output, show=args.show)


def main() -> None:
    """离线局部重建与结果可视化入口。"""
    local_main()


if __name__=="__main__":
    main()
