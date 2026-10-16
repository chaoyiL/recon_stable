"""黑框区域、十字标记和冻结参考弧长坐标的整体重建入口。"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from utils.config import ConfigError, load_config
from utils.reconstruction_interface import (
    MaterialCoordinateSystems,
    OverallReconstructionProvider,
    load_material_coordinate_specs,
)
from reconstruction.surface_reconstruction.run_reconstruction import SurfaceReconstructionProvider


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def reconstruct_material_coordinates(
    frame_bgr: np.ndarray,
    provider: OverallReconstructionProvider,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    surface_mask: np.ndarray | None = None,
) -> MaterialCoordinateSystems:
    """用十字标记整体重建生成三套材料坐标。"""
    specs = load_material_coordinate_specs(config_path)
    mask = None if surface_mask is None else np.asarray(surface_mask, np.bool_)
    result = provider.reconstruct(np.asarray(frame_bgr), specs, mask)
    result.validate_against(specs)
    return result


def build_reconstruction_provider(
    config_path: str | Path,
) -> SurfaceReconstructionProvider:
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    surface = config.get("get_surface")
    if not isinstance(surface, Mapping):
        raise ConfigError("缺少 get_surface 配置段")
    raw = surface.get("overall_reconstruction")
    if not isinstance(raw, Mapping):
        raise ConfigError("get_surface.overall_reconstruction 必须是字典")
    unknown = set(raw) - {"provider", "config_file", "row_direction"}
    if unknown:
        raise ConfigError(
            "get_surface.overall_reconstruction 包含未知字段: "
            f"{sorted(unknown)}"
        )
    if raw.get("provider") != "marker_bicubic":
        raise ConfigError(
            "get_surface.overall_reconstruction.provider 必须是 marker_bicubic"
        )
    config_file = raw.get("config_file")
    if not isinstance(config_file, str) or not config_file.strip():
        raise ConfigError(
            "get_surface.overall_reconstruction.config_file 必须是非空路径"
        )
    provider_config = Path(config_file).expanduser()
    if not provider_config.is_absolute():
        provider_config = path.parent / provider_config
    direction = raw.get("row_direction", "auto")
    if direction not in {"auto", "down", "up", "left", "right"}:
        raise ConfigError(
            "get_surface.overall_reconstruction.row_direction "
            "必须是 auto/down/up/left/right"
        )
    return SurfaceReconstructionProvider(provider_config, direction)


def save_material_coordinates(
    output_path: str | Path,
    systems: MaterialCoordinateSystems,
    surface_mask: np.ndarray,
    frame_bgr: np.ndarray,
    provider: SurfaceReconstructionProvider,
    *,
    surface_coefficients=None,
    geometry_stage="final",
    metadata=None,
) -> Path:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if geometry_stage not in {"final", "preview_reference"}:
        raise ValueError("geometry_stage must be final or preview_reference")
    observation = systems.observation
    if observation.st is not None:
        chart = provider.coordinate_chart
        if chart is None or any(getattr(systems, name).chart_id != chart.chart_id
                                for name in ("geometry", "lightfield", "observation")):
            raise ValueError("All output grids must share the provider coordinate chart")
    values: dict[str, np.ndarray] = {
        "format_version": np.asarray(2, np.int32),
        "overall_reconstruction": np.asarray("marker_bicubic"),
        "surface_mask": np.asarray(surface_mask, np.bool_),
        "observed_bgr": np.asarray(frame_bgr, np.uint8),
        # recon-local reads the observation grid through these aliases.
        "xyz": observation.xyz,
        "uv": observation.uv,
        "valid_mask": observation.valid_mask,
        "geometry_valid_mask": observation.valid_mask,
        "coordinates": observation.coordinates,
        "material_coordinates": observation.coordinates,
        "geometry_stage": np.asarray(geometry_stage),
        "image_shape": np.asarray(frame_bgr.shape[:2], np.int32),
        "camera_matrix": np.asarray(provider.session.camera, np.float64),
        "dist_coeffs": np.asarray(provider.session.distortion, np.float64),
        "template_xy_mm": np.asarray(provider.session.xy, np.float64),
    }
    if observation.st is not None:
        from utils.surface_coordinates import save_arclength_chart, FRAME_CONVENTION
        chart = provider.coordinate_chart
        chart_file = provider.coordinate_chart_file
        if chart_file is None:
            chart_file = output.with_name(output.stem+"_coordinates.npz")
            save_arclength_chart(chart_file, chart)
            provider.coordinate_chart_file = Path(chart_file).resolve()
        values.update(st=observation.st.astype(np.float32),
                      coordinate_kind=np.asarray(observation.coordinate_kind),
                      coordinate_chart_id=np.asarray(observation.chart_id),
                      coordinate_chart_file=np.asarray(os.path.relpath(chart_file, output.resolve().parent)),
                      frame_convention=np.asarray(FRAME_CONVENTION),
                      material_bounds_mm=np.asarray(chart.parameter_bounds, np.float64))
    for name in ("geometry", "lightfield", "observation"):
        coordinate_map = getattr(systems, name)
        values[f"{name}_xyz"] = coordinate_map.xyz
        values[f"{name}_uv"] = coordinate_map.uv
        values[f"{name}_coordinates"] = coordinate_map.coordinates
        if name != "geometry":
            values[f"{name}_valid_mask"] = coordinate_map.valid_mask
        else:
            values["geometry_grid_valid_mask"] = coordinate_map.valid_mask
        if coordinate_map.st is not None:
            values[f"{name}_st"] = coordinate_map.st.astype(np.float32)
    frame_result = provider.last_frame_result
    if frame_result is not None:
        tracking = frame_result["tracking"]
        from reconstruction.surface_reconstruction.markers import marker_reference_sizes, build_marker_image_mask
        sizes = marker_reference_sizes(frame_bgr, tracking["pixels"])
        values["reference_marker_sizes"] = sizes
        values["marker_image_mask"] = build_marker_image_mask(frame_bgr, tracking["pixels"], tracking["valid"], sizes)
        info = frame_result["info"]
        values.update(
            marker_pixels=np.asarray(tracking["pixels"], np.float32),
            marker_valid=np.asarray(tracking["valid"], np.bool_),
            marker_confidence=np.asarray(tracking["confidence"], np.float32),
            reconstruction_status=np.asarray(info["status"]),
            reprojection_rms_px=np.asarray(
                info.get("reprojection_rms_px", np.nan), np.float32
            ),
            metric_max_error=np.asarray(
                info.get("metric_max_error", np.nan), np.float32
            ),
        )
        if "marker_fit_valid" in frame_result:
            values["marker_fit_valid"] = np.asarray(frame_result["marker_fit_valid"], bool)
        values["fit_info"] = np.asarray(json.dumps(info, ensure_ascii=False))
    coefficients = surface_coefficients
    if coefficients is None and frame_result is not None:
        coefficients = frame_result.get("coefficients")
    if coefficients is not None:
        coefficients = np.asarray(coefficients, np.float64)
        if coefficients.shape != (16, 3) or not np.isfinite(coefficients).all():
            raise ValueError("surface_coefficients must be finite 16x3")
        values["surface_coefficients"] = coefficients
    if metadata:
        if set(metadata) & (set(values) | {"metadata", "surface_coefficients", "st"}):
            raise ValueError("metadata must not override geometry arrays or their semantics")
        values["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                values[key] = np.asarray(value)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="黑框有效域和十字标记双三次整体曲面重建"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="输出三套材料坐标 NPZ")
    parser.add_argument("--mask-output", help="可选：另存黑框内部二值 mask PNG")
    parser.add_argument("--coordinate-reference", help="复用已有参考弧长坐标图 NPZ")
    args = parser.parse_args()
    image = cv2.imread(str(Path(args.image).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"无法读取图片: {args.image}")
    provider = build_reconstruction_provider(args.config)
    if args.coordinate_reference:
        from utils.surface_coordinates import load_arclength_chart
        provider.bind_coordinate_chart(load_arclength_chart(args.coordinate_reference))
        provider.coordinate_chart_file = Path(args.coordinate_reference).expanduser().resolve()
    systems = reconstruct_material_coordinates(image, provider, args.config)
    mask = provider.last_frame_result["inner_region"].mask
    output = save_material_coordinates(args.output, systems, mask, image, provider,
                                       metadata={"source_image": str(Path(args.image).expanduser().resolve())})
    if args.mask_output:
        mask_output = Path(args.mask_output).expanduser()
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_output), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"无法保存 mask: {mask_output}")
        print(f"黑框区域 mask 已保存: {mask_output}")
    print(f"三套材料坐标已保存: {output}")


if __name__ == "__main__":
    main()
