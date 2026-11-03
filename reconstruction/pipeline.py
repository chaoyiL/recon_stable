"""单张图片的 marker 整体拟合、参考颜色配对和局部重建。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.color_difference import CAMERA_RGB_TRANSFER_FUNCTION
from utils.image_io import (
    load_linear_image, linear_rgb_to_tracking_bgr8, validate_linear_rgb,
)
from reconstruction.get_surface import build_reconstruction_provider
from reconstruction.local_recon import (
    LocalReconstructionResult,
    default_reconstruction_output,
    load_local_reconstruction_settings,
    reconstruct_from_input,
)
from reconstruction.surface_reconstruction.markers import build_marker_image_mask, marker_reference_sizes
from reconstruction.surface_reconstruction.surface import BicubicSurfaceAdapter
from utils.config import load_config
from utils.normal_mapping import NormalCalibration
from utils.reconstruction_interface import MaterialCoordinateMap, load_material_coordinate_specs
from utils.surface_coordinates import (
    COORDINATE_KIND, FRAME_CONVENTION, load_arclength_chart, sample_surface_grid,
)
from utils.surface_projection import CameraModel, project_points
from utils.surface_sampling import bilinear_sample, sample_mask_to_surface
from utils.surface_geometry import surface_grid_curve_lengths


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config.yaml'


@dataclass(frozen=True)
class MaterialSamplingGrid:
    """同一个 LUT 下固定的原生材料参数 (mm) 和规则 st 网格。"""

    parameters: np.ndarray
    st: np.ndarray
    chart_id: str


def load_sampling_grid(
    path: str | Path, calibration: NormalCalibration, surface: BicubicSurfaceAdapter,
    camera: CameraModel, shape: tuple[int, int],
) -> MaterialSamplingGrid:
    """Accept a complete arc-length chart or a saved inverse grid from that chart.

    A saved inverse grid is interpolated at regular target st coordinates.
    The chart identity is retained; XYZ, UV and colors are evaluated anew.
    """
    metadata = calibration.metadata
    for key, expected in dict(coordinate_kind=COORDINATE_KIND, frame_convention=FRAME_CONVENTION,
                              residual_domain='material_grid', color_space='linear_rgb').items():
        if metadata.get(key) != expected:
            raise ValueError(f'LUT requires {key}={expected}')
    with np.load(path, allow_pickle=False) as data:
        if 'st_table' in data:
            chart = load_arclength_chart(path)
            chart_id = chart.chart_id
            if chart_id != metadata.get('coordinate_chart_id'):
                raise ValueError('LUT/material-grid coordinate_chart_id mismatch')
            grid = sample_surface_grid(surface, chart, *shape)
            return MaterialSamplingGrid(grid.parameters, grid.st, chart_id)
        if (int(data['format_version']) != 1
                or str(data['grid_kind']) != 'frozen_material_grid'):
            raise ValueError('Unsupported frozen material-grid format')
        for key in ('coordinate_kind', 'coordinate_chart_id', 'frame_convention', 'residual_domain', 'color_space'):
            if str(data[key]) != metadata.get(key):
                raise ValueError(f'LUT/material-grid {key} mismatch')
        for key, expected in dict(template_xy_mm=surface.model.xy, camera_matrix=camera.matrix,
                                  dist_coeffs=camera.distortion, image_shape=camera.image_shape).items():
            if not np.array_equal(data[key], expected):
                raise ValueError(f'Material-grid {key} differs from the overall reconstruction configuration')
        parameters = np.asarray(data['material_coordinates'], np.float64)
        st = np.asarray(data['st'], np.float64)
        chart_id = str(data['coordinate_chart_id'])
    if (parameters.ndim != 3 or parameters.shape[-1] != 2
            or min(parameters.shape[:2]) < 2 or st.shape != parameters.shape):
        raise ValueError('Invalid frozen material-grid shape')
    source_shape = parameters.shape[:2]
    lo = np.asarray(surface.parameter_bounds)[[0, 2]]
    hi = np.asarray(surface.parameter_bounds)[[1, 3]]
    ss, tt = np.meshgrid(np.linspace(0, 1, source_shape[1]), np.linspace(0, 1, source_shape[0]))
    if (not np.isfinite(parameters).all() or not np.isfinite(st).all()
            or np.any(parameters < lo) or np.any(parameters > hi)
            or not np.allclose(st, np.stack((ss, tt), axis=-1), rtol=0, atol=1e-7)):
        raise ValueError('Invalid native material parameters or regular st grid')
    if source_shape != shape:
        ss, tt = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))
        st = np.stack((ss, tt), axis=-1)
        query = st * [source_shape[1]-1, source_shape[0]-1]
        sampled = bilinear_sample(parameters, query, allow_boundary=True)
        if not sampled.valid.all():
            raise ValueError('Frozen material-grid resampling failed')
        parameters = sampled.values
    return MaterialSamplingGrid(parameters, st, chart_id)


def sample_material_map(
    surface: BicubicSurfaceAdapter, camera: CameraModel, grid: MaterialSamplingGrid,
    surface_mask: np.ndarray,
) -> MaterialCoordinateMap:
    """Evaluate saved material identities on this frame, then project to its image."""
    shape = grid.parameters.shape[:2]
    xyz = surface.evaluate(grid.parameters.reshape(-1, 2)).reshape(*shape, 3)
    uv, valid = project_points(xyz, camera)
    uv = uv.astype(np.float32)
    valid &= sample_mask_to_surface(surface_mask, uv)
    return MaterialCoordinateMap(xyz, uv, grid.parameters, valid, grid.st,
                                 COORDINATE_KIND, grid.chart_id)


def make_reconstruction_input(
    current_map: MaterialCoordinateMap, reference_map: MaterialCoordinateMap,
    image: np.ndarray, reference_image: np.ndarray,
    surface_mask: np.ndarray, reference_mask: np.ndarray,
    marker_mask: np.ndarray, reference_marker_mask: np.ndarray,
) -> dict[str, np.ndarray | str]:
    """Assemble the existing local solver contract in memory, including both UV maps."""
    if (current_map.chart_id != reference_map.chart_id
            or current_map.coordinate_kind != reference_map.coordinate_kind):
        raise ValueError('Reference/current material-grid identities differ')
    image = validate_linear_rgb(image)
    reference_image = validate_linear_rgb(reference_image)
    return dict(
        xyz=current_map.xyz, uv=current_map.uv,
        material_coordinates=current_map.coordinates, st=current_map.st,
        valid_mask=current_map.valid_mask, geometry_valid_mask=current_map.valid_mask,
        geometry_stage='final', observed_linear_rgb=image,
        reference_linear_rgb=reference_image,
        reference_uv=reference_map.uv, reference_material_coordinates=reference_map.coordinates,
        reference_st=reference_map.st,
        surface_mask=surface_mask, reference_surface_mask=reference_mask,
        marker_image_mask=marker_mask, reference_marker_image_mask=reference_marker_mask,
        photometric_valid_mask=current_map.valid_mask & reference_map.valid_mask,
        coordinate_kind=COORDINATE_KIND, coordinate_chart_id=current_map.chart_id,
        frame_convention=FRAME_CONVENTION, residual_domain='material_grid', color_space='linear_rgb',
        rgb_transfer_function=CAMERA_RGB_TRANSFER_FUNCTION)


def _fit_frame(session, image, label):
    fitted = session.process(image)
    if (fitted['coefficients'] is None or fitted['inner_region'] is None
            or fitted['error'] or fitted['info']['status'] == 'ambiguous'):
        raise ValueError(f"{label}整体重建失败: {fitted['error'] or fitted['info']['status']}")
    return fitted


def _configured_path(value, base, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'reconstruction.{name} 必须是非空路径')
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def reconstruct_image(
    image_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    reference_image: str | None = None,
    material_grid_file: str | None = None,
    calibration_file: str | None = None,
    output: str | None = None,
    lsmr_atol: float | None = None,
    lsmr_btol: float | None = None,
    lsmr_max_iterations: int | None = None,
) -> tuple[LocalReconstructionResult, Path]:
    """Fit the current marker surface and solve its local normal displacement.

    The reference initializes fixed marker IDs and supplies background colors.
    The displacement baseline remains the current fitted surface, in mm.
    No calibration model is trained or modified by this workflow.
    """
    config_path = Path(config_path).expanduser().resolve()
    image_path = Path(image_path).expanduser().resolve()
    config = load_config(config_path)
    raw = config.get('reconstruction')
    if not isinstance(raw, dict):
        raise ValueError('config.yaml 缺少 reconstruction 配置段')
    unknown = set(raw) - {'reference_image', 'material_grid_file', 'output_dir', 'match_surface_aspect'}
    if unknown:
        raise ValueError(f'Unknown reconstruction settings: {sorted(unknown)}')
    match_aspect = raw.get('match_surface_aspect', True)
    if not isinstance(match_aspect, bool):
        raise ValueError('reconstruction.match_surface_aspect 必须是布尔值')
    reference_path = (Path(reference_image).expanduser().resolve() if reference_image is not None
                      else _configured_path(raw.get('reference_image'), config_path.parent, 'reference_image'))
    grid_path = (Path(material_grid_file).expanduser().resolve() if material_grid_file is not None
                 else _configured_path(raw.get('material_grid_file'), config_path.parent, 'material_grid_file'))
    output_path = (Path(output).expanduser() if output is not None else
                   default_reconstruction_output(
                       config_path, raw.get('output_dir'), image_path.stem + '_local_result.npz'))
    if output_path.suffix.lower() != '.npz':
        raise ValueError('重建输出必须使用 .npz 扩展名')
    if output_path.resolve() in {image_path, reference_path.resolve(), grid_path.resolve()}:
        raise ValueError('重建输出不能覆盖图片或材料网格')
    settings = load_local_reconstruction_settings(
        config_path, input_override=str(image_path), output_override=str(output_path),
        calibration_override=calibration_file, lsmr_atol_override=lsmr_atol,
        lsmr_btol_override=lsmr_btol, lsmr_max_iterations_override=lsmr_max_iterations)
    if output_path.resolve() == settings.calibration_file.resolve():
        raise ValueError('重建输出不能覆盖 LUT')
    calibration = NormalCalibration.load(settings.calibration_file)
    image, reference = load_linear_image(image_path), load_linear_image(reference_path)
    provider = build_reconstruction_provider(config_path)
    session = provider.session
    reference_fit = _fit_frame(session, reference, '参考图')
    reference_surface = BicubicSurfaceAdapter(session.model, reference_fit['coefficients'])
    camera = CameraModel(session.camera, session.distortion, reference.shape[:2])
    spec = load_material_coordinate_specs(config_path).observation
    grid = load_sampling_grid(grid_path, calibration, reference_surface, camera, (spec.rows, spec.columns))
    reference_mask = reference_fit['inner_region'].mask
    reference_tracking_bgr = linear_rgb_to_tracking_bgr8(reference)
    sizes = marker_reference_sizes(
        reference, reference_fit['tracking']['pixels'],
        tracking_bgr=reference_tracking_bgr)
    reference_marker_mask = build_marker_image_mask(reference, reference_fit['tracking']['pixels'],
                                                   reference_fit['tracking']['valid'], sizes,
                                                   tracking_bgr=reference_tracking_bgr)
    # Keep the reference's marker IDs while fitting an independent current shape.
    session.model.reset()
    current_fit = _fit_frame(session, image, '当前图')
    current_surface = BicubicSurfaceAdapter(session.model, current_fit['coefficients'])
    current_mask = current_fit['inner_region'].mask
    if match_aspect:
        pilot_map = sample_material_map(current_surface, camera, grid, current_mask)
        width, height = surface_grid_curve_lengths(pilot_map.xyz, pilot_map.valid_mask)
        long_edge_points = max(3, spec.rows, spec.columns)
        if height >= width:
            shape = (long_edge_points, max(3, 1+round((long_edge_points-1)*width/height)))
        else:
            shape = (max(3, 1+round((long_edge_points-1)*height/width)), long_edge_points)
        grid = load_sampling_grid(grid_path, calibration, current_surface, camera, shape)
    reference_map = sample_material_map(reference_surface, camera, grid, reference_mask)
    current_map = sample_material_map(current_surface, camera, grid, current_mask)
    current_tracking_bgr = linear_rgb_to_tracking_bgr8(image)
    marker_mask = build_marker_image_mask(
        image, current_fit['tracking']['pixels'], current_fit['tracking']['valid'],
        sizes, tracking_bgr=current_tracking_bgr)
    data = make_reconstruction_input(current_map, reference_map, image, reference,
                                     current_mask, reference_mask, marker_mask, reference_marker_mask)
    print(f"整体重建完成: marker={int(current_fit['tracking']['valid'].sum())} "
          f"reprojection_rms_px={current_fit['info']['reprojection_rms_px']:.6g} "
          f"solver_grid={current_map.xyz.shape[0]}x{current_map.xyz.shape[1]}")
    return reconstruct_from_input(data, calibration, settings), output_path
