"""Full-device analytic renderer for PyTorch CUDA tensors."""
from __future__ import annotations

import copy
import math
import warnings
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

from .geometry import camera_frame
from .surface_coordinates import (set_center_material_mm, normal_direction,
                                  analytic_parameter, require_flat)
from .light_distribution import analytic_light_samples
from .geometry_gpu import (camera_rays_torch, height_normal_torch,
                           intersect_surface_torch, project_world_torch,
                           pack_pose_parameters)


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('GPU rendering requires PyTorch CUDA; install the physical-gpu extra') from exc
    return torch


def _dtype(torch, name: str):
    return {'float32': torch.float32, 'float64': torch.float64}[name]


class AnalyticGpuContext:
    """Reusable CUDA constants for one camera, surface and lighting setup."""

    supports_curve = False
    pack_pose_parameters = staticmethod(pack_pose_parameters)

    def __init__(self, config: dict, device: str | None = None):
        if not self.supports_curve:
            require_flat(config, 'AnalyticGpuContext (planar calibration context)')
        torch = require_torch()
        self.torch = torch
        self.base_config = copy.deepcopy(config)
        self.device = torch.device(device or config.get('render', {}).get('device', 'cuda:0'))
        if self.device.type != 'cuda':
            raise ValueError('AnalyticGpuContext requires a CUDA device')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available to PyTorch')
        try:
            torch.empty(1, device=self.device).add_(1)
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            raise RuntimeError(f'CUDA device {self.device} cannot execute tensors') from exc
        render = config.get('render', {})
        self.geometry_dtype = _dtype(torch, render.get('geometry_dtype', 'float64'))
        self.radiometry_dtype = _dtype(torch, render.get('radiometry_dtype', 'float32'))
        self.pixel_chunk_size = int(render.get('pixel_chunk_size', 16384))
        self.light_chunk_size = int(render.get('light_chunk_size', 16))
        self.oom_retry_count = 0
        self.cuda_graphs = render.get('cuda_graphs', True)
        self.cuda_graph_capture_count = 0
        self.cuda_graph_replay_count = 0
        self.cuda_graph_fallback_reason = None
        self._shading_graph = None
        self._shading_graph_key = None
        self._render_count = 0
        self._compiled_workspace = None
        from .renderer_compiled import CompiledLighting
        factor_function = None
        if self.supports_curve:
            from .curve_renderer_kernels import curve_light_block_factor
            factor_function = curve_light_block_factor
        self.compiled_lighting = CompiledLighting(torch, render.get('compile_lighting', 'off'),
                                                  factor_function=factor_function)
        self.compile_warmup_seconds = 0.
        self.compile_prepare_seconds = 0.
        self.compile_graph_capture_seconds = 0.
        from .renderer_kernels import shade_light_block
        self._eager_light_block = shade_light_block
        if self.supports_curve:
            from .curve_renderer_kernels import shade_curve_light_block
            self._eager_light_block = shade_curve_light_block
        self.rays = camera_rays_torch(config, device=self.device,
                                      dtype=self.geometry_dtype, torch=torch)
        camera = config['camera']
        self.camera_origin = torch.as_tensor(camera['position_mm'], device=self.device,
                                             dtype=self.geometry_dtype)
        rotation = torch.as_tensor(camera_frame(camera), device=self.device,
                                   dtype=self.geometry_dtype)
        self.camera_forward = rotation[:, 2]
        self.cos4 = (self.rays @ self.camera_forward) ** 4
        self.response = torch.as_tensor(camera['response_rgb_3x3'], device=self.device,
                                        dtype=self.radiometry_dtype)
        points, normals, powers, exponents = [], [], [], []
        for strip in config['lights']:
            samples, normal, sample_powers = analytic_light_samples(strip, config)
            count = len(samples)
            points.append(samples)
            normals.append(np.broadcast_to(normal, (count, 3)))
            powers.append(sample_powers)
            exponents.append(np.full(count, strip['cosine_exponent'], float))
        self.light_points = torch.as_tensor(np.concatenate(points), device=self.device,
                                            dtype=self.geometry_dtype)
        self.light_normals = torch.as_tensor(np.concatenate(normals), device=self.device,
                                             dtype=self.geometry_dtype)
        self.light_powers = torch.as_tensor(np.concatenate(powers), device=self.device,
                                            dtype=self.radiometry_dtype)
        self.light_exponents = torch.as_tensor(np.concatenate(exponents), device=self.device,
                                               dtype=self.geometry_dtype)
        self.unit_light_exponents = bool(np.all(np.concatenate(exponents) == 1))
        self.light_normalization = (self.light_exponents + 1) / (2 * math.pi)
        self.material_grid_cache = {}

    def config_for_pose(self, pose=None, config=None):
        result = copy.deepcopy(config or self.base_config)
        if result['lights'] != self.base_config['lights']:
            raise ValueError('Lighting changed after AnalyticGpuContext creation; '
                             'create a new context for the new point powers/distribution')
        if pose is not None:
            set_center_material_mm(result, pose.center_xy_mm)
            result['indentation']['depth_mm'] = float(pose.depth_mm)
        return result

    def camera_ground_truth(self, config: dict):
        torch = self.torch
        t = intersect_surface_torch(self.camera_origin, self.rays, config, torch=torch)
        hit = torch.isfinite(t)
        safe_t = torch.where(hit, t, torch.zeros_like(t))
        positions = self.camera_origin + safe_t[..., None] * self.rays
        _, normals, _, _, _ = height_normal_torch(
            positions[..., 0], positions[..., 1], config, optical=True, torch=torch)
        hit &= torch.sum(normals * -self.rays, dim=-1) > 0
        depth = torch.sum((positions - self.camera_origin) * self.camera_forward, dim=-1)
        nan = torch.full_like(positions, float('nan'))
        return {'mask': hit,
                'position_mm': torch.where(hit[..., None], positions, nan),
                'normal_world': torch.where(hit[..., None], normals, nan),
                'depth_mm': torch.where(hit, depth, torch.full_like(depth, float('nan')))}

    def render_tensors(self, *, pose=None, config=None):
        """Render one frame, reducing chunks and retrying after a CUDA OOM."""
        torch = self.torch
        while True:
            try:
                with torch.inference_mode():
                    result = self._render_tensors_once(pose=pose, config=config)
                self._render_count += 1
                return result
            except torch.cuda.OutOfMemoryError:
                self._shading_graph = None
                self._shading_graph_key = None
                self._compiled_workspace = None
                if self.pixel_chunk_size <= 1 and self.light_chunk_size <= 1:
                    raise
                self.oom_retry_count += 1
                if self.pixel_chunk_size >= self.light_chunk_size \
                        and self.pixel_chunk_size > 1:
                    self.pixel_chunk_size = max(1, self.pixel_chunk_size // 2)
                else:
                    self.light_chunk_size = max(1, self.light_chunk_size // 2)
                torch.cuda.empty_cache()

    def _get_shading_graph(self, config):
        # Single-frame rendering avoids capture costs. Reused scan contexts opt in
        # on their second frame, after the reference/first frame has warmed CUDA.
        if not self.cuda_graphs or self._render_count == 0 \
                or self.cuda_graph_fallback_reason is not None:
            return None
        surface, indentation = config['surface'], config['indentation']
        pixels = min(self.pixel_chunk_size, self.rays.shape[0] * self.rays.shape[1])
        key = (pixels, self.light_chunk_size, tuple(sorted(self._lighting_meta(config).items())),
               self.compiled_lighting.enabled)
        if self._shading_graph_key != key:
            self._shading_graph = None
            from .renderer_cuda_graph import ShadingCudaGraph
            try:
                self._shading_graph = ShadingCudaGraph(self, config, pixels)
            except self.torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as exc:
                self.cuda_graph_fallback_reason = str(exc)
                warnings.warn(f'CUDA graph capture unavailable; using eager CUDA: {exc}',
                              RuntimeWarning, stacklevel=2)
                return None
            self._shading_graph_key = key
            self.cuda_graph_capture_count += 1
        self._shading_graph.set_pose(config)
        return self._shading_graph

    def compile_metrics(self):
        return dict(self.compiled_lighting.metrics(), warmup_seconds=self.compile_warmup_seconds,
                    prepare_seconds=self.compile_prepare_seconds,
                    graph_capture_seconds=self.compile_graph_capture_seconds)

    def _lighting_meta(self, cfg):
        return dict(width=float(cfg['surface']['size_mm'][0]),
                    height=float(cfg['surface']['size_mm'][1]),
                    radius=float(cfg['indentation']['radius_mm']),
                    sign=float(normal_direction(cfg)),
                    shadows=cfg['render']['shadows'], unit_exponents=self.unit_light_exponents)

    def _light_block_arguments(self, point, normal, shadow_origin, pose, start, stop):
        return (point, normal, shadow_origin, self.light_points[start:stop],
                self.light_normals[start:stop], self.light_powers[start:stop],
                self.light_exponents[start:stop], self.light_normalization[start:stop], pose)

    def _prepare_lighting(self, point, normal, cfg, pose):
        compiler = self.compiled_lighting
        if not compiler.enabled:
            return
        shadow_origin = point[:, None, :] + normal[:, None, :] * 1e-6 \
            if cfg['render']['shadows'] else None
        meta = self._lighting_meta(cfg)
        started = time.perf_counter()
        for start in range(0, len(self.light_points), self.light_chunk_size):
            stop = min(start + self.light_chunk_size, len(self.light_points))
            compiler.prepare(self._light_block_arguments(point, normal, shadow_origin, pose,
                                                          start, stop), meta)
        self.compile_prepare_seconds += time.perf_counter() - started

    def _shade_points(self, point, normal, cfg, *, pose_parameters=None):
        """Integrate all lights without host synchronization or dynamic shapes."""
        torch = self.torch
        accumulated = torch.zeros((len(point), 3), device=self.device,
                                  dtype=self.radiometry_dtype)
        shadow_origin = (point[:, None, :] + normal[:, None, :] * 1e-6
                         if cfg['render']['shadows'] else None)
        if pose_parameters is None:
            pose_parameters = torch.as_tensor(self.pack_pose_parameters(cfg),
                                               device=self.device, dtype=self.geometry_dtype)
        meta = self._lighting_meta(cfg)
        for light_start in range(0, len(self.light_points), self.light_chunk_size):
            light_stop = min(light_start + self.light_chunk_size, len(self.light_points))
            arguments = self._light_block_arguments(
                point, normal, shadow_origin, pose_parameters, light_start, light_stop)
            if self.compiled_lighting.enabled:
                factor, _ = self.compiled_lighting.function(*arguments, **meta)
                # Preserve ATen's RGB multiplication/reduction order. Compiling
                # this reduction moved one RGB8 quantization boundary in testing.
                contribution = torch.sum(
                    factor[..., None] * self.light_powers[None, light_start:light_stop, :], dim=1)
            else:
                contribution, _ = self._eager_light_block(*arguments, **meta)
            accumulated += contribution
        return accumulated

    def _render_tensors_once(self, *, pose=None, config=None):
        torch = self.torch
        cfg = self.config_for_pose(pose, config)
        if self._render_count > 0:
            self.compiled_lighting.initialize()
        gt = self.camera_ground_truth(cfg)
        mask = gt['mask']
        positions = gt['position_mm'][mask]
        normals = gt['normal_world'][mask]
        irradiance = torch.zeros((len(positions), 3), device=self.device,
                                 dtype=self.radiometry_dtype)
        pixel_step = self.pixel_chunk_size
        workspace = None
        if self.compiled_lighting.enabled and len(positions):
            from .renderer_cuda_graph import ShadingWorkspace
            count = min(pixel_step, self.rays.shape[0] * self.rays.shape[1])
            if self._compiled_workspace is None or len(self._compiled_workspace.positions) != count:
                self._compiled_workspace = ShadingWorkspace(self, cfg, count)
            workspace = self._compiled_workspace
            workspace.set_pose(cfg)
            workspace.load(positions[:count], normals[:count])
            self._prepare_lighting(workspace.positions, workspace.normals, cfg, workspace.pose)
        graph = self._get_shading_graph(cfg) if len(positions) else None
        for pixel_start in range(0, len(positions), pixel_step):
            pixel_stop = min(pixel_start + pixel_step, len(positions))
            point = positions[pixel_start:pixel_stop]
            normal = normals[pixel_start:pixel_stop]
            if graph is None:
                if self.compiled_lighting.enabled:
                    workspace.load(point, normal)
                    accumulated = self._shade_points(workspace.positions, workspace.normals, cfg,
                                                       pose_parameters=workspace.pose)[:len(point)]
                else:
                    accumulated = self._shade_points(point, normal, cfg)
            else:
                accumulated = graph.replay(point, normal)
                self.cuda_graph_replay_count += 1
            irradiance[pixel_start:pixel_stop] = accumulated

        reflectance = torch.as_tensor(cfg['surface']['reflectance_rgb'], device=self.device,
                                      dtype=self.radiometry_dtype)
        radiance = irradiance * reflectance / math.pi
        shape = (*mask.shape, 3)
        radiance_image = torch.zeros(shape, device=self.device, dtype=self.radiometry_dtype)
        radiance_image[mask] = radiance * 1e6
        camera = cfg['camera']
        area = (2 * analytic_parameter(cfg, 'camera', 'pinhole_half_width_mm')) ** 2
        sensor = (radiance_image * (area / float(camera['focal_length_mm']) ** 2)
                  * self.cos4.to(self.radiometry_dtype)[..., None])
        return sensor, radiance_image, gt, cfg

    def encode_linear_rgb(self, sensor: Tensor, config: dict) -> Tensor:
        """Return unclipped, unencoded exposure-scaled camera-linear RGB."""
        return sensor @ self.response.T * float(config['render']['exposure_scale'])

    def encode_display(self, sensor, config):
        """Make a clipped linear uint8 preview; calibration uses encode_linear_rgb."""
        torch = self.torch
        rgb = AnalyticGpuContext.encode_linear_rgb(self, sensor, config)
        return torch.round(255 * torch.clamp(rgb, 0, 1)).to(torch.uint8)

    def project_world(self, points, config):
        return project_world_torch(points, config, torch=self.torch)


def render_host(config: dict, device: str | None = None):
    torch = require_torch()
    with torch.inference_mode():
        context = AnalyticGpuContext(config, device)
        sensor, radiance, gt, _ = context.render_tensors(config=config)
        host_gt = {key: value.detach().cpu().numpy() for key, value in gt.items()}
        return (sensor.detach().cpu().numpy(), radiance.detach().cpu().numpy(), host_gt)
