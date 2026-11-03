"""CUDA curve context reusing the validated radiometry and execution engine."""
from __future__ import annotations

import copy
from .config import validate_config
from .renderer_gpu import AnalyticGpuContext, require_torch
from .curve_geometry_gpu import pack_curve_pose, curve_ground_truth_torch


class CurveGpuContext(AnalyticGpuContext):
    supports_curve = True
    pack_pose_parameters = staticmethod(pack_curve_pose)

    def __init__(self, config: dict, device: str | None = None, *,
                 allow_contact_tangent: bool = False):
        self.allow_contact_tangent = allow_contact_tangent
        validate_config(config, allow_curve=True, allow_contact_tangent=allow_contact_tangent)
        if config['surface'].get('bending', {}).get('radius_mm') is None:
            raise ValueError('CurveGpuContext requires a finite bending radius')
        super().__init__(config, device)

    def config_for_pose(self, pose=None, config=None):
        result = copy.deepcopy(config or self.base_config)
        for key in ('surface', 'camera', 'lights', 'render'):
            if result[key] != self.base_config[key]:
                raise ValueError(f'{key} changed after CurveGpuContext creation; create a new context')
        if result['indentation']['radius_mm'] != self.base_config['indentation']['radius_mm']:
            raise ValueError('Sphere radius changed; create a new CurveGpuContext')
        if pose is not None:
            if pose.center_uv is None:
                raise ValueError('Curved poses require center_uv')
            result['indentation']['center_uv'] = list(pose.center_uv)
            result['indentation']['depth_mm'] = float(pose.depth_mm)
        validate_config(result, allow_curve=True,
                        allow_contact_tangent=self.allow_contact_tangent)
        return result

    def camera_ground_truth(self, config: dict):
        pose = self.camera_origin.new_tensor(self.pack_pose_parameters(config))
        return curve_ground_truth_torch(self.camera_origin, self.rays, self.camera_forward,
                                        pose, config, self.torch)

    def _lighting_meta(self, cfg):
        return dict(width=float(cfg['surface']['size_mm'][0]),
                    height=float(cfg['surface']['size_mm'][1]),
                    radius=float(cfg['indentation']['radius_mm']),
                    curve_radius=float(cfg['surface']['bending']['radius_mm']),
                    shadows=cfg['render']['shadows'], unit_exponents=self.unit_light_exponents)


def render_curve_host(config: dict, device: str | None = None):
    torch = require_torch()
    with torch.inference_mode():
        context = CurveGpuContext(config, device)
        sensor, radiance, gt, _ = context.render_tensors(config=config)
        return (sensor.cpu().numpy(), radiance.cpu().numpy(),
                {key: value.cpu().numpy() for key, value in gt.items()})
