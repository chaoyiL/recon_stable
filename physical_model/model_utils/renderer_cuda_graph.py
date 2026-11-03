"""Fixed-shape CUDA graph for repeated analytic lighting and shadow queries."""
from __future__ import annotations

import time

from .geometry_gpu import pack_pose_parameters


class ShadingWorkspace:
    """Fixed point/normal/pose buffers for compiled and graph execution."""
    def __init__(self, context, config, pixel_count):
        torch = context.torch
        self.torch = torch
        self.positions = torch.zeros((pixel_count, 3), device=context.device,
                                     dtype=context.geometry_dtype)
        self.normals = torch.zeros_like(self.positions)
        self.pack_pose = context.pack_pose_parameters
        self.pose = torch.empty(len(self.pack_pose(config)), device=context.device,
                                dtype=context.geometry_dtype)
        self.set_pose(config)
    def set_pose(self, config):
        self.pose.copy_(self.pose.new_tensor(self.pack_pose(config)))

    def load(self, positions, normals):
        count = len(positions)
        self.positions[:count].copy_(positions)
        self.normals[:count].copy_(normals)
        if count < len(self.normals):
            self.normals[count:].zero_()


class ShadingCudaGraph(ShadingWorkspace):
    """Callers copy outputs before the next replay overwrites graph storage."""

    def __init__(self, context, config, pixel_count):
        super().__init__(context, config, pixel_count)
        torch = context.torch
        context._prepare_lighting(self.positions, self.normals, config, self.pose)
        # Warm up on a side stream, as required by PyTorch's CUDA graph API.
        with torch.cuda.device(context.device):
            current = torch.cuda.current_stream(context.device)
            warmup = torch.cuda.Stream(device=context.device)
            warmup.wait_stream(current)
            started = time.perf_counter()
            with torch.cuda.stream(warmup):
                for _ in range(2):
                    context._shade_points(self.positions, self.normals, config,
                                          pose_parameters=self.pose)
            current.wait_stream(warmup)
            warmup.synchronize()
            context.compile_warmup_seconds += time.perf_counter() - started
            self.graph = torch.cuda.CUDAGraph()
            started = time.perf_counter()
            with torch.cuda.graph(self.graph):
                self.output = context._shade_points(
                    self.positions, self.normals, config, pose_parameters=self.pose)
            context.compile_graph_capture_seconds += time.perf_counter() - started

    def replay(self, positions, normals):
        count = len(positions)
        self.load(positions, normals)
        self.graph.replay()
        return self.output[:count]
