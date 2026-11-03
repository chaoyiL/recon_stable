"""Optional Inductor backend with explicit initialization and CUDA-only fallback."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import time
import warnings


class CompiledLighting:
    def __init__(self, torch, mode, factor_function=None, *, validator=None,
                 region='light_geometry_and_factor'):
        self.torch = torch
        self.factor_function = factor_function
        self.validator = validator
        self.region = region
        self.mode = mode
        self.enabled = False
        self.reason = None
        self.failure_stage = None
        self.prepared = set()
        self.first_call_seconds = 0.
        self.validation_seconds = 0.
        self.validation_status = 'not_run'
        self.versions = {'torch': torch.__version__, 'cuda': torch.version.cuda}
        self.options = {'triton.cudagraphs': False, 'use_fast_math': False,
                        'emulate_precision_casts': True,
                        'eager_numerics.division_rounding': True,
                        'eager_numerics.disable_ftz': True}
        self.function = None
        self.initialized = False
        if mode == 'required':
            self.initialize()

    def initialize(self):
        if self.mode == 'off' or self.initialized:
            return
        self.initialized = True
        torch = self.torch
        try:
            if importlib.util.find_spec('triton') is None:
                raise RuntimeError('Triton is unavailable; install the gpu-compile extra')
            for package in ('triton-windows', 'triton'):
                try:
                    self.versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
            cache = Path(__file__).resolve().parents[1] / '.uv-cache' / 'torch-compile' / 'renderer'
            os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(cache / 'inductor'))
            os.environ.setdefault('TRITON_CACHE_DIR', str(cache / 'triton'))
            missing = set(self.options) - set(torch._inductor.list_options())
            if missing:
                raise RuntimeError(f'Unsupported compiler options: {sorted(missing)}')
            if self.factor_function is None:
                from .renderer_kernels import light_block_factor
                self.factor_function = light_block_factor
            self.function = torch.compile(self.factor_function, backend='inductor',
                                          fullgraph=True, dynamic=False, options=self.options)
            self.enabled = True
        except Exception as exc:
            self.fail('backend_initialization', exc)

    def fail(self, stage, exc):
        if isinstance(exc, self.torch.cuda.OutOfMemoryError):
            raise exc
        self.failure_stage = stage
        self.reason = f'{type(exc).__name__}: {exc}'
        self.enabled = False
        self.function = None
        if self.mode == 'required':
            raise RuntimeError(f'Required lighting compilation failed at {stage}: {exc}') from exc
        warnings.warn(f'Lighting compilation unavailable; using existing CUDA path: {self.reason}',
                      RuntimeWarning, stacklevel=2)

    @staticmethod
    def key(args, meta):
        tensors = tuple((tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
                        if x is not None else None for x in args)
        return tensors, tuple(sorted(meta.items()))

    def prepare(self, args, meta):
        if not self.enabled:
            return
        key = self.key(args, meta)
        if key in self.prepared:
            return
        torch = self.torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Lighting kernel must compile before CUDA graph capture')
        try:
            print(f'lighting: compile P={args[0].shape[0]}, L={args[3].shape[0]}, '
                  f'{args[0].dtype}/{args[5].dtype}', flush=True)
            started = time.perf_counter()
            actual = self.function(*args, **meta)
            torch.cuda.synchronize(args[0].device)
            self.first_call_seconds += time.perf_counter() - started
        except Exception as exc:
            self.fail('first_call', exc)
            return
        try:
            started = time.perf_counter()
            expected = self.factor_function(*args, **meta)
            if self.validator is not None:
                self.validator(actual, expected, torch)
            else:
                tolerance = 3e-10 if actual[0].dtype == torch.float64 else 2e-6
                torch.testing.assert_close(actual[0], expected[0], rtol=tolerance, atol=1e-15)
                if not torch.equal(actual[1], expected[1]):
                    raise RuntimeError('Compiled lighting changed shadow visibility')
            self.validation_seconds += time.perf_counter() - started
            self.validation_status = 'initial_blocks_passed'
        except Exception as exc:
            self.validation_status = 'failed'
            self.fail('initial_block_validation', exc)
            return
        self.prepared.add(key)

    def metrics(self):
        return dict(requested=self.mode, effective=self.enabled and bool(self.prepared),
                    state=('waiting_for_reuse' if self.mode == 'auto' and not self.initialized
                           else 'fallback' if self.reason else 'compiled' if self.prepared
                           else 'ready' if self.enabled else 'disabled'),
                    backend='inductor' if self.enabled and self.prepared else 'eager_cuda',
                    region=self.region,
                    options=self.options, versions=self.versions,
                    variant_count=len(self.prepared), first_call_seconds=self.first_call_seconds,
                    validation_seconds=self.validation_seconds,
                    validation_status=self.validation_status,
                    fallback_reason=self.reason, failure_stage=self.failure_stage)
