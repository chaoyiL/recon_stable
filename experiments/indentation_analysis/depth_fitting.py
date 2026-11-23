"""Fixed-radius sphere fits on nested, deepest-first contact subsets."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from numbers import Real

import numpy as np

from .geometry import SphereFit, fit_sphere


@dataclass(frozen=True)
class DepthScanOptions:
    """Step and consecutive-RMSE stability requirements, in millimetres."""

    step_mm: float = .02
    min_fit_points: int = 100
    stable_steps: int = 5
    rmse_atol_mm: float = .001
    rmse_rtol: float = .01

    def __post_init__(self) -> None:
        for name, allow_zero in (('step_mm', False), ('rmse_atol_mm', True),
                                 ('rmse_rtol', True)):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
                raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
            object.__setattr__(self, name, float(value))
        for name, minimum in (('min_fit_points', 4), ('stable_steps', 1)):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < minimum):
                raise ValueError(f'{name} must be an integer >= {minimum}')
            object.__setattr__(self, name, int(value))


def depth_scan_options(config: dict) -> DepthScanOptions:
    """Read and validate ``indentation_analysis.depth_scan`` settings."""
    if not isinstance(config, dict):
        raise ValueError('config must be a mapping')
    analysis = config.get('indentation_analysis', {})
    if not isinstance(analysis, dict):
        raise ValueError('indentation_analysis must be a mapping')
    options = analysis.get('depth_scan', {})
    if not isinstance(options, dict):
        raise ValueError('indentation_analysis.depth_scan must be a mapping')
    unknown = set(options) - {item.name for item in fields(DepthScanOptions)}
    if unknown:
        raise ValueError(f'Unknown depth_scan options: {", ".join(sorted(map(str, unknown)))}')
    return DepthScanOptions(**options)


@dataclass(frozen=True)
class DepthSample:
    depth_mm: float
    minimum_indentation_mm: float
    sample_count: int
    fit: SphereFit | None
    error: str | None


@dataclass(frozen=True)
class DepthConvergence:
    """All visited levels and the selected sample, when convergence succeeds."""

    samples: list[DepthSample]
    sample: DepthSample | None
    status: str


def _scan_depths(points: np.ndarray, displacement: np.ndarray, baseline_normals: np.ndarray,
                 radius_mm: float, *, step_mm: float, min_fit_points: int):
    xyz = np.asarray(points, np.float64)
    signed = np.asarray(displacement, np.float64)
    normals = np.asarray(baseline_normals, np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0
            or signed.shape != (len(xyz),) or normals.shape != xyz.shape
            or not all(np.isfinite(value).all() for value in (xyz, signed, normals))):
        raise ValueError('Require finite Nx3 points/normals and N displacements, N >= 1')
    if np.any(signed >= 0):
        raise ValueError('Displacement must be strictly negative inside a contact')
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('Baseline normals must be nonzero')
    for name, value in (('radius_mm', radius_mm), ('step_mm', step_mm)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) \
                or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and strictly positive')
    if isinstance(min_fit_points, (bool, np.bool_)) \
            or not isinstance(min_fit_points, (int, np.integer)) or min_fit_points < 4:
        raise ValueError('min_fit_points must be an integer >= 4')

    normals = normals / lengths[:, None]
    deepest = float(-signed.min())
    offsets = signed - signed.min()
    span = float(offsets.max())
    levels = np.arange(0., span, step_mm)
    if len(levels) > 1 and np.isclose(levels[-1], span, rtol=0., atol=1e-12):
        levels = levels[:-1]
    levels = np.append(levels, span)
    for depth in levels:
        selected = offsets <= depth
        count = int(selected.sum())
        fit, error = None, None
        if count < min_fit_points:
            error = 'insufficient_points'
        else:
            side = normals[selected].mean(axis=0)
            try:
                fit = fit_sphere(xyz[selected], radius_mm, side_normal=side)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                error = str(exc)
        yield DepthSample(float(depth), deepest-float(depth), count, fit, error)


def scan_depths(points: np.ndarray, displacement: np.ndarray, baseline_normals: np.ndarray,
                radius_mm: float, *, step_mm: float = .02,
                min_fit_points: int = 20) -> list[DepthSample]:
    """Fit each subset whose signed displacement is at most ``min + depth``.

    Inputs are finite N×3 camera XYZ, N strictly negative displacements and
    N×3 nonzero oriented baseline normals. Depth and displacement use mm.
    Every level has an independent, equal-weight fixed-radius radial fit.
    The mean unit baseline normal selects the side for the sphere center.
    The last level always includes the shallowest candidate.
    """
    return list(_scan_depths(points, displacement, baseline_normals, radius_mm,
                             step_mm=step_mm, min_fit_points=min_fit_points))


class _ConvergenceTracker:
    def __init__(self, options: DepthScanOptions):
        if not isinstance(options, DepthScanOptions):
            raise ValueError('options must be a DepthScanOptions instance')
        self.options = options
        self.window: deque[DepthSample] = deque(maxlen=options.stable_steps + 1)
        self.previous: DepthSample | None = None

    def add(self, sample: DepthSample) -> bool:
        window, previous, options = self.window, self.previous, self.options
        if sample.fit is None or sample.sample_count < options.min_fit_points:
            window.clear()
            self.previous = None
            return False
        regular = (previous is not None and np.isclose(
            sample.depth_mm - previous.depth_mm, options.step_mm,
            rtol=1e-10, atol=np.finfo(float).eps * max(1., abs(sample.depth_mm))))
        if previous is None or not regular or sample.sample_count <= previous.sample_count:
            window.clear()
        window.append(sample)
        self.previous = sample
        if len(window) == options.stable_steps + 1:
            rms = np.array([item.fit.radial_rms_mm for item in window])
            tolerance = max(options.rmse_atol_mm, float(rms.mean()) * options.rmse_rtol)
            return float(rms.max() - rms.min()) <= tolerance
        return False


def find_convergence(samples: list[DepthSample], *,
                     options: DepthScanOptions) -> DepthSample | None:
    """Find the first stable level in an already computed depth scan."""
    tracker = _ConvergenceTracker(options)
    return next((sample for sample in samples if tracker.add(sample)), None)


def scan_until_converged(points: np.ndarray, displacement: np.ndarray,
                         baseline_normals: np.ndarray, radius_mm: float, *,
                         options: DepthScanOptions) -> DepthConvergence:
    """Stop at the first stable run of expanding, regularly spaced fit subsets.

    A window contains ``stable_steps + 1`` successful fits; each adjacent pair
    must add at least one point. Its RMSE range must be at most the larger of
    the absolute tolerance and the relative tolerance times the window mean.
    Failed fits and repeated subsets reset the run. A final fractional step is
    recorded but cannot extend a run of regular steps.
    """
    tracker = _ConvergenceTracker(options)
    samples: list[DepthSample] = []
    for sample in _scan_depths(points, displacement, baseline_normals, radius_mm,
                               step_mm=options.step_mm,
                               min_fit_points=options.min_fit_points):
        samples.append(sample)
        if tracker.add(sample):
            return DepthConvergence(samples, sample, 'converged')
    return DepthConvergence(samples, None, 'not_converged')
