"""Bounded PWM updates from measured linear RGB responses."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import lsq_linear


def _pwm_bounds(
    current: ArrayLike, max_pwm: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    current_array = np.asarray(current, dtype=np.float64)
    if current_array.ndim != 2 or current_array.shape[1] != 3 or not len(current_array):
        raise ValueError("current must have shape (N, 3) with N > 0")

    try:
        upper = np.broadcast_to(np.asarray(max_pwm, dtype=np.float64), current_array.shape)
    except ValueError as exc:
        raise ValueError("max_pwm must be broadcastable to the shape of current") from exc
    if not np.all(np.isfinite(upper)) or np.any(upper < 0) or np.any(upper != np.rint(upper)):
        raise ValueError("max_pwm must contain finite nonnegative integers")
    if (
        not np.all(np.isfinite(current_array))
        or np.any(current_array != np.rint(current_array))
        or np.any(current_array < 0)
        or np.any(current_array > upper)
    ):
        raise ValueError("current must contain integers between zero and max_pwm")
    return current_array, upper


def bounded_pwm(
    current: ArrayLike,
    baseline: ArrayLike,
    target: ArrayLike,
    jacobian: ArrayLike,
    max_pwm: ArrayLike,
    regularization: float = 0.00015,
) -> NDArray[np.int64]:
    """Propose an integer PWM update using a measured local linear model.

    ``current`` has shape (N, 3), in LED order and RGB channel order. ``baseline``
    and ``target`` are corresponding one-dimensional linear RGB feature vectors.
    ``jacobian`` has shape (M, 3*N), with each column giving feature change per
    PWM count for a C-order flattened channel. ``max_pwm`` is an integer scalar
    or an array broadcastable to (N, 3).

    The continuous update minimizes ``||baseline + jacobian @ delta - target||²
    + regularization * ||delta||²`` with ``0 <= current + delta <= max_pwm``.
    The returned (N, 3) PWM values are rounded to integers.
    """
    current_array, upper = _pwm_bounds(current, max_pwm)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    jacobian_array = np.asarray(jacobian, dtype=np.float64)
    if baseline_array.ndim != 1 or not baseline_array.size or target_array.shape != baseline_array.shape:
        raise ValueError("baseline and target must be nonempty vectors of equal length")
    if jacobian_array.shape != (baseline_array.size, current_array.size):
        raise ValueError("jacobian must have shape (len(baseline), current.size)")
    if not all(np.all(np.isfinite(a)) for a in (baseline_array, target_array, jacobian_array)):
        raise ValueError("baseline, target and jacobian must contain finite values")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")

    flat_current = current_array.ravel()
    flat_upper = upper.ravel()
    free = flat_upper > 0
    if not np.any(free):
        return current_array.astype(np.int64)

    matrix = jacobian_array[:, free]
    residual = target_array - baseline_array
    if regularization > 0:
        matrix = np.vstack((matrix, np.sqrt(regularization) * np.eye(free.sum())))
        residual = np.concatenate((residual, np.zeros(free.sum())))

    result = lsq_linear(
        matrix,
        residual,
        bounds=(-flat_current[free], flat_upper[free] - flat_current[free]),
        method="bvls",
    )
    if not result.success:
        raise RuntimeError(f"PWM least-squares optimization failed: {result.message}")
    proposal = flat_current.copy()
    proposal[free] += result.x
    return np.rint(proposal).astype(np.int64).reshape(current_array.shape)


def propose_pwm(
    current: ArrayLike,
    baseline: ArrayLike,
    samples: ArrayLike,
    target: ArrayLike,
    probe_steps: ArrayLike,
    max_pwm: ArrayLike,
    regularization: float = 0.00015,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Estimate the response matrix from single-channel probes and propose PWM.

    Each row of ``samples`` is a linear RGB feature vector measured with one
    channel changed by the corresponding nonzero integer in ``probe_steps``.
    Rows follow the C-order flattening of ``current``, so samples has shape
    (3*N, M) and probe_steps has shape (3*N,). A negative step is valid when
    probing downward. Both the baseline and every probe use the same features.
    Returns the proposed (N, 3) integer PWM array and the (M, 3*N) Jacobian.
    """
    current_array, upper = _pwm_bounds(current, max_pwm)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    samples_array = np.asarray(samples, dtype=np.float64)
    steps = np.asarray(probe_steps, dtype=np.float64)
    if baseline_array.ndim != 1 or not baseline_array.size:
        raise ValueError("baseline must be a nonempty vector")
    if samples_array.shape != (current_array.size, baseline_array.size):
        raise ValueError("samples must have shape (current.size, len(baseline))")
    if steps.shape != (current_array.size,):
        raise ValueError("probe_steps must have shape (current.size,)")
    if (
        not np.all(np.isfinite(steps))
        or np.any(steps == 0)
        or np.any(steps != np.rint(steps))
        or np.any(current_array.ravel() + steps < 0)
        or np.any(current_array.ravel() + steps > upper.ravel())
    ):
        raise ValueError("probe_steps must be nonzero integers that keep PWM within bounds")
    if not np.all(np.isfinite(samples_array)) or not np.all(np.isfinite(baseline_array)):
        raise ValueError("baseline and samples must contain finite values")

    jacobian = ((samples_array - baseline_array) / steps[:, None]).T
    proposal = bounded_pwm(current_array, baseline_array, target, jacobian, upper, regularization)
    return proposal, jacobian
