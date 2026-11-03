"""Active-set slope and joint slope-depth Gauss--Newton solvers.

The depth method follows the upstream per-point GN3 equations: depth is scaled
in millimetres, eliminated with a scalar Schur complement, constrained to a
physical interval, and accepted only when the actual camera-linear objective
does not increase. Shadow visibility is frozen at each point's reference
position; light geometry and active source/receiver branches are refreshed at
every current state.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .photometry import DepthResponseField, _camera_post, _per_light_factors


def _beta_array(value: Any, count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape == (2,):
        result = np.broadcast_to(result, (count, 2)).copy()
    if result.shape != (count, 2) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape (N,2) or (2,)')
    return result.copy()


def _depth_array(value: Any, count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(count, float(result), dtype=np.float64)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite scalar or array with shape (N,)')
    return result.copy()


def _color_array(value: Any, count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count, 3) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape (N,3)')
    return result


def _color_weights(value: Any | None, count: int) -> np.ndarray:
    if value is None:
        return np.ones((count, 3), dtype=np.float64)
    weights = np.asarray(value, dtype=np.float64)
    if weights.ndim == 0:
        weights = np.full((count, 3), float(weights), dtype=np.float64)
    elif weights.shape == (3,):
        weights = np.broadcast_to(weights, (count, 3)).copy()
    if weights.shape != (count, 3) or not np.isfinite(weights).all() \
            or np.any(weights < 0.0):
        raise ValueError('color_weight must be a nonnegative scalar, (3,), or (N,3)')
    return weights


def _validate_solver_settings(*, steps: int, damping: float, prior_weight: float,
                              max_step: float) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    for name, value in (('damping', damping), ('prior_weight', prior_weight),
                        ('max_step', max_step)):
        if not math.isfinite(float(value)):
            raise ValueError(f'{name} must be finite')
    if damping < 0.0 or prior_weight < 0.0 or max_step <= 0.0:
        raise ValueError('damping/prior_weight must be nonnegative and max_step positive')


def _validate_depth_settings(*, depth_scale_mm: float, max_depth_mm: float,
                             max_depth_step_mm: float, depth_damping: float,
                             depth_effect_threshold: float, rank_rtol: float,
                             line_search_steps: int) -> None:
    for name, value in (
        ('depth_scale_mm', depth_scale_mm), ('max_depth_mm', max_depth_mm),
        ('max_depth_step_mm', max_depth_step_mm), ('depth_damping', depth_damping),
        ('depth_effect_threshold', depth_effect_threshold), ('rank_rtol', rank_rtol),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f'{name} must be finite')
    if depth_scale_mm <= 0.0 or max_depth_mm <= 0.0 or max_depth_step_mm <= 0.0:
        raise ValueError('depth scales and bounds must be positive')
    if depth_damping < 0.0 or depth_effect_threshold < 0.0 or rank_rtol < 0.0:
        raise ValueError('depth damping and thresholds must be nonnegative')
    if isinstance(line_search_steps, bool) or not isinstance(line_search_steps, int) \
            or line_search_steps < 0:
        raise ValueError('line_search_steps must be a nonnegative integer')


def _optical_normal(normal: np.ndarray, basis: np.ndarray,
                    beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(1.0 + np.sum(beta * beta, axis=-1))
    tangent = np.einsum('nci,ni->nc', basis, beta)
    optical = (-normal + tangent) / scale[:, None]
    derivative = basis / scale[:, None, None]
    derivative -= optical[:, :, None] * beta[:, None, :] / scale[:, None, None] ** 2
    return optical, derivative


def _matrix_from_components(source_factor: np.ndarray, emitter_active: np.ndarray,
                            visible: np.ndarray, direction: np.ndarray,
                            optical: np.ndarray, camera_post: np.ndarray,
                            camera_rgb: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    receiver_active = np.sum(direction * optical[:, None, :], axis=-1) > 0.0
    source_eligible = emitter_active & visible
    active = source_eligible & receiver_active
    gain = source_factor * active * camera_post[:, None]
    matrix = np.einsum('nl,lc,nlj->ncj', gain, camera_rgb, direction,
                       optimize=True)
    return matrix, active, source_eligible


def _position_response(field: DepthResponseField, beta: np.ndarray, depth: np.ndarray,
                       *, need_depth_derivative: bool
                       ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Build B(X(delta), beta) and its analytic fixed-branch depth derivative."""
    model = field.light_model
    count, light_count = len(beta), len(model.points)
    position = field.position - depth[:, None] * field.normal
    optical, _ = _optical_normal(field.normal, field.basis, beta)
    matrix = np.empty((count, 3, 3), dtype=np.float64)
    matrix_depth = np.empty_like(matrix) if need_depth_derivative else None
    active_all = np.empty((count, light_count), dtype=bool)
    eligible_all = np.empty_like(active_all)
    chunk = model.point_chunk_size

    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        rows = slice(start, stop)
        source_factor, emitter_active, visible, direction = _per_light_factors(
            position[rows], optical[rows], model,
            field.reference_visible[rows])
        # The supplied visibility is the exact reference-face mask, independent
        # of the current slope and depth. The helper returns it unchanged.
        post = _camera_post(position[rows], model)
        current_matrix, active, eligible = _matrix_from_components(
            source_factor, emitter_active, visible, direction, optical[rows], post,
            model.camera_rgb)
        matrix[rows] = current_matrix
        active_all[rows] = active
        eligible_all[rows] = eligible
        if not need_depth_derivative:
            continue

        point_position = position[rows]
        reference_normal = field.normal[rows]
        vector = model.points[None, :, :] - point_position[:, None, :]
        distance = np.linalg.norm(vector, axis=-1)
        safe_distance = np.maximum(distance, np.finfo(np.float64).tiny)
        direction_rate = (reference_normal[:, None, :] - direction * np.sum(
            direction * reference_normal[:, None, :], axis=-1, keepdims=True)) \
            / safe_distance[..., None]
        distance_rate = np.sum(direction * reference_normal[:, None, :], axis=-1)

        cosine_emit = np.maximum(-np.sum(
            direction * model.emission_normals[None, :, :], axis=-1), 0.0)
        cosine_emit_rate = -np.sum(
            direction_rate * model.emission_normals[None, :, :], axis=-1)
        exponent = model.exponents[None, :]
        safe_cosine = np.where(emitter_active, cosine_emit, 1.0)
        power = safe_cosine ** exponent
        power_rate = np.zeros_like(power)
        differentiable_power = emitter_active & (exponent > 0.0)
        if np.any(differentiable_power):
            power_rate[differentiable_power] = (
                exponent * safe_cosine ** (exponent - 1.0) * cosine_emit_rate
            )[differentiable_power]
        power_rate *= emitter_active

        distance2 = safe_distance * safe_distance
        distance2_rate = 2.0 * safe_distance * distance_rate
        source_rate = model.normalization[None, :] * (
            power_rate / distance2 - power * distance2_rate / (distance2 * distance2))
        source_rate *= emitter_active

        camera_ray = point_position - model.camera_position
        camera_distance = np.linalg.norm(camera_ray, axis=-1)
        safe_camera_distance = np.maximum(camera_distance, np.finfo(np.float64).tiny)
        camera_direction = camera_ray / safe_camera_distance[:, None]
        point_velocity = -reference_normal
        camera_direction_rate = (point_velocity - camera_direction * np.sum(
            camera_direction * point_velocity, axis=-1, keepdims=True)) \
            / safe_camera_distance[:, None]
        camera_cosine_raw = camera_direction @ model.camera_forward
        camera_cosine = np.maximum(camera_cosine_raw, 0.0)
        camera_cosine_rate = (camera_direction_rate @ model.camera_forward) \
            * (camera_cosine_raw > 0.0)
        post_rate = (4.0 * model.camera_aperture * camera_cosine ** 3
                     * camera_cosine_rate * model.camera_post_scale)

        visible_active = active
        gain = source_factor * visible_active * post[:, None]
        gain_rate = (source_rate * post[:, None]
                     + source_factor * post_rate[:, None]) * visible_active
        value = np.einsum(
            'nl,lc,nlj->ncj', gain_rate, model.camera_rgb, direction, optimize=True)
        direction_part = np.einsum(
            'nl,lc,nlj->ncj', gain, model.camera_rgb, direction_rate, optimize=True)
        matrix_depth[rows] = value + direction_part

    return matrix, matrix_depth, active_all, eligible_all, optical


def linearize_depth_response(field: DepthResponseField, beta: np.ndarray,
                             depth_mm: np.ndarray) -> tuple[
                                 np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray]:
    """Return response and slope/depth Jacobians at one candidate state."""
    beta_array = _beta_array(beta, len(field.position), 'beta')
    depth_array = _depth_array(depth_mm, len(field.position), 'depth_mm')
    matrix, matrix_depth, active, eligible, optical = _position_response(
        field, beta_array, depth_array, need_depth_derivative=True)
    optical, normal_derivative = _optical_normal(field.normal, field.basis, beta_array)
    camera_linear = np.einsum('nij,nj->ni', matrix, optical)
    prediction = camera_linear - field.background
    beta_jacobian = np.einsum('nij,njk->nik', matrix, normal_derivative)
    depth_jacobian = np.einsum('nij,nj->ni', matrix_depth, optical)
    return (prediction, camera_linear, beta_jacobian, depth_jacobian, matrix,
            active, eligible)


def _candidate_response(field: DepthResponseField, beta: np.ndarray,
                        depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix, _, active, eligible, optical = _position_response(
        field, beta, depth, need_depth_derivative=False)
    camera_linear = np.einsum('nij,nj->ni', matrix, optical)
    return camera_linear - field.background, active, eligible


def _objective(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray,
              beta: np.ndarray, prior: np.ndarray, prior_weight: float) -> np.ndarray:
    residual = prediction - target
    color = np.sum(weights * residual * residual, axis=-1)
    prior_cost = prior_weight * np.sum((beta - prior) ** 2, axis=-1)
    return color + prior_cost


def _solve_2x2(system: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    update = np.zeros_like(rhs)
    failed = np.zeros(len(rhs), dtype=bool)
    for row in range(len(rhs)):
        try:
            value = np.linalg.solve(system[row], rhs[row])
        except np.linalg.LinAlgError:
            failed[row] = True
            continue
        if not np.isfinite(value).all():
            failed[row] = True
            continue
        update[row] = value
    return update, failed


def _projected_schur_solve(y: np.ndarray, beta_jacobian: np.ndarray,
                           depth_jacobian: np.ndarray, prior_delta: np.ndarray, *,
                           prior_weight: float, damping: float, depth_damping: float,
                           depth_scale_mm: float, depth_effect_threshold: float,
                           rank_rtol: float, diagnostics: bool = True) -> dict[str, np.ndarray]:
    """Solve the whitened pointwise 2+1 slope/depth normal system."""
    count = len(y)
    identity = np.broadcast_to(np.eye(2), (count, 2, 2))
    hessian_beta = np.einsum('nci,ncj->nij', beta_jacobian, beta_jacobian)
    gradient_beta = np.einsum('nci,nc->ni', beta_jacobian, y)
    depth_column = depth_jacobian * depth_scale_mm
    beta_depth_cross = np.einsum('nci,nc->ni', beta_jacobian, depth_column)
    depth_hessian = np.sum(depth_column * depth_column, axis=-1)
    depth_gradient = np.sum(depth_column * y, axis=-1)
    weak_depth = np.sqrt(np.maximum(depth_hessian, 0.0)) <= depth_effect_threshold
    solve_denominator = np.maximum(depth_hessian + depth_damping, np.finfo(np.float64).tiny)
    solve_hessian = hessian_beta - (
        beta_depth_cross[:, :, None] * beta_depth_cross[:, None, :]
        / solve_denominator[:, None, None])
    solve_gradient = gradient_beta - beta_depth_cross * depth_gradient[:, None] \
        / solve_denominator[:, None]
    solve_hessian[weak_depth] = hessian_beta[weak_depth]
    solve_gradient[weak_depth] = gradient_beta[weak_depth]
    solve_hessian = 0.5 * (solve_hessian + np.swapaxes(solve_hessian, -1, -2))

    effective_damping = max(float(damping), np.finfo(np.float64).eps)
    prior_gradient = prior_weight * prior_delta
    system = solve_hessian + (prior_weight + effective_damping) * identity
    rhs = solve_gradient + prior_gradient
    delta_beta, solve_failed = _solve_2x2(system, rhs)
    residual_after_beta = y - np.einsum('nci,ni->nc', beta_jacobian, delta_beta)
    depth_step_normalized = np.sum(depth_column * residual_after_beta, axis=-1) \
        / solve_denominator
    depth_step_normalized[weak_depth] = 0.0
    depth_step_mm = depth_step_normalized * depth_scale_mm
    depth_step_mm[solve_failed] = 0.0
    result = {
        'delta_beta': delta_beta,
        'depth_step_mm': depth_step_mm,
        'depth_step_normalized': depth_step_normalized,
        'depth_effect_weak': weak_depth,
        'solve_failed': solve_failed,
        'fixed_depth_system': hessian_beta + (prior_weight + effective_damping) * identity,
        'fixed_depth_rhs': gradient_beta,
        'fixed_depth_cross': beta_depth_cross,
        'fixed_depth_prior_rhs': prior_gradient,
        'fixed_depth_column': depth_column,
        'fixed_depth_norm': depth_hessian,
    }
    if not diagnostics:
        return result

    undamped_denominator = np.maximum(depth_hessian, np.finfo(np.float64).tiny)
    observable_hessian = hessian_beta - (
        beta_depth_cross[:, :, None] * beta_depth_cross[:, None, :]
        / undamped_denominator[:, None, None])
    observable_hessian[weak_depth] = hessian_beta[weak_depth]
    observable_hessian = 0.5 * (observable_hessian + np.swapaxes(observable_hessian, -1, -2))
    eigenvalues = np.linalg.eigvalsh(observable_hessian)
    largest = np.maximum(eigenvalues[:, -1], 0.0)
    smallest = eigenvalues[:, 0]
    rank_threshold = largest * rank_rtol ** 2
    slope_rank = np.sum(eigenvalues > rank_threshold[:, None], axis=-1)
    condition = np.full(count, np.inf, dtype=np.float64)
    positive = smallest > 0.0
    condition[positive] = largest[positive] / np.maximum(
        smallest[positive], np.finfo(np.float64).tiny)
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    result.update({
        'depth_effect_norm': np.sqrt(np.maximum(depth_hessian, 0.0)),
        'slope_rank': slope_rank,
        'rank_deficient': slope_rank < 2,
        'slope_min_eigenvalue': smallest,
        'slope_condition': condition,
        'projected_jacobian_singular_values': singular_values,
        'projected_jacobian_min_singular_value': singular_values[:, 0],
        'projected_jacobian_condition': np.sqrt(condition),
        'projected_hessian': solve_hessian,
        'projected_gradient': solve_gradient,
        'observability_hessian': observable_hessian,
        'observability_gradient': np.where(
            weak_depth[:, None], gradient_beta,
            gradient_beta - beta_depth_cross * depth_gradient[:, None]
            / undamped_denominator[:, None]),
        'beta_depth_cross': beta_depth_cross,
        'depth_hessian': depth_hessian,
        'depth_gradient': depth_gradient,
    })
    return result


def _fixed_depth_beta_step(y: np.ndarray, beta_jacobian: np.ndarray,
                           depth_column: np.ndarray, depth_step_mm: np.ndarray,
                           *, prior_delta: np.ndarray, prior_weight: float,
                           depth_scale_mm: float, cache: dict[str, np.ndarray]
                           ) -> tuple[np.ndarray, np.ndarray]:
    normalized_depth_step = depth_step_mm / depth_scale_mm
    rhs = (cache['fixed_depth_rhs']
           - cache['fixed_depth_cross'] * normalized_depth_step[:, None]
           + cache['fixed_depth_prior_rhs'])
    update, failed = _solve_2x2(cache['fixed_depth_system'], rhs)
    return update, failed


def _limit_norm(value: np.ndarray, maximum: float) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(value, axis=-1)
    factor = np.minimum(maximum / np.maximum(norm, np.finfo(np.float64).tiny), 1.0)
    return value * factor[:, None], norm * factor


def _depth_objective(field: DepthResponseField, beta: np.ndarray, depth: np.ndarray,
                     target: np.ndarray, weights: np.ndarray, prior: np.ndarray,
                     prior_weight: float, *, diagnostics: bool = False
                     ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    prediction, active, eligible = _candidate_response(field, beta, depth)
    cost = _objective(prediction, target, weights, beta, prior, prior_weight)
    return cost, active if diagnostics else None, eligible if diagnostics else None


def predict_depth_gauss_newton(
        field: DepthResponseField, color_delta: Any, initial_beta: Any,
        initial_depth_mm: Any = 0.0, *, steps: int = 3, damping: float = 1e-4,
        prior_beta: Any | None = None, prior_weight: float = 0.0,
        max_step: float = 0.5, color_weight: Any | None = None,
        depth_scale_mm: float = 1.0, max_depth_mm: float = 5.0,
        max_depth_step_mm: float = 1.0, depth_damping: float = 0.0,
        depth_effect_threshold: float = 1e-8, rank_rtol: float = 1e-6,
        line_search_steps: int = 4, diagnostics: bool = True
        ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """Estimate each point's tangent slope and indentation depth jointly.

    Inputs are signed camera-linear RGB deltas, LUT slope initial values, and
    optional depth seeds. Ground-truth slopes and depths are never accepted by
    this function. The prior centre defaults to the supplied LUT slope.
    """
    if not isinstance(field, DepthResponseField):
        raise TypeError('field must be a DepthResponseField')
    _validate_solver_settings(steps=steps, damping=damping,
                              prior_weight=prior_weight, max_step=max_step)
    _validate_depth_settings(
        depth_scale_mm=depth_scale_mm, max_depth_mm=max_depth_mm,
        max_depth_step_mm=max_depth_step_mm, depth_damping=depth_damping,
        depth_effect_threshold=depth_effect_threshold, rank_rtol=rank_rtol,
        line_search_steps=line_search_steps)
    count = len(field.position)
    target = _color_array(color_delta, count, 'color_delta')
    beta = _beta_array(initial_beta, count, 'initial_beta')
    prior = beta.copy() if prior_beta is None else _beta_array(prior_beta, count, 'prior_beta')
    depth = _depth_array(initial_depth_mm, count, 'initial_depth_mm')
    if np.any(depth < 0.0) or np.any(depth > max_depth_mm):
        raise ValueError('initial_depth_mm must lie in [0, max_depth_mm]')
    weights = _color_weights(color_weight, count)
    sqrt_weights = np.sqrt(weights)

    prediction = linearize_depth_response(field, beta, depth)[0]
    objective = _objective(prediction, target, weights, beta, prior, prior_weight)
    last_beta_step = np.zeros(count, dtype=np.float64)
    last_depth_step = np.zeros(count, dtype=np.float64)
    solve_failed_any = np.zeros(count, dtype=bool)
    depth_weak_any = np.zeros(count, dtype=bool)
    rank_deficient_any = np.zeros(count, dtype=bool)
    line_search_failed_any = np.zeros(count, dtype=bool)
    accepted_steps = np.zeros(count, dtype=np.int32)
    backtrack_total = np.zeros(count, dtype=np.int32)
    depth_bound_hit_count = np.zeros(count, dtype=np.int32)
    depth_bound_active_any = np.zeros(count, dtype=bool)

    for iteration in range(steps):
        (prediction, _, beta_jacobian, depth_jacobian, _, active_current,
         eligible_current) = linearize_depth_response(field, beta, depth)
        residual = target - prediction
        whitened_residual = sqrt_weights * residual
        whitened_beta = sqrt_weights[:, :, None] * beta_jacobian
        whitened_depth = sqrt_weights * depth_jacobian
        prior_delta = prior - beta
        step_details = _projected_schur_solve(
            whitened_residual, whitened_beta, whitened_depth, prior_delta,
            prior_weight=prior_weight, damping=damping, depth_damping=depth_damping,
            depth_scale_mm=depth_scale_mm,
            depth_effect_threshold=depth_effect_threshold, rank_rtol=rank_rtol,
            diagnostics=diagnostics)
        proposed_beta = step_details['delta_beta']
        proposed_depth = step_details['depth_step_mm']
        solve_failed = step_details['solve_failed'].copy()

        lower = np.maximum(-depth, -max_depth_step_mm)
        upper = np.minimum(max_depth_mm - depth, max_depth_step_mm)
        bounded_depth = np.minimum(np.maximum(proposed_depth, lower), upper)
        depth_bound_active = np.abs(bounded_depth - proposed_depth) > (8.0 * np.finfo(float).eps)
        depth_column = step_details['fixed_depth_column']
        bounded_beta, bounded_failed = _fixed_depth_beta_step(
            whitened_residual, whitened_beta, depth_column, bounded_depth,
            prior_delta=prior_delta, prior_weight=prior_weight,
            depth_scale_mm=depth_scale_mm, cache=step_details)
        proposed_beta = np.where(depth_bound_active[:, None], bounded_beta, proposed_beta)
        solve_failed |= bounded_failed & depth_bound_active

        proposed_beta, beta_norm = _limit_norm(proposed_beta, max_step)
        depth_residual = whitened_residual - np.einsum('nci,ni->nc', whitened_beta, proposed_beta)
        depth_norm = step_details['fixed_depth_norm']
        depth_denominator = np.maximum(depth_norm + depth_damping, np.finfo(float).tiny)
        normalized_depth = np.sum(depth_column * depth_residual, axis=-1) / depth_denominator
        normalized_depth[step_details['depth_effect_weak']] = 0.0
        rebacksolved_depth = normalized_depth * depth_scale_mm
        rebacksolved_depth = np.minimum(np.maximum(rebacksolved_depth, lower), upper)
        post_bound_active = (~depth_bound_active) & (
            np.abs(rebacksolved_depth - normalized_depth * depth_scale_mm)
            > (8.0 * np.finfo(float).eps))
        post_bound_beta, post_bound_failed = _fixed_depth_beta_step(
            whitened_residual, whitened_beta, depth_column, rebacksolved_depth,
            prior_delta=prior_delta, prior_weight=prior_weight,
            depth_scale_mm=depth_scale_mm, cache=step_details)
        post_bound_beta, _ = _limit_norm(post_bound_beta, max_step)
        proposed_beta = np.where(post_bound_active[:, None], post_bound_beta, proposed_beta)
        solve_failed |= post_bound_active & post_bound_failed
        proposed_depth = np.where(depth_bound_active, bounded_depth, rebacksolved_depth)
        proposed_beta[solve_failed] = 0.0
        proposed_depth[solve_failed] = 0.0

        no_op = (np.linalg.norm(proposed_beta, axis=-1) <= np.finfo(float).eps) \
            & (np.abs(proposed_depth) <= np.finfo(float).eps)
        accepted = no_op.copy()
        accepted_beta = beta.copy()
        accepted_depth = depth.copy()
        accepted_objective = objective.copy()
        accepted_active = active_current.copy()
        accepted_eligible = eligible_current.copy()
        accepted_scale = np.zeros(count, dtype=np.float64)
        alpha = np.ones(count, dtype=np.float64)

        for _ in range(line_search_steps + 1):
            pending = np.flatnonzero(~accepted)
            if not len(pending):
                break
            candidate_beta = beta[pending] + alpha[pending, None] * proposed_beta[pending]
            candidate_depth = depth[pending] + alpha[pending] * proposed_depth[pending]
            candidate_cost, candidate_active, candidate_eligible = _depth_objective(
                field.take_rows(pending), candidate_beta, candidate_depth, target[pending],
                weights[pending], prior[pending], prior_weight, diagnostics=diagnostics)
            tolerance = 8.0 * np.finfo(float).eps * (1.0 + np.abs(objective[pending]))
            improves = candidate_cost <= objective[pending] + tolerance
            accepted_rows = pending[improves]
            if len(accepted_rows):
                accepted_beta[accepted_rows] = candidate_beta[improves]
                accepted_depth[accepted_rows] = candidate_depth[improves]
                accepted_objective[accepted_rows] = candidate_cost[improves]
                accepted_scale[accepted_rows] = alpha[accepted_rows]
                if diagnostics:
                    accepted_active[accepted_rows] = candidate_active[improves]
                    accepted_eligible[accepted_rows] = candidate_eligible[improves]
                accepted[accepted_rows] = True
            failed_rows = pending[~improves]
            if len(failed_rows):
                alpha[failed_rows] *= 0.5
                backtrack_total[failed_rows] += 1

        line_search_failed_any |= ~accepted
        accepted_delta_beta = accepted_beta - beta
        accepted_delta_depth = accepted_depth - depth
        last_beta_step = np.linalg.norm(accepted_delta_beta, axis=-1)
        last_depth_step = accepted_delta_depth
        if diagnostics:
            solve_failed_any |= solve_failed
            depth_weak_any |= step_details['depth_effect_weak']
            rank_deficient_any |= step_details['rank_deficient']
            active_switch = np.sum(accepted_active != active_current, axis=-1)
            eligible_switch = np.sum(accepted_eligible != eligible_current, axis=-1)
            if iteration == 0:
                active_switch_total = active_switch.astype(np.int32)
                eligibility_switch_total = eligible_switch.astype(np.int32)
            else:
                active_switch_total += active_switch.astype(np.int32)
                eligibility_switch_total += eligible_switch.astype(np.int32)
            hit_bound = depth_bound_active | post_bound_active
            depth_bound_hit_count += (accepted & hit_bound).astype(np.int32)
            depth_bound_active_any |= accepted & hit_bound
            accepted_steps += (accepted & ~no_op).astype(np.int32)
        beta, depth, objective = accepted_beta, accepted_depth, accepted_objective

    if not diagnostics:
        return (beta, depth), {'diagnostics_enabled': False}

    (prediction, camera_linear, beta_jacobian, depth_jacobian, _, _,
     _) = linearize_depth_response(field, beta, depth)
    final_y = sqrt_weights * (target - prediction)
    final_a = sqrt_weights[:, :, None] * beta_jacobian
    final_b = sqrt_weights * depth_jacobian
    final_diagnostics = _projected_schur_solve(
        final_y, final_a, final_b, prior - beta, prior_weight=prior_weight,
        damping=damping, depth_damping=depth_damping,
        depth_scale_mm=depth_scale_mm,
        depth_effect_threshold=depth_effect_threshold, rank_rtol=rank_rtol)
    boundary_tolerance = 8.0 * np.finfo(float).eps * max(1.0, max_depth_mm)
    diagnostics_result: dict[str, Any] = {
        'diagnostics_enabled': True,
        'residual_rgb': prediction - target,
        'residual_norm': np.linalg.norm(prediction - target, axis=-1),
        'camera_linear_rgb': camera_linear,
        'depth_mm': depth,
        'depth_step_mm': last_depth_step,
        'step_norm': last_beta_step,
        'accepted_steps': accepted_steps,
        'line_search_backtracks': backtrack_total,
        'line_search_failed': line_search_failed_any,
        'depth_bound_hit_count': depth_bound_hit_count,
        'depth_constraint_active': depth_bound_active_any,
        'depth_at_lower_bound': depth <= boundary_tolerance,
        'depth_at_upper_bound': depth >= max_depth_mm - boundary_tolerance,
        'solve_failed': solve_failed_any | final_diagnostics['solve_failed'],
        'depth_effect_weak': depth_weak_any | final_diagnostics['depth_effect_weak'],
        'depth_effect_norm': final_diagnostics['depth_effect_norm'],
        'slope_rank': final_diagnostics['slope_rank'],
        'rank_deficient': rank_deficient_any | final_diagnostics['rank_deficient'],
        'slope_min_eigenvalue': final_diagnostics['slope_min_eigenvalue'],
        'slope_condition': final_diagnostics['slope_condition'],
        'projected_jacobian_singular_values': final_diagnostics[
            'projected_jacobian_singular_values'],
        'projected_jacobian_min_singular_value': final_diagnostics[
            'projected_jacobian_min_singular_value'],
        'projected_jacobian_condition': final_diagnostics['projected_jacobian_condition'],
        'active_switch_count': active_switch_total,
        'active_switch_total': int(np.sum(active_switch_total)),
        'source_eligibility_switch_count': eligibility_switch_total,
        'source_eligibility_switch_total': int(np.sum(eligibility_switch_total)),
        'active_changed': active_switch_total > 0,
        'reference_shadow_visibility_frozen': True,
        'reference_shadow_visibility': 'frozen_at_X0',
        'depth_scale_mm': float(depth_scale_mm),
        'max_depth_mm': float(max_depth_mm),
        'max_depth_step_mm': float(max_depth_step_mm),
        'depth_damping': float(depth_damping),
        'damping': float(damping),
        'prior_weight': float(prior_weight),
    }
    return (beta, depth), diagnostics_result


def predict_active_gauss_newton(
        field: DepthResponseField, color_delta: Any, initial_beta: Any, *,
        steps: int = 3, damping: float = 1e-4, prior_beta: Any | None = None,
        prior_weight: float = 0.0, max_step: float = 0.5,
        color_weight: Any | None = None, diagnostics: bool = True
        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the upstream fixed-position active-set slope-only GN baseline."""
    if not isinstance(field, DepthResponseField):
        raise TypeError('field must be a DepthResponseField')
    _validate_solver_settings(steps=steps, damping=damping,
                              prior_weight=prior_weight, max_step=max_step)
    count = len(field.position)
    target = _color_array(color_delta, count, 'color_delta')
    beta = _beta_array(initial_beta, count, 'initial_beta')
    prior = beta.copy() if prior_beta is None else _beta_array(prior_beta, count, 'prior_beta')
    weights = _color_weights(color_weight, count)
    identity = np.broadcast_to(np.eye(2), (count, 2, 2))
    solve_failed_any = np.zeros(count, dtype=bool)
    step_norm = np.zeros(count, dtype=np.float64)

    for _ in range(steps):
        optical, normal_derivative = _optical_normal(field.normal, field.basis, beta)
        matrix, _, _, _, _ = _position_response(
            field, beta, np.zeros(count, dtype=np.float64), need_depth_derivative=False)
        prediction = np.einsum('nij,nj->ni', matrix, optical) - field.background
        jacobian = np.einsum('nij,njk->nik', matrix, normal_derivative)
        residual = prediction - target
        hessian = np.einsum('nci,nc,ncj->nij', jacobian, weights, jacobian)
        gradient = np.einsum('nci,nc,nc->ni', jacobian, weights, residual)
        if prior_weight:
            hessian += prior_weight * identity
            gradient += prior_weight * (beta - prior)
        effective_damping = max(float(damping), np.finfo(float).eps)
        hessian += effective_damping * identity
        update, failed = _solve_2x2(hessian, -gradient)
        update[failed] = 0.0
        solve_failed_any |= failed
        update, step_norm = _limit_norm(update, max_step)
        beta += update

    if not diagnostics:
        return beta, {'diagnostics_enabled': False}
    optical, normal_derivative = _optical_normal(field.normal, field.basis, beta)
    matrix, _, active, eligible, optical_from_response = _position_response(
        field, beta, np.zeros(count, dtype=np.float64), need_depth_derivative=False)
    prediction = np.einsum('nij,nj->ni', matrix, optical) - field.background
    result = {
        'diagnostics_enabled': True,
        'residual_rgb': prediction - target,
        'residual_norm': np.linalg.norm(prediction - target, axis=-1),
        'camera_linear_rgb': prediction + field.background,
        'active_changed': np.any(active != field.reference_active, axis=-1),
        'active_switch_count': np.sum(active != field.reference_active, axis=-1).astype(np.int32),
        'step_norm': step_norm,
        'solve_failed': solve_failed_any,
        'optical_normal': optical_from_response,
    }
    return beta, result


__all__ = [
    'linearize_depth_response',
    'predict_active_gauss_newton',
    'predict_depth_gauss_newton',
]
