"""Metrics for one-value RGB-bin compatibility of archived calibration samples."""
from __future__ import annotations

from typing import Any

import numpy as np


_ANGLE_GROUP_TOLERANCE = 1e-13
_COVERAGE_UPDATE_TOLERANCE = 1e-14


def bin_samples(colors: np.ndarray, lut: Any) -> dict[str, np.ndarray]:
    """Assign each signed linear RGB residual to its nearest LUT node.

    ``colors`` is an ``[N, 3]`` float array in RGB order. The returned flattened
    ids use ``(r * S + g) * S + b`` and ``out_of_range`` marks rows outside the
    LUT's unclipped color range before edge clamping.
    """
    color = np.asarray(colors, dtype=np.float64)
    if color.ndim != 2 or color.shape[1] != 3:
        raise ValueError("colors must have shape [N, 3]")
    if not np.isfinite(color).all():
        raise ValueError("colors must contain only finite values")

    size = int(lut.size)
    color_min = np.asarray(lut.color_min, dtype=np.float64)
    color_max = np.asarray(lut.color_max, dtype=np.float64)
    if size < 2 or color_min.shape != (3,) or color_max.shape != (3,):
        raise ValueError("lut must provide a cubic RGB range with size >= 2")
    span = color_max - color_min
    if not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("lut color range must be finite and positive")

    out_of_range = np.any((color < color_min) | (color > color_max), axis=1)
    normalized = np.clip((color - color_min) / span, 0.0, 1.0)
    coordinates = np.rint(normalized * (size - 1)).astype(np.int64)
    bin_ids = (coordinates[:, 0] * size + coordinates[:, 1]) * size + coordinates[:, 2]
    return {
        "bin_ids": bin_ids,
        "coordinates": coordinates,
        "out_of_range": out_of_range,
    }


def maximum_disc(
    points: np.ndarray,
    weights: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, float]:
    """Return the best weighted closed-disc center and its actual covered weight.

    The center is unrestricted in slope space. Identical slope locations are
    merged before an exact boundary-angle sweep; event ordering and numerical
    tolerances follow the LUT quality handoff document.
    """
    values = np.asarray(points, dtype=np.float64)
    point_weights = np.asarray(weights, dtype=np.float64)
    radius = float(tau)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("points must be a nonempty [N, 2] array")
    if point_weights.shape != (values.shape[0],):
        raise ValueError("weights must have shape [N]")
    if not np.isfinite(values).all() or not np.isfinite(point_weights).all():
        raise ValueError("points and weights must be finite")
    if np.any(point_weights < 0):
        raise ValueError("weights must be nonnegative")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("tau must be finite and nonnegative")

    unique_points, inverse = np.unique(values, axis=0, return_inverse=True)
    unique_weights = np.bincount(inverse, weights=point_weights)
    total_weight = float(np.sum(unique_weights))
    if total_weight <= 0:
        return unique_points[0].copy(), 0.0

    if radius == 0.0 or unique_points.shape[0] == 1:
        winner = int(np.argmax(unique_weights))
        return unique_points[winner].copy(), float(unique_weights[winner])

    minimum = np.min(unique_points, axis=0)
    maximum = np.max(unique_points, axis=0)
    midpoint = (minimum + maximum) / 2.0
    final_radius = _closed_radius(radius)
    if np.all(np.linalg.norm(unique_points - midpoint, axis=1) <= final_radius):
        return midpoint, total_weight

    best_center: np.ndarray | None = None
    best_swept_weight = -np.inf
    two_pi = 2.0 * np.pi

    # np.unique returns lexicographically sorted points, fixing anchor order.
    for anchor_index, anchor in enumerate(unique_points):
        active_weight = float(unique_weights[anchor_index])
        events: list[tuple[float, bool, float]] = []

        for other_index, other in enumerate(unique_points):
            if other_index == anchor_index:
                continue
            delta_vector = other - anchor
            distance = float(np.hypot(delta_vector[0], delta_vector[1]))
            if distance > 2.0 * radius:
                continue

            direction = float(np.arctan2(delta_vector[1], delta_vector[0]))
            half_width = float(np.arccos(np.clip(distance / (2.0 * radius), -1.0, 1.0)))
            start = (direction - half_width) % two_pi
            end = (direction + half_width) % two_pi
            weight = float(unique_weights[other_index])
            if start > end:
                # This interval covers angle zero; its end leaves and start re-enters.
                active_weight += weight
                events.append((end, False, weight))
                events.append((start, True, weight))
            else:
                events.append((start, True, weight))
                events.append((end, False, weight))

        events.sort(key=lambda item: item[0])
        candidate_center = anchor + np.array([radius, 0.0])
        if active_weight > best_swept_weight + _COVERAGE_UPDATE_TOLERANCE:
            best_swept_weight = active_weight
            best_center = candidate_center

        event_index = 0
        while event_index < len(events):
            group_start = events[event_index][0]
            group_end = event_index + 1
            while (group_end < len(events)
                   and events[group_end][0] - group_start <= _ANGLE_GROUP_TOLERANCE):
                group_end += 1

            entered = 0.0
            left = 0.0
            for angle, entering, weight in events[event_index:group_end]:
                if entering:
                    entered += weight
                else:
                    left += weight
            active_weight += entered
            angle = group_start
            candidate_center = anchor + radius * np.array([np.cos(angle), np.sin(angle)])
            if active_weight > best_swept_weight + _COVERAGE_UPDATE_TOLERANCE:
                best_swept_weight = active_weight
                best_center = candidate_center
            active_weight -= left
            event_index = group_end

    if best_center is None:
        best_center = unique_points[0].copy()
    covered = _covered_weight(values, point_weights, best_center, final_radius)
    return best_center, covered


def evaluate_quality(
    colors: np.ndarray,
    slopes: np.ndarray,
    poses: np.ndarray,
    lut: Any,
    taus: np.ndarray | list[float],
    weighting: str = "sample",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one-value incompatibility for all occupied nearest-node bins.

    ``colors`` and ``slopes`` contain paired raw rows, and ``poses`` identifies
    the complete calibration image for each row. The returned sparse ``bins``
    arrays are ordered by ascending flattened LUT node id.
    """
    color, slope, pose, tau_values = _validate_samples(colors, slopes, poses, taus)
    weights = _sample_weights(pose, weighting)
    assignments = bin_samples(color, lut)
    bin_ids = assignments["bin_ids"]
    unique_bins, inverse = np.unique(bin_ids, return_inverse=True)
    bin_count = unique_bins.size
    tau_count = tau_values.size
    size = int(lut.size)
    sample_counts = np.bincount(inverse, minlength=bin_count).astype(np.int64)
    bin_weights = np.bincount(inverse, weights=weights, minlength=bin_count)
    pose_counts = np.zeros(bin_count, dtype=np.int64)
    unique_slope_counts = np.zeros(bin_count, dtype=np.int64)
    original_valid = np.zeros(bin_count, dtype=np.bool_)
    coordinates = _unflatten_bins(unique_bins, size)
    slope_min = np.empty((bin_count, 2), dtype=np.float64)
    slope_max = np.empty((bin_count, 2), dtype=np.float64)
    tilt_min = np.empty(bin_count, dtype=np.float64)
    tilt_max = np.empty(bin_count, dtype=np.float64)
    normal_rms = np.empty(bin_count, dtype=np.float64)
    centers = np.empty((tau_count, bin_count, 2), dtype=np.float64)
    incompatibility = np.empty((tau_count, bin_count), dtype=np.float64)
    original_valid_table = getattr(lut, "original_valid", None)
    if original_valid_table is not None:
        original_valid_table = np.asarray(original_valid_table, dtype=np.bool_)
        if original_valid_table.shape != (size, size, size):
            raise ValueError("lut original_valid must match the cubic LUT dimensions")

    normals = _slopes_to_normals(slope)
    tilts = np.degrees(np.arctan(np.hypot(slope[:, 0], slope[:, 1])))
    normal_coverage, theta_edges, phi_edges = _normal_coverage(slope)

    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(sample_counts)[:-1]]
    for bin_index, start in enumerate(starts):
        sample_indices = order[start:start + sample_counts[bin_index]]
        bin_slopes = slope[sample_indices]
        bin_weights_for_samples = weights[sample_indices]
        bin_poses = pose[sample_indices]
        bin_normals = normals[sample_indices]
        bin_tilts = tilts[sample_indices]
        slope_min[bin_index] = np.min(bin_slopes, axis=0)
        slope_max[bin_index] = np.max(bin_slopes, axis=0)
        pose_counts[bin_index] = np.unique(bin_poses).size
        unique_slope_counts[bin_index] = np.unique(bin_slopes, axis=0).shape[0]
        tilt_min[bin_index] = np.min(bin_tilts)
        tilt_max[bin_index] = np.max(bin_tilts)
        mean_normal = np.sum(bin_normals * bin_weights_for_samples[:, None], axis=0)
        mean_normal /= np.linalg.norm(mean_normal)
        angles = np.arccos(np.clip(bin_normals @ mean_normal, -1.0, 1.0))
        normal_rms[bin_index] = np.degrees(np.sqrt(
            np.sum(bin_weights_for_samples * angles**2) / bin_weights[bin_index]
        ))
        if original_valid_table is not None:
            r, g, b = coordinates[bin_index]
            original_valid[bin_index] = original_valid_table[r, g, b]

        node_slope = np.asarray(lut.slopes[
            coordinates[bin_index, 0],
            coordinates[bin_index, 1],
            coordinates[bin_index, 2],
        ], dtype=np.float64)
        for tau_index, tau in enumerate(tau_values):
            center, covered = maximum_disc(
                bin_slopes, bin_weights_for_samples, float(tau))
            node_covered = _covered_weight(
                bin_slopes, bin_weights_for_samples, node_slope, _closed_radius(float(tau)))
            if node_covered > covered:
                center = node_slope.copy()
                covered = node_covered
            centers[tau_index, bin_index] = center
            incompatibility[tau_index, bin_index] = np.clip(
                1.0 - covered / bin_weights[bin_index], 0.0, 1.0)

    total_weight = float(np.sum(weights))
    out_of_range_fraction = float(
        np.sum(weights[assignments["out_of_range"]]) / total_weight)
    singleton_mask = sample_counts[inverse] == 1
    singleton_sample_fraction = float(np.sum(weights[singleton_mask]) / total_weight)
    curves = []
    for tau_index, tau in enumerate(tau_values):
        global_incompatibility = float(
            np.sum(bin_weights * incompatibility[tau_index]) / total_weight)
        curves.append({"tau": float(tau), "incompatibility": global_incompatibility})

    report: dict[str, Any] = {
        "curves": curves,
        "coverage": {
            "sample_count": int(color.shape[0]),
            "weighted_sample_count": total_weight,
            "occupied_bin_count": int(bin_count),
            "singleton_sample_fraction": singleton_sample_fraction,
            "out_of_range_fraction": out_of_range_fraction,
        },
        "holdout": None,
    }
    bins: dict[str, np.ndarray] = {
        "bin_ids": unique_bins.astype(np.int64, copy=False),
        "sample_count": sample_counts,
        "weight": bin_weights.astype(np.float64, copy=False),
        "pose_count": pose_counts,
        "unique_slope_count": unique_slope_counts,
        "original_valid": original_valid,
        "tilt_min_degrees": tilt_min,
        "tilt_max_degrees": tilt_max,
        "normal_rms_degrees": normal_rms,
        "coordinates": coordinates,
        "slope_min": slope_min,
        "slope_max": slope_max,
        "optimal_center": centers,
        "incompatibility": incompatibility,
        "taus": tau_values.copy(),
        "normal_coverage_count": normal_coverage,
        "theta_bin_edges_degrees": theta_edges,
        "phi_bin_edges_degrees": phi_edges,
        "out_of_range_fraction": np.asarray(out_of_range_fraction, dtype=np.float64),
    }
    return report, bins


def evaluate_holdout(
    holdout_colors: np.ndarray,
    holdout_slopes: np.ndarray,
    holdout_poses: np.ndarray,
    reference_bins: dict[str, np.ndarray],
    lut: Any,
    taus: np.ndarray | list[float],
    weighting: str = "sample",
) -> dict[str, Any]:
    """Transfer reference-bin centers to an independent pose holdout set."""
    color, slope, pose, tau_values = _validate_samples(
        holdout_colors, holdout_slopes, holdout_poses, taus, allow_empty=True)
    if not np.array_equal(np.asarray(reference_bins["taus"], dtype=np.float64), tau_values):
        raise ValueError("holdout taus must match the reference bin taus")
    ref_ids = np.asarray(reference_bins["bin_ids"], dtype=np.int64)
    centers = np.asarray(reference_bins["optimal_center"], dtype=np.float64)
    if centers.shape != (tau_values.size, ref_ids.size, 2):
        raise ValueError("reference optimal_center must have shape [T, B, 2]")

    if color.shape[0] == 0:
        return {
            "unseen_bin_fraction": None,
            "out_of_range_fraction": None,
            "curves": [
                {
                    "tau": float(tau),
                    "transferred_disc_failure_or_unseen": None,
                    "transferred_disc_known_bin_error": None,
                }
                for tau in tau_values
            ],
        }

    weights = _sample_weights(pose, weighting)
    assignments = bin_samples(color, lut)
    holdout_ids = assignments["bin_ids"]
    ref_lookup = {int(bin_id): index for index, bin_id in enumerate(ref_ids)}
    ref_indices = np.fromiter(
        (ref_lookup.get(int(bin_id), -1) for bin_id in holdout_ids),
        dtype=np.int64,
        count=holdout_ids.size,
    )
    unseen = ref_indices < 0
    total_weight = float(np.sum(weights))
    unseen_bin_fraction = float(np.sum(weights[unseen]) / total_weight)
    out_of_range_fraction = float(
        np.sum(weights[assignments["out_of_range"]]) / total_weight)
    known = ~unseen
    known_weight = float(np.sum(weights[known]))
    curves = []
    for tau_index, tau in enumerate(tau_values):
        known_failed = np.zeros(holdout_ids.shape, dtype=np.bool_)
        if known_weight > 0:
            rows = np.flatnonzero(known)
            transferred_centers = centers[tau_index, ref_indices[rows]]
            if tau == 0:
                known_failed[rows] = np.any(slope[rows] != transferred_centers, axis=1)
            else:
                distances = np.linalg.norm(slope[rows] - transferred_centers, axis=1)
                known_failed[rows] = distances > _closed_radius(float(tau))
            known_error: float | None = float(
                np.sum(weights[known_failed]) / known_weight)
        else:
            known_error = None
        failure_or_unseen = float(
            np.sum(weights[known_failed | unseen]) / total_weight)
        curves.append({
            "tau": float(tau),
            "transferred_disc_failure_or_unseen": failure_or_unseen,
            "transferred_disc_known_bin_error": known_error,
        })
    return {
        "unseen_bin_fraction": unseen_bin_fraction,
        "out_of_range_fraction": out_of_range_fraction,
        "curves": curves,
    }


def split_samples(
    image_index: np.ndarray,
    image_paths: np.ndarray | list[str],
    lut: Any,
    split: str = "auto",
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Split complete poses using the selected training-only LUT metadata."""
    indices = np.asarray(image_index)
    paths = _image_path_strings(image_paths)
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("image_index must be a nonempty integer vector")
    if paths.ndim != 1 or paths.size == 0 or np.unique(paths).size != paths.size:
        raise ValueError("image_paths must be a nonempty vector of unique strings")
    if np.any(indices < 0) or np.any(indices >= paths.size):
        raise ValueError("image_index contains an invalid image_paths index")
    if split not in {"auto", "all"}:
        raise ValueError("split must be 'auto' or 'all'")

    metadata = getattr(lut, "metadata", {}) or {}
    validation_status = metadata.get("validation_status")
    image_keys = paths[indices]
    if split == "all" or (split == "auto" and validation_status != "training_only"):
        reference_mask = np.ones(indices.shape, dtype=np.bool_)
        return reference_mask, None, {
            "split": "all",
            "requested_split": split,
            "validation_status": validation_status,
            "reference_images": np.unique(image_keys).tolist(),
            "holdout_images": None,
        }

    training_images = _metadata_images(metadata, "training_images")
    validation_images = _metadata_images(metadata, "validation_images")
    if not training_images or not validation_images:
        raise ValueError("training-only LUT must have nonempty training_images and validation_images")
    if set(training_images) & set(validation_images):
        raise ValueError("training_images and validation_images must be disjoint")
    training_set = set(training_images)
    validation_set = set(validation_images)
    unassigned = sorted(set(image_keys.tolist()) - training_set - validation_set)
    if unassigned:
        raise ValueError(f"training/validation image lists do not cover sample poses: {unassigned}")
    reference_mask = np.isin(image_keys, training_images)
    holdout_mask = np.isin(image_keys, validation_images)
    if not np.all(reference_mask | holdout_mask):
        raise ValueError("training/validation image lists must cover every sample")
    return reference_mask, holdout_mask, {
        "split": "pose_holdout",
        "requested_split": split,
        "validation_status": validation_status,
        "reference_images": training_images,
        "holdout_images": validation_images,
    }


def _validate_samples(
    colors: np.ndarray,
    slopes: np.ndarray,
    poses: np.ndarray,
    taus: np.ndarray | list[float],
    *,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    color = np.asarray(colors, dtype=np.float64)
    slope = np.asarray(slopes, dtype=np.float64)
    pose = np.asarray(poses)
    tau_values = np.asarray(taus, dtype=np.float64)
    if color.ndim != 2 or color.shape[1] != 3:
        raise ValueError("colors must have shape [N, 3]")
    if slope.shape != (color.shape[0], 2):
        raise ValueError("slopes must have shape [N, 2] and pair with colors")
    if pose.ndim != 1 or pose.shape[0] != color.shape[0]:
        raise ValueError("poses must have shape [N] and pair with colors")
    if color.shape[0] == 0 and not allow_empty:
        raise ValueError("samples must be nonempty")
    if not np.isfinite(color).all() or not np.isfinite(slope).all():
        raise ValueError("colors and slopes must contain only finite values")
    if tau_values.ndim != 1 or tau_values.size < 1:
        raise ValueError("taus must be a nonempty vector")
    if not np.isfinite(tau_values).all() or np.any(tau_values < 0):
        raise ValueError("taus must be finite and nonnegative")
    if np.unique(tau_values).size != tau_values.size:
        raise ValueError("taus must be distinct")
    return color, slope, pose, np.sort(tau_values)


def _sample_weights(poses: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "sample":
        return np.ones(poses.shape[0], dtype=np.float64)
    if weighting != "pose_equal":
        raise ValueError("weighting must be 'sample' or 'pose_equal'")
    _, inverse, counts = np.unique(poses, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(np.float64)


def _closed_radius(tau: float) -> float:
    if tau == 0.0:
        return 0.0
    return tau + 1e-12 * max(1.0, tau)


def _covered_weight(
    points: np.ndarray,
    weights: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> float:
    if radius == 0:
        covered = np.all(points == center, axis=1)
    else:
        covered = np.linalg.norm(points - center, axis=1) <= radius
    return float(np.sum(weights[covered]))


def _unflatten_bins(bin_ids: np.ndarray, size: int) -> np.ndarray:
    red = bin_ids // (size * size)
    green = (bin_ids // size) % size
    blue = bin_ids % size
    return np.column_stack((red, green, blue)).astype(np.int64, copy=False)


def _slopes_to_normals(slopes: np.ndarray) -> np.ndarray:
    normals = np.column_stack((-slopes[:, 0], -slopes[:, 1], np.ones(slopes.shape[0])))
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _normal_coverage(slopes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_edges = np.linspace(0.0, 90.0, 19)
    phi_edges = np.linspace(-180.0, 180.0, 37)
    theta = np.degrees(np.arctan(np.hypot(slopes[:, 0], slopes[:, 1])))
    phi = np.degrees(np.arctan2(-slopes[:, 1], -slopes[:, 0]))
    coverage, _, _ = np.histogram2d(theta, phi, bins=(theta_edges, phi_edges))
    return coverage.astype(np.int64), theta_edges, phi_edges


def _image_path_strings(image_paths: np.ndarray | list[str]) -> np.ndarray:
    raw = np.asarray(image_paths)
    if raw.dtype.kind == "S":
        return np.char.decode(raw, "utf-8")
    if raw.dtype.kind == "U":
        return raw.astype(str, copy=False)
    if raw.dtype.kind == "O":
        values = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw]
        return np.asarray(values, dtype=str)
    return raw.astype(str)


def _metadata_images(metadata: dict[str, Any], name: str) -> list[str]:
    value = metadata.get(name)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"training-only LUT metadata must contain {name}")
    images = [str(item) for item in value]
    if len(images) != len(set(images)):
        raise ValueError(f"{name} must not contain duplicate image paths")
    return images
