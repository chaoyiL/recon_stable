"""Score every curvature on the same material identities for each sphere pose."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np

from .experiment import depth_rmse, pose_metric


PREDICTION_FIELDS = {
    "naive_lut": "slope_naive_lut",
    "fixed_position_slope_gn3": "slope_fixed_position_gn3",
    "depth_slope_gn3": "slope_depth_gn3",
}


def score_common_domain(
    output: Path,
    condition_ids: Sequence[str],
    pose_ids: Sequence[str],
    slope_tolerances: list[float],
) -> tuple[dict, dict]:
    """Return per-condition scores and coverage on cross-curvature intersections.

    Grid rows/columns identify the same reference arclength point in all raw
    files. Each method keeps every common point, including weak/rank-deficient
    solver results. An empty intersection is reported explicitly; callers must
    not aggregate it as a successfully evaluated pose.
    """
    scores = {condition: {"pose_metrics": {}} for condition in condition_ids}
    coverage = {}
    for pose in pose_ids:
        archives = {}
        identities = {}
        for condition in condition_ids:
            path = output / "raw" / condition / f"{pose}.npz"
            with np.load(path, allow_pickle=False) as saved:
                archives[condition] = {key: saved[key] for key in saved.files if key != "metadata"}
            indices = archives[condition]["grid_index"]
            identities[condition] = [tuple(index) for index in indices.tolist()]
        common = set.intersection(*(set(identities[condition]) for condition in condition_ids))
        coverage[pose] = {
            "common_point_count": len(common),
            "full_point_counts": {condition: len(identities[condition]) for condition in condition_ids},
        }
        if not common:
            continue
        for condition in condition_ids:
            arrays = archives[condition]
            selected = np.array([identity in common for identity in identities[condition]], dtype=bool)
            area = arrays["area_mm2"][selected]
            weight = area / area.sum()
            methods = {}
            for method, field in PREDICTION_FIELDS.items():
                methods[method] = {
                    "sample_count": len(common),
                    "lut_out_of_range_fraction": float(np.sum(weight * ~arrays["lut_color_in_range"][selected])),
                    "lut_original_support_fraction": float(np.sum(weight * arrays["lut_original_node_support"][selected])),
                    **pose_metric(arrays[field][selected], arrays["slope_truth"][selected], area, slope_tolerances),
                }
            depth_row = methods["depth_slope_gn3"]
            predicted = arrays["depth_estimate_gn3_mm"][selected]
            truth = arrays["depth_truth_mm"][selected]
            depth_row["depth_rmse_mm"] = depth_rmse(predicted, truth, area)
            depth_row["depth_bias_mm"] = float(np.sum(weight * (predicted - truth)))
            for name in ("rank_deficient", "depth_effect_weak"):
                depth_row[f"depth_diag_{name}_count"] = int(np.count_nonzero(
                    arrays[f"depth_diagnostic_{name}"][selected]))
            scores[condition]["pose_metrics"][pose] = {"sample_count": len(common), "methods": methods}
    return scores, coverage
