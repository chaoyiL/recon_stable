"""Input checks, CLI and report writing for signed-RGB LUT quality evaluation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from calibrate.utils.norm_pipeline import parse_settings
from utils.normal_mapping import NormalCalibration

from . import metric


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_TAUS = (0.0, 0.025, 0.05, 0.1, 0.2)
METRIC_VERSION = "lut_single_value_incompatibility_v1"
REPORT_FORMAT_VERSION = 2

_SAMPLE_FIELDS = (
    "color_residual_samples",
    "slope_samples",
    "image_index",
    "image_paths",
)
_SAMPLE_METADATA_FIELDS = (
    "run_signature",
    "simulation_config_signature",
    "light_geometry_model",
    "coordinate_kind",
    "coordinate_chart_id",
    "coordinate_chart_file",
    "frame_convention",
    "residual_domain",
    "color_space",
    "rgb_transfer_function",
    "image_encoding",
    "observation_model_id",
    "backend",
    "reference_image",
    "sphere_radius_mm",
    "sphere_side",
    "boundary_model",
    "contact_offset_mm",
)
_MATCHED_METADATA_FIELDS = (
    "run_signature",
    "simulation_config_signature",
    "light_geometry_model",
    "coordinate_kind",
    "coordinate_chart_id",
    "frame_convention",
    "residual_domain",
    "color_space",
    "rgb_transfer_function",
    "image_encoding",
    "observation_model_id",
    "backend",
    "reference_image",
    "sphere_radius_mm",
    "sphere_side",
    "boundary_model",
    "contact_offset_mm",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_value) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_scalar(data: Any, key: str) -> Any:
    if key not in data:
        raise ValueError(f"Normal samples are missing required field {key!r}")
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"Normal sample field {key!r} must be scalar")
    return value.item()


def _read_string_vector(data: Any, key: str) -> np.ndarray:
    raw = np.asarray(data[key])
    if raw.ndim != 1 or raw.dtype.kind not in "SU":
        raise ValueError(f"Normal sample field {key!r} must be a string vector")
    if raw.dtype.kind == "S":
        try:
            return np.asarray([value.decode("utf-8") for value in raw], dtype=str)
        except UnicodeDecodeError as exc:
            raise ValueError(f"Normal sample field {key!r} contains invalid UTF-8") from exc
    return raw.astype(str)


def _read_samples(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in _SAMPLE_FIELDS if key not in archive]
            if missing:
                raise ValueError(f"Normal samples are missing required fields: {missing}")
            colors = np.asarray(archive["color_residual_samples"], dtype=np.float64)
            slopes = np.asarray(archive["slope_samples"], dtype=np.float64)
            image_index = np.asarray(archive["image_index"])
            image_paths = _read_string_vector(archive, "image_paths")

            if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] == 0:
                raise ValueError("color_residual_samples must be a nonempty [N,3] array")
            if slopes.shape != (len(colors), 2):
                raise ValueError("slope_samples must have shape [N,2] and pair with colors")
            if not np.isfinite(colors).all() or not np.isfinite(slopes).all():
                raise ValueError("Normal sample colors and slopes must be finite")
            if image_index.shape != (len(colors),) or image_index.dtype.kind not in "iu":
                raise ValueError("image_index must be an integer [N] array")
            if (image_paths.ndim != 1 or not len(image_paths)
                    or any(not value.strip() for value in image_paths)
                    or len(set(image_paths.tolist())) != len(image_paths)):
                raise ValueError("image_paths must be a nonempty [P] array of unique pose keys")
            if (image_index < 0).any() or (image_index >= len(image_paths)).any():
                raise ValueError("image_index contains a value outside image_paths")

            metadata = {
                key: _read_scalar(archive, key)
                for key in _SAMPLE_METADATA_FIELDS
                if key in archive
            }
            if "format_version" in archive and int(_read_scalar(archive, "format_version")) != 1:
                raise ValueError("Grouped image samples require format_version=1")
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not read normal samples {path}: {exc}") from exc

    return {
        "colors": colors,
        "slopes": slopes,
        "image_index": image_index.astype(np.int64, copy=False),
        "image_paths": image_paths,
    }, metadata


def _float_equal(first: Any, second: Any) -> bool:
    try:
        a, b = float(first), float(second)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


def _metadata_equal(key: str, first: Any, second: Any) -> bool:
    if key in {"contact_offset_mm", "sphere_radius_mm"}:
        return _float_equal(first, second)
    if key == "backend":
        first_value, second_value = str(first).lower(), str(second).lower()
        if first_value == "mujoco":
            first_value = "xense"
        if second_value == "mujoco":
            second_value = "xense"
        return first_value == second_value
    return str(first) == str(second)


def _validate_metadata(
    sample_metadata: dict[str, Any],
    lut: NormalCalibration,
    expected_contact_offset_mm: float,
) -> list[str]:
    lut_metadata = lut.metadata
    warnings = []
    for key in _MATCHED_METADATA_FIELDS:
        if key in sample_metadata and key in lut_metadata and not _metadata_equal(
            key, sample_metadata[key], lut_metadata[key]
        ):
            left, right = sample_metadata[key], lut_metadata[key]
            raise ValueError(f"Sample/LUT metadata mismatch for {key}: {left!r} != {right!r}")

    missing_declarations = [
        key for key in _MATCHED_METADATA_FIELDS
        if key != "contact_offset_mm" and (key not in sample_metadata or key not in lut_metadata)
    ]
    if missing_declarations:
        warnings.append(
            "Source compatibility cannot be fully verified because declarations are missing "
            "from one or both artifacts: " + ", ".join(missing_declarations) + "."
        )

    if "contact_offset_mm" in sample_metadata:
        if not _float_equal(sample_metadata["contact_offset_mm"], expected_contact_offset_mm):
            raise ValueError(
                "Sample contact_offset_mm differs from config.yaml contact_surface.offset_mm; "
                "rerun prepare, samples, and LUT calibration"
            )
    if "contact_offset_mm" in lut_metadata:
        if not _float_equal(lut_metadata["contact_offset_mm"], expected_contact_offset_mm):
            raise ValueError(
                "LUT contact_offset_mm differs from config.yaml contact_surface.offset_mm; "
                "rerun prepare, samples, and LUT calibration"
            )
    if "contact_offset_mm" in sample_metadata and "contact_offset_mm" in lut_metadata:
        if not _float_equal(sample_metadata["contact_offset_mm"], lut_metadata["contact_offset_mm"]):
            raise ValueError("Sample/LUT metadata mismatch for contact_offset_mm")
    else:
        warnings.append(
            "contact_offset_mm is not declared by both artifacts; contact-surface compatibility "
            "cannot be fully verified."
        )

    if sample_metadata.get("color_space") not in (None, "linear_rgb"):
        raise ValueError("Sample color_space must be linear_rgb when declared")
    if lut_metadata.get("color_space") not in (None, "linear_rgb"):
        raise ValueError("LUT color_space must be linear_rgb when declared")

    if lut_metadata.get("validation_status") == "training_only":
        training_images = lut_metadata.get("training_images")
        validation_images = lut_metadata.get("validation_images")
        if not isinstance(training_images, list) or not training_images:
            raise ValueError("training_only LUT must declare nonempty training_images")
        if not isinstance(validation_images, list) or not validation_images:
            raise ValueError("training_only LUT must declare nonempty validation_images")
        training_set, validation_set = set(map(str, training_images)), set(map(str, validation_images))
        if training_set & validation_set:
            raise ValueError("LUT training_images and validation_images overlap")
    else:
        warnings.append(
            "This LUT is not marked training_only; split=auto will use all archived samples as reference."
        )

    return warnings


def _read_source_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid sample source manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Sample source manifest root must be an object: {path}")
    return value


def _validate_source_manifest(
    source_manifest: dict[str, Any] | None,
    sample_metadata: dict[str, Any],
    lut: NormalCalibration,
) -> list[str]:
    if source_manifest is None:
        return ["No manifest.json was found beside the sample file; archived run provenance is unavailable."]
    warnings = []
    lut_metadata = lut.metadata
    for key in _MATCHED_METADATA_FIELDS:
        if key not in source_manifest:
            continue
        for artifact_name, metadata in (("samples", sample_metadata), ("LUT", lut_metadata)):
            if key in metadata and not _metadata_equal(key, source_manifest[key], metadata[key]):
                raise ValueError(
                    f"Source manifest/{artifact_name} mismatch for {key}: "
                    f"{source_manifest[key]!r} != {metadata[key]!r}"
                )
    if not any(key in source_manifest for key in ("run_signature", "simulation_config_signature")):
        warnings.append(
            "Source manifest has no run or simulation signature; only its declared coordinate and "
            "other shared metadata can be checked."
        )
    return warnings


def _validate_pose_split(samples: dict[str, np.ndarray], lut: NormalCalibration) -> None:
    if lut.metadata.get("validation_status") != "training_only":
        return
    training = set(map(str, lut.metadata["training_images"]))
    validation = set(map(str, lut.metadata["validation_images"]))
    image_paths = set(samples["image_paths"][samples["image_index"]].tolist())
    if training & validation:
        raise ValueError("LUT training_images and validation_images overlap")
    if not image_paths <= training | validation:
        missing = sorted(image_paths - (training | validation))
        raise ValueError(
            "training_only LUT pose lists must cover all sample poses "
            f"(missing={missing[:5]})"
        )


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, float, Path]:
    config_path = Path(args.config).expanduser().resolve()
    settings = parse_settings(config_path)
    sample_path = Path(args.samples).expanduser().resolve() if args.samples else settings.input_file
    if args.lut:
        lut_path = Path(args.lut).expanduser().resolve()
    else:
        lut_path = settings.output.with_name(f"{settings.output.stem}_train{settings.output.suffix}")
    output_path = Path(args.out).expanduser().resolve() if args.out else settings.work_dir / "quality"
    return config_path, sample_path, lut_path, settings.contact_offset_mm, output_path


def _check_output_path(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty output directory: {path}")


def _validation_status(settings: Any, sample_path: Path) -> tuple[str, Path | None]:
    report_path = (settings.work_dir / "verification" / "validation.json").resolve()
    if not report_path.is_file():
        return "unknown", None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid calibration validation report {report_path}: {exc}") from exc
    # A custom --samples file may have no relationship to the configured validation report.
    configured_samples = settings.input_file.resolve()
    if sample_path.resolve() != configured_samples:
        return "unknown", None
    return str(report.get("status", "unknown")), report_path


def _normalize_taus(values: list[float], detail_tau: float) -> tuple[list[float], float]:
    taus = np.asarray(values, dtype=np.float64)
    if taus.ndim != 1 or len(taus) < 2 or not np.isfinite(taus).all() or (taus < 0).any():
        raise ValueError("--taus must contain at least two finite nonnegative values")
    if len(np.unique(taus)) != len(taus):
        raise ValueError("--taus values must be unique")
    taus = np.sort(taus)
    if not np.isfinite(detail_tau) or detail_tau not in taus:
        raise ValueError("--detail-tau must be one of --taus")
    return taus.tolist(), float(detail_tau)


def _write_curves(path: Path, reference: dict[str, Any], holdout: dict[str, Any] | None) -> None:
    reference_curves = reference.get("curves")
    if not isinstance(reference_curves, list) or not reference_curves:
        raise ValueError("Metric reference report must contain a nonempty curves list")
    holdout_by_tau = {}
    if holdout is not None:
        for row in holdout.get("curves", []):
            holdout_by_tau[float(row["tau"])] = row

    keys = ["tau"]
    for row in reference_curves:
        for key in row:
            if key != "tau" and key not in keys:
                keys.append(key)
    holdout_keys = []
    for row in holdout_by_tau.values():
        for key in row:
            if key != "tau" and key not in holdout_keys:
                holdout_keys.append(key)
    columns = keys + [f"holdout_{key}" for key in holdout_keys]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in reference_curves:
            result = {key: row.get(key) for key in keys}
            held = holdout_by_tau.get(float(row["tau"]), {})
            result.update({f"holdout_{key}": held.get(key) for key in holdout_keys})
            writer.writerow(result)


def _write_per_bin(path: Path, bins: dict[str, Any]) -> None:
    values = {}
    for key, value in bins.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"Metric per-bin field {key!r} cannot be an object array")
        values[key] = array
    if "bin_ids" not in values or "optimal_center" not in values or "incompatibility" not in values:
        raise ValueError("Metric per-bin result is missing required arrays")
    np.savez_compressed(path, **values)


def _select_examples(indices: np.ndarray, slopes: np.ndarray, center: np.ndarray, limit: int) -> np.ndarray:
    limit = min(limit, len(indices))
    distances = np.linalg.norm(slopes - center, axis=1)
    nearest = np.argsort(distances, kind="stable")
    farthest = nearest[::-1]
    selected = []
    seen = set()
    left = right = 0
    while len(selected) < limit:
        for order, cursor_name in ((nearest, "left"), (farthest, "right")):
            cursor = left if cursor_name == "left" else right
            while cursor < len(order) and int(order[cursor]) in seen:
                cursor += 1
            if cursor < len(order):
                value = int(order[cursor])
                selected.append(value)
                seen.add(value)
                cursor += 1
            if cursor_name == "left":
                left = cursor
            else:
                right = cursor
            if len(selected) >= limit:
                break
    return indices[np.asarray(selected, dtype=np.int64)]


def _sample_weights(poses: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "sample":
        return np.ones(len(poses), dtype=np.float64)
    if weighting != "pose_equal":
        raise ValueError("weighting must be 'sample' or 'pose_equal'")
    _, inverse, counts = np.unique(poses, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _write_top_bins(
    path: Path,
    bins: dict[str, Any],
    reference_bin_ids: np.ndarray,
    reference_slopes: np.ndarray,
    reference_source_indices: np.ndarray,
    taus: list[float],
    detail_tau: float,
    top_bins: int,
    examples_per_bin: int,
) -> None:
    ids = np.asarray(bins["bin_ids"], dtype=np.int64)
    counts = np.asarray(bins["sample_count"], dtype=np.int64)
    incompatibility = np.asarray(bins["incompatibility"], dtype=np.float64)
    centers = np.asarray(bins["optimal_center"], dtype=np.float64)
    tau_index = taus.index(detail_tau)
    if incompatibility.shape != (len(taus), len(ids)) or centers.shape != (len(taus), len(ids), 2):
        raise ValueError("Metric per-bin arrays have incompatible dimensions")
    if counts.shape != ids.shape:
        raise ValueError("Metric sample_count must align with bin_ids")
    order = np.lexsort((ids, -counts, -incompatibility[tau_index]))[:top_bins]
    rows = []
    for item in order:
        bin_id = int(ids[item])
        positions = np.flatnonzero(reference_bin_ids == bin_id)
        selected = _select_examples(
            positions,
            reference_slopes[positions],
            centers[tau_index, item],
            examples_per_bin,
        )
        rows.append({
            "bin_id": bin_id,
            "coordinates": np.asarray(bins.get("coordinates", np.empty((0, 3)))[item]).tolist(),
            "sample_count": int(counts[item]),
            "weight": float(np.asarray(bins.get("weight", counts))[item]),
            "pose_count": int(np.asarray(bins.get("pose_count", np.zeros_like(counts)))[item]),
            "unique_slope_count": int(np.asarray(bins.get("unique_slope_count", np.zeros_like(counts)))[item]),
            "incompatibility": float(incompatibility[tau_index, item]),
            "optimal_center": centers[tau_index, item].tolist(),
            "original_sample_indices": reference_source_indices[selected].astype(int).tolist(),
        })
    _write_json(path, {"format_version": REPORT_FORMAT_VERSION, "detail_tau": detail_tau, "bins": rows})


def _load_calibration(path: Path) -> NormalCalibration:
    try:
        return NormalCalibration.load(path)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not load signed normal LUT {path}: {exc}") from exc


def _load_runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for package in ("matplotlib", "numba", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _dry_run(
    samples: dict[str, np.ndarray],
    lut: NormalCalibration,
    reference_mask: np.ndarray,
    holdout_mask: np.ndarray | None,
    split_metadata: dict[str, Any],
    weighting: str,
) -> dict[str, Any]:
    reference_colors = samples["colors"][reference_mask]
    assigned = metric.bin_samples(reference_colors, lut)
    counts = np.bincount(assigned["bin_ids"])
    occupied_counts = counts[counts > 0]
    weights = _sample_weights(samples["image_index"][reference_mask], weighting)
    total_weight = float(weights.sum())
    return {
        "status": "dry_run",
        "source_sample_count": int(len(samples["colors"])),
        "reference_sample_count": int(reference_mask.sum()),
        "held_out_sample_count": int(holdout_mask.sum()) if holdout_mask is not None else 0,
        "split": split_metadata,
        "weighting": weighting,
        "occupied_bin_count": int(len(occupied_counts)),
        "maximum_bin_sample_count": int(occupied_counts.max()) if len(occupied_counts) else 0,
        "squared_bin_count_sum": int(np.sum(occupied_counts.astype(np.int64) ** 2)),
        "out_of_range_fraction": float(weights[assigned["out_of_range"]].sum() / total_weight),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate single-value incompatibility in a signed RGB normal LUT")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="calibration YAML (default: config.yaml)")
    parser.add_argument("--samples", help="override normal_samples.npz")
    parser.add_argument("--lut", help="override the default training LUT")
    parser.add_argument("--out", help="output directory (default: normal_calibration.work_dir/quality)")
    parser.add_argument("--split", choices=("auto", "all"), default="auto")
    parser.add_argument("--weighting", choices=("sample", "pose_equal"), default="sample")
    parser.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    parser.add_argument("--detail-tau", type=float, default=0.1)
    parser.add_argument("--top-bins", type=int, default=8)
    parser.add_argument("--examples-per-bin", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and report workload without writing files")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    taus, detail_tau = _normalize_taus(args.taus, args.detail_tau)
    if args.top_bins < 1 or args.examples_per_bin < 1:
        raise ValueError("--top-bins and --examples-per-bin must be positive integers")
    config_path, sample_path, lut_path, expected_offset, output_path = _resolve_paths(args)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if not sample_path.is_file():
        raise FileNotFoundError(f"Samples do not exist: {sample_path}")
    if not lut_path.is_file():
        raise FileNotFoundError(f"LUT does not exist: {lut_path}")

    samples, sample_metadata = _read_samples(sample_path)
    lut = _load_calibration(lut_path)
    warnings = _validate_metadata(sample_metadata, lut, expected_offset)
    source_manifest_path = sample_path.parent / "manifest.json"
    source_manifest = _read_source_manifest(source_manifest_path)
    warnings.extend(_validate_source_manifest(source_manifest, sample_metadata, lut))
    _validate_pose_split(samples, lut)
    reference_mask, holdout_mask, split_metadata = metric.split_samples(
        samples["image_index"], samples["image_paths"], lut, split=args.split
    )
    reference_mask = np.asarray(reference_mask, dtype=bool)
    holdout_mask = None if holdout_mask is None else np.asarray(holdout_mask, dtype=bool)
    if reference_mask.shape != samples["image_index"].shape or not reference_mask.any():
        raise ValueError("split_samples returned an invalid or empty reference mask")
    if holdout_mask is not None and (
        holdout_mask.shape != reference_mask.shape or np.any(reference_mask & holdout_mask) or not holdout_mask.any()
    ):
        raise ValueError("split_samples returned an invalid holdout mask")

    if args.dry_run:
        result = _dry_run(samples, lut, reference_mask, holdout_mask, split_metadata, args.weighting)
        result.update({
            "samples": str(sample_path),
            "lut": str(lut_path),
            "lut_status": lut.metadata.get("validation_status", "unknown"),
            "warnings": warnings,
        })
        print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_value))
        return result

    _check_output_path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    settings = parse_settings(config_path)
    calibration_status, validation_path = _validation_status(settings, sample_path)
    input_files = {"samples": sample_path, "lut": lut_path, "config": config_path}
    if source_manifest is not None:
        input_files["source_manifest"] = source_manifest_path.resolve()
    if validation_path is not None:
        input_files["calibration_validation"] = validation_path
    manifest: dict[str, Any] = {
        "format_version": REPORT_FORMAT_VERSION,
        "metric_version": METRIC_VERSION,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "taus": taus,
            "detail_tau": detail_tau,
            "weighting": args.weighting,
            "split": args.split,
            "top_bins": args.top_bins,
            "examples_per_bin": args.examples_per_bin,
        },
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)} for key, path in input_files.items()
        },
        "input_metadata": {
            "sample": sample_metadata,
            "lut": lut.metadata,
            "source_manifest": {
                "path": str(source_manifest_path.resolve()) if source_manifest is not None else None,
                "declarations": {
                    key: source_manifest[key]
                    for key in _MATCHED_METADATA_FIELDS
                    if source_manifest is not None and key in source_manifest
                },
            },
            "calibration_status": calibration_status,
            "validation_report": str(validation_path) if validation_path else None,
        },
        "split": split_metadata,
        "environment": _load_runtime_versions(),
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("runner", Path(__file__).resolve()),
                ("metric", Path(metric.__file__).resolve()),
                ("plots", Path(__file__).with_name("plots.py").resolve()),
            )
        },
        "warnings": warnings,
    }
    _write_json(output_path / "manifest.json", manifest)

    try:
        reference_indices = np.flatnonzero(reference_mask)
        reference_colors = samples["colors"][reference_mask]
        reference_slopes = samples["slopes"][reference_mask]
        reference_poses = samples["image_index"][reference_mask]
        reference_report, bins = metric.evaluate_quality(
            reference_colors, reference_slopes, reference_poses, lut, taus, weighting=args.weighting
        )
        holdout_report = None
        if holdout_mask is not None:
            holdout_report = metric.evaluate_holdout(
                samples["colors"][holdout_mask],
                samples["slopes"][holdout_mask],
                samples["image_index"][holdout_mask],
                bins,
                lut,
                taus,
                weighting=args.weighting,
            )

        quality = {
            "format_version": REPORT_FORMAT_VERSION,
            "metric_version": METRIC_VERSION,
            "interpretation": {
                "quantity": "single_value_incompatibility",
                "definition": (
                    "For each occupied nearest-RGB-node bin, the best possible single (p,q) answer is "
                    "chosen to cover the largest total sample weight within tau; incompatibility is the "
                    "uncovered weight fraction, aggregated by bin weight."
                ),
                "scope": (
                    "This measures disagreement among archived raw calibration slopes assigned to each "
                    "nearest RGB node. It is not LUT prediction error, normal-angle error, or a lower bound "
                    "on continuous interpolation error."
                ),
                "tau_units": "Euclidean distance in dimensionless local slopes (p,q)",
                "rgb_input": "signed linear-RGB residuals from the archived sample file",
            },
            "calibration_status": calibration_status,
            "lut_status": lut.metadata.get("validation_status", "unknown"),
            "parameters": manifest["parameters"],
            "split": split_metadata,
            "source": {
                "sample_count": int(len(samples["colors"])),
                "reference_sample_count": int(reference_mask.sum()),
                "held_out_sample_count": int(holdout_mask.sum()) if holdout_mask is not None else 0,
                "samples": str(sample_path),
                "lut": str(lut_path),
                "source_manifest": str(source_manifest_path.resolve()) if source_manifest is not None else None,
                "configured_contact_offset_mm": expected_offset,
                "sample_contact_offset_mm": sample_metadata.get("contact_offset_mm"),
                "lut_contact_offset_mm": lut.metadata.get("contact_offset_mm"),
            },
            "reference": {
                "curves": reference_report["curves"],
                "coverage": reference_report["coverage"],
            },
            "holdout": holdout_report,
            "warnings": warnings,
        }
        _write_json(output_path / "quality.json", quality)
        _write_curves(output_path / "curves.csv", reference_report, holdout_report)
        _write_per_bin(output_path / "per_bin.npz", bins)

        assigned = metric.bin_samples(reference_colors, lut)
        _write_top_bins(
            output_path / "top_bins.json",
            bins,
            assigned["bin_ids"],
            reference_slopes,
            reference_indices,
            taus,
            detail_tau,
            args.top_bins,
            args.examples_per_bin,
        )

        from .plots import render_plots

        plot_files = render_plots(
            output_path,
            reference_report,
            holdout_report,
            bins,
            reference_colors,
            reference_slopes,
            reference_poses,
            lut,
            taus,
            detail_tau=detail_tau,
            top_bins=args.top_bins,
        )
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["outputs"] = {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in sorted(output_path.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        manifest["plots"] = {name: str(path) for name, path in plot_files.items()}
        _write_json(output_path / "manifest.json", manifest)
        print(f"LUT quality report written to {output_path}")
        return quality
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(output_path / "manifest.json", manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
