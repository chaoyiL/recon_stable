"""Train paired curvature-conditioned background and local normal models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from calibrate.utils.train_normal_mlp import train_normal_mlp
from optical_learning.generation_02.model import (
    FOURIER_FREQUENCIES,
    WIDTHS,
    BackgroundMLP,
    TrainingOptions,
    fit_background,
)
from utils.color_difference import CAMERA_RGB_TRANSFER_FUNCTION
from utils.normal_models import predict_normal_model
from utils.region_annotations import RegionAnnotation, dilate_contact_mask, region_boundary_points
from utils.surface_coordinates import COORDINATE_KIND, FRAME_CONVENTION
from utils.surface_curvature import compute_surface_curvature
from utils.surface_domain import DOMAIN_KIND
from utils.surface_geometry import local_slopes_to_normals
from utils.surface_sampling import sample_mask_to_surface


BACKGROUND_OPTIONS = TrainingOptions(epochs=12)


@dataclass
class _FrameSamples:
    record: dict
    features: np.ndarray
    background_valid: np.ndarray
    contact_valid: np.ndarray


def _frame_samples(record: dict, settings, half_width: float) -> _FrameSamples:
    frame = record['frame']
    grid = frame.systems.observation
    shape = grid.xyz.shape[:2]
    if frame.geometry_stage != 'final' or record.get('preview_errors'):
        raise ValueError(f'{frame.image_key}: training requires accepted final geometry')
    if (frame.chart.chart_id != grid.chart_id or frame.chart.domain_kind != DOMAIN_KIND
            or grid.coordinate_kind != COORDINATE_KIND):
        raise ValueError(f'{frame.image_key}: invalid observation coordinate chart')
    if not np.isclose(frame.contact_offset_mm, settings.contact_offset_mm, rtol=0, atol=1e-9):
        raise ValueError(f'{frame.image_key}: contact surface offset differs from settings')

    expected_shapes = dict(
        observed_color_grid=(*shape, 3), photometric_valid_mask=shape,
        slope_grid=(*shape, 2), sample_valid_mask=shape,
        curvature_grid_per_mm=(*shape, 3), curvature_valid_mask=shape,
        reference_normals=(*shape, 3), tangent_frames=(*shape, 3, 2),
    )
    for name, expected in expected_shapes.items():
        if np.shape(record[name]) != expected:
            raise ValueError(f'{frame.image_key}: {name} must have shape {expected}')
    if np.shape(grid.st) != (*shape, 2):
        raise ValueError(f'{frame.image_key}: observation st does not match the grid')

    image_shape = frame.linear_rgb.shape[:2]
    region = RegionAnnotation(
        image=frame.image_key, image_size=image_shape[::-1], shape='polygon',
        contour_px=np.asarray(record['contact_boundary_px']).tolist(),
    )
    contour = region_boundary_points(region, 512)
    contact = np.zeros(image_shape, np.uint8)
    cv2.fillPoly(contact, [np.rint(contour).astype(np.int32)], 1)
    exclusion = dilate_contact_mask(contact, settings.background_contact_exclusion_margin_px)
    outside = sample_mask_to_surface(~exclusion, grid.uv)

    curvature = compute_surface_curvature(
        grid.xyz, grid.valid_mask, grid.coordinates, frame.template_xy_mm,
        smoothing_half_width_mm=half_width,
    )
    features = np.concatenate((grid.st, curvature.values), axis=-1).astype(np.float32)
    photo = np.asarray(record['photometric_valid_mask'], bool)
    common = (photo & grid.valid_mask & curvature.valid & np.isfinite(features).all(axis=-1)
              & np.isfinite(record['observed_color_grid']).all(axis=-1)
              & np.all((grid.st >= 0) & (grid.st <= 1), axis=-1))
    background_valid = common & outside
    contact_valid = (common & np.asarray(record['sample_valid_mask'], bool)
                     & np.asarray(record['curvature_valid_mask'], bool)
                     & np.isfinite(record['curvature_grid_per_mm']).all(axis=-1)
                     & np.isfinite(record['slope_grid']).all(axis=-1)
                     & np.isfinite(record['reference_normals']).all(axis=-1)
                     & np.isfinite(record['tangent_frames']).all(axis=(-1, -2)))
    if not background_valid.any() or not contact_valid.any():
        raise ValueError(f'{frame.image_key}: background and contact each need valid observations')
    return _FrameSamples(record, features, background_valid, contact_valid)


def _fit_background(frames: list[_FrameSamples], half_width: float) -> BackgroundMLP:
    features = np.concatenate([frame.features[frame.background_valid] for frame in frames])
    colors = np.concatenate([
        frame.record['observed_color_grid'][frame.background_valid] for frame in frames
    ])
    return fit_background(
        features, colors, np.ones(len(colors), np.uint8), BACKGROUND_OPTIONS,
        curvature_smoothing_half_width_mm=half_width,
    )


def _normal_samples(frames: list[_FrameSamples], model: BackgroundMLP, half_width: float) -> dict:
    fields = {name: [] for name in ('colors', 'slopes', 'groups', 'st', 'curvature', 'normals', 'frames')}
    for index, frame in enumerate(frames):
        record, valid = frame.record, frame.contact_valid
        prediction = model.predict(frame.features[valid], curvature_smoothing_half_width_mm=half_width)
        fields['colors'].append(np.asarray(record['observed_color_grid'][valid] - prediction, np.float32))
        fields['slopes'].append(np.asarray(record['slope_grid'][valid], np.float32))
        fields['groups'].append(np.full(valid.sum(), index, np.int32))
        fields['st'].append(np.asarray(frame.features[valid, :2], np.float32))
        fields['curvature'].append(np.asarray(record['curvature_grid_per_mm'][valid], np.float32))
        fields['normals'].append(np.asarray(record['reference_normals'][valid], np.float32))
        fields['frames'].append(np.asarray(record['tangent_frames'][valid], np.float32))
    return {name: np.concatenate(values) for name, values in fields.items()}


def _fit_normal(samples: dict, settings):
    return train_normal_mlp(
        samples['colors'], samples['slopes'], samples['groups'], method='mlp8',
        st=samples['st'], curvature_per_mm=samples['curvature'],
        curvature_valid_mask=np.ones(len(samples['colors']), bool), options=settings.mlp_options,
    )


def _background_validation(frames: list[_FrameSamples], model: BackgroundMLP, half_width: float) -> dict:
    errors = []
    per_image = []
    for frame in frames:
        mask = frame.background_valid
        predicted = model.predict(frame.features[mask], curvature_smoothing_half_width_mm=half_width)
        error = np.asarray(predicted - frame.record['observed_color_grid'][mask], np.float64)
        errors.append(error)
        per_image.append(dict(image_key=frame.record['frame'].image_key, samples=len(error),
                              rgb_rmse=float(np.sqrt(np.mean(error ** 2)))))
    error = np.concatenate(errors)
    return dict(sample_count=len(error), rgb_rmse=float(np.sqrt(np.mean(error ** 2))),
                channel_rmse=np.sqrt(np.mean(error ** 2, axis=0)).tolist(), images=per_image)


def _normal_validation(samples: dict, frames: list[_FrameSamples], model, settings) -> dict:
    prediction = predict_normal_model(
        samples['colors'], model, st=samples['st'], curvature_per_mm=samples['curvature'],
        curvature_valid_mask=np.ones(len(samples['colors']), bool),
    )
    predicted, predicted_valid = local_slopes_to_normals(
        prediction.slopes, samples['normals'], samples['frames'])
    actual, actual_valid = local_slopes_to_normals(samples['slopes'], samples['normals'], samples['frames'])
    if not (predicted_valid & actual_valid).all():
        raise ValueError('Validation contains invalid local normal frames')
    angles = np.degrees(np.arccos(np.clip(np.sum(predicted * actual, axis=-1), -1, 1)))
    out_of_range = np.any((samples['colors'] < model.color_min) | (samples['colors'] > model.color_max), axis=1)
    unsupported = ~prediction.supported_mask
    images = []
    for index, frame in enumerate(frames):
        selected = samples['groups'] == index
        rms = float(np.sqrt(np.mean(angles[selected] ** 2)))
        out_fraction = float(np.mean(out_of_range[selected]))
        unsupported_fraction = float(np.mean(unsupported[selected]))
        images.append(dict(
            image_key=frame.record['frame'].image_key, sample_count=int(selected.sum()),
            rms_degrees=rms, median_degrees=float(np.median(angles[selected])),
            p95_degrees=float(np.percentile(angles[selected], 95)),
            out_of_range_fraction=out_fraction, unsupported_fraction=unsupported_fraction,
            passed=(rms <= settings.validation_rms_degrees
                    and out_fraction <= settings.maximum_out_of_range_fraction
                    and unsupported_fraction <= settings.maximum_unsupported_fraction),
        ))
    return dict(
        passed=all(image['passed'] for image in images), sample_count=len(angles),
        rms_degrees=float(np.sqrt(np.mean(angles ** 2))),
        median_degrees=float(np.median(angles)), p95_degrees=float(np.percentile(angles, 95)),
        out_of_range_fraction=float(np.mean(out_of_range)), unsupported_fraction=float(np.mean(unsupported)),
        all_held_out_points_evaluated=True, images=images,
        thresholds=dict(rms_degrees=settings.validation_rms_degrees,
                        maximum_out_of_range_fraction=settings.maximum_out_of_range_fraction,
                        maximum_unsupported_fraction=settings.maximum_unsupported_fraction),
    )


def train_contact_models(
    records: list[dict],
    settings,
    lut_directory: str | Path,
    *,
    background_directory: str | Path,
    curvature_smoothing_half_width_mm: float = 0.0,
) -> dict:
    """Train background RGB and MLP8 slopes from accepted final-geometry records.

    Each record supplies current-frame RGB, an image-space contact contour,
    sphere slope labels and unsmoothed local curvature on the observation grid.
    Background RGB uses points outside the dilated contour and current-frame
    curvature smoothed by the specified half-width in mm. MLP8 receives signed
    linear RGB residuals, material st, and unsmoothed k11/k12/k22 in mm^-1.

    Whole-image holdout trains both models on training images only. After
    validation, both exported models are fitted on all records. The report
    records held-out quality even when a configured threshold is missed.
    """
    records = list(records)
    if len(records) < 3:
        raise ValueError('Whole-image validation and MLP early stopping require at least three images')
    if not np.isfinite(curvature_smoothing_half_width_mm) or curvature_smoothing_half_width_mm < 0:
        raise ValueError('Curvature smoothing half-width must be finite and nonnegative')
    first = records[0]['frame']
    keys = [record['frame'].image_key for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Calibration image keys must be unique')
    frames = []
    for record in records:
        frame = record['frame']
        if (frame.chart.chart_id != first.chart.chart_id
                or not np.array_equal(frame.template_xy_mm, first.template_xy_mm)
                or np.shape(frame.systems.observation.st) != np.shape(first.systems.observation.st)
                or not np.allclose(frame.systems.observation.st, first.systems.observation.st, rtol=0, atol=1e-7)):
            raise ValueError(f'{frame.image_key}: calibration frames must share one material chart and grid')
        frames.append(_frame_samples(record, settings, curvature_smoothing_half_width_mm))

    ordered = np.array(sorted(range(len(frames)), key=lambda i: keys[i]))
    np.random.default_rng(42).shuffle(ordered)
    count = min(len(frames) - 2, max(1, math.ceil(settings.validation_fraction * len(frames))))
    validation = [frames[i] for i in ordered[:count]]
    training = [frames[i] for i in ordered[count:]]
    half_width = float(curvature_smoothing_half_width_mm)

    background = _fit_background(training, half_width)
    background_validation = _background_validation(validation, background, half_width)
    training_samples = _normal_samples(training, background, half_width)
    validation_samples = _normal_samples(validation, background, half_width)
    candidate = _fit_normal(training_samples, settings)
    normal_validation = _normal_validation(validation_samples, validation, candidate, settings)
    del background, training_samples, validation_samples, candidate

    final_background = _fit_background(frames, half_width)
    samples = _normal_samples(frames, final_background, half_width)
    final_normal = _fit_normal(samples, settings)
    status = 'passed' if normal_validation['passed'] else 'failed'
    lut_output = Path(lut_directory).expanduser().resolve()
    background_output = Path(background_directory).expanduser().resolve()
    lut_output.mkdir(parents=True, exist_ok=True)
    background_output.mkdir(parents=True, exist_ok=True)
    background_file = background_output / 'background_mlp.npz'
    normal_file = lut_output / 'normal_lut.npz'
    report = dict(
        format_version=1, status=status, coordinate_chart_id=first.chart.chart_id,
        training_images=[frame.record['frame'].image_key for frame in training],
        validation_images=[frame.record['frame'].image_key for frame in validation],
        validation_split='whole_image', final_fit='all_images',
        contact_exclusion_margin_px=settings.background_contact_exclusion_margin_px,
        curvature_smoothing_half_width_mm=half_width,
        background=dict(model_file=str(background_file),
                        feature_order=['s', 't', 'k11_per_mm', 'k12_per_mm', 'k22_per_mm'],
                        widths=list(WIDTHS), fourier_frequencies=list(FOURIER_FREQUENCIES),
                        options=asdict(BACKGROUND_OPTIONS),
                        training_samples=sum(int(frame.background_valid.sum()) for frame in training),
                        all_samples=sum(int(frame.background_valid.sum()) for frame in frames),
                        validation=background_validation),
        normal=dict(model_file=str(normal_file), method='mlp8',
                    curvature='unsmoothed_local_tangent_frame',
                    all_samples=len(samples['colors']), validation=normal_validation,
                    best_epoch=final_normal.training_info['best_epoch']),
    )
    metadata = dict(
        method='mlp8', validation_status=status, coordinate_kind=COORDINATE_KIND,
        coordinate_chart_id=first.chart.chart_id, coordinate_chart_file='reference_coordinates.npz',
        domain_kind=DOMAIN_KIND, frame_convention=FRAME_CONVENTION,
        residual_domain='material_grid', color_space='linear_rgb',
        rgb_transfer_function=CAMERA_RGB_TRANSFER_FUNCTION, contact_offset_mm=settings.contact_offset_mm,
        reference_image=str(settings.reference_image) if settings.reference_image is not None else None,
        background_model=os.path.relpath(background_file, lut_output),
        background_curvature_smoothing_half_width_mm=half_width,
        normal_curvature_smoothing_half_width_mm=0.0,
    )
    final_background.save(background_file, first.chart.chart_id)
    replace(final_normal, metadata=metadata).save(normal_file)
    (lut_output / 'training.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
