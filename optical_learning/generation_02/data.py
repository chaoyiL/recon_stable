"""Sample fixed-surface background RGB from the annotated calibration frames."""

from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from calibrate.utils.norm_pipeline import (
    CalibrationSettings,
    FrameCalibrationData,
    load_frame_calibration_data,
)
from utils.image_io import load_linear_image
from utils.region_annotations import load_regions, region_masks
from utils.surface_curvature import compute_surface_curvature
from utils.surface_sampling import SampledField, sample_linear_rgb_to_surface


SESSION_GAP_SECONDS = 60
FEATURE_ORDER = ('s', 't', 'k11_per_mm', 'k12_per_mm', 'k22_per_mm')


@dataclass(frozen=True)
class FixedSurface:
    features: np.ndarray  # rows x columns x 5, using frozen st and curvature in mm^-1
    valid: np.ndarray
    chart_id: str

    @property
    def st(self) -> np.ndarray:
        return self.features[..., :2]


@dataclass(frozen=True)
class Observations:
    rgb: np.ndarray  # frames x rows x columns x 3, linear RGB
    valid: np.ndarray  # frames x rows x columns
    keys: list[str]
    sessions: np.ndarray  # one integer session per frame
    skipped: dict[str, str]


def load_fixed_surface(settings: CalibrationSettings, smoothing_half_width_mm: float = 0.0) -> FixedSurface:
    with np.load(settings.work_dir / 'reference_surface.npz', allow_pickle=False) as cache:
        curvature = compute_surface_curvature(
            cache['observation_xyz'], cache['observation_valid_mask'],
            cache['observation_coordinates'], cache['template_xy_mm'],
            smoothing_half_width_mm=smoothing_half_width_mm)
        features = np.concatenate((cache['observation_st'], curvature.values), axis=-1).astype(np.float32)
        return FixedSurface(features, curvature.valid, str(cache['coordinate_chart_id']))


def sample_frame(
    frame: FrameCalibrationData,
    image: np.ndarray,
    surface: FixedSurface,
    exclusion: np.ndarray,
) -> SampledField:
    grid = frame.systems.observation
    if (grid.chart_id != surface.chart_id or grid.st.shape != surface.st.shape
            or not np.allclose(grid.st, surface.st, rtol=0, atol=1e-7)):
        raise ValueError(f'{frame.image_key}: material grid differs from the fixed surface')
    if not np.array_equal(image, frame.linear_rgb):
        raise ValueError(f'{frame.image_key}: raw RGB differs from the geometry cache')
    pixel_valid = frame.surface_mask & ~frame.marker_image_mask & ~exclusion & (image < 1).all(axis=-1)
    sampled = sample_linear_rgb_to_surface(image, grid.uv, pixel_valid)
    return SampledField(sampled.values, sampled.valid & grid.valid_mask & surface.valid)


def load_observations(settings: CalibrationSettings, image_dir: Path, surface: FixedSurface,
                      *, contact_exclusion_margin_px: int | None = None) -> Observations:
    if contact_exclusion_margin_px is None:
        contact_exclusion_margin_px = settings.background_contact_exclusion_margin_px
    if (isinstance(contact_exclusion_margin_px, bool) or not isinstance(contact_exclusion_margin_px, int)
            or contact_exclusion_margin_px < 0):
        raise ValueError('The contact exclusion margin must be a nonnegative integer in pixels')
    manifest = json.loads((settings.work_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['coordinate_chart_id'] != surface.chart_id:
        raise ValueError('Manifest and reference surface have different coordinate charts')
    annotations = load_regions(settings.annotations_file, image_root=image_dir)
    rgb, valid, keys, sessions = [], [], [], []
    skipped = {}
    previous = None
    session = -1
    for row in sorted(manifest['images'], key=lambda row: row['key']):
        key, index = row['key'], row['index']
        annotation = annotations.get(key)
        if row['status'] == 'skip' or (annotation and annotation.status == 'skip'):
            skipped[key] = row['reason'] or 'annotation marked skip'
            continue
        if annotation is None or annotation.status != 'accepted' or annotation.semantics != 'contact_boundary':
            skipped[key] = 'accepted contact annotation required'
            continue

        stem = f'{index:06d}.npz'
        frame = load_frame_calibration_data(settings.work_dir / 'overall' / stem, settings)
        with np.load(settings.work_dir / 'frames' / stem, allow_pickle=False) as record:
            if (frame.image_key != key or frame.image_index != index or frame.geometry_stage != 'final'
                    or str(record['image_key']) != key or int(record['image_index']) != index
                    or str(record['coordinate_chart_id']) != surface.chart_id):
                raise ValueError(f'{key}: contact record and final geometry do not match')
            annotation = replace(annotation, contour_px=record['contact_boundary_px'].tolist())
        image = load_linear_image(image_dir / key)
        exclusion = region_masks(annotation, image.shape, outer_margin_px=contact_exclusion_margin_px)['exclusion']
        sampled = sample_frame(frame, image, surface, exclusion)
        if not sampled.valid.any():
            skipped[key] = 'no valid background points after masking'
            continue

        captured = datetime.strptime(Path(key).stem, 'capture_%Y%m%d_%H%M%S_%f')
        if previous is None or (captured - previous).total_seconds() > SESSION_GAP_SECONDS:
            session += 1
        previous = captured
        rgb.append(sampled.values)
        valid.append(sampled.valid)
        keys.append(key)
        sessions.append(session)

    if session < 1:
        raise ValueError('The whole-session holdout requires at least two capture sessions')
    return Observations(np.stack(rgb), np.stack(valid), keys, np.asarray(sessions), skipped)


def sample_reference(settings: CalibrationSettings, surface: FixedSurface) -> SampledField:
    if settings.reference_image is None:
        raise ValueError('The final evaluation requires a recorded reference image')
    frame = load_frame_calibration_data(settings.work_dir / 'reference_surface.npz', settings)
    image = load_linear_image(settings.reference_image)
    return sample_frame(frame, image, surface, np.zeros(image.shape[:2], dtype=bool))
