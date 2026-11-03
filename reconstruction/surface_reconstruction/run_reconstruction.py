"""Single entry: camera/video/still reconstruction without a reference frame."""

import os

# Tiny least-squares systems are much faster without oversubscribed BLAS threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import threading
import cv2
import numpy as np
try:
    from .surface import SurfaceModel, local_coordinates
    from .tracking import TrackingSeed, MarkerTracker, require_regular_template
    from .display import SurfaceView, marker_overlay
    from .markers import inspect_cross_candidates, find_inner_region
except ImportError:  # Keep ``python run_reconstruction.py`` working.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from surface import SurfaceModel, local_coordinates
    from tracking import TrackingSeed, MarkerTracker, require_regular_template
    from display import SurfaceView, marker_overlay
    from markers import inspect_cross_candidates, find_inner_region

try:
    from utils.color_difference import bgr8_to_linear_rgb
    from utils.image_io import (
        load_linear_image, linear_rgb_to_tracking_bgr8, validate_linear_rgb,
    )
except ModuleNotFoundError:  # Direct script execution puts this directory first.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.color_difference import bgr8_to_linear_rgb
    from utils.image_io import (
        load_linear_image, linear_rgb_to_tracking_bgr8, validate_linear_rgb,
    )

ROOT = Path(__file__).resolve().parent


def load_inputs(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    if len(config["image_size"]) != 2 or any(
        not isinstance(v, int) or v <= 0 for v in config["image_size"]
    ):
        raise ValueError("image_size must contain positive integer width and height")
    base = Path(config_path).resolve().parent
    with np.load(base / config["camera_file"], allow_pickle=False) as data:
        camera = data["camera_matrix"].astype(float)
        distortion = data["dist_coeffs"].astype(float)
    if (
        camera.shape != (3, 3)
        or not np.isfinite(camera).all()
        or not np.isfinite(distortion).all()
        or min(camera[0, 0], camera[1, 1]) <= 0
    ):
        raise ValueError("Invalid camera calibration")
    data = np.genfromtxt(
        base / config["template_file"], delimiter=",", names=True, encoding="utf-8-sig"
    )
    ids = data["id"].astype(int)
    xy = np.column_stack((data["x_mm"], data["y_mm"]))
    require_regular_template(ids, xy)
    if not np.isfinite(xy).all():
        raise ValueError("Invalid physical template")
    return config, ids, xy, camera, distortion


class SurfaceReconstructionSession:
    """Persistent IDs; observation acquisition is independent of shape fitting."""

    def __init__(self, config_path=ROOT / 'config.json', row_direction='auto'):
        if row_direction not in {'auto', 'down', 'up', 'left', 'right'}:
            raise ValueError('Invalid row_direction')
        self.config_path = Path(config_path).expanduser().resolve()
        self.config, self.ids, self.xy, self.camera, self.distortion = load_inputs(self.config_path)
        self.expected_size = tuple(self.config['image_size'])
        self.model = SurfaceModel(self.xy, self.camera, self.distortion, self.config.get('solver'))
        self.row_direction = row_direction
        self.tracker = None
        self.segment = 0
        self.lost_frames = 0
        self.shape_failed_frames = 0

    def reset(self):
        self.tracker = None
        self.lost_frames = self.shape_failed_frames = 0
        self.model.reset()

    def fit_observations(self, tracking, *, exclusion_mask=None, prior_coefficients=None):
        pixels = np.asarray(tracking['pixels'], np.float64)
        fit_valid = np.asarray(tracking['valid'], bool).copy() & np.isfinite(pixels).all(1)
        if exclusion_mask is not None:
            mask = np.asarray(exclusion_mask, bool)
            if mask.shape != self.expected_size[::-1]:
                raise ValueError('Fit exclusion mask shape mismatch')
            ij = np.rint(np.nan_to_num(pixels)).astype(int)
            inside = (ij >= 0).all(1) & (ij[:, 0] < mask.shape[1]) & (ij[:, 1] < mask.shape[0])
            fit_valid[inside] &= ~mask[ij[inside, 1], ij[inside, 0]]
        points = coefficients = None
        error = ''
        if prior_coefficients is not None:
            self.model.previous = np.array(prior_coefficients, np.float64, copy=True)
            self.model.alternative = None
        try:
            points, coefficients, info = self.model.fit(pixels, fit_valid, tracking['confidence'])
            fit_valid[info.get('rejected_observation_ids', [])] = False
            self.shape_failed_frames = 0
        except (ValueError, RuntimeError, cv2.error) as exc:
            self.shape_failed_frames += 1
            info = {'status': 'insufficient_observations' if fit_valid.sum() < 32 else 'fit_rejected'}
            error = str(exc)
        return dict(tracking=tracking, marker_fit_valid=fit_valid, points=points,
                    coefficients=coefficients, info=info, error=error, segment=self.segment)

    def process(self, frame_linear_rgb, *, fit=True):
        """Process one float32 linear-RGB frame; adapt only the tracking input."""
        frame_linear_rgb = validate_linear_rgb(frame_linear_rgb)
        if (frame_linear_rgb.shape[1], frame_linear_rgb.shape[0]) != self.expected_size:
            raise ValueError(
                f'Calibration is for {self.expected_size}; '
                f'input is {frame_linear_rgb.shape[1::-1]}')
        tracking_bgr = linear_rgb_to_tracking_bgr8(frame_linear_rgb)
        tracking = dict(pixels=np.full((len(self.ids), 2), np.nan),
                        valid=np.zeros(len(self.ids), bool), confidence=np.zeros(len(self.ids)))
        region = None
        try:
            gray = cv2.cvtColor(tracking_bgr, cv2.COLOR_BGR2GRAY)
            region = find_inner_region(gray)
            if self.tracker is None:
                seed, _, _ = TrackingSeed.build(
                    frame_linear_rgb, self.ids, self.xy, self.row_direction,
                    region=region, tracking_bgr=tracking_bgr)
                self.tracker = MarkerTracker(seed)
                self.model.reset()
                self.segment += 1
                self.lost_frames = self.shape_failed_frames = 0
                tracking = dict(pixels=seed.pixels.copy(), valid=np.ones(len(self.ids), bool),
                                confidence=np.ones(len(self.ids)))
            else:
                tracking = self.tracker.update(
                    frame_linear_rgb, tracking_bgr=tracking_bgr)
                # Invalid slots still locate conservative photometric holes.
                tracking['pixels'] = np.where(np.isfinite(tracking['pixels']),
                                             tracking['pixels'], self.tracker.positions)
            self.lost_frames = self.lost_frames+1 if tracking['valid'].sum() < 12 else 0
            if fit:
                result = self.fit_observations(tracking)
                rejected = result['info'].get('rejected_observation_ids', [])
                if rejected:
                    tracking['valid'][rejected] = False
                    tracking['confidence'][rejected] = 0
                    self.tracker.valid[rejected] = False
                    projected = cv2.projectPoints(result['points'], np.zeros(3), np.zeros(3),
                                                  self.camera, self.distortion)[0].reshape(-1, 2)
                    self.tracker.positions[rejected] = projected[rejected]
                if self.shape_failed_frames >= 3:
                    self.reset()
            else:
                result = dict(tracking=tracking, points=None, coefficients=None,
                              info={'status': 'tracked'}, error='', segment=self.segment)
            if self.lost_frames >= 5:
                self.reset()
        except (ValueError, RuntimeError, cv2.error) as exc:
            result = dict(tracking=tracking, points=None, coefficients=None,
                          info={'status': 'region_not_found' if region is None else 'tracking_failed'},
                          error=str(exc), segment=self.segment)
            self.reset()
        result['inner_region'] = region
        return result


class SurfaceReconstructionProvider:
    """Sample every frame through one frozen reference coordinate chart."""

    def __init__(self, config_path=ROOT / 'config.json', row_direction='auto'):
        self.session = SurfaceReconstructionSession(config_path, row_direction)
        self.last_frame_result = None
        self.coordinate_chart = None
        self.coordinate_chart_file = None

    def bind_coordinate_chart(self, chart):
        try:
            from .surface import BicubicSurfaceAdapter
        except ImportError:
            from surface import BicubicSurfaceAdapter
        bounds = BicubicSurfaceAdapter(self.session.model, np.zeros((16, 3))).parameter_bounds
        if not np.allclose(bounds, chart.parameter_bounds, atol=1e-12, rtol=0):
            raise ValueError('Coordinate chart does not match physical template')
        self.coordinate_chart = chart

    def reset(self, *, clear_coordinate_chart=False):
        self.session.reset()
        self.last_frame_result = None
        if clear_coordinate_chart:
            self.coordinate_chart = None
            self.coordinate_chart_file = None

    def track_frame(self, frame_linear_rgb):
        self.last_frame_result = self.session.process(
            np.asarray(frame_linear_rgb), fit=False)
        return self.last_frame_result

    def _surface_mask(self, explicit):
        region = None if self.last_frame_result is None else self.last_frame_result.get('inner_region')
        mask = None if region is None else region.mask
        if explicit is not None:
            explicit = np.asarray(explicit, bool)
            mask = explicit if mask is None else mask & explicit
        return mask

    def sample_coefficients(self, coefficients, specs, surface_mask=None):
        from utils.reconstruction_interface import MaterialCoordinateMap, MaterialCoordinateSystems
        from utils.surface_coordinates import build_arclength_chart, sample_surface_grid
        from utils.surface_projection import CameraModel, project_points
        from utils.surface_sampling import sample_mask_to_surface
        try:
            from .surface import BicubicSurfaceAdapter
        except ImportError:
            from surface import BicubicSurfaceAdapter
        surface = BicubicSurfaceAdapter(self.session.model, coefficients)
        if self.coordinate_chart is None:
            self.bind_coordinate_chart(build_arclength_chart(surface))
        chart = self.coordinate_chart
        mask = self._surface_mask(surface_mask)
        camera = CameraModel(self.session.camera, self.session.distortion,
                             self.session.expected_size[::-1])
        maps = {}
        for name in ('geometry', 'lightfield', 'observation'):
            spec = getattr(specs, name)
            grid = sample_surface_grid(surface, chart, spec.rows, spec.columns)
            uv, projected = project_points(grid.xyz, camera)
            valid = grid.coordinate_valid & projected
            if mask is not None:
                valid &= sample_mask_to_surface(mask, uv)
            maps[name] = MaterialCoordinateMap(grid.xyz, uv, grid.parameters, valid,
                                               grid.st, chart.coordinate_kind, chart.chart_id)
        result = MaterialCoordinateSystems(**maps)
        result.validate_against(specs)
        return result

    def fit_tracked_frame(self, specs, surface_mask=None, fit_exclusion_mask=None,
                          prior_coefficients=None):
        if self.last_frame_result is None:
            raise RuntimeError('No cached tracking observations')
        cached = self.last_frame_result
        result = self.session.fit_observations(cached['tracking'], exclusion_mask=fit_exclusion_mask,
                                               prior_coefficients=prior_coefficients)
        result['inner_region'] = cached.get('inner_region')
        self.last_frame_result = result
        if result['coefficients'] is None or result['info']['status'] == 'ambiguous':
            raise RuntimeError(result['error'] or result['info']['status'])
        return self.sample_coefficients(result['coefficients'], specs, surface_mask)

    def reconstruct(self, frame_linear_rgb, specs, surface_mask=None):
        result = self.session.process(np.asarray(frame_linear_rgb))
        self.last_frame_result = result
        if result['coefficients'] is None or result['info']['status'] == 'ambiguous':
            raise RuntimeError(f"Marker surface reconstruction failed: {result['error'] or result['info']['status']}")
        return self.sample_coefficients(result['coefficients'], specs, surface_mask)


class LatestCamera:
    """Drain camera continuously; processing always takes the latest available frame."""

    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.stopped = False
        self.done = False
        self.latest = None
        self.index = 0
        self.consumed = 0
        self.worker = threading.Thread(target=self._read, daemon=True)
        self.worker.start()

    def _read(self):
        try:
            while not self.stopped:
                ok, encoded_bgr = self.capture.read()
                if not ok:
                    break
                frame_linear_rgb = bgr8_to_linear_rgb(encoded_bgr)
                with self.condition:
                    self.index += 1
                    self.latest = (self.index, time.perf_counter(), frame_linear_rgb)
                    self.condition.notify_all()
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def read(self):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.done or (self.latest is not None and self.latest[0] > self.consumed),
                timeout=5.0,
            )
            if not ready or self.latest is None or self.latest[0] <= self.consumed:
                return None
            packet = self.latest
            self.consumed = packet[0]
            return packet

    def close(self):
        self.stopped = True
        self.worker.join(timeout=1.0)
        self.capture.release()
        self.worker.join(timeout=1.0)


def load_linear_recording(directory):
    """Read a capture.py frame manifest and return its nominal FPS and frame list."""
    directory = Path(directory).expanduser()
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read recording manifest in {directory}: {exc}") from exc
    if (not isinstance(manifest, dict)
            or manifest.get("format_version") != 1
            or manifest.get("color_space") != "linear_rgb"
            or manifest.get("rgb_transfer_function") != "bt709"):
        raise ValueError(f"Unsupported linear recording manifest: {directory}")
    try:
        fps = float(manifest["fps"])
        records = manifest["frames"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid recording manifest: {directory}") from exc
    if not np.isfinite(fps) or fps <= 0 or not isinstance(records, list):
        raise ValueError(f"Invalid recording FPS or frame table: {directory}")
    frames = []
    previous_index = -1
    previous_timestamp = -1.0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid frame record in {directory}")
        try:
            index = record["index"]
            timestamp = float(record["timestamp_s"])
            filename = record["file"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid frame record in {directory}") from exc
        if (not isinstance(index, int) or isinstance(index, bool) or index <= previous_index
                or not np.isfinite(timestamp) or timestamp < previous_timestamp
                or not isinstance(filename, str) or Path(filename).name != filename
                or Path(filename).suffix.lower() != ".npz"):
            raise ValueError(f"Invalid frame record in {directory}")
        frames.append((index, timestamp, directory / filename))
        previous_index, previous_timestamp = index, timestamp
    return fps, frames


def unique_output(base):
    """Keep previous results; never mix a new run with stale successful estimates."""
    base = Path(base)
    if base.exists():
        base = base.with_name(base.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    base.mkdir(parents=True)
    return base


def make_writer(path, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise OSError(f"Cannot encode video: {path}")
    return writer


def save_points(writer, index, timestamp, segment, result, points, xy):
    local = (
        local_coordinates(points, xy)[0] if points is not None else np.full((len(xy), 3), np.nan)
    )
    xyz = points if points is not None else local
    for k in range(len(xy)):
        observed = bool(result["valid"][k])
        source = (
            ("observed_fit" if observed else "model_inferred")
            if points is not None
            else "unavailable"
        )
        row = [
            index,
            timestamp,
            segment,
            k,
            int(observed),
            source,
            float(result["confidence"][k]),
            *result["pixels"][k],
            *xyz[k],
            *local[k],
        ]
        writer.writerow(
            ["" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in row]
        )


def run(args):
    session = SurfaceReconstructionSession(args.config, args.row_direction)
    config, ids, xy = session.config, session.ids, session.xy
    source = (
        args.image
        if args.image is not None
        else (args.video if args.video is not None else str(config["camera_index"]))
    )
    is_image = args.image is not None
    recording_dir = (Path(source).expanduser()
                     if not is_image and not str(source).isdigit()
                     and Path(source).expanduser().is_dir() else None)
    is_camera = not is_image and recording_dir is None and str(source).isdigit()
    expected = session.expected_size
    model = session.model
    viewer = SurfaceView(model, *expected)
    basename = Path(source).stem if not is_camera else "camera_" + str(source)
    output = unique_output(args.out or ROOT / "output" / basename)
    marker_dir = ROOT / "marker_detection" / output.name
    if not is_image and not args.no_record:
        marker_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(
        source=str(source),
        frames=0,
        reconstructed_frames=0,
        fully_observed_frames=0,
        ambiguous_frames=0,
        failed_frames=0,
        segments=0,
        complete=False,
        solver="bicubic differential isometry + image observations + pose-aligned temporal prior",
        reference_required=False,
        image_size=list(expected),
        settings=config,
        coordinate_system="camera XYZ mm; local shape uses a proper best-fit rigid alignment to the template",
        id_convention="row-major, per segment; identical marks have an unresolved physical 180-degree symmetry",
        accuracy_note="Reprojection and metric consistency are diagnostics, not measured 3D accuracy.",
        timing_note="Wall throughput includes capture, tracking, solve, render, encoding, CSV and UI. Not sensor-to-display latency.",
        camera_frames_dropped=0,
    )
    capture = None
    latest = None
    video_writer = None
    marker_writer = None
    segment = 0
    fps_smooth = 0.0
    count = 0
    metrics = []
    start = time.perf_counter()
    last_index = 0
    try:
        if is_image:
            still = load_linear_image(source)
            fps = 1.0
        elif recording_dir is not None:
            fps, recording_frames = load_linear_recording(recording_dir)
            summary["input_fps"] = fps
        else:
            capture = cv2.VideoCapture(int(source) if is_camera else str(source))
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {source}")
            if is_camera:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, expected[0])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, expected[1])
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            summary["input_fps"] = fps
            if is_camera:
                latest = LatestCamera(capture)
        with (output / ("reconstructed_points.csv" if is_image else "frame_points.csv")).open(
            "w", newline="", encoding="utf-8"
        ) as pf, (output / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as mf:
            pw = csv.writer(pf)
            mw = csv.writer(mf)
            pw.writerow(
                [
                    "frame",
                    "time_s",
                    "segment",
                    "id",
                    "observed",
                    "point_source",
                    "confidence",
                    "u_px",
                    "v_px",
                    "X_mm",
                    "Y_mm",
                    "Z_mm",
                    "X_local_mm",
                    "Y_local_mm",
                    "Z_local_mm",
                ]
            )
            mw.writerow(
                [
                    "frame",
                    "time_s",
                    "segment",
                    "observed",
                    "status",
                    "reprojection_rms_px",
                    "metric_max_error",
                    "processing_ms",
                    "error",
                ]
            )
            while args.max_frames is None or count < args.max_frames:
                frame_start = time.perf_counter()
                if is_image:
                    if count:
                        break
                    frame = still
                    timestamp = 0.0
                    frame_index = 0
                elif recording_dir is not None:
                    if count >= len(recording_frames):
                        break
                    frame_index, timestamp, frame_path = recording_frames[count]
                    frame = load_linear_image(frame_path)
                elif latest is not None:
                    packet = latest.read()
                    if packet is None:
                        break
                    frame_index, acquired, frame = packet
                    timestamp = acquired - start
                    summary["camera_frames_dropped"] += max(0, frame_index - last_index - 1)
                    last_index = frame_index
                else:
                    ok, encoded_bgr = capture.read()
                    if not ok:
                        break
                    frame = bgr8_to_linear_rgb(encoded_bgr)
                    frame_index = count
                    timestamp = count / fps
                frame_result = session.process(frame)
                result = frame_result["tracking"]
                info = frame_result["info"]
                error = frame_result["error"]
                points = frame_result["points"]
                coeff = frame_result["coefficients"]
                segment = frame_result["segment"]
                overlay = (
                    inspect_cross_candidates(frame, error)[0]
                    if error
                    else marker_overlay(
                        frame,
                        result["pixels"],
                        result["valid"],
                        info["status"],
                        segment,
                    )
                )
                shape = viewer.render(coeff, info, fps_smooth)
                combined = np.hstack((overlay, shape))
                save_points(pw, frame_index, timestamp, segment, result, points, xy)
                if is_image:
                    if coeff is not None:
                        viewer.save_obj(output / "surface.obj", coeff)
                    summary["image_diagnostics"] = dict(info, error=error)
                elif not args.no_record:
                    if video_writer is None:
                        video_writer = make_writer(
                            output / "reconstruction.avi", fps, (expected[0] * 2, expected[1])
                        )
                        marker_writer = make_writer(
                            marker_dir / "detected_markers.avi", fps, expected
                        )
                        summary["recording_note"] = (
                            "Video encodes processed frames at input FPS; use time_s in CSV for timing, especially for live capture with dropped frames."
                        )
                    video_writer.write(combined)
                    marker_writer.write(overlay)
                key = -1
                if args.preview or (is_camera and not args.no_preview):
                    cv2.imshow("Surface reconstruction | Esc: exit | R: reacquire", combined)
                    key = cv2.waitKey(1) & 255
                elapsed = time.perf_counter() - frame_start
                fps_smooth = (
                    1 / max(elapsed, 1e-9)
                    if count == 0
                    else 0.9 * fps_smooth + 0.1 / max(elapsed, 1e-9)
                )
                mw.writerow(
                    [
                        frame_index,
                        timestamp,
                        segment,
                        int(result["valid"].sum()),
                        info["status"],
                        info.get("reprojection_rms_px", ""),
                        info.get("metric_max_error", ""),
                        elapsed * 1000,
                        error,
                    ]
                )
                if count % 30 == 0:
                    pf.flush()
                    mf.flush()
                count += 1
                summary["frames"] = count
                summary["segments"] = segment
                summary["reconstructed_frames"] += int(coeff is not None)
                summary["fully_observed_frames"] += int(result["valid"].all())
                summary["ambiguous_frames"] += int(info["status"] == "ambiguous")
                summary["failed_frames"] += int(coeff is None)
                if coeff is not None:
                    metrics.append([info["reprojection_rms_px"], info["metric_max_error"]])
                if count % 120 == 0:
                    print(
                        f"{count} frames | 3D {summary['reconstructed_frames']} | {fps_smooth:.1f} FPS | {info['status']}",
                        flush=True,
                    )
                if key == 27:
                    break
                if key in (ord("r"), ord("R")):
                    session.reset()
            if count == 0:
                raise ValueError("No decodable frames")
            summary["complete"] = True
    except Exception as exc:
        summary["error"] = str(exc)
        raise
    finally:
        if latest is not None:
            latest.close()
        elif capture is not None:
            capture.release()
        if video_writer is not None:
            video_writer.release()
        if marker_writer is not None:
            marker_writer.release()
        if args.preview or (is_camera and not args.no_preview):
            cv2.destroyAllWindows()
        duration = time.perf_counter() - start
        summary["wall_seconds"] = duration
        summary["wall_throughput_fps"] = count / max(duration, 1e-9)
        if metrics:
            summary["median_reprojection_rms_px"] = float(np.median(np.array(metrics)[:, 0]))
            summary["median_metric_max_error"] = float(np.median(np.array(metrics)[:, 1]))
        (output / "diagnostics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "frames",
                    "reconstructed_frames",
                    "fully_observed_frames",
                    "ambiguous_frames",
                    "segments",
                    "wall_throughput_fps",
                )
            },
            indent=2,
        )
    )
    print(f"Results: {output}\nMarkers: {marker_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="Video file, linear frame directory, or camera index (default from config)")
    source.add_argument("--image", help="Single image; no reference image required")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-frames", type=int)
    ui = parser.add_mutually_exclusive_group()
    ui.add_argument("--preview", action="store_true", help="Show marker and live 3D views")
    ui.add_argument("--no-preview", action="store_true", help="Disable camera display")
    parser.add_argument(
        "--no-record", action="store_true", help="Save CSV/diagnostics but no videos"
    )
    parser.add_argument(
        "--row-direction", choices=["auto", "down", "up", "left", "right"], default="auto"
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
