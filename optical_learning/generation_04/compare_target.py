"""Apply the saved fourth-generation backgrounds and Tables to one target image."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from optical_learning.generation_02.model import BackgroundMLP
from reconstruction.get_surface import build_reconstruction_provider
from reconstruction.local_recon import (
    load_local_reconstruction_settings, load_reconstruction_arrays, reconstruct_from_input,
)
from reconstruction.pipeline import (
    DEFAULT_CONFIG_PATH, _fit_frame, load_sampling_grid, make_reconstruction_input, sample_material_map,
)
from reconstruction.surface_reconstruction.markers import build_marker_image_mask, marker_reference_sizes
from reconstruction.surface_reconstruction.surface import BicubicSurfaceAdapter
from reconstruction.viewer import view_reconstruction
from utils.calibration_store import lut_directory
from utils.color_difference import signed_color_difference
from utils.config import load_config
from utils.image_io import linear_rgb_to_tracking_bgr8, load_linear_image
from utils.normal_models import load_normal_model
from utils.surface_curvature import compute_surface_curvature
from utils.surface_domain import extend_surface
from utils.surface_geometry import OffsetSurface
from utils.surface_projection import CameraModel
from utils.surface_sampling import rasterize_surface_grid, sample_linear_rgb_to_surface
from experiments.indentation_analysis.depth_fitting import depth_scan_options

from .contact_diagnostics import ANCHOR_RADIUS_PX, DEPTH_MM, PAIRS, THRESHOLD_MM, distance_rows, scan_contacts, write_rows
from .plateau import EXTERIOR_GUARD_PX, FACE_INSET_PX, PLATEAU_UV, face_metrics, plateau_masks


ROOT = DEFAULT_CONFIG_PATH.parent
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / 'runs/paired_curvature/per_frame'
ANCHORS_UV = ((226, 209), (312, 209), (402, 208))
METHODS = ('reference', 'learned')


def matched_inputs(image_path: Path, prior: dict, reference_table, model: BackgroundMLP):
    reference_path = ROOT / prior['reference_image']
    image, reference = load_linear_image(image_path), load_linear_image(reference_path)
    provider = build_reconstruction_provider(DEFAULT_CONFIG_PATH, contact_surface=True)
    session, offset = provider.session, provider.contact_offset_mm
    if offset != prior['contact_offset_mm']:
        raise ValueError('Contact-surface offset differs from the model calibration')
    fixed_fit = _fit_frame(session, reference, 'reference image')
    session.model.reset()  # Keep the reference marker identities for the current fit.
    current_fit = _fit_frame(session, image, image_path.name)
    camera = CameraModel(session.camera, session.distortion, reference.shape[:2])
    grid_path = max((lut_directory(DEFAULT_CONFIG_PATH) / 'table').glob('*/lut_material_grid.npz'))
    grid = load_sampling_grid(grid_path, reference_table, camera, tuple(prior['grid_shape']),
                              template_xy_mm=session.xy, contact_offset_mm=offset)

    fits = (fixed_fit, current_fit)
    image_masks = tuple(fit['inner_region'].mask for fit in fits)
    maps = tuple(sample_material_map(OffsetSurface(extend_surface(
        BicubicSurfaceAdapter(session.model, fit['coefficients']), grid.parameter_bounds), offset),
        camera, grid, mask) for fit, mask in zip(fits, image_masks, strict=True))
    tracking_rgb = tuple(linear_rgb_to_tracking_bgr8(frame) for frame in (reference, image))
    sizes = marker_reference_sizes(reference, fixed_fit['tracking']['pixels'], tracking_bgr=tracking_rgb[0])
    marker_masks = tuple(build_marker_image_mask(frame, fit['tracking']['pixels'], fit['tracking']['valid'], sizes,
                                                tracking_bgr=bgr)
                         for frame, fit, bgr in zip((reference, image), fits, tracking_rgb, strict=True))
    fixed, current = maps
    reference_mask, image_mask = image_masks
    reference_marker, image_marker = marker_masks
    data = make_reconstruction_input(current, fixed, image, reference, image_mask, reference_mask,
                                     image_marker, reference_marker, offset)
    _, fixed_delta, _, fixed_valid, _ = load_reconstruction_arrays(data, reference_table.metadata)
    half_width = model.curvature_smoothing_half_width_mm
    curvature = compute_surface_curvature(current.xyz, current.valid_mask, current.coordinates, session.xy,
                                          smoothing_half_width_mm=half_width)
    prediction = model.predict(np.concatenate((current.st, curvature.values), axis=-1),
                               curvature_smoothing_half_width_mm=half_width)
    observed = sample_linear_rgb_to_surface(image, current.uv, image_mask & ~image_marker & np.all(image < 1, axis=-1))
    learned_delta, shared = signed_color_difference(observed.values, prediction,
                                                   fixed_valid & curvature.valid & observed.valid)
    if not shared.any():
        raise ValueError('The two backgrounds have no common photometric support')
    data['photometric_valid_mask'] = shared
    data.update(curvature.npz_fields())
    return data, {'reference': fixed_delta, 'learned': learned_delta}, float(current_fit['info']['reprojection_rms_px'])


def plot_displacement(path: Path, results: dict, uv: np.ndarray, photo: np.ndarray,
                      image: np.ndarray, anchors: np.ndarray, outline: np.ndarray | None) -> None:
    fields = np.stack([results[m].displacement for m in METHODS], axis=-1)
    rendered = rasterize_surface_grid(fields, uv, results['reference'].valid_mask, image.shape[:2])
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.6), layout='constrained', sharex=True, sharey=True)
    limit = float(np.abs(fields[results['reference'].valid_mask]).max())
    left, top = np.floor(uv[photo].min(axis=0)) - 5
    right, bottom = np.ceil(uv[photo].max(axis=0)) + 5
    axes[0].imshow(linear_rgb_to_tracking_bgr8(image)[..., ::-1])
    axes[0].set_title('Observed contact')
    for index, (ax, title) in enumerate(zip(axes[1:], ('Captured reference + Table', 'Learned background + Table'))):
        panel = ax.imshow(np.ma.array(rendered.values[..., index], mask=~rendered.valid), cmap='coolwarm',
                          vmin=-limit, vmax=limit)
        ax.set_title(title)
    for ax in axes:
        ax.set(xlabel='Image u (px)', xlim=(left, right), ylim=(bottom, top))
        if outline is not None:
            ax.add_patch(plt.Polygon(outline, fill=False, color='#fbce55', lw=1.35, linestyle='--'))
        for contact, (u, v) in enumerate(anchors, 1):
            ax.text(u, v - 18, str(contact), ha='center', color='white' if ax is axes[0] else '#252525')
    axes[0].set_ylabel('Image v (px)')
    figure.colorbar(panel, ax=axes[1:], label='Signed normal displacement (mm)')
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_scan(path: Path, scans: list[dict], positions: list[dict]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11, 5.3), layout='constrained')
    for index, (first, second) in enumerate(PAIRS):
        top, bottom = axes[:, index]
        for method, caption in zip(METHODS, ('Captured reference', 'Learned background')):
            rows = [row for row in scans if row['method'] == method and row['contact'] == index + 1
                    and row['radial_rmse_mm'] is not None]
            top.plot([row['depth_mm'] for row in rows], [row['radial_rmse_mm'] for row in rows], label=caption)
            rows = [row for row in positions if row['method'] == method and
                    (row['contact_a'], row['contact_b']) == (first, second)]
            bottom.plot([row['depth_mm'] for row in rows], [row['error_mm'] for row in rows])
        top.set_title(f'Ball {index + 1}')
        bottom.set_title(f'Centers {first}–{second}')
        bottom.axhline(0, linewidth=.7, color='.55')
        for ax in (top, bottom):
            ax.axvline(DEPTH_MM, linewidth=.7, linestyle='--', color='.6')
            ax.set_xlabel('Fit depth (mm)')
    axes[0, 0].set_ylabel('Radial RMSE (mm)')
    axes[1, 0].set_ylabel('Distance error vs scan (mm)')
    axes[0, 0].legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def rms(values: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(values.astype(np.float64)**2))) if values.size else None


def diagnose(output: Path, results: dict, uv: np.ndarray, anchors: np.ndarray,
             radii: tuple[float, ...], options, scan_distances: dict, *, anchor_window: bool = False) -> dict:
    report, scans, positions, fits = {}, [], [], []
    for method in METHODS:
        contacts, rows, centers, candidates, selected = scan_contacts(
            results[method], uv, anchors, radii, options, method, anchor_window=anchor_window)
        scans.extend(rows)
        positions.extend(distance_rows(method, centers, scan_distances))
        fits.extend(contacts)
        directory = output / method
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'fit_selection.npz', uv=uv, candidate_masks=candidates, fit_masks=selected)
        fixed_centers = {row['contact']: np.array([row[f'center_{axis}_mm'] for axis in 'xyz'])
                         for row in contacts if row['status'] == 'ok'}
        metrics = {}
        for first, second in PAIRS:
            if first in fixed_centers and second in fixed_centers:
                key = f'{first}-{second}'
                distance = float(np.linalg.norm(fixed_centers[first] - fixed_centers[second]))
                metrics[key] = dict(distance_mm=distance, error_mm=distance - scan_distances[key])
        report[method] = dict(spheres=contacts, center_distances_at_fixed_depth=metrics,
            distance_rmse_mm=rms(np.asarray([pair['error_mm'] for pair in metrics.values()])) if len(metrics) == 3 else None)
    write_rows(output / 'sphere_fits.csv', fits)
    write_rows(output / 'depth_scan.csv', scans, ('method', 'contact', 'radius_mm', 'depth_mm', 'minimum_indentation_mm',
               'sample_count', 'radial_rmse_mm', 'center_x_mm', 'center_y_mm', 'center_z_mm', 'status'))
    write_rows(output / 'position_scan.csv', positions,
               ('method', 'depth_mm', 'contact_a', 'contact_b', 'distance_mm', 'scan_distance_mm', 'error_mm'))
    plot_scan(output / 'depth_scan.png', scans, positions)
    return report


def run(image_path: Path, output: Path, anchors: np.ndarray, *, plateau_outline: np.ndarray | None = None) -> dict:
    prior_dir = MODEL_DIR / '3b2'
    prior = json.loads((prior_dir / 'comparison.json').read_text())
    model_file = MODEL_DIR / 'model.npz'
    model = BackgroundMLP.load(model_file)
    files = {method: prior_dir / method / 'table.npz' for method in METHODS}
    tables = {method: load_normal_model(path) for method, path in files.items()}
    data, residuals, reprojection_rms = matched_inputs(image_path, prior, tables['reference'], model)
    with np.load(model_file, allow_pickle=False) as saved:
        if str(saved['chart_id']) != data['coordinate_chart_id']:
            raise ValueError('The saved light field and Table use different material coordinate charts')
    uv, photo = data['uv'], data['photometric_valid_mask']
    anchor_distance = np.linalg.norm(uv[..., None, :] - anchors, axis=-1).min(axis=-1)
    if plateau_outline is None:
        outside = photo & (anchor_distance >= 55.)
    else:
        face, outside = plateau_masks(uv, photo, plateau_outline, anchor_distance >= ANCHOR_RADIUS_PX)
        if face.sum() < 3:
            raise ValueError('The plateau audit requires at least three valid face points')
    right = photo & (uv[..., 0] > 445.)
    right_outside = right & outside
    config = load_config(DEFAULT_CONFIG_PATH)
    options = depth_scan_options(config)
    radii = tuple(float(value) for value in config['indentation_analysis']['sphere_radii_mm'])
    if radii != tuple(sphere['radius_mm'] for sphere in prior['scan_spheres']):
        raise ValueError('Configured sphere radii differ from the scan fit')
    print(f'{image_path.stem}: {int(photo.sum()):,} common photometric points.', flush=True)
    report = dict(image=str(image_path.relative_to(ROOT)), grid_shape=list(photo.shape),
                  photometric_points=int(photo.sum()), noncontact_points=int(outside.sum()),
                  right_photometric_points=int(right.sum()), right_noncontact_points=int(right_outside.sum()),
                  marker_reprojection_rms_px=reprojection_rms,
                  contact_offset_mm=data['contact_offset_mm'], coordinate_chart_id=data['coordinate_chart_id'],
                  curvature_smoothing_half_width_mm=model.curvature_smoothing_half_width_mm,
                  model=str(model_file.relative_to(ROOT)), scan=prior['scan'],
                  scan_fit_archive=str((prior_dir / 'comparison.json').relative_to(ROOT)),
                  scan_center_distances_mm=prior['scan_center_distances_mm'], comparison_depth_mm=DEPTH_MM,
                  depth_scan=asdict(options),
                  anchors_uv=anchors.tolist(), background_difference_rms=rms((residuals['reference'] - residuals['learned'])[photo]),
                  methods={})
    if plateau_outline is not None:
        report['plateau'] = dict(outline_uv=plateau_outline.tolist(), face_inset_px=FACE_INSET_PX,
                                 exterior_guard_px=EXTERIOR_GUARD_PX, cap_exclusion_radius_px=ANCHOR_RADIUS_PX,
                                 face_points=int(face.sum()))
        np.savez_compressed(output / 'plateau_masks.npz', uv=uv, face=face, exterior=outside, right_exterior=right_outside)
    results = {}
    for method in METHODS:
        settings = load_local_reconstruction_settings(DEFAULT_CONFIG_PATH, input_override=str(image_path),
                                                      calibration_override=str(files[method]))
        result = reconstruct_from_input(dict(data, color_residual_grid=residuals[method]), tables[method], settings)
        results[method] = result
        directory = output / method
        directory.mkdir(exist_ok=True)
        path = result.save(directory / f'{image_path.stem}.npz')
        view_reconstruction(path, image_output=path.with_suffix('.png'), show=False)
        report['methods'][method] = dict(table=str(files[method].relative_to(ROOT)),
            noncontact_rgb_rmse=rms(residuals[method][outside]),
            right_noncontact_rgb_rmse=rms(residuals[method][right_outside]),
            right_photometric_points_over_0p2_mm=int((right & (result.displacement < -THRESHOLD_MM)).sum()),
            right_noncontact_points_over_0p2_mm=int((right_outside & (result.displacement < -THRESHOLD_MM)).sum()),
            right_displacement_rms_mm=rms(result.displacement[right_outside]))
        if plateau_outline is not None:
            report['methods'][method]['plateau_face'] = face_metrics(result, face, THRESHOLD_MM)

    scan_distances = prior['scan_center_distances_mm']
    global_fits = diagnose(output, results, uv, anchors, radii, options, scan_distances)
    for method in METHODS:
        report['methods'][method].update(global_fits[method])
    if plateau_outline is not None or any(row['status'] != 'ok' for result in global_fits.values() for row in result['spheres']):
        report['anchor_window_diagnostic'] = diagnose(output / 'anchor_windows', results, uv, anchors, radii, options,
                                                      scan_distances, anchor_window=True)
    plot_displacement(output / 'displacement_comparison.png', results, uv, photo, data['observed_linear_rgb'],
                      anchors, plateau_outline)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'runs/compare_3b1')
    args = parser.parse_args()
    image = ROOT / 'assets/test_pics/3b/3b1.npz'
    args.output_dir.mkdir(parents=True)
    run(image, args.output_dir, np.asarray(ANCHORS_UV, dtype=float), plateau_outline=np.asarray(PLATEAU_UV, dtype=float))
    print(f'Saved {args.output_dir}.')


if __name__ == '__main__':
    main()
