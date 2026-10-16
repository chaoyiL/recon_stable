"""Plot saved slope fields and audit displacement sign crossings without refitting."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from reconstruction.local_recon import _load_reconstruction_input, lookup_slopes, parse_zero_color_protection
from utils.color_difference import signed_difference_preview
from utils.color_protection import signed_color_protection_factors
from utils.config import load_config
from utils.normal_mapping import NormalCalibration, lookup_slopes_raw
from utils.surface_geometry import compute_surface_geometry
from utils.surface_projection import CameraModel, project_points
from utils.surface_sampling import rasterize_surface_grid


ZERO_SLOPE_TOLERANCE = 1e-6
NEAR_ZERO_SLOPE = .01


def sign_crossings(depth: np.ndarray, interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and edges connecting strictly opposite signs in the 4-neighbour grid.

    Edges have (row_a, column_a, row_b, column_b). Exact zero boundary values
    are excluded; a crossing is not assumed to be a stationary point of depth.
    """
    mask = np.zeros(depth.shape, bool)
    edges = []
    for axis in (0, 1):
        first, second = [slice(None)]*2, [slice(None)]*2
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        crossing = interior[first] & interior[second] & (depth[first]*depth[second] < 0)
        mask[first] |= crossing
        mask[second] |= crossing
        row, column = np.nonzero(crossing)
        edges.append(np.column_stack((row, column, row+(axis == 0), column+(axis == 1))))
    return mask, np.concatenate(edges)


def crossing_nodes(edges: np.ndarray, depth: np.ndarray, minimum_amplitude_mm: float) -> np.ndarray:
    a, b = (edges[:, 0], edges[:, 1]), (edges[:, 2], edges[:, 3])
    selected = edges[np.maximum(abs(depth[a]), abs(depth[b])) >= minimum_amplitude_mm]
    mask = np.zeros(depth.shape, bool)
    mask[selected[:, 0], selected[:, 1]] = True
    mask[selected[:, 2], selected[:, 3]] = True
    return mask


def load_case(result_path, calibration, radii, minimum_crossing_amplitude_mm):
    index = result_path.name.partition('_local_result')[0]
    root = result_path.parent
    input_path = root/f'{index}_recon_input.npz'
    xyz, colors, geometry_valid, photo, _ = _load_reconstruction_input(input_path, calibration.metadata)
    with np.load(result_path, allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    with np.load(root/f'{index}.npz', allow_pickle=False) as data:
        image_path = root/Path(str(data['source_image'])).name
        camera = CameraModel(data['camera_matrix'], data['dist_coeffs'], tuple(data['image_shape']))
        if not np.array_equal(data['xyz'], xyz):
            raise ValueError(f'{index}: baseline and reconstruction input XYZ differ')
    with np.load(input_path, allow_pickle=False) as data:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or not np.array_equal(image, data['observed_bgr']):
            raise ValueError(f'{index}: source photograph and cached colors differ')
        uv = data['uv']
        surface_mask = data['surface_mask']
    if not np.array_equal(photo, saved['photometric_valid_mask']):
        raise ValueError(f'{index}: saved and replayed photometric masks differ')
    replay, _, confidence = lookup_slopes(colors, calibration, photo,
        zero_color_inner_radius=radii[0], zero_color_outer_radius=radii[1],
        zero_color_negative_inner_radius=radii[2], zero_color_negative_outer_radius=radii[3],
        trusted_no_contact_mask=saved.get('trusted_no_contact_mask'))
    replay_error = float(np.max(np.abs(replay-saved['slopes'])))
    if replay_error > 1e-6 or not np.allclose(confidence, saved['confidence'], atol=1e-6):
        raise ValueError(f'{index}: supplied LUT/settings do not reproduce saved slopes/confidence')
    raw, _, _ = lookup_slopes_raw(colors, calibration, photo)
    interior = saved['valid_mask'] & ~saved['boundary_mask']
    crossing, edges = sign_crossings(saved['displacement'], interior)
    analysis_crossing = crossing_nodes(edges, saved['displacement'], minimum_crossing_amplitude_mm)
    geometry = compute_surface_geometry(xyz, geometry_valid)
    depth = saved['displacement']
    derivatives = np.zeros((*depth.shape, 2), np.float64)
    derivatives[1:-1, 1:-1, 0] = (depth[1:-1, 2:]-depth[1:-1, :-2])/2
    derivatives[1:-1, 1:-1, 1] = (depth[2:, 1:-1]-depth[:-2, 1:-1])/2
    gradient = np.einsum('...ab,...b->...a', geometry['k'], derivatives)
    protection_factor, _ = signed_color_protection_factors(colors, *radii)
    return dict(index=index, result_path=result_path, image_path=image_path, image=image,
        xyz=xyz, uv=uv, camera=camera,
        colors=colors, photo=photo, geometry_valid=geometry_valid, surface_mask=surface_mask,
        saved=saved, interior=interior, geometry=geometry, gradient=gradient,
        crossing=crossing, analysis_crossing=analysis_crossing, edges=edges, raw=raw, replay_error=replay_error,
        magnitude=np.linalg.norm(saved['slopes'], axis=-1), intensity=np.max(abs(colors), axis=-1),
        protection_factor=protection_factor)


def region_statistics(case, mask):
    mask = mask & case['interior']
    photo = mask & case['photo']
    result = dict(grid_point_count=int(mask.sum()), valid_color_count=int(photo.sum()),
                  missing_color_count=int((mask & ~case['photo']).sum()))
    if not photo.any():
        return result
    magnitude, intensity = case['magnitude'][photo], case['intensity'][photo]
    result.update(
        zero_slope_fraction=float(np.mean(magnitude <= ZERO_SLOPE_TOLERANCE)),
        near_zero_slope_fraction=float(np.mean(magnitude < NEAR_ZERO_SLOPE)),
        slope_norm_p05_p50_p95=np.percentile(magnitude, [5, 50, 95]).tolist(),
        color_max_abs_p05_p50_p95=np.percentile(intensity, [5, 50, 95]).tolist(),
        signed_rgb_median=np.median(case['colors'][photo], axis=0).tolist(),
        absolute_rgb_median=np.median(abs(case['colors'][photo]), axis=0).tolist(),
        raw_lut_slope_norm_median=float(np.median(np.linalg.norm(case['raw'][photo], axis=-1))),
        reconstructed_central_gradient_norm_median=float(np.median(
            np.linalg.norm(case['gradient'][photo], axis=-1))),
        absolute_depth_p50_p95=np.percentile(abs(case['saved']['displacement'][photo]), [50, 95]).tolist())
    return result


def summarize(case, radii, depth_band_mm):
    crossing = case['crossing']
    zero = case['magnitude'] <= ZERO_SLOPE_TOLERANCE
    depth, edges = case['saved']['displacement'], case['edges']
    a, b = (edges[:, 0], edges[:, 1]), (edges[:, 2], edges[:, 3])
    peak = np.maximum(abs(depth[a]), abs(depth[b]))
    photo_edges = case['photo'][a] & case['photo'][b]
    both_zero = photo_edges & zero[a] & zero[b]
    chord = np.linalg.norm(case['xyz'][b]-case['xyz'][a], axis=-1)
    jumps = abs(depth[b]-depth[a])
    secants = jumps/np.maximum(chord, 1e-12)
    color_valid = crossing & case['photo']
    return dict(index=case['index'], result_file=str(case['result_path']), image=str(case['image_path']),
        displacement_range_mm=[float(depth[case['interior']].min()), float(depth[case['interior']].max())],
        saved_slope_replay_max_abs_error=case['replay_error'],
        trusted_no_contact_count=int(case['saved']['trusted_no_contact_mask'].sum()),
        sign_crossing_points=region_statistics(case, crossing),
        sign_crossing_zero_slope_points=region_statistics(case, crossing & zero & case['photo']),
        sign_crossing_nonzero_slope_points=region_statistics(case, crossing & ~zero & case['photo']),
        amplitude_filtered_crossing_points=region_statistics(case, case['analysis_crossing']),
        amplitude_filtered_zero_slope_points=region_statistics(case, case['analysis_crossing'] & zero & case['photo']),
        crossing_amplitude_sensitivity=[dict(minimum_endpoint_amplitude_mm=threshold,
            statistics=region_statistics(case, crossing_nodes(edges, depth, threshold)))
            for threshold in (0, .0001, .001, .01, .02, .05)],
        near_zero_depth_band=region_statistics(case, abs(depth) <= depth_band_mm),
        all_valid_interior=region_statistics(case, case['interior']),
        crossing_color_bins={
            'inner_or_below': int((color_valid & (case['protection_factor'] == 0)).sum()),
            'transition': int((color_valid & (case['protection_factor'] > 0) & (case['protection_factor'] < 1)).sum()),
            'outer_or_above': int((color_valid & (case['protection_factor'] >= 1)).sum())},
        crossing_edges=dict(count=len(edges), both_colors_valid_count=int(photo_edges.sum()),
            both_endpoint_slopes_zero_count=int(both_zero.sum()),
            both_endpoint_depths_under_002mm_fraction=float(np.mean(peak < .02)) if len(edges) else None,
            both_endpoint_depths_under_005mm_fraction=float(np.mean(peak < .05)) if len(edges) else None,
            zero_slope_edges_abs_depth_jump_p50_p95_mm=np.percentile(jumps[both_zero], [50, 95]).tolist()
                if both_zero.any() else None,
            zero_slope_edges_abs_depth_secant_over_3d_chord_p50_p95=np.percentile(secants[both_zero], [50, 95]).tolist()
                if both_zero.any() else None))


def image_axes(axis, case):
    yy, xx = np.nonzero(case['surface_mask'])
    axis.set(xlim=(xx.min()-12, xx.max()+12), ylim=(yy.max()+12, yy.min()-12),
             xlabel='Image x (px)', ylabel='Image y (px)', aspect='equal')
    axis.set_facecolor('#b8b8b8')


def zero_contour(axis, case):
    uv = case['uv']
    neighbourhood = cv2.dilate(case['analysis_crossing'].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    depth = np.ma.array(case['saved']['displacement'], mask=~(case['interior'] & neighbourhood))
    if depth.count() == 0:
        return
    if depth.min() < 0 < depth.max():
        axis.contour(uv[..., 0], uv[..., 1], depth, levels=[0], colors='black', linewidths=1.6)
        axis.contour(uv[..., 0], uv[..., 1], depth, levels=[0], colors='#fff35c', linewidths=.65)


def plot_case(case, output_path, slope_limit, depth_limit, minimum_crossing_amplitude_mm):
    figure = Figure(figsize=(15, 8), layout='constrained')
    axes = figure.subplots(2, 2, sharex=True, sharey=True)
    shape = case['image'].shape[:2]
    fields = np.dstack((case['colors'], case['magnitude']))
    color = rasterize_surface_grid(fields, case['uv'], case['photo'], shape)
    depth = rasterize_surface_grid(case['saved']['displacement'], case['uv'], case['geometry_valid'], shape)
    axes[0, 0].imshow(case['image'][..., ::-1])
    axes[0, 0].set_title('Photograph + zero-displacement contour')
    artist = axes[0, 1].imshow(np.ma.array(depth.values[..., 0], mask=~depth.valid),
                             cmap='coolwarm', vmin=-depth_limit, vmax=depth_limit)
    axes[0, 1].set_title('Signed normal displacement d')
    figure.colorbar(artist, ax=axes[0, 1], label='mm', shrink=.8)
    artist = axes[1, 0].imshow(np.ma.array(color.values[..., 3], mask=~color.valid),
                             cmap='YlOrRd', vmin=0, vmax=slope_limit)
    figure.colorbar(artist, ax=axes[1, 0], label='sqrt(p^2 + q^2), dimensionless', shrink=.8)
    axes[1, 0].set_title('Used slopes: magnitude + projected tangent arrows')
    tangent = np.einsum('...ca,...a->...c', case['geometry']['frame'], case['saved']['slopes'])
    base_uv, _ = project_points(case['xyz'], case['camera'])
    end_uv, _ = project_points(case['xyz']+1e-3*tangent, case['camera'])
    arrow = (end_uv-base_uv)/1e-3*2  # A common 2 mm tangent-vector display multiplier.
    rows, columns = case['photo'].shape
    subset = np.zeros((rows, columns), bool)
    subset[::max(1, rows//30), ::max(1, columns//12)] = True
    subset &= case['photo'] & case['geometry']['valid'] & (case['magnitude'] > ZERO_SLOPE_TOLERANCE)
    axes[1, 0].quiver(*case['uv'][subset].T, *arrow[subset].T,
                      color='#122a4f', angles='xy', scale_units='xy', scale=1, width=.003)
    rgb = signed_difference_preview(color.values[..., :3], color.valid, gain=1)[..., ::-1]
    axes[1, 1].imshow(rgb)
    axes[1, 1].set_title('Signed linear-RGB difference (neutral gray = zero)')
    for axis in axes.flat:
        zero_contour(axis, case)
        image_axes(axis, case)
    figure.suptitle(f"{case['result_path'].stem} | {case['image_path'].name}", fontsize=15)
    figure.supxlabel(f'Yellow: d = 0, keeping crossings with an endpoint |d| >= {minimum_crossing_amplitude_mm:g} mm. '
        'Image coordinates and scale match.\n'
        'Arrows project 2 mm * (p e1 + q e2); no normalization.\n'
        'Color display: clip(0.5 + dRGB, 0, 1). Black color pixels have no measurement; '
        'gray slope pixels have no photometric support.', fontsize=10)
    figure.savefig(output_path, dpi=180)


def plot_crossing_summary(cases, radii, minimum_crossing_amplitude_mm, output_path):
    figure = Figure(figsize=(17, 4*len(cases)), layout='constrained')
    axes = figure.subplots(len(cases), 3, squeeze=False)
    for row, case in enumerate(cases):
        crossing = case['analysis_crossing'] & case['photo']
        zero = case['magnitude'] <= ZERO_SLOPE_TOLERANCE
        axis = axes[row, 0]
        axis.imshow(case['image'][..., ::-1], alpha=.4)
        for mask, color, label in ((case['analysis_crossing'] & ~case['photo'], 'gray', 'Missing color'),
                                  (crossing & ~zero, '#f06c00', 'Nonzero used slope'),
                                  (crossing & zero, '#0048c9', 'Zero used slope')):
            axis.scatter(*case['uv'][mask].T, s=4, color=color, label=label)
        image_axes(axis, case)
        axis.set_title(f"{case['result_path'].stem}: sign-crossing nodes")
        axis.legend(fontsize=8, loc='upper right')
        intensity = case['protection_factor'][crossing]
        axis = axes[row, 1]
        axis.scatter(intensity, np.linalg.norm(case['raw'][crossing], axis=-1),
                     s=6, alpha=.3, color='#8c42b5', label='Raw LUT slope')
        axis.scatter(intensity, case['magnitude'][crossing], s=7, alpha=.5,
                     color='#008d9b', label='Saved / used slope')
        axis.set(xlabel='Signed color protection factor', ylabel='Slope magnitude',
                 title='Same crossing points: raw LUT vs used slopes')
        axis.legend(fontsize=8)
        axis = axes[row, 2]
        bins = np.linspace(0, 1, 35)
        axis.hist([case['protection_factor'][crossing & zero], case['protection_factor'][crossing & ~zero]],
                  bins=bins, stacked=True, color=['#0048c9', '#f06c00'],
                  label=['Zero used slope', 'Nonzero used slope'])
        axis.set(xlabel='Signed color protection factor', ylabel='Grid point count',
                 title=f"Valid crossing points: {crossing.sum()} | zero slopes: "
                       f"{zero[crossing].mean() if crossing.any() else 0:.1%}")
        axis.legend(fontsize=8)
        for axis in axes[row, 1:]:
            for radius in (0, 1):
                axis.axvline(radius, color='#444444', linestyle='--', linewidth=1)
            axis.grid(alpha=.2)
    figure.suptitle('Displacement sign crossings: slope and color audit', fontsize=16)
    figure.supxlabel('Crossings use strictly opposite signs on adjacent interior grid nodes; '
        f'at least one endpoint has |d| >= {minimum_crossing_amplitude_mm:g} mm (analysis cutoff).\n'
        'Fixed zero boundaries and missing photometry are excluded from statistics. '
        f'Zero slope: magnitude <= {ZERO_SLOPE_TOLERANCE:g}; dashed lines: factor = 0 (zero), 1 (full).')
    figure.savefig(output_path, dpi=180)


def save_crossing_points(cases, path):
    fields = ('index', 'result_file', 'row', 'column', 'image_x_px', 'image_y_px', 'displacement_mm',
              'passes_amplitude_filter', 'photometric_valid', 'p', 'q', 'slope_norm',
              'raw_lut_slope_norm', 'delta_r', 'delta_g', 'delta_b', 'color_max_abs', 'confidence')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for row, column in np.argwhere(case['crossing']):
                ij = (row, column)
                photo = bool(case['photo'][ij])
                p, q = case['saved']['slopes'][ij] if photo else ('', '')
                r, g, b = case['colors'][ij] if photo else ('', '', '')
                writer.writerow(dict(index=case['index'], result_file=str(case['result_path']), row=row, column=column,
                    image_x_px=case['uv'][ij][0], image_y_px=case['uv'][ij][1],
                    displacement_mm=case['saved']['displacement'][ij],
                    passes_amplitude_filter=bool(case['analysis_crossing'][ij]), photometric_valid=photo,
                    p=p, q=q, slope_norm=case['magnitude'][ij] if photo else '',
                    raw_lut_slope_norm=np.linalg.norm(case['raw'][ij]) if photo else '',
                    delta_r=r, delta_g=g, delta_b=b, color_max_abs=case['intensity'][ij] if photo else '',
                    confidence=case['saved']['confidence'][ij]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('assets/tests_pic'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--calibration', type=Path, required=True, help='LUT that reproduces the saved outputs')
    parser.add_argument('--result-version', type=int,
                        help='Select *_local_result.<version>.npz, e.g. 3; omit for unversioned results')
    parser.add_argument('--depth-band-mm', type=float, default=.02)
    parser.add_argument('--minimum-crossing-amplitude-mm', type=float, default=.001,
                        help='Analysis/display cutoff for adjacent sign changes, not a sensor accuracy claim')
    args = parser.parse_args()
    if args.depth_band_mm <= 0 or args.minimum_crossing_amplitude_mm < 0:
        parser.error('Depth band must be positive; crossing amplitude must be nonnegative')
    version_suffix = '' if args.result_version is None else f'.{args.result_version}'
    output = args.output_dir or args.input_dir/f'slope_color_analysis{version_suffix}'
    radii = parse_zero_color_protection(load_config(args.config)['local_reconstruction'])
    calibration = NormalCalibration.load(args.calibration)
    paths = sorted(args.input_dir.glob(f'*_local_result{version_suffix}.npz'))
    if not paths:
        parser.error('No saved local reconstruction results found')
    cases = [load_case(path, calibration, radii, args.minimum_crossing_amplitude_mm) for path in paths]
    source_images = {p for p in args.input_dir.glob('*.png') if '_local_result' not in p.stem}
    missing = source_images-{case['image_path'] for case in cases}
    if missing:
        parser.error(f'Photographs without matched reconstruction results: {sorted(map(str, missing))}')
    output.mkdir(parents=True, exist_ok=True)
    slope_limit = max(float(case['magnitude'][case['photo']].max()) for case in cases)
    depth_limit = max(float(abs(case['saved']['displacement'][case['interior']]).max()) for case in cases)
    summaries = []
    for case in cases:
        destination = output/f"{case['index']}_slope_color.png"
        plot_case(case, destination, slope_limit, depth_limit, args.minimum_crossing_amplitude_mm)
        summary = summarize(case, radii, args.depth_band_mm)
        summary['figure'] = str(destination)
        summaries.append(summary)
        print(f"{case['index']}: {destination}", flush=True)
    plot_crossing_summary(cases, radii, args.minimum_crossing_amplitude_mm, output/'crossing_summary.png')
    save_crossing_points(cases, output/'crossing_points.csv')
    report = dict(result_version=args.result_version, calibration_file=str(args.calibration),
        calibration_validation_status=calibration.metadata.get('validation_status'),
        note='Saved reconstructions are audited without refitting or modifying their inputs.',
        definition='Unique interior grid nodes adjacent to a strictly opposite displacement sign (4-neighbours).',
        depth_meaning='Signed normal displacement from the fitted current-frame baseline, in mm.',
        slope_meaning='Saved LUT slopes after configured protection; missing photometry is not a zero observation.',
        zero_color_protection=dict(inner_radius=radii[0], outer_radius=radii[1],
                                   negative_inner_radius=radii[2], negative_outer_radius=radii[3]),
        zero_slope_tolerance=ZERO_SLOPE_TOLERANCE, near_zero_slope_threshold=NEAR_ZERO_SLOPE,
        minimum_crossing_endpoint_amplitude_mm=args.minimum_crossing_amplitude_mm,
        near_zero_depth_band_half_width_mm=args.depth_band_mm, cases=summaries)
    (output/'crossing_statistics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    fields = ('index', 'result_file', 'image', 'crossing_points', 'valid_color_points', 'zero_slope_percent',
              'filtered_valid_color_points', 'filtered_zero_slope_percent',
              'median_slope', 'median_max_abs_drgb', 'zero_slope_median_max_abs_drgb', 'zero_slope_raw_lut_median')
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            stats, zeros = summary['sign_crossing_points'], summary['sign_crossing_zero_slope_points']
            filtered = summary['amplitude_filtered_crossing_points']
            writer.writerow(dict(index=summary['index'], result_file=summary['result_file'], image=summary['image'],
                crossing_points=stats['grid_point_count'], valid_color_points=stats['valid_color_count'],
                zero_slope_percent=100*stats.get('zero_slope_fraction', 0),
                filtered_valid_color_points=filtered['valid_color_count'],
                filtered_zero_slope_percent=100*filtered.get('zero_slope_fraction', 0),
                median_slope=stats.get('slope_norm_p05_p50_p95', [None]*3)[1],
                median_max_abs_drgb=stats.get('color_max_abs_p05_p50_p95', [None]*3)[1],
                zero_slope_median_max_abs_drgb=zeros.get('color_max_abs_p05_p50_p95', [None]*3)[1],
                zero_slope_raw_lut_median=zeros.get('raw_lut_slope_norm_median')))
    print(f'Crossing statistics: {output / "crossing_statistics.json"}', flush=True)


if __name__ == '__main__':
    main()
