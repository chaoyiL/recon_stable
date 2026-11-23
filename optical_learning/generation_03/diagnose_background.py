"""Compare fixed-shape background error and test a 3b2 right-side background correction."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from calibrate.utils.norm_pipeline import parse_settings
from optical_learning.generation_02.evaluate import color_metrics
from optical_learning.lut_background_3b2.diagnosis import IMAGE_ANCHORS_UV
from optical_learning.lut_background_3b2.experiment import BACKGROUND, CONFIG, IMAGE, matched_inputs
from reconstruction.local_recon import load_local_reconstruction_settings, reconstruct_from_input
from utils.config import load_config, get_curvature_smoothing_half_width_mm
from utils.normal_models import load_normal_model
from utils.surface_sampling import rasterize_surface_grid, sample_linear_rgb_to_surface


HERE = Path(__file__).resolve().parent
CURRENT = HERE/'runs/configured_40px/margin_40'
WIDE = HERE/'runs/initial/margin_60'
SECOND_RECON = HERE.parent/'lut_background_3b2/table/runs/initial'
RIGHT_U = 445.
BLEND_START_U = 425.
NONCONTACT_RADIUS = 55.


def reference_scores(work_dir: Path) -> dict:
    paths = {'generation_02': BACKGROUND.parent, 'margin_40': CURRENT/'background', 'margin_60': WIDE/'background'}
    with ExitStack() as stack:
        grids = {name: stack.enter_context(np.load(path/'evaluation_grids.npz', allow_pickle=False))
                 for name, path in paths.items()}
        ref = grids['generation_02']
        st, mask, rgb = ref['st'], ref['reference_valid'], ref['reference_rgb']
        with np.load(work_dir/'reference_surface.npz', allow_pickle=False) as geometry:
            np.testing.assert_array_equal(st, geometry['observation_st'])
            u, v = np.moveaxis(geometry['observation_uv'], -1, 0)
        defect = (455 <= u) & (u <= 505) & (111 <= v) & (v <= 135)
        interior = ((.1 <= st) & (st <= .9)).all(axis=-1) & ~defect
        scores = {}
        for name, grid in grids.items():
            for field in ('st', 'reference_valid', 'reference_rgb'):
                np.testing.assert_array_equal(ref[field], grid[field])
            error = grid['prediction'] - rgb
            scores[name] = dict(full=color_metrics(error[mask]), interior=color_metrics(error[mask & interior]))
    return scores


def plot_fields(path, observed, residuals, displacements, uv, photo, image_shape):
    fields = np.concatenate((observed, observed - residuals['generation_02'],
                             observed - residuals['margin_40'],
                             np.stack([residuals[k][..., 0] for k in residuals], axis=-1),
                             np.stack(list(displacements.values()), axis=-1)), axis=-1)
    render = rasterize_surface_grid(fields, uv, photo, image_shape)
    height, width = image_shape
    left, top = np.floor(uv[photo].min(axis=0)).astype(int)
    right, bottom = np.ceil(uv[photo].max(axis=0)).astype(int)
    extent = (0, width, height, 0)
    figure, axes = plt.subplots(3, 3, figsize=(12, 6.7), layout='constrained')
    for column, name in enumerate(('Observed 3b2', 'Generation 2 background', '40 px learned background')):
        rgb = np.clip(render.values[..., column*3:(column+1)*3], 0, 1)
        rgb = np.where(rgb <= .0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-.055)
        axes[0, column].imshow(np.where(render.valid[..., None], rgb, .13), extent=extent)
        axes[0, column].set_title(name)
    for row, start, limit, label, titles in (
        (1, 9, .12, 'Observed minus background, R', ('Fixed reference', 'Generation 2', '40 px learned')),
        (2, 12, .7, 'Normal displacement (mm)', ('Fixed reference', '40 px learned', 'Right background replaced')),
    ):
        for column, title in enumerate(titles):
            ax = axes[row, column]
            values = np.ma.array(render.values[..., start + column], mask=~render.valid)
            panel = ax.imshow(values, extent=extent, cmap='coolwarm', vmin=-limit, vmax=limit)
            ax.set_title(title)
            ax.axvline(RIGHT_U, color='0.25', linestyle=':', linewidth=1)
        figure.colorbar(panel, ax=axes[row], label=label, shrink=.85)
    for row, panels in enumerate(axes):
        for ax in panels:
            ax.set(xlim=(left-4, right+4), ylim=(bottom+4, top-4), yticks=[])
            if row < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Image u (px)')
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(output: Path) -> dict:
    settings = parse_settings(CONFIG)
    half_width = get_curvature_smoothing_half_width_mm(load_config(CONFIG))
    table_dir = CURRENT/'3b2'
    fixed_table = load_normal_model(table_dir/'reference/table.npz')
    data, third, uv = matched_inputs(settings.reference_image, half_width, fixed_table,
                                    background_file=CURRENT/'background/model.npz')
    other, second, uv2 = matched_inputs(settings.reference_image, half_width, fixed_table)
    for key in ('xyz', 'st', 'photometric_valid_mask'):
        np.testing.assert_array_equal(data[key], other[key])
    np.testing.assert_array_equal(uv, uv2)
    photo = data['photometric_valid_mask']
    distance = np.linalg.norm(uv[..., None, :] - IMAGE_ANCHORS_UV, axis=-1).min(axis=-1)
    outside = photo & (distance >= NONCONTACT_RADIUS)
    right = outside & (uv[..., 0] >= RIGHT_U)
    residuals = {'reference': third['reference'], 'generation_02': second['learned'], 'margin_40': third['learned']}
    scores = {name: dict(noncontact=color_metrics(values[outside]), right=color_metrics(values[right]))
              for name, values in residuals.items()}

    # Keep the learned Table and all contact-region colors fixed. Blend in the
    # measured background on the right with a short transition in image space.
    position = np.clip((uv[..., 0] - BLEND_START_U) / (RIGHT_U - BLEND_START_U), 0, 1)
    blend = position**2 * (3 - 2*position)
    contact_neighborhood = distance < NONCONTACT_RADIUS
    blend = np.where(photo & ~contact_neighborhood, blend, 0.)
    corrected = third['learned'] + blend[..., None]*(third['reference'] - third['learned'])
    np.testing.assert_array_equal(corrected[photo & contact_neighborhood], third['learned'][photo & contact_neighborhood])
    model_path = table_dir/'learned/table.npz'
    local_settings = load_local_reconstruction_settings(CONFIG, input_override=str(IMAGE),
                                                        calibration_override=str(model_path))
    result = reconstruct_from_input(dict(data, color_residual_grid=corrected), load_normal_model(model_path),
                                    local_settings)
    depths = {}
    for name, path in (('reference', table_dir/'reference'), ('margin_40', table_dir/'learned'),
                       ('generation_02', SECOND_RECON/'learned')):
        with np.load(path/'3b2.npz', allow_pickle=False) as reconstruction:
            np.testing.assert_array_equal(reconstruction['photometric_valid_mask'], photo)
            depths[name] = reconstruction['displacement']
    np.testing.assert_array_equal(result.photometric_valid_mask, photo)
    depths['right_background_replaced'] = result.displacement
    artifact = {name: dict(points_over_0p2_mm=int(((field < -.2) & right).sum()),
                           displacement_rms_mm=float(np.sqrt(np.mean(field[right].astype(float)**2))))
                for name, field in depths.items()}

    image = data['observed_linear_rgb']
    source_valid = data['surface_mask'] & ~data['marker_image_mask'] & np.all(image < 1, axis=-1)
    observed = sample_linear_rgb_to_surface(image, uv, source_valid).values
    displayed = {name: depths[name] for name in ('reference', 'margin_40', 'right_background_replaced')}
    plot_fields(output/'3b2_background_bias.png', observed, residuals, displayed, uv, photo, image.shape[:2])
    report = dict(reference=reference_scores(settings.work_dir), noncontact_radius_px=NONCONTACT_RADIUS,
                  right_u_px=RIGHT_U, blend_start_u_px=BLEND_START_U, residuals=scores, right_displacement=artifact)
    (output/'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE/'runs/background_bias')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True)
    run(args.output_dir)
    print(f'Background diagnosis saved to {args.output_dir}.')


if __name__ == '__main__':
    main()
