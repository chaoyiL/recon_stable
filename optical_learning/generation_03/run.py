"""Measure how an expanded contact exclusion affects background and 3b2 reconstruction."""

import argparse
import json
from pathlib import Path

from calibrate.utils.norm_pipeline import parse_settings
from optical_learning.generation_02.run import run as fit_background
from optical_learning.lut_background_3b2.experiment import CONFIG, run as compare_backgrounds
from optical_learning.lut_background_3b2.table.run import fit_models


HERE = Path(__file__).resolve().parent
SWEEP_MARGINS_PX = (20, 40, 60)


def run(output: Path, *, sweep: bool = False) -> list[dict]:
    margin = parse_settings(CONFIG).background_contact_exclusion_margin_px
    margins = SWEEP_MARGINS_PX if sweep else (margin,)
    comparison = []
    for margin in margins:
        print(f'\nBackground sampling exclusion: {margin} px.', flush=True)
        directory = output / f'margin_{margin:02d}'
        background = directory / 'background'
        background.mkdir(parents=True)
        fit = fit_background(background, CONFIG, contact_exclusion_margin_px=margin,
                             holdout_exclusion_margin_px=max(margins))

        recon = directory / '3b2'
        reconstruction = compare_backgrounds(recon, 'table', fit_models, background_file=background / 'model.npz')
        learned = reconstruction['methods']['learned']
        comparison.append(dict(
            margin_px=margin,
            training_observations=fit['observations']['train'],
            all_observations=fit['observations']['all'],
            covered_material_points=fit['coverage']['all']['covered'],
            holdout_rmse=fit['holdout']['mlp_full_valid']['rgb_rmse'],
            holdout_samples=fit['holdout']['mlp_full_valid']['n'],
            reference_rmse=fit['reference']['mlp_full_valid']['rgb_rmse'],
            reference_samples=fit['reference']['mlp_full_valid']['n'],
            target_background_difference_rms=reconstruction['background_difference_rms'],
            target_right_background_difference_rms=reconstruction['right_background_difference_rms'],
            target_photometric_points=reconstruction['photometric_points'],
            right_press_points=learned['right_photometric_points_over_0p2_mm'],
            sphere_states=[sphere['status'] for sphere in learned['spheres']],
            sphere_window_points=[row['below_minus_0p2_mm'] for row in learned['anchor_diagnostics']],
            center_distance_rmse_mm=learned.get('distance_rmse_mm'),
        ))

    for key in ('holdout_samples', 'reference_samples', 'target_photometric_points'):
        if len({row[key] for row in comparison}) != 1:
            raise ValueError(f'Evaluation support differs between margins: {key}')
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='configured')
    parser.add_argument('--sweep', action='store_true', help='Repeat the 20/40/60 px experiment')
    args = parser.parse_args()
    if args.run_name in {'', '.', '..'} or Path(args.run_name).name != args.run_name:
        parser.error('--run-name must be a single directory name')
    output = HERE / 'runs' / args.run_name
    output.mkdir(parents=True)
    for row in run(output, sweep=args.sweep):
        print(row)


if __name__ == '__main__':
    main()
