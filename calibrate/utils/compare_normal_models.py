"""Compare calibration methods using the same whole-image holdout split."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import csv
import json

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from utils.calibration_store import METHODS
from calibrate.utils.norm_pipeline import fit_and_validate_samples


def _comparison_figure(results, output):
    ordered = next((report['validation_images'] for report in results.values()
                    if report.get('validation_images')), [])
    figure = Figure(figsize=(max(9, len(ordered)*1.1), 9), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True)
    if not ordered:
        axes[0].text(.5, .5, 'No completed image evaluations', ha='center', va='center',
                     transform=axes[0].transAxes)
    for method, report in results.items():
        values = {row['image_key']: row for row in report.get('images', [])}
        if not values:
            continue
        x = np.arange(len(ordered))
        axes[0].plot(x, [values[key]['rms_degrees'] for key in ordered], marker='o', label=method)
        axes[1].plot(x, [100*values[key]['out_of_range_fraction'] for key in ordered], marker='o', label=method)
        axes[2].plot(x, [100*values[key]['unsupported_fraction'] for key in ordered], marker='o', label=method)
    thresholds = next((report['thresholds'] for report in results.values() if report.get('thresholds')), None)
    if thresholds:
        axes[0].axhline(thresholds['rms_degrees'], color='black', linestyle='--', label='limit')
        axes[1].axhline(100*thresholds['maximum_out_of_range_fraction'], color='black', linestyle='--')
        axes[2].axhline(100*thresholds['maximum_unsupported_fraction'], color='black', linestyle='--')
    axes[0].set_ylabel('Normal angle RMS (degrees)')
    axes[1].set_ylabel('RGB outside training range (%)')
    axes[2].set_ylabel('Unsupported samples (%)')
    axes[2].set_xticks(np.arange(len(ordered)), [Path(key).stem for key in ordered], rotation=35, ha='right')
    for axis in axes:
        axis.grid(alpha=.25)
    if ordered:
        axes[0].legend(ncols=min(5, len(results)+1))
    figure.suptitle('Whole-image normal calibration comparison')
    figure.savefig(output, dpi=140)
    figure.clear()


def compare_normal_models(settings, sample_path=None):
    """Fit all methods independently and retain failed candidates for review."""
    if settings.lut_dir is None:
        raise ValueError('Comparisons require a versioned calibration lut_dir')
    source = Path(sample_path or settings.input_file).resolve()
    version = datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:6]
    output = settings.lut_dir/'comparisons'/version
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    reference_split = None
    sample_digest = None
    for method in METHODS:
        print(f'compare: fitting {method}')
        report = fit_and_validate_samples(source, replace(settings, method=method, export_output=False),
                                          raise_on_failure=False)
        digest = report.get('sample_sha256')
        if digest is not None:
            if sample_digest is not None and digest != sample_digest:
                raise ValueError('Comparison sample file changed between methods')
            sample_digest = digest
        split = report.get('validation_images')
        if split is not None:
            if reference_split is not None and split != reference_split:
                raise ValueError('Comparison methods returned different held-out image groups')
            reference_split = split
        results[method] = report
    columns = ('method', 'image_key', 'sample_count', 'rms_degrees', 'median_degrees', 'p95_degrees',
               'out_of_range_fraction', 'unsupported_fraction', 'passed')
    csv_path = output/'per_image.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for method, report in results.items():
            writer.writerows(dict(method=method, **row) for row in report.get('images', []))
    figure_path = output/'comparison.png'
    _comparison_figure(results, figure_path)
    summary = dict(sample_file=str(source), sample_sha256=sample_digest,
                   split_role='historical_development_holdout_not_independent_test',
                   validation_images=reference_split or [], results=results,
                   per_image_csv=str(csv_path), figure_file=str(figure_path), directory=str(output))
    summary_path = output/'comparison.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    summary['summary_file'] = str(summary_path)
    return summary
