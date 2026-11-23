"""Move historical assets LUTs into the versioned, git-ignored model store."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from utils.calibration_store import (activate_run, create_run, lut_directory,
                                     _atomic_json)
from utils.normal_mapping import NormalCalibration


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _npz_keys(path: Path) -> set[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            return {name.removesuffix('.npy') for name in archive.namelist()}
    except (OSError, zipfile.BadZipFile):
        return set()


def _copy_exact(source: Path, target: Path) -> str:
    digest = _sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256(target) != digest:
            raise ValueError(f'Refusing to overwrite a different archived model: {target}')
    else:
        temporary = target.with_suffix(target.suffix+'.tmp')
        shutil.copy2(source, temporary)
        if _sha256(temporary) != digest:
            temporary.unlink()
            raise OSError(f'Calibration copy verification failed: {source}')
        temporary.replace(target)
    return digest


def _copy_chart_if_matching(source: Path, target: Path, chart_id: str) -> bool:
    if not source.is_file():
        return False
    with np.load(source, allow_pickle=False) as data:
        if 'chart_id' not in data or str(data['chart_id']) != chart_id:
            return False
    shutil.copy2(source, target)
    return True


def migrate_luts(config_path: str | Path) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    project = config_path.parent
    assets = project/'assets'
    store = lut_directory(config_path)
    store.mkdir(parents=True, exist_ok=True)
    manifest_file = store/'migration_manifest.json'
    manifest = (json.loads(manifest_file.read_text(encoding='utf-8')) if manifest_file.exists()
                else dict(format_version=1, files=[], adopted_runs=[]))
    sources = []
    for source in sorted(assets.rglob('*.npz')):
        keys = _npz_keys(source)
        model = {'slopes', 'color_min', 'color_max', 'variances'} <= keys
        grid = 'grid_kind' in keys and source.name == 'lut_material_grid.npz'
        if model or grid:
            sources.append((source, 'model' if model else 'material_grid'))

    # Archive exact bytes before changing metadata in any selected bundle.
    for source, role in sources:
        relative = source.relative_to(assets)
        target = store/'legacy'/relative
        digest = _copy_exact(source, target)
        record = dict(original=str(source.relative_to(project)),
                      archived=str(target.relative_to(store)), sha256=digest, role=role)
        manifest['files'] = [row for row in manifest['files'] if row['original'] != record['original']]
        manifest['files'].append(record)
        if role == 'model':
            model = NormalCalibration.load(source)
            chart_name = model.metadata.get('coordinate_chart_file')
            if chart_name:
                chart = (source.parent/chart_name).resolve()
                _copy_chart_if_matching(chart, target.parent/'reference_coordinates.npz',
                                        model.metadata.get('coordinate_chart_id'))

    current = store/'table'/'current.json'
    if not current.exists():
        candidates = []
        for source, role in sources:
            if role != 'model':
                continue
            model = NormalCalibration.load(source)
            grid = source.parent/'lut_material_grid.npz'
            if model.metadata.get('validation_status') != 'passed' or not grid.is_file():
                continue
            with np.load(grid, allow_pickle=False) as data:
                if str(data.get('coordinate_chart_id', '')) != model.metadata.get('coordinate_chart_id'):
                    continue
            modern = model.metadata.get('domain_kind') == 'black_frame_tangent_v1'
            candidates.append((modern, source.stat().st_mtime_ns, source, model, grid))
        if candidates:
            _, _, source, model, grid = max(candidates, key=lambda row: row[:2])
            run = create_run(store, 'table')
            metadata = dict(model.metadata, method='table')
            chart = source.parent/metadata.get('coordinate_chart_file', 'reference_coordinates.npz')
            has_chart = _copy_chart_if_matching(chart, run.chart_file, metadata['coordinate_chart_id'])
            metadata['migration'] = dict(original=str(source.relative_to(project)),
                                         sha256=_sha256(source), chart_unavailable=not has_chart)
            if has_chart:
                metadata['coordinate_chart_file'] = run.chart_file.name
            else:
                # Old reconstruction used the self-contained frozen grid even
                # after prepare replaced this model's original chart file.
                metadata.pop('coordinate_chart_file', None)
            replace(model, metadata=metadata).save(run.model_file)
            with np.load(grid, allow_pickle=False) as data:
                values = {key: data[key] for key in data.files if key != 'coordinate_chart_file'}
            if has_chart:
                values['coordinate_chart_file'] = np.asarray(run.chart_file.name)
            np.savez_compressed(run.grid_file, **values)
            report = dict(metadata.get('validation') or {}, status='passed',
                          migrated=True, model_file=str(run.model_file))
            _atomic_json(run.report_file, report)
            activate_run(run)
            manifest['adopted_runs'].append(dict(method='table', directory=str(run.directory.relative_to(store)),
                                                  original=str(source.relative_to(project))))

    # Persist the reversible move map before deleting each verified source.
    _atomic_json(manifest_file, manifest)
    moved = 0
    for source, _ in sources:
        target = store/'legacy'/source.relative_to(assets)
        if _sha256(source) != _sha256(target):
            raise OSError(f'Archived model changed; keeping source: {source}')
        source.unlink()
        moved += 1
    return dict(moved=moved, manifest_file=str(manifest_file),
                current_file=str(current) if current.exists() else None)


def main() -> None:
    parser = argparse.ArgumentParser(description='迁移 assets 中历史 LUT，校验 SHA-256 后移除原文件')
    parser.add_argument('--config', default=Path(__file__).resolve().parents[1]/'config.yaml')
    args = parser.parse_args()
    print(json.dumps(migrate_luts(args.config), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
