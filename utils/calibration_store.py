"""Versioned normal-calibration artifacts shared by training and reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Mapping
from uuid import uuid4

import numpy as np

from utils.config import load_config


METHODS = ('table', 'mlp3', 'mlp5', 'mlp6', 'mlp8')


def validate_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f'normal_calibration.method must be one of {METHODS}; got {method!r}')
    return method


def selected_method(config: Mapping) -> str:
    raw = config.get('normal_calibration') or {}
    if not isinstance(raw, Mapping):
        raise ValueError('normal_calibration must be a mapping')
    return validate_method(raw.get('method', 'table'))


def _path(value, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base/path).resolve()


def lut_directory(config_path: str | Path, config: Mapping | None = None) -> Path:
    source = Path(config_path).expanduser().resolve()
    config = load_config(source) if config is None else config
    raw = config.get('normal_calibration') or {}
    return _path(raw.get('lut_dir', 'lut'), source.parent)


@dataclass(frozen=True)
class CalibrationRun:
    directory: Path
    method: str

    @property
    def model_file(self) -> Path:
        return self.directory/'normal_lut.npz'

    @property
    def training_file(self) -> Path:
        return self.directory/'normal_lut_train.npz'

    @property
    def grid_file(self) -> Path:
        return self.directory/'lut_material_grid.npz'

    @property
    def chart_file(self) -> Path:
        return self.directory/'reference_coordinates.npz'

    @property
    def report_file(self) -> Path:
        return self.directory/'validation.json'


@dataclass(frozen=True)
class CalibrationArtifacts:
    model_file: Path
    grid_file: Path | None
    directory: Path
    method: str


def create_run(lut_dir: str | Path, method: str) -> CalibrationRun:
    method = validate_method(method)
    version = datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:6]
    directory = Path(lut_dir).expanduser().resolve()/method/version
    directory.mkdir(parents=True, exist_ok=False)
    return CalibrationRun(directory, method)


def stage_reference_chart(run: CalibrationRun, source_chart: str | Path) -> Path:
    source = Path(source_chart).resolve()
    if source != run.chart_file.resolve():
        shutil.copyfile(source, run.chart_file)
    return run.chart_file


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def _metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'])) if 'metadata' in data else {}
    if not isinstance(metadata, dict):
        raise ValueError('Calibration metadata must be a mapping')
    return metadata


def activate_run(run: CalibrationRun) -> Path:
    """Atomically select one complete, validated model/grid bundle."""
    validate_method(run.method)
    if run.directory.parent.name != run.method:
        raise ValueError('Calibration run must be stored below its method directory')
    report = json.loads(run.report_file.read_text(encoding='utf-8'))
    if report.get('status') != 'passed':
        raise ValueError('Only passed calibration runs can become current')
    metadata = _metadata(run.model_file)
    if metadata.get('validation_status') != 'passed':
        raise ValueError('Selected model must declare passed validation')
    actual_method = metadata.get('method', metadata.get('fit_method', 'table'))
    if actual_method != run.method:
        raise ValueError('Calibration method differs from its storage directory')
    grid = None
    if metadata.get('coordinate_chart_id'):
        with np.load(run.grid_file, allow_pickle=False) as data:
            if str(data['coordinate_chart_id']) != metadata['coordinate_chart_id']:
                raise ValueError('Calibration model/material grid identities differ')
        if run.chart_file.is_file():
            with np.load(run.chart_file, allow_pickle=False) as chart:
                if str(chart['chart_id']) != metadata['coordinate_chart_id']:
                    raise ValueError('Calibration model/coordinate chart identities differ')
        elif not metadata.get('migration', {}).get('chart_unavailable', False):
            raise ValueError('Calibration bundle is missing its frozen coordinate chart')
        grid = run.grid_file.name
    pointer = run.directory.parent/'current.json'
    _atomic_json(pointer, dict(format_version=1, method=run.method,
        run=run.directory.name, model_file=run.model_file.name,
        grid_file=grid, validation_status='passed'))
    return pointer


def _contained_path(root: Path, value: str) -> Path:
    path = (root/value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Calibration registry path escapes its method directory')
    return path


def resolve_artifacts(
    config_path: str | Path, *, model_override: str | Path | None = None,
    grid_override: str | Path | None = None,
) -> CalibrationArtifacts:
    """Resolve the selected method once so model and grid always share a version.

    Explicit CLI paths resolve against cwd. Legacy explicit config paths remain
    accepted only when no method/store configuration is present.
    """
    source = Path(config_path).expanduser().resolve()
    config = load_config(source)
    method = selected_method(config)
    raw = config.get('normal_calibration') or {}
    local = config.get('local_reconstruction') or {}
    reconstruction = config.get('reconstruction') or {}
    legacy = 'method' not in raw and 'lut_dir' not in raw
    registry_selected = False
    if model_override is not None:
        model = Path(model_override).expanduser().resolve()
        directory = model.parent
        neighbor = directory/'lut_material_grid.npz'
        grid = neighbor if neighbor.is_file() else None
    elif legacy and local.get('calibration_file'):
        model = _path(local['calibration_file'], source.parent)
        directory = model.parent
        grid = (_path(reconstruction['material_grid_file'], source.parent)
                if reconstruction.get('material_grid_file') else None)
    else:
        registry_selected = True
        method_root = lut_directory(source, config)/method
        pointer_file = method_root/'current.json'
        if not pointer_file.is_file():
            raise ValueError(f'No validated {method} model in {method_root}; run calibrate-norm --stage fit')
        pointer = json.loads(pointer_file.read_text(encoding='utf-8'))
        if (pointer.get('format_version') != 1 or pointer.get('method') != method
                or pointer.get('validation_status') != 'passed'):
            raise ValueError(f'Invalid calibration registry: {pointer_file}')
        directory = _contained_path(method_root, pointer['run'])
        model = _contained_path(directory, pointer['model_file'])
        grid = (_contained_path(directory, pointer['grid_file'])
                if pointer.get('grid_file') else None)
    if grid_override is not None:
        grid = Path(grid_override).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f'Selected calibration does not exist: {model}')
    if registry_selected:
        metadata = _metadata(model)
        if metadata.get('method', 'table') != method or metadata.get('validation_status') != 'passed':
            raise ValueError('Selected calibration differs from its validated method registry')
    if model_override is not None:
        method = validate_method(_metadata(model).get('method', 'table'))
    if grid is not None and not grid.is_file():
        raise FileNotFoundError(f'Selected material grid does not exist: {grid}')
    return CalibrationArtifacts(model, grid, directory, method)
