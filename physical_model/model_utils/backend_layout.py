"""Analytic simulation configuration and render output locations."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / 'configs'
BACKEND_RESULT_DIRS = {'analytic': 'analytic'}


def canonical_backend(value: str) -> str:
    """Validate the only backend included in this package."""
    if value not in BACKEND_RESULT_DIRS:
        raise ValueError(f'Unknown tactile backend: {value}')
    return value


def simulation_config_path(backend: str = 'analytic') -> Path:
    return CONFIG_DIR / f'{canonical_backend(backend)}.json'


def new_render_output_dir(backend: str = 'analytic', *, curved: bool = False) -> Path:
    """Choose a unique output location under physical_model/render_results."""
    canonical_backend(backend)
    root = 'curve' if curved else 'planar'
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    return PROJECT_ROOT / 'render_results' / root / run
