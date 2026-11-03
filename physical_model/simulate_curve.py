"""Render analytic cylindrical bending or launch a curved LUT calibration."""
from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physical_model.model_utils.curve_simulation import main


if __name__ == '__main__':
    main()
