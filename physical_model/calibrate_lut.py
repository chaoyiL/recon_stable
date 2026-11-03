"""Calibrate a signed linear-RGB to local-slope LUT from analytic simulation."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physical_model.cali_utils.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
