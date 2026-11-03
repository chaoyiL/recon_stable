"""Thin command-line entry point for the local compensation experiment."""

from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physical_model.compensation_utils.runner import main


if __name__ == '__main__':
    main()
