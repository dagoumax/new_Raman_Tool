"""Shared import/processing entry points used by desktop and scripted workflows."""

from pathlib import Path

from raman_tool.models import Spectrum
from raman_tool.readers import SUPPORTED_FORMATS, read_file
from raman_tool.processing import subtract_baseline
from raman_tool.sorting import natural_sorted


def collect_spectrum_files(directory: str | Path) -> list[Path]:
    """Visit each file once on case-sensitive and case-insensitive filesystems."""
    return natural_sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS
    )


def load_spectrum(
    filepath: str | Path, *, row_groups=None, col_merge=1, calibration=None,
    row_mode="mean", baseline_options: dict | None = None,
) -> Spectrum:
    spectrum = read_file(
        filepath, row_groups=row_groups, col_merge=col_merge,
        calibration=calibration, row_mode=row_mode,
    )
    if baseline_options is not None:
        spectrum = subtract_baseline(spectrum, **baseline_options)
    return spectrum
