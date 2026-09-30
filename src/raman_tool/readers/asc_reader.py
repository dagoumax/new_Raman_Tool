"""ASC spectrum reader with comma/whitespace tables and descriptive headers."""

from pathlib import Path

from raman_tool.models import Spectrum
from raman_tool.readers.text_reader import read_text_spectrum


def read_asc(
    filepath: str | Path,
    calibration: tuple[float, float] | None = None,
) -> Spectrum:
    return read_text_spectrum(filepath, calibration, allow_headers=True)
