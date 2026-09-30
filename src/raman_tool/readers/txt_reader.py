"""TXT spectrum reader; one column is intensity, two columns are x/y."""

from pathlib import Path

from raman_tool.models import Spectrum
from raman_tool.readers.text_reader import read_text_spectrum


def read_txt(
    filepath: str | Path,
    calibration: tuple[float, float] | None = None,
) -> Spectrum:
    return read_text_spectrum(filepath, calibration)
