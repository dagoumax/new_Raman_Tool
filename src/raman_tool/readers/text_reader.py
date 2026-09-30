"""Shared numeric-table parsing and round-trip metadata for TXT and ASC."""

import json
from pathlib import Path
import re

import numpy as np

from raman_tool.models import Spectrum
from raman_tool.safety import check_data_points, check_text_file_size
from raman_tool.validation import validate_calibration


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CALIBRATION = re.compile(
    rf"Calibration\s+raman_shift\s*=\s*({_NUMBER})\s*\*\s*pixel\s*\+\s*({_NUMBER})",
    re.IGNORECASE,
)


def _text_lines(filepath: Path):
    with filepath.open(encoding="utf-8-sig", errors="replace") as stream:
        yield from stream


def _read_metadata(line: str, metadata: dict) -> None:
    header = line[1:].strip()
    if header.startswith("Pixel (px)"):
        metadata["x_unit"] = "px"
    elif header.startswith("Raman Shift (cm^-1)"):
        metadata["x_unit"] = "cm-1"
    elif header.startswith("Calibration"):
        match = _CALIBRATION.fullmatch(header)
        if match is None:
            raise ValueError("Invalid Calibration header")
        metadata["calibration"] = validate_calibration(*map(float, match.groups()))
    elif header.startswith("Processing History"):
        try:
            history = json.loads(header[len("Processing History"):].strip())
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid Processing History header") from exc
        if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
            raise ValueError("Processing History must be a list of records")
        metadata["processing_history"] = history


def read_text_spectrum(
    filepath: str | Path,
    calibration: tuple[float, float] | None = None,
    *,
    allow_headers: bool = False,
) -> Spectrum:
    """Parse whole numeric rows; never extract numbers out of descriptive text.

    Header calibration describes existing coordinates and is not reapplied.
    An explicit calibration only transforms an uncalibrated pixel axis.
    Legacy headerless two-column data continues to use cm-1.
    """
    filepath = Path(filepath)
    check_text_file_size(filepath)
    if calibration is not None:
        calibration = validate_calibration(*calibration)
    metadata = {"filepath": str(filepath.absolute()), "calibration": None}
    rows = []
    columns = None
    for number, raw_line in enumerate(_text_lines(filepath), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("#", ";")):
            _read_metadata(line, metadata)
            continue
        line = re.split(r"[#;]", line, maxsplit=1)[0].strip().removesuffix(",").strip()
        if not line:
            continue
        parts = re.split(r"[,\s]+", line)
        try:
            float(parts[0])
        except ValueError:
            if allow_headers:
                continue
            raise ValueError(f"Invalid numeric data on line {number}: {filepath}")
        try:
            if re.search(r",\s*,", line):
                raise ValueError("Missing column")
            row = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric data on line {number}: {filepath}") from exc
        if columns is None:
            columns = len(row)
        if len(row) != columns:
            raise ValueError(f"Inconsistent column count on line {number}: {filepath}")
        # Additional columns can contain other image rows. This reader's
        # documented two-column selection needs no extra copy of that matrix.
        rows.append(row[:2])
        if len(rows) % 1024 == 0:
            check_data_points(len(rows), "Text spectrum")
    if not rows:
        raise ValueError(f"Text file contains no numeric data: {filepath}")
    check_data_points(len(rows), "Text spectrum")
    data = np.asarray(rows, dtype=np.float64)
    if columns == 1:
        x = np.arange(len(rows), dtype=np.float64)
        intensity = data[:, 0]
        metadata["x_unit"] = "px"
    else:
        x, intensity = data[:, 0], data[:, 1]
        metadata.setdefault("x_unit", "cm-1")
    if calibration is not None and metadata["x_unit"] == "px":
        x = calibration[0] * x + calibration[1]
        metadata.update(calibration=calibration, x_unit="cm-1")
    return Spectrum(x, intensity, filename=filepath.name, metadata=metadata)
