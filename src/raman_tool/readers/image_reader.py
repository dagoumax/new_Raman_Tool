"""Shared image loading, row selection and detector-pixel binning."""

from pathlib import Path
import re

import numpy as np

from raman_tool.models import Spectrum
from raman_tool.safety import (
    check_data_points,
    check_image_array_values,
    check_image_pixels,
    configure_pillow_limits,
)
from raman_tool.validation import integer_parameter, validate_calibration


MAX_ROW_SPECTRA = 500


def _parse_row_groups(spec: str) -> list[tuple[int, int]]:
    """Parse complete 1-based ranges, rejecting misspellings and zero indices."""
    spec = re.sub(r"\s*([-~])\s*", r"\1", spec.strip())
    groups = []
    for part in re.split(r"[,;\s]+", spec):
        if not part:
            continue
        match = re.fullmatch(r"([1-9]\d*)(?:[-~]([1-9]\d*))?", part)
        if match is None:
            raise ValueError(f"Invalid row group: {part!r}; use 1-based ranges such as 1-40, 91-130")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        groups.append((min(start, end), max(start, end)))
    if not groups:
        raise ValueError("Row groups must contain at least one 1-based row range")
    return groups


def _merge_columns(arr: np.ndarray, factor: int) -> np.ndarray:
    """Average detector columns, retaining a shorter final bin if necessary."""
    factor = integer_parameter(factor, "col_merge", minimum=1)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("Column binning requires a nonempty 2D array")
    if factor > arr.shape[1]:
        raise ValueError(f"col_merge ({factor}) exceeds the number of detector pixels ({arr.shape[1]})")
    if factor == 1:
        return arr
    starts = np.arange(0, arr.shape[1], factor)
    counts = np.minimum(factor, arr.shape[1] - starts)
    return np.add.reduceat(arr.astype(np.float64, copy=False), starts, axis=1) / counts


def load_image_array(filepath: str | Path, *, grayscale: bool = False) -> tuple[np.ndarray, tuple]:
    """Decode a bounded image using each format's established color conversion."""
    Image = configure_pillow_limits()
    with Image.open(filepath) as img:
        check_image_pixels(*img.size)
        check_image_array_values(img.width * img.height * len(img.getbands()))
        arr = np.asarray(img.convert("L") if grayscale else img, dtype=np.float64)
    original_shape = arr.shape
    check_image_array_values(arr.size)
    if arr.ndim == 3:
        arr = np.mean(arr[:, :, :3], axis=2)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("Image must contain a nonempty row/column array")
    # A vertical 1D export represents a spectrum, not a one-pixel detector.
    if arr.shape[1] == 1:
        arr = arr.T
    return arr, original_shape


def reduce_image_array(
    arr: np.ndarray, row_groups: str | None = None, col_merge: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected/binned rows and their centers in original pixel units."""
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("Image reduction requires a nonempty 2D array")
    rows, cols = arr.shape
    if row_groups is not None:
        indices = []
        selected = set()
        for start, end in _parse_row_groups(row_groups):
            if end > rows:
                raise ValueError(f"Row group {start}-{end} exceeds image height ({rows})")
            for index in range(start - 1, end):
                if index in selected:
                    raise ValueError(f"Row {index + 1} is selected more than once")
                selected.add(index)
                indices.append(index)
        arr = arr[indices, :]
    arr = _merge_columns(arr, col_merge)
    centers = _merge_columns(np.arange(cols, dtype=np.float64).reshape(1, -1), col_merge)[0]
    return arr, centers


def read_image(
    filepath: str | Path,
    *,
    format_name: str,
    grayscale: bool = False,
    row_groups: str | None = None,
    col_merge: int = 1,
    calibration: tuple[float, float] | None = None,
    row_mode: str = "mean",
) -> Spectrum:
    if row_mode not in {"mean", "sum"}:
        raise ValueError("row_mode must be 'mean' or 'sum'")
    if calibration is not None:
        calibration = validate_calibration(*calibration)
    filepath = Path(filepath)
    arr, original_shape = load_image_array(filepath, grayscale=grayscale)
    total_rows, original_cols = arr.shape
    arr, pixels = reduce_image_array(arr, row_groups=row_groups, col_merge=col_merge)
    check_data_points(len(pixels), "Image spectrum")
    x = pixels if calibration is None else calibration[0] * pixels + calibration[1]
    intensity = np.sum(arr, axis=0) if row_mode == "sum" else np.mean(arr, axis=0)
    individual_rows = [row.copy() for row in arr[:MAX_ROW_SPECTRA]]
    return Spectrum(
        x,
        intensity,
        filename=filepath.name,
        metadata={
            "filepath": str(filepath.absolute()),
            "format": format_name,
            "image_shape": original_shape,
            "row_groups": row_groups,
            "row_mode": row_mode,
            "col_merge": int(col_merge),
            "selected_rows": arr.shape[0],
            "output_cols": arr.shape[1],
            "individual_rows": individual_rows,
            "individual_rows_truncated": arr.shape[0] > MAX_ROW_SPECTRA,
            "total_rows": total_rows,
            "original_cols": original_cols,
            "pixel_centers": pixels,
            "calibration": calibration,
            "x_unit": "cm-1" if calibration is not None else "px",
        },
    )
