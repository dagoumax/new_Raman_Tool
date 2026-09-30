"""BMP spectrum image reader using shared detector reduction."""

from pathlib import Path

from raman_tool.models import Spectrum
from raman_tool.readers.image_reader import _merge_columns, _parse_row_groups, read_image


def read_bmp(
    filepath: str | Path,
    row_groups: str | None = None,
    col_merge: int = 1,
    calibration: tuple[float, float] | None = None,
    row_mode: str = "mean",
) -> Spectrum:
    return read_image(
        filepath,
        format_name="BMP",
        grayscale=False,
        row_groups=row_groups,
        col_merge=col_merge,
        calibration=calibration,
        row_mode=row_mode,
    )
