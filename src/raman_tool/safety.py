"""Safety limits for local file parsing and numerical processing."""

from __future__ import annotations

from pathlib import Path
import warnings

from raman_tool.config import get_config, reload_config


def _safety_config() -> dict:
    return get_config()["safety"]


def refresh_limits() -> None:
    """Reload config-backed module constants for compatibility with callers."""
    reload_config()
    _sync_constants()


def _sync_constants() -> None:
    global MAX_INPUT_FILE_BYTES, MAX_TEXT_FILE_BYTES, MAX_DATA_POINTS
    global MIN_SUPPORTED_IMAGE_PIXELS, MAX_IMAGE_PIXELS, MAX_IMAGE_ARRAY_VALUES
    global MAX_BASELINE_POINTS

    safety = _safety_config()
    MAX_INPUT_FILE_BYTES = int(safety["max_input_file_mb"]) * 1024 * 1024
    MAX_TEXT_FILE_BYTES = int(safety["max_text_file_mb"]) * 1024 * 1024
    MAX_DATA_POINTS = int(safety["max_data_points"])
    MIN_SUPPORTED_IMAGE_PIXELS = int(safety["min_supported_image_pixels"])
    MAX_IMAGE_PIXELS = int(safety["max_image_pixels"])
    MAX_IMAGE_ARRAY_VALUES = MAX_IMAGE_PIXELS * int(safety["max_image_channels"])
    MAX_BASELINE_POINTS = int(safety["max_baseline_points"])


class InputLimitError(ValueError):
    """Raised when an input would consume too much memory or CPU."""


def check_file_size(filepath: str | Path, max_bytes: int | None = None) -> None:
    _sync_constants()
    limit = MAX_INPUT_FILE_BYTES if max_bytes is None else max_bytes
    path = Path(filepath)
    size = path.stat().st_size
    if size > limit:
        raise InputLimitError(
            f"Input file is too large: {path} ({size} bytes > {limit} bytes)"
        )


def check_text_file_size(filepath: str | Path) -> None:
    _sync_constants()
    check_file_size(filepath, MAX_TEXT_FILE_BYTES)


def check_data_points(count: int, context: str = "data") -> None:
    _sync_constants()
    if count > MAX_DATA_POINTS:
        raise InputLimitError(
            f"{context} has too many points: {count} > {MAX_DATA_POINTS}"
        )


def check_image_pixels(width: int, height: int) -> None:
    _sync_constants()
    pixels = int(width) * int(height)
    if pixels > MAX_IMAGE_PIXELS:
        raise InputLimitError(
            f"Image is too large: {width}x{height} ({pixels} pixels > {MAX_IMAGE_PIXELS})"
        )


def check_image_array_values(count: int) -> None:
    _sync_constants()
    if count > MAX_IMAGE_ARRAY_VALUES:
        raise InputLimitError(
            f"Image array is too large: {count} values > {MAX_IMAGE_ARRAY_VALUES}"
        )


def check_baseline_points(count: int) -> None:
    _sync_constants()
    if count > MAX_BASELINE_POINTS:
        raise InputLimitError(
            f"arPLS input has too many points: {count} > {MAX_BASELINE_POINTS}"
        )


def configure_pillow_limits():
    _sync_constants()
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    return Image


_sync_constants()
