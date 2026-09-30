"""Shared input rules used by readers, processing functions and user interfaces."""

from __future__ import annotations

import numpy as np


def finite_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def positive_float(value, name: str) -> float:
    number = finite_float(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be > 0")
    return number


def integer_parameter(value, name: str, *, minimum: int = 0) -> int:
    number = finite_float(value, name)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def numeric_vector(values, name: str, *, minimum: int = 1) -> np.ndarray:
    try:
        array = np.asarray(values)
        if array.dtype.kind in {"c", "b"}:
            raise ValueError("complex and boolean data are unsupported")
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric 1D array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if array.size < minimum:
        raise ValueError(f"{name} requires at least {minimum} data points")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_spectrum_arrays(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = numeric_vector(x, "raman_shift")
    y = numeric_vector(y, "intensity")
    if x.shape != y.shape:
        raise ValueError(f"raman_shift shape {x.shape} does not match intensity shape {y.shape}")
    if x.size > 1 and not (np.all(x[1:] > x[:-1]) or np.all(x[1:] < x[:-1])):
        raise ValueError("raman_shift must be strictly monotonic (ascending or descending), without duplicates")
    return x, y


def validate_range(start, end, name: str = "range") -> tuple[float, float]:
    start = finite_float(start, f"{name} start")
    end = finite_float(end, f"{name} end")
    if start >= end:
        raise ValueError(f"{name} start must be less than end")
    return start, end


def region_mask(x: np.ndarray, start, end, *, name: str = "region", minimum: int = 1) -> np.ndarray:
    start, end = validate_range(start, end, name)
    mask = (x >= start) & (x <= end)
    count = np.count_nonzero(mask)
    if count < minimum:
        raise ValueError(f"{name} [{start}, {end}] requires at least {minimum} data points; found {count}")
    return mask


def validate_calibration(slope, intercept) -> tuple[float, float]:
    slope = finite_float(slope, "calibration slope")
    intercept = finite_float(intercept, "calibration intercept")
    if slope == 0:
        raise ValueError("calibration slope must be nonzero")
    return slope, intercept


def calibration_from_points(pixel1, shift1, pixel2, shift2) -> tuple[float, float]:
    """Validate two detector references and return one shared linear calibration."""
    pixel1 = finite_float(pixel1, "calibration pixel1")
    pixel2 = finite_float(pixel2, "calibration pixel2")
    shift1 = finite_float(shift1, "calibration shift1")
    shift2 = finite_float(shift2, "calibration shift2")
    if pixel1 == pixel2:
        raise ValueError("Calibration pixel positions must be different")
    slope = (shift2 - shift1) / (pixel2 - pixel1)
    return validate_calibration(slope, shift1 - slope * pixel1)
