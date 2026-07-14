"""Sorting helpers."""

from pathlib import Path
import re
from typing import Any, Iterable


def natural_sort_key(value: Any) -> list[Any]:
    """Sort names with embedded numbers in numeric order."""
    name = Path(value).name if isinstance(value, (str, Path)) else str(value)
    parts = re.split(r"(\d+)", name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def natural_sorted(values: Iterable[Any]) -> list[Any]:
    return sorted(values, key=natural_sort_key)
