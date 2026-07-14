"""Configurable gas Raman peak library."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from raman_tool.config import default_config_path


DEFAULT_GAS_LIBRARY: dict[str, dict[str, Any]] = {
    "N2": {
        "name": "氮气",
        "center": 2330.0,
        "half_width": 20.0,
        "coefficient": 1.0,
        "color": "#2c3e50",
        "enabled": True,
        "quantitative": True,
    },
    "O2": {
        "name": "氧气",
        "center": 1555.0,
        "half_width": 25.0,
        "coefficient": 1.0,
        "color": "#e74c3c",
        "enabled": True,
        "quantitative": True,
    },
    "CO2": {
        "name": "二氧化碳",
        "center": 1388.0,
        "half_width": 15.0,
        "coefficient": 1.0,
        "color": "#27ae60",
        "enabled": True,
        "quantitative": True,
    },
    "H2O": {
        "name": "水蒸气",
        "center": 3657.0,
        "half_width": 25.0,
        "coefficient": 1.0,
        "color": "#3498db",
        "enabled": True,
        "quantitative": False,
    },
    "CH4": {
        "name": "甲烷",
        "center": 2917.0,
        "half_width": 25.0,
        "coefficient": 1.0,
        "color": "#8e44ad",
        "enabled": True,
        "quantitative": False,
    },
    "H2": {
        "name": "氢气",
        "center": 4155.0,
        "half_width": 25.0,
        "coefficient": 1.0,
        "color": "#f39c12",
        "enabled": True,
        "quantitative": False,
    },
    "CO": {
        "name": "一氧化碳",
        "center": 2143.0,
        "half_width": 25.0,
        "coefficient": 1.0,
        "color": "#1abc9c",
        "enabled": True,
        "quantitative": False,
    },
    "SO2": {
        "name": "二氧化硫",
        "center": 1151.0,
        "half_width": 25.0,
        "coefficient": 1.0,
        "color": "#c0392b",
        "enabled": True,
        "quantitative": False,
    },
    "NO": {
        "name": "一氧化氮",
        "center": 1876.0,
        "half_width": 25.0,
        "coefficient": 1.0,
        "color": "#7f8c8d",
        "enabled": True,
        "quantitative": False,
    },
    "NH3": {
        "name": "氨气",
        "center": 3334.0,
        "half_width": 25.0,
        "coefficient": 1.0,
        "color": "#2ecc71",
        "enabled": True,
        "quantitative": False,
    },
    "C2H6": {
        "name": "乙烷",
        "center": 2954.0,
        "half_width": 25.0,
        "coefficient": 1.0,
        "color": "#d35400",
        "enabled": True,
        "quantitative": False,
    },
    "CO2_V2": {
        "name": "二氧化碳 v2",
        "label": "CO2(v2)",
        "center": 1285.0,
        "half_width": 15.0,
        "coefficient": 1.0,
        "color": "#27ae60",
        "enabled": True,
        "quantitative": False,
    },
}

_library_cache: dict[str, dict[str, Any]] | None = None
_library_cache_path: Path | None = None


def default_gas_library_path() -> Path:
    env_path = os.environ.get("RAMAN_TOOL_GAS_LIBRARY")
    if env_path:
        return Path(env_path).expanduser()
    return default_config_path().with_name("gas_library.json")


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_key(key: str) -> str:
    allowed = []
    for ch in key.strip():
        if ch.isalnum() or ch in {"_", "-", "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"}:
            allowed.append(ch)
    return "".join(allowed)


def _default_for_key(key: str) -> dict[str, Any]:
    for default_key, entry in DEFAULT_GAS_LIBRARY.items():
        if default_key.casefold() == key.casefold():
            return entry
    return {}


def _coerce_entry(key: str, raw: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    clean_key = _coerce_key(key)
    if not clean_key:
        return None
    default = _default_for_key(clean_key)
    center = _to_float(raw.get("center", default.get("center", 0.0)), 0.0)
    half_width = _to_float(raw.get("half_width", default.get("half_width", 25.0)), 25.0)
    coefficient = _to_float(raw.get("coefficient", default.get("coefficient", 1.0)), 1.0)
    if center <= 0:
        return None
    return clean_key, {
        "name": str(raw.get("name", default.get("name", clean_key)) or clean_key),
        "label": str(raw.get("label", default.get("label", clean_key)) or clean_key),
        "center": center,
        "half_width": max(0.1, half_width),
        "coefficient": max(0.0, coefficient),
        "color": str(raw.get("color", default.get("color", "#999999")) or "#999999"),
        "enabled": _to_bool(raw.get("enabled", default.get("enabled", True)), True),
        "quantitative": _to_bool(raw.get("quantitative", default.get("quantitative", False)), False),
    }


def coerce_gas_library(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    library: dict[str, dict[str, Any]] = {}
    for key, entry in raw.items():
        if isinstance(key, str) and isinstance(entry, dict):
            coerced = _coerce_entry(key, entry)
            if coerced is not None:
                clean_key, clean_entry = coerced
                library[clean_key] = clean_entry
    return library


def load_gas_library(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    lib_path = Path(path) if path is not None else default_gas_library_path()
    if not lib_path.exists():
        return coerce_gas_library(deepcopy(DEFAULT_GAS_LIBRARY))
    data = json.loads(lib_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Gas library must contain a JSON object")
    library = coerce_gas_library(data)
    if not library:
        raise ValueError("Gas library cannot be empty")
    return library


def get_gas_library() -> dict[str, dict[str, Any]]:
    global _library_cache, _library_cache_path
    current_path = default_gas_library_path()
    if _library_cache is None or _library_cache_path != current_path:
        _library_cache = load_gas_library(current_path)
        _library_cache_path = current_path
    return deepcopy(_library_cache)


def reload_gas_library() -> dict[str, dict[str, Any]]:
    global _library_cache, _library_cache_path
    _library_cache_path = default_gas_library_path()
    _library_cache = load_gas_library(_library_cache_path)
    return deepcopy(_library_cache)


def save_gas_library(library: dict[str, dict[str, Any]], path: str | Path | None = None) -> Path:
    clean = coerce_gas_library(library)
    if not clean:
        raise ValueError("Gas library cannot be empty")
    lib_path = Path(path) if path is not None else default_gas_library_path()
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    lib_path.write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    global _library_cache, _library_cache_path
    if lib_path == default_gas_library_path():
        _library_cache = clean
        _library_cache_path = lib_path
    return lib_path


def reset_gas_library(path: str | Path | None = None) -> Path:
    return save_gas_library(DEFAULT_GAS_LIBRARY, path)


def get_quantitative_gases() -> tuple[str, ...]:
    library = get_gas_library()
    gases = tuple(
        key for key, entry in library.items()
        if entry.get("enabled", True) and entry.get("quantitative", False)
    )
    if gases:
        return gases
    core = tuple(gas for gas in ("O2", "N2", "CO2") if gas in library and library[gas].get("enabled", True))
    if core:
        return core
    return tuple(key for key, entry in library.items() if entry.get("enabled", True))


def get_reference_peaks() -> dict[float, str]:
    peaks: dict[float, str] = {}
    for key, entry in get_gas_library().items():
        if not entry.get("enabled", True):
            continue
        label = str(entry.get("label") or key)
        peaks[float(entry["center"])] = label
    return peaks


def get_gas_colors() -> dict[str, str]:
    colors: dict[str, str] = {}
    for key, entry in get_gas_library().items():
        color = str(entry.get("color") or "#999999")
        colors[key] = color
        colors[str(entry.get("label") or key)] = color
    return colors


def find_gas_key(query: str, library: dict[str, dict[str, Any]] | None = None) -> str | None:
    if library is None:
        library = get_gas_library()
    clean_query = _coerce_key(str(query))
    if clean_query in library:
        return clean_query
    folded = clean_query.casefold()
    for key in library:
        if key.casefold() == folded:
            return key
    return None


def get_gas_choices() -> list[tuple[str, str]]:
    return [
        (key, f"{key} ({entry.get('name', key)})")
        for key, entry in get_gas_library().items()
        if entry.get("enabled", True)
    ]
