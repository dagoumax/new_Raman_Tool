"""Application configuration loading and persistence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
from typing import Any


DEFAULT_CONFIG: dict[str, dict[str, int]] = {
    "safety": {
        "max_input_file_mb": 512,
        "max_text_file_mb": 128,
        "max_data_points": 2_000_000,
        "min_supported_image_pixels": 2048 * 2048,
        "max_image_pixels": 4096 * 4096,
        "max_image_channels": 4,
        "max_baseline_points": 200_000,
    }
}

_config_cache: dict[str, Any] | None = None


def default_config_path() -> Path:
    env_path = os.environ.get("RAMAN_TOOL_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "RamanTool" / "config.toml"
    return Path.home() / ".raman_tool" / "config.toml"


def _parse_simple_toml(text: str) -> dict[str, Any]:
    """Parse the small TOML subset this app writes.

    Python 3.10 does not include tomllib, so keep config parsing dependency-free
    for our simple [safety] integer settings file.
    """
    data: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            section = data.setdefault(name, {})
            continue
        if "=" not in line or section is None:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        try:
            section[key] = int(value)
        except ValueError:
            section[key] = value
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _coerce_safety(config: dict[str, Any]) -> dict[str, Any]:
    safety = config.setdefault("safety", {})
    defaults = DEFAULT_CONFIG["safety"]
    for key, default in defaults.items():
        try:
            value = int(safety.get(key, default))
        except (TypeError, ValueError):
            value = default
        safety[key] = max(1, value)
    safety["min_supported_image_pixels"] = max(
        defaults["min_supported_image_pixels"],
        safety["min_supported_image_pixels"],
    )
    safety["max_image_pixels"] = max(
        safety["min_supported_image_pixels"],
        safety["max_image_pixels"],
    )
    safety["max_image_channels"] = max(1, safety["max_image_channels"])
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_config_path()
    loaded: dict[str, Any] = {}
    if config_path.exists():
        loaded = _parse_simple_toml(config_path.read_text(encoding="utf-8"))
    return _coerce_safety(_deep_merge(DEFAULT_CONFIG, loaded))


def get_config() -> dict[str, Any]:
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def reload_config() -> dict[str, Any]:
    global _config_cache
    _config_cache = load_config()
    return _config_cache


def save_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    config = _coerce_safety(_deep_merge(DEFAULT_CONFIG, config))
    config_path = Path(path) if path is not None else default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    safety = config["safety"]
    lines = ["[safety]"]
    for key in DEFAULT_CONFIG["safety"]:
        lines.append(f"{key} = {int(safety[key])}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    global _config_cache
    _config_cache = config
    return config_path
