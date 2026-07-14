"""Workflow preset storage for processing and display settings."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from raman_tool.config import default_config_path


DEFAULT_WORKFLOW_PRESETS: dict[str, dict[str, Any]] = {
    "标准气体分析": {
        "baseline_method": "arPLS",
        "baseline_lam": 100000.0,
        "baseline_degree": 3,
        "auto_baseline": True,
        "batch_baseline": True,
        "concentration_gas": "N2",
        "concentration_window": 10.0,
        "concentration_strategy": "peak_max",
        "row_mode": "mean",
        "col_merge": 1,
        "row_groups": "",
        "show_individual_rows": False,
        "calibration_px1": "",
        "calibration_shift1": "",
        "calibration_px2": "",
        "calibration_shift2": "",
        "show_gas_peaks": True,
        "show_auto_peaks": True,
    },
    "峰面积定量": {
        "baseline_method": "arPLS",
        "baseline_lam": 100000.0,
        "baseline_degree": 3,
        "auto_baseline": True,
        "batch_baseline": True,
        "concentration_gas": "N2",
        "concentration_window": 12.0,
        "concentration_strategy": "peak_area",
        "row_mode": "mean",
        "col_merge": 1,
        "row_groups": "",
        "show_individual_rows": False,
        "calibration_px1": "",
        "calibration_shift1": "",
        "calibration_px2": "",
        "calibration_shift2": "",
        "show_gas_peaks": True,
        "show_auto_peaks": True,
    },
    "快速查看": {
        "baseline_method": "arPLS",
        "baseline_lam": 100000.0,
        "baseline_degree": 3,
        "auto_baseline": False,
        "batch_baseline": False,
        "concentration_gas": "N2",
        "concentration_window": 10.0,
        "concentration_strategy": "peak_max",
        "row_mode": "mean",
        "col_merge": 1,
        "row_groups": "",
        "show_individual_rows": False,
        "calibration_px1": "",
        "calibration_shift1": "",
        "calibration_px2": "",
        "calibration_shift2": "",
        "show_gas_peaks": False,
        "show_auto_peaks": False,
    },
}

PRESET_KEYS = set(next(iter(DEFAULT_WORKFLOW_PRESETS.values())).keys())


def default_presets_path() -> Path:
    env_path = os.environ.get("RAMAN_TOOL_PRESETS")
    if env_path:
        return Path(env_path).expanduser()
    return default_config_path().with_name("presets.json")


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _coerce_preset(raw: dict[str, Any]) -> dict[str, Any]:
    preset = deepcopy(DEFAULT_WORKFLOW_PRESETS["标准气体分析"])
    for key, value in raw.items():
        if key in PRESET_KEYS:
            preset[key] = value

    preset["baseline_method"] = "poly" if str(preset["baseline_method"]).lower() == "poly" else "arPLS"
    preset["baseline_lam"] = max(1.0, _to_float(preset["baseline_lam"], 100000.0))
    preset["baseline_degree"] = min(5, max(1, _to_int(preset["baseline_degree"], 3)))
    preset["auto_baseline"] = _to_bool(preset["auto_baseline"], True)
    preset["batch_baseline"] = _to_bool(preset["batch_baseline"], True)
    preset["concentration_gas"] = str(preset["concentration_gas"] or "N2").strip() or "N2"
    preset["concentration_window"] = max(0.1, _to_float(preset["concentration_window"], 10.0))
    preset["concentration_strategy"] = (
        "peak_area" if str(preset["concentration_strategy"]) == "peak_area" else "peak_max"
    )
    preset["row_mode"] = "sum" if str(preset["row_mode"]) == "sum" else "mean"
    preset["col_merge"] = max(1, _to_int(preset["col_merge"], 1))
    preset["row_groups"] = str(preset["row_groups"] or "").strip()
    preset["show_individual_rows"] = _to_bool(preset["show_individual_rows"], False)
    for key in ("calibration_px1", "calibration_shift1", "calibration_px2", "calibration_shift2"):
        preset[key] = str(preset[key] or "").strip()
    preset["show_gas_peaks"] = _to_bool(preset["show_gas_peaks"], True)
    preset["show_auto_peaks"] = _to_bool(preset["show_auto_peaks"], True)
    return preset


def _load_stored_presets(preset_path: Path) -> dict[str, Any]:
    if not preset_path.exists():
        return {}
    data = json.loads(preset_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Preset file must contain a JSON object")
    return data


def load_workflow_presets(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    preset_path = Path(path) if path is not None else default_presets_path()
    presets = deepcopy(DEFAULT_WORKFLOW_PRESETS)
    for name, raw in _load_stored_presets(preset_path).items():
        if not isinstance(name, str) or not name.strip():
            continue
        clean_name = name.strip()
        if raw is None:
            presets.pop(clean_name, None)
        elif isinstance(raw, dict):
            presets[clean_name] = _coerce_preset(raw)
    return {name: _coerce_preset(value) for name, value in presets.items()}


def save_workflow_preset(
    name: str,
    preset: dict[str, Any],
    path: str | Path | None = None,
) -> Path:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Preset name cannot be empty")

    preset_path = Path(path) if path is not None else default_presets_path()
    try:
        presets = _load_stored_presets(preset_path)
    except (OSError, ValueError, json.JSONDecodeError):
        presets = {}
    presets[clean_name] = _coerce_preset(preset)

    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text(
        json.dumps(presets, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return preset_path



def delete_workflow_preset(name: str, path: str | Path | None = None) -> Path:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Preset name cannot be empty")

    preset_path = Path(path) if path is not None else default_presets_path()
    available = load_workflow_presets(preset_path)
    if clean_name not in available:
        raise KeyError(f"Preset not found: {clean_name}")
    stored = _load_stored_presets(preset_path)
    if clean_name in DEFAULT_WORKFLOW_PRESETS:
        stored[clean_name] = None
    else:
        stored.pop(clean_name, None)

    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text(
        json.dumps(stored, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return preset_path
