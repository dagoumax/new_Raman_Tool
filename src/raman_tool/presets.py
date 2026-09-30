"""Workflow preset storage for processing and display settings."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from raman_tool.config import default_config_path
from raman_tool.validation import positive_float, integer_parameter, calibration_from_points
from raman_tool.readers.image_reader import _parse_row_groups


DEFAULT_WORKFLOW_PRESETS: dict[str, dict[str, Any]] = {
    "标准气体分析": {
        "baseline_method": "arPLS",
        "baseline_lam": 100000.0,
        "baseline_degree": 3,
        "auto_baseline": True,
        "batch_baseline": True,
        "concentration_gas": "",
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
        "concentration_gas": "",
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
        "concentration_gas": "",
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


def _to_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def validate_workflow_preset(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill omitted defaults and reject values that processing would reject.

    A preset never silently substitutes or clamps explicitly supplied numerical
    parameters. Spectrum-dependent constraints are checked during processing.
    """
    if not isinstance(raw, dict):
        raise ValueError("Workflow preset must be an object")
    preset = deepcopy(DEFAULT_WORKFLOW_PRESETS["标准气体分析"])
    for key, value in raw.items():
        if key in PRESET_KEYS:
            preset[key] = value

    method = str(preset["baseline_method"]).lower()
    if method not in {"poly", "arpls"}:
        raise ValueError("baseline_method must be arPLS or poly")
    preset["baseline_method"] = "poly" if method == "poly" else "arPLS"
    preset["baseline_lam"] = positive_float(preset["baseline_lam"], "baseline_lam")
    preset["baseline_degree"] = integer_parameter(preset["baseline_degree"], "baseline_degree", minimum=0)
    preset["auto_baseline"] = _to_bool(preset["auto_baseline"], "auto_baseline")
    preset["batch_baseline"] = _to_bool(preset["batch_baseline"], "batch_baseline")
    preset["concentration_gas"] = str(preset["concentration_gas"] or "").strip()
    preset["concentration_window"] = positive_float(preset["concentration_window"], "concentration_window")
    if str(preset["concentration_strategy"]) not in {"peak_max", "peak_area"}:
        raise ValueError("concentration_strategy must be peak_max or peak_area")
    if str(preset["row_mode"]) not in {"sum", "mean"}:
        raise ValueError("row_mode must be sum or mean")
    preset["col_merge"] = integer_parameter(preset["col_merge"], "col_merge", minimum=1)
    preset["row_groups"] = str(preset["row_groups"] or "").strip()
    if preset["row_groups"]:
        _parse_row_groups(preset["row_groups"])
    preset["show_individual_rows"] = _to_bool(preset["show_individual_rows"], "show_individual_rows")
    for key in ("calibration_px1", "calibration_shift1", "calibration_px2", "calibration_shift2"):
        preset[key] = "" if preset[key] is None else str(preset[key]).strip()
    calibration = [preset[key] for key in
                   ("calibration_px1", "calibration_shift1", "calibration_px2", "calibration_shift2")]
    if any(calibration):
        if not all(calibration):
            raise ValueError("Two-point calibration requires all four fields")
        calibration_from_points(*calibration)
    preset["show_gas_peaks"] = _to_bool(preset["show_gas_peaks"], "show_gas_peaks")
    preset["show_auto_peaks"] = _to_bool(preset["show_auto_peaks"], "show_auto_peaks")
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
            try:
                presets[clean_name] = validate_workflow_preset(raw)
            except ValueError as exc:
                raise ValueError(f"Invalid preset {clean_name!r}: {exc}") from exc
        else:
            raise ValueError(f"Invalid preset {clean_name!r}: expected an object or deletion marker")
    return {name: validate_workflow_preset(value) for name, value in presets.items()}


def save_workflow_preset(
    name: str,
    preset: dict[str, Any],
    path: str | Path | None = None,
) -> Path:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Preset name cannot be empty")

    preset_path = Path(path) if path is not None else default_presets_path()
    validated = validate_workflow_preset(preset)
    presets = _load_stored_presets(preset_path)
    presets[clean_name] = validated

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
