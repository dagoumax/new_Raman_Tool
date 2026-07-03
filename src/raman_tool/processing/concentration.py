"""Gas concentration calculation.

The main concentration model follows Raman_smoothing_processing:
extract O2/N2/CO2 peak intensities, apply per-gas coefficients, then
normalize the three corrected intensities. Smoothing is intentionally not
applied here; smoothing belongs to batch/display post-processing.
"""

from __future__ import annotations

import numpy as np

from raman_tool.models import Spectrum


GAS_RAMAN_SHIFTS = {
    "N2": {"center": 2330.0, "half_width": 20.0, "coefficient": 1.0, "name": "氮气"},
    "O2": {"center": 1555.0, "half_width": 25.0, "coefficient": 1.0, "name": "氧气"},
    "CO2": {"center": 1388.0, "half_width": 15.0, "coefficient": 1.0, "name": "二氧化碳"},
    # Extra gases are kept for compatibility with the old single-gas API.
    "H2O": {"center": 3657.0, "half_width": 25.0, "coefficient": 1.0, "name": "水蒸气"},
    "CH4": {"center": 2917.0, "half_width": 25.0, "coefficient": 1.0, "name": "甲烷"},
    "H2": {"center": 4155.0, "half_width": 25.0, "coefficient": 1.0, "name": "氢气"},
    "CO": {"center": 2143.0, "half_width": 25.0, "coefficient": 1.0, "name": "一氧化碳"},
    "SO2": {"center": 1151.0, "half_width": 25.0, "coefficient": 1.0, "name": "二氧化硫"},
    "NO": {"center": 1876.0, "half_width": 25.0, "coefficient": 1.0, "name": "一氧化氮"},
    "NH3": {"center": 3334.0, "half_width": 25.0, "coefficient": 1.0, "name": "氨气"},
    "C2H6": {"center": 2954.0, "half_width": 25.0, "coefficient": 1.0, "name": "乙烷"},
}

CONCENTRATION_GASES = ("O2", "N2", "CO2")


def _find_peak_indices(raman_shift: np.ndarray, center: float, half_width: float) -> np.ndarray:
    lo = center - half_width
    hi = center + half_width
    return np.where((raman_shift >= lo) & (raman_shift <= hi))[0]


def _peak_intensity_area(y: np.ndarray, x: np.ndarray, indices: np.ndarray) -> float:
    if len(indices) < 3:
        return 0.0
    x_win = x[indices]
    y_win = y[indices]
    raw_area = float(np.trapezoid(y_win, x_win))
    endpoint_area = float((y_win[0] + y_win[-1]) * (x_win[-1] - x_win[0]) / 2.0)
    return max(raw_area - endpoint_area, 0.0)


def _peak_intensity_max(y: np.ndarray, indices: np.ndarray) -> float:
    if len(indices) < 3:
        return 0.0
    y_win = y[indices]
    endpoint_baseline = float((y_win[0] + y_win[-1]) / 2.0)
    return max(float(np.max(y_win)) - endpoint_baseline, 0.0)


def _peak_info(x: np.ndarray, y: np.ndarray, indices: np.ndarray, strategy: str) -> dict:
    if len(indices) == 0:
        return {"center": 0.0, "height": 0.0, "area": 0.0, "intensity": 0.0}

    y_win = y[indices]
    idx_max = indices[int(np.argmax(y_win))]
    height = _peak_intensity_max(y, indices)
    area = _peak_intensity_area(y, x, indices)
    intensity = area if strategy == "peak_area" else height
    return {
        "center": float(x[idx_max]),
        "height": float(height),
        "area": float(area),
        "intensity": float(intensity),
    }


def calculate_gas_concentrations(
    spectrum: Spectrum,
    strategy: str = "peak_max",
    coefficients: dict[str, float] | None = None,
    windows: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """Calculate raw O2/N2/CO2 concentrations without temporal smoothing.

    Args:
        spectrum: Input spectrum. Baseline correction should be applied before
            calling this function if needed.
        strategy: ``"peak_max"`` or ``"peak_area"``.
        coefficients: Optional per-gas multipliers, e.g. ``{"O2": 1.0}``.
        windows: Optional per-gas ``(center, half_width)`` overrides.

    Returns:
        A dictionary containing fractions, percentages, peak details and the
        corrected intensity denominator.
    """
    strategy = strategy.lower()
    if strategy not in {"peak_max", "peak_area"}:
        raise ValueError("strategy must be 'peak_max' or 'peak_area'")

    x = np.asarray(spectrum.raman_shift, dtype=np.float64)
    y = np.asarray(spectrum.intensity, dtype=np.float64)

    peaks: dict[str, dict] = {}
    corrected: dict[str, float] = {}
    for gas in CONCENTRATION_GASES:
        info = GAS_RAMAN_SHIFTS[gas]
        center = float(info["center"])
        half_width = float(info["half_width"])
        if windows and gas in windows:
            center, half_width = windows[gas]

        indices = _find_peak_indices(x, center, half_width)
        peak = _peak_info(x, y, indices, strategy)
        coeff = float(coefficients.get(gas, info["coefficient"]) if coefficients else info["coefficient"])
        peaks[gas] = {
            **peak,
            "window_center": float(center),
            "half_width": float(half_width),
            "coefficient": coeff,
        }
        corrected[gas] = peak["intensity"] * coeff

    denominator = float(sum(corrected.values()))
    fractions = {
        gas: (corrected[gas] / denominator if denominator > 1e-15 else 0.0)
        for gas in CONCENTRATION_GASES
    }
    percentages = {gas: fractions[gas] * 100.0 for gas in CONCENTRATION_GASES}

    return {
        "strategy": strategy,
        "fractions": fractions,
        "percentages": percentages,
        "peaks": peaks,
        "corrected_intensities": corrected,
        "denominator": denominator,
    }


def calculate_concentration(
    spectrum: Spectrum,
    gas_name: str,
    window: float = 10.0,
    reference_gas: str = "N2",
    reference_concentration: float = 78.0,
    strategy: str = "peak_max",
) -> dict:
    """Calculate one gas concentration using the new O2/N2/CO2 model.

    ``reference_gas`` and ``reference_concentration`` are accepted for backward
    compatibility but are no longer used by the O2/N2/CO2 normalized model.
    """
    gas_key = gas_name.upper()
    gas_info = GAS_RAMAN_SHIFTS.get(gas_key)
    if gas_info is None:
        raise ValueError(f"未知气体: {gas_name}. 支持的气体: {list(GAS_RAMAN_SHIFTS.keys())}")

    windows = None
    if gas_key in CONCENTRATION_GASES:
        windows = {gas_key: (float(gas_info["center"]), float(window))}

    all_result = calculate_gas_concentrations(spectrum, strategy=strategy, windows=windows)
    peak = all_result["peaks"].get(gas_key, {"center": 0.0, "area": 0.0, "height": 0.0, "intensity": 0.0})
    concentration = all_result["percentages"].get(gas_key, 0.0)

    ref_key = reference_gas.upper()
    ref_info = GAS_RAMAN_SHIFTS.get(ref_key, {"name": reference_gas})
    ref_peak = all_result["peaks"].get(ref_key, {"center": 0.0, "area": 0.0, "height": 0.0, "intensity": 0.0})

    return {
        "gas": gas_info["name"],
        "gas_key": gas_key,
        "concentration": round(concentration, 4),
        "peak_center": peak["center"],
        "peak_area": round(peak["area"], 4),
        "peak_height": round(peak["height"], 4),
        "peak_intensity": round(peak["intensity"], 4),
        "reference_gas": ref_info["name"],
        "reference_peak_center": ref_peak["center"],
        "reference_peak_area": round(ref_peak["area"], 4),
        "reference_concentration": reference_concentration,
        "all_concentrations": all_result,
    }


def calculate_concentration_from_ratio(
    area_ratio: float,
    gas_name: str,
    reference_gas: str = "N2",
    reference_concentration: float = 78.0,
) -> float:
    """Backward-compatible ratio helper.

    The GUI now uses normalized O2/N2/CO2 concentrations, but this helper is
    kept for existing scripts that pass a precomputed target/reference ratio.
    """
    gas_info = GAS_RAMAN_SHIFTS.get(gas_name.upper())
    if gas_info is None:
        raise ValueError(f"未知气体: {gas_name}")
    return reference_concentration * area_ratio
