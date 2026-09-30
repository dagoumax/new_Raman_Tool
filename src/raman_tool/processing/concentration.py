"""Gas-library-driven Raman concentration calculation."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from raman_tool.models import Spectrum


from raman_tool.gas_library import DEFAULT_GAS_LIBRARY, coerce_gas_library, find_gas_key, get_gas_library
from raman_tool.validation import finite_float, positive_float, validate_spectrum_arrays


GAS_RAMAN_SHIFTS = DEFAULT_GAS_LIBRARY


def _find_peak_indices(raman_shift: np.ndarray, center: float, half_width: float) -> np.ndarray:
    lo = center - half_width
    hi = center + half_width
    return np.where((raman_shift >= lo) & (raman_shift <= hi))[0]


def _peak_info(x: np.ndarray, y: np.ndarray, indices: np.ndarray, strategy: str) -> dict:
    if len(indices) < 3:
        return {"center": 0.0, "height": 0.0, "area": 0.0, "intensity": 0.0}
    x_win = x[indices]
    y_win = y[indices]
    if x_win[0] > x_win[-1]:
        x_win, y_win = x_win[::-1], y_win[::-1]
    # Subtract a line between the window endpoints before integration. This
    # prevents sloped backgrounds becoming peak heights and avoids subtracting
    # two large areas when the actual peak is small.
    fraction = (x_win - x_win[0]) / (x_win[-1] - x_win[0])
    residual = (y_win - y_win[0]) - fraction * (y_win[-1] - y_win[0])
    # Only suppress floating-point subtraction error; this is not an
    # experimental noise estimate or a gas detection limit.
    roundoff = np.max(np.abs(y_win)) * np.finfo(np.float64).eps * 32
    residual[np.abs(residual) <= roundoff] = 0.0
    idx_max = int(np.argmax(residual))
    height = max(float(residual[idx_max]), 0.0)
    area = max(float(trapezoid(residual, x_win)), 0.0)
    intensity = area if strategy == "peak_area" else height
    return {
        "center": float(x_win[idx_max]),
        "height": float(height),
        "area": float(area),
        "intensity": float(intensity),
    }


def calculate_gas_concentrations(
    spectrum: Spectrum,
    strategy: str = "peak_max",
    coefficients: dict[str, float] | None = None,
    windows: dict[str, tuple[float, float]] | None = None,
    *,
    library: dict | None = None,
) -> dict:
    """Calculate weighted signal percentages for enabled quantitative gases.

    These percentages are normalized within the selected gas library. They are
    not independently validated physical concentrations. Converting them to
    composition requires response coefficients calibrated with reference
    samples, representative gas selection and validated acquisition conditions.

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
    strategy = str(strategy).lower()
    if strategy not in {"peak_max", "peak_area"}:
        raise ValueError("strategy must be 'peak_max' or 'peak_area'")

    if not spectrum.is_raman_shift:
        raise ValueError("气体定量需要 cm⁻¹ 横轴；请先完成像素到拉曼位移校准")

    x, y = validate_spectrum_arrays(spectrum.raman_shift, spectrum.intensity)

    gas_library = get_gas_library() if library is None else coerce_gas_library(library)
    concentration_gases = tuple(
        key for key, info in gas_library.items()
        if info.get("enabled", True) and info.get("quantitative", False)
    )
    if not concentration_gases:
        raise ValueError("气体峰位库中没有启用定量的气体")
    for name, overrides in (("coefficients", coefficients), ("windows", windows)):
        if overrides is not None:
            unknown = set(overrides) - set(concentration_gases)
            if unknown:
                raise ValueError(f"{name} contain unknown or inactive gases: {sorted(unknown)}")

    peaks: dict[str, dict] = {}
    corrected: dict[str, float] = {}
    unavailable: list[str] = []
    for gas in concentration_gases:
        if gas not in gas_library:
            continue
        info = gas_library[gas]
        center = float(info["center"])
        half_width = float(info["half_width"])
        if windows and gas in windows:
            try:
                center, half_width = windows[gas]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{gas} window must contain center and half_width") from exc
        center = finite_float(center, f"{gas} window center")
        half_width = positive_float(half_width, f"{gas} half_width")

        indices = _find_peak_indices(x, center, half_width)
        if np.min(x) > center - half_width or np.max(x) < center + half_width:
            unavailable.append(f"{gas}: 光谱未完整覆盖窗口 [{center - half_width}, {center + half_width}]")
        elif len(indices) < 3 or not (np.min(x[indices]) < center < np.max(x[indices])):
            unavailable.append(f"{gas}: 窗口内需要至少 3 个数据点并包含峰中心两侧")
        peak = _peak_info(x, y, indices, strategy)
        coeff = positive_float(
            coefficients.get(gas, info["coefficient"]) if coefficients else info["coefficient"],
            f"{gas} coefficient",
        )
        peaks[gas] = {
            **peak,
            "window_center": float(center),
            "half_width": float(half_width),
            "coefficient": coeff,
            "window_points": len(indices),
        }
        corrected[gas] = peak["intensity"] * coeff

    if unavailable:
        raise ValueError("无法归一化：定量窗口缺失或采样不足；" + "; ".join(unavailable))

    active_gases = tuple(corrected)
    denominator = float(sum(corrected.values()))
    if not np.isfinite(denominator):
        raise ValueError("加权信号超过可计算范围，请检查强度及修正系数")
    if denominator <= 0:
        raise ValueError("定量窗口内没有高于背景的有效峰信号，无法计算加权信号占比")
    fractions = {
        gas: corrected[gas] / denominator
        for gas in active_gases
    }
    percentages = {gas: fractions[gas] * 100.0 for gas in active_gases}
    warnings = [
        "结果为所选气体的加权信号占比；未经标准样品标定与验证，不能直接视为实际组分浓度。"
    ]
    if all(peaks[gas]["coefficient"] == 1.0 for gas in active_gases):
        warnings.append("所有修正系数均为 1，结果未补偿不同气体的响应差异。")
    for gas in active_gases:
        if corrected[gas] == 0:
            warnings.append(f"{gas} 窗口内未得到正峰信号；0% 不等于确认该气体不存在。")

    return {
        "strategy": strategy,
        "result_kind": "weighted_signal_fraction",
        "background_model": "linear_window_endpoints",
        "valid": True,
        "validity": {"complete_windows": True, "positive_total_signal": True, "physical_concentration_validated": False},
        "warnings": warnings,
        "x_unit": spectrum.x_unit,
        "gases": list(active_gases),
        "fractions": fractions,
        "percentages": percentages,
        "peaks": peaks,
        "corrected_intensities": corrected,
        "denominator": denominator,
    }


def calculate_concentration(
    spectrum: Spectrum,
    gas_name: str,
    window: float | None = None,
    reference_gas: str | None = None,
    reference_concentration: float = 78.0,
    strategy: str = "peak_max",
    *,
    library: dict | None = None,
) -> dict:
    """Return focused details for one quantitative gas and all dynamic results.

    ``reference_gas`` and ``reference_concentration`` remain optional only for
    compatibility with older CLI/TUI callers. The normalized model does not
    require a fixed reference gas.
    """
    gas_library = get_gas_library() if library is None else coerce_gas_library(library)
    concentration_gases = tuple(
        key for key, info in gas_library.items()
        if info.get("enabled", True) and info.get("quantitative", False)
    )
    gas_key = find_gas_key(gas_name, gas_library)
    gas_info = gas_library.get(gas_key) if gas_key else None
    if gas_info is None:
        raise ValueError(f"未知气体: {gas_name}. 支持的气体: {list(gas_library.keys())}")
    if gas_key not in concentration_gases:
        raise ValueError(f"气体 {gas_key} 未在气体峰位库中启用定量")

    windows = None
    if window is not None:
        windows = {gas_key: (float(gas_info["center"]), float(window))}

    all_result = calculate_gas_concentrations(spectrum, strategy=strategy, windows=windows, library=gas_library)
    peak = all_result["peaks"].get(gas_key, {"center": 0.0, "area": 0.0, "height": 0.0, "intensity": 0.0})
    concentration = all_result["percentages"].get(gas_key, 0.0)

    result = {
        "gas": gas_info["name"],
        "gas_key": gas_key,
        "x_unit": spectrum.x_unit,
        "result_kind": all_result["result_kind"],
        "valid": all_result["valid"],
        "warnings": all_result["warnings"],
        "concentration": round(concentration, 4),
        "peak_center": peak["center"],
        "peak_area": round(peak["area"], 4),
        "peak_height": round(peak["height"], 4),
        "peak_intensity": round(peak["intensity"], 4),
        "all_concentrations": all_result,
    }
    if reference_gas:
        reference_concentration = positive_float(reference_concentration, "reference_concentration")
        ref_key = find_gas_key(reference_gas, gas_library) or str(reference_gas)
        ref_info = gas_library.get(ref_key, {"name": reference_gas})
        ref_peak = all_result["peaks"].get(
            ref_key, {"center": 0.0, "area": 0.0, "height": 0.0, "intensity": 0.0}
        )
        result.update(
            {
                "reference_gas": ref_info["name"],
                "reference_peak_center": ref_peak["center"],
                "reference_peak_area": round(ref_peak["area"], 4),
                "reference_concentration": reference_concentration,
            }
        )
    return result


def calculate_concentration_from_ratio(
    area_ratio: float,
    gas_name: str,
    reference_gas: str = "N2",
    reference_concentration: float = 78.0,
) -> float:
    """Backward-compatible ratio helper.

    The main application uses the gas library's dynamic normalized model. This
    helper is kept for existing scripts that pass a precomputed ratio.
    """
    library = get_gas_library()
    gas_key = find_gas_key(gas_name, library)
    gas_info = library.get(gas_key) if gas_key else None
    if gas_info is None:
        raise ValueError(f"未知气体: {gas_name}")
    area_ratio = finite_float(area_ratio, "area_ratio")
    reference_concentration = positive_float(reference_concentration, "reference_concentration")
    if area_ratio < 0:
        raise ValueError("area_ratio must be >= 0")
    if find_gas_key(reference_gas, library) is None:
        raise ValueError(f"未知参考气体: {reference_gas}")
    return reference_concentration * area_ratio
