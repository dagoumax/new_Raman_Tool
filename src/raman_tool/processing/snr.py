"""信噪比、峰面积与扣除背景后的半峰宽计算。"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths

from raman_tool.models import Spectrum
from raman_tool.gas_library import get_reference_peaks
from raman_tool.validation import finite_float, integer_parameter, positive_float, region_mask, validate_spectrum_arrays


def _ascending(spectrum: Spectrum) -> tuple[np.ndarray, np.ndarray]:
    x, y = validate_spectrum_arrays(spectrum.raman_shift, spectrum.intensity)
    return (x[::-1], y[::-1]) if x[0] > x[-1] else (x, y)


def _match_gas(position: float, tolerance: float) -> str | None:
    best_gas = None
    best_dist = float("inf")
    for ref_pos, gas_name in get_reference_peaks().items():
        dist = abs(position - ref_pos)
        if dist < tolerance and dist < best_dist:
            best_dist, best_gas = dist, gas_name
    return best_gas


def _optional_region(x, start, end, name, minimum):
    if (start is None) != (end is None):
        raise ValueError(f"{name} requires both start and end")
    return None if start is None else region_mask(x, start, end, name=name, minimum=minimum)


def calculate_snr(
    spectrum: Spectrum,
    peak_start: float | None = None,
    peak_end: float | None = None,
    noise_start: float | None = None,
    noise_end: float | None = None,
) -> dict:
    """峰高（区域最大值减最小值）除以噪声区域的标准差。

    默认噪声选区为横坐标最高的 10% 数据点（至少 2 点），与文件顺序无关。
    平坦信号不具备可用 SNR；有信号且噪声为零时返回无穷大。
    """
    x, y = _ascending(spectrum)
    if len(x) < 2:
        raise ValueError("SNR requires at least 2 data points")
    peak_mask = _optional_region(x, peak_start, peak_end, "peak region", 2)
    noise_mask = _optional_region(x, noise_start, noise_end, "noise region", 2)
    peak_x, peak_y = (x, y) if peak_mask is None else (x[peak_mask], y[peak_mask])
    signal = float(np.max(peak_y) - np.min(peak_y))
    if signal <= 0:
        raise ValueError("peak region has no positive signal above background; SNR is undefined")
    peak_center = peak_x[np.argmax(peak_y)]
    peak_area = 0.0 if peak_mask is None else trapezoid(peak_y - np.min(peak_y), peak_x)
    noise_y = y[-max(2, int(np.ceil(len(x) * 0.1))):] if noise_mask is None else y[noise_mask]
    noise_rms = float(np.std(noise_y))
    snr = float("inf") if noise_rms == 0 else signal / noise_rms
    return {
        "snr": float(snr), "signal": signal, "noise_rms": noise_rms,
        "peak_center": float(peak_center), "peak_area": float(peak_area),
        "x_unit": spectrum.x_unit,
        "noise_points": len(noise_y),
        "noise_region_source": "automatic_high_x_tail" if noise_mask is None else "user",
    }


def _local_fwhm(x, y, idx, baseline):
    height = float(y[idx] - baseline)
    if height <= 0:
        return float("nan"), False
    half = baseline + height / 2.0
    left, right = int(idx), int(idx)
    while left > 0 and y[left] > half:
        left -= 1
    while right < len(y) - 1 and y[right] > half:
        right += 1
    if y[left] > half or y[right] > half or left == right:
        return float("nan"), False
    left_x = float(np.interp(half, y[left:left + 2], x[left:left + 2]))
    right_x = float(np.interp(half, y[right - 1:right + 1][::-1], x[right - 1:right + 1][::-1]))
    return right_x - left_x, True


def find_peak(spectrum: Spectrum, start: float, end: float) -> dict:
    """在指定区域寻峰；面积扣除区域最小值，半峰宽按同一背景计算。"""
    x, y = _ascending(spectrum)
    mask = region_mask(x, start, end, name="peak region", minimum=3)
    x_roi, y_roi = x[mask], y[mask]
    idx_max = int(np.argmax(y_roi))
    baseline = float(np.min(y_roi))
    height = float(y_roi[idx_max] - baseline)
    if height <= 0:
        raise ValueError("peak region has no positive signal above background")
    fwhm, fwhm_valid = _local_fwhm(x_roi, y_roi, idx_max, baseline)
    return {
        "center": float(x_roi[idx_max]), "height": height,
        "area": float(trapezoid(y_roi - baseline, x_roi)),
        "fwhm": float(fwhm), "fwhm_valid": fwhm_valid,
        "warnings": [] if fwhm_valid else ["选区未包含峰两侧的半高交点，无法计算半峰宽"],
        "x_unit": spectrum.x_unit,
    }


def find_peaks_auto(
    spectrum: Spectrum,
    height: float | None = None,
    distance: int = 10,
    prominence: float | None = None,
    rel_height: float = 0.05,
    rel_prominence: float = 0.05,
    match_tolerance: float | None = None,
) -> list[dict]:
    """自动寻峰，使用峰突出度定义半峰宽，并按横坐标升序返回。

    height 保持为原始峰顶强度；面积扣除峰基部端点之间的线性背景。
    """
    x, y = _ascending(spectrum)
    distance = integer_parameter(distance, "distance", minimum=1)
    rel_height = finite_float(rel_height, "rel_height")
    rel_prominence = finite_float(rel_prominence, "rel_prominence")
    if not (0 <= rel_height <= 1 and 0 <= rel_prominence <= 1):
        raise ValueError("relative peak thresholds must be between 0 and 1")
    if height is not None:
        height = finite_float(height, "height")
    if prominence is not None:
        prominence = finite_float(prominence, "prominence")
        if prominence < 0:
            raise ValueError("prominence must be >= 0")
    if match_tolerance is not None:
        match_tolerance = positive_float(match_tolerance, "match_tolerance")
    if len(x) < 3:
        return []
    y_span = float(np.max(y) - np.min(y))
    y_std = float(np.std(y))
    if height is None:
        height = float(np.min(y)) + max(y_std * 2, y_span * rel_height)
    if prominence is None:
        prominence = max(y_std, y_span * rel_prominence)
    if match_tolerance is None:
        match_tolerance = max(50, (x[-1] - x[0]) * 0.05)
    peaks_idx, props = find_peaks(y, height=height, distance=distance, prominence=prominence)
    if not len(peaks_idx):
        return []
    _, _, left_ips, right_ips = peak_widths(
        y, peaks_idx, rel_height=0.5,
        prominence_data=(props["prominences"], props["left_bases"], props["right_bases"]),
    )
    point_indices = np.arange(len(x), dtype=float)
    results = []
    for i, idx in enumerate(peaks_idx):
        center = float(x[idx])
        left, right = int(props["left_bases"][i]), int(props["right_bases"][i])
        area_x, area_y = x[left:right + 1], y[left:right + 1]
        background = np.interp(area_x, [x[left], x[right]], [y[left], y[right]])
        area = float(trapezoid(np.maximum(area_y - background, 0.0), area_x))
        fwhm = float(np.interp(right_ips[i], point_indices, x) - np.interp(left_ips[i], point_indices, x))
        results.append({
            "center": center, "height": float(y[idx]), "area": area,
            "fwhm": fwhm, "prominence": float(props["prominences"][i]),
            "matched_gas": _match_gas(center, match_tolerance) if spectrum.is_raman_shift else None,
            "x_unit": spectrum.x_unit,
        })
    return results
