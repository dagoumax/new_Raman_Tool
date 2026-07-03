"""信噪比 (SNR) 计算与寻峰."""

import numpy as np
from scipy.signal import find_peaks
from raman_tool.models import Spectrum

# 气体参考峰 (cm⁻¹)
GAS_REF_PEAKS = {
    2331.0: "N2",
    1555.0: "O2",
    1388.0: "CO2",
    3657.0: "H2O",
    2917.0: "CH4",
    4155.0: "H2",
    2143.0: "CO",
    1151.0: "SO2",
    1876.0: "NO",
    3334.0: "NH3",
    2954.0: "C2H6",
    1285.0: "CO2",
}


def _match_gas(position: float, tolerance: float) -> str | None:
    """将实测峰位匹配到最近的气体参考峰."""
    best_gas = None
    best_dist = float("inf")
    for ref_pos, gas_name in GAS_REF_PEAKS.items():
        dist = abs(position - ref_pos)
        if dist < tolerance and dist < best_dist:
            best_dist = dist
            best_gas = gas_name
    return best_gas


def calculate_snr(
    spectrum: Spectrum,
    peak_start: float | None = None,
    peak_end: float | None = None,
    noise_start: float | None = None,
    noise_end: float | None = None,
) -> dict:
    """计算光谱的信噪比 (SNR).

    信噪比 = 峰值信号强度 / 噪声 RMS

    Args:
        spectrum: 输入光谱
        peak_start: 信号区域起始拉曼位移 (cm⁻¹)，None 时自动选择最高峰
        peak_end: 信号区域终止拉曼位移 (cm⁻¹)
        noise_start: 噪声区域起始拉曼位移 (cm⁻¹)，None 时自动选择平坦区域
        noise_end: 噪声区域终止拉曼位移 (cm⁻¹)

    Returns:
        dict: {"snr": 信噪比值, "signal": 信号强度, "noise_rms": 噪声RMS,
               "peak_center": 峰值位置, "peak_area": 峰面积}
    """
    x = spectrum.raman_shift
    y = spectrum.intensity

    if peak_start is not None and peak_end is not None:
        peak_mask = (x >= peak_start) & (x <= peak_end)
        peak_y = y[peak_mask]
        signal = np.max(peak_y) - np.min(peak_y)
        peak_center = x[peak_mask][np.argmax(peak_y)]
        peak_area = np.trapezoid(peak_y - np.min(peak_y), x[peak_mask])
    else:
        signal = np.max(y) - np.min(y)
        peak_center = x[np.argmax(y)]
        peak_area = 0.0

    if noise_start is not None and noise_end is not None:
        noise_mask = (x >= noise_start) & (x <= noise_end)
        noise_y = y[noise_mask]
        noise_rms = np.std(noise_y)
    else:
        # 自动选择噪声区域：光谱最后 10% 的区域
        n = len(x)
        noise_start_idx = int(n * 0.9)
        noise_y = y[noise_start_idx:]
        noise_rms = np.std(noise_y)

    if noise_rms == 0:
        snr = float("inf")
    else:
        snr = signal / noise_rms

    return {
        "snr": float(snr),
        "signal": float(signal),
        "noise_rms": float(noise_rms),
        "peak_center": float(peak_center),
        "peak_area": float(peak_area),
    }


def find_peak(
    spectrum: Spectrum,
    start: float,
    end: float,
) -> dict:
    """在指定区域内寻找峰值.

    Args:
        spectrum: 输入光谱
        start: 起始拉曼位移
        end: 终止拉曼位移

    Returns:
        dict: {"center": 峰中心位置, "height": 峰高, "area": 峰面积,
               "fwhm": 半峰宽}
    """
    x = spectrum.raman_shift
    y = spectrum.intensity

    mask = (x >= start) & (x <= end)
    x_roi = x[mask]
    y_roi = y[mask]

    if len(y_roi) == 0:
        raise ValueError(f"在范围 [{start}, {end}] 内无数据点")

    idx_max = np.argmax(y_roi)
    center = x_roi[idx_max]
    height = y_roi[idx_max]

    baseline = np.min(y_roi)
    y_baselined = y_roi - baseline

    area = float(np.trapezoid(y_baselined, x_roi))

    # 计算半峰宽 (Full Width at Half Maximum)
    half_max = height / 2
    above_half = y_roi >= half_max
    indices = np.where(above_half)[0]
    if len(indices) >= 2:
        fwhm = x_roi[indices[-1]] - x_roi[indices[0]]
    else:
        fwhm = 0.0

    return {
        "center": float(center),
        "height": float(height - baseline),
        "area": area,
        "fwhm": float(fwhm),
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
    """自动检测光谱中的所有峰，并尝试匹配已知气体。

    Args:
        spectrum: 输入光谱
        height: 峰高阈值，None 时按最大强度的 5% 自动设置
        distance: 峰之间的最小间距 (像素)
        prominence: 峰的突出度阈值，None 时自动设置
        rel_height: 相对高度阈值 (相对于最大强度)
        rel_prominence: 相对突出度阈值 (相对于最大强度)
        match_tolerance: 气体匹配容差 (x 轴单位)，None 时取 x 范围的 5%

    Returns:
        [{center, height, area, fwhm, prominence, matched_gas}, ...] 按位置排序
    """
    x = spectrum.raman_shift
    y = spectrum.intensity

    y_max = np.max(y)
    y_std = np.std(y)
    x_range = np.max(x) - np.min(x)

    if height is None:
        height = max(y_std * 2, y_max * rel_height)
    if prominence is None:
        prominence = max(y_std, y_max * rel_prominence)
    if match_tolerance is None:
        match_tolerance = max(50, x_range * 0.05)

    peaks_idx, props = find_peaks(
        y, height=height, distance=distance, prominence=prominence
    )

    results = []
    for i, idx in enumerate(peaks_idx):
        center = float(x[idx])
        peak_height = float(y[idx])

        half_max = peak_height / 2
        left = idx
        while left > 0 and y[left] > half_max:
            left -= 1
        right = idx
        while right < len(y) - 1 and y[right] > half_max:
            right += 1
        fwhm = float(x[min(right, len(x) - 1)] - x[max(left, 0)])

        area_start = max(0, idx - int(fwhm / (x[1] - x[0]) if len(x) > 1 and x[1] != x[0] else 5))
        area_end = min(len(y), idx + int(fwhm / (x[1] - x[0]) if len(x) > 1 and x[1] != x[0] else 5))
        if area_end > area_start:
            area = float(np.trapezoid(y[area_start:area_end], x[area_start:area_end]))
        else:
            area = 0.0

        prom = float(props["prominences"][i]) if "prominences" in props else 0.0

        matched = _match_gas(center, match_tolerance)

        results.append({
            "center": center,
            "height": peak_height,
            "area": area,
            "fwhm": fwhm,
            "prominence": prom,
            "matched_gas": matched,
        })

    return results
