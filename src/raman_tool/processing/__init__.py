"""数据处理模块.

包含基线校正、信噪比计算、气体浓度计算等功能。
"""

from raman_tool.processing.baseline import (
    arPLS,
    poly_baseline,
    subtract_baseline,
)
from raman_tool.processing.snr import calculate_snr, find_peak, find_peaks_auto
from raman_tool.processing.concentration import (
    calculate_gas_concentrations,
    calculate_concentration,
    calculate_concentration_from_ratio,
)

__all__ = [
    "arPLS",
    "poly_baseline",
    "subtract_baseline",
    "calculate_snr",
    "find_peak",
    "find_peaks_auto",
    "calculate_gas_concentrations",
    "calculate_concentration",
    "calculate_concentration_from_ratio",
]
