"""TXT 文本数据读取器.

支持: 两列 (拉曼位移, 强度) 或单列 (自动用索引作 x 轴)
支持以 "#" 或 ";" 开头的注释行
"""

from pathlib import Path
import numpy as np
from raman_tool.safety import check_data_points, check_text_file_size
from raman_tool.models import Spectrum


def read_txt(
    filepath: str | Path,
    calibration: tuple[float, float] | None = None,
) -> Spectrum:
    """读取 TXT 格式光谱数据.

    Args:
        filepath: 文件路径
        calibration: 线性校准 (a, b) 使 raman = a * pixel + b

    Returns:
        Spectrum 对象
    """
    filepath = Path(filepath)
    check_text_file_size(filepath)
    data = np.loadtxt(filepath, comments=("#", ";"))

    if data.ndim == 0 or len(data) == 0:
        raise ValueError(f"TXT 文件无有效数据: {filepath}")

    is_index_axis = False  # x 轴是否为像素索引 (需校准)

    if data.ndim == 1:
        intensity = data
        raman_shift = np.arange(len(intensity), dtype=np.float64)
        is_index_axis = True
    elif data.ndim == 2:
        if data.shape[1] >= 2:
            raman_shift = data[:, 0]
            intensity = data[:, 1]
            # 两列及以上: 第 1 列已是拉曼位移, 不校准
        else:
            intensity = data[:, 0]
            raman_shift = np.arange(len(intensity), dtype=np.float64)
            is_index_axis = True
    else:
        raise ValueError(f"TXT 数据格式不支持 (ndim={data.ndim}): {filepath}")

    check_data_points(len(intensity), "TXT data")

    if calibration and is_index_axis:
        raman_shift = calibration[0] * raman_shift + calibration[1]

    return Spectrum(
        raman_shift=raman_shift,
        intensity=intensity,
        filename=filepath.name,
        metadata={
            "filepath": str(filepath.absolute()),
            "calibration": calibration,
        },
    )
