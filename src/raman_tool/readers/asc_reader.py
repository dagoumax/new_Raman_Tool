"""ASC 文本数据读取器.

支持: 两列 (拉曼位移, 强度) 或单列 (自动用索引作 x 轴)
自动跳过非数字行。
"""

from pathlib import Path
import re
import numpy as np
from raman_tool.safety import check_data_points, check_text_file_size
from raman_tool.models import Spectrum


def read_asc(
    filepath: str | Path,
    calibration: tuple[float, float] | None = None,
) -> Spectrum:
    """读取 ASC 格式光谱数据.

    Args:
        filepath: 文件路径
        calibration: 线性校准 (a, b) 使 raman = a * pixel + b

    Returns:
        Spectrum 对象
    """
    filepath = Path(filepath)
    check_text_file_size(filepath)
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    lines = content.strip().splitlines()
    data_rows = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[,\s]+", line)
        numeric_parts = []
        for p in parts:
            try:
                numeric_parts.append(float(p))
            except ValueError:
                pass
        if len(numeric_parts) >= 1:
            data_rows.append(numeric_parts)

    if not data_rows:
        raise ValueError(f"ASC 文件中未找到有效数据: {filepath}")

    data = np.array(data_rows, dtype=object)
    # 按最长行填充 NaN 再转为 float
    max_cols = max(len(r) for r in data_rows)
    padded = np.full((len(data_rows), max_cols), np.nan)
    for i, row in enumerate(data_rows):
        padded[i, :len(row)] = row

    is_index_axis = False

    if padded.shape[1] >= 2:
        raman_shift = padded[:, 0]
        intensity = padded[:, 1]
    else:
        intensity = padded[:, 0]
        raman_shift = np.arange(len(intensity), dtype=np.float64)
        is_index_axis = True

    # 去除 NaN
    mask = ~np.isnan(intensity)
    raman_shift = raman_shift[mask]
    intensity = intensity[mask]

    check_data_points(len(intensity), "ASC data")

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
