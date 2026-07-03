"""TIFF 图像读取器.

将 TIFF 光谱图像转换为数组，支持行分组、列合并和拉曼位移校准。
"""

import re
from pathlib import Path
import numpy as np
from raman_tool.models import Spectrum


def _parse_row_groups(spec: str) -> list[tuple[int, int]]:
    groups = []
    for part in re.split(r"[,;\s\n]+", spec.strip()):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"(\d+)\s*[-~]\s*(\d+)", part)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            groups.append((start, end) if start <= end else (end, start))
        else:
            try:
                n = int(part)
                groups.append((n, n))
            except ValueError:
                pass
    return groups


def _merge_columns(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    cols = arr.shape[1]
    new_cols = cols // factor
    if new_cols == 0:
        return arr
    trimmed = arr[:, : new_cols * factor]
    reshaped = trimmed.reshape(arr.shape[0], new_cols, factor)
    return reshaped.mean(axis=2)


def read_tif(
    filepath: str | Path,
    row_groups: str | None = None,
    col_merge: int = 1,
    calibration: tuple[float, float] | None = None,
    row_mode: str = "mean",
) -> Spectrum:
    """读取 TIFF 格式光谱图像数据.

    Args:
        filepath: 文件路径
        row_groups: 行分组规格 (1-based), 如 "1-40, 91-130"
        col_merge: 列合并因子 (默认 1=不变)
        calibration: 线性校准参数 (a, b) 使得 raman_shift = a * pixel + b

    Returns:
        Spectrum 对象。metadata 中可能包含 "rows" (各行光谱列表) 供多行叠加显示。
    """
    from PIL import Image

    filepath = Path(filepath)
    img = Image.open(filepath)
    arr = np.array(img, dtype=np.float64)
    original_shape = arr.shape

    if arr.ndim == 1:
        raman = np.arange(len(arr), dtype=np.float64)
        if calibration:
            raman = calibration[0] * raman + calibration[1]
        return Spectrum(
            raman_shift=raman,
            intensity=arr,
            filename=filepath.name,
            metadata={"filepath": str(filepath.absolute()), "format": "TIFF", "image_shape": original_shape},
        )

    if arr.ndim == 3:
        arr = np.mean(arr[:, :, :3], axis=2)

    rows, cols = arr.shape

    # 单行/单列 → 直接作为 1D 光谱返回
    if rows == 1:
        raman = np.arange(cols, dtype=np.float64)
        if calibration:
            raman = calibration[0] * raman + calibration[1]
        return Spectrum(
            raman_shift=raman,
            intensity=arr[0, :],
            filename=filepath.name,
            metadata={"filepath": str(filepath.absolute()), "format": "TIFF", "image_shape": original_shape},
        )
    if cols == 1:
        raman = np.arange(rows, dtype=np.float64)
        if calibration:
            raman = calibration[0] * raman + calibration[1]
        return Spectrum(
            raman_shift=raman,
            intensity=arr[:, 0],
            filename=filepath.name,
            metadata={"filepath": str(filepath.absolute()), "format": "TIFF", "image_shape": original_shape},
        )

    # 2D 图像: 行分组 + 列合并
    row_spectra = []
    MAX_ROW_SPECTRA = 500  # 限制存储行数防止内存溢出

    if row_groups:
        groups = _parse_row_groups(row_groups)
        selected_blocks = []
        for start, end in groups:
            s = max(0, start - 1)
            e = min(rows, end)
            if s < e:
                selected_blocks.append(arr[s:e, :])
                # 每组内各行: 限制总数
                for r in range(s, e):
                    if len(row_spectra) < MAX_ROW_SPECTRA:
                        row_spectra.append(arr[r, :].copy())

        if not selected_blocks:
            raise ValueError(f"行分组 '{row_groups}' 未匹配到有效范围 (总行数: {rows})")
        arr = np.concatenate(selected_blocks, axis=0)
    else:
        # 没有行分组: 全部行, 限制单行存储数量
        if rows <= MAX_ROW_SPECTRA:
            row_spectra = [arr[r, :].copy() for r in range(rows)]

    # 列合并
    if col_merge > 1:
        arr = _merge_columns(arr, col_merge)
        if row_spectra:
            row_spectra = [_merge_columns(r.reshape(1, -1), col_merge).flatten() for r in row_spectra]

    final_rows, final_cols = arr.shape
    intensity = np.sum(arr, axis=0) if row_mode == "sum" else np.mean(arr, axis=0)

    # 拉曼位移校准
    raman_shift = np.arange(final_cols, dtype=np.float64)
    if calibration:
        raman_shift = calibration[0] * raman_shift + calibration[1]

    return Spectrum(
        raman_shift=raman_shift,
        intensity=intensity,
        filename=filepath.name,
        metadata={
            "filepath": str(filepath.absolute()),
            "format": "TIFF",
            "image_shape": original_shape,
            "row_groups": row_groups,
            "col_merge": col_merge,
            "selected_rows": final_rows,
            "output_cols": final_cols,
            "individual_rows": row_spectra if row_spectra else None,
            "total_rows": rows,
            "calibration": calibration,
        },
    )
