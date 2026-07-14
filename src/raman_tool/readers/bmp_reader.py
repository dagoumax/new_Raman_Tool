"""BMP 图像读取器.

将 BMP 光谱图像转换为数组，支持行分组和列合并。
"""

import re
from pathlib import Path
import numpy as np
from raman_tool.safety import check_image_array_values, check_image_pixels, configure_pillow_limits
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


def read_bmp(
    filepath: str | Path,
    row_groups: str | None = None,
    col_merge: int = 1,
    calibration: tuple[float, float] | None = None,
    row_mode: str = "mean",
) -> Spectrum:
    Image = configure_pillow_limits()

    filepath = Path(filepath)
    with Image.open(filepath) as img:
        check_image_pixels(*img.size)
        arr = np.array(img, dtype=np.float64)
    original_shape = arr.shape

    check_image_array_values(arr.size)

    if arr.ndim == 1:
        return Spectrum(
            raman_shift=np.arange(len(arr), dtype=np.float64),
            intensity=arr,
            filename=filepath.name,
            metadata={"filepath": str(filepath.absolute()), "format": "BMP", "image_shape": original_shape},
        )

    if arr.ndim == 3:
        arr = np.mean(arr[:, :, :3], axis=2)

    rows, cols = arr.shape

    if rows == 1:
        raman = np.arange(cols, dtype=np.float64)
        if calibration:
            raman = calibration[0] * raman + calibration[1]
        return Spectrum(
            raman_shift=raman,
            intensity=arr[0, :],
            filename=filepath.name,
            metadata={"filepath": str(filepath.absolute()), "format": "BMP", "image_shape": original_shape},
        )
    if cols == 1:
        raman = np.arange(rows, dtype=np.float64)
        if calibration:
            raman = calibration[0] * raman + calibration[1]
        return Spectrum(
            raman_shift=raman,
            intensity=arr[:, 0],
            filename=filepath.name,
            metadata={"filepath": str(filepath.absolute()), "format": "BMP", "image_shape": original_shape},
        )

    row_spectra = []
    MAX_ROW_SPECTRA = 500

    if row_groups:
        groups = _parse_row_groups(row_groups)
        selected_blocks = []
        for start, end in groups:
            s = max(0, start - 1)
            e = min(rows, end)
            if s < e:
                selected_blocks.append(arr[s:e, :])
                for r in range(s, e):
                    if len(row_spectra) < MAX_ROW_SPECTRA:
                        row_spectra.append(arr[r, :].copy())
        if selected_blocks:
            arr = np.concatenate(selected_blocks, axis=0)
        else:
            raise ValueError(f"行分组 '{row_groups}' 未匹配到有效范围 (总行数: {rows})")
    else:
        if rows <= MAX_ROW_SPECTRA:
            row_spectra = [arr[r, :].copy() for r in range(rows)]

    if col_merge > 1:
        arr = _merge_columns(arr, col_merge)
        if row_spectra:
            row_spectra = [_merge_columns(r.reshape(1, -1), col_merge).flatten() for r in row_spectra]

    final_rows, final_cols = arr.shape
    intensity = np.sum(arr, axis=0) if row_mode == "sum" else np.mean(arr, axis=0)

    raman_shift = np.arange(final_cols, dtype=np.float64)
    if calibration:
        raman_shift = calibration[0] * raman_shift + calibration[1]

    return Spectrum(
        raman_shift=raman_shift,
        intensity=intensity,
        filename=filepath.name,
        metadata={
            "filepath": str(filepath.absolute()),
            "format": "BMP",
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
