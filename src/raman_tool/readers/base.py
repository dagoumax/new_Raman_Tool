"""文件读取器基类与格式检测."""

from pathlib import Path
from raman_tool.models import Spectrum

SUPPORTED_FORMATS = {
    ".txt": "TXT 文本数据",
    ".asc": "ASC 文本数据",
    ".sif": "SIF (Andor Solis)",
    ".tif": "TIFF 图像",
    ".tiff": "TIFF 图像",
    ".bmp": "BMP 图像",
    ".jpg": "JPG 图像",
    ".jpeg": "JPG 图像",
}


def detect_format(filepath: str | Path) -> str:
    suffix = Path(filepath).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"不支持的格式: {suffix}。支持的格式: {list(SUPPORTED_FORMATS.keys())}"
        )
    return suffix


def read_file(
    filepath: str | Path,
    row_groups: str | None = None,
    col_merge: int = 1,
    calibration: tuple[float, float] | None = None,
    row_mode: str = "mean",
) -> Spectrum:
    """根据扩展名自动选择合适的读取器读取文件.

    Args:
        filepath: 文件路径
        row_groups: TIF/BMP 行分组规格 (如 "1-40, 91-130")
        col_merge: TIF/BMP 列合并因子 (默认 1=不变)
        calibration: 线性校准 (a, b) 使 raman = a * pixel + b

    Returns:
        Spectrum 对象
    """
    from raman_tool.readers.txt_reader import read_txt
    from raman_tool.readers.asc_reader import read_asc
    from raman_tool.readers.sif_reader import read_sif
    from raman_tool.readers.tif_reader import read_tif
    from raman_tool.readers.bmp_reader import read_bmp
    from raman_tool.readers.jpg_reader import read_jpg

    suffix = detect_format(filepath)

    if suffix in (".tif", ".tiff"):
        return read_tif(filepath, row_groups=row_groups, col_merge=col_merge, calibration=calibration, row_mode=row_mode)
    if suffix == ".bmp":
        return read_bmp(filepath, row_groups=row_groups, col_merge=col_merge, calibration=calibration, row_mode=row_mode)
    if suffix in (".jpg", ".jpeg"):
        return read_jpg(filepath, row_groups=row_groups, col_merge=col_merge, calibration=calibration, row_mode=row_mode)

    readers = {
        ".txt": read_txt,
        ".asc": read_asc,
        ".sif": read_sif,
    }
    return readers[suffix](filepath, calibration=calibration)
