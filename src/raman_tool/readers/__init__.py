"""文件读取器模块.

支持格式: TXT, ASC, SIF, TIF, BMP, JPG
图像格式支持行分组和列合并参数。
"""

from raman_tool.readers.txt_reader import read_txt
from raman_tool.readers.asc_reader import read_asc
from raman_tool.readers.sif_reader import read_sif
from raman_tool.readers.tif_reader import read_tif
from raman_tool.readers.bmp_reader import read_bmp
from raman_tool.readers.jpg_reader import read_jpg
from raman_tool.readers.base import detect_format, read_file, SUPPORTED_FORMATS

__all__ = [
    "read_txt", "read_asc", "read_sif", "read_tif", "read_bmp", "read_jpg",
    "detect_format", "read_file", "SUPPORTED_FORMATS",
]
