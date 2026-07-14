"""SIF (Andor Solis) 文件读取器.

解析 Andor SIF 二进制格式文件中的光谱数据。
支持两种格式:
  1. 标准 1D 光谱 ("65538 N" 标签后接 N 字节 uint16 LE 数据)
  2. 2D 图像 ("0\n0\n" 标记后接 float32 LE 数据)
波长校准来自 65567 标签中的多项式系数:
    wavelength(nm) = c0 + c1 * pixel + c2 * pixel^2
拉曼位移: shift(cm^-1) = (1/laser_nm - 1/wavelength_nm) * 1e7
"""

from pathlib import Path
import struct
import re
import numpy as np
from raman_tool.safety import check_data_points, check_file_size
from raman_tool.models import Spectrum


class SIFError(Exception):
    pass


def _parse_calibration(data: bytes, text: str) -> tuple[list[float], float]:
    coeffs = [543.5, 0.08, 0.0]
    laser_wl = 532.0
    cal_match = re.search(r"(\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+e[+-]\d+)", text[:600])
    if cal_match:
        coeffs = [float(cal_match.group(i)) for i in range(1, 4)]
    wl_match = re.search(r"(\d{3})\s*\n\s*\d+\s*\n\s*\d+\s*\n\s*\d+\s*\n\s*\d+\s*\n\s*Raman shift", text)
    if wl_match:
        laser_wl = float(wl_match.group(1))
    return coeffs, laser_wl


def _is_marker(value):
    """Check if a uint16 value is a marker/metadata (not spectral data)."""
    if value in [0, 259, 65530, 65535]:
        return True
    low = value & 0xFF
    high = (value >> 8) & 0xFF
    if 32 <= low <= 126 and 32 <= high <= 126:
        return True
    if (low == 0 and 32 <= high <= 126) or (high == 0 and 32 <= low <= 126):
        return True
    return False


def _filter_marked_data(vals):
    """Filter out marker values from uint16 data that has interleaved markers."""
    filtered = []
    for v in vals:
        if not _is_marker(int(v)):
            filtered.append(float(v))
    return np.array(filtered, dtype=np.float64), len(filtered)


def _scan_float32_data(data, xml_start, min_vals=100):
    """Scan file for float32 blocks that look like spectral data.
    Fallback for SIF files where uint16 parsing fails."""
    import struct
    best = None
    scan_end = min(xml_start, len(data))
    for start in range(400, scan_end - min_vals * 4, 8):
        n = (scan_end - start) // 4
        if n < min_vals:
            break
        try:
            vals = struct.unpack_from('<' + str(n) + 'f', data, start)
            valid = [v for v in vals if 10 < v < 100000]
            if len(valid) >= min_vals:
                avg = sum(valid) / len(valid)
                if best is None or len(valid) > best[0]:
                    best = (len(valid), start, np.array(valid, dtype=np.float64))
        except:
            continue
    if best:
        return best[2], len(best[2])
    raise SIFError("无法找到float32光谱数据")


def _compute_wavelength(num_pixels: int, coeffs: list[float]) -> np.ndarray:
    pixels = np.arange(num_pixels, dtype=np.float64)
    wl = np.full(num_pixels, coeffs[0], dtype=np.float64)
    if len(coeffs) >= 2:
        wl += coeffs[1] * pixels
    if len(coeffs) >= 3:
        wl += coeffs[2] * pixels ** 2
    return wl


def _compute_raman_shift(wavelengths: np.ndarray, laser_wl: float) -> np.ndarray:
    valid = wavelengths > 0
    shifts = np.zeros_like(wavelengths)
    if laser_wl > 0:
        shifts[valid] = (1.0 / laser_wl - 1.0 / wavelengths[valid]) * 1e7
    return shifts


def _read_intensity_1d(data: bytes, text: str) -> tuple:
    for m in re.finditer(r"65538\s+(\d+)\n", text):
        nbytes = int(m.group(1))
        if nbytes < 8:
            continue
        data_offset = m.end()
        num_pixels = nbytes // 2
        if data_offset + nbytes > len(data):
            continue
        vals = struct.unpack_from("<{}H".format(num_pixels), data, data_offset)
        intensity = np.array(vals, dtype=np.float64)
        # Check for marker-based format (e.g. DR316B_LD,DD)
        if np.sum(intensity == 259) > 10:
            filtered, n = _filter_marked_data(intensity)
            if n > 50:
                return filtered, n, "uint16_filtered_{}px".format(n)
        # Standard format (e.g. DU420_BVF)
        valid_ratio = np.sum((intensity > 0) & (intensity < 65535)) / num_pixels
        if valid_ratio > 0.5:
            return intensity, num_pixels, "uint16_LE_{}px".format(num_pixels)
    raise SIFError("无法定位光谱数据区域")


def _trim_ascii_zero_trailer(data: bytes, end: int) -> int:
    """Trim Andor text trailer lines such as ``0\n0\n`` before the XML block."""
    while end >= 2 and data[end - 2:end] == b"0\n":
        end -= 2
    return end


def read_sif(filepath: str | Path, calibration: tuple | None = None) -> Spectrum:
    filepath = Path(filepath)
    check_file_size(filepath)
    data = filepath.read_bytes()
    try:
        text = data.decode("ascii", errors="replace")
    except Exception:
        text = data[:2000].decode("latin-1", errors="replace")
    cal_coeffs, laser_wl = _parse_calibration(data, text)

    # Parse expected pixel count from header (e.g. "2000 256 53" -> 2000)
    expected_pixels = 0
    dim_match = re.search(r"\b(\d{3,4})\s+(\d{2,4})\s+(\d+)", text[:600])
    if dim_match:
        expected_pixels = int(dim_match.group(1))
        check_data_points(expected_pixels, "SIF expected pixels")
    xml_start = data.find(b"<?xml")
    if xml_start == -1:
        xml_start = len(data)

    # Strategy 1: Try uint16-based reading
    intensity = None
    try:
        intensity, num_pixels, fmt = _read_intensity_1d(data, text)
    except SIFError:
        pass

    # If uint16 gave too few pixels vs expected, try float32 from XML-bounded area
    if intensity is not None and expected_pixels > 100 and num_pixels < expected_pixels * 0.8:
        intensity = None  # Switch to float32 fallback

    # Strategy 2: float32 from XML-bounded area (reference tool approach)
    if intensity is None:
        found_float32 = False
        data_end = _trim_ascii_zero_trailer(data, xml_start)
        if expected_pixels > 100 and xml_start > expected_pixels * 4 + 300:
            data_start = xml_start - expected_pixels * 4
            if data_end == xml_start and data_start >= 0:
                vals = struct.unpack_from("<" + str(expected_pixels) + "f", data, data_start)
                valid_ratio = sum(10 < v < 100000 for v in vals) / expected_pixels
                if valid_ratio > 0.3:
                    intensity = np.array(vals, dtype=np.float64)
                    num_pixels = expected_pixels
                    fmt = "float32_LE_xml_" + str(expected_pixels) + "px"
                    found_float32 = True
            else:
                num_pixels = (data_end - data_start) // 4
                data_start = data_end - num_pixels * 4
                if num_pixels > 100 and data_start >= 0:
                    vals = struct.unpack_from("<" + str(num_pixels) + "f", data, data_start)
                    valid_ratio = sum(10 < v < 100000 for v in vals) / num_pixels
                    if valid_ratio > 0.3:
                        intensity = np.array(vals, dtype=np.float64)
                        fmt = "float32_LE_xml_trimmed_" + str(num_pixels) + "px"
                        found_float32 = True

        # Strategy 3: Original "0\n0\n" marker path
        if not found_float32:
            marker_pos = data.find(b"0\n0\n", 0, xml_start)
            if marker_pos == -1:
                marker_pos = data.rfind(b"\n0\n", 0, xml_start)
            if marker_pos == -1:
                raise SIFError("无法定位数据起始位置: {}".format(filepath))
            data_start = marker_pos + 4
            data_end = _trim_ascii_zero_trailer(data, xml_start)
            remaining = data_end - data_start
            num_floats = remaining // 4
            if num_floats < 10:
                raise SIFError("数据区域太小: {}".format(filepath))
            intensity = np.frombuffer(data, dtype=np.float32, count=num_floats, offset=data_start).astype(np.float64)
            num_pixels = len(intensity)
            fmt = "float32_LE_2D"
    else:
        # Strategy 1 worked (uint16 path)
        pass

    check_data_points(num_pixels, "SIF pixels")

    # Compute wavelengths and Raman shift
    wavelengths = _compute_wavelength(num_pixels, cal_coeffs)
    raman_shift = _compute_raman_shift(wavelengths, laser_wl)
    if calibration is not None:
        raman_shift = calibration[0] * np.arange(num_pixels, dtype=np.float64) + calibration[1]

    # Extract metadata from text header
    spectrometer = ""
    spec_match = re.search(r"SR\d+[A-Za-z]*\d*", text)
    if spec_match:
        spectrometer = spec_match.group()
    ccd_model = ""
    ccd_match = re.search(r"DR\d+[A-Za-z]*[-\w]*", text)
    if ccd_match:
        ccd_model = ccd_match.group()
    ccd_height = 1
    if dim_match:
        ccd_height = int(dim_match.group(2))
    else:
        dim_match2 = re.search(r"\n\s{2,}(\d{3,5})\s+(\d+)\s+\d+", text[:600])
        if dim_match2:
            ccd_height = int(dim_match2.group(2))
    return Spectrum(
        raman_shift=raman_shift,
        intensity=intensity,
        filename=filepath.name,
        metadata={
            "filepath": str(filepath.absolute()),
            "format": "SIF",
            "num_pixels": num_pixels,
            "ccd_height": ccd_height,
            "spectrometer": spectrometer,
            "ccd_model": ccd_model,
            "laser_wavelength": laser_wl,
            "calibration_coefficients": cal_coeffs,
            "wavelengths": wavelengths,
            "data_format": fmt,
        },
    )
