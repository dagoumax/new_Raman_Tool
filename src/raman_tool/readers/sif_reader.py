"""SIF (Andor Solis) 文件读取器.

解析 Andor SIF 二进制格式文件中的光谱数据。
支持单帧标准 SIF (1D/2D float32)，并兼容旧版紧凑 uint16/float32 文件。
标准二维帧沿探测器行取平均，不猜测多帧/多区域文件的选择规则。
标准帧数据为 float32，波长校准来自明确的校准区块 (65540):
    wavelength(nm) = c0 + c1 * pixel + c2 * pixel^2 + c3 * pixel^3
拉曼位移: shift(cm^-1) = (1/laser_nm - 1/wavelength_nm) * 1e7

Unrecognized or incomplete calibration remains in pixel units. Standard block
layout and 1-based polynomial coordinates were checked against the maintainer's
parser: https://github.com/fujiisoup/sif_parser/blob/master/sif_parser/_sif_open.py
and https://github.com/fujiisoup/sif_parser/blob/master/sif_parser/utils.py
"""

from pathlib import Path
import struct
import re
import numpy as np
from raman_tool.safety import check_data_points, check_file_size
from raman_tool.models import Spectrum
from raman_tool.validation import validate_calibration


class SIFError(Exception):
    pass


def _parse_calibration(data: bytes, text: str) -> tuple[list[float] | None, float | None]:
    """Read a complete named calibration block, without default coefficients.

    Other 65540 sections are present in SIF headers, so the following coefficient
    and axis-calibration rows must all have the expected numeric structure.
    """
    for block in re.finditer(
        r"(?m)^65540[^\r\n]*\r?\n([^\r\n]+)\r?\n([^\r\n]+)\r?\n([^\r\n]+)\r?\n([^\r\n]+)",
        text,
    ):
        try:
            coeffs = [float(v) for v in block.group(1).split()]
            other_axes = [[float(v) for v in block.group(i).split()] for i in (2, 3)]
            if len(coeffs) != 4 or any(len(axis) != 4 for axis in other_axes):
                continue
            if not np.all(np.isfinite(coeffs)) or coeffs[0] <= 0:
                continue
            laser_wl = float(block.group(4))
        except ValueError:
            continue
        if np.isfinite(laser_wl) and laser_wl > 0:
            return coeffs, laser_wl
        return coeffs, None
    return None, None


def _read_standard_frame(data: bytes, text: str) -> tuple | None:
    """Read the declared single float32 frame, never the 65538 user-text block."""
    frame = re.search(
        r"(?<!\d)65541[ \t]+([\d \t]+)\r?\n65538[ \t]+([\d \t]+)\r?\n", text
    )
    if frame is None:
        return None
    dimensions = [int(v) for v in frame.group(1).split()]
    roi = [int(v) for v in frame.group(2).split()]
    if len(dimensions) != 8 or len(roi) < 6:
        raise SIFError("Invalid SIF frame dimensions")
    frame_count, subimages, total, frame_length = dimensions[4:]
    if frame_count != 1 or subimages != 1:
        raise SIFError("Multiple SIF frames/subimages require an explicit selection; export one frame first")
    left, top, right, bottom, ybin, xbin = roi[:6]
    if min(left, bottom, xbin, ybin) < 1 or right < left or top < bottom:
        raise SIFError("Invalid SIF detector region")
    raw_width, raw_height = right - left + 1, top - bottom + 1
    if raw_width % xbin or raw_height % ybin:
        raise SIFError("SIF detector region is not divisible by its binning")
    width, height = raw_width // xbin, raw_height // ybin
    if total != frame_length or frame_length != width * height:
        raise SIFError("SIF frame size does not match declared detector dimensions")
    check_data_points(frame_length, "SIF frame values")
    offset = frame.end()
    timestamp = re.match(rb"[ \t]*\d+\r?\n", data[offset:offset + 128])
    if timestamp is None:
        raise SIFError("SIF frame timestamp is missing")
    offset += timestamp.end()
    # The optional flags are ASCII lines, immediately before binary frame data.
    zero_flag = re.match(rb"0\r?\n", data[offset:offset + 4])
    if zero_flag:
        offset += zero_flag.end()
    elif data[offset:offset + 2] == b"1\n":
        offset += 2
        extra_timestamp = re.match(rb"[ \t]*\d+\r?\n", data[offset:offset + 128])
        if extra_timestamp is None:
            raise SIFError("SIF frame timestamp array is incomplete")
        offset += extra_timestamp.end()
    if offset + frame_length * 4 > len(data):
        raise SIFError("SIF float32 frame is truncated")
    image = np.frombuffer(data, dtype="<f4", count=frame_length, offset=offset).reshape(height, width)
    if not np.all(np.isfinite(image)):
        raise SIFError("SIF float32 frame contains non-finite values")
    pixels = left + np.arange(width, dtype=np.float64) * xbin + (xbin - 1) / 2
    return image.mean(axis=0, dtype=np.float64), pixels, {
        "image_shape": (height, width),
        "selected_rows": height,
        "row_mode": "mean",
        "frame_data_offset": offset,
        "frame_data_values": frame_length,
        "detector_xbin": xbin,
        "detector_ybin": ybin,
    }


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


def _compute_wavelength(num_pixels: int, coeffs: list[float], pixels=None) -> np.ndarray:
    if pixels is None:
        pixels = np.arange(1, num_pixels + 1, dtype=np.float64)
    return np.polynomial.polynomial.polyval(pixels, coeffs)


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
    if calibration is not None:
        calibration = validate_calibration(*calibration)
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

    # A real standard SIF frame must be parsed before legacy compact formats.
    # The earlier 65538 block can be binary user text and is not detector data.
    standard = _read_standard_frame(data, text)
    frame_metadata = {}
    pixels = None
    intensity = None
    if standard is not None:
        intensity, pixels, frame_metadata = standard
        num_pixels = len(intensity)
        fmt = "float32_LE_standard_frame"
    else:
        version = re.search(r"(?m)^655(\d{2})[ \t]+0[ \t]+0[ \t]+1", text)
        if version is not None and int(version.group(1)) >= 48:
            raise SIFError("Unsupported SIF frame layout; cannot safely locate detector data")
        try:
            intensity, num_pixels, fmt = _read_intensity_1d(data, text)
        except SIFError:
            pass

    # If uint16 gave too few pixels vs expected, try float32 from XML-bounded area
    if standard is None and intensity is not None and expected_pixels > 100 and num_pixels < expected_pixels * 0.8:
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
            intensity = np.frombuffer(data, dtype="<f4", count=num_floats, offset=data_start).astype(np.float64)
            num_pixels = len(intensity)
            fmt = "float32_LE_2D"
    else:
        # Strategy 1 worked (uint16 path)
        pass

    check_data_points(num_pixels, "SIF pixels")

    # Display uncalibrated data as pixels, with an explicit reason in metadata.
    if pixels is None:
        pixels = np.arange(1, num_pixels + 1, dtype=np.float64)
    raman_shift = pixels - 1
    wavelengths = None
    x_unit = "px"
    calibration_source = "missing_or_invalid"
    calibration_warning = "SIF 缺少有效的波长校准或激光波长；当前使用像素坐标，请先校准。"
    if cal_coeffs is not None and laser_wl is not None:
        candidate_wavelengths = _compute_wavelength(num_pixels, cal_coeffs, pixels)
        differences = np.diff(candidate_wavelengths)
        if (np.all(np.isfinite(candidate_wavelengths)) and np.all(candidate_wavelengths > 0)
                and (len(differences) == 0 or np.all(differences > 0) or np.all(differences < 0))):
            wavelengths = candidate_wavelengths
            raman_shift = _compute_raman_shift(wavelengths, laser_wl)
            x_unit = "cm-1"
            calibration_source = "sif_header"
            calibration_warning = None
    if calibration is not None:
        raman_shift = calibration[0] * (pixels - 1) + calibration[1]
        x_unit = "cm-1"
        calibration_source = "user"
        calibration_warning = None

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
            "x_unit": x_unit,
            "calibration": calibration,
            "calibration_source": calibration_source,
            "calibration_warning": calibration_warning,
            "pixel_centers": pixels - 1,
            "num_pixels": num_pixels,
            "ccd_height": ccd_height,
            "spectrometer": spectrometer,
            "ccd_model": ccd_model,
            "laser_wavelength": laser_wl,
            "calibration_coefficients": cal_coeffs,
            "wavelengths": wavelengths,
            "data_format": fmt,
            **frame_metadata,
        },
    )
