"""End-to-end reader regressions for coordinates, metadata and detector data."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from raman_tool.exporters import export_spectrum
from raman_tool.models import Spectrum
from raman_tool.readers import read_file
from raman_tool.readers.sif_reader import SIFError, read_sif


@pytest.mark.parametrize("suffix", [".txt", ".asc"])
@pytest.mark.parametrize("unit", ["px", "cm-1"])
def test_export_reimport_preserves_units_calibration_history_and_float_values(tmp_path, suffix, unit):
    original = Spectrum(
        [1.0, 1.0 + 1e-12, 2.5],
        [0.000000000123456789, -3.456789123456789, 10.0],
        metadata={
            "x_unit": unit,
            "calibration": (2.345678912345678, -100.3456789123456),
            "processing_history": [{"sequence": 1, "action": "基线校正", "parameters": {"degree": 2}}],
        },
    )
    loaded = read_file(export_spectrum(original, tmp_path / f"spectrum{suffix}", include_history=True))
    assert loaded.x_unit == unit
    np.testing.assert_array_equal(loaded.raman_shift, original.raman_shift)
    np.testing.assert_array_equal(loaded.intensity, original.intensity)
    assert loaded.metadata["calibration"] == original.metadata["calibration"]
    assert loaded.metadata["processing_history"] == original.metadata["processing_history"]


@pytest.mark.parametrize("suffix", [".txt", ".asc"])
def test_explicit_calibration_uses_exported_pixel_positions(tmp_path, suffix):
    spectrum = Spectrum([0.5, 2.5, 4.5], [10, 20, 30], metadata={"x_unit": "px"})
    path = export_spectrum(spectrum, tmp_path / f"binned{suffix}")
    loaded = read_file(path, calibration=(2, 100))
    np.testing.assert_array_equal(loaded.raman_shift, [101, 105, 109])
    assert loaded.x_unit == "cm-1"


@pytest.mark.parametrize("suffix", [".txt", ".asc"])
def test_already_calibrated_coordinates_are_not_calibrated_twice(tmp_path, suffix):
    path = export_spectrum(Spectrum([100, 102, 104], [10, 20, 30]), tmp_path / f"calibrated{suffix}")
    np.testing.assert_array_equal(read_file(path, calibration=(2, 100)).raman_shift, [100, 102, 104])


def test_asc_does_not_extract_numbers_from_comments_or_headers(tmp_path):
    path = tmp_path / "comments.asc"
    path.write_text(
        "Experiment 2026 exposure 10\n# Calibration\traman_shift = 2 * pixel + 100\n"
        "; 7 88 999\n100,10\n102,20\n104,30\n", encoding="utf-8"
    )
    loaded = read_file(path)
    np.testing.assert_array_equal(loaded.raman_shift, [100, 102, 104])
    np.testing.assert_array_equal(loaded.intensity, [10, 20, 30])
    assert loaded.metadata["calibration"] == (2, 100)


@pytest.mark.parametrize("suffix", [".txt", ".asc"])
def test_single_two_column_row_is_one_spectral_point(tmp_path, suffix):
    path = tmp_path / f"one{suffix}"
    path.write_text("100 45\n", encoding="utf-8")
    loaded = read_file(path)
    assert loaded.size == 1
    assert loaded.raman_shift[0] == 100
    assert loaded.intensity[0] == 45


@pytest.mark.parametrize("content", ["100 10\n102\n", "100 10\n102,,30\n", "100 10\n102 broken 30\n"])
def test_asc_rejects_partial_numeric_rows(tmp_path, content):
    path = tmp_path / "bad.asc"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_file(path)


@pytest.mark.parametrize("suffix", [".tif", ".bmp", ".jpg"])
@pytest.mark.parametrize("shape", [(2, 8), (1, 8), (8, 1)])
def test_image_binning_uses_original_pixel_centers(tmp_path, suffix, shape):
    path = tmp_path / f"pixels{suffix}"
    Image.fromarray(np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)).save(path)
    raw = read_file(path, col_merge=2)
    calibrated = read_file(path, col_merge=2, calibration=(2, 100))
    np.testing.assert_array_equal(raw.raman_shift, [0.5, 2.5, 4.5, 6.5])
    np.testing.assert_array_equal(calibrated.raman_shift, [101, 105, 109, 113])
    assert raw.x_unit == "px"
    assert calibrated.x_unit == "cm-1"
    assert all(row.shape == calibrated.intensity.shape for row in calibrated.metadata["individual_rows"])


def test_image_partial_final_bin_and_row_sum_preserve_all_selected_pixels(tmp_path):
    path = tmp_path / "odd.tif"
    arr = np.array([[0, 2, 4, 6, 8], [10, 12, 14, 16, 18], [20, 22, 24, 26, 28]], dtype=np.uint8)
    Image.fromarray(arr).save(path)
    spectrum = read_file(path, row_groups="1, 3", col_merge=2, row_mode="sum")
    np.testing.assert_array_equal(spectrum.raman_shift, [0.5, 2.5, 4])
    np.testing.assert_array_equal(spectrum.intensity, [22, 30, 36])
    assert spectrum.metadata["selected_rows"] == 2


@pytest.mark.parametrize("kwargs", [
    {"col_merge": 0}, {"col_merge": 1.5}, {"col_merge": 9},
    {"row_groups": "0-2"}, {"row_groups": "1-2junk"}, {"row_groups": "1-5"},
    {"row_groups": "1-2,2"}, {"row_mode": "median"}, {"calibration": (0, 100)},
])
def test_image_invalid_parameters_fail_explicitly(tmp_path, kwargs):
    path = tmp_path / "image.tif"
    Image.fromarray(np.ones((2, 8), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError):
        read_file(path, **kwargs)


def _write_standard_sif(path, *, calibration_block=True, laser="532", coeffs="543 0.5 0 0", truncated=False):
    """Known frame bytes, unrelated user-text bytes, and explicit standard layout."""
    header = b"Andor Technology Multi-Channel File\n65538 1\n65567 0 0 1\n"
    # 65538 length marks user text in real SIF; these must never become intensity.
    header += b"65538 16\n" + np.full(8, 9999, dtype="<u2").tobytes() + b"\n"
    if calibration_block:
        header += f"65540 \x03 \x00 \x01 \x00 \x01 \x00\n{coeffs}\n0 1 0 0\n0 1 0 0\n{laser}\n".encode()
    header += b"Pixel number65541 1 2 4 1 1 1 8 8\n65538 1 2 4 1 1 1 0\n         0\n0\n"
    pixels = np.array([[1, 3, 5, 7], [9, 11, 13, 15]], dtype="<f4")
    payload = pixels.tobytes()
    path.write_bytes(header + (payload[:8] if truncated else payload))
    return len(header)


def test_standard_sif_reads_declared_float_frame_and_header_calibration(tmp_path):
    path = tmp_path / "standard.sif"
    offset = _write_standard_sif(path)
    spectrum = read_sif(path)
    np.testing.assert_array_equal(spectrum.intensity, [5, 7, 9, 11])
    assert spectrum.metadata["frame_data_offset"] == offset
    assert spectrum.metadata["frame_data_values"] == 8
    assert spectrum.metadata["image_shape"] == (2, 4)
    wavelengths = np.array([543.5, 544, 544.5, 545])
    np.testing.assert_allclose(spectrum.raman_shift, (1 / 532 - 1 / wavelengths) * 1e7)
    assert spectrum.x_unit == "cm-1"
    assert spectrum.metadata["calibration_source"] == "sif_header"


@pytest.mark.parametrize("options", [
    {"calibration_block": False}, {"laser": "0"}, {"laser": "nan"},
    {"coeffs": "543 0 0 0"}, {"coeffs": "543 nan 0 0"},
])
def test_sif_incomplete_or_invalid_calibration_stays_pixels(tmp_path, options):
    path = tmp_path / "uncalibrated.sif"
    _write_standard_sif(path, **options)
    spectrum = read_sif(path)
    np.testing.assert_array_equal(spectrum.raman_shift, [0, 1, 2, 3])
    assert spectrum.x_unit == "px"
    assert spectrum.metadata["calibration_warning"]
    calibrated = read_sif(path, calibration=(-2, 100))
    np.testing.assert_array_equal(calibrated.raman_shift, [100, 98, 96, 94])
    assert calibrated.x_unit == "cm-1"
    assert calibrated.metadata["calibration_source"] == "user"


def test_sif_rejects_truncated_standard_frame_instead_of_using_user_text(tmp_path):
    path = tmp_path / "truncated.sif"
    _write_standard_sif(path, truncated=True)
    with pytest.raises(SIFError, match="truncated"):
        read_sif(path)


def test_recorded_sif_frame_has_known_detector_values():
    path = Path(__file__).parent.parent / "test_data" / "10_1.sif"
    if not path.exists():
        pytest.skip("Recorded detector sample is not packaged")
    spectrum = read_sif(path)
    assert spectrum.size == 1024
    np.testing.assert_array_equal(spectrum.intensity[:10], [165, 166, 164, 163, 163, 164, 162, 164, 165, 162])
    assert spectrum.metadata["image_shape"] == (1, 1024)
    assert spectrum.metadata["frame_data_offset"] == 2857
    assert spectrum.metadata["data_format"] == "float32_LE_standard_frame"
    assert spectrum.metadata["calibration_coefficients"] == [544.155205887116, 0.158690231175482, -1.85266976214663e-05, 0]


def test_sif_detector_roi_and_xbin_preserve_original_pixel_centers(tmp_path):
    path = tmp_path / "binned.sif"
    _write_standard_sif(path)
    path.write_bytes(path.read_bytes().replace(
        b"65538 1 2 4 1 1 1 0\n", b"65538 11 2 18 1 1 2 0\n"
    ))
    spectrum = read_sif(path)
    np.testing.assert_array_equal(spectrum.metadata["pixel_centers"], [10.5, 12.5, 14.5, 16.5])
    wavelengths = 543 + 0.5 * np.array([11.5, 13.5, 15.5, 17.5])
    np.testing.assert_allclose(spectrum.raman_shift, (1 / 532 - 1 / wavelengths) * 1e7)
    overridden = read_sif(path, calibration=(2, 100))
    np.testing.assert_array_equal(overridden.raman_shift, [121, 125, 129, 133])


def test_sif_multiple_frames_fail_with_actionable_selection_message(tmp_path):
    path = tmp_path / "multiple.sif"
    _write_standard_sif(path)
    path.write_bytes(path.read_bytes().replace(
        b"65541 1 2 4 1 1 1 8 8", b"65541 1 2 4 1 2 1 16 8"
    ))
    with pytest.raises(SIFError, match="explicit selection"):
        read_sif(path)


def test_sif_flag_one_timestamp_block_precedes_float_frame(tmp_path):
    path = tmp_path / "flag_one.sif"
    offset = _write_standard_sif(path)
    replacement = b"         0\n1\n             6941752\n"
    original = b"         0\n0\n"
    path.write_bytes(path.read_bytes().replace(original, replacement))
    spectrum = read_sif(path)
    np.testing.assert_array_equal(spectrum.intensity, [5, 7, 9, 11])
    assert spectrum.metadata["frame_data_offset"] == offset + len(replacement) - len(original)
