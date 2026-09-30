"""测试 Spectrum 数据模型."""

import pytest
import numpy as np
from raman_tool.models import Spectrum

# 测试数据
TEST_X = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
TEST_Y = np.array([10.0, 50.0, 30.0, 20.0, 5.0])


class TestSpectrum:
    def test_create_spectrum(self):
        spec = Spectrum(TEST_X, TEST_Y)
        assert spec.size == 5
        assert spec.filename == ""

    def test_create_with_filename(self):
        spec = Spectrum(TEST_X, TEST_Y, filename="test.txt")
        assert spec.filename == "test.txt"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            Spectrum(np.array([1, 2, 3]), np.array([1, 2]))

    def test_list_input_converted(self):
        spec = Spectrum([100, 200, 300], [10, 20, 30])
        assert isinstance(spec.raman_shift, np.ndarray)
        assert isinstance(spec.intensity, np.ndarray)

    def test_crop(self):
        spec = Spectrum(TEST_X, TEST_Y)
        cropped = spec.crop(100, 300)
        assert cropped.size == 3
        assert cropped.raman_shift[0] == 100.0
        assert cropped.raman_shift[-1] == 300.0

    def test_normalize(self):
        spec = Spectrum(TEST_X, TEST_Y)
        norm = spec.normalize()
        assert np.isclose(np.max(norm.intensity), 1.0)

    def test_normalize_zero(self):
        spec = Spectrum(TEST_X, np.zeros(5))
        norm = spec.normalize()
        assert np.all(norm.intensity == 0)

    def test_metadata(self):
        spec = Spectrum(TEST_X, TEST_Y, metadata={"key": "value"})
        assert spec.metadata["key"] == "value"

    def test_axis_unit_defaults_to_raman_shift_for_legacy_spectra(self):
        spec = Spectrum(TEST_X, TEST_Y)

        assert spec.x_unit == "cm-1"
        assert spec.x_unit_label == "cm⁻¹"
        assert spec.is_raman_shift

    def test_pixel_axis_has_distinct_labels(self):
        spec = Spectrum(TEST_X, TEST_Y, metadata={"x_unit": "pixels"})

        assert spec.x_unit == "px"
        assert spec.x_label == "像素位置 (px)"
        assert spec.x_plot_label == "像素位置 (px)"
        assert not spec.is_raman_shift


@pytest.mark.parametrize("x,y", [
    ([], []),
    ([[1, 2]], [[3, 4]]),
    (1, 2),
    ([1, 2], [3, np.nan]),
    ([1, np.inf], [3, 4]),
    ([1, 2], [3 + 1j, 4]),
    ([True, False], [1, 2]),
    ([1, 1, 2], [1, 2, 3]),
    ([1, 3, 2], [1, 2, 3]),
])
def test_invalid_spectrum_data_rejected(x, y):
    with pytest.raises(ValueError):
        Spectrum(x, y)


def test_descending_spectrum_preserves_order():
    spectrum = Spectrum((3, 2, 1), (10, 20, 30))
    np.testing.assert_array_equal(spectrum.raman_shift, [3, 2, 1])
    assert spectrum.intensity.dtype == np.float64


@pytest.mark.parametrize("unit", ["nm", "unknown", 0, False])
def test_unknown_axis_unit_is_not_assumed_calibrated(unit):
    with pytest.raises(ValueError, match="unit"):
        Spectrum([1, 2], [3, 4], metadata={"x_unit": unit})


@pytest.mark.parametrize("start,end", [(500, 100), (100, 100), (100, np.inf), (600, 700)])
def test_crop_rejects_invalid_or_empty_region(start, end):
    with pytest.raises(ValueError):
        Spectrum(TEST_X, TEST_Y).crop(start, end)
