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
