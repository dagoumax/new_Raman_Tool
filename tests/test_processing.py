"""测试数据处理模块."""

import pytest
import numpy as np
from raman_tool.models import Spectrum
from raman_tool.gas_library import reload_gas_library
from raman_tool.processing import (
    calculate_snr,
    find_peak,
    calculate_gas_concentrations,
    calculate_concentration,
    subtract_baseline,
    poly_baseline,
)


@pytest.fixture(autouse=True)
def isolate_gas_library(monkeypatch, tmp_path):
    monkeypatch.setenv("RAMAN_TOOL_GAS_LIBRARY", str(tmp_path / "missing-gas-library.json"))
    reload_gas_library()
    yield
    reload_gas_library()


def create_gaussian_peak(center: float, height: float, sigma: float, noise: float = 0):
    """创建模拟拉曼峰."""
    x = np.linspace(0, 4000, 2000)
    y = height * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    if noise > 0:
        rng = np.random.default_rng(42)
        y += rng.normal(0, noise, len(x))
    return x, y


class TestSNR:
    def test_basic_snr(self):
        x, y = create_gaussian_peak(1500, 100, 10)
        spec = Spectrum(x, y)
        result = calculate_snr(spec)
        assert result["snr"] > 0
        assert "peak_center" in result
        assert "noise_rms" in result

    def test_snr_with_regions(self):
        x, y = create_gaussian_peak(1500, 100, 10, noise=1)
        spec = Spectrum(x, y)
        result = calculate_snr(
            spec, peak_start=1490, peak_end=1510, noise_start=3500, noise_end=4000
        )
        assert result["snr"] > 0

    def test_snr_zero_noise(self):
        x, y = create_gaussian_peak(1500, 100, 10, noise=0)
        spec = Spectrum(x, y)
        result = calculate_snr(spec)
        assert result["noise_rms"] == 0


class TestFindPeak:
    def test_find_peak_basic(self):
        x, y = create_gaussian_peak(1500, 100, 10)
        spec = Spectrum(x, y)
        result = find_peak(spec, 1490, 1510)
        assert abs(result["center"] - 1500) < 2
        assert result["height"] > 0
        assert result["area"] > 0

    def test_find_peak_empty_range(self):
        x = np.array([100, 200, 300], dtype=np.float64)
        y = np.array([1, 2, 3], dtype=np.float64)
        spec = Spectrum(x, y)
        with pytest.raises(ValueError):
            find_peak(spec, 500, 600)


class TestBaseline:
    def test_poly_baseline(self):
        x = np.linspace(100, 3000, 1000)
        y = 10 * np.sin(x / 500) + 5 * x + 100
        spec = Spectrum(x, y)
        baseline = poly_baseline(spec, degree=1)
        assert baseline.shape == y.shape
        assert abs(baseline[0] - y[0]) < abs(y[0])

    def test_subtract_baseline_poly(self):
        x = np.linspace(100, 3000, 1000)
        baseline = 0.1 * x + 200
        peak = 50 * np.exp(-0.5 * ((x - 1500) / 20) ** 2)
        y = baseline + peak
        spec = Spectrum(x, y)
        corrected = subtract_baseline(spec, method="poly", degree=1)
        assert corrected.intensity.max() < y.max()

    def test_subtract_baseline_invalid_method(self):
        spec = Spectrum(np.array([1, 2, 3], dtype=np.float64), np.array([4, 5, 6], dtype=np.float64))
        with pytest.raises(ValueError):
            subtract_baseline(spec, method="invalid")

    def test_subtract_baseline_arpls(self):
        x = np.linspace(100, 3000, 500)
        y = 2 * x + 100 + 20 * np.sin(x / 100)
        spec = Spectrum(x, y)
        corrected = subtract_baseline(spec, method="arPLS")
        assert corrected.intensity.shape == y.shape


class TestConcentration:
    def test_calculate_gas_concentrations_normalized(self):
        x = np.linspace(100, 4000, 4000)
        o2_peak = 200 * np.exp(-0.5 * ((x - 1555) / 5) ** 2)
        n2_peak = 600 * np.exp(-0.5 * ((x - 2330) / 5) ** 2)
        co2_peak = 200 * np.exp(-0.5 * ((x - 1388) / 5) ** 2)
        spec = Spectrum(x, o2_peak + n2_peak + co2_peak)

        result = calculate_gas_concentrations(spec, strategy="peak_max")
        percentages = result["percentages"]
        assert percentages["N2"] > percentages["O2"]
        assert percentages["N2"] > percentages["CO2"]
        assert sum(percentages.values()) == pytest.approx(100.0)

    def test_calculate_gas_concentrations_peak_area(self):
        x = np.linspace(100, 4000, 4000)
        y = (
            200 * np.exp(-0.5 * ((x - 1555) / 8) ** 2)
            + 600 * np.exp(-0.5 * ((x - 2330) / 8) ** 2)
            + 200 * np.exp(-0.5 * ((x - 1388) / 8) ** 2)
        )
        spec = Spectrum(x, y)

        result = calculate_gas_concentrations(spec, strategy="peak_area")
        assert result["peaks"]["O2"]["area"] > 0
        assert result["peaks"]["N2"]["intensity"] > result["peaks"]["O2"]["intensity"]
        assert sum(result["percentages"].values()) == pytest.approx(100.0)

    def test_calculate_concentration_n2(self):
        x, y = create_gaussian_peak(2331, 1000, 5)
        spec = Spectrum(x, y)
        result = calculate_concentration(spec, "N2", window=20)
        assert "concentration" in result
        assert result["gas"] == "氮气"

    def test_calculate_concentration_unknown_gas(self):
        spec = Spectrum(np.array([100, 200], dtype=np.float64), np.array([10, 20], dtype=np.float64))
        with pytest.raises(ValueError):
            calculate_concentration(spec, "UNKNOWN")

    def test_multi_gas_spectrum(self):
        """测试多气体光谱."""
        x = np.linspace(100, 4000, 4000)
        n2_peak = 1000 * np.exp(-0.5 * ((x - 2331) / 5) ** 2)
        o2_peak = 800 * np.exp(-0.5 * ((x - 1555) / 5) ** 2)
        y = n2_peak + o2_peak

        spec = Spectrum(x, y)
        result_o2 = calculate_concentration(spec, "O2", window=20)
        assert result_o2["concentration"] > 0
        assert result_o2["gas"] == "氧气"
