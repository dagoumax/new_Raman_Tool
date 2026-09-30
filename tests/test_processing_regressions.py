"""Numerical regressions and analytic reference cases; not instrument certification."""

import numpy as np
import pytest

from raman_tool.models import Spectrum
from raman_tool.processing import (
    arPLS, calculate_gas_concentrations, calculate_concentration,
    calculate_snr, find_peak, find_peaks_auto, poly_baseline, subtract_baseline,
)
from raman_tool.validation import validate_calibration


def gas_entry(center, half_width=10, coefficient=1):
    return {"center": center, "half_width": half_width, "coefficient": coefficient,
            "enabled": True, "quantitative": True}


@pytest.fixture
def gaussian():
    x = np.linspace(-10, 10, 2001)
    y = 12 * np.exp(-0.5 * x ** 2)
    return x, y


@pytest.mark.parametrize("background", [0, 100, -100])
@pytest.mark.parametrize("reverse", [False, True])
def test_peak_matches_analytic_gaussian_width_and_area(gaussian, background, reverse):
    x, y = gaussian
    x, y = (x[::-1], y[::-1]) if reverse else (x, y)
    result = find_peak(Spectrum(x, y + background), -8, 8)
    assert result["center"] == pytest.approx(0)
    assert result["height"] == pytest.approx(12)
    assert result["area"] == pytest.approx(12 * np.sqrt(2 * np.pi), rel=1e-6)
    assert result["fwhm"] == pytest.approx(2 * np.sqrt(2 * np.log(2)), rel=2e-5)
    assert result["fwhm_valid"]


def test_clipped_peak_reports_unavailable_width(gaussian):
    x, y = gaussian
    result = find_peak(Spectrum(x, y), 0, 4)
    assert not result["fwhm_valid"]
    assert np.isnan(result["fwhm"])
    assert result["warnings"]


def test_auto_peak_background_and_descending_invariant(gaussian):
    x, y = gaussian
    ordinary = find_peaks_auto(Spectrum(x, y))[0]
    shifted_reversed = find_peaks_auto(Spectrum(x[::-1], y[::-1] + 100))[0]
    assert shifted_reversed["fwhm"] == pytest.approx(ordinary["fwhm"])
    assert shifted_reversed["area"] == pytest.approx(ordinary["area"])
    assert shifted_reversed["area"] == pytest.approx(12 * np.sqrt(2 * np.pi), rel=1e-6)


def test_nonuniform_coordinate_peak_width():
    x = np.linspace(-2.1, 2.1, 2001) ** 3
    y = np.exp(-0.5 * x ** 2)
    result = find_peaks_auto(Spectrum(x, y))[0]
    assert result["fwhm"] == pytest.approx(2 * np.sqrt(2 * np.log(2)), rel=1e-4)
    assert result["area"] == pytest.approx(np.sqrt(2 * np.pi), rel=1e-4)


@pytest.mark.parametrize("kwargs", [
    {"noise_start": 100, "noise_end": 200},
    {"noise_start": 10, "noise_end": 11},
    {"noise_start": -1},
    {"peak_end": 1},
    {"peak_start": 1, "peak_end": -1},
    {"noise_start": np.nan, "noise_end": 1},
])
def test_snr_invalid_regions_are_explicit_errors(gaussian, kwargs):
    x, y = gaussian
    with pytest.raises(ValueError):
        calculate_snr(Spectrum(x, y), **kwargs)


def test_snr_selected_area_and_automatic_noise_are_direction_invariant():
    x = np.linspace(-10, 10, 2001)
    y = 12 * np.exp(-0.5 * x ** 2) + 0.1 * np.sin(x * 30)
    forward = calculate_snr(Spectrum(x, y), -4, 4)
    backward = calculate_snr(Spectrum(x[::-1], y[::-1]), -4, 4)
    assert backward == forward
    assert forward["peak_area"] > 0


def test_zero_signal_is_not_infinite_snr():
    with pytest.raises(ValueError, match="no positive signal"):
        calculate_snr(Spectrum([1, 2, 3], [5, 5, 5]))


@pytest.mark.parametrize("kwargs", [
    {"lam": 0}, {"lam": -1}, {"lam": np.nan},
    {"max_iter": 0}, {"max_iter": 1.5}, {"max_iter": True},
    {"tol": 0}, {"tol": np.inf}, {"tol": 1},
])
def test_arpls_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        arPLS(np.array([1., 2., 3., 4.]), **kwargs)


@pytest.mark.parametrize("values", [[], [1, 2], [[1, 2, 3]], [1, 2, np.nan]])
def test_arpls_rejects_invalid_signals(values):
    with pytest.raises(ValueError):
        arPLS(values)


@pytest.mark.parametrize("degree", [-1, 1.5, 3, np.inf, True])
def test_polynomial_rejects_invalid_degree(degree):
    with pytest.raises(ValueError):
        poly_baseline(Spectrum([1, 2, 3], [2, 4, 6]), degree)


def test_polynomial_reference_line_at_large_x_offset():
    x = 1e8 + np.linspace(0, 1, 100)
    expected = 2 * (x - 1e8) + 5
    baseline = poly_baseline(Spectrum(x, expected), degree=1)
    np.testing.assert_allclose(baseline, expected, atol=1e-7)


def test_arpls_reference_constant_and_linear_signal():
    x = np.arange(101, dtype=float)
    for y in (np.full_like(x, 7), 3 + x * 0.05):
        corrected = subtract_baseline(Spectrum(x, y), lam=1e4)
        np.testing.assert_allclose(corrected.intensity, 0, atol=1e-6)


@pytest.mark.parametrize("slope,intercept", [(0, 1), (np.inf, 1), (1, np.nan)])
def test_calibration_rejects_nonfinite_or_zero_slope(slope, intercept):
    with pytest.raises(ValueError):
        validate_calibration(slope, intercept)


def test_negative_calibration_slope_is_valid():
    assert validate_calibration(-2, 100) == (-2.0, 100.0)


@pytest.mark.parametrize("strategy", ["peak_max", "peak_area"])
@pytest.mark.parametrize("reverse", [False, True])
def test_reference_mixture_weighted_signal_percentages(strategy, reverse):
    # Equal-width peaks with heights 3:1 and coefficients 1:2 imply 60%:40%.
    x = np.linspace(80, 220, 2801)
    y = 30 + 3 * np.exp(-0.5 * ((x - 100) / 1.2) ** 2) + np.exp(-0.5 * ((x - 200) / 1.2) ** 2)
    if reverse:
        x, y = x[::-1], y[::-1]
    library = {"A": gas_entry(100), "B": gas_entry(200, coefficient=2)}
    result = calculate_gas_concentrations(Spectrum(x, y), strategy=strategy, library=library)
    assert result["percentages"] == pytest.approx({"A": 60, "B": 40}, rel=1e-8)
    assert result["result_kind"] == "weighted_signal_fraction"
    assert result["valid"]
    assert not result["validity"]["physical_concentration_validated"]
    assert result["warnings"]
    focused = calculate_concentration(Spectrum(x, y), "A", strategy=strategy, library=library)
    assert focused["concentration"] == pytest.approx(60)


@pytest.mark.parametrize("x", [
    np.linspace(90, 110, 201),  # Missing B's window, formerly yielded A=100%.
    np.linspace(95, 220, 2501),  # A window truncated despite a visible peak.
    np.array([80, 100, 120, 180, 200, 220]),  # Only one point per window.
])
def test_missing_or_insufficient_windows_cannot_be_normalized(x):
    y = 3 * np.exp(-0.5 * ((x - 100) / 1.2) ** 2) + np.exp(-0.5 * ((x - 200) / 1.2) ** 2)
    with pytest.raises(ValueError, match="窗口"):
        calculate_gas_concentrations(Spectrum(x, y), library={"A": gas_entry(100), "B": gas_entry(200)})


def test_flat_spectrum_cannot_yield_zero_percentages():
    x = np.linspace(80, 120, 401)
    with pytest.raises(ValueError, match="有效峰信号"):
        calculate_gas_concentrations(Spectrum(x, np.ones_like(x)), library={"A": gas_entry(100)})


@pytest.mark.parametrize("strategy", ["peak_max", "peak_area"])
@pytest.mark.parametrize("background", [2.34, 1e8])
def test_sloping_background_is_not_quantitative_signal(strategy, background):
    x = np.linspace(80, 120, 401)
    y = background + x * 0.37
    with pytest.raises(ValueError, match="有效峰信号"):
        calculate_gas_concentrations(Spectrum(x, y), strategy=strategy, library={"A": gas_entry(100)})


@pytest.mark.parametrize("strategy", ["peak_max", "peak_area"])
def test_reference_mixture_is_invariant_to_linear_background(strategy):
    x = np.linspace(80, 220, 2801)
    y = 3 * np.exp(-0.5 * ((x - 100) / 1.2) ** 2) + np.exp(-0.5 * ((x - 200) / 1.2) ** 2)
    library = {"A": gas_entry(100), "B": gas_entry(200, coefficient=2)}
    result = calculate_gas_concentrations(Spectrum(x, y + 80 + x * 0.37), strategy=strategy, library=library)
    assert result["percentages"] == pytest.approx({"A": 60, "B": 40}, rel=1e-8)
    assert result["peaks"]["A"]["center"] == pytest.approx(100)


@pytest.mark.parametrize("kwargs", [
    {"windows": {"A": (100, 0)}}, {"windows": {"A": (np.nan, 10)}},
    {"coefficients": {"A": -1}}, {"coefficients": {"A": np.inf}},
    {"coefficients": {"typo": 1}}, {"windows": {"A": (100,)}},
])
def test_invalid_quantitative_parameters_are_rejected(kwargs):
    x = np.linspace(80, 120, 401)
    y = np.exp(-0.5 * ((x - 100) / 1.2) ** 2)
    with pytest.raises(ValueError):
        calculate_gas_concentrations(Spectrum(x, y), library={"A": gas_entry(100)}, **kwargs)


def test_library_snapshot_does_not_depend_on_global_configuration(monkeypatch):
    monkeypatch.setenv("RAMAN_TOOL_GAS_LIBRARY", "does-not-exist.json")
    x = np.linspace(80, 120, 401)
    y = np.exp(-0.5 * ((x - 100) / 1.2) ** 2)
    result = calculate_gas_concentrations(Spectrum(x, y), library={"A": gas_entry(100)})
    assert result["gases"] == ["A"]
