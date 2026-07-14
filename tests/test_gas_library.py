import numpy as np
import pytest

from raman_tool.gas_library import load_gas_library, save_gas_library
from raman_tool.models import Spectrum
from raman_tool.processing import calculate_gas_concentrations, find_peaks_auto


def test_gas_library_defaults_include_core_gases(tmp_path):
    library = load_gas_library(tmp_path / "missing.json")

    assert library["N2"]["center"] == pytest.approx(2330.0)
    assert library["O2"]["quantitative"] is True
    assert library["CO2_V2"]["label"] == "CO2(v2)"


def test_save_and_load_gas_library_roundtrip(tmp_path):
    path = tmp_path / "gas_library.json"
    save_gas_library(
        {
            "AR": {
                "name": "argon",
                "center": 1000,
                "half_width": 11,
                "coefficient": 0.8,
                "color": "#abcdef",
                "enabled": True,
                "quantitative": False,
            }
        },
        path,
    )

    loaded = load_gas_library(path)

    assert loaded["AR"]["center"] == pytest.approx(1000.0)
    assert loaded["AR"]["half_width"] == pytest.approx(11.0)


def test_find_peaks_auto_uses_configured_library(monkeypatch, tmp_path):
    path = tmp_path / "gas_library.json"
    monkeypatch.setenv("RAMAN_TOOL_GAS_LIBRARY", str(path))
    save_gas_library(
        {
            "AR": {
                "name": "argon",
                "center": 1000,
                "half_width": 10,
                "coefficient": 1,
                "color": "#abcdef",
                "enabled": True,
                "quantitative": False,
            }
        },
        path,
    )
    from raman_tool.gas_library import reload_gas_library

    reload_gas_library()
    x = np.linspace(900, 1100, 500)
    y = 100 * np.exp(-0.5 * ((x - 1000) / 3) ** 2)
    peaks = find_peaks_auto(Spectrum(x, y), match_tolerance=5)

    assert peaks
    assert peaks[0]["matched_gas"] == "AR"


def test_configured_quantitative_centers_affect_concentration(monkeypatch, tmp_path):
    path = tmp_path / "gas_library.json"
    monkeypatch.setenv("RAMAN_TOOL_GAS_LIBRARY", str(path))
    save_gas_library(
        {
            "O2": {"name": "O2", "center": 1000, "half_width": 10, "coefficient": 1, "enabled": True, "quantitative": True},
            "N2": {"name": "N2", "center": 1200, "half_width": 10, "coefficient": 1, "enabled": True, "quantitative": True},
            "CO2": {"name": "CO2", "center": 1400, "half_width": 10, "coefficient": 1, "enabled": True, "quantitative": True},
        },
        path,
    )
    from raman_tool.gas_library import reload_gas_library

    reload_gas_library()
    x = np.linspace(800, 1600, 2000)
    y = (
        100 * np.exp(-0.5 * ((x - 1000) / 3) ** 2)
        + 300 * np.exp(-0.5 * ((x - 1200) / 3) ** 2)
        + 100 * np.exp(-0.5 * ((x - 1400) / 3) ** 2)
    )

    result = calculate_gas_concentrations(Spectrum(x, y))

    assert result["percentages"]["N2"] > result["percentages"]["O2"]
    assert result["peaks"]["N2"]["window_center"] == pytest.approx(1200.0)



def test_gas_library_preserves_mixed_case_and_subscript_keys(tmp_path):
    path = tmp_path / "gas_library.json"
    save_gas_library(
        {
            "CBrF₃": {
                "name": "哈龙1301",
                "center": 760,
                "half_width": 12,
                "coefficient": 1,
                "enabled": True,
                "quantitative": True,
            }
        },
        path,
    )

    loaded = load_gas_library(path)

    assert "CBrF₃" in loaded
    assert loaded["CBrF₃"]["name"] == "哈龙1301"
