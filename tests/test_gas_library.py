import numpy as np
import pytest

from raman_tool.gas_library import (
    coerce_gas_library,
    get_quantitative_gas_choices,
    load_gas_library,
    reload_gas_library,
    save_gas_library,
)
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


def test_find_peaks_auto_does_not_match_cm1_library_on_pixel_axis(monkeypatch, tmp_path):
    path = tmp_path / "gas_library.json"
    monkeypatch.setenv("RAMAN_TOOL_GAS_LIBRARY", str(path))
    save_gas_library(
        {
            "AR": {
                "name": "argon",
                "center": 1000,
                "half_width": 10,
                "coefficient": 1,
                "enabled": True,
                "quantitative": False,
            }
        },
        path,
    )
    reload_gas_library()
    x = np.linspace(900, 1100, 500)
    y = 100 * np.exp(-0.5 * ((x - 1000) / 3) ** 2)

    peaks = find_peaks_auto(Spectrum(x, y, metadata={"x_unit": "px"}), match_tolerance=5)

    assert peaks
    assert peaks[0]["matched_gas"] is None


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


def test_quantitative_chain_uses_only_dynamic_custom_gases(monkeypatch, tmp_path):
    path = tmp_path / "gas_library.json"
    monkeypatch.setenv("RAMAN_TOOL_GAS_LIBRARY", str(path))
    save_gas_library(
        {
            "CBrF₃": {
                "name": "哈龙1301",
                "center": 760,
                "half_width": 12,
                "coefficient": 1,
                "enabled": True,
                "quantitative": True,
            },
            "Xe": {
                "name": "氙气",
                "center": 900,
                "half_width": 10,
                "coefficient": 0.5,
                "enabled": True,
                "quantitative": True,
            },
            "AR": {
                "name": "氩气",
                "center": 1000,
                "half_width": 10,
                "coefficient": 1,
                "enabled": True,
                "quantitative": False,
            },
        },
        path,
    )
    reload_gas_library()
    x = np.linspace(700, 1050, 2000)
    y = (
        200 * np.exp(-0.5 * ((x - 760) / 3) ** 2)
        + 100 * np.exp(-0.5 * ((x - 900) / 3) ** 2)
        + 1000 * np.exp(-0.5 * ((x - 1000) / 3) ** 2)
    )

    result = calculate_gas_concentrations(Spectrum(x, y))

    assert result["gases"] == ["CBrF₃", "Xe"]
    assert set(result["percentages"]) == {"CBrF₃", "Xe"}
    assert [key for key, _label in get_quantitative_gas_choices()] == ["CBrF₃", "Xe"]
    assert sum(result["percentages"].values()) == pytest.approx(100.0)


def test_quantitative_chain_requires_at_least_one_enabled_gas(monkeypatch, tmp_path):
    path = tmp_path / "gas_library.json"
    monkeypatch.setenv("RAMAN_TOOL_GAS_LIBRARY", str(path))
    save_gas_library(
        {
            "AR": {
                "name": "氩气",
                "center": 1000,
                "half_width": 10,
                "coefficient": 1,
                "enabled": True,
                "quantitative": False,
            }
        },
        path,
    )
    reload_gas_library()

    with pytest.raises(ValueError, match="没有启用定量"):
        calculate_gas_concentrations(
            Spectrum(np.linspace(900, 1100, 100), np.ones(100))
        )



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


@pytest.mark.parametrize("field", ["center", "half_width", "coefficient"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 0, "invalid"])
def test_invalid_gas_numbers_are_rejected_not_silently_replaced(field, value):
    entry = {"center": 1000, "half_width": 10, "coefficient": 1}
    entry[field] = value
    with pytest.raises(ValueError, match=field):
        coerce_gas_library({"test_gas": entry})
