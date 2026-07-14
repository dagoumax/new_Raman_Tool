import json

from raman_tool.presets import delete_workflow_preset, load_workflow_presets, save_workflow_preset


def test_default_workflow_presets_are_available(tmp_path):
    presets = load_workflow_presets(tmp_path / "missing.json")

    assert "标准气体分析" in presets
    assert presets["标准气体分析"]["baseline_method"] == "arPLS"
    assert presets["峰面积定量"]["concentration_strategy"] == "peak_area"


def test_save_and_load_workflow_preset_roundtrip(tmp_path):
    path = tmp_path / "presets.json"
    save_workflow_preset(
        "my preset",
        {
            "baseline_method": "poly",
            "baseline_lam": 50000,
            "baseline_degree": 4,
            "auto_baseline": False,
            "batch_baseline": True,
            "concentration_gas": "O2",
            "concentration_window": 14,
            "concentration_strategy": "peak_area",
            "row_mode": "sum",
            "col_merge": 2,
            "row_groups": "1-40, 91-130",
            "show_individual_rows": True,
            "calibration_px1": "500",
            "calibration_shift1": "1555",
            "calibration_px2": "800",
            "calibration_shift2": "2331",
            "show_gas_peaks": False,
            "show_auto_peaks": True,
        },
        path,
    )

    loaded = load_workflow_presets(path)

    assert loaded["my preset"]["baseline_method"] == "poly"
    assert loaded["my preset"]["baseline_degree"] == 4
    assert loaded["my preset"]["concentration_gas"] == "O2"
    assert loaded["my preset"]["row_mode"] == "sum"
    assert loaded["my preset"]["row_groups"] == "1-40, 91-130"
    assert loaded["my preset"]["show_individual_rows"] is True
    assert loaded["my preset"]["calibration_px1"] == "500"
    assert loaded["my preset"]["calibration_shift2"] == "2331"


def test_preset_preserves_mixed_case_gas_key(tmp_path):
    path = tmp_path / "presets.json"
    save_workflow_preset("halon", {"concentration_gas": "CBrF₃"}, path)

    assert load_workflow_presets(path)["halon"]["concentration_gas"] == "CBrF₃"


def test_invalid_preset_values_are_coerced(tmp_path):
    path = tmp_path / "presets.json"
    path.write_text(json.dumps({"bad": {"baseline_degree": "oops", "col_merge": 0, "show_gas_peaks": "false"}}), encoding="utf-8")

    loaded = load_workflow_presets(path)

    assert loaded["bad"]["baseline_degree"] == 3
    assert loaded["bad"]["col_merge"] == 1
    assert loaded["bad"]["show_gas_peaks"] is False



def test_delete_workflow_preset(tmp_path):
    path = tmp_path / "presets.json"
    save_workflow_preset("to delete", {"baseline_method": "poly"}, path)

    delete_workflow_preset("to delete", path)
    loaded = load_workflow_presets(path)

    assert "to delete" not in loaded


def test_delete_builtin_workflow_preset_persists(tmp_path):
    path = tmp_path / "presets.json"

    delete_workflow_preset("标准气体分析", path)

    assert "标准气体分析" not in load_workflow_presets(path)
    assert json.loads(path.read_text(encoding="utf-8"))["标准气体分析"] is None


def test_saving_another_preset_keeps_builtin_deletion(tmp_path):
    path = tmp_path / "presets.json"
    delete_workflow_preset("标准气体分析", path)

    save_workflow_preset("custom", {"baseline_method": "poly"}, path)

    loaded = load_workflow_presets(path)
    assert "标准气体分析" not in loaded
    assert "custom" in loaded
