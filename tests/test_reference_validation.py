import json
from pathlib import Path

import numpy as np
import pytest

from raman_tool import safety
from raman_tool.config import DEFAULT_CONFIG
from raman_tool.cli import main
from raman_tool.reference_validation import validate_standard, write_validation_report


FIXTURE = Path(__file__).parent / "fixtures" / "standards" / "synthetic-mixture.json"


def write_manifest(tmp_path, **updates):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["file"] = str(FIXTURE.with_suffix(".txt").resolve())
    data.update(updates)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("strategy", ["peak_max", "peak_area"])
def test_analytic_reference_pipeline_matches_declared_truth(tmp_path, strategy):
    report = validate_standard(write_manifest(tmp_path, strategy=strategy))
    assert report["passed"]
    assert report["sample_kind"] == "synthetic"
    assert "no measured standard" in report["scope"]
    assert report["comparisons"]["N2"]["actual_percent"] == pytest.approx(78)
    assert report["comparisons"]["O2"]["actual_percent"] == pytest.approx(21)
    assert report["comparisons"]["CO2"]["actual_percent"] == pytest.approx(1)
    assert len(report["source_sha256"]) == 64
    assert report["settings"]["gas_library"]["O2"]["coefficient"] == 2


def test_reference_mismatch_fails_and_cli_returns_failure(tmp_path):
    path = write_manifest(tmp_path, expected_percentages={"N2": 70, "O2": 29, "CO2": 1})
    report_path = tmp_path / "report.json"
    assert main(["validate-standard", str(path), "--report", str(report_path)]) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert not report["passed"]
    assert report["comparisons"]["N2"]["error_percentage_points"] == pytest.approx(8)
    assert report["comparisons"]["CO2"]["passed"]


@pytest.mark.parametrize("updates", [
    {"expected_percentages": {"CO2": 100}},
    {"expected_percentages": {"N2": 78, "O2": 21, "CO2": 0}},
    {"sample_kind": None}, {"acquisition_conditions": ""},
    {"tolerance_percentage_points": -1},
    {"tolerance_percentage_points": float("nan")},
])
def test_reference_requires_complete_finite_explicit_truth(tmp_path, updates):
    with pytest.raises(ValueError):
        validate_standard(write_manifest(tmp_path, **updates))


def test_validation_report_does_not_overwrite_existing_file(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("prior")
    with pytest.raises(FileExistsError):
        write_validation_report({"passed": True}, target)
    assert target.read_text() == "prior"


@pytest.mark.parametrize("key", [
    "schema_version", "sample_kind", "sample_id", "reference_source",
    "acquisition_conditions", "gas_library", "expected_percentages",
    "tolerance_percentage_points", "file",
])
def test_missing_required_manifest_fields_raise_valueerror(tmp_path, key):
    path = write_manifest(tmp_path)
    data = json.loads(path.read_text())
    del data[key]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_standard(path)
    assert main(["validate-standard", str(path)]) == 1


@pytest.mark.parametrize("updates", [
    {"schema_version": True}, {"schema_version": 1.0},
    {"sample_kind": []}, {"file": []}, {"gas_library": []},
    {"gas_library": {"N2": None}}, {"expected_percentages": []},
    {"import_options": []}, {"import_options": {"unknown": 1}},
    {"import_options": {"col_merge": 0}}, {"import_options": {"col_merge": True}},
    {"import_options": {"row_mode": []}}, {"import_options": {"row_groups": [1, 3]}},
    {"import_options": {"calibration": [1]}},
    {"import_options": {"calibration": [0, 100]}},
    {"import_options": {"col_merge": 2}},  # Must not silently ignore image options on TXT.
    {"baseline_options": []}, {"baseline_options": {"method": []}},
    {"baseline_options": {"degree": 1.2}}, {"baseline_options": {"max_iter": 0}},
    {"baseline_options": {"tol": 1}}, {"baseline_options": {"unknown": 1}},
    {"strategy": []}, {"strategy": "unknown"},
    {"expected_peak_centers": []}, {"expected_peak_centers": {"CO2": {}}},
    {"expected_peak_centers": {"unknown": {"center": 1, "tolerance_cm1": 1}}},
    {"unknown": "typo"},
])
def test_bad_manifest_types_or_unknown_fields_raise_valueerror(tmp_path, updates):
    with pytest.raises(ValueError):
        validate_standard(write_manifest(tmp_path, **updates))


@pytest.mark.parametrize("change", [
    {"coefficient": None}, {"quantitative": "true"}, {"enabled": 1},
    {"coefficent": 1}, {"x_unit": "px"},
])
def test_reference_library_rejects_silent_coercion(tmp_path, change):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["gas_library"]["N2"].update(change)
    with pytest.raises(ValueError):
        validate_standard(write_manifest(tmp_path, gas_library=data["gas_library"]))


def test_reference_library_requires_explicit_response_coefficients(tmp_path):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del data["gas_library"]["N2"]["coefficient"]
    with pytest.raises(ValueError, match="coefficient"):
        validate_standard(write_manifest(tmp_path, gas_library=data["gas_library"]))


def test_duplicate_json_fields_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1, "schema_version": 2}')
    with pytest.raises(ValueError, match="Duplicate"):
        validate_standard(path)


@pytest.mark.parametrize("provenance", [
    None, [], {}, {"status": []}, {"status": "unknown", "source": "test"},
    {"status": "uncalibrated", "source": ""},
    {"status": "calibrated", "source": "test"},
])
def test_measured_comparison_requires_explicit_coefficient_provenance(tmp_path, provenance):
    with pytest.raises(ValueError, match="coefficient_provenance"):
        validate_standard(write_manifest(tmp_path, sample_kind="measured", coefficient_provenance=provenance))


def test_uncalibrated_measured_agreement_is_not_physical_concentration_validation(tmp_path):
    report = validate_standard(write_manifest(
        tmp_path, sample_kind="measured",
        reference_source="Synthetic test fixture exercising the measured-reference workflow, not a real standard.",
        coefficient_provenance={"status": "uncalibrated", "source": "Test fixture multipliers; no measured calibration."},
    ))
    assert report["passed"]
    assert not report["response_coefficients_declared_calibrated"]
    assert not report["physical_concentration_validated"]
    assert "uncalibrated" in report["scope"]
    assert "numerical agreement only" in report["scope"]
    assert report["settings"]["coefficient_provenance"]["status"] == "uncalibrated"


def test_declared_calibration_does_not_become_independent_instrument_approval(tmp_path):
    provenance = {"status": "calibrated", "source": "Test-only certificate placeholder", "applicability": "Test fixture only"}
    report = validate_standard(write_manifest(tmp_path, sample_kind="measured", coefficient_provenance=provenance))
    assert report["passed"]
    assert report["response_coefficients_declared_calibrated"]
    assert not report["physical_concentration_validated"]
    assert "this sample only" in report["scope"]


def test_single_gas_100_percent_reference_cannot_pass_tautologically(tmp_path):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="single normalized signal"):
        validate_standard(write_manifest(
            tmp_path, sample_kind="measured", gas_library={"CO2": data["gas_library"]["CO2"]},
            expected_percentages={"CO2": 100},
            coefficient_provenance={"status": "uncalibrated", "source": "Defaults"},
        ))


def test_pure_reference_with_other_windows_retains_limited_scope(tmp_path):
    x = np.arange(1000., 2501.)
    y = 10 + 100 * np.maximum(1 - np.abs(x - 1388) / 40, 0)
    data_path = tmp_path / "synthetic-pure-co2-for-workflow-test.txt"
    np.savetxt(data_path, np.column_stack((x, y)))
    path = write_manifest(
        tmp_path, file=str(data_path), sample_kind="measured",
        reference_source="Synthetic pure-CO2 fixture for validator testing, not a measured standard.",
        expected_percentages={"CO2": 100, "N2": 0, "O2": 0}, tolerance_percentage_points=1,
        coefficient_provenance={"status": "uncalibrated", "source": "Default test coefficients"},
        expected_peak_centers={"CO2": {"center": 1388, "tolerance_cm1": 1}},
    )
    report = validate_standard(path)
    assert report["passed"]
    assert report["peak_comparisons"]["CO2"]["passed"]
    assert any("相对响应系数" in warning for warning in report["warnings"])
    assert not report["physical_concentration_validated"]


def test_peak_position_mismatch_fails_even_when_percentages_agree(tmp_path):
    report = validate_standard(write_manifest(
        tmp_path, expected_peak_centers={"CO2": {"center": 1398, "tolerance_cm1": 1}},
    ))
    assert all(item["passed"] for item in report["comparisons"].values())
    assert not report["peak_comparisons"]["CO2"]["passed"]
    assert not report["passed"]


def test_hashing_does_not_materialize_whole_source_file(tmp_path, monkeypatch):
    path = write_manifest(tmp_path)
    def forbid_read_bytes(_path):
        pytest.fail("Source/manifest hashing must not call Path.read_bytes")
    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    assert validate_standard(path)["passed"]


def test_oversized_reference_is_rejected_before_open_or_hash(tmp_path, monkeypatch):
    source = tmp_path / "oversized.txt"
    source.write_bytes(b" " * (1024 * 1024 + 1))
    path = write_manifest(tmp_path, file=str(source))
    limits = DEFAULT_CONFIG["safety"].copy()
    limits["max_input_file_mb"] = 1
    monkeypatch.setattr(safety, "_safety_config", lambda: limits)
    actual_open = Path.open
    def guarded_open(self, *args, **kwargs):
        if self == source:
            pytest.fail("Oversized reference should fail before source opening/hash")
        return actual_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(safety.InputLimitError, match="too large"):
        validate_standard(path)


def test_report_serialization_error_preserves_existing_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("original")
    with pytest.raises(ValueError):
        write_validation_report({"bad": float("nan")}, path, overwrite=True)
    assert path.read_text() == "original"
