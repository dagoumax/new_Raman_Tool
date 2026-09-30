"""Compare the complete processing pipeline with declared reference compositions.

Synthetic references verify software arithmetic. Measured references additionally
need independently known composition, acquisition conditions, and calibrated
response coefficients; a passing synthetic comparison is not instrument approval.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from raman_tool import __version__
from raman_tool import safety
from raman_tool.gas_library import coerce_gas_library
from raman_tool.models import normalize_x_unit, RAMAN_SHIFT_UNIT
from raman_tool.processing import calculate_gas_concentrations
from raman_tool.validation import finite_float, integer_parameter, positive_float, validate_calibration
from raman_tool.workflows import load_spectrum


_MANIFEST_KEYS = {
    "schema_version", "sample_kind", "sample_id", "reference_source",
    "acquisition_conditions", "gas_library", "expected_percentages",
    "tolerance_percentage_points", "file", "import_options", "baseline_options",
    "strategy", "coefficient_provenance", "expected_peak_centers",
}


def _object(value, name: str, *, allowed: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if allowed is not None and set(value) - allowed:
        raise ValueError(f"Unknown {name} fields: {sorted(set(value) - allowed)}")
    return value


def _required_text(value: dict, key: str, context: str = "Reference manifest") -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{context} requires nonempty {key}")
    return text


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON number is not allowed: {value}")


def _checked_limit(path: Path, *, text_file: bool = False) -> int:
    safety.check_file_size(path)
    limit = safety.MAX_INPUT_FILE_BYTES
    if text_file or path.suffix.lower() in {".json", ".txt", ".asc"}:
        safety.check_text_file_size(path)
        limit = min(limit, safety.MAX_TEXT_FILE_BYTES)
    return limit


def _file_sha256(path: Path, *, text_file: bool = False) -> str:
    """Hash in bounded chunks, enforcing parser limits before opening the file."""
    limit = _checked_limit(path, text_file=text_file)
    initial = path.stat()
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            count += len(chunk)
            if count > limit:
                raise safety.InputLimitError(f"Input file grew beyond its size limit: {path}")
            digest.update(chunk)
    final = path.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError(f"Reference input changed while hashing: {path}")
    return digest.hexdigest()


def _reference_library(raw) -> dict:
    raw = _object(raw, "gas_library")
    allowed = {"name", "label", "x_unit", "center", "half_width", "coefficient",
               "color", "enabled", "quantitative"}
    for gas, entry in raw.items():
        entry = _object(entry, f"gas_library.{gas}", allowed=allowed)
        for field in ("center", "half_width", "coefficient", "quantitative"):
            if field not in entry:
                raise ValueError(f"gas_library.{gas} requires explicit {field}")
        for field in ("enabled", "quantitative"):
            if field in entry and not isinstance(entry[field], bool):
                raise ValueError(f"gas_library.{gas}.{field} must be boolean")
        if normalize_x_unit(entry.get("x_unit")) != RAMAN_SHIFT_UNIT:
            raise ValueError("Reference gas windows must use cm-1 units")
    library = coerce_gas_library(raw)
    if set(library) != set(raw):
        raise ValueError("Reference gas keys must be valid and must not change after normalization")
    return library


def _coefficient_provenance(manifest: dict, kind: str) -> dict | None:
    raw = manifest.get("coefficient_provenance")
    if raw is None and kind == "synthetic":
        return None
    raw = _object(raw, "coefficient_provenance", allowed={"status", "source", "applicability"})
    if not isinstance(raw.get("status"), str) or raw["status"] not in {"calibrated", "uncalibrated"}:
        raise ValueError("coefficient_provenance.status must be calibrated or uncalibrated")
    _required_text(raw, "source", "coefficient_provenance")
    if raw["status"] == "calibrated" or "applicability" in raw:
        _required_text(raw, "applicability", "coefficient_provenance")
    return raw.copy()


def _processing_options(manifest: dict) -> tuple[dict, dict | None]:
    options = _object(manifest.get("import_options", {}), "import_options",
                      allowed={"row_groups", "col_merge", "calibration", "row_mode"}).copy()
    groups = options.get("row_groups")
    if groups is not None and not isinstance(groups, str):
        raise ValueError("import_options.row_groups must be a row specification string or null")
    if "col_merge" in options:
        options["col_merge"] = integer_parameter(options["col_merge"], "col_merge", minimum=1)
    if "row_mode" in options:
        mode = options["row_mode"]
        if not isinstance(mode, str) or mode not in {"mean", "sum"}:
            raise ValueError("import_options.row_mode must be mean or sum")
    calibration = options.get("calibration")
    if calibration is not None:
        if not isinstance(calibration, (list, tuple)) or len(calibration) != 2:
            raise ValueError("import_options.calibration must contain slope and intercept")
        options["calibration"] = validate_calibration(*calibration)
    baseline = manifest.get("baseline_options")
    if baseline is not None:
        baseline = _object(baseline, "baseline_options", allowed={"method", "lam", "degree", "max_iter", "tol"}).copy()
        method = baseline.get("method", "arPLS")
        if not isinstance(method, str) or method not in {"arPLS", "poly"}:
            raise ValueError("baseline_options.method must be arPLS or poly")
        for key in ("lam", "tol"):
            if key in baseline:
                baseline[key] = positive_float(baseline[key], key)
        if baseline.get("tol", 1e-6) >= 1:
            raise ValueError("tol must be less than 1")
        for key, minimum in (("degree", 0), ("max_iter", 1)):
            if key in baseline:
                baseline[key] = integer_parameter(baseline[key], key, minimum=minimum)
    return options, baseline


def validate_standard(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    _checked_limit(manifest_path, text_file=True)
    manifest_stat = manifest_path.stat()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
    manifest = _object(manifest, "manifest", allowed=_MANIFEST_KEYS)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Reference manifest requires schema_version=1")
    kind = manifest.get("sample_kind")
    if not isinstance(kind, str) or kind not in {"synthetic", "measured"}:
        raise ValueError("sample_kind must explicitly be synthetic or measured")
    for key in ("sample_id", "reference_source", "acquisition_conditions", "file"):
        _required_text(manifest, key)
    library = _reference_library(manifest.get("gas_library"))
    gases = {key for key, info in library.items()
             if info.get("enabled", True) and info.get("quantitative", False)}
    if len(gases) < 2:
        raise ValueError("Reference composition validation requires at least two quantitative gases; "
                         "a single normalized signal is always 100% and cannot validate purity")
    provenance = _coefficient_provenance(manifest, kind)
    import_options, baseline_options = _processing_options(manifest)
    strategy = manifest.get("strategy", "peak_max")
    if not isinstance(strategy, str) or strategy not in {"peak_max", "peak_area"}:
        raise ValueError("strategy must be peak_max or peak_area")
    expected = manifest.get("expected_percentages", {})
    if not gases or not isinstance(expected, dict) or set(expected) != gases:
        raise ValueError("Expected composition must cover every enabled quantitative gas exactly")
    expected = {key: finite_float(value, f"expected {key}") for key, value in expected.items()}
    if any(value < 0 or value > 100 for value in expected.values()):
        raise ValueError("Expected percentages must be within 0..100")
    if abs(sum(expected.values()) - 100) > 1e-6:
        raise ValueError("Expected reference composition must sum to 100%")
    tolerances = manifest.get("tolerance_percentage_points")
    if not isinstance(tolerances, dict):
        tolerances = dict.fromkeys(gases, tolerances)
    if set(tolerances) != gases:
        raise ValueError("Tolerance must be defined for every quantitative gas")
    tolerances = {gas: positive_float(value, f"tolerance {gas}") for gas, value in tolerances.items()}
    peak_specs = _object(manifest.get("expected_peak_centers", {}), "expected_peak_centers")
    if set(peak_specs) - gases:
        raise ValueError("expected_peak_centers must only reference enabled quantitative gases")
    for gas, spec in peak_specs.items():
        _object(spec, f"expected_peak_centers.{gas}", allowed={"center", "tolerance_cm1"})
        finite_float(spec.get("center"), f"{gas} expected peak center")
        positive_float(spec.get("tolerance_cm1"), f"{gas} peak tolerance_cm1")
    source_path = (manifest_path.parent / manifest["file"]).resolve()
    if source_path.suffix.lower() not in {".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}:
        if (import_options.get("row_groups") or import_options.get("col_merge", 1) != 1
                or import_options.get("row_mode", "mean") != "mean"):
            raise ValueError("Row selection, row sum and column merging are supported only for image formats")
    source_hash = _file_sha256(source_path)
    source_stat = source_path.stat()
    spectrum = load_spectrum(
        source_path, **import_options, baseline_options=baseline_options,
    )
    after_load = source_path.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (after_load.st_size, after_load.st_mtime_ns):
        raise ValueError("Reference spectrum changed while loading; rerun with a stable file")
    result = calculate_gas_concentrations(
        spectrum, strategy=strategy, library=library,
    )
    comparisons = {}
    for gas in sorted(gases):
        actual = result["percentages"][gas]
        error = actual - expected[gas]
        comparisons[gas] = {
            "expected_percent": expected[gas], "actual_percent": actual,
            "error_percentage_points": error,
            "tolerance_percentage_points": tolerances[gas],
            "passed": abs(error) <= tolerances[gas],
        }
    peak_comparisons = {}
    for gas, spec in peak_specs.items():
        actual = result["peaks"][gas]["center"]
        expected_center = finite_float(spec["center"], f"{gas} expected peak center")
        tolerance = positive_float(spec["tolerance_cm1"], f"{gas} peak tolerance_cm1")
        detected = result["peaks"][gas]["intensity"] > 0
        peak_comparisons[gas] = {
            "expected_cm1": expected_center, "actual_cm1": actual,
            "error_cm1": actual - expected_center, "tolerance_cm1": tolerance,
            "positive_peak_signal": detected,
            "passed": detected and abs(actual - expected_center) <= tolerance,
        }
    warnings = list(result.get("warnings", []))
    calibrated = provenance is not None and provenance["status"] == "calibrated"
    if kind == "measured" and not calibrated:
        warnings.append("修正系数来源标记为未经标定；本次仅比较加权信号占比，不能据此验证实际纯度或混合物浓度。")
    pure_reference = sum(value > 0 for value in expected.values()) == 1
    if kind == "measured" and pure_reference:
        warnings.append("单一纯气体样品不能确定不同气体的相对响应系数；其他窗口用于检查非目标信号，不能替代独立混合标样。")
    if kind == "synthetic":
        scope = "Synthetic arithmetic/pipeline verification only; no measured standard sample was validated."
    elif not calibrated:
        scope = ("Exploratory weighted-signal comparison with a declared measured reference using uncalibrated "
                 "response coefficients; PASS means numerical agreement only, not validated gas purity or "
                 "physical composition, mixture response, detection limits, or instrument approval.")
    else:
        scope = ("Comparison with a declared measured reference using coefficients declared calibrated for the "
                 "recorded applicability; validates agreement for this sample only, not other mixtures, "
                 "acquisition settings, detection limits, or independent coefficient calibration.")
    manifest_hash = _file_sha256(manifest_path, text_file=True)
    final_manifest_stat = manifest_path.stat()
    if (manifest_stat.st_size, manifest_stat.st_mtime_ns) != (final_manifest_stat.st_size, final_manifest_stat.st_mtime_ns):
        raise ValueError("Reference manifest changed during validation; rerun with a stable file")
    return {
        "schema_version": 1, "tool_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_id": manifest["sample_id"], "sample_kind": kind,
        "reference_source": manifest["reference_source"],
        "acquisition_conditions": manifest["acquisition_conditions"],
        "source_file": str(source_path), "source_sha256": source_hash,
        "manifest_sha256": manifest_hash,
        "settings": {
            "import_options": import_options,
            "baseline_options": baseline_options,
            "gas_library": library, "strategy": result["strategy"],
            "coefficient_provenance": provenance,
        },
        "comparisons": comparisons, "peak_comparisons": peak_comparisons,
        "passed": all(item["passed"] for item in [*comparisons.values(), *peak_comparisons.values()]),
        "comparison_kind": "weighted_signal_percentages",
        "response_coefficients_declared_calibrated": calibrated,
        "physical_concentration_validated": False,
        "scope": scope, "warnings": warnings,
    }


def write_validation_report(report: dict, path: str | Path, *, overwrite: bool = False) -> Path:
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(serialized)
    return path
