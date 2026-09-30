"""Read-only import audit of user-supplied samples; writes reports in the project.

This records file hashes, declared calibration, and raw spectral diagnostics.
It neither fits calibration to the data nor treats directory names as certified
composition. Scripts found in the source directories are never executed.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from raman_tool.readers import read_file, SUPPORTED_FORMATS


def describe(path: Path) -> dict:
    result = {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        spectrum = read_file(path)
        x, y = spectrum.raman_shift, spectrum.intensity
        result.update(status="read", points=spectrum.size, unit=spectrum.x_unit,
                      x_range=[float(x.min()), float(x.max())],
                      intensity_range=[float(y.min()), float(y.max())],
                      median_pixel_spacing=float(np.median(np.abs(np.diff(x)))) if x.size > 1 else None)
        result["metadata"] = {key: spectrum.metadata.get(key) for key in (
            "format", "calibration_source", "calibration_warning", "calibration_coefficients",
            "laser_wavelength", "image_shape", "frame_data_offset", "frame_data_values",
            "data_format", "ccd_model", "spectrometer", "detector_xbin", "detector_ybin"
        )}
        peaks, properties = find_peaks(y, prominence=0)
        ranked = np.argsort(properties["prominences"])[-10:][::-1]
        result["strongest_raw_peaks"] = [
            {"x": float(x[peaks[index]]), "intensity": float(y[peaks[index]]),
             "prominence": float(properties["prominences"][index])} for index in ranked
        ]
        result["reference_windows"] = {}
        if spectrum.is_raman_shift:
            for label, center in {"CO2_lower": 1285.0, "CO2_upper": 1388.0,
                                  "O2": 1555.0, "N2": 2330.0}.items():
                mask = np.abs(x - center) <= 35
                if np.count_nonzero(mask) < 3:
                    continue
                wx, wy = x[mask], y[mask]
                baseline = wy[0] + (wy[-1] - wy[0]) * (wx - wx[0]) / (wx[-1] - wx[0])
                corrected = wy - baseline
                i = int(np.argmax(corrected))
                result["reference_windows"][label] = {
                    "reference_center": center, "half_width": 35,
                    "maximum_above_endpoint_line_at": float(wx[i]),
                    "height_above_endpoint_line": float(corrected[i]),
                }
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc))
    return result


def audit(directory: Path) -> dict:
    inventory = sorted(path for path in directory.rglob("*") if path.is_file())
    samples = [path for path in inventory if path.suffix.lower() in SUPPORTED_FORMATS]
    groups = Counter(str(path.parent.relative_to(directory)) for path in samples)
    records = []
    for index, path in enumerate(samples, 1):
        record = describe(path)
        record["relative_path"] = str(path.relative_to(directory))
        records.append(record)
        if index % 100 == 0:
            print(f"Read {index}/{len(samples)} in {directory.name}", flush=True)
    return {
        "directory": str(directory.resolve()),
        "total_files": len(inventory),
        "extensions": dict(Counter(path.suffix.lower() for path in inventory)),
        "sample_counts_by_subdirectory": dict(groups),
        "read_status_counts": dict(Counter(record["status"] for record in records)),
        "unit_counts": dict(Counter(record.get("unit", "error") for record in records)),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--co2", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Default reader; uncorrected peak prominence; local endpoint-line window heights; no calibration fitting",
        "limitations": ["CO2 identity is user-provided, without certified purity/tolerance", "Candidate directory identity is unconfirmed", "Parsed calibration is file provenance, not independent accuracy certification"],
        "co2": audit(args.co2), "unconfirmed_candidates": audit(args.candidate_directory),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {field: report[key][field] for field in ("total_files", "read_status_counts", "unit_counts")}
                      for key in ("co2", "unconfirmed_candidates")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
