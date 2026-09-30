"""Spectrum export helpers."""

import json

from pathlib import Path

from raman_tool.models import Spectrum
from raman_tool.validation import validate_calibration


SUPPORTED_EXPORT_FORMATS = {".asc", ".txt"}


def normalize_export_path(filepath: str | Path, suffix: str | None = None) -> Path:
    path = Path(filepath)
    if suffix:
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        path = path.with_suffix(suffix.lower())
    elif path.suffix.lower() not in SUPPORTED_EXPORT_FORMATS:
        path = path.with_suffix(".txt")
    return path


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Cannot create unique output path for {path}")


def export_spectrum(
    spectrum: Spectrum,
    filepath: str | Path,
    suffix: str | None = None,
    overwrite: bool = False,
    include_history: bool = False,
) -> Path:
    path = normalize_export_path(filepath, suffix)
    if path.suffix.lower() not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {path.suffix}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    x_header = "Pixel (px)" if not spectrum.is_raman_shift else "Raman Shift (cm^-1)"
    lines = [f"# {x_header}\tIntensity"]
    calibration = spectrum.metadata.get("calibration")
    if calibration is not None:
        calibration = validate_calibration(*calibration)
        lines.append(f"# Calibration\traman_shift = {calibration[0]:.17g} * pixel + {calibration[1]:.17g}")
    if include_history:
        history = spectrum.metadata.get("processing_history", [])
        lines.append(
            "# Processing History\t"
            + json.dumps(history, ensure_ascii=False, separators=(",", ":"))
        )
    lines.extend(
        f"{x:.17g}\t{y:.17g}"
        for x, y in zip(spectrum.raman_shift, spectrum.intensity)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
