import numpy as np

from raman_tool.exporters import export_spectrum, normalize_export_path, unique_path
from raman_tool.models import Spectrum


def test_export_spectrum_writes_two_columns(tmp_path):
    spec = Spectrum(
        raman_shift=np.array([100.0, 200.5]),
        intensity=np.array([10.0, 20.25]),
        filename="sample.sif",
    )

    out = export_spectrum(spec, tmp_path / "sample.asc")

    assert out.suffix == ".asc"
    assert out.read_text(encoding="utf-8").splitlines() == [
        "# Raman Shift (cm^-1)\tIntensity",
        "100\t10",
        "200.5\t20.25",
    ]


def test_normalize_export_path_uses_selected_suffix(tmp_path):
    assert normalize_export_path(tmp_path / "sample.sif", ".txt").name == "sample.txt"
    assert normalize_export_path(tmp_path / "sample", ".asc").name == "sample.asc"


def test_unique_path_appends_number_for_existing_file(tmp_path):
    existing = tmp_path / "sample.asc"
    existing.write_text("old", encoding="utf-8")

    assert unique_path(existing).name == "sample_1.asc"



def test_export_spectrum_refuses_overwrite_by_default(tmp_path):
    import pytest

    spec = Spectrum(
        raman_shift=np.array([100.0]),
        intensity=np.array([10.0]),
        filename="sample.sif",
    )
    out = tmp_path / "sample.asc"
    out.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_spectrum(spec, out)


def test_export_spectrum_can_overwrite_when_requested(tmp_path):
    spec = Spectrum(
        raman_shift=np.array([100.0]),
        intensity=np.array([10.0]),
        filename="sample.sif",
    )
    out = tmp_path / "sample.asc"
    out.write_text("old", encoding="utf-8")

    export_spectrum(spec, out, overwrite=True)

    assert out.read_text(encoding="utf-8").splitlines()[1] == "100\t10"


def test_export_spectrum_can_embed_processing_history(tmp_path):
    spec = Spectrum(
        raman_shift=np.array([100.0]),
        intensity=np.array([10.0]),
        metadata={
            "processing_history": [
                {"sequence": 1, "action": "基线校正", "parameters": {"method": "poly"}}
            ]
        },
    )

    out = export_spectrum(spec, tmp_path / "traceable.asc", include_history=True)
    lines = out.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "# Raman Shift (cm^-1)\tIntensity"
    assert lines[1].startswith("# Processing History\t")
    assert "基线校正" in lines[1]


def test_export_spectrum_marks_pixel_axis_and_calibration(tmp_path):
    spec = Spectrum(
        raman_shift=np.array([100.0, 102.0]),
        intensity=np.array([10.0, 20.0]),
        metadata={"x_unit": "px", "calibration": (2.0, 100.0)},
    )

    out = export_spectrum(spec, tmp_path / "pixels.txt")
    lines = out.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "# Pixel (px)\tIntensity"
    assert lines[1] == "# Calibration\traman_shift = 2 * pixel + 100"
