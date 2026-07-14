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
        "# Raman Shift\tIntensity",
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
