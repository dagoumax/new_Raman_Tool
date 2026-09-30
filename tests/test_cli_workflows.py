import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from raman_tool import cli
from raman_tool.processing import subtract_baseline
from raman_tool.workflows import collect_spectrum_files, load_spectrum


def test_cli_honors_degree_using_actual_linear_baseline(tmp_path, monkeypatch):
    source = tmp_path / "linear.txt"
    np.savetxt(source, np.column_stack((np.arange(10.), 2 * np.arange(10.) + 7)))
    seen = {}
    def capture(spectrum, baseline, corrected, **kwargs):
        seen["result"] = corrected.intensity
        return None
    monkeypatch.setattr(cli, "plot_baseline", capture)
    monkeypatch.setattr(cli, "save_figure", lambda *a, **kw: "mock.png")
    # Degree 0 must remove the constant mean, preserving the slope.
    assert cli.main(["baseline", str(source), "-m", "poly", "-d", "0", "--no-show"]) == 0
    assert np.allclose(seen["result"], 2 * np.arange(10.) - 9)


def test_batch_visits_mixed_case_files_once_and_excludes_directories(tmp_path, monkeypatch):
    for name in ("a.txt", "b.TxT", "c.ASC"):
        (tmp_path / name).write_text("1 2\n2 3\n3 4\n")
    (tmp_path / "directory.txt").mkdir()
    assert len(collect_spectrum_files(tmp_path)) == 3
    captured = []
    monkeypatch.setattr(cli, "plot_spectrum", lambda spec, **kw: captured.append(spec.filename))
    monkeypatch.setattr(cli, "save_figure", lambda *a, **kw: "mock.png")
    assert cli.main(["batch", str(tmp_path)]) == 0
    assert sorted(captured) == ["a.txt", "b.TxT", "c.ASC"]


def test_cli_and_shared_import_use_same_calibration_and_validation(tmp_path, capsys):
    source = tmp_path / "pixels.txt"
    source.write_text("10\n20\n30\n")
    assert cli.main(["info", str(source), "--calibration", "2,100"]) == 0
    output = capsys.readouterr().out
    assert "100.00" in output and "104.00" in output
    assert cli.main(["info", str(source), "--calibration", "0,100"]) == 1
    assert cli.main(["snr", str(source), "--noise", "100,200"]) == 1
    assert cli.main(["snr", str(source), "--peak", "bad"]) == 1


def test_shared_load_and_process_matches_direct_core(tmp_path):
    source = tmp_path / "raw.txt"
    np.savetxt(source, np.column_stack((np.arange(12.), np.arange(12.) ** 2 + 5)))
    raw = load_spectrum(source)
    options = {"method": "poly", "degree": 1}
    direct = subtract_baseline(raw, **options)
    combined = load_spectrum(source, baseline_options=options)
    assert np.allclose(direct.intensity, combined.intensity)


def test_module_entrypoint_returns_nonzero_for_failed_command(tmp_path):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, "-m", "raman_tool", "info", str(tmp_path / "missing.txt")],
        capture_output=True, env=environment, timeout=30,
    )
    assert result.returncode == 1
