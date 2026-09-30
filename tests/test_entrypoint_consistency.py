"""The same inputs/options must survive all UI adapters without substitutions."""

import io
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListWidgetItem
from rich.console import Console

from raman_tool import cli, tui
from raman_tool.gas_library import DEFAULT_GAS_LIBRARY, get_quantitative_gas_choices
from raman_tool.history import SpectrumSession
from raman_tool.processing import calculate_concentration
from raman_tool.qt_gui import RamanQtGUI
from raman_tool.workflows import load_spectrum


_APP = QApplication.instance() or QApplication([])


def drain(window):
    deadline = time.monotonic() + 10
    while window._workers.workers and time.monotonic() < deadline:
        _APP.processEvents()
        QTest.qWait(5)
    _APP.processEvents()
    assert not window._workers.workers


@pytest.mark.parametrize("strategy", ["peak_max", "peak_area"])
@pytest.mark.parametrize("baseline", [False, True])
def test_qt_single_batch_cli_and_terminal_match(tmp_path, monkeypatch, capsys, strategy, baseline):
    library_path = tmp_path / "gases.json"
    library_path.write_text(json.dumps(DEFAULT_GAS_LIBRARY), encoding="utf-8")
    monkeypatch.setenv("RAMAN_TOOL_GAS_LIBRARY", str(library_path))
    x = np.arange(1000., 2500.5, .5)
    y = 10 + .002 * x
    for center, height in [(1388, 120), (1555, 200), (2330, 600)]:
        y += height * np.exp(-.5 * ((x - center) / 4) ** 2)
    path = tmp_path / "mixture.txt"
    np.savetxt(path, np.column_stack((x, y)))
    options = {"method": "poly", "degree": 1} if baseline else None
    spec = load_spectrum(path, baseline_options=options)
    # Intentionally differs from the CO2 library window, exercising UI override.
    expected = calculate_concentration(spec, "CO2", window=6, strategy=strategy)
    percentages = expected["all_concentrations"]["percentages"]
    win = RamanQtGUI()
    monkeypatch.setattr(win, "_update_plot", lambda: None)
    captured = []
    monkeypatch.setattr(win, "_on_batch_done", lambda ok, fail, rows: captured.append((ok, fail, rows)))
    try:
        win.current_session = SpectrumSession(spec)
        win.current_spectrum = win.current_session.current
        win.conc_gas.setCurrentIndex(win.conc_gas.findData("CO2"))
        win.conc_window.setText("6")
        win.conc_strategy.setCurrentIndex(1 if strategy == "peak_area" else 0)
        win._on_concentration()
        assert f"{expected['concentration']:.4f}%" in win.conc_result.text()
        win.batch_baseline_cb.setChecked(baseline)
        win.baseline_method.setCurrentIndex(1)
        win.baseline_degree.setCurrentText("1")
        item = QListWidgetItem(path.name)
        item.setData(Qt.UserRole, str(path))
        win.file_list.addItem(item)
        win._on_batch()
        drain(win)
        assert captured[0][:2] == (1, 0)
        assert captured[0][2][0]["percentages"] == pytest.approx(percentages, abs=1e-12)
    finally:
        win.close()
        drain(win)

    args = ["concentration", str(path), "CO2", "--window", "6", "--strategy", strategy]
    if baseline:
        args += ["--baseline", "-m", "poly", "-d", "1"]
    assert cli.main(args) == 0
    output = capsys.readouterr().out
    for gas, value in percentages.items():
        assert f"{gas}: {value:.4f}%" in output

    terminal = tui.RamanTUI()
    terminal.current_spectrum = spec
    monkeypatch.setattr(terminal, "_press_enter", lambda: None)
    choices = [gas for gas, label in get_quantitative_gas_choices()]
    monkeypatch.setattr(tui.IntPrompt, "ask", lambda *a, **kw: choices.index("CO2") + 1)
    monkeypatch.setattr(tui.FloatPrompt, "ask", lambda *a, **kw: 6)
    monkeypatch.setattr(tui.Prompt, "ask", lambda *a, **kw: strategy)
    rendered = io.StringIO()
    monkeypatch.setattr(tui, "console", Console(file=rendered, width=240, force_terminal=False))
    terminal._calc_concentration()
    for value in percentages.values():
        assert f"{value:.4f} %" in rendered.getvalue()


def test_applying_preset_does_not_discard_other_file_edits(monkeypatch):
    win = RamanQtGUI()
    try:
        raw = load_spectrum(__file__.replace("test_entrypoint_consistency.py", "fixtures/standards/synthetic-mixture.txt"))
        session = SpectrumSession(raw)
        session.apply(raw, "recorded processing")
        win.spectrum_sessions["other-file"] = session
        win._apply_workflow_preset({"auto_baseline": False})
        assert win.spectrum_sessions["other-file"] is session
        assert session.audit_count == 1
    finally:
        win.close()
