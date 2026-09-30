import os
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from raman_tool.history import SpectrumSession
from raman_tool.models import Spectrum
from raman_tool.qt_gui import RamanQtGUI


_APP = QApplication.instance() or QApplication([])


def wait_for_workers(window):
    deadline = time.monotonic() + 5
    while window._workers.workers and time.monotonic() < deadline:
        _APP.processEvents()
        QTest.qWait(5)
    _APP.processEvents()
    assert not window._workers.workers


def test_operation_panel_preset_roundtrip():
    window = RamanQtGUI()
    try:
        window.baseline_method.setCurrentIndex(1)
        window.baseline_lam.setText("54321")
        window.baseline_degree.setCurrentText("4")
        window.auto_baseline_cb.setChecked(False)
        window.batch_baseline_cb.setChecked(True)
        window.conc_gas.addItem("CBrF₃ (Halon 1301)", "CBrF₃")
        window.conc_gas.setCurrentIndex(window.conc_gas.count() - 1)
        window.conc_window.setText("14")
        window.conc_strategy.setCurrentIndex(1)
        window.row_mode_combo.setCurrentIndex(1)
        window.img_col_merge.setText("2")
        window.img_row_groups.setText("1-40, 91-130")
        window.img_show_rows_cb.setChecked(True)
        window.cal_px1.setText("500")
        window.cal_rs1.setText("1555")
        window.cal_px2.setText("800")
        window.cal_rs2.setText("2331")

        preset = window._collect_workflow_preset()
        window.img_row_groups.clear()
        window.img_show_rows_cb.setChecked(False)
        window.cal_px1.clear()
        window.conc_gas.setCurrentIndex(0)
        window._apply_workflow_preset(preset)

        assert window.baseline_degree.currentText() == "4"
        assert window.conc_gas.currentData() == "CBrF₃"
        assert window.img_row_groups.text() == "1-40, 91-130"
        assert window.img_show_rows_cb.isChecked()
        assert window.cal_px1.text() == "500"
        assert window.cal_rs2.text() == "2331"
    finally:
        window.close()


def test_qt_allows_repeated_baseline_and_restores_raw_data():
    window = RamanQtGUI()
    try:
        raw = Spectrum(
            np.arange(20, dtype=np.float64),
            np.linspace(5.0, 10.0, 20) + np.sin(np.arange(20)),
            filename="sample.txt",
        )
        window.current_session = SpectrumSession(raw)
        window.current_spectrum = window.current_session.current
        window.baseline_method.setCurrentIndex(1)
        window.baseline_degree.setCurrentText("1")

        window._on_baseline()
        wait_for_workers(window)
        first_result = window.current_spectrum.intensity.copy()
        window._on_baseline()
        wait_for_workers(window)

        assert window.current_session.state_count == 3
        assert not np.array_equal(window.current_spectrum.intensity, first_result)
        assert np.array_equal(window.current_session.raw.intensity, raw.intensity)
        assert window.baseline_undo_btn.isEnabled()

        window._on_restore_raw()
        assert np.array_equal(window.current_spectrum.intensity, raw.intensity)
        assert window.current_session.audit_log[-1]["action"] == "恢复原始数据"

        window._on_undo_processing()
        assert not np.array_equal(window.current_spectrum.intensity, raw.intensity)
    finally:
        window.close()
        wait_for_workers(window)


def test_qt_disables_gas_actions_for_uncalibrated_pixel_axis():
    window = RamanQtGUI()
    try:
        window.current_spectrum = Spectrum(
            np.arange(20, dtype=np.float64),
            np.ones(20, dtype=np.float64),
            metadata={"x_unit": "px"},
        )
        window._update_axis_dependent_ui()

        assert not window.gas_peaks_cb.isEnabled()
        assert not window.conc_calculate_btn.isEnabled()

        window.current_spectrum = Spectrum(
            np.arange(20, dtype=np.float64),
            np.ones(20, dtype=np.float64),
            metadata={"x_unit": "cm-1"},
        )
        window._update_axis_dependent_ui()

        assert window.gas_peaks_cb.isEnabled()
        assert window.conc_calculate_btn.isEnabled()
    finally:
        window.close()
