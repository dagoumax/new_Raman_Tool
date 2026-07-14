import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from raman_tool.qt_gui import RamanQtGUI


_APP = QApplication.instance() or QApplication([])


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
