"""Qt GUI (PySide6).

基于 PySide6 的桌面图形界面，支持拖拽导入、光谱显示、
信噪比计算、基线校正、气体浓度分析等功能。
"""

import sys
from pathlib import Path

import numpy as np

# 在导入任何 matplotlib 相关模块前设好后端
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QGroupBox, QLabel, QLineEdit,
    QPushButton, QTextEdit, QFileDialog, QMessageBox, QStatusBar,
    QMenu, QTabWidget, QComboBox, QCheckBox,
    QProgressBar, QDockWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QInputDialog,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent

from raman_tool.models import Spectrum
from raman_tool.history import SpectrumSession, SessionCache
from raman_tool.readers import detect_format
from raman_tool.sorting import natural_sorted
from raman_tool.exporters import export_spectrum, normalize_export_path
from raman_tool.processing import calculate_snr, calculate_concentration
from raman_tool.visualization import plot_spectrum, save_figure
from raman_tool.config import get_config, save_config
from raman_tool.presets import delete_workflow_preset, load_workflow_presets, save_workflow_preset, validate_workflow_preset
from raman_tool.safety import refresh_limits
from raman_tool.validation import integer_parameter, positive_float, calibration_from_points
from raman_tool.workflows import collect_spectrum_files
from raman_tool.gas_library import (
    default_gas_library_path, get_gas_library,
    get_quantitative_gas_choices, reload_gas_library,
    save_gas_library,
)


from raman_tool.qt_dialogs import GasLibraryDialog, SafetySettingsDialog, ImageViewerWindow, ConcentrationResultDialog
from raman_tool.qt_workers import SpectrumLoadThread, BaselineThread, BatchProcessThread, BatchExportThread, WorkerManager


class RamanQtGUI(QMainWindow):
    """拉曼光谱处理工具 Qt 主窗口."""

    def __init__(self):
        super().__init__()

        self.current_spectrum: Spectrum | None = None
        self.current_file: str = ""
        self.spectrum_sessions = SessionCache()
        self._workers = WorkerManager(self)
        self._workers.worker_finished.connect(self._on_worker_finished)
        self._workers.idle.connect(self._on_workers_idle)
        self._closing = False
        self._load_request = 0
        self._baseline_worker = None
        self._batch_worker = None
        self.last_batch_failures: list[dict[str, str]] = []
        self.current_session: SpectrumSession | None = None
        self.current_cache_key: str = ""
        self.current_figure: plt.Figure | None = None
        self.canvas: FigureCanvas | None = None
        self._toolbar_ref = None

        # SNR 选区
        self.snr_signal_range: tuple[float, float] | None = None
        self.snr_noise_range: tuple[float, float] | None = None
        self._snr_selecting: str | None = None
        self._show_gas_peaks: bool = True
        self._show_auto_peaks: bool = True
        self._detected_peaks: list[dict] = []

        # 十字光标
        self._cursor_x: float | None = None
        self._cursor_y: float | None = None
        self._cursor_lines: list = []
        self._cursor_h_line = None
        self._cursor_v_line = None
        self._snr_span = None
        self._snr_selecting = None

        self.setWindowTitle("拉曼光谱数据处理工具")
        self.resize(1400, 850)
        self.setMinimumSize(1000, 600)

        self._setup_menus()
        self._setup_central()
        self._setup_docks()
        self._setup_statusbar()

        self._apply_style()

    def _on_worker_finished(self, worker):
        if worker is self._baseline_worker:
            self._baseline_worker = None
            self.baseline_apply_btn.setEnabled(True)
        if worker is self._batch_worker:
            self._batch_worker = None
            self.batch_start_btn.setEnabled(True)
            self.batch_cancel_btn.setEnabled(False)
        self._update_history_ui()

    def _on_workers_idle(self):
        if self._closing:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        self._closing = True
        if self._workers.workers:
            self._load_request += 1
            self._workers.interrupt_all()
            self.setEnabled(False)
            self.status_bar.showMessage("正在等待当前计算安全结束...")
            event.ignore()
            return
        self._clear_canvas()
        for viewer in self.findChildren(ImageViewerWindow) + self.findChildren(ConcentrationResultDialog):
            viewer.close()
        self.spectrum_sessions.close()
        event.accept()

    def _invalidate_loads(self):
        self._load_request += 1
        for worker in tuple(self._workers.workers):
            if isinstance(worker, SpectrumLoadThread):
                worker.requestInterruption()

    def _on_cancel_batch(self):
        if self._batch_worker is not None:
            self._batch_worker.requestInterruption()
            self.batch_cancel_btn.setEnabled(False)
            self.batch_status.setText("正在取消，等待当前文件处理结束...")

    def _setup_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")

        open_action = QAction("打开文件(&O)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_files)
        file_menu.addAction(open_action)

        open_dir_action = QAction("打开目录(&D)...", self)
        open_dir_action.setShortcut("Ctrl+Shift+O")
        open_dir_action.triggered.connect(self._on_open_directory)
        file_menu.addAction(open_dir_action)

        file_menu.addSeparator()

        export_action = QAction("导出图表(&E)...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export_chart)
        file_menu.addAction(export_action)

        export_spectrum_action = QAction("导出当前光谱数据(&S)...", self)
        export_spectrum_action.setShortcut("Ctrl+Shift+E")
        export_spectrum_action.triggered.connect(self._on_export_spectrum)
        file_menu.addAction(export_spectrum_action)

        export_raw_action = QAction("导出原始光谱数据...", self)
        export_raw_action.triggered.connect(self._on_export_raw_spectrum)
        file_menu.addAction(export_raw_action)

        export_all_spectra_action = QAction("导出全部光谱数据(&A)...", self)
        export_all_spectra_action.triggered.connect(self._on_export_all_spectra)
        file_menu.addAction(export_all_spectra_action)

        file_menu.addSeparator()

        quit_action = QAction("退出(&Q)", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        process_menu = menubar.addMenu("处理(&P)")

        snr_action = QAction("信噪比计算(&S)...", self)
        snr_action.triggered.connect(self._on_calc_snr)
        process_menu.addAction(snr_action)

        baseline_action = QAction("基线校正(&B)...", self)
        baseline_action.triggered.connect(self._on_baseline)
        process_menu.addAction(baseline_action)

        self.undo_processing_action = QAction("撤销处理", self)
        self.undo_processing_action.setShortcut("Ctrl+Z")
        self.undo_processing_action.setEnabled(False)
        self.undo_processing_action.triggered.connect(self._on_undo_processing)
        process_menu.addAction(self.undo_processing_action)

        self.redo_processing_action = QAction("重做处理", self)
        self.redo_processing_action.setShortcut("Ctrl+Y")
        self.redo_processing_action.setEnabled(False)
        self.redo_processing_action.triggered.connect(self._on_redo_processing)
        process_menu.addAction(self.redo_processing_action)

        self.restore_raw_action = QAction("恢复原始光谱", self)
        self.restore_raw_action.setEnabled(False)
        self.restore_raw_action.triggered.connect(self._on_restore_raw)
        process_menu.addAction(self.restore_raw_action)

        conc_action = QAction("气体浓度分析(&C)...", self)
        conc_action.triggered.connect(self._on_concentration)
        process_menu.addAction(conc_action)

        process_menu.addSeparator()

        batch_action = QAction("批量处理(&T)...", self)
        batch_action.triggered.connect(self._on_batch)
        process_menu.addAction(batch_action)

        presets_menu = menubar.addMenu("预设(&R)")
        apply_preset_action = QAction("应用工作流预设...", self)
        apply_preset_action.triggered.connect(self._on_apply_workflow_preset)
        presets_menu.addAction(apply_preset_action)

        save_preset_action = QAction("保存当前为预设...", self)
        save_preset_action.triggered.connect(self._on_save_workflow_preset)
        presets_menu.addAction(save_preset_action)

        delete_preset_action = QAction("删除工作流预设...", self)
        delete_preset_action.triggered.connect(self._on_delete_workflow_preset)
        presets_menu.addAction(delete_preset_action)

        settings_menu = menubar.addMenu("设置(&S)")

        self.show_file_list_action = QAction("显示文件列表", self)
        self.show_file_list_action.setCheckable(True)
        self.show_file_list_action.setChecked(True)
        self.show_file_list_action.toggled.connect(
            lambda checked: self._set_dock_visible("file_list", checked)
        )
        settings_menu.addAction(self.show_file_list_action)

        self.show_control_panel_action = QAction("显示操作面板", self)
        self.show_control_panel_action.setCheckable(True)
        self.show_control_panel_action.setChecked(True)
        self.show_control_panel_action.toggled.connect(
            lambda checked: self._set_dock_visible("control_panel", checked)
        )
        settings_menu.addAction(self.show_control_panel_action)

        self.show_log_panel_action = QAction("显示输出日志", self)
        self.show_log_panel_action.setCheckable(True)
        self.show_log_panel_action.setChecked(True)
        self.show_log_panel_action.toggled.connect(
            lambda checked: self._set_dock_visible("log_panel", checked)
        )
        settings_menu.addAction(self.show_log_panel_action)

        settings_menu.addSeparator()
        gas_library_action = QAction("气体峰位库...", self)
        gas_library_action.triggered.connect(self._on_gas_library_settings)
        settings_menu.addAction(gas_library_action)

        safety_action = QAction("安全限制...", self)
        safety_action.triggered.connect(self._on_safety_settings)
        settings_menu.addAction(safety_action)

        help_menu = menubar.addMenu("帮助(&H)")
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _get_baseline_options(self) -> dict:
        if self.baseline_method.currentIndex() == 0:
            lam_text = self.baseline_lam.text().strip()
            lam = positive_float(lam_text if lam_text else 1e5, "lam")
            return {"method": "arPLS", "lam": lam}
        return {"method": "poly", "degree": int(self.baseline_degree.currentText())}

    def _format_baseline_options(self, options: dict) -> str:
        if options.get("method") == "poly":
            return f"poly, degree={int(options.get('degree', 3))}"
        return f"arPLS, lam={float(options.get('lam', 1e5)):.1e}"

    def _collect_workflow_preset(self) -> dict:
        method = "arPLS" if self.baseline_method.currentIndex() == 0 else "poly"
        strategy = "peak_area" if self.conc_strategy.currentIndex() == 1 else "peak_max"
        row_mode = "sum" if self.row_mode_combo.currentIndex() == 1 else "mean"
        gas_name = str(self.conc_gas.currentData() or "")
        lam = positive_float(self.baseline_lam.text().strip() or "100000", "lam")
        window = positive_float(self.conc_window.text().strip() or "10", "聚焦气体窗口")
        col_merge = integer_parameter(self.img_col_merge.text().strip() or "1", "列合并", minimum=1)

        calibration_values = [
            self.cal_px1.text().strip(),
            self.cal_rs1.text().strip(),
            self.cal_px2.text().strip(),
            self.cal_rs2.text().strip(),
        ]
        self._get_calibration()
        return {
            "baseline_method": method,
            "baseline_lam": lam,
            "baseline_degree": int(self.baseline_degree.currentText()),
            "auto_baseline": self.auto_baseline_cb.isChecked(),
            "batch_baseline": self.batch_baseline_cb.isChecked(),
            "concentration_gas": gas_name,
            "concentration_window": window,
            "concentration_strategy": strategy,
            "row_mode": row_mode,
            "col_merge": col_merge,
            "row_groups": self.img_row_groups.text().strip(),
            "show_individual_rows": self.img_show_rows_cb.isChecked(),
            "calibration_px1": calibration_values[0],
            "calibration_shift1": calibration_values[1],
            "calibration_px2": calibration_values[2],
            "calibration_shift2": calibration_values[3],
            "show_gas_peaks": self.gas_peaks_cb.isChecked(),
            "show_auto_peaks": self.auto_peaks_cb.isChecked(),
        }

    def _apply_workflow_preset(self, preset: dict):
        preset = validate_workflow_preset(preset)
        self._invalidate_loads()
        self.baseline_method.setCurrentIndex(1 if preset.get("baseline_method") == "poly" else 0)
        self.baseline_lam.setText(str(preset.get("baseline_lam", 100000.0)))
        degree = str(int(preset.get("baseline_degree", 3)))
        idx = self.baseline_degree.findText(degree)
        if idx < 0:
            self.baseline_degree.addItem(degree)
        self.baseline_degree.setCurrentText(degree)
        self.auto_baseline_cb.setChecked(bool(preset.get("auto_baseline", True)))
        self.batch_baseline_cb.setChecked(bool(preset.get("batch_baseline", True)))

        gas = str(preset.get("concentration_gas") or "").casefold()
        for i in range(self.conc_gas.count()):
            if str(self.conc_gas.itemData(i) or "").casefold() == gas:
                self.conc_gas.setCurrentIndex(i)
                break
        self.conc_window.setText(str(preset.get("concentration_window", 10.0)))
        self.conc_strategy.setCurrentIndex(1 if preset.get("concentration_strategy") == "peak_area" else 0)

        self.row_mode_combo.setCurrentIndex(1 if preset.get("row_mode") == "sum" else 0)
        self.img_col_merge.setText(str(preset["col_merge"]))
        self.img_row_groups.setText(str(preset.get("row_groups", "")))
        self.img_show_rows_cb.setChecked(bool(preset.get("show_individual_rows", False)))
        self.cal_px1.setText(str(preset.get("calibration_px1", "")))
        self.cal_rs1.setText(str(preset.get("calibration_shift1", "")))
        self.cal_px2.setText(str(preset.get("calibration_px2", "")))
        self.cal_rs2.setText(str(preset.get("calibration_shift2", "")))
        self.gas_peaks_cb.setChecked(bool(preset.get("show_gas_peaks", True)))
        self.auto_peaks_cb.setChecked(bool(preset.get("show_auto_peaks", True)))
        # Cache keys already separate import/calibration/baseline settings.
        # Applying a preset must not discard edits to other loaded files.
        self.current_session = None
        self.current_cache_key = ""
        if self.current_file:
            self._load_file(self.current_file)
        elif self.current_spectrum is not None:
            if self._show_auto_peaks:
                self._detect_peaks()
            else:
                self._detected_peaks = []
                self.peak_table.setRowCount(0)
            self._update_plot()

    def _on_apply_workflow_preset(self):
        try:
            presets = load_workflow_presets()
        except Exception as e:
            QMessageBox.critical(self, "预设读取失败", str(e))
            return
        names = list(presets.keys())
        if not names:
            QMessageBox.information(self, "预设", "没有可用的工作流预设")
            return
        name, ok = QInputDialog.getItem(self, "应用工作流预设", "选择预设:", names, 0, False)
        if not ok or not name:
            return
        self._apply_workflow_preset(presets[name])
        self._log(f"已应用工作流预设: {name}")
        self.status_bar.showMessage(f"已应用工作流预设: {name}")

    def _on_save_workflow_preset(self):
        name, ok = QInputDialog.getText(self, "保存工作流预设", "预设名称:")
        if not ok or not name.strip():
            return
        try:
            path = save_workflow_preset(name, self._collect_workflow_preset())
        except Exception as e:
            QMessageBox.critical(self, "预设保存失败", str(e))
            return
        self._log(f"工作流预设已保存: {name.strip()} -> {path}")
        QMessageBox.information(self, "预设已保存", f"工作流预设已保存。\n{path}")

    def _on_delete_workflow_preset(self):
        try:
            presets = load_workflow_presets()
        except Exception as e:
            QMessageBox.critical(self, "预设读取失败", str(e))
            return
        names = list(presets.keys())
        if not names:
            QMessageBox.information(self, "预设", "没有可删除的工作流预设")
            return
        name, ok = QInputDialog.getItem(self, "删除工作流预设", "选择预设:", names, 0, False)
        if not ok or not name:
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除预设吗？\n{name}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            path = delete_workflow_preset(name)
        except Exception as e:
            QMessageBox.critical(self, "预设删除失败", str(e))
            return
        self._log(f"工作流预设已删除: {name} -> {path}")
        QMessageBox.information(self, "预设已删除", f"工作流预设已删除。\n{path}")

    def _on_gas_library_settings(self):
        dialog = GasLibraryDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            path = save_gas_library(dialog.to_library())
            reload_gas_library()
        except Exception as e:
            QMessageBox.critical(self, "气体峰位库保存失败", str(e))
            return
        self._refresh_concentration_gases()
        self._update_axis_dependent_ui()
        if self.current_spectrum is not None:
            if self._show_auto_peaks:
                self._detect_peaks()
            self._update_plot()
        self._log(f"气体峰位库已保存: {path}")
        QMessageBox.information(self, "峰位库已保存", f"气体峰位库已更新。\n{path}")

    def _refresh_concentration_gases(self):
        current = self.conc_gas.currentData() if hasattr(self, "conc_gas") else None
        self.conc_gas.clear()
        choices = get_quantitative_gas_choices()
        for key, label in choices:
            self.conc_gas.addItem(label, key)
        if not choices:
            self.conc_gas.addItem("气体库中没有启用定量的气体", None)
        for i, (key, _label) in enumerate(choices):
            if key.casefold() == str(current or "").casefold():
                self.conc_gas.setCurrentIndex(i)
                break
        if hasattr(self, "conc_window"):
            self._on_concentration_gas_changed()

    def _on_concentration_gas_changed(self, _index: int | None = None):
        if not hasattr(self, "conc_window"):
            return
        gas_key = self.conc_gas.currentData()
        gas_info = get_gas_library().get(str(gas_key)) if gas_key else None
        if gas_info:
            self.conc_window.setText(f"{float(gas_info['half_width']):g}")

    def _on_safety_settings(self):
        dialog = SafetySettingsDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        path = save_config(dialog.to_config())
        refresh_limits()
        self._log(f"安全限制设置已保存: {path}")
        QMessageBox.information(self, "设置已保存", f"安全限制已更新。\n{path}")

    def _setup_central(self):
        """设置中央绘图区域."""
        central = QWidget()
        self.setCentralWidget(central)
        self.central_layout = QVBoxLayout(central)
        self.central_layout.setContentsMargins(0, 0, 0, 0)

        placeholder = QLabel("请拖拽文件到此处或使用「文件 → 打开文件」载入光谱数据")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self.central_layout.addWidget(placeholder)
        self._placeholder = placeholder

    def _set_dock_visible(self, dock_name: str, visible: bool):
        dock_map = {
            "file_list": getattr(self, "file_list_dock", None),
            "control_panel": getattr(self, "control_panel_dock", None),
            "log_panel": getattr(self, "log_panel_dock", None),
        }
        dock = dock_map.get(dock_name)
        if dock is None:
            return
        dock.setVisible(visible)
        self._refit_central_plot()

    def _on_dock_visibility_changed(self, dock_name: str, visible: bool):
        action_map = {
            "file_list": getattr(self, "show_file_list_action", None),
            "control_panel": getattr(self, "show_control_panel_action", None),
            "log_panel": getattr(self, "show_log_panel_action", None),
        }
        action = action_map.get(dock_name)
        if action is not None and action.isChecked() != visible:
            action.blockSignals(True)
            action.setChecked(visible)
            action.blockSignals(False)
        self._refit_central_plot()

    def _refit_central_plot(self):
        if self.current_figure is None or self.canvas is None:
            return
        try:
            self.current_figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        except Exception:
            pass
        self.current_figure.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.92)
        self.canvas.draw_idle()

    def _setup_docks(self):
        """设置停靠面板."""
        # 左侧: 文件列表
        left_dock = QDockWidget("文件列表", self)
        left_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self.file_list_dock = left_dock
        left_dock.setMinimumWidth(220)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.itemDoubleClicked.connect(self._on_file_double_click)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._on_file_context_menu)
        left_layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        btn_add = QPushButton("添加")
        btn_add.clicked.connect(self._on_open_files)
        btn_remove = QPushButton("移除")
        btn_remove.clicked.connect(self._on_remove_file)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self._on_clear_files)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)
        btn_layout.addWidget(btn_clear)
        left_layout.addLayout(btn_layout)

        left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)
        left_dock.visibilityChanged.connect(
            lambda visible: self._on_dock_visibility_changed("file_list", visible)
        )

        # 右侧: 控制面板
        right_dock = QDockWidget("操作面板", self)
        self.control_panel_dock = right_dock
        right_dock.setMinimumWidth(260)
        self.control_tabs = QTabWidget()
        self._setup_snr_tab()
        self._setup_baseline_tab()
        self._setup_concentration_tab()
        self._setup_batch_tab()
        self._setup_image_tab()
        right_dock.setWidget(self.control_tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)
        right_dock.visibilityChanged.connect(
            lambda visible: self._on_dock_visibility_changed("control_panel", visible)
        )

        # 底部: 日志
        bottom_dock = QDockWidget("输出日志", self)
        self.log_panel_dock = bottom_dock
        bottom_dock.setMinimumHeight(100)
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(120)
        bottom_dock.setWidget(self.log_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, bottom_dock)
        bottom_dock.visibilityChanged.connect(
            lambda visible: self._on_dock_visibility_changed("log_panel", visible)
        )

    def _setup_snr_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 选区状态
        self.snr_signal_range: tuple[float, float] | None = None
        self.snr_noise_range: tuple[float, float] | None = None

        g1 = QGroupBox("信号区域")
        g1l = QVBoxLayout(g1)
        hint1 = QLabel("点击下方按钮后在图上框选")
        hint1.setStyleSheet("color: #666; font-size: 11px;")
        g1l.addWidget(hint1)

        self.snr_signal_label = QLabel("未选择")
        self.snr_signal_label.setStyleSheet("color: #27ae60;")
        g1l.addWidget(self.snr_signal_label)

        btn_signal = QPushButton("📐 框选信号区域")
        btn_signal.setStyleSheet("background: #27ae60;")
        btn_signal.clicked.connect(self._on_select_signal)
        g1l.addWidget(btn_signal)
        layout.addWidget(g1)

        g2 = QGroupBox("噪声区域")
        g2l = QVBoxLayout(g2)
        hint2 = QLabel("点击下方按钮后在图上框选")
        hint2.setStyleSheet("color: #666; font-size: 11px;")
        g2l.addWidget(hint2)

        self.snr_noise_label = QLabel("未选择")
        self.snr_noise_label.setStyleSheet("color: #e74c3c;")
        g2l.addWidget(self.snr_noise_label)

        btn_noise = QPushButton("📐 框选噪声区域")
        btn_noise.setStyleSheet("background: #e74c3c; color: white;")
        btn_noise.clicked.connect(self._on_select_noise)
        g2l.addWidget(btn_noise)
        layout.addWidget(g2)

        btn_calc = QPushButton("计算信噪比")
        btn_calc.clicked.connect(self._on_calc_snr)
        layout.addWidget(btn_calc)

        btn_clear = QPushButton("清除选区")
        btn_clear.clicked.connect(self._on_clear_snr_selection)
        layout.addWidget(btn_clear)

        self.snr_result = QLabel("")
        self.snr_result.setWordWrap(True)
        layout.addWidget(self.snr_result)

        layout.addStretch()
        self.control_tabs.addTab(tab, "信噪比")

    def _on_select_signal(self):
        """激活信号区域框选."""
        if self.canvas is None:
            QMessageBox.warning(self, "提示", "请先载入光谱")
            return
        self._snr_selecting = "signal"
        self.status_bar.showMessage("请在图上拖拽框选信号区域 (峰值所在范围)...")

    def _on_select_noise(self):
        """激活噪声区域框选."""
        if self.canvas is None:
            QMessageBox.warning(self, "提示", "请先载入光谱")
            return
        self._snr_selecting = "noise"
        self.status_bar.showMessage("请在图上拖拽框选噪声区域 (平坦基线范围)...")

    def _on_clear_snr_selection(self):
        self.snr_signal_range = None
        self.snr_noise_range = None
        self.snr_signal_label.setText("未选择")
        self.snr_noise_label.setText("未选择")
        self._update_plot()
        self.status_bar.showMessage("SNR 选区已清除")

    def _setup_baseline_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g1 = QGroupBox("基线校正参数")
        g1l = QVBoxLayout(g1)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("方法:"))
        self.baseline_method = QComboBox()
        self.baseline_method.addItems(["arPLS (自适应)", "多项式拟合 (poly)"])
        self.baseline_method.currentIndexChanged.connect(self._on_baseline_method_changed)
        r1.addWidget(self.baseline_method)
        g1l.addLayout(r1)

        # arPLS lambda 参数
        self._lam_layout = QHBoxLayout()
        self._lam_layout.addWidget(QLabel("平滑参数 lam:"))
        self.baseline_lam = QLineEdit("100000")
        self._lam_layout.addWidget(self.baseline_lam)
        g1l.addLayout(self._lam_layout)

        # poly 阶数 (默认隐藏)
        self._degree_layout = QHBoxLayout()
        self._degree_layout.addWidget(QLabel("阶数:"))
        self.baseline_degree = QComboBox()
        self.baseline_degree.addItems(["0", "1", "2", "3", "4", "5"])
        self.baseline_degree.setCurrentText("3")
        self._degree_layout.addWidget(self.baseline_degree)
        g1l.addLayout(self._degree_layout)

        # 初始状态: arPLS 因此隐藏阶数
        self._hide_layout(self._degree_layout)

        # 自动基线校正勾选框
        self.auto_baseline_cb = QCheckBox("载入光谱后自动执行基线校正")
        self.auto_baseline_cb.setChecked(True)
        g1l.addWidget(self.auto_baseline_cb)

        layout.addWidget(g1)

        self.baseline_apply_btn = QPushButton("执行基线校正")
        self.baseline_apply_btn.clicked.connect(self._on_baseline)
        layout.addWidget(self.baseline_apply_btn)

        history_actions = QHBoxLayout()
        self.baseline_undo_btn = QPushButton("撤销")
        self.baseline_undo_btn.clicked.connect(self._on_undo_processing)
        history_actions.addWidget(self.baseline_undo_btn)
        self.baseline_redo_btn = QPushButton("重做")
        self.baseline_redo_btn.clicked.connect(self._on_redo_processing)
        history_actions.addWidget(self.baseline_redo_btn)
        self.baseline_restore_btn = QPushButton("恢复原始")
        self.baseline_restore_btn.clicked.connect(self._on_restore_raw)
        history_actions.addWidget(self.baseline_restore_btn)
        layout.addLayout(history_actions)

        history_group = QGroupBox("处理历史")
        history_layout = QVBoxLayout(history_group)
        self.baseline_history_list = QListWidget()
        self.baseline_history_list.setMaximumHeight(190)
        self.baseline_history_list.setAlternatingRowColors(True)
        history_layout.addWidget(self.baseline_history_list)
        layout.addWidget(history_group)
        self._update_history_ui()
        layout.addStretch()

        self.control_tabs.addTab(tab, "基线校正")

    def _hide_layout(self, layout):
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w:
                w.setVisible(False)

    def _show_layout(self, layout):
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w:
                w.setVisible(True)

    def _on_baseline_method_changed(self, idx):
        if idx == 0:  # arPLS
            self._hide_layout(self._degree_layout)
            self._show_layout(self._lam_layout)
        else:  # poly
            self._hide_layout(self._lam_layout)
            self._show_layout(self._degree_layout)

    def _ensure_current_session(self) -> SpectrumSession | None:
        if self.current_session is None and self.current_spectrum is not None:
            self.current_session = SpectrumSession(self.current_spectrum)
            if self.current_cache_key:
                self.spectrum_sessions[self.current_cache_key] = self.current_session
        return self.current_session

    def _refresh_from_current_session(self) -> None:
        self._clear_analysis_results()
        if self.current_session is None:
            self._update_history_ui()
            return
        self.current_spectrum = self.current_session.current
        if self._show_auto_peaks:
            self._detect_peaks()
        else:
            self._detected_peaks = []
            self.peak_table.setRowCount(0)
        self._update_plot()
        self._update_history_ui()
        self._update_axis_dependent_ui()

    def _clear_analysis_results(self):
        self.snr_result.clear()
        self.conc_result.clear()

    def _format_history_record(self, record: dict) -> str:
        timestamp = str(record.get("timestamp", ""))
        clock = timestamp.split("T", 1)[-1][:8] if "T" in timestamp else timestamp
        params = record.get("parameters", {})
        details = ", ".join(f"{key}={value}" for key, value in params.items())
        source = record.get("source", "")
        suffix = " | ".join(part for part in (source, details) if part)
        text = f"{record.get('sequence', '')}. {clock} {record.get('action', '')}"
        return f"{text}\n{suffix}" if suffix else text

    def _update_history_ui(self) -> None:
        session = self.current_session
        busy = self._baseline_worker is not None
        can_undo = bool(session and session.can_undo and not busy)
        can_redo = bool(session and session.can_redo and not busy)
        can_restore = bool(session and not session.is_raw_current and not busy)

        if hasattr(self, "undo_processing_action"):
            self.undo_processing_action.setEnabled(can_undo)
            self.redo_processing_action.setEnabled(can_redo)
            self.restore_raw_action.setEnabled(can_restore)
        if hasattr(self, "baseline_undo_btn"):
            self.baseline_undo_btn.setEnabled(can_undo)
            self.baseline_redo_btn.setEnabled(can_redo)
            self.baseline_restore_btn.setEnabled(can_restore)
        if hasattr(self, "baseline_history_list"):
            self.baseline_history_list.clear()
            if self.spectrum_sessions.capacity_warning:
                self.baseline_history_list.addItem(self.spectrum_sessions.capacity_warning)
            if session is not None:
                if session.discarded_state_count:
                    self.baseline_history_list.addItem(
                        f"较早的 {session.discarded_state_count} 个撤销快照已释放；原始数据与完整操作记录保留"
                    )
                for record in session.recent_audit_log:
                    self.baseline_history_list.addItem(self._format_history_record(record))
                if self.baseline_history_list.count():
                    self.baseline_history_list.scrollToBottom()

    def _on_undo_processing(self) -> None:
        if self._baseline_worker is not None:
            return
        session = self._ensure_current_session()
        if session is None or not session.can_undo:
            return
        session.undo()
        self._refresh_from_current_session()
        self._log("已撤销上一步光谱处理")
        self.status_bar.showMessage("已撤销上一步光谱处理")

    def _on_redo_processing(self) -> None:
        if self._baseline_worker is not None:
            return
        session = self._ensure_current_session()
        if session is None or not session.can_redo:
            return
        session.redo()
        self._refresh_from_current_session()
        self._log("已重做光谱处理")
        self.status_bar.showMessage("已重做光谱处理")

    def _on_restore_raw(self) -> None:
        if self._baseline_worker is not None:
            return
        session = self._ensure_current_session()
        if session is None or session.is_raw_current:
            return
        session.restore_raw()
        self._refresh_from_current_session()
        self._log("已恢复原始光谱；此前处理历史仍然保留")
        self.status_bar.showMessage("已恢复原始光谱，可撤销恢复操作")

    def _setup_concentration_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g1 = QGroupBox("加权信号占比")
        g1l = QVBoxLayout(g1)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("结果聚焦气体:"))
        self.conc_gas = QComboBox()
        self._refresh_concentration_gases()
        r1.addWidget(self.conc_gas)
        g1l.addLayout(r1)

        layout.addWidget(g1)

        g2 = QGroupBox("峰提取与计算")
        g2l = QVBoxLayout(g2)

        r4 = QHBoxLayout()
        self.conc_window_label = QLabel("聚焦气体半窗口 (cm⁻¹):")
        r4.addWidget(self.conc_window_label)
        self.conc_window = QLineEdit("10")
        r4.addWidget(self.conc_window)
        g2l.addLayout(r4)
        self.conc_gas.currentIndexChanged.connect(self._on_concentration_gas_changed)
        self._on_concentration_gas_changed()

        r5 = QHBoxLayout()
        r5.addWidget(QLabel("归一化方法:"))
        self.conc_strategy = QComboBox()
        self.conc_strategy.addItems(["峰高归一化 (peak_max)", "峰面积归一化 (peak_area)"])
        r5.addWidget(self.conc_strategy)
        g2l.addLayout(r5)

        layout.addWidget(g2)

        self.conc_calculate_btn = QPushButton("计算气体库定量结果")
        self.conc_calculate_btn.clicked.connect(self._on_concentration)
        layout.addWidget(self.conc_calculate_btn)

        self.conc_result = QLabel("")
        self.conc_result.setWordWrap(True)
        layout.addWidget(self.conc_result)

        note = QLabel("结果表示启用定量气体的加权信号占比。实际组分浓度需经标准样品校准验证；默认系数 1 不代表已校准。")
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addStretch()
        self.control_tabs.addTab(tab, "气体信号占比")

    def _setup_batch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g1 = QGroupBox("批量处理")
        g1l = QVBoxLayout(g1)

        self.batch_baseline_cb = QCheckBox("执行基线校正")
        g1l.addWidget(self.batch_baseline_cb)

        self.batch_start_btn = QPushButton("开始批量处理")
        self.batch_start_btn.clicked.connect(self._on_batch)
        g1l.addWidget(self.batch_start_btn)
        self.batch_cancel_btn = QPushButton("取消批量任务")
        self.batch_cancel_btn.setEnabled(False)
        self.batch_cancel_btn.clicked.connect(self._on_cancel_batch)
        g1l.addWidget(self.batch_cancel_btn)

        layout.addWidget(g1)

        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        layout.addWidget(self.batch_progress)

        self.batch_status = QLabel("")
        self.batch_status.setWordWrap(True)
        layout.addWidget(self.batch_status)

        layout.addStretch()
        self.control_tabs.addTab(tab, "批量处理")

    def _setup_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g1 = QGroupBox("行分组")
        g1l = QVBoxLayout(g1)
        hint = QLabel("指定行范围 (1-based)")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        g1l.addWidget(hint)
        g1l.addWidget(QLabel("支持逗号/空格/换行分隔，示例: 1-40, 91-130"))
        self.img_row_groups = QLineEdit()
        self.img_row_groups.setPlaceholderText("留空 = 全部行取平均")
        g1l.addWidget(self.img_row_groups)
        self.img_show_rows_cb = QCheckBox("显示各行数据 (叠加到图表)")
        self.img_show_rows_cb.setChecked(False)
        self.img_show_rows_cb.toggled.connect(self._update_plot)
        g1l.addWidget(self.img_show_rows_cb)
        layout.addWidget(g1)

        g2 = QGroupBox("列合并")
        g2l = QVBoxLayout(g2)
        r = QHBoxLayout()
        r.addWidget(QLabel("合并因子 n:"))
        self.img_col_merge = QLineEdit("1")
        self.img_col_merge.setMaximumWidth(80)
        r.addWidget(self.img_col_merge)
        r.addWidget(QLabel("(n=1 不变, n=2 每两列合并取平均)"))
        r.addStretch()
        g2l.addLayout(r)
        layout.addWidget(g2)

        # 行处理模式
        g_row = QGroupBox("行处理方式")
        g_rowl = QVBoxLayout(g_row)
        self.row_mode_combo = QComboBox()
        self.row_mode_combo.addItems(["平均 (mean)", "求和 (sum)"])
        g_rowl.addWidget(self.row_mode_combo)
        layout.addWidget(g_row)

        btn_apply_image_settings = QPushButton("应用设置并重绘光谱")
        btn_apply_image_settings.clicked.connect(self._apply_image_settings_to_current_file)
        layout.addWidget(btn_apply_image_settings)

        g3 = QGroupBox("拉曼位移校准 (像素 -> cm⁻¹)")
        g3l = QVBoxLayout(g3)
        g3l.addWidget(QLabel("将 TIF 像素列索引校准为拉曼位移。"))
        g3l.addWidget(QLabel("输入已知气体峰的位置 (如 N₂ 对应 2331 cm⁻¹)"))

        cr1 = QHBoxLayout()
        cr1.addWidget(QLabel("峰1 像素:"))
        self.cal_px1 = QLineEdit()
        self.cal_px1.setPlaceholderText="如 500"
        cr1.addWidget(self.cal_px1)
        cr1.addWidget(QLabel("cm⁻¹:"))
        self.cal_rs1 = QLineEdit()
        self.cal_rs1.setPlaceholderText("如 1555 (O2)")
        cr1.addWidget(self.cal_rs1)
        g3l.addLayout(cr1)

        cr2 = QHBoxLayout()
        cr2.addWidget(QLabel("峰2 像素:"))
        self.cal_px2 = QLineEdit()
        self.cal_px2.setPlaceholderText("如 800")
        cr2.addWidget(self.cal_px2)
        cr2.addWidget(QLabel("cm⁻¹:"))
        self.cal_rs2 = QLineEdit()
        self.cal_rs2.setPlaceholderText("如 2331 (N2)")
        cr2.addWidget(self.cal_rs2)
        g3l.addLayout(cr2)

        self.cal_status = QLabel("")
        self.cal_status.setStyleSheet("color: #666; font-size: 11px;")
        g3l.addWidget(self.cal_status)

        # 气体峰位显示开关
        self.gas_peaks_cb = QCheckBox("显示气体库参考峰位")
        self.gas_peaks_cb.setChecked(True)
        self.gas_peaks_cb.toggled.connect(self._on_gas_peaks_toggled)
        g3l.addWidget(self.gas_peaks_cb)

        # 自动寻峰开关
        self.auto_peaks_cb = QCheckBox("自动寻峰 + 显示峰位数值")
        self.auto_peaks_cb.setChecked(True)
        self.auto_peaks_cb.toggled.connect(self._on_auto_peaks_toggled)
        g3l.addWidget(self.auto_peaks_cb)

        layout.addWidget(g3)

        # 寻峰结果表
        g4 = QGroupBox("寻峰结果")
        g4l = QVBoxLayout(g4)
        self.peak_table = QTableWidget(0, 4)
        self.peak_table.setHorizontalHeaderLabels(["气体", "位置", "高度", "半峰宽"])
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.peak_table.setMaximumHeight(160)
        self.peak_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.peak_table.setAlternatingRowColors(True)
        g4l.addWidget(self.peak_table)
        layout.addWidget(g4)

        layout.addStretch()
        self.control_tabs.addTab(tab, "图像设置")

    def _apply_image_settings_to_current_file(self):
        if not self.current_file:
            QMessageBox.warning(self, "提示", "请先载入一个 TIF/BMP/JPG 图像文件")
            return

        suffix = Path(self.current_file).suffix.lower()
        if suffix not in (".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
            QMessageBox.warning(self, "提示", "行分组设置仅适用于图像文件")
            return

        self._load_file(self.current_file)

    def _get_calibration(self) -> tuple[float, float] | None:
        """从校准输入计算线性校准参数 (a, b) 使得 raman = a * pixel + b."""
        fields = [self.cal_px1.text().strip(), self.cal_rs1.text().strip(),
                  self.cal_px2.text().strip(), self.cal_rs2.text().strip()]
        if not any(fields):
            return None
        if not all(fields):
            raise ValueError("请填写完整的两点校准参数，或清空全部校准字段")
        return calibration_from_points(*fields)

    def _on_gas_peaks_toggled(self, checked):
        self._show_gas_peaks = checked
        self._update_plot()

    def _on_auto_peaks_toggled(self, checked):
        self._show_auto_peaks = checked
        if checked and self.current_spectrum:
            self._detect_peaks()
        self._update_plot()

    def _update_axis_dependent_ui(self):
        spectrum = self.current_spectrum
        has_raman_axis = bool(spectrum and spectrum.is_raman_shift)
        has_quantitative_gas = bool(self.conc_gas.currentData()) if hasattr(self, "conc_gas") else False
        if hasattr(self, "gas_peaks_cb"):
            self.gas_peaks_cb.setEnabled(has_raman_axis)
            self.gas_peaks_cb.setToolTip(
                "" if has_raman_axis else "气体库峰位使用 cm⁻¹；请先完成拉曼位移校准"
            )
        if hasattr(self, "conc_calculate_btn"):
            self.conc_calculate_btn.setEnabled(has_raman_axis and has_quantitative_gas)
            self.conc_gas.setEnabled(has_raman_axis)
            self.conc_window.setEnabled(has_raman_axis)
            self.conc_strategy.setEnabled(has_raman_axis)

    def _detect_peaks(self):
        if self.current_spectrum is None:
            return
        try:
            from raman_tool.processing import find_peaks_auto
            self._detected_peaks = find_peaks_auto(self.current_spectrum)
            self._update_peak_table()
        except Exception:
            self._detected_peaks = []
            self.peak_table.setRowCount(0)

    def _update_peak_table(self):
        self.peak_table.setRowCount(0)
        for p in self._detected_peaks:
            row = self.peak_table.rowCount()
            self.peak_table.insertRow(row)
            gas = p.get("matched_gas") or "—"
            unit = self.current_spectrum.x_unit_label if self.current_spectrum else ""
            self.peak_table.setItem(row, 0, QTableWidgetItem(gas))
            self.peak_table.setItem(row, 1, QTableWidgetItem(f"{p['center']:.1f} {unit}"))
            self.peak_table.setItem(row, 2, QTableWidgetItem(f"{p['height']:.0f}"))
            self.peak_table.setItem(row, 3, QTableWidgetItem(f"{p['fwhm']:.1f}"))

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QDockWidget { font-weight: bold; }
            QGroupBox { font-weight: bold; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; padding: 0 4px; }
            QPushButton { padding: 5px 12px; background: #3498db; color: white;
                           border: none; border-radius: 3px; }
            QPushButton:hover { background: #2980b9; }
            QPushButton:pressed { background: #1c6ea4; }
            QListWidget::item:alternate { background: #f0f0f0; }
            QListWidget::item:selected { background: #3498db; color: white; }
        """)

    def _log(self, text: str):
        self.log_widget.append(text)

    def _on_open_files(self):
        patterns = "光谱文件 (*.txt *.asc *.sif *.tif *.tiff *.bmp *.jpg *.jpeg);;所有文件 (*.*)"
        files, _ = QFileDialog.getOpenFileNames(self, "选择光谱文件", "", patterns)
        if files:
            self._add_files(files)

    def _on_open_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择包含光谱文件的目录")
        if dir_path:
            all_files = collect_spectrum_files(dir_path)
            if all_files:
                self._add_files([str(f) for f in all_files])
            else:
                QMessageBox.information(self, "提示", "所选目录中未找到支持格式的文件")

    def _add_files(self, files: list[str]):
        added = 0
        for f in files:
            try:
                detect_format(f)
            except ValueError:
                continue
            existing = [self.file_list.item(i).data(Qt.UserRole)
                       for i in range(self.file_list.count())]
            if f in existing:
                continue
            item = QListWidgetItem(Path(f).name)
            item.setData(Qt.UserRole, f)
            item.setToolTip(f)
            self.file_list.addItem(item)
            added += 1

        if added > 0:
            self.status_bar.showMessage(f"已添加 {added} 个文件")
            self._sort_file_list()
            if self.file_list.count() == 1:
                self.file_list.setCurrentRow(0)
                self._load_selected_file()

    def _sort_file_list(self):
        files = [
            self.file_list.item(i).data(Qt.UserRole)
            for i in range(self.file_list.count())
        ]
        self.file_list.clear()
        for f in natural_sorted(files):
            item = QListWidgetItem(Path(f).name)
            item.setData(Qt.UserRole, f)
            item.setToolTip(f)
            self.file_list.addItem(item)

    def _on_remove_file(self):
        self._invalidate_loads()
        for item in self.file_list.selectedItems():
            filepath = str(item.data(Qt.UserRole) or "")
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            for cache_key in [key for key in self.spectrum_sessions if key.startswith(f"{filepath}|")]:
                self.spectrum_sessions.pop(cache_key, None)

    def _on_clear_files(self):
        self._invalidate_loads()
        if self._baseline_worker is not None:
            self._baseline_worker.requestInterruption()
        self.file_list.clear()
        self.spectrum_sessions.clear()
        self.current_session = None
        self.current_cache_key = ""
        self.current_file = ""
        self.current_spectrum = None
        self._clear_analysis_results()
        self.cal_status.clear()
        self._update_history_ui()
        self._update_axis_dependent_ui()
        self._clear_canvas()

    def _on_file_double_click(self, item):
        self._load_selected_file()

    def _on_file_context_menu(self, pos):
        """文件列表右键菜单 — 图像格式显示灰度图."""
        item = self.file_list.itemAt(pos)
        if not item:
            return
        filepath = str(item.data(Qt.UserRole) or "")
        if not filepath:
            return
        suffix = Path(filepath).suffix.lower()
        if suffix not in (".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
            return
        menu = QMenu(self)
        img_action = QAction("显示 2D 灰度图", self)
        img_action.triggered.connect(lambda checked, fp=filepath: self._open_image_viewer(fp))
        menu.addAction(img_action)
        menu.exec(self.file_list.mapToGlobal(pos))

    def _open_image_viewer(self, filepath: str):
        try:
            from raman_tool.readers.image_reader import load_image_array, reduce_image_array

            arr, _ = load_image_array(filepath, grayscale=Path(filepath).suffix.lower() in {".jpg", ".jpeg"})
            row_groups = self.img_row_groups.text().strip()
            col_merge = integer_parameter(self.img_col_merge.text(), "列合并", minimum=1)
            arr, _ = reduce_image_array(arr, row_groups=row_groups or None, col_merge=col_merge)

            viewer = ImageViewerWindow(filepath, arr, self)
            viewer.setAttribute(Qt.WA_DeleteOnClose)
            if row_groups or col_merge > 1:
                viewer.setWindowTitle(
                    f"图像 - {Path(filepath).name} | 行: {row_groups or '全部'} | 列合并: {col_merge}"
                )
            viewer.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开图像: {e}")

    def _load_selected_file(self):
        item = self.file_list.currentItem()
        if not item:
            return
        filepath = item.data(Qt.UserRole)
        self._load_file(filepath)

    def _load_file(self, filepath: str, row_groups: str | None = None, col_merge: int = 1):
        if self._closing or self._capacity_blocked():
            return
        self._invalidate_loads()
        suffix = Path(filepath).suffix.lower()
        try:
            calibration = self._get_calibration()
            if suffix in (".tif", ".tiff", ".bmp", ".jpg", ".jpeg") and col_merge == 1:
                col_merge = integer_parameter(self.img_col_merge.text(), "列合并", minimum=1)
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        row_mode = "sum" if self.row_mode_combo.currentIndex() == 1 else "mean"

        # TIF/BMP/JPG: 从「图像设置」面板读取行/列参数
        if suffix in (".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
            if row_groups is None:
                rg_text = self.img_row_groups.text().strip()
                row_groups = rg_text if rg_text else None

        self.status_bar.showMessage(f"正在载入 {Path(filepath).name}...")
        self._log(f"载入: {Path(filepath).name}")

        try:
            baseline_options = self._get_baseline_options() if self.auto_baseline_cb.isChecked() else None
        except ValueError as exc:
            QMessageBox.critical(self, "基线参数错误", str(exc))
            return
        baseline_key = tuple(sorted(baseline_options.items())) if baseline_options else None
        cache_key = (
            f"{filepath}|rows={row_groups}|cols={col_merge}|mode={row_mode}|"
            f"cal={calibration}|baseline={baseline_key}"
        )
        if cache_key in self.spectrum_sessions:
            self.current_session = self.spectrum_sessions[cache_key]
            self.current_cache_key = cache_key
            self.current_file = filepath
            self._refresh_from_current_session()
            self.status_bar.showMessage(f"已应用图像设置: {Path(filepath).name}")
            return

        self.load_thread = SpectrumLoadThread(
            filepath,
            row_groups=row_groups,
            col_merge=col_merge,
            calibration=calibration,
            row_mode=row_mode,
            cache_key=cache_key,
            baseline_options=baseline_options,
            request_id=self._load_request,
        )
        self.load_thread.finished_loading.connect(self._on_spectrum_loaded)
        self.load_thread.error_occurred.connect(self._on_load_error)
        self._workers.start(self.load_thread)

    def _on_spectrum_loaded(self, session: SpectrumSession, filepath: str, cache_key: str):
        worker = self.sender()
        if self._closing or worker.request_id != self._load_request:
            return
        if worker.baseline_options is not None:
            self._log(f"  已自动执行基线校正 ({self._format_baseline_options(worker.baseline_options)})")
        self._clear_analysis_results()
        self.spectrum_sessions[cache_key] = session
        self.current_session = session
        self.current_cache_key = cache_key
        spectrum = session.current
        self.current_spectrum = spectrum
        self.current_file = filepath

        # 自动寻峰
        if self._show_auto_peaks:
            self._detect_peaks()

        self._update_plot()
        self._update_history_ui()
        self._update_axis_dependent_ui()
        n = spectrum.size
        xr = f"{spectrum.raman_shift[0]:.1f} - {spectrum.raman_shift[-1]:.1f}"
        yr = f"{spectrum.intensity.min():.1f} - {spectrum.intensity.max():.1f}"

        unit = spectrum.x_unit_label
        self.status_bar.showMessage(
            f"已载入: {Path(filepath).name} | 数据点: {n} | 范围: {xr} {unit} | 强度: {yr}"
        )
        self._log(f"  数据点: {n}, 横轴范围: {xr} {unit}, 强度: {yr}")
        cal = spectrum.metadata.get("calibration")
        if cal:
            a, b = cal
            self.cal_status.setText(f"当前校准: raman = {a:.4f} * pixel + {b:.2f}")
        else:
            self.cal_status.setText("" if spectrum.is_raman_shift else "当前横轴为像素，尚未校准")

    def _on_load_error(self, error: str):
        if self._closing or self.sender().request_id != self._load_request:
            return
        self.status_bar.showMessage("载入失败")
        self._log(f"[错误] 载入失败: {error}")

    def _update_plot(self):
        if self.current_spectrum is None:
            return

        self._clear_canvas()
        # 重置光标线段引用，避免残留旧图表的已关闭线段对象
        # 否则 _draw_cursor 会操作旧线段而非在新坐标轴上创建新线段
        self._cursor_h_line = None
        self._cursor_v_line = None
        self._cursor_lines = []
        self._cursor_x = None
        self._cursor_y = None

        fig = plot_spectrum(
            self.current_spectrum, show=False,
            show_gas_peaks=self._show_gas_peaks,
            detected_peaks=self._detected_peaks if self._show_auto_peaks else None,
            show_individual_rows=self.img_show_rows_cb.isChecked(),
        )
        self.current_figure = fig
        ax = fig.axes[0]

        # 重绘已有的 SNR 选区
        if self.snr_signal_range:
            ax.axvspan(self.snr_signal_range[0], self.snr_signal_range[1],
                       alpha=0.15, color="green", label="信号区域")
        if self.snr_noise_range:
            ax.axvspan(self.snr_noise_range[0], self.snr_noise_range[1],
                       alpha=0.15, color="red", label="噪声区域")
        if self.snr_signal_range or self.snr_noise_range:
            ax.legend(loc="upper right", fontsize=8)

        # 十字光标 (全量重绘时)
        self._draw_cursor(ax)

        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        self._toolbar_ref = toolbar  # 用于光标检测工具栏模式

        # SNR 框选 + 光标交互
        self._snr_span = None
        self._snr_selecting = None
        canvas.mpl_connect("button_press_event", self._on_canvas_click)
        canvas.mpl_connect("key_press_event", self._on_key_press)
        canvas.setFocusPolicy(Qt.StrongFocus)
        canvas.setFocus()

        self.central_layout.addWidget(toolbar)
        self.central_layout.addWidget(canvas)
        self.canvas = canvas

    def _on_canvas_click(self, event):
        """画布点击: 左键=置光标/SNR框选, 右键=取消光标."""
        from matplotlib.backend_bases import MouseButton
        if event.inaxes is None:
            return

        # 导航工具栏激活时跳过
        if hasattr(self, '_toolbar_ref') and self._toolbar_ref.mode != "":
            return

        if event.button == MouseButton.RIGHT:
            self._clear_cursor()
            self._redraw_cursor_only()
            return

        # 左键: SNR 框选模式
        if self._snr_selecting:
            from matplotlib.widgets import SpanSelector
            mode = self._snr_selecting
            self._snr_selecting = None
            color = "green" if mode == "signal" else "red"

            def on_select(xmin, xmax):
                unit = self.current_spectrum.x_unit_label if self.current_spectrum else ""
                if mode == "signal":
                    self.snr_signal_range = (xmin, xmax)
                    self.snr_signal_label.setText(f"信号: {xmin:.1f} ~ {xmax:.1f} {unit}")
                else:
                    self.snr_noise_range = (xmin, xmax)
                    self.snr_noise_label.setText(f"噪声: {xmin:.1f} ~ {xmax:.1f} {unit}")
                self.status_bar.showMessage(f"{'信号' if mode == 'signal' else '噪声'}区域已选择")
                self._update_plot()

            self._snr_span = SpanSelector(
                event.inaxes, on_select, "horizontal",
                props=dict(alpha=0.3, facecolor=color),
                interactive=True, drag_from_anywhere=True,
            )
            return

        # 普通左键: 放置十字光标
        if event.button == MouseButton.LEFT and self.current_spectrum is not None:
            x = event.xdata
            if x is not None:
                spec = self.current_spectrum
                idx = np.argmin(np.abs(spec.raman_shift - x))
                self._cursor_x = float(spec.raman_shift[idx])
                self._cursor_y = float(spec.intensity[idx])
                self._update_cursor_status()
                self._redraw_cursor_only()

    def _on_key_press(self, event):
        """方向键移动光标: 左右=单步, 上下=跳转邻近峰."""
        if self._cursor_x is None or self.current_spectrum is None:
            return

        spec = self.current_spectrum
        idx = np.argmin(np.abs(spec.raman_shift - self._cursor_x))
        step = 1  # 单像素步进

        if event.key == "right":
            idx = min(spec.size - 1, idx + step)
        elif event.key == "left":
            idx = max(0, idx - step)
        elif event.key == "up" or event.key == "down":
            # 跳转到邻近的峰
            if self._detected_peaks:
                peaks = sorted(self._detected_peaks, key=lambda p: p["center"])
                cur_x = spec.raman_shift[idx]
                if event.key == "up":
                    # 找右边最近的峰
                    for p in peaks:
                        if p["center"] > cur_x:
                            idx = np.argmin(np.abs(spec.raman_shift - p["center"]))
                            break
                else:
                    # 找左边最近的峰
                    for p in reversed(peaks):
                        if p["center"] < cur_x:
                            idx = np.argmin(np.abs(spec.raman_shift - p["center"]))
                            break
        elif event.key == "escape":
            self._clear_cursor()
            self._redraw_cursor_only()
            return
        else:
            return

        self._cursor_x = float(spec.raman_shift[idx])
        self._cursor_y = float(spec.intensity[idx])
        self._update_cursor_status()
        self._redraw_cursor_only()

    def _draw_cursor(self, ax):
        """在坐标轴上绘制/更新十字光标 (复用线对象)."""
        if self._cursor_x is None:
            return
        cx, cy = self._cursor_x, self._cursor_y

        # 复用或创建光标线
        if hasattr(self, '_cursor_h_line') and self._cursor_h_line is not None:
            self._cursor_h_line.set_ydata([cy, cy])
            self._cursor_h_line.set_visible(True)
        else:
            self._cursor_h_line = ax.axhline(cy, color="red", linewidth=1, alpha=0.7, linestyle="--", zorder=10)
            self._cursor_lines.append(self._cursor_h_line)

        if hasattr(self, '_cursor_v_line') and self._cursor_v_line is not None:
            self._cursor_v_line.set_xdata([cx, cx])
            self._cursor_v_line.set_visible(True)
        else:
            self._cursor_v_line = ax.axvline(cx, color="red", linewidth=1, alpha=0.7, linestyle="--", zorder=10)
            self._cursor_lines.append(self._cursor_v_line)

    def _redraw_cursor_only(self):
        """仅重绘光标，不触发全量 _update_plot."""
        if self.canvas is None or self.current_figure is None:
            return
        ax = self.current_figure.axes[0]
        self._draw_cursor(ax)
        self.canvas.draw()

    def _update_cursor_status(self):
        if self._cursor_x is None:
            return
        unit = self.current_spectrum.x_unit_label if self.current_spectrum else ""
        self.status_bar.showMessage(
            f"光标: x={self._cursor_x:.2f} {unit}  强度={self._cursor_y:.1f}  |  方向键移动 | 右键取消 | Esc 取消"
        )

    def _clear_cursor(self):
        self._cursor_x = None
        self._cursor_y = None
        if self._cursor_h_line is not None:
            self._cursor_h_line.set_visible(False)
        if self._cursor_v_line is not None:
            self._cursor_v_line.set_visible(False)
        self._cursor_lines = []

    def _confirm_overwrite_path(self, path: Path) -> bool:
        if not path.exists():
            return True
        answer = QMessageBox.question(
            self,
            "确认覆盖",
            f"文件已存在，是否覆盖？\n{path}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _clear_canvas(self):
        if self._placeholder:
            self._placeholder.setParent(None)
            self._placeholder = None

        # 关闭旧图表防止窗口泄漏
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None

        while self.central_layout.count():
            item = self.central_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.canvas = None

    def _on_export_chart(self):
        if self.current_figure is None:
            QMessageBox.warning(self, "提示", "请先生成图表")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出图表", "spectrum.png",
            "PNG 图片 (*.png);;PDF 文档 (*.pdf);;SVG 矢量图 (*.svg)"
        )
        if filepath:
            path = Path(filepath)
            if not self._confirm_overwrite_path(path):
                return
            try:
                save_figure(self.current_figure, path)
                self._log(f"图表已导出: {filepath}")
                self.status_bar.showMessage(f"已导出: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {e}")

    def _on_export_spectrum(self):
        if self.current_spectrum is None:
            QMessageBox.warning(self, "提示", "请先载入光谱文件")
            return

        default_name = Path(self.current_file).with_suffix(".asc").name if self.current_file else "spectrum.asc"
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出当前光谱数据",
            default_name,
            "ASC 文件 (*.asc);;TXT 文件 (*.txt)",
        )
        if not filepath:
            return
        suffix = ".txt" if "TXT" in selected_filter.upper() else ".asc"
        path = normalize_export_path(filepath, suffix)
        if not self._confirm_overwrite_path(path):
            return
        try:
            spectrum = (
                self.current_session.spectrum_for_export()
                if self.current_session is not None
                else self.current_spectrum
            )
            path = export_spectrum(spectrum, path, overwrite=True, include_history=True)
            self._log(f"当前光谱数据已导出: {path}")
            self.status_bar.showMessage(f"已导出当前光谱数据: {path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出当前光谱数据失败: {e}")

    def _on_export_raw_spectrum(self):
        session = self._ensure_current_session()
        if session is None:
            QMessageBox.warning(self, "提示", "请先载入光谱文件")
            return

        source = Path(self.current_file) if self.current_file else Path("spectrum")
        default_name = f"{source.stem}_original.asc"
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出原始光谱数据",
            default_name,
            "ASC 文件 (*.asc);;TXT 文件 (*.txt)",
        )
        if not filepath:
            return
        suffix = ".txt" if "TXT" in selected_filter.upper() else ".asc"
        path = normalize_export_path(filepath, suffix)
        if not self._confirm_overwrite_path(path):
            return
        try:
            path = export_spectrum(session.raw, path, overwrite=True)
            self._log(f"原始光谱数据已导出: {path}")
            self.status_bar.showMessage(f"已导出原始光谱数据: {path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出原始光谱数据失败: {e}")

    def _on_calc_snr(self):
        if self.current_spectrum is None:
            QMessageBox.warning(self, "提示", "请先载入光谱文件")
            return

        try:
            peak_start = None
            peak_end = None
            noise_start = None
            noise_end = None

            if self.snr_signal_range:
                peak_start, peak_end = self.snr_signal_range
            if self.snr_noise_range:
                noise_start, noise_end = self.snr_noise_range

            result = calculate_snr(
                self.current_spectrum, peak_start, peak_end, noise_start, noise_end
            )

            text = (
                f"SNR: {result['snr']:.2f}\n"
                f"信号强度: {result['signal']:.2f}\n"
                f"噪声 RMS: {result['noise_rms']:.4f}\n"
                f"峰中心: {result['peak_center']:.2f} {self.current_spectrum.x_unit_label}"
            )
            if result["peak_area"] > 0:
                text += f"\n峰面积: {result['peak_area']:.4f}"

            self.snr_result.setText(text)
            self._log("=== 信噪比计算 ===\n" + text.replace("\n", "\n  "))
            self.control_tabs.setCurrentIndex(0)

        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, "参数错误", f"请检查输入: {e}")

    def _on_baseline(self):
        if self._closing or self._baseline_worker is not None or self._capacity_blocked():
            return
        if self.current_spectrum is None:
            QMessageBox.warning(self, "提示", "请先载入光谱文件")
            return

        try:
            session = self._ensure_current_session()
            if session is None:
                return
            options = self._get_baseline_options()
            self.status_bar.showMessage(
                f"正在进行基线校正 ({self._format_baseline_options(options)})..."
            )
            worker = BaselineThread(session, options)
            self._baseline_worker = worker
            worker.completed.connect(self._on_baseline_done)
            worker.error_occurred.connect(self._on_baseline_error)
            self.baseline_apply_btn.setEnabled(False)
            self._update_history_ui()
            self._workers.start(worker)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"校正失败: {e}")

    def _capacity_blocked(self):
        warning = self.spectrum_sessions.capacity_warning
        if warning:
            QMessageBox.warning(self, "缓存容量已满", warning)
        return bool(warning)

    def _on_baseline_done(self, corrected):
        worker = self.sender()
        if self._closing or worker.session is not self.current_session:
            return
        if worker.session.revision != worker.revision:
            self._log("基线计算期间处理状态已改变，已忽略过期结果")
            return
        worker.session.apply(corrected, "基线校正", worker.options, "手动")
        self._refresh_from_current_session()
        self._log(f"基线校正完成 ({self._format_baseline_options(worker.options)})")
        self.status_bar.showMessage("基线校正完成；原始数据和处理历史已保留")

    def _on_baseline_error(self, error):
        if not self._closing and self.sender().session is self.current_session:
            QMessageBox.critical(self, "错误", f"校正失败: {error}")

    def _on_concentration(self):
        if self.current_spectrum is None:
            QMessageBox.warning(self, "提示", "请先载入光谱文件")
            return

        try:
            gas_name = self.conc_gas.currentData()
            if not gas_name:
                raise ValueError("气体峰位库中没有启用定量的气体")
            window = float(self.conc_window.text())
            strategy = "peak_area" if self.conc_strategy.currentIndex() == 1 else "peak_max"

            result = calculate_concentration(
                self.current_spectrum,
                gas_name=str(gas_name),
                window=window,
                strategy=strategy,
            )
            all_conc = result.get("all_concentrations", {})
            percentages = all_conc.get("percentages", {})
            peaks = all_conc.get("peaks", {})
            gas_library = get_gas_library()
            unit = self.current_spectrum.x_unit_label

            gas_lines = []
            for gas, percentage in percentages.items():
                info = gas_library.get(gas, {})
                name = str(info.get("name") or gas)
                peak = peaks.get(gas, {})
                gas_lines.append(
                    f"{gas} ({name}): {percentage:.4f}%  "
                    f"I={peak.get('intensity', 0.0):.4f}  "
                    f"峰位={peak.get('center', 0.0):.2f} {unit}"
                )

            text = (
                f"聚焦气体: {result['gas']} ({result['gas_key']})\n"
                f"加权信号占比: {result['concentration']:.4f}%\n"
                f"峰中心: {result['peak_center']:.2f} {unit}\n"
                f"峰高: {result['peak_height']:.4f}\n"
                f"峰面积: {result['peak_area']:.4f}\n\n"
                + "\n".join(gas_lines)
                + "\n\n" + "\n".join(all_conc.get("warnings", []))
            )
            self.conc_result.setText(text)
            self._log("=== 气体加权信号占比 ===\n" + text.replace("\n", "\n  "))

        except ValueError as e:
            QMessageBox.warning(self, "错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {e}")

    def _on_batch(self):
        if self._closing or self._batch_worker is not None:
            return
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "提示", "请先添加文件到列表")
            return

        do_baseline = self.batch_baseline_cb.isChecked()
        files = [Path(self.file_list.item(i).data(Qt.UserRole))
                for i in range(self.file_list.count())]
        rg_text = self.img_row_groups.text().strip()
        row_groups = rg_text if rg_text else None
        try:
            col_merge = integer_parameter(self.img_col_merge.text(), "列合并", minimum=1)
            calibration = self._get_calibration()
            baseline_options = self._get_baseline_options() if do_baseline else None
            focus_gas = self.conc_gas.currentData()
            window = positive_float(self.conc_window.text(), "定量半窗口")
            windows = None
            if focus_gas:
                info = get_gas_library()[focus_gas]
                windows = {focus_gas: (float(info["center"]), window)}
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        row_mode = "sum" if self.row_mode_combo.currentIndex() == 1 else "mean"
        strategy = "peak_area" if self.conc_strategy.currentIndex() == 1 else "peak_max"

        self.batch_progress.setVisible(True)
        self.batch_progress.setMaximum(len(files))
        self.batch_progress.setValue(0)

        self.batch_thread = BatchProcessThread(
            files,
            do_baseline,
            row_groups=row_groups,
            col_merge=col_merge,
            calibration=calibration,
            row_mode=row_mode,
            strategy=strategy,
            baseline_options=baseline_options,
            windows=windows,
        )
        self.batch_thread.progress_update.connect(self._on_batch_progress)
        self.batch_thread.file_done.connect(self._on_batch_file_done)
        self.batch_thread.all_done.connect(self._on_batch_done)
        self._start_batch_worker(self.batch_thread)

    def _on_export_all_spectra(self):
        if self._closing or self._batch_worker is not None:
            return
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "提示", "请先添加文件到列表")
            return

        format_text, ok = QInputDialog.getItem(
            self,
            "导出全部光谱数据",
            "导出格式:",
            ["ASC (*.asc)", "TXT (*.txt)"],
            0,
            False,
        )
        if not ok:
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择光谱数据导出目录")
        if not output_dir:
            return

        files = [Path(self.file_list.item(i).data(Qt.UserRole))
                for i in range(self.file_list.count())]
        rg_text = self.img_row_groups.text().strip()
        row_groups = rg_text if rg_text else None
        try:
            col_merge = integer_parameter(self.img_col_merge.text(), "列合并", minimum=1)
            calibration = self._get_calibration()
            baseline_options = self._get_baseline_options() if self.batch_baseline_cb.isChecked() else None
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        row_mode = "sum" if self.row_mode_combo.currentIndex() == 1 else "mean"
        suffix = ".txt" if format_text.startswith("TXT") else ".asc"

        self.batch_progress.setVisible(True)
        self.batch_progress.setMaximum(len(files))
        self.batch_progress.setValue(0)
        self.batch_status.setText("正在批量导出光谱数据...")

        self.batch_export_thread = BatchExportThread(
            files,
            Path(output_dir),
            suffix,
            do_baseline=self.batch_baseline_cb.isChecked(),
            row_groups=row_groups,
            col_merge=col_merge,
            calibration=calibration,
            row_mode=row_mode,
            baseline_options=baseline_options,
        )
        self.batch_export_thread.progress_update.connect(self._on_batch_progress)
        self.batch_export_thread.file_done.connect(self._on_batch_file_done)
        self.batch_export_thread.all_done.connect(self._on_batch_export_done)
        self._start_batch_worker(self.batch_export_thread)

    def _start_batch_worker(self, worker):
        self._batch_worker = worker
        self.last_batch_failures = []
        self.batch_start_btn.setEnabled(False)
        self.batch_cancel_btn.setEnabled(True)
        self._workers.start(worker)

    def _on_batch_progress(self, current: int, total: int):
        self.batch_progress.setValue(current)

    def _on_batch_file_done(self, filename: str, success: bool, reason: str = ""):
        if self._closing:
            return
        if success:
            self._log(f"  ✓ {filename}")
        else:
            self._log(f"  ✗ {filename}: {reason}")

    def _on_batch_done(self, ok: int, fail: int, results: list):
        worker = self.sender()
        self.last_batch_failures = list(worker.failures)
        if self._closing:
            return
        self.batch_progress.setVisible(False)
        state = "已取消" if worker.cancelled else "完成"
        self.batch_status.setText(f"{state}: {ok} 成功, {fail} 失败")
        self._log(f"批量处理完成: {ok} 成功, {fail} 失败")
        if results or self.last_batch_failures:
            dialog = ConcentrationResultDialog(results, self, failures=self.last_batch_failures)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.show()

    def _on_batch_export_done(self, ok: int, fail: int, output_dir: str):
        worker = self.sender()
        self.last_batch_failures = list(worker.failures)
        if self._closing:
            return
        self.batch_progress.setVisible(False)
        state = "已取消" if worker.cancelled else "导出完成"
        self.batch_status.setText(f"{state}: {ok} 成功, {fail} 失败")
        self._log(f"批量导出光谱数据完成: {ok} 成功, {fail} 失败, 输出目录: {output_dir}")
        if fail:
            self._log("导出失败明细已保留，可从日志查看文件名与原因")
        if worker.failure_report_path:
            self._log(f"导出失败报告: {worker.failure_report_path}")
        if worker.report_error:
            self._log(f"无法保存失败报告: {worker.report_error}")

    def _on_about(self):
        QMessageBox.about(
            self, "关于",
            "<h3>拉曼光谱数据处理工具 v0.1.0</h3>"
            "<p>支持多种文件格式读取、可视化、信噪比计算、气体浓度分析。</p>"
            "<p>支持格式: TXT, ASC, SIF, TIFF, BMP</p>"
            "<p>基于 PySide6 + matplotlib 构建</p>"
        )

    # ── 拖拽支持 ──
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            try:
                detect_format(path)
                files.append(path)
            except ValueError:
                pass

        if files:
            self._add_files(files)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("RamanTool")
    app.setOrganizationName("RamanTool")

    window = RamanQtGUI()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
