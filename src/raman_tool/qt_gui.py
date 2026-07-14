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
    QMenuBar, QMenu, QTabWidget, QSplitter, QComboBox, QCheckBox,
    QProgressBar, QDockWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QInputDialog, QFormLayout, QSpinBox, QDialogButtonBox,
)
from PySide6.QtCore import Qt, Signal, QThread, QMimeData
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent

from raman_tool.models import Spectrum
from raman_tool.readers import read_file, detect_format, SUPPORTED_FORMATS
from raman_tool.sorting import natural_sorted
from raman_tool.exporters import export_spectrum, normalize_export_path, unique_path
from raman_tool.processing import calculate_snr, calculate_concentration, calculate_gas_concentrations, subtract_baseline
from raman_tool.visualization import plot_spectrum, plot_baseline, save_figure, plot_multiple
from raman_tool.config import get_config, save_config, default_config_path
from raman_tool.presets import delete_workflow_preset, load_workflow_presets, save_workflow_preset
from raman_tool.safety import refresh_limits
from raman_tool.gas_library import (
    DEFAULT_GAS_LIBRARY, default_gas_library_path, get_gas_choices,
    load_gas_library, reload_gas_library, save_gas_library,
)


class GasLibraryDialog(QDialog):
    COLUMNS = ["key", "name", "center", "half_width", "coefficient", "color", "enabled", "quantitative"]
    HEADERS = ["气体", "名称", "峰位", "半窗宽", "系数", "颜色", "启用", "定量"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("气体峰位库")
        self.resize(820, 520)

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        buttons_row = QHBoxLayout()
        add_btn = QPushButton("添加")
        add_btn.clicked.connect(self._add_row)
        buttons_row.addWidget(add_btn)

        remove_btn = QPushButton("删除选中")
        remove_btn.clicked.connect(self._remove_selected)
        buttons_row.addWidget(remove_btn)

        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self._reset_defaults)
        buttons_row.addWidget(reset_btn)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        hint = QLabel(f"库文件: {default_gas_library_path()}")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialog_buttons.accepted.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(dialog_buttons)

        self._load(load_gas_library())

    def _load(self, library: dict):
        self.table.setRowCount(0)
        for key, entry in library.items():
            self._add_row(key, entry)

    def _add_row(self, key: str = "", entry: dict | None = None):
        if not isinstance(key, str):
            key = ""
        if entry is None:
            entry = {
                "name": "",
                "center": 1000.0,
                "half_width": 25.0,
                "coefficient": 1.0,
                "color": "#999999",
                "enabled": True,
                "quantitative": False,
            }
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = {
            "key": key,
            "name": entry.get("name", ""),
            "center": entry.get("center", 1000.0),
            "half_width": entry.get("half_width", 25.0),
            "coefficient": entry.get("coefficient", 1.0),
            "color": entry.get("color", "#999999"),
            "enabled": entry.get("enabled", True),
            "quantitative": entry.get("quantitative", False),
        }
        for col, field in enumerate(self.COLUMNS):
            if field in {"enabled", "quantitative"}:
                item = QTableWidgetItem("")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if values[field] else Qt.Unchecked)
            else:
                item = QTableWidgetItem(str(values[field]))
            self.table.setItem(row, col, item)

    def _remove_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _reset_defaults(self):
        if QMessageBox.question(
            self,
            "恢复默认",
            "确定要恢复默认气体峰位库吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) == QMessageBox.Yes:
            self._load(DEFAULT_GAS_LIBRARY)

    def to_library(self) -> dict:
        library = {}
        for row in range(self.table.rowCount()):
            values = {}
            key = ""
            for col, field in enumerate(self.COLUMNS):
                item = self.table.item(row, col)
                if field == "key":
                    key = item.text().strip() if item else ""
                elif field in {"enabled", "quantitative"}:
                    values[field] = bool(item and item.checkState() == Qt.Checked)
                else:
                    values[field] = item.text().strip() if item else ""
            if key:
                library[key] = values
        return library


class SafetySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("安全限制设置")
        self.setMinimumWidth(420)
        safety = get_config()["safety"]
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.max_input_mb = self._spin(1, 4096, safety["max_input_file_mb"])
        self.max_text_mb = self._spin(1, 2048, safety["max_text_file_mb"])
        self.max_data_points = self._spin(1_000, 50_000_000, safety["max_data_points"])
        self.max_image_side = self._spin(2048, 16384, int(safety["max_image_pixels"] ** 0.5))
        self.max_image_channels = self._spin(1, 8, safety["max_image_channels"])
        self.max_baseline_points = self._spin(1_000, 5_000_000, safety["max_baseline_points"])
        form.addRow("最大输入文件 (MB):", self.max_input_mb)
        form.addRow("最大文本文件 (MB):", self.max_text_mb)
        form.addRow("最大光谱点数:", self.max_data_points)
        form.addRow("最大图像边长 (px):", self.max_image_side)
        form.addRow("最大图像通道数:", self.max_image_channels)
        form.addRow("arPLS 最大点数:", self.max_baseline_points)
        hint = QLabel(f"配置文件: {default_config_path()}")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(int(value))
        spin.setSingleStep(max(1, (maximum - minimum) // 100))
        return spin

    def to_config(self) -> dict:
        side = self.max_image_side.value()
        return {
            "safety": {
                "max_input_file_mb": self.max_input_mb.value(),
                "max_text_file_mb": self.max_text_mb.value(),
                "max_data_points": self.max_data_points.value(),
                "min_supported_image_pixels": 2048 * 2048,
                "max_image_pixels": side * side,
                "max_image_channels": self.max_image_channels.value(),
                "max_baseline_points": self.max_baseline_points.value(),
            }
        }


class ImageViewerWindow(QMainWindow):
    """独立的 2D 灰度图显示窗口 (带十字光标)."""

    def __init__(self, filepath: str, img_array: np.ndarray, parent=None):
        super().__init__(parent)
        self.filepath = str(filepath)
        self._img_array = img_array
        self._cursor_col: int | None = None
        self._cursor_row: int | None = None
        self._h_line = None
        self._v_line = None

        self.setWindowTitle(f"图像 - {Path(self.filepath).name}")
        self.resize(900, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._fig, self._ax = plt.subplots(figsize=(8, 6))
        self._ax.imshow(self._img_array, cmap="gray", aspect="auto", origin="upper")
        self._ax.set_title(Path(self.filepath).name)
        self._ax.set_xlabel("列")
        self._ax.set_ylabel("行")

        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("key_press_event", self._on_key)
        self._canvas.setFocusPolicy(Qt.StrongFocus)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("左键定位 | 方向键移动 | 右键/Esc 取消")

    def showEvent(self, event):
        super().showEvent(event)
        self._canvas.setFocus()

    def _on_click(self, event):
        from matplotlib.backend_bases import MouseButton
        if event.inaxes != self._ax:
            return
        if self._toolbar.mode != "":
            return
        if event.button == MouseButton.RIGHT:
            self._clear_cursor()
            return
        if event.button == MouseButton.LEFT:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self._cursor_col = int(round(x))
                self._cursor_row = int(round(y))
                self._update_cursor()

    def _on_key(self, event):
        if self._cursor_col is None:
            return
        h, w = self._img_array.shape
        if event.key == "right":
            self._cursor_col = min(w - 1, self._cursor_col + 1)
        elif event.key == "left":
            self._cursor_col = max(0, self._cursor_col - 1)
        elif event.key == "up":
            self._cursor_row = max(0, self._cursor_row - 1)
        elif event.key == "down":
            self._cursor_row = min(h - 1, self._cursor_row + 1)
        elif event.key == "escape":
            self._clear_cursor()
            return
        else:
            return
        self._update_cursor()

    def _update_cursor(self):
        # 创建或更新光标线 (复用对象，不反复 remove/redraw)
        col, row = self._cursor_col, self._cursor_row
        if self._h_line is None:
            self._h_line = self._ax.axhline(row, color="red", linewidth=1.2, alpha=0.9, zorder=100)
        else:
            self._h_line.set_ydata([row, row])
        if self._v_line is None:
            self._v_line = self._ax.axvline(col, color="red", linewidth=1.2, alpha=0.9, zorder=100)
        else:
            self._v_line.set_xdata([col, col])

        val = self._img_array[row, col]
        self.status_bar.showMessage(
            f"行={row}  列={col}  值={val:.1f}  |  方向键移动 | 右键/Esc 取消"
        )
        self._canvas.draw_idle()

    def _clear_cursor(self):
        self._cursor_col = None
        self._cursor_row = None
        if self._h_line:
            self._h_line.set_visible(False)
        if self._v_line:
            self._v_line.set_visible(False)
        self._canvas.draw_idle()
        self.status_bar.showMessage("左键定位 | 方向键移动 | 右键/Esc 取消")


class ConcentrationResultDialog(QDialog):
    """Batch concentration curve dialog."""

    def __init__(self, results: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量浓度结果")
        self.resize(980, 620)
        self._results = results
        self._gases = self._collect_gases(results)
        self._cursor_index: int | None = None
        self._cursor_v_line = None
        self._cursor_markers = {}
        self._stats = self._compute_fluctuation_stats(results, self._gases)

        layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()
        layout.addLayout(content_layout)

        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        content_layout.addWidget(chart_widget, stretch=1)

        tool_box = QGroupBox("功能")
        tool_layout = QVBoxLayout(tool_box)
        tool_box.setMaximumWidth(240)
        content_layout.addWidget(tool_box)

        self._fig, self._ax = plt.subplots(figsize=(8, 5))
        x = list(range(1, len(results) + 1))
        self._gas_lines = {
            gas: self._ax.plot(
                x,
                [self._row_percent(row, gas) for row in results],
                marker="o",
                linewidth=1.2,
                label=gas,
            )[0]
            for gas in self._gases
        }
        self._ax.set_xlabel("文件序号")
        self._ax.set_ylabel("浓度 (%)")
        self._ax.set_title(" / ".join(self._gases) + " 浓度曲线" if self._gases else "浓度曲线")
        self._ax.grid(True, alpha=0.3)
        if self._gas_lines:
            self._ax.legend(loc="best")
        self._fig.tight_layout()

        self._canvas = FigureCanvas(self._fig)
        toolbar = NavigationToolbar(self._canvas, self)
        chart_layout.addWidget(toolbar)
        chart_layout.addWidget(self._canvas)
        self._canvas.mpl_connect("button_press_event", self._on_chart_click)
        self._canvas.mpl_connect("key_press_event", self._on_key_press)
        self._canvas.setFocusPolicy(Qt.StrongFocus)

        summary_parts = [f"共 {len(results)} 个结果"]
        for gas in self._gases:
            vals = [self._row_percent(row, gas) for row in results]
            summary_parts.append(f"{gas}: {np.mean(vals):.4f}%")
        summary = QLabel(" | ".join(summary_parts))
        summary.setWordWrap(True)
        chart_layout.addWidget(summary)

        tool_layout.addWidget(QLabel("显示气体:"))
        self._gas_checks = {}
        for gas in self._gases:
            cb = QCheckBox(gas)
            cb.setChecked(True)
            cb.toggled.connect(self._update_visible_gases)
            tool_layout.addWidget(cb)
            self._gas_checks[gas] = cb

        tool_layout.addWidget(QLabel(""))
        tool_layout.addWidget(QLabel("波动分析:"))
        self._stats_label = QLabel(self._format_stats_text())
        self._stats_label.setWordWrap(True)
        tool_layout.addWidget(self._stats_label)

        tool_layout.addWidget(QLabel(""))
        tool_layout.addWidget(QLabel("光标读数:"))
        self._cursor_status = QLabel("左键定位曲线\n方向键移动\n右键/Esc 取消")
        self._cursor_status.setWordWrap(True)
        tool_layout.addWidget(self._cursor_status)

        tool_layout.addWidget(QLabel(""))
        self._export_cb = QCheckBox("导出 TXT")
        self._export_cb.toggled.connect(self._on_export_toggled)
        tool_layout.addWidget(self._export_cb)

        self._save_btn = QPushButton("选择位置并保存")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_results_txt)
        tool_layout.addWidget(self._save_btn)

        self._save_status = QLabel("")
        self._save_status.setWordWrap(True)
        tool_layout.addWidget(self._save_status)
        tool_layout.addStretch()
        self._canvas.setFocus()

    @staticmethod
    def _collect_gases(results: list[dict]) -> list[str]:
        gases: list[str] = []
        for row in results:
            row_gases = row.get("gases") or list(row.get("percentages", {}).keys())
            for gas in row_gases:
                if gas not in gases:
                    gases.append(gas)
        return gases

    @staticmethod
    def _row_percent(row: dict, gas: str) -> float:
        return float(row.get("percentages", {}).get(gas, row.get(gas, 0.0)))

    @staticmethod
    def _row_intensity(row: dict, gas: str) -> float:
        peaks = row.get("peaks", {})
        return float(peaks.get(gas, {}).get("intensity", row.get(f"{gas}_I", 0.0)))

    @staticmethod
    def _compute_fluctuation_stats(results: list[dict], gases: list[str]) -> dict:
        stats = {}
        for gas in gases:
            vals = np.array([ConcentrationResultDialog._row_percent(row, gas) for row in results], dtype=np.float64)
            if vals.size == 0:
                stats[gas] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "cv": 0.0}
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
            stats[gas] = {
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "range": max_val - min_val,
                "cv": std / mean if abs(mean) > 1e-12 else 0.0,
            }
        return stats

    def _format_stats_text(self) -> str:
        lines = []
        for gas in self._gases:
            s = self._stats[gas]
            lines.append(
                f"{gas}: std={s['std']:.4f}%\n"
                f"  min={s['min']:.4f}% max={s['max']:.4f}%\n"
                f"  range={s['range']:.4f}% CV={s['cv']:.4f}"
            )
        return "\n".join(lines)

    def _update_visible_gases(self, *_args):
        any_visible = False
        for gas, line in self._gas_lines.items():
            visible = self._gas_checks[gas].isChecked()
            line.set_visible(visible)
            marker = self._cursor_markers.get(gas)
            if marker is not None:
                marker.set_visible(visible and self._cursor_index is not None)
            any_visible = any_visible or visible

        if any_visible:
            self._ax.legend(
                [line for line in self._gas_lines.values() if line.get_visible()],
                [gas for gas, line in self._gas_lines.items() if line.get_visible()],
                loc="best",
            )
        legend = self._ax.get_legend()
        if legend is not None:
            legend.set_visible(any_visible)
        self._ax.relim(visible_only=True)
        self._ax.autoscale_view()
        self._update_cursor_status()
        self._canvas.draw_idle()

    def _on_chart_click(self, event):
        from matplotlib.backend_bases import MouseButton
        if event.inaxes != self._ax:
            return
        if event.button == MouseButton.RIGHT:
            self._clear_cursor()
            return
        if event.button != MouseButton.LEFT or event.xdata is None or not self._results:
            return
        idx = int(round(event.xdata)) - 1
        idx = max(0, min(len(self._results) - 1, idx))
        self._set_cursor_index(idx)

    def _on_key_press(self, event):
        if self._cursor_index is None:
            return
        if event.key == "right":
            self._set_cursor_index(min(len(self._results) - 1, self._cursor_index + 1))
        elif event.key == "left":
            self._set_cursor_index(max(0, self._cursor_index - 1))
        elif event.key == "escape":
            self._clear_cursor()

    def _set_cursor_index(self, idx: int):
        self._cursor_index = idx
        x = idx + 1
        row = self._results[idx]
        if self._cursor_v_line is None:
            self._cursor_v_line = self._ax.axvline(x, color="red", linewidth=1, linestyle="--", alpha=0.75, zorder=20)
        else:
            self._cursor_v_line.set_xdata([x, x])
            self._cursor_v_line.set_visible(True)

        for gas in self._gases:
            y = self._row_percent(row, gas)
            marker = self._cursor_markers.get(gas)
            if marker is None:
                marker = self._ax.plot([x], [y], "o", color="red", markersize=5, zorder=25)[0]
                self._cursor_markers[gas] = marker
            else:
                marker.set_data([x], [y])
            marker.set_visible(self._gas_checks[gas].isChecked())

        self._update_cursor_status()
        self._canvas.setFocus()
        self._canvas.draw_idle()

    def _update_cursor_status(self):
        if self._cursor_index is None:
            self._cursor_status.setText("左键定位曲线\n方向键移动\n右键/Esc 取消")
            return
        row = self._results[self._cursor_index]
        lines = [f"序号: {row['index']}", f"文件: {row['filename']}"]
        for gas in self._gases:
            if self._gas_checks[gas].isChecked():
                lines.append(f"{gas}: {self._row_percent(row, gas):.4f}%")
        self._cursor_status.setText("\n".join(lines))

    def _clear_cursor(self):
        self._cursor_index = None
        if self._cursor_v_line is not None:
            self._cursor_v_line.set_visible(False)
        for marker in self._cursor_markers.values():
            marker.set_visible(False)
        self._update_cursor_status()
        self._canvas.draw_idle()

    def _on_export_toggled(self, checked: bool):
        self._save_btn.setEnabled(checked)
        if not checked:
            self._save_status.setText("")

    def _save_results_txt(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "保存浓度结果",
            "concentration_results.txt",
            "TXT 文件 (*.txt);;所有文件 (*.*)",
        )
        if not filepath:
            return
        path = Path(filepath)
        if path.suffix.lower() != ".txt":
            path = path.with_suffix(".txt")

        header = ["index", "filename"]
        for gas in self._gases:
            header.extend([f"{gas}_percent", f"{gas}_I"])
        lines = ["\t".join(header)]
        for row in self._results:
            values = [str(row["index"]), row["filename"]]
            for gas in self._gases:
                values.append(f"{self._row_percent(row, gas):.6f}")
                values.append(f"{self._row_intensity(row, gas):.6f}")
            lines.append("\t".join(values))
        lines.append("")
        lines.append("fluctuation_analysis")
        lines.append("\t".join(["gas", "mean_percent", "std_percent", "min_percent", "max_percent", "range_percent", "cv"]))
        for gas in self._gases:
            s = self._stats[gas]
            lines.append("\t".join([
                gas,
                f"{s['mean']:.6f}",
                f"{s['std']:.6f}",
                f"{s['min']:.6f}",
                f"{s['max']:.6f}",
                f"{s['range']:.6f}",
                f"{s['cv']:.6f}",
            ]))
        if path.exists():
            answer = QMessageBox.question(
                self,
                "确认覆盖",
                f"文件已存在，是否覆盖？\n{path}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self._save_status.setText("已取消保存")
                return
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._save_status.setText(f"已保存:\n{path}")


class SpectrumLoadThread(QThread):
    """后台加载光谱文件的线程."""
    finished_loading = Signal(object, str, str)  # spectrum, filepath, cache_key
    error_occurred = Signal(str)

    def __init__(
        self,
        filepath: str,
        row_groups: str | None = None,
        col_merge: int = 1,
        calibration: tuple | None = None,
        row_mode: str = "mean",
        cache_key: str | None = None,
    ):
        super().__init__()
        self.filepath = filepath
        self.row_groups = row_groups
        self.col_merge = col_merge
        self.calibration = calibration
        self.row_mode = row_mode
        self.cache_key = cache_key or filepath

    def run(self):
        try:
            spectrum = read_file(
                Path(self.filepath),
                row_groups=self.row_groups,
                col_merge=self.col_merge,
                calibration=self.calibration,
                row_mode=self.row_mode,
            )
            self.finished_loading.emit(spectrum, self.filepath, self.cache_key)
        except Exception as e:
            self.error_occurred.emit(str(e))


class BatchProcessThread(QThread):
    """后台批处理线程."""
    progress_update = Signal(int, int)
    file_done = Signal(str, bool)
    all_done = Signal(int, int, object)

    def __init__(
        self,
        files,
        do_baseline,
        row_groups: str | None = None,
        col_merge: int = 1,
        calibration: tuple | None = None,
        row_mode: str = "mean",
        strategy: str = "peak_max",
        baseline_options: dict | None = None,
    ):
        super().__init__()
        self.files = files
        self.do_baseline = do_baseline
        self.row_groups = row_groups
        self.col_merge = col_merge
        self.calibration = calibration
        self.row_mode = row_mode
        self.strategy = strategy
        self.baseline_options = baseline_options or {"method": "arPLS"}

    def run(self):
        ok = 0
        fail = 0
        total = len(self.files)
        results = []
        for i, f in enumerate(self.files):
            try:
                spectrum = read_file(
                    f,
                    row_groups=self.row_groups,
                    col_merge=self.col_merge,
                    calibration=self.calibration,
                    row_mode=self.row_mode,
                )
                if self.do_baseline:
                    spectrum = subtract_baseline(spectrum, **self.baseline_options)
                conc = calculate_gas_concentrations(spectrum, strategy=self.strategy)
                percentages = conc["percentages"]
                peaks = conc["peaks"]
                row = {
                    "index": ok + 1,
                    "filename": f.name,
                    "gases": list(percentages.keys()),
                    "percentages": percentages,
                    "peaks": peaks,
                }
                for gas, value in percentages.items():
                    row[gas] = value
                    row[f"{gas}_I"] = peaks.get(gas, {}).get("intensity", 0.0)
                results.append(row)
                ok += 1
                self.file_done.emit(f.name, True)
            except Exception:
                fail += 1
                self.file_done.emit(f.name, False)
            self.progress_update.emit(i + 1, total)
        self.all_done.emit(ok, fail, results)


class BatchExportThread(QThread):
    """后台批量导出光谱数据的线程."""
    progress_update = Signal(int, int)
    file_done = Signal(str, bool)
    all_done = Signal(int, int, str)

    def __init__(
        self,
        files,
        output_dir: Path,
        suffix: str,
        do_baseline: bool = False,
        row_groups: str | None = None,
        col_merge: int = 1,
        calibration: tuple | None = None,
        row_mode: str = "mean",
        baseline_options: dict | None = None,
    ):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.suffix = suffix
        self.do_baseline = do_baseline
        self.row_groups = row_groups
        self.col_merge = col_merge
        self.calibration = calibration
        self.row_mode = row_mode
        self.baseline_options = baseline_options or {"method": "arPLS"}

    def run(self):
        ok = 0
        fail = 0
        total = len(self.files)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(self.files):
            try:
                spectrum = read_file(
                    f,
                    row_groups=self.row_groups,
                    col_merge=self.col_merge,
                    calibration=self.calibration,
                    row_mode=self.row_mode,
                )
                if self.do_baseline:
                    spectrum = subtract_baseline(spectrum, **self.baseline_options)
                out_path = unique_path(self.output_dir / f.with_suffix(self.suffix).name)
                export_spectrum(spectrum, out_path)
                ok += 1
                self.file_done.emit(out_path.name, True)
            except Exception:
                fail += 1
                self.file_done.emit(f.name, False)
            self.progress_update.emit(i + 1, total)
        self.all_done.emit(ok, fail, str(self.output_dir))


class RamanQtGUI(QMainWindow):
    """拉曼光谱处理工具 Qt 主窗口."""

    def __init__(self):
        super().__init__()

        self.current_spectrum: Spectrum | None = None
        self.current_file: str = ""
        self.loaded_spectra: dict[str, Spectrum] = {}
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
            try:
                lam = float(lam_text) if lam_text else 1e5
            except ValueError as exc:
                raise ValueError("arPLS 平滑参数 lam 必须是数字") from exc
            return {"method": "arPLS", "lam": lam}
        return {"method": "poly", "degree": int(self.baseline_degree.currentText())}

    def _subtract_baseline_with_options(self, spectrum: Spectrum, options: dict) -> Spectrum:
        method = options.get("method", "arPLS")
        if method == "poly":
            return subtract_baseline(spectrum, method="poly", degree=int(options.get("degree", 3)))
        return subtract_baseline(spectrum, method="arPLS", lam=float(options.get("lam", 1e5)))

    def _format_baseline_options(self, options: dict) -> str:
        if options.get("method") == "poly":
            return f"poly, degree={int(options.get('degree', 3))}"
        return f"arPLS, lam={float(options.get('lam', 1e5)):.1e}"

    def _collect_workflow_preset(self) -> dict:
        method = "arPLS" if self.baseline_method.currentIndex() == 0 else "poly"
        strategy = "peak_area" if self.conc_strategy.currentIndex() == 1 else "peak_max"
        row_mode = "sum" if self.row_mode_combo.currentIndex() == 1 else "mean"
        gas_name = str(self.conc_gas.currentData() or self.conc_gas.currentText() or "N2")
        try:
            lam = float(self.baseline_lam.text().strip() or "100000")
        except ValueError as exc:
            raise ValueError("arPLS 平滑参数 lam 必须是数字") from exc
        if lam < 1:
            raise ValueError("arPLS 平滑参数 lam 必须大于或等于 1")
        try:
            window = float(self.conc_window.text().strip() or "10")
        except ValueError as exc:
            raise ValueError("聚焦气体窗口必须是数字") from exc
        if window < 0.1:
            raise ValueError("聚焦气体窗口必须大于或等于 0.1")
        try:
            col_merge = int(self.img_col_merge.text().strip() or "1")
        except ValueError as exc:
            raise ValueError("列合并因子必须是整数") from exc
        if col_merge < 1:
            raise ValueError("列合并因子必须大于或等于 1")

        calibration_values = [
            self.cal_px1.text().strip(),
            self.cal_rs1.text().strip(),
            self.cal_px2.text().strip(),
            self.cal_rs2.text().strip(),
        ]
        if any(calibration_values):
            if not all(calibration_values):
                raise ValueError("拉曼位移校准的四个参数必须全部填写")
            try:
                px1, _rs1, px2, _rs2 = map(float, calibration_values)
            except ValueError as exc:
                raise ValueError("拉曼位移校准参数必须是数字") from exc
            if px1 == px2:
                raise ValueError("拉曼位移校准的两个像素位置不能相同")
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
        self.baseline_method.setCurrentIndex(1 if preset.get("baseline_method") == "poly" else 0)
        self.baseline_lam.setText(str(preset.get("baseline_lam", 100000.0)))
        degree = str(int(preset.get("baseline_degree", 3)))
        idx = self.baseline_degree.findText(degree)
        self.baseline_degree.setCurrentIndex(idx if idx >= 0 else 2)
        self.auto_baseline_cb.setChecked(bool(preset.get("auto_baseline", True)))
        self.batch_baseline_cb.setChecked(bool(preset.get("batch_baseline", True)))

        gas = str(preset.get("concentration_gas", "N2")).casefold()
        for i in range(self.conc_gas.count()):
            if str(self.conc_gas.itemData(i) or "").casefold() == gas:
                self.conc_gas.setCurrentIndex(i)
                break
        self.conc_window.setText(str(preset.get("concentration_window", 10.0)))
        self.conc_strategy.setCurrentIndex(1 if preset.get("concentration_strategy") == "peak_area" else 0)

        self.row_mode_combo.setCurrentIndex(1 if preset.get("row_mode") == "sum" else 0)
        self.img_col_merge.setText(str(max(1, int(preset.get("col_merge", 1)))))
        self.img_row_groups.setText(str(preset.get("row_groups", "")))
        self.img_show_rows_cb.setChecked(bool(preset.get("show_individual_rows", False)))
        self.cal_px1.setText(str(preset.get("calibration_px1", "")))
        self.cal_rs1.setText(str(preset.get("calibration_shift1", "")))
        self.cal_px2.setText(str(preset.get("calibration_px2", "")))
        self.cal_rs2.setText(str(preset.get("calibration_shift2", "")))
        self.gas_peaks_cb.setChecked(bool(preset.get("show_gas_peaks", True)))
        self.auto_peaks_cb.setChecked(bool(preset.get("show_auto_peaks", True)))
        self.loaded_spectra.clear()
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
        if self.current_spectrum is not None:
            if self._show_auto_peaks:
                self._detect_peaks()
            self._update_plot()
        self._log(f"气体峰位库已保存: {path}")
        QMessageBox.information(self, "峰位库已保存", f"气体峰位库已更新。\n{path}")

    def _refresh_concentration_gases(self):
        current = self.conc_gas.currentData() if hasattr(self, "conc_gas") else "N2"
        self.conc_gas.clear()
        choices = get_gas_choices()
        for key, label in choices:
            self.conc_gas.addItem(label, key)
        for i, (key, _label) in enumerate(choices):
            if key.casefold() == str(current or "N2").casefold():
                self.conc_gas.setCurrentIndex(i)
                break

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
        self.baseline_degree.addItems(["1", "2", "3", "4", "5"])
        self.baseline_degree.setCurrentIndex(2)
        self._degree_layout.addWidget(self.baseline_degree)
        g1l.addLayout(self._degree_layout)

        # 初始状态: arPLS 因此隐藏阶数
        self._hide_layout(self._degree_layout)

        # 自动基线校正勾选框
        self.auto_baseline_cb = QCheckBox("载入光谱后自动执行基线校正")
        self.auto_baseline_cb.setChecked(True)
        g1l.addWidget(self.auto_baseline_cb)

        layout.addWidget(g1)

        btn = QPushButton("执行基线校正")
        btn.clicked.connect(self._on_baseline)
        layout.addWidget(btn)
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

    def _setup_concentration_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g1 = QGroupBox("浓度结果")
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
        r4.addWidget(QLabel("聚焦气体窗口 (px):"))
        self.conc_window = QLineEdit("10")
        r4.addWidget(self.conc_window)
        g2l.addLayout(r4)

        r5 = QHBoxLayout()
        r5.addWidget(QLabel("浓度算法:"))
        self.conc_strategy = QComboBox()
        self.conc_strategy.addItems(["峰高归一化 (peak_max)", "峰面积归一化 (peak_area)"])
        r5.addWidget(self.conc_strategy)
        g2l.addLayout(r5)

        layout.addWidget(g2)

        btn = QPushButton("计算 O2/N2/CO2 浓度")
        btn.clicked.connect(self._on_concentration)
        layout.addWidget(btn)

        self.conc_result = QLabel("")
        self.conc_result.setWordWrap(True)
        layout.addWidget(self.conc_result)

        layout.addStretch()
        self.control_tabs.addTab(tab, "气体浓度")

    def _setup_batch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g1 = QGroupBox("批量处理")
        g1l = QVBoxLayout(g1)

        self.batch_baseline_cb = QCheckBox("执行基线校正")
        g1l.addWidget(self.batch_baseline_cb)

        btn = QPushButton("开始批量处理")
        btn.clicked.connect(self._on_batch)
        g1l.addWidget(btn)

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
        cr2.addWidget(QLabel("cm-1:"))
        self.cal_rs2 = QLineEdit()
        self.cal_rs2.setPlaceholderText("如 2331 (N2)")
        cr2.addWidget(self.cal_rs2)
        g3l.addLayout(cr2)

        self.cal_status = QLabel("")
        self.cal_status.setStyleSheet("color: #666; font-size: 11px;")
        g3l.addWidget(self.cal_status)

        # 气体峰位显示开关
        self.gas_peaks_cb = QCheckBox("显示参考气体峰位 (N₂, O₂, CO₂...)")
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
        try:
            px1 = float(self.cal_px1.text())
            rs1 = float(self.cal_rs1.text())
            px2 = float(self.cal_px2.text())
            rs2 = float(self.cal_rs2.text())
        except ValueError:
            return None
        if px1 == px2:
            return None
        a = (rs2 - rs1) / (px2 - px1)
        b = rs1 - a * px1
        return (a, b)

    def _on_gas_peaks_toggled(self, checked):
        self._show_gas_peaks = checked
        self._update_plot()

    def _on_auto_peaks_toggled(self, checked):
        self._show_auto_peaks = checked
        if checked and self.current_spectrum:
            self._detect_peaks()
        self._update_plot()

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
            unit = "cm-1" if self.current_spectrum and self.current_spectrum.metadata.get("calibration") else "px"
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
        patterns = "光谱文件 (*.txt *.asc *.sif *.tif *.tiff *.bmp);;所有文件 (*.*)"
        files, _ = QFileDialog.getOpenFileNames(self, "选择光谱文件", "", patterns)
        if files:
            self._add_files(files)

    def _on_open_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择包含光谱文件的目录")
        if dir_path:
            supported = set(SUPPORTED_FORMATS.keys())
            dir_p = Path(dir_path)
            all_files = []
            for ext in supported:
                all_files.extend(dir_p.glob(f"*{ext}"))
                all_files.extend(dir_p.glob(f"*{ext.upper()}"))
            all_files = natural_sorted(all_files)
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
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

    def _on_clear_files(self):
        self.file_list.clear()
        self.loaded_spectra.clear()
        self.current_spectrum = None
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
            from raman_tool.readers.tif_reader import _merge_columns, _parse_row_groups
            from raman_tool.safety import check_image_pixels, configure_pillow_limits

            Image = configure_pillow_limits()
            with Image.open(filepath) as img:
                check_image_pixels(*img.size)
                arr = np.array(img)
            if arr.ndim == 3:
                arr = np.mean(arr[:, :, :3], axis=2)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = arr.astype(np.float64, copy=False)

            row_groups = self.img_row_groups.text().strip()
            if row_groups:
                rows = arr.shape[0]
                selected_blocks = []
                for start, end in _parse_row_groups(row_groups):
                    s = max(0, start - 1)
                    e = min(rows, end)
                    if s < e:
                        selected_blocks.append(arr[s:e, :])
                if not selected_blocks:
                    raise ValueError(f"行分组 '{row_groups}' 未匹配到有效范围 (总行数 {rows})")
                arr = np.concatenate(selected_blocks, axis=0)

            try:
                col_merge = max(1, int(self.img_col_merge.text()))
            except ValueError:
                col_merge = 1
            if col_merge > 1:
                arr = _merge_columns(arr, col_merge)

            viewer = ImageViewerWindow(filepath, arr, self)
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
        suffix = Path(filepath).suffix.lower()
        calibration = self._get_calibration()
        row_mode = "sum" if self.row_mode_combo.currentIndex() == 1 else "mean"

        # TIF/BMP/JPG: 从「图像设置」面板读取行/列参数
        if suffix in (".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
            if row_groups is None:
                rg_text = self.img_row_groups.text().strip()
                row_groups = rg_text if rg_text else None
            if col_merge == 1:
                try:
                    col_merge = max(1, int(self.img_col_merge.text()))
                except ValueError:
                    col_merge = 1

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
        if cache_key in self.loaded_spectra:
            self.current_spectrum = self.loaded_spectra[cache_key]
            self.current_file = filepath
            if self._show_auto_peaks:
                self._detect_peaks()
            self._update_plot()
            self.status_bar.showMessage(f"已应用图像设置: {Path(filepath).name}")
            return

        self.load_thread = SpectrumLoadThread(
            filepath,
            row_groups=row_groups,
            col_merge=col_merge,
            calibration=calibration,
            row_mode=row_mode,
            cache_key=cache_key,
        )
        self.load_thread.finished_loading.connect(self._on_spectrum_loaded)
        self.load_thread.error_occurred.connect(self._on_load_error)
        self.load_thread.start()

    def _on_spectrum_loaded(self, spectrum: Spectrum, filepath: str, cache_key: str):
        # 自动基线校正
        if self.auto_baseline_cb.isChecked():
            try:
                baseline_options = self._get_baseline_options()
                spectrum = self._subtract_baseline_with_options(spectrum, baseline_options)
                self._log(f"  已自动执行基线校正 ({self._format_baseline_options(baseline_options)})")
            except Exception as e:
                self._log(f"  自动基线校正失败: {e}")

        self.loaded_spectra[cache_key] = spectrum
        self.current_spectrum = spectrum
        self.current_file = filepath

        # 自动寻峰
        if self._show_auto_peaks:
            self._detect_peaks()

        self._update_plot()
        n = spectrum.size
        xr = f"{spectrum.raman_shift[0]:.1f} - {spectrum.raman_shift[-1]:.1f}"
        yr = f"{spectrum.intensity.min():.1f} - {spectrum.intensity.max():.1f}"

        cal = spectrum.metadata.get("calibration")
        unit = "cm-1" if cal else "px"
        self.status_bar.showMessage(
            f"已载入: {Path(filepath).name} | 数据点: {n} | 范围: {xr} {unit} | 强度: {yr}"
        )
        self._log(f"  数据点: {n}, 拉曼位移: {xr} {unit}, 强度: {yr}")
        if cal:
            a, b = cal
            self.cal_status.setText(f"当前校准: raman = {a:.4f} * pixel + {b:.2f}")

    def _on_load_error(self, error: str):
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
                if mode == "signal":
                    self.snr_signal_range = (xmin, xmax)
                    self.snr_signal_label.setText(f"信号: {xmin:.1f} ~ {xmax:.1f} px")
                else:
                    self.snr_noise_range = (xmin, xmax)
                    self.snr_noise_label.setText(f"噪声: {xmin:.1f} ~ {xmax:.1f} px")
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
        cal = self.current_spectrum.metadata.get("calibration") if self.current_spectrum else None
        unit = "cm-1" if cal else "px"
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
            path = export_spectrum(self.current_spectrum, path, overwrite=True)
            self._log(f"当前光谱数据已导出: {path}")
            self.status_bar.showMessage(f"已导出当前光谱数据: {path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出当前光谱数据失败: {e}")

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
                f"峰中心: {result['peak_center']:.2f} px"
            )
            if result["peak_area"] > 0:
                text += f"\n峰面积: {result['peak_area']:.4f}"

            self.snr_result.setText(text)
            self._log("=== 信噪比计算 ===\n" + text.replace("\n", "\n  "))
            self.control_tabs.setCurrentIndex(0)

        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, "参数错误", f"请检查输入: {e}")

    def _on_baseline(self):
        if self.current_spectrum is None:
            QMessageBox.warning(self, "提示", "请先载入光谱文件")
            return

        try:
            method_idx = self.baseline_method.currentIndex()
            method = "arPLS" if method_idx == 0 else "poly"

            self.status_bar.showMessage(f"正在进行基线校正 ({method})...")

            if method == "arPLS":
                lam_text = self.baseline_lam.text().strip()
                lam = float(lam_text) if lam_text else 1e5
                corrected = subtract_baseline(self.current_spectrum, method="arPLS", lam=lam)
                self._log(f"基线校正完成 (arPLS, lam={lam:.1e})")
            else:
                degree = int(self.baseline_degree.currentText())
                corrected = subtract_baseline(self.current_spectrum, method="poly", degree=degree)
                self._log(f"基线校正完成 (poly, degree={degree})")

            # 直接在原图位置更新光谱
            self.current_spectrum = corrected
            if self._show_auto_peaks:
                self._detect_peaks()
            self._update_plot()
            self.status_bar.showMessage("基线校正完成 — 图表已更新")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"校正失败: {e}")

    def _on_concentration(self):
        if self.current_spectrum is None:
            QMessageBox.warning(self, "提示", "请先载入光谱文件")
            return

        try:
            gas_name = str(self.conc_gas.currentData() or self.conc_gas.currentText())
            window = float(self.conc_window.text())
            strategy = "peak_area" if self.conc_strategy.currentIndex() == 1 else "peak_max"

            result = calculate_concentration(
                self.current_spectrum,
                gas_name=gas_name,
                window=window,
                reference_gas="N2",
                reference_concentration=78.0,
                strategy=strategy,
            )
            all_conc = result.get("all_concentrations", {})
            percentages = all_conc.get("percentages", {})
            peaks = all_conc.get("peaks", {})

            text = (
                f"气体: {result['gas']}\n"
                f"浓度: {result['concentration']:.4f} %\n"
                f"峰中心: {result['peak_center']:.2f} px\n"
                f"峰高: {result['peak_height']:.4f}\n"
                f"峰面积: {result['peak_area']:.4f}\n"
                f"\nO2: {percentages.get('O2', 0.0):.4f}%  I={peaks.get('O2', {}).get('intensity', 0.0):.4f}\n"
                f"N2: {percentages.get('N2', 0.0):.4f}%  I={peaks.get('N2', {}).get('intensity', 0.0):.4f}\n"
                f"CO2: {percentages.get('CO2', 0.0):.4f}%  I={peaks.get('CO2', {}).get('intensity', 0.0):.4f}"
            )
            self.conc_result.setText(text)
            self._log("=== 气体浓度计算 ===\n" + text.replace("\n", "\n  "))

        except ValueError as e:
            QMessageBox.warning(self, "错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {e}")

    def _on_batch(self):
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "提示", "请先添加文件到列表")
            return

        do_baseline = self.batch_baseline_cb.isChecked()
        files = [Path(self.file_list.item(i).data(Qt.UserRole))
                for i in range(self.file_list.count())]
        rg_text = self.img_row_groups.text().strip()
        row_groups = rg_text if rg_text else None
        try:
            col_merge = max(1, int(self.img_col_merge.text()))
        except ValueError:
            col_merge = 1
        calibration = self._get_calibration()
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
            baseline_options=self._get_baseline_options(),
        )
        self.batch_thread.progress_update.connect(self._on_batch_progress)
        self.batch_thread.file_done.connect(self._on_batch_file_done)
        self.batch_thread.all_done.connect(self._on_batch_done)
        self.batch_thread.start()

    def _on_export_all_spectra(self):
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
            col_merge = max(1, int(self.img_col_merge.text()))
        except ValueError:
            col_merge = 1
        calibration = self._get_calibration()
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
            baseline_options=self._get_baseline_options(),
        )
        self.batch_export_thread.progress_update.connect(self._on_batch_progress)
        self.batch_export_thread.file_done.connect(self._on_batch_file_done)
        self.batch_export_thread.all_done.connect(self._on_batch_export_done)
        self.batch_export_thread.start()

    def _on_batch_progress(self, current: int, total: int):
        self.batch_progress.setValue(current)

    def _on_batch_file_done(self, filename: str, success: bool):
        if success:
            self._log(f"  ✓ {filename}")
        else:
            self._log(f"  ✗ {filename}")

    def _on_batch_done(self, ok: int, fail: int, results: list):
        self.batch_progress.setVisible(False)
        self.batch_status.setText(f"完成: {ok} 成功, {fail} 失败")
        self._log(f"批量处理完成: {ok} 成功, {fail} 失败")
        if results:
            dialog = ConcentrationResultDialog(results, self)
            dialog.exec()

    def _on_batch_export_done(self, ok: int, fail: int, output_dir: str):
        self.batch_progress.setVisible(False)
        self.batch_status.setText(f"导出完成: {ok} 成功, {fail} 失败")
        self._log(f"批量导出光谱数据完成: {ok} 成功, {fail} 失败, 输出目录: {output_dir}")
        QMessageBox.information(self, "完成", f"光谱数据导出完成\n成功: {ok}\n失败: {fail}\n目录: {output_dir}")

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
