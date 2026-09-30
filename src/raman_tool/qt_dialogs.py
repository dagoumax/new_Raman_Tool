"""Settings dialogs, batch reports and image viewers for the Qt application."""

from pathlib import Path

import numpy as np

# 在导入任何 matplotlib 相关模块前设好后端
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTextEdit, QFileDialog, QMessageBox, QCheckBox, QStatusBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
    QFormLayout, QSpinBox, QDialogButtonBox,
)
from PySide6.QtCore import Qt

from raman_tool.config import get_config, default_config_path
from raman_tool.gas_library import DEFAULT_GAS_LIBRARY, default_gas_library_path, load_gas_library


class GasLibraryDialog(QDialog):
    COLUMNS = ["key", "name", "center", "half_width", "coefficient", "color", "enabled", "quantitative"]
    HEADERS = ["气体", "名称", "峰位 (cm⁻¹)", "半窗宽 (cm⁻¹)", "系数", "颜色", "启用", "定量"]

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

    def closeEvent(self, event):
        plt.close(self._fig)
        super().closeEvent(event)

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

    def __init__(self, results: list[dict], parent=None, failures: list[dict] | None = None):
        super().__init__(parent)
        self.setWindowTitle("批量加权信号占比")
        self.resize(980, 620)
        self._results = results
        self._failures = failures or []
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
        self._ax.set_ylabel("加权信号占比 (%)")
        self._ax.set_title(" / ".join(self._gases) + " 信号占比" if self._gases else "信号占比")
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
        self._warnings = list(dict.fromkeys(warning for row in results for warning in row.get("warnings", [])))
        note = QLabel("\n".join(self._warnings) or "结果为启用定量气体的加权信号占比；需经标准样品校准后才能解释为实际浓度。")
        note.setWordWrap(True)
        chart_layout.addWidget(note)
        if self._failures:
            failures_text = QTextEdit()
            failures_text.setReadOnly(True)
            failures_text.setMaximumHeight(110)
            failures_text.setPlainText("失败文件：\n" + "\n".join(
                f"{failure['filename']}: {failure['reason']}" for failure in self._failures
            ))
            chart_layout.addWidget(failures_text)

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

    def closeEvent(self, event):
        plt.close(self._fig)
        super().closeEvent(event)

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
        lines = ["# result_kind: weighted_signal_fraction",
                 "# 结果为启用定量气体的加权信号占比；实际浓度需标准样品校准验证。"]
        lines.extend(f"# {warning}" for warning in self._warnings)
        lines.append("\t".join(header))
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
        if self._failures:
            lines.extend(["", "failed_files", "filename\treason"])
            for failure in self._failures:
                reason = failure["reason"].replace("\n", " ").replace("\t", " ")
                lines.append(f"{failure['filename']}\t{reason}")
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
        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            self._save_status.setText(f"保存失败: {exc}")
            return
        self._save_status.setText(f"已保存:\n{path}")
