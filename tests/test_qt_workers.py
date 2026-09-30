"""Exercise real queued Qt signals and overlapping worker lifetimes offscreen."""

import os
import json
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListWidgetItem, QMessageBox, QFileDialog

from raman_tool.history import SpectrumSession
from raman_tool.models import Spectrum
from raman_tool.qt_gui import RamanQtGUI
from raman_tool.qt_dialogs import ConcentrationResultDialog, ImageViewerWindow
from raman_tool import qt_workers

_APP = QApplication.instance() or QApplication([])


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        _APP.processEvents()
        QTest.qWait(5)
    _APP.processEvents()
    assert predicate(), "Qt task did not reach its expected state"


def spectrum(name="test.txt"):
    x = np.arange(20, dtype=float)
    return Spectrum(x, 10 + x + np.sin(x), filename=name)


@pytest.fixture
def window(monkeypatch):
    win = RamanQtGUI()
    # These tests concern state/threads; plotting itself is covered in the Qt preset tests.
    monkeypatch.setattr(win, "_update_plot", lambda: None)
    monkeypatch.setattr(win, "_detect_peaks", lambda: None)
    yield win
    win.close()
    wait_until(lambda: not win._workers.workers)
    _APP.processEvents()


def set_current(win):
    win.current_session = SpectrumSession(spectrum())
    win.current_spectrum = win.current_session.current
    win.baseline_method.setCurrentIndex(1)
    win.baseline_degree.setCurrentText("1")


def test_manual_baseline_is_background_and_rejects_duplicate(window, monkeypatch):
    set_current(window)
    entered, release = threading.Event(), threading.Event()
    thread_ids = []
    real_subtract = qt_workers.subtract_baseline

    def blocked(spec, **options):
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return real_subtract(spec, **options)

    monkeypatch.setattr(qt_workers, "subtract_baseline", blocked)
    try:
        window._on_baseline()
        wait_until(entered.is_set)
        first_worker = window._baseline_worker
        window._on_baseline()
        window._on_undo_processing()
        assert window._baseline_worker is first_worker
        assert len(window._workers.workers) == 1
        assert window.current_session.state_count == 1
        assert not window.baseline_apply_btn.isEnabled()
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        wait_until(lambda: bool(ticks))
        assert thread_ids[0] != threading.get_ident()
    finally:
        release.set()
    wait_until(lambda: not window._workers.workers)
    assert window.current_session.state_count == 2
    assert window.baseline_apply_btn.isEnabled()
    assert window.baseline_undo_btn.isEnabled()


def test_manual_result_cannot_overwrite_newer_session_revision(window, monkeypatch):
    set_current(window)
    entered, release = threading.Event(), threading.Event()

    def blocked(spec, **options):
        entered.set()
        assert release.wait(5)
        return Spectrum(spec.raman_shift, np.zeros(spec.size), filename=spec.filename)

    monkeypatch.setattr(qt_workers, "subtract_baseline", blocked)
    try:
        window._on_baseline()
        wait_until(entered.is_set)
        replacement = Spectrum(np.arange(20), np.full(20, 42.0))
        window.current_session.apply(replacement, "other change")
        revision = window.current_session.revision
    finally:
        release.set()
    wait_until(lambda: not window._workers.workers)
    assert window.current_session.revision == revision
    assert np.all(window.current_session.current.intensity == 42)


def test_fast_file_switch_retains_old_worker_and_ignores_stale_result(window, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    window.auto_baseline_cb.setChecked(False)

    def read(path, **options):
        if path.name == "first.txt":
            entered.set()
            assert release.wait(5)
        return spectrum(path.name)

    monkeypatch.setattr(qt_workers, "load_spectrum", read)
    try:
        window._load_file("first.txt")
        wait_until(entered.is_set)
        first_worker = window.load_thread
        window._load_file("second.txt")
        assert first_worker in window._workers.workers
        wait_until(lambda: window.current_file == "second.txt")
        assert first_worker in window._workers.workers
    finally:
        release.set()
    wait_until(lambda: not window._workers.workers)
    assert window.current_file == "second.txt"
    assert window.current_spectrum.filename == "second.txt"
    assert len(window.spectrum_sessions) == 1


def test_auto_baseline_uses_options_captured_before_read(window, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    observed = []
    main_thread_id = threading.get_ident()
    window.baseline_method.setCurrentIndex(1)
    window.baseline_degree.setCurrentText("1")
    real_subtract = qt_workers.subtract_baseline

    def read(path, **options):
        entered.set()
        assert release.wait(5)
        return spectrum(path.name)

    def baseline(spec, **options):
        observed.append((dict(options), threading.get_ident()))
        return real_subtract(spec, **options)

    monkeypatch.setattr(qt_workers, "load_spectrum", read)
    monkeypatch.setattr(qt_workers, "subtract_baseline", baseline)
    try:
        window._load_file("auto.txt")
        wait_until(entered.is_set)
        window.baseline_degree.setCurrentText("5")
        window.auto_baseline_cb.setChecked(False)
    finally:
        release.set()
    wait_until(lambda: not window._workers.workers)
    assert observed[0][0] == {"method": "poly", "degree": 1}
    assert observed[0][1] != main_thread_id
    assert window.current_session.state_count == 2
    assert window.current_session.audit_log[-1]["parameters"]["degree"] == 1


def test_clear_files_invalidates_outstanding_load(window, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    window.auto_baseline_cb.setChecked(False)

    def read(path, **options):
        entered.set()
        assert release.wait(5)
        return spectrum(path.name)

    monkeypatch.setattr(qt_workers, "load_spectrum", read)
    try:
        window._load_file("old.txt")
        wait_until(entered.is_set)
        window._on_clear_files()
    finally:
        release.set()
    wait_until(lambda: not window._workers.workers)
    assert window.current_spectrum is None
    assert not window.spectrum_sessions


def test_close_waits_for_worker_without_blocking_event_loop(window, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    window.auto_baseline_cb.setChecked(False)

    def read(path, **options):
        entered.set()
        assert release.wait(5)
        return spectrum(path.name)

    monkeypatch.setattr(qt_workers, "load_spectrum", read)
    try:
        window.show()
        window._load_file("slow.txt")
        wait_until(entered.is_set)
        start = time.monotonic()
        assert not window.close()
        assert time.monotonic() - start < 0.5
        assert window.isVisible()
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        wait_until(lambda: bool(ticks))
    finally:
        release.set()
    wait_until(lambda: not window._workers.workers and not window.isVisible())
    assert window.current_spectrum is None


def test_batch_duplicate_cancel_and_failure_reason(window, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    for name in ("broken.txt", "not_started.txt"):
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, name)
        window.file_list.addItem(item)
    calls = []

    def read(path, **options):
        calls.append(path.name)
        entered.set()
        assert release.wait(5)
        raise ValueError("missing calibration")

    monkeypatch.setattr(qt_workers, "load_spectrum", read)
    try:
        window._on_batch()
        wait_until(entered.is_set)
        original = window._batch_worker
        window._on_batch()
        window._on_export_all_spectra()
        assert window._batch_worker is original
        assert len(window._workers.workers) == 1
        assert not window.batch_start_btn.isEnabled()
        window._on_cancel_batch()
    finally:
        release.set()
    wait_until(lambda: not window._workers.workers)
    assert calls == ["broken.txt"]
    assert window.last_batch_failures[0]["filename"] == "broken.txt"
    assert "missing calibration" in window.last_batch_failures[0]["reason"]
    assert window.batch_start_btn.isEnabled()
    assert "已取消" in window.batch_status.text()


def test_export_directory_failure_is_reported_for_each_file(tmp_path):
    invalid_dir = tmp_path / "regular_file"
    invalid_dir.write_text("not a directory", encoding="utf-8")
    worker = qt_workers.BatchExportThread([Path("one.txt"), Path("two.txt")], invalid_dir, ".txt")
    reports = []
    worker.all_done.connect(lambda *args: reports.append(args))
    worker.run()
    assert reports[0][:2] == (0, 2)
    assert [failure["filename"] for failure in worker.failures] == ["one.txt", "two.txt"]
    assert all("FileExistsError" in failure["reason"] for failure in worker.failures)


@pytest.mark.parametrize("value", ["0", "abc", "nan", "-2"])
def test_invalid_image_merge_rejected_before_worker_dispatch(window, monkeypatch, value):
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    window.img_col_merge.setText(value)
    window._load_file("image.tif")
    assert messages
    assert not window._workers.workers


def test_incomplete_calibration_rejected_instead_of_silently_ignored(window, monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    window.cal_px1.setText("10")
    window._load_file("image.tif")
    assert "校准" in messages[0]
    assert not window._workers.workers


def test_auto_baseline_failure_cannot_cache_uncorrected_result(window, monkeypatch):
    monkeypatch.setattr(qt_workers, "load_spectrum", lambda *args, **kwargs: spectrum())

    def failed(*args, **kwargs):
        raise ValueError("baseline failed")

    monkeypatch.setattr(qt_workers, "subtract_baseline", failed)
    window._load_file("auto.txt")
    wait_until(lambda: not window._workers.workers)
    assert window.current_spectrum is None
    assert not window.spectrum_sessions
    assert "auto.txt: baseline failed" in window.log_widget.toPlainText()


def test_capacity_warning_blocks_growth_but_not_undo(window, monkeypatch):
    set_current(window)
    window.current_session.apply(spectrum(), "changed")
    window.spectrum_sessions.capacity_warning = "cache full"
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    window._on_baseline()
    window._load_file("next.txt")
    assert len(messages) == 2
    assert not window._workers.workers
    window._on_undo_processing()
    assert window.current_session.cursor == 0


def test_failed_exports_write_machine_readable_file_report(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("invalid unit")

    monkeypatch.setattr(qt_workers, "load_spectrum", failed)
    worker = qt_workers.BatchExportThread(["broken.txt"], tmp_path, ".txt")
    worker.run()
    payload = json.loads(worker.failure_report_path.read_text(encoding="utf-8"))
    assert payload["failures"][0]["filename"] == "broken.txt"
    assert "invalid unit" in payload["failures"][0]["reason"]
    assert payload["cancelled"] is False


def test_batch_report_exports_failure_reasons_and_result_meaning(tmp_path, monkeypatch):
    path = tmp_path / "report.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    dialog = ConcentrationResultDialog([], failures=[{"filename": "broken.txt", "reason": "bad calibration"}])
    try:
        dialog._save_results_txt()
        text = path.read_text(encoding="utf-8")
        assert "weighted_signal_fraction" in text
        assert "broken.txt\tbad calibration" in text
    finally:
        dialog.close()


def test_image_viewer_uses_shared_row_selection_and_binning(window, tmp_path):
    from PIL import Image

    path = tmp_path / "image.tif"
    array = np.arange(24, dtype=np.uint8).reshape(4, 6)
    Image.fromarray(array).save(path)
    window.img_row_groups.setText("2-3")
    window.img_col_merge.setText("2")
    window._open_image_viewer(str(path))
    viewer = window.findChildren(ImageViewerWindow)[0]
    assert np.array_equal(viewer._img_array, array[1:3].reshape(2, 3, 2).mean(axis=2))


def test_changing_processing_state_clears_obsolete_analysis_results(window):
    set_current(window)
    window.snr_result.setText("old SNR")
    window.conc_result.setText("old percentage")
    window._refresh_from_current_session()
    assert not window.snr_result.text()
    assert not window.conc_result.text()


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_preset_uses_same_positive_finite_parameter_rules(window, value):
    window.baseline_lam.setText(value)
    with pytest.raises(ValueError):
        window._get_baseline_options()
    with pytest.raises(ValueError):
        window._collect_workflow_preset()


def test_preset_accepts_same_small_positive_values_as_calculation(window):
    window.baseline_lam.setText("0.25")
    window.conc_window.setText("0.05")
    preset = window._collect_workflow_preset()
    assert preset["baseline_lam"] == 0.25
    assert preset["concentration_window"] == 0.05


@pytest.mark.parametrize("degree", [0, 8])
def test_qt_applies_valid_preset_degree_without_substitution(window, degree):
    window._apply_workflow_preset({"baseline_method": "poly", "baseline_degree": degree})
    assert window._get_baseline_options() == {"method": "poly", "degree": degree}
    assert window._collect_workflow_preset()["baseline_degree"] == degree
