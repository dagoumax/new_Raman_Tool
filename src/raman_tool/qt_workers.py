"""Background tasks and their lifetime management, independent of window widgets."""

from copy import deepcopy
import json
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from raman_tool.exporters import export_spectrum, unique_path
from raman_tool.history import SpectrumSession, copy_spectrum
from raman_tool.gas_library import get_gas_library
from raman_tool.processing import calculate_gas_concentrations, subtract_baseline
from raman_tool.workflows import load_spectrum


class WorkerManager(QObject):
    """Retain every running thread and release it only after Qt reports finished.

    Interruption is cooperative between file reads and numerical operations. The
    window defers closing while tasks drain instead of blocking its event loop.
    """

    worker_finished = Signal(object)
    idle = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.workers: set[QThread] = set()

    def start(self, worker: QThread):
        worker.setParent(self)
        self.workers.add(worker)
        worker.finished.connect(self._finished)
        worker.start()

    @Slot()
    def _finished(self):
        worker = self.sender()
        self.workers.discard(worker)
        self.worker_finished.emit(worker)
        worker.deleteLater()
        if not self.workers:
            self.idle.emit()

    def interrupt_all(self):
        for worker in tuple(self.workers):
            worker.requestInterruption()


class SpectrumLoadThread(QThread):
    finished_loading = Signal(object, str, str)  # session, filepath, cache key
    error_occurred = Signal(str)

    def __init__(self, filepath, row_groups=None, col_merge=1, calibration=None,
                 row_mode="mean", cache_key=None, baseline_options=None, request_id=0):
        super().__init__()
        self.filepath = str(filepath)
        self.cache_key = cache_key or self.filepath
        self.request_id = request_id
        self.read_options = dict(row_groups=row_groups, col_merge=col_merge,
                                 calibration=calibration, row_mode=row_mode)
        self.baseline_options = deepcopy(baseline_options)

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            spectrum = load_spectrum(Path(self.filepath), **self.read_options)
            if self.isInterruptionRequested():
                return
            session = SpectrumSession(spectrum)
            if self.baseline_options is not None:
                corrected = subtract_baseline(spectrum, **self.baseline_options)
                if self.isInterruptionRequested():
                    return
                session.apply(corrected, "基线校正", self.baseline_options, "自动加载")
            if not self.isInterruptionRequested():
                self.finished_loading.emit(session, self.filepath, self.cache_key)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error_occurred.emit(f"{Path(self.filepath).name}: {exc}")


class BaselineThread(QThread):
    completed = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, session: SpectrumSession, options: dict):
        super().__init__()
        self.session = session
        self.revision = session.revision
        self.spectrum = copy_spectrum(session.current)
        self.options = deepcopy(options)

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            corrected = subtract_baseline(self.spectrum, **self.options)
            if not self.isInterruptionRequested():
                self.completed.emit(corrected)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error_occurred.emit(str(exc))


class BatchThread(QThread):
    progress_update = Signal(int, int)
    file_done = Signal(str, bool, str)

    def __init__(self, files, do_baseline=False, row_groups=None, col_merge=1,
                 calibration=None, row_mode="mean", baseline_options=None):
        super().__init__()
        self.files = [Path(path) for path in files]
        self.do_baseline = do_baseline
        self.read_options = dict(row_groups=row_groups, col_merge=col_merge,
                                 calibration=calibration, row_mode=row_mode)
        self.baseline_options = deepcopy(baseline_options or {"method": "arPLS"})
        self.failures: list[dict[str, str]] = []
        self.cancelled = False

    def _read(self, path):
        spectrum = load_spectrum(path, **self.read_options)
        if self.do_baseline and not self.isInterruptionRequested():
            spectrum = subtract_baseline(spectrum, **self.baseline_options)
        return spectrum

    def _failure(self, path, exc):
        reason = f"{type(exc).__name__}: {exc}"
        self.failures.append({"filename": path.name, "path": str(path), "reason": reason})
        self.file_done.emit(path.name, False, reason)


class BatchProcessThread(BatchThread):
    all_done = Signal(int, int, object)

    def __init__(self, files, do_baseline, row_groups=None, col_merge=1,
                 calibration=None, row_mode="mean", strategy="peak_max", baseline_options=None,
                 windows=None):
        super().__init__(files, do_baseline, row_groups, col_merge, calibration,
                         row_mode, baseline_options)
        self.strategy = strategy
        self.library = deepcopy(get_gas_library())
        self.windows = deepcopy(windows)

    def run(self):
        results = []
        for index, path in enumerate(self.files):
            if self.isInterruptionRequested():
                break
            try:
                spectrum = self._read(path)
                if self.isInterruptionRequested():
                    break
                conc = calculate_gas_concentrations(
                    spectrum, strategy=self.strategy, library=self.library, windows=self.windows,
                )
                if self.isInterruptionRequested():
                    break
                row = {"index": index + 1, "filename": path.name,
                       "gases": list(conc["percentages"]), **conc}
                results.append(row)
                self.file_done.emit(path.name, True, "")
            except Exception as exc:
                self._failure(path, exc)
            self.progress_update.emit(index + 1, len(self.files))
        self.cancelled = self.isInterruptionRequested()
        self.all_done.emit(len(results), len(self.failures), results)


class BatchExportThread(BatchThread):
    all_done = Signal(int, int, str)

    def __init__(self, files, output_dir, suffix, do_baseline=False, row_groups=None,
                 col_merge=1, calibration=None, row_mode="mean", baseline_options=None):
        super().__init__(files, do_baseline, row_groups, col_merge, calibration,
                         row_mode, baseline_options)
        self.output_dir = Path(output_dir)
        self.suffix = suffix
        self.failure_report_path = None
        self.report_error = ""

    def run(self):
        ok = 0
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            for path in self.files:
                self._failure(path, exc)
            self.all_done.emit(0, len(self.failures), str(self.output_dir))
            return
        for index, path in enumerate(self.files):
            if self.isInterruptionRequested():
                break
            try:
                spectrum = self._read(path)
                if self.isInterruptionRequested():
                    break
                out_path = unique_path(self.output_dir / path.with_suffix(self.suffix).name)
                export_spectrum(spectrum, out_path)
                ok += 1
                self.file_done.emit(out_path.name, True, "")
            except Exception as exc:
                self._failure(path, exc)
            self.progress_update.emit(index + 1, len(self.files))
        self.cancelled = self.isInterruptionRequested()
        if self.failures:
            try:
                report = unique_path(self.output_dir / "export_failures.json")
                report.write_text(json.dumps({"failures": self.failures, "cancelled": self.cancelled},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
                self.failure_report_path = report
            except OSError as exc:
                self.report_error = str(exc)
        self.all_done.emit(ok, len(self.failures), str(self.output_dir))
