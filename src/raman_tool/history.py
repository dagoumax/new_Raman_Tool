"""Bounded reversible snapshots, complete audit records, and a spillable session cache."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import tempfile
from typing import Any
import weakref

import numpy as np

from raman_tool.config import get_config
from raman_tool.models import Spectrum


def copy_spectrum(spectrum: Spectrum) -> Spectrum:
    return Spectrum(
        spectrum.raman_shift.copy(), spectrum.intensity.copy(),
        filename=spectrum.filename, metadata=spectrum.metadata.copy(),
    )


def _freeze_arrays(value: Any) -> None:
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for child in value.values():
            _freeze_arrays(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _freeze_arrays(child)


def _snapshot_metadata(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        snapshot = value.copy() if value.flags.writeable else value
        snapshot.setflags(write=False)
        return snapshot
    if isinstance(value, dict):
        return {key: _snapshot_metadata(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_snapshot_metadata(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_metadata(child) for child in value)
    return deepcopy(value)


def _array_bytes(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, Spectrum):
        return sum(_array_bytes(v, seen) for v in
                   (value.raman_shift, value.intensity, value.metadata))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_array_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(v, seen) for v in value)
    return 0


@dataclass
class SpectrumSession:
    """Keep the original snapshot and newest undo states within a memory budget.

    State IDs never repeat, including after undo branches and pruning. Audit
    records are append-only in a temporary stream; only a bounded tail stays in
    memory. The stream belongs to this session and is not a durable project file.
    """

    source: Spectrum
    max_states: int | None = None
    max_bytes: int | None = None
    display_records: int | None = None
    _states: list[Spectrum] = field(init=False, repr=False)
    _state_ids: list[int] = field(init=False, repr=False)
    _cursor: int = field(init=False, default=0)
    revision: int = field(init=False, default=0)
    discarded_state_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        limits = get_config()["runtime"]
        self.max_states = int(self.max_states if self.max_states is not None else limits["max_history_states"])
        self.max_bytes = int(self.max_bytes if self.max_bytes is not None else limits["max_history_mb"] * 1024**2)
        self.display_records = int(self.display_records if self.display_records is not None else limits["history_display_records"])
        if self.max_states < 2 or self.max_bytes < 1 or self.display_records < 1:
            raise ValueError("History limits require at least two states and positive capacities")
        raw = copy_spectrum(self.source)
        raw.metadata = deepcopy(raw.metadata)
        if _array_bytes(raw) > self.max_bytes:
            raise ValueError("原始光谱超过会话内存限制，请提高 runtime.max_history_mb 或减少导入数据")
        _freeze_arrays((raw.raman_shift, raw.intensity, raw.metadata))
        self._states = [raw]
        self._state_ids = [0]
        self.source = raw
        self._next_state_id = 1
        self._audit_count = 0
        self._audit_stream = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self._audit_tail = deque(maxlen=self.display_records)
        self._capacity_callbacks: list[weakref.WeakMethod] = []

    @property
    def current(self) -> Spectrum:
        return self._states[self._cursor]

    @property
    def raw(self) -> Spectrum:
        result = copy_spectrum(self.source)
        result.metadata = deepcopy(self.source.metadata)
        return result

    @property
    def cursor(self) -> int:
        return self._state_ids[self._cursor]

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def memory_bytes(self) -> int:
        return _array_bytes(self._states)

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._states) - 1

    @property
    def is_raw_current(self) -> bool:
        return np.array_equal(self.current.intensity, self.source.intensity)

    @property
    def audit_count(self) -> int:
        return self._audit_count

    @property
    def recent_audit_log(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._audit_tail))

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        self._audit_stream.seek(0)
        records = [json.loads(line) for line in self._audit_stream]
        self._audit_stream.seek(0, 2)
        return records

    def add_capacity_listener(self, callback) -> None:
        reference = weakref.WeakMethod(callback)
        if reference not in self._capacity_callbacks:
            self._capacity_callbacks.append(reference)

    def _notify_capacity(self) -> None:
        for reference in self._capacity_callbacks[:]:
            callback = reference()
            if callback is None:
                self._capacity_callbacks.remove(reference)
            else:
                callback(self)

    def apply(self, spectrum: Spectrum, action: str,
              parameters: dict[str, Any] | None = None, source: str = "手动") -> Spectrum:
        if spectrum.x_unit != self.current.x_unit or not np.array_equal(
            spectrum.raman_shift, self.current.raman_shift
        ):
            raise ValueError("Processed spectrum axis does not match the current spectrum")
        candidate = copy_spectrum(spectrum)
        candidate.metadata = _snapshot_metadata(spectrum.metadata)
        _freeze_arrays((candidate.raman_shift, candidate.intensity))
        if _array_bytes([self.source, candidate]) > self.max_bytes:
            raise ValueError("处理结果与原始数据超过会话内存限制，请提高 runtime.max_history_mb")
        # Verify serializability before changing state or the audit stream.
        json.dumps(parameters or {}, ensure_ascii=False, allow_nan=False)
        discarded = len(self._states) - self._cursor - 1
        from_state = self.cursor
        del self._states[self._cursor + 1:]
        del self._state_ids[self._cursor + 1:]
        self._states.append(candidate)
        self._state_ids.append(self._next_state_id)
        self._next_state_id += 1
        self._cursor = len(self._states) - 1
        pruned = 0
        while len(self._states) > self.max_states or self.memory_bytes > self.max_bytes:
            del self._states[1]
            del self._state_ids[1]
            self._cursor -= 1
            pruned += 1
        self.discarded_state_count += pruned
        self._record(action, parameters or {}, source, from_state, self.cursor,
                     discarded_redo_states=discarded, pruned_undo_states=pruned)
        self._notify_capacity()
        return self.current

    def undo(self) -> Spectrum:
        if not self.can_undo:
            raise IndexError("No processing step to undo")
        previous = self.cursor
        self._cursor -= 1
        self._record("撤销", {}, "用户", previous, self.cursor)
        self._notify_capacity()
        return self.current

    def redo(self) -> Spectrum:
        if not self.can_redo:
            raise IndexError("No processing step to redo")
        previous = self.cursor
        self._cursor += 1
        self._record("重做", {}, "用户", previous, self.cursor)
        self._notify_capacity()
        return self.current

    def restore_raw(self) -> Spectrum:
        if self.is_raw_current:
            return self.current
        return self.apply(self.source, "恢复原始数据", {}, "用户")

    def spectrum_for_export(self) -> Spectrum:
        spectrum = copy_spectrum(self.current)
        # Keep imported provenance when extending an exported spectrum.
        inherited = deepcopy(self.source.metadata.get("processing_history", []))
        spectrum.metadata.update({
            "raw_data_preserved": True, "processing_state": self.cursor,
            "processing_history": inherited + self.audit_log,
            "pruned_undo_states": self.discarded_state_count,
        })
        return spectrum

    def _record(self, action, parameters, source, from_state, to_state,
                discarded_redo_states=0, pruned_undo_states=0) -> None:
        record = {
            "sequence": self._audit_count + 1,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "action": action, "parameters": deepcopy(parameters), "source": source,
            "from_state": from_state, "to_state": to_state,
            "discarded_redo_states": discarded_redo_states,
            "pruned_undo_states": pruned_undo_states,
        }
        self._audit_stream.seek(0, 2)
        self._audit_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._audit_stream.flush()
        self._audit_tail.append(record)
        self._audit_count += 1
        self.revision += 1

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_serialized_audit"] = self.audit_log
        state.pop("_audit_stream")
        state.pop("_capacity_callbacks")
        return state

    def __setstate__(self, state):
        records = state.pop("_serialized_audit")
        self.__dict__.update(state)
        self._capacity_callbacks = []
        self._audit_stream = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        for record in records:
            self._audit_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        for spectrum in self._states:
            _freeze_arrays((spectrum.raman_shift, spectrum.intensity, spectrum.metadata))


class SessionCache(MutableMapping):
    """LRU resident budget with lossless private disk spill for evicted sessions.

    Pickles are generated and consumed only in this instance's private temporary
    directory. Never use this implementation to open external project files.
    Cache spill failures retain the live session and surface a capacity warning.
    """

    def __init__(self, max_sessions=None, max_bytes=None, max_disk_bytes=None):
        limits = get_config()["runtime"]
        self.max_sessions = max_sessions if max_sessions is not None else limits["max_cache_sessions"]
        self.max_bytes = max_bytes if max_bytes is not None else limits["max_cache_mb"] * 1024**2
        self.max_disk_bytes = max_disk_bytes if max_disk_bytes is not None else limits["max_cache_disk_mb"] * 1024**2
        if min(self.max_sessions, self.max_bytes, self.max_disk_bytes) < 1:
            raise ValueError("Cache limits must be positive")
        self._resident = OrderedDict()
        self._spilled: dict[str, Path] = {}
        self._live = weakref.WeakValueDictionary()
        self._directory = tempfile.TemporaryDirectory(prefix="raman-sessions-")
        self._serial = 0
        self.capacity_warning = ""

    @property
    def memory_bytes(self):
        return sum(session.memory_bytes for session in self._resident.values())

    @property
    def resident_count(self):
        return len(self._resident)

    @property
    def disk_bytes(self):
        return sum(path.stat().st_size for path in self._spilled.values())

    def __len__(self):
        return len(self._resident) + len(self._spilled)

    def __iter__(self):
        return iter(list(self._resident) + list(self._spilled))

    def __contains__(self, key):
        return key in self._resident or key in self._spilled

    def __getitem__(self, key):
        if key in self._resident:
            self._resident.move_to_end(key)
            return self._resident[key]
        path = self._spilled[key]
        session = self._live.get(key)
        if session is None:
            with path.open("rb") as stream:
                session = pickle.load(stream)
        path.unlink()
        del self._spilled[key]
        self._resident[key] = session
        self._live[key] = session
        session.add_capacity_listener(self._session_changed)
        self.enforce_limits()
        return session

    def __setitem__(self, key, value):
        if key in self._spilled:
            self._spilled.pop(key).unlink()
        self._resident[key] = value
        self._live[key] = value
        self._resident.move_to_end(key)
        value.add_capacity_listener(self._session_changed)
        self.enforce_limits()

    def __delitem__(self, key):
        if key in self._resident:
            del self._resident[key]
        else:
            self._spilled.pop(key).unlink()
        self._live.pop(key, None)
        self.enforce_limits()

    def clear(self):
        self._resident.clear()
        self._live.clear()
        for path in self._spilled.values():
            path.unlink(missing_ok=True)
        self._spilled.clear()
        self.capacity_warning = ""

    def close(self):
        self.clear()
        self._directory.cleanup()

    def _session_changed(self, session):
        for key, value in list(self._live.items()):
            if value is session:
                if key in self._spilled:
                    self._spilled.pop(key).unlink()
                    self._resident[key] = session
                self._resident.move_to_end(key)
                break
        self.enforce_limits()

    def enforce_limits(self):
        self.capacity_warning = ""
        while self.resident_count > self.max_sessions or self.memory_bytes > self.max_bytes:
            key, session = next(iter(self._resident.items()))
            self._serial += 1
            path = Path(self._directory.name) / f"{self._serial}.cache"
            try:
                with path.open("wb") as stream:
                    pickle.dump(session, stream, protocol=pickle.HIGHEST_PROTOCOL)
                if self.disk_bytes + path.stat().st_size > self.max_disk_bytes:
                    raise OSError("会话磁盘缓存已满；请导出处理结果并清理文件列表")
            except (OSError, pickle.PickleError) as exc:
                path.unlink(missing_ok=True)
                self.capacity_warning = str(exc)
                # Preserve edited data instead of dropping a session silently.
                break
            self._spilled[key] = path
            del self._resident[key]
