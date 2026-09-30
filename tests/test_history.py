import numpy as np
import pytest

from raman_tool.history import SpectrumSession, SessionCache
from raman_tool.models import Spectrum


def make_spectrum(offset: float = 0.0) -> Spectrum:
    return Spectrum(
        np.array([100.0, 200.0, 300.0]),
        np.array([10.0, 20.0, 15.0]) + offset,
        filename="sample.txt",
    )


def test_session_preserves_independent_raw_snapshot():
    source = make_spectrum()
    session = SpectrumSession(source)

    source.intensity[0] = 999.0

    assert session.raw.intensity[0] == 10.0
    assert session.current.intensity[0] == 10.0


def test_session_allows_multiple_processing_steps_and_undo_redo():
    session = SpectrumSession(make_spectrum())
    session.apply(make_spectrum(-1), "基线校正", {"method": "poly", "degree": 2})
    session.apply(make_spectrum(-2), "基线校正", {"method": "arPLS", "lam": 100000})

    assert session.state_count == 3
    assert session.cursor == 2
    assert np.allclose(session.raw.intensity, [10.0, 20.0, 15.0])

    assert np.allclose(session.undo().intensity, [9.0, 19.0, 14.0])
    assert np.allclose(session.redo().intensity, [8.0, 18.0, 13.0])
    assert [record["action"] for record in session.audit_log] == [
        "基线校正",
        "基线校正",
        "撤销",
        "重做",
    ]


def test_new_step_after_undo_keeps_audit_and_discards_only_redo_state():
    session = SpectrumSession(make_spectrum())
    session.apply(make_spectrum(-1), "step 1")
    session.apply(make_spectrum(-2), "step 2")
    session.undo()

    session.apply(make_spectrum(-3), "replacement")

    assert not session.can_redo
    assert session.audit_log[-1]["discarded_redo_states"] == 1
    assert any(record["action"] == "step 2" for record in session.audit_log)


def test_restore_raw_is_reversible_and_export_contains_history():
    session = SpectrumSession(make_spectrum())
    session.apply(make_spectrum(-2), "基线校正", {"method": "poly"})

    restored = session.restore_raw()
    exported = session.spectrum_for_export()

    assert np.array_equal(restored.intensity, session.raw.intensity)
    assert session.can_undo
    assert exported.metadata["raw_data_preserved"] is True
    assert exported.metadata["processing_history"][-1]["action"] == "恢复原始数据"


def test_history_prunes_snapshots_but_keeps_raw_and_complete_audit():
    session = SpectrumSession(make_spectrum(), max_states=3, display_records=2)
    for i in range(1, 7):
        session.apply(make_spectrum(-i), f"step {i}")
    assert session.state_count == 3
    assert session.discarded_state_count == 4
    assert session.audit_count == 6
    assert len(session.recent_audit_log) == 2
    assert len(session.spectrum_for_export().metadata["processing_history"]) == 6
    assert session.cursor == 6
    session.undo()
    assert session.cursor == 5
    session.undo()
    assert session.cursor == 0
    assert np.array_equal(session.current.intensity, make_spectrum().intensity)
    session.apply(make_spectrum(-7), "new branch")
    assert session.cursor == 7
    assert session.revision == 9


def test_history_byte_limit_and_readonly_raw():
    session = SpectrumSession(make_spectrum(), max_bytes=144)
    for i in range(6):
        session.apply(make_spectrum(-i), "step")
    assert session.memory_bytes <= 144
    with pytest.raises(ValueError):
        session.current.intensity[0] = 200
    with pytest.raises(ValueError):
        session.source.intensity[0] = 200
    mutable_export = session.raw
    mutable_export.intensity[0] = 200
    assert session.raw.intensity[0] == 10


def test_history_rejects_result_that_cannot_fit_with_raw_without_losing_state():
    session = SpectrumSession(make_spectrum(), max_bytes=64)
    with pytest.raises(ValueError, match="内存"):
        session.apply(make_spectrum(-1), "step")
    assert session.state_count == 1
    assert session.audit_count == 0


def test_cache_spills_and_restores_history_and_undo():
    cache = SessionCache(max_sessions=1)
    first = SpectrumSession(make_spectrum())
    first.apply(make_spectrum(-1), "correction")
    cache["first"] = first
    cache["second"] = SpectrumSession(make_spectrum(5))
    assert cache.resident_count == 1
    assert len(cache) == 2
    assert cache.disk_bytes > 0
    del first
    restored = cache["first"]
    assert restored.current.intensity[0] == 9
    assert restored.audit_count == 1
    assert restored.can_undo
    restored.undo()
    assert restored.current.intensity[0] == 10
    cache.close()


def test_cache_preserves_active_edits_after_spilling_and_rebalances_on_growth():
    cache = SessionCache(max_sessions=1, max_bytes=96)
    first = SpectrumSession(make_spectrum())
    cache["a"] = first
    cache["b"] = SpectrumSession(make_spectrum())
    first.apply(make_spectrum(-3), "late edit")
    assert cache["a"].current.intensity[0] == 7
    first.apply(make_spectrum(-4), "another")
    assert cache.memory_bytes <= 96
    assert cache["a"].audit_count == 2
    cache.close()


def test_cache_disk_full_keeps_data_and_reports_capacity():
    cache = SessionCache(max_sessions=1, max_disk_bytes=1)
    first = SpectrumSession(make_spectrum())
    first.apply(make_spectrum(-1), "correction")
    cache["a"] = first
    cache["b"] = SpectrumSession(make_spectrum())
    assert cache.capacity_warning
    assert cache["a"].current.intensity[0] == 9
    cache.close()


def test_spilled_session_undo_is_persisted_after_last_live_reference_dropped():
    import gc
    cache = SessionCache(max_bytes=48)
    session = SpectrumSession(make_spectrum())
    cache["a"] = session
    session.apply(make_spectrum(-1), "step")
    session.undo()
    assert cache.resident_count == 0
    del session
    gc.collect()
    restored = cache["a"]
    assert restored.cursor == 0
    assert restored.can_redo
    assert restored.audit_log[-1]["action"] == "撤销"
    cache.close()


def test_processed_metadata_does_not_follow_later_caller_mutations():
    session = SpectrumSession(make_spectrum())
    result = make_spectrum(-1)
    result.metadata["nested"] = {"data": np.array([1., 2.])}
    session.apply(result, "step")
    result.metadata["nested"]["data"][0] = 999
    result.metadata["nested"]["new"] = "later"
    assert session.current.metadata["nested"]["data"][0] == 1
    assert "new" not in session.current.metadata["nested"]
