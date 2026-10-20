"""Regression (2026-10-20): /api/status showed drift n=0 forever. The engine's drift state was updated on every close,
but the stored safety view was only rewritten when the drift LEVEL changed."""
from test_engine_core import mk_engine, open_long, run


def test_stored_safety_view_follows_the_drift_sample_size(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    run(eng.heartbeat_cycle())
    assert store.kv_get("safety_engine")["drift"]["n"] == 0
    open_long(eng)
    pos = store.get_positions()[0]
    assert run(eng.close_position(pos, "manual"))
    assert eng.drift_state.n == 1                          # the engine itself always knew
    run(eng.heartbeat_cycle())
    assert store.kv_get("safety_engine")["drift"]["n"] == 1


def test_a_restart_seeds_the_drift_monitor_from_stored_trades(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng)
    assert run(eng.close_position(store.get_positions()[0], "manual"))
    eng2, _ = mk_engine(store, tmp_path)                  # same DB: the restarted engine remembers the evidence
    assert eng2.drift_state.n == 1
