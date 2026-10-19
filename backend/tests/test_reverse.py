"""Reversal (user 2026-10-19): an opposite-side signal on a held coin closes it with a normal order (exit "reverse")
and opens the other side through every entry check; no reduce-only anywhere."""
from test_engine_core import closed_of, mk_engine, open_long, run


def open_short(eng, sym="AAAUSDT", prob=0.9):
    async def go():
        await eng.update_symbol(sym)
        await eng.try_open(sym, "short", prob, [], closed_of(eng, sym))
    run(go())


def test_opposite_signal_reverses_the_position(store, tmp_path):
    eng, data = mk_engine(store, tmp_path, settings={"short_enabled": True})
    open_long(eng)
    assert [p["side"] for p in store.get_positions()] == ["long"]
    open_short(eng)
    pos = store.get_positions()
    assert [(p["symbol"], p["side"]) for p in pos] == [("AAAUSDT", "short")]
    assert [t["exit_reason"] for t in store.get_trades(5)] == ["reverse"]
    assert {o.side for o in eng.ex.open_conditional_orders("AAAUSDT")} == {"BUY"}     # TP/SL for the short only


def test_same_side_signal_does_nothing(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng)
    open_long(eng)
    assert len(store.get_positions()) == 1 and store.get_trades(5) == []


def test_reverse_disabled_or_below_margin_keeps_the_position(store, tmp_path):
    eng, data = mk_engine(store, tmp_path, settings={"short_enabled": True, "reverse_enabled": False})
    open_long(eng)
    open_short(eng)
    assert [p["side"] for p in store.get_positions()] == ["long"] and store.get_trades(5) == []
    eng.cfg = eng.cfg.merge({"reverse_enabled": True, "reverse_min_prob_margin": 0.5})
    open_short(eng, prob=0.9)                      # threshold 0.70 + 0.5 margin > 0.9
    assert [p["side"] for p in store.get_positions()] == ["long"] and store.get_trades(5) == []
