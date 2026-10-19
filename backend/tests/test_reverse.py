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


# ------------------------------------------------------------------ backtester parity
def _sim_setup():
    import numpy as np
    from test_sim_helpers import build, flat, plain_cfg
    n = 40
    d = flat(n)
    for i in range(n):
        d["h"][i], d["l"][i] = 100.5, 99.5          # quiet: no TP / SL inside the window
    sig = np.zeros((n, 2))
    sig[20, 0] = 0.9                                # long (after the 14-bar ATR warm-up)
    sig[30, 1] = 0.9                                # then the model calls short
    exits = {"exit_mode": "atr", "sl_atr": 5.0, "tp_atr": 8.0, "sl_usd": 10.0}
    return build(d), sig, exits, plain_cfg


def test_sim_reverses_on_an_opposite_model_signal():
    from tb.sim.backtest import run_backtest
    kl, sig, exits, plain_cfg = _sim_setup()
    r = run_backtest({"X": kl}, {"X": sig}, plain_cfg(**exits, thr_long=0.7, thr_short=0.7))
    assert r.stats.get("reversals") == 1
    assert [t["exit_reason"] for t in r.trades] == ["reverse"] and r.trades[0]["side"] == "long"
    assert r.stats["entries"] == 2                  # the short opened after the reverse


def test_sim_reverse_off_or_below_margin_keeps_the_long():
    from tb.sim.backtest import run_backtest
    kl, sig, exits, plain_cfg = _sim_setup()
    for kw in ({"reverse_enabled": False}, {"reverse_margin": 0.5}):
        r = run_backtest({"X": kl}, {"X": sig}, plain_cfg(**exits, thr_long=0.7, thr_short=0.7, **kw))
        assert not r.stats.get("reversals") and r.trades == [] and r.stats["skip_has_position"] == 1
