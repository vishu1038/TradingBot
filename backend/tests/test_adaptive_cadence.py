"""Adaptive decision cadence (entry_cfg.decision_mode = "adaptive"): events, extra confidence, min gap, the variable
regular gap, live / shadow parity, and fixed mode unchanged."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from test_engine_core import mk_engine, run
from test_v3_engine_core import doubles, hour_now, write_v3  # noqa: F401  (doubles = fixture)
from tb import adaptive as A
from tb.adaptive import AdaptiveState, CoinState, regular_gap
from tb.entry import BAR_MS, EntryCfg

HOUR = 3_600_000
T0 = 1_760_000_000_000 // HOUR * HOUR            # a UTC hour boundary (ms)
THR_L, THR_S = 0.60, 0.60

OFF = dict(ev_range_atr=0.0, ev_volume_z=0.0, ev_breakout=False, ev_oi_jump_pct=0.0, ev_prob_jump=0.0)


def cfg(**kw) -> EntryCfg:
    base = dict(decision_mode="adaptive", base_gap_bars=24, min_regular_gap_bars=24, max_regular_gap_bars=24,
                min_gap_bars=1, ev_extra_conf=0.05, **OFF)
    base.update(kw)
    return EntryCfg(**base)


def flat(n: int = 700, end_open: int = T0 + 4 * BAR_MS, seed: int = 1) -> pd.DataFrame:
    """Quiet 5m bars around 100 (tiny noise); the last bar OPENS at ``end_open`` (4 bars into an hour by default)."""
    rng = np.random.default_rng(seed)
    t = end_open - np.arange(n - 1, -1, -1, dtype="int64") * BAR_MS
    c = 100.0 + rng.uniform(-0.01, 0.01, n)
    o = np.concatenate([[100.0], c[:-1]])
    h = np.full(n, 100.05)                                   # constant range: no bar is 1.5 x ATR by chance
    l = np.full(n, 99.95)
    v = 100.0 + rng.uniform(0, 10, n)
    return pd.DataFrame({"open_time": t, "open": o, "high": h, "low": l, "close": c, "volume": v})


def seeded(sym: str = "ETHUSDT", frame: pd.DataFrame | None = None, last_prob=None, **ent_kw):
    """State whose coin made its last (regular) decision at the start of the hour of the frame's last bar."""
    st = AdaptiveState()
    f = frame if frame is not None else flat()
    bar = int(f["open_time"].iloc[-1])
    g = bar // HOUR * HOUR
    st.coin[sym] = CoinState(g, g, dict(last_prob or {"long": None, "short": None}))
    return st, f, cfg(**ent_kw)


def step(st, f, ent, pl=0.8, ps=0.1, metrics=None, sym="ETHUSDT"):
    return st.step(sym, f, metrics, pl, ps, ent, THR_L, THR_S, True)


# ----------------------------------------------------------------------------- events
def test_no_event_means_no_intra_hour_decision():
    st, f, ent = seeded(**{**OFF, "ev_range_atr": 1.5, "ev_volume_z": 2.5, "ev_breakout": True, "ev_oi_jump_pct": 1.0,
                           "ev_prob_jump": 0.1})
    d = step(st, f, ent)
    assert d.kind is None and d.side is None                      # p=0.8 >> threshold, but nothing happened on the bar


def test_range_event_triggers():
    f = flat()
    f.loc[f.index[-1], "high"] = 100.3
    st, f, ent = seeded(frame=f, ev_range_atr=1.5)
    d = step(st, f, ent)
    assert d.kind == "event" and d.side == "long" and "range_atr" in d.events and d.reason.startswith("event range_atr=")


def test_volume_event_triggers():
    f = flat()
    f.loc[f.index[-1], "volume"] = 400.0
    st, f, ent = seeded(frame=f, ev_volume_z=2.5)
    d = step(st, f, ent)
    assert d.kind == "event" and d.side == "long" and d.events["volume_z"] > 2.5


def test_breakout_event_is_side_specific():
    f = flat()
    f.loc[f.index[-1], ["close", "high"]] = [100.2, 100.2]          # above the prior closed 1h bar's high
    st, f, ent = seeded(frame=f, ev_breakout=True)
    d = step(st, f, ent, pl=0.8, ps=0.1)
    assert d.kind == "event" and d.side == "long" and "breakout" in d.events and "breakout" in d.reason
    st2, f2, ent2 = seeded(frame=f, ev_breakout=True)
    assert step(st2, f2, ent2, pl=0.1, ps=0.8).side is None            # an UP breakout is no reason to go short
    g = flat()
    g.loc[g.index[-1], ["close", "low"]] = [99.8, 99.8]
    st3, g, ent3 = seeded(frame=g, ev_breakout=True)
    d3 = step(st3, g, ent3, pl=0.1, ps=0.8)
    assert d3.kind == "event" and d3.side == "short"


def test_open_interest_event_triggers():
    f = flat()
    ct = f["open_time"].to_numpy()[-6:]
    m = pd.DataFrame({"create_time": ct, "sum_open_interest": [1000.0, 1000.0, 1000.0, 1000.0, 1010.0, 1015.0]})
    st, f, ent = seeded(frame=f, ev_oi_jump_pct=1.0)
    d = step(st, f, ent, metrics=m)
    assert d.kind == "event" and d.events["oi_jump_pct"] >= 1.0
    st2, f2, ent2 = seeded(frame=f, ev_oi_jump_pct=1.0)
    m2 = m.assign(sum_open_interest=[1000.0] * 5 + [1005.0])
    assert step(st2, f2, ent2, metrics=m2).kind is None                # 0.5 % is not a jump
    st3, f3, ent3 = seeded(frame=f, ev_oi_jump_pct=1.0)
    assert step(st3, f3, ent3, metrics=None).kind is None              # no metrics: the event cannot fire


def test_probability_jump_event_triggers():
    st, f, ent = seeded(last_prob={"long": 0.55, "short": 0.1}, ev_prob_jump=0.10)
    d = step(st, f, ent, pl=0.70)
    assert d.kind == "event" and d.side == "long" and d.events["prob_jump"] == pytest.approx(0.15)
    st2, f2, ent2 = seeded(last_prob={"long": 0.65, "short": 0.1}, ev_prob_jump=0.10)
    assert step(st2, f2, ent2, pl=0.70).kind is None                   # +0.05 only
    st3, f3, ent3 = seeded(last_prob=None, ev_prob_jump=0.10)
    assert step(st3, f3, ent3, pl=0.95).kind is None                   # nothing to compare with yet


# ----------------------------------------------------------------------------- extra confidence
def test_event_entry_needs_threshold_plus_extra_confidence():
    f = flat()
    f.loc[f.index[-1], "volume"] = 400.0
    st, f, ent = seeded(frame=f, ev_volume_z=2.5, ev_extra_conf=0.05)
    d = step(st, f, ent, pl=0.63)                                      # above the 0.60 threshold, below 0.65
    assert d.kind == "event" and d.side is None
    st2, f2, ent2 = seeded(frame=f, ev_volume_z=2.5, ev_extra_conf=0.05)
    d2 = step(st2, f2, ent2, pl=0.66)
    assert d2.side == "long" and d2.prob == pytest.approx(0.66)


def test_regular_decision_uses_the_normal_threshold():
    st, f, ent = seeded(base_gap_bars=12, min_regular_gap_bars=12, max_regular_gap_bars=12)
    f = flat(end_open=T0 + 11 * BAR_MS)
    st.coin["ETHUSDT"] = CoinState(T0 - BAR_MS, T0 - BAR_MS)
    d = step(st, f, ent, pl=0.61)                                      # 12 bars after the last regular look
    assert d.kind == "regular" and d.side == "long" and d.gap == 12


# ----------------------------------------------------------------------------- min gap + only-hourly
def test_min_gap_bars_spaces_decision_points_of_one_coin():
    f = flat(n=800)
    f.loc[f.index[-6:], "volume"] = 400.0                              # volume spike on every one of the last bars
    ent = cfg(ev_volume_z=2.0, min_gap_bars=3)
    st = AdaptiveState()
    bar0 = int(f["open_time"].iloc[-6])
    st.coin["ETHUSDT"] = CoinState(bar0 - 100 * BAR_MS, bar0 - 100 * BAR_MS)
    st.coin["ETHUSDT"].last_regular_ms = bar0 - 2 * BAR_MS             # recent regular look: none due for 24 bars
    st.coin["ETHUSDT"].last_decision_ms = bar0 - 5 * BAR_MS
    kinds = []
    for k in range(6, 0, -1):
        sub = f.iloc[: len(f) - k + 1]
        d = step(st, sub, ent)
        kinds.append(d.kind)
    # decisions at the 1st bar (gap 5 since the last one) then only every 3 bars
    assert kinds == ["event", None, None, "event", None, None]


def test_without_events_only_regular_looks_happen_and_scoring_is_free():
    ent = cfg(base_gap_bars=12, min_regular_gap_bars=12, max_regular_gap_bars=12, ev_range_atr=1.5, ev_volume_z=2.5,
              ev_breakout=True, ev_oi_jump_pct=1.0, ev_prob_jump=0.1)
    full = flat(n=900, end_open=T0 + 30 * BAR_MS)
    st = AdaptiveState()
    kinds = []
    for k in range(30, -1, -1):
        sub = full.iloc[: len(full) - k]
        d = step(st, sub, ent, pl=0.8)
        kinds.append((int(sub["open_time"].iloc[-1]), d.kind))
    got = [(t - T0) // BAR_MS for t, kd in kinds if kd]
    assert all(kd == "regular" for _, kd in kinds if kd)
    assert got == [11, 23]                                            # the bars that CLOSE on the hour grid


# ----------------------------------------------------------------------------- variable regular gap
def walk(n: int, hist_vol: float, last_vol: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = np.concatenate([rng.normal(0, hist_vol, n - 12), rng.normal(0, last_vol, 12)])
    c = 100.0 * np.exp(np.cumsum(r))
    t = T0 - np.arange(n - 1, -1, -1, dtype="int64") * BAR_MS
    return pd.DataFrame({"open_time": t, "open": c, "high": c, "low": c, "close": c, "volume": 100.0})


def test_regular_gap_is_short_for_a_volatile_coin_and_long_for_a_quiet_one():
    vol_gap, vr = regular_gap(walk(2600, 0.001, 0.003), 12, 3, 24)
    quiet_gap, qr = regular_gap(walk(2600, 0.001, 0.0006), 12, 3, 24)
    assert vr < 1 < qr
    assert vol_gap < 12 < quiet_gap
    assert 3 <= vol_gap <= 6 and 16 <= quiet_gap <= 24


def test_regular_gap_clamps_and_degrades_to_the_base_gap():
    assert regular_gap(walk(2600, 0.001, 0.05), 12, 3, 24)[0] == 3             # a wildly volatile coin: floor
    assert regular_gap(walk(2600, 0.001, 0.00001), 12, 3, 24)[0] == 24         # a dead coin: ceiling
    assert regular_gap(walk(2600, 0.001, 0.05), 12, 6, 24)[0] == 6
    assert regular_gap(walk(2600, 0.001, 0.00001), 12, 3, 18)[0] == 18
    assert regular_gap(walk(100, 0.001, 0.05), 12, 3, 24) == (12, None)        # < 1 day of history: base gap
    flat_ = walk(2600, 0.001, 0.0)
    assert regular_gap(flat_, 12, 3, 24)[0] == 24                              # zero current vol


def test_volatile_coin_looks_again_sooner_than_a_quiet_one():
    ent = cfg(base_gap_bars=12, min_regular_gap_bars=3, max_regular_gap_bars=24)
    res = {}
    for name, last in (("vol", 0.004), ("quiet", 0.0003)):
        f = walk(2600, 0.001, last)
        bar = int(f["open_time"].iloc[-1])
        st = AdaptiveState()
        st.coin["X"] = CoinState(bar - 7 * BAR_MS, bar - 7 * BAR_MS)               # last regular look 7 bars ago
        res[name] = st.step("X", f, None, 0.7, 0.1, ent, THR_L, THR_S, True).kind
    assert res == {"vol": "regular", "quiet": None}


# ----------------------------------------------------------------------------- config
def test_entry_cfg_defaults_and_validation():
    d = EntryCfg.resolve(__import__("tb.config", fromlist=["Settings"]).Settings(), {})
    assert d.decision_mode == "fixed" and not d.adaptive
    a = EntryCfg.resolve(__import__("tb.config", fromlist=["Settings"]).Settings(), {"decision_mode": "adaptive"})
    assert (a.adaptive, a.base_gap_bars, a.min_regular_gap_bars, a.max_regular_gap_bars, a.min_gap_bars) == (True, 12, 3, 24, 1)
    assert (a.ev_range_atr, a.ev_volume_z, a.ev_breakout, a.ev_oi_jump_pct, a.ev_prob_jump, a.ev_extra_conf) == \
        (1.5, 2.5, True, 1.0, 0.10, 0.05)
    bad = EntryCfg.resolve(__import__("tb.config", fromlist=["Settings"]).Settings(),
                           {"decision_mode": "nonsense", "ev_range_atr": "x", "min_gap_bars": -3})
    assert bad.decision_mode == "fixed" and bad.ev_range_atr == 1.5 and bad.min_gap_bars == 1


# ----------------------------------------------------------------------------- engine: live + shadow parity, fixed unchanged
ADAPT = dict(decision_mode="adaptive", base_gap_bars=12, min_regular_gap_bars=12, max_regular_gap_bars=12, cooldown_bars=0,
             ev_range_atr=0.0, ev_volume_z=0.0, ev_breakout=False, ev_oi_jump_pct=0.0, ev_prob_jump=0.0)


def models(d, champ_cfg, chal_cfg=None):
    write_v3(d, "champ", "champion", kind="mtf", history_bars_5m=100, entry_cfg=champ_cfg)
    if chal_cfg is not None:
        write_v3(d, "chal", "challenger", kind="mtf", history_bars_5m=100, entry_cfg=chal_cfg)


def test_adaptive_scores_every_bar_but_enters_only_at_regular_points_and_events(store, tmp_path, doubles):
    models(tmp_path / "models", ADAPT)
    eng, _ = mk_engine(store, tmp_path, now=hour_now() + 1500)           # 4 bars after the hour, no event configured
    run(eng.bars_cycle())
    assert len(store.get_signals(None, 10)) == 2                         # scored (and recorded) on a non-hour bar ...
    assert store.get_positions() == []                                   # ... but no decision point -> no entry
    eng2, _ = mk_engine(store, tmp_path, now=hour_now())                  # the bar that closes on the hour: regular look
    run(eng2.bars_cycle())
    assert len(store.get_positions()) == 2
    assert eng2.decisions_status()["decision_mode"] == "adaptive"
    assert eng2.decisions_status()["last_decision"]["kind"] == "regular"


def test_event_entry_is_logged_with_its_event(store, tmp_path, doubles, caplog):
    models(tmp_path / "models", {**ADAPT, "ev_range_atr": 0.001})        # any bar range is an event
    eng, _ = mk_engine(store, tmp_path, now=hour_now() + 1500)
    with caplog.at_level(logging.INFO, logger="tb.engine"):
        run(eng.bars_cycle())
    assert {p["symbol"] for p in store.get_positions()} == {"AAAUSDT", "BBBUSDT"}
    msgs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("adaptive entry")]
    assert any(m.startswith("adaptive entry AAAUSDT long: event range_atr=") and "p=0.90" in m for m in msgs)
    st = eng.decisions_status()
    assert st["last_decision"]["kind"] == "event" and st["decision_counts"]["entries_event"] == 2


def test_shadow_model_decides_exactly_like_the_live_champion(store, tmp_path, doubles):
    for now in (hour_now() + 1500, hour_now()):
        for extra in ({}, {"ev_range_atr": 0.001}):
            import shutil
            shutil.rmtree(tmp_path / "models", ignore_errors=True)
            models(tmp_path / "models", {**ADAPT, **extra}, {**ADAPT, **extra})
            eng, _ = mk_engine(store, tmp_path, now=now)
            run(eng.bars_cycle())
            for s in ("AAAUSDT", "BBBUSDT"):
                assert eng.shadow.has("champ", s) == eng.shadow.has("chal", s)
            live = {p["symbol"] for p in store.get_positions()}
            assert live == {s for s in ("AAAUSDT", "BBBUSDT") if eng.shadow.has("chal", s)}
            store.clear_positions() if hasattr(store, "clear_positions") else None
            run(eng.close_all("manual"))


def test_fixed_mode_is_unchanged(store, tmp_path, doubles):
    models(tmp_path / "models", {"decision_every_bars": 12, "cooldown_bars": 0, **{"ev_range_atr": 0.001}})   # (ev_* ignored in fixed)
    eng, _ = mk_engine(store, tmp_path, now=hour_now() + 1500)
    run(eng.bars_cycle())
    assert store.get_signals(None, 10) == [] and store.get_positions() == []     # not a decision bar: nothing happens
    assert "decision_mode" not in eng.decisions_status() and not eng.model_entry(eng.model_info).adaptive
    eng2, _ = mk_engine(store, tmp_path, now=hour_now())
    run(eng2.bars_cycle())
    assert len(store.get_positions()) == 2
