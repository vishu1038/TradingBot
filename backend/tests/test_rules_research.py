"""The rules research harness: ATR-exit simulation (TP closes everything), reversal, baselines, statistics helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tb.config import Settings
from tb.sim import rules_research as RR
from tb.sim.model_backtest import Market, SimCfg

BAR = 300_000
T0 = 1_735_689_600_000


def flat_market(n: int = 400, mods: dict | None = None) -> tuple[Market, pd.DataFrame]:
    t = T0 + np.arange(n) * BAR
    o = np.full(n, 100.0)
    h = np.full(n, 100.1)
    l = np.full(n, 99.9)
    c = np.full(n, 100.0)
    for i, (hi, lo) in (mods or {}).items():
        h[i], l[i] = hi, lo
    kl = pd.DataFrame({"open_time": t, "open": o, "high": h, "low": l, "close": c, "volume": 1.0, "quote_volume": 100.0})
    fu = pd.DataFrame({"calc_time": np.zeros(0, "int64"), "funding_rate": np.zeros(0)})
    return Market({"X": (kl, fu)}), kl


def cand(bar_idx: int, side: str, atr: float = 2.0) -> pd.DataFrame:
    return pd.DataFrame({"sym": ["X"], "bar": [T0 + bar_idx * BAR], "side": [side], "prob": [1.0], "atr": [atr]})


CFG = Settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0, "fee_taker": 0.0}, allow_internal=True)


@pytest.mark.parametrize("side,d", [("long", 1.0), ("short", -1.0)])
def test_tp_closes_the_whole_position_at_1_5_atr(side, d):
    mkt, _ = flat_market(mods={101: (100.0 + d * 3.5 if side == "long" else 100.1, 100.0 + d * 3.5 if side == "short" else 99.9)})
    tr = RR.simulate_atr(cand(100, side), mkt, CFG, RR.SPEC, SimCfg(), T0 + 10**9)
    assert len(tr) == 1 and tr[0]["reason"] == "tp"
    # qty = min(10 USD / (3 ATR), 160 / 100) = 1.6 ; TP = 3 USD away -> +4.8
    assert tr[0]["pnl"] == pytest.approx(1.6 * 3.0)
    assert tr[0]["side"] == side


def test_sl_is_three_atr_away_and_tp_first_bar_wins_over_a_later_stop():
    mkt, _ = flat_market(mods={101: (103.5, 99.9), 102: (100.1, 90.0)})        # bar 101 reaches the TP, bar 102 would hit the SL
    tr = RR.simulate_atr(cand(100, "long"), mkt, CFG, RR.SPEC, SimCfg(), T0 + 10**9)
    assert [t["reason"] for t in tr] == ["tp"] and tr[0]["pnl"] > 0
    mkt, _ = flat_market(mods={101: (100.1, 93.9)})                            # 6.1 below the entry: SL at -6
    tr = RR.simulate_atr(cand(100, "long"), mkt, CFG, RR.SPEC, SimCfg(), T0 + 10**9)
    assert tr[0]["reason"] == "sl" and tr[0]["pnl"] == pytest.approx(-1.6 * 6.0)
    mkt, _ = flat_market(mods={101: (103.5, 93.9)})                            # both in one bar: pessimistic SL
    assert RR.simulate_atr(cand(100, "long"), mkt, CFG, RR.SPEC, SimCfg(), T0 + 10**9)[0]["reason"] == "sl"


def test_time_stop_and_dollar_floor_and_cooldown():
    mkt, _ = flat_market(n=1500)
    tr = RR.simulate_atr(cand(100, "long"), mkt, CFG, RR.SPEC, SimCfg(), T0 + 10**9)
    assert tr[0]["reason"] == "timeout"                                        # 1152 bars later
    # tiny ATR: the TP is worth < $2 -> skipped by the dollar floor (never shrunk)
    assert RR.simulate_atr(cand(100, "long", atr=0.2), mkt, CFG, RR.SPEC, SimCfg(), T0 + 10**9) == []
    # a second signal on the same coin 3 bars later is inside cooldown_bars (6)
    two = pd.concat([cand(100, "long"), cand(103, "long")])
    mkt, _ = flat_market(mods={101: (103.5, 99.9)})
    assert len(RR.simulate_atr(two, mkt, CFG, RR.SPEC, SimCfg(), T0 + 10**9)) == 1


def test_opposite_signal_reverses_a_held_coin():
    mkt, _ = flat_market(n=600, mods={400: (103.5, 99.9)})
    two = pd.concat([cand(100, "long"), cand(120, "short")])
    info: dict = {}
    tr = RR.simulate_atr(two, mkt, CFG, RR.SPEC, SimCfg(), T0 + 10**9, info)
    assert info["reversals"] == 1
    assert [t["reason"] for t in tr][0] == "reverse" and tr[0]["side"] == "long" and tr[1]["side"] == "short"


def test_protection_inert_for_the_research_exit_and_would_not_be_for_a_nearer_stop():
    assert RR.protection_inert(Settings(), RR.SPEC)
    assert RR.protection_inert(Settings().merge({"breakeven_r": 1.0, "partial_tp_r": 1.0, "trail": {"enabled": True, "activate_r": 1.0,
                                                                                                       "distance_atr": 1.0}}, allow_internal=True), RR.SPEC)
    from tb.exits import ExitSpec
    assert not RR.protection_inert(Settings(), ExitSpec("atr", "1h", tp_atr=3.0, sl_atr=1.5))        # TP beyond 1 R: not inert


def test_segments_cover_the_period_without_gaps():
    a, b = RR.PERIODS["dev"]
    segs = RR.segments(a, b)
    assert segs[0][0] == a and segs[-1][1] == b and all(x[1] == y[0] for x, y in zip(segs, segs[1:]))
    assert all(x[1] > x[0] for x in segs) and 9 <= len(segs) <= 13


def test_summarize_and_stats_helpers():
    tr = [dict(side="long", t_out=3, pnl=2.0, reason="tp"), dict(side="short", t_out=1, pnl=-1.0, reason="sl"),
          dict(side="long", t_out=2, pnl=-4.0, reason="sl")]
    s = RR.summarize(tr, days=1.5)
    assert s["n"] == 3 and s["per_day"] == pytest.approx(2.0) and s["total"] == pytest.approx(-3.0)
    assert s["maxdd"] == pytest.approx(5.0) and (s["n_long"], s["n_short"]) == (2, 1) and s["win"] == pytest.approx(1 / 3)
    assert RR.zscore(1.0, [0.0, 1.0, -1.0, 0.5]) == pytest.approx((1.0 - 0.125) / np.std([0.0, 1.0, -1.0, 0.5], ddof=1))
    p, zc = RR.bonferroni(3.0, 12)
    assert p == pytest.approx(min(1.0, 12 * 0.0013499), rel=1e-3) and zc < 3.0
    assert RR.bonferroni(-1.0, 12)[0] == 1.0


def test_baseline_draws_match_counts_sides_and_regime():
    n = 5000
    pool = pd.DataFrame({"sym": pd.Categorical(np.where(np.arange(n) % 2 == 0, "A", "B")), "bar": T0 + np.arange(n) * BAR,
                         "atr": np.float32(1.0), "reg_btc": np.where(np.arange(n) % 3 == 0, 0, np.where(np.arange(n) % 2 == 0, 1, -1)).astype("int8"),
                         "reg_coin": np.zeros(n, "int8")})
    r = RR.draw_random(pool, 40, 25, np.random.default_rng(0))
    assert (r["side"] == "long").sum() == 40 and (r["side"] == "short").sum() == 25 and r["sym"].map(type).eq(str).all()
    g = RR.draw_regime_only(pool, "btc", 100, np.random.default_rng(0))
    assert len(g) == 100
    reg = pool.set_index("bar")["reg_btc"]
    assert (reg.loc[g["bar"]].to_numpy() != 0).all()
    assert (np.where(reg.loc[g["bar"]].to_numpy() > 0, "long", "short") == g["side"].to_numpy()).all()
    assert RR.draw_regime_only(pool, "coin", 100, np.random.default_rng(0)).empty              # no trending bars -> nothing to draw
