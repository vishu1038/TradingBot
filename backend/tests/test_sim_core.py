"""Hand-computed cases for the trade template in the event-driven backtester."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

from test_sim_helpers import BAR, T0, build, flat, mk_klines, plain_cfg, sig_at
from tb.sim import run_backtest
from tb.sim.template import SimConfig, breakeven_winrate, liq_price, size_and_levels

FEE = 0.0005


def run1(d, sigs, cfg, funding=None):
    kl = build(d)
    f = {"X": funding} if funding is not None else None
    return run_backtest({"X": kl}, {"X": sigs}, cfg, f)


def test_long_hits_tp_exactly():
    d = flat(10)
    d["h"][3] = 103.2            # TP = 100 * (1 + 5/160) = 103.125
    r = run1(d, sig_at(10, [0]), plain_cfg())
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t["exit_reason"] == "tp" and t["side"] == "long"
    assert t["entry_price"] == 100.0 and t["exit_price"] == pytest.approx(103.125)
    assert t["qty"] == pytest.approx(1.6)                   # 160 / 100
    assert t["gross_pnl"] == pytest.approx(5.0)
    fees = FEE * 1.6 * 100 + FEE * 1.6 * 103.125            # 0.08 + 0.0825
    assert t["fees"] == pytest.approx(fees) and fees == pytest.approx(0.1625)
    assert t["pnl"] == pytest.approx(5.0 - 0.1625)
    assert r.equity[-1] == pytest.approx(1000 + 4.8375)


def test_long_hits_sl_with_fees_and_slippage():
    d = flat(10)
    d["l"][3] = 90.0             # SL = 100 * (1 - 10/160) = 93.75
    # no slippage first
    t = run1(d, sig_at(10, [0]), plain_cfg()).trades[0]
    assert t["exit_reason"] == "sl" and t["exit_price"] == pytest.approx(93.75)
    assert t["gross_pnl"] == pytest.approx(-10.0)
    assert t["pnl"] == pytest.approx(-10.0 - FEE * 1.6 * (100 + 93.75))
    # with slippage 0.02% (entry + stops) and +0.05% extra on stop fills
    cfg = plain_cfg(slippage=0.0002, stop_slippage_extra=0.0005)
    t = run1(d, sig_at(10, [0]), cfg).trades[0]
    entry = 100.0 * 1.0002
    qty = 160.0 / entry
    exit_px = entry * (1 - 10 / 160) * (1 - 0.0007)     # levels hang off the slipped fill
    assert t["entry_price"] == pytest.approx(entry)
    assert t["exit_price"] == pytest.approx(exit_px)
    gross = qty * (exit_px - entry)
    fees = FEE * qty * entry + FEE * qty * exit_px
    assert t["pnl"] == pytest.approx(gross - fees)
    assert t["pnl"] < -10.0 - 0.16  # strictly worse than the zero-cost loss


def test_same_bar_tp_and_sl_resolves_sl_first():
    d = flat(10)
    d["h"][3], d["l"][3] = 104.0, 90.0     # both barriers inside one candle
    t = run1(d, sig_at(10, [0]), plain_cfg()).trades[0]
    assert t["exit_reason"] == "sl" and t["gross_pnl"] == pytest.approx(-10.0)


def test_entry_bar_can_stop_out_and_entry_is_next_open_no_lookahead():
    d = flat(10)
    d["o"][1], d["h"][1], d["l"][1], d["c"][1] = 101.0, 101.0, 101.0, 101.0   # next bar open != signal close
    d["h"][2] = 110.0
    r = run1(d, sig_at(10, [0]), plain_cfg())
    t = r.trades[0]
    assert t["entry_price"] == 101.0                         # open of the bar AFTER the signal bar
    assert t["opened_at"] == (T0 + BAR) // 1000
    # a signal on the very last bar has no next bar -> no trade
    assert run1(flat(6), sig_at(6, [5]), plain_cfg()).trades == []
    # the signal bar's own range is irrelevant: a TP-sized spike inside the SIGNAL bar does not exit
    d2 = flat(8)
    d2["h"][0] = 200.0
    assert run1(d2, sig_at(8, [0]), plain_cfg()).trades == []   # still open at the end, no exit


def test_trailing_stop_ratchets_and_exits():
    n = 30
    d = flat(n)
    for i in range(n):                       # constant range 2 -> ATR14 == 2 from bar 13 on
        d["h"][i], d["l"][i] = 101.0, 99.0
    for i in range(22, 27):                  # price stays up there (no gap-down opens)
        d["o"][i] = d["c"][i] = 102.0
    # entry at bar 21 (signal bar 20), risk = 6.25, activate at 0.4R = 2.5 -> 102.5 (< TP 103.125)
    d["h"][22], d["l"][22] = 102.6, 100.0    # best 102.6 -> trail 100.6 (effective from next bar)
    d["h"][23], d["l"][23] = 103.0, 101.5    # ratchets to 101.0 ; stop 100.6 not touched
    d["h"][24], d["l"][24] = 102.0, 100.8    # low pierces 101.0 but not 100.6 -> proves it ratcheted
    cfg = plain_cfg(trail_enabled=True, trail_activate_r=0.4, trail_distance_atr=1.0)
    r = run1(d, sig_at(n, [20]), cfg)
    t = r.trades[0]
    assert t["exit_reason"] == "trail"
    assert t["exit_price"] == pytest.approx(101.0)
    assert t["gross_pnl"] == pytest.approx(1.6 * 1.0)
    # never loosens: a pullback bar (lower high) between does not lower the stop
    d["h"][23], d["l"][23] = 102.0, 101.5    # best stays 102.6 -> stop 100.6, then bar 24 low 100.8 survives
    d["h"][24], d["l"][24] = 102.0, 100.8
    d["h"][25], d["l"][25] = 101.0, 100.5    # pierces 100.6
    t2 = run1(d, sig_at(n, [20]), cfg).trades[0]
    assert t2["exit_reason"] == "trail" and t2["exit_price"] == pytest.approx(100.6)


def test_trail_off_leaves_fixed_levels():
    d = flat(30)
    for i in range(30):
        d["h"][i], d["l"][i] = 101.0, 99.0
    d["h"][22] = 102.9
    d["l"][24] = 95.0                        # above SL 93.75 and nothing trails -> still open
    r = run1(d, sig_at(30, [20]), plain_cfg())
    assert r.trades == []


def test_funding_sign_long_pays_short_receives_when_rate_positive():
    n = 20
    # bars start 07:00 UTC; the 08:00 settlement falls on row 12
    t0 = T0 + 7 * 3_600_000
    fund = pd.DataFrame({"calc_time": [t0 + 12 * BAR], "funding_rate": [0.0001]})
    for side, sgn in (("long", +1), ("short", -1)):
        d = flat(n)
        d["t0"] = t0
        if side == "long":
            d["h"][14] = 103.2                 # TP
        else:
            d["l"][14] = 96.8                  # TP for a short = 100 * (1 - 5/160) = 96.875
        cfg = plain_cfg(short_enabled=True)
        r = run1(d, sig_at(n, [0], side), cfg, funding=fund)
        t = r.trades[0]
        assert t["side"] == side and t["exit_reason"] == "tp"
        assert t["funding"] == pytest.approx(sgn * 0.0001 * 1.6 * 100.0)   # + = paid
        exit_px = 103.125 if side == "long" else 96.875
        assert t["pnl"] == pytest.approx(5.0 - FEE * 1.6 * (100 + exit_px) - t["funding"])
        # the wallet reflects the funding cash flow
        assert r.equity[-1] == pytest.approx(1000 + t["pnl"])
    # settlement before the position opened costs nothing
    fund2 = pd.DataFrame({"calc_time": [t0 + 1 * BAR], "funding_rate": [0.01]})
    d = flat(n); d["t0"] = t0; d["h"][14] = 103.2
    assert run1(d, sig_at(n, [3]), plain_cfg(), funding=fund2).trades[0]["funding"] == 0.0


def test_liquidation_price_for_8x_isolated():
    assert liq_price(100.0, +1, 8, 0.005) == pytest.approx(100 * 0.875 / 0.995)   # ~87.94
    assert liq_price(100.0, -1, 8, 0.005) == pytest.approx(100 * 1.125 / 1.005)   # ~111.94
    # liq sits beyond the SL (6.25 %) for fixed mode -> SL normally first
    cfg = SimConfig()
    qty, tp, sl = size_and_levels(100.0, 100.0, +1, float("nan"), cfg)
    assert sl == pytest.approx(93.75) and liq_price(100.0, +1, 8, cfg.mmr) < sl


def test_stop_fires_before_liquidation_unless_gapped():
    d = flat(10)
    d["l"][3] = 80.0                         # path passes the stop (93.75) on the way to < liq (87.94)
    t = run1(d, sig_at(10, [0]), plain_cfg()).trades[0]
    assert t["exit_reason"] == "sl" and t["exit_price"] == pytest.approx(93.75)
    d = flat(10)
    d["o"][3] = d["h"][3] = d["l"][3] = d["c"][3] = 80.0   # gap straight through stop AND liq
    t = run1(d, sig_at(10, [0]), plain_cfg()).trades[0]
    assert t["exit_reason"] == "liquidation"
    assert t["exit_price"] == pytest.approx(80.0)
    # the whole isolated margin (20) is lost, not more: net = -(margin) - entry fee... fees included
    assert t["pnl"] == pytest.approx(-20.0 - FEE * 160.0)


def test_kill_switch_closes_all_and_blocks_new_entries():
    n = 12
    d = flat(n)
    for i in range(2, n):                    # -3 % from bar 2 on (inside SL of 6.25 %)
        d["o"][i] = d["h"][i] = d["l"][i] = d["c"][i] = 97.0
    kl = build(d)
    sigs = {s: sig_at(n, [0, 6]) for s in ("A", "B")}
    cfg = plain_cfg(start_equity=100.0, kill_dd_pct=5.0)
    r = run_backtest({"A": kl, "B": kl.copy()}, sigs, cfg)
    assert len(r.trades) == 2
    assert {t["exit_reason"] for t in r.trades} == {"kill"}
    assert r.kill_time == T0 + 2 * BAR
    assert r.stats["skip_kill"] == 2         # the row-6 signals are refused
    assert r.stats["kill_closes"] == 2
    # all closed at the bar-2 close of 97: equity = 100 - 2*(9.6 loss) - fees
    exp = 100 - 2 * 1.6 * 3 - 2 * FEE * 1.6 * 100 - 2 * FEE * 1.6 * 97
    assert r.equity[-1] == pytest.approx(exp)


def _walk(n, seed, sigma=0.0008):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * 1.0001
    l = np.minimum(o, c) * 0.9999
    return mk_klines(o, h, l, c)


def test_corr_cap_blocks_third_correlated_position():
    n = 400
    base = _walk(n, 1)
    kls = {"A": base, "B": base.copy(), "C": base.copy(), "E": _walk(n, 99)}
    sigs = {"A": sig_at(n, [300], prob=0.9), "B": sig_at(n, [300], prob=0.8),
            "C": sig_at(n, [300], prob=0.75), "E": sig_at(n, [300], prob=0.72)}
    cfg = plain_cfg(thr_long=0.7)
    r = run_backtest(kls, sigs, cfg)
    syms = {t["symbol"] for t in r.trades} | {p for p in ()}
    # A and B open first (highest prob); C is perfectly correlated with both -> blocked; E is not
    assert r.stats["skip_corr"] == 1
    assert r.stats["entries"] == 3
    # positions are still open at the end of data; check which ones via equity/trade counts
    opened_syms = r.stats["entries"]
    # same data without the cap opens all four
    r2 = run_backtest(kls, sigs, plain_cfg(thr_long=0.7, corr_max_correlated=99))
    assert r2.stats["entries"] == 4 and r2.stats["skip_corr"] == 0
    # opposite direction is not capped: C as a short with shorts enabled opens
    sigs3 = dict(sigs)
    sigs3["C"] = sig_at(n, [300], side="short", prob=0.95)
    r3 = run_backtest(kls, sigs3, plain_cfg(thr_long=0.7, short_enabled=True))
    assert r3.stats["skip_corr"] == 0 and r3.stats["entries"] == 4


def test_slot_cap_is_floor_equity_over_slice():
    n = 6
    kl = build(flat(n))
    kls = {s: kl.copy() for s in "ABC"}
    sigs = {s: sig_at(n, [0], prob=0.9 - 0.1 * i) for i, s in enumerate("ABC")}
    r = run_backtest(kls, sigs, plain_cfg(start_equity=45.0, thr_long=0.5, corr_max_correlated=99))
    assert r.stats["entries"] == 2 and r.stats["skip_no_slot"] == 1     # floor(45/20) = 2


def test_equity_equals_start_plus_sum_of_trade_pnl_when_flat():
    rng = np.random.default_rng(5)
    n = 3000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.concatenate([[100.0], c[:-1]])
    kl = mk_klines(o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c)
    sigs = {"X": sig_at(n, range(10, n - 400, 37))}
    r = run_backtest({"X": kl}, sigs, plain_cfg(slippage=0.0002, stop_slippage_extra=0.0005))
    assert len(r.trades) > 20
    open_end = r.stats["open_at_end"]
    if open_end == 0:
        assert r.equity[-1] == pytest.approx(1000 + sum(t["pnl"] for t in r.trades), abs=1e-6)
    assert r.metrics["n_trades"] == len(r.trades)
    assert r.metrics["fees_paid"] == pytest.approx(sum(t["fees"] for t in r.trades))


def test_breakeven_winrate_fixed():
    cfg = SimConfig()
    be = breakeven_winrate(cfg)
    N = 160.0
    win = 5 - N * 2 * 0.0005
    loss = 10 + N * (2 * 0.0005 + 0.0002 + 0.0005)
    assert be["breakeven_winrate"] == pytest.approx(loss / (win + loss))
    assert 0.67 < be["breakeven_winrate"] < 0.70           # above the zero-cost 66.7 %
    assert be["zero_cost_winrate"] == pytest.approx(10 / 15)
    ba = breakeven_winrate(SimConfig(exit_mode="atr"), atr_frac=0.003)
    assert ba["notional"] == pytest.approx(160.0)           # 10/0.003 = 3333 -> capped
    assert ba["breakeven_winrate"] > ba["zero_cost_winrate"] == pytest.approx(1.0 / 2.5)
    with pytest.raises(ValueError):
        breakeven_winrate(SimConfig(exit_mode="atr"))


def test_atr_mode_sizing_and_levels():
    cfg = SimConfig(exit_mode="atr", tp_atr=1.5, sl_atr=1.0, sl_usd=10.0)
    # ATR 1.0 at price 100: sl dist 1 -> qty = min(10/1, 160/100) = 1.6 (capped by the slice)
    qty, tp, sl = size_and_levels(100.0, 100.0, +1, 1.0, cfg)
    assert qty == pytest.approx(1.6) and tp == pytest.approx(101.5) and sl == pytest.approx(99.0)
    # large ATR: sl dist 20 -> qty = 10/20 = 0.5 so the SL loses exactly sl_usd
    qty, tp, sl = size_and_levels(100.0, 100.0, -1, 20.0, cfg)
    assert qty == pytest.approx(0.5) and sl == pytest.approx(120.0) and tp == pytest.approx(70.0)
    assert size_and_levels(100.0, 100.0, +1, float("nan"), cfg) is None
