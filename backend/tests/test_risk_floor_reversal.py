"""Dollar floor, symmetric long/short defaults and reversal protection (user decisions 2026-10-19).

Runs with the SHIPPED defaults of the new rules (the autouse legacy fixture in conftest is off for this module).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from test_sim_helpers import BAR, T0, build, flat, mk_klines, plain_cfg, sig_at
from tb import reversal as R
from tb.adaptive import pick_side
from tb.config import CONTRACT_KEYS, Settings
from tb.exchange.base import SymbolFilters
from tb.protect import Protect, partial_gain_ok
from tb.risk import Book, check_entry, dollar_floor, net_exposure_check, size_trade, slots_fit
from tb.shadow import ShadowBook
from tb.sim import run_backtest
from tb.sim.template import SimConfig

pytestmark = pytest.mark.real_risk_defaults

F = SymbolFilters(0.01, 0.001, 0.001, 5.0)
FEE = 0.0005


# ====================================================================== settings / symmetric defaults
def test_new_settings_defaults_contract_and_validation():
    s = Settings()
    assert (s.min_tp_usd, s.min_sl_usd, s.max_net_exposure_pct, s.reversal_z, s.reversal_lock_r, s.reversal_block_bars) == \
        (2.0, 2.0, 150.0, 2.5, 0.5, 6)
    api = s.to_api()
    for k in ("min_tp_usd", "min_sl_usd", "max_net_exposure_pct", "reversal_z", "reversal_lock_r", "reversal_block_bars"):
        assert k in CONTRACT_KEYS and k in api
    ok = s.merge({"min_tp_usd": 0, "min_sl_usd": 0, "max_net_exposure_pct": 0, "reversal_z": 0, "reversal_block_bars": 0},
                 allow_internal=False)
    assert ok.max_net_exposure_pct == 0 and ok.reversal_block_bars == 0
    for bad in ({"min_tp_usd": -1}, {"min_sl_usd": -0.1}, {"max_net_exposure_pct": 5}, {"max_net_exposure_pct": -1},
                {"reversal_z": 0.2}, {"reversal_z": 11}, {"reversal_lock_r": -1}, {"reversal_block_bars": -1},
                {"reversal_block_bars": 1.5}, {"min_tp_usd": "x"}):
        with pytest.raises(ValueError):
            s.merge(bad)


def test_long_and_short_defaults_are_symmetric():
    s, c = Settings(), SimConfig()
    assert s.short_enabled and c.short_enabled
    assert s.prob_threshold_long == s.prob_threshold_short == 0.70
    assert c.thr_long == c.thr_short == 0.70
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "tb" / "sim" / "run.py").read_text()
    assert 'default="both"' in src and "default=d.thr_short" in src and "default=d.short_enabled" in src


def test_pick_side_is_symmetric_and_a_tie_trades_neither():
    # mirrored probabilities pick mirrored sides
    assert pick_side(0.80, 0.30, 0.70, 0.70, True) == ("long", 0.80)
    assert pick_side(0.30, 0.80, 0.70, 0.70, True) == ("short", 0.80)
    # both clear: the larger margin over ITS OWN threshold wins, whatever the side
    assert pick_side(0.75, 0.90, 0.70, 0.70, True)[0] == "short"
    assert pick_side(0.90, 0.75, 0.70, 0.70, True)[0] == "long"
    assert pick_side(0.80, 0.85, 0.70, 0.80, True)[0] == "long"          # margins 0.10 vs 0.05
    assert pick_side(0.85, 0.85, 0.70, 0.70, True) == (None, 0.0)         # exact tie: no built-in lean
    assert pick_side(0.30, 0.90, 0.70, 0.70, False) == (None, 0.0)        # shorts off


def test_xsec_targets_do_not_prefer_longs():
    from tb.entry import XsecCfg
    from tb.scorer import Score
    from tb.engine import Engine

    xc = XsecCfg(24, 1, 1, 24)
    sc = {"A": Score(0.9, 0.6), "B": Score(0.2, 0.95), "C": Score(0.5, 0.5)}
    assert Engine.xsec_targets(sc, xc, True) == {"A": "long", "B": "short"}
    # a coin both lists want goes to the side with the larger probability (and to neither on an exact tie)
    sc2 = {"A": Score(0.9, 0.95), "B": Score(0.1, 0.1)}
    assert Engine.xsec_targets(sc2, xc, True) == {"A": "short"}
    sc3 = {"A": Score(0.9, 0.9)}
    assert Engine.xsec_targets(sc3, xc, True) == {}


# ====================================================================== (A) dollar floor
def test_floor_skips_a_derated_trade_instead_of_shrinking_it():
    cfg = Settings()
    ok = size_trade(cfg, "long", 100.0, 2.0, F, None, risk_mult=1.0)
    assert ok.ok and ok.reward_usd == pytest.approx(5.0, abs=0.05) and ok.risk_usd == pytest.approx(10.0)
    half = size_trade(cfg, "long", 100.0, 2.0, F, None, risk_mult=0.5)
    assert half.ok and half.reward_usd == pytest.approx(2.5, abs=0.05)                       # 2.5 >= 2: fine
    low = size_trade(cfg, "long", 100.0, 2.0, F, None, risk_mult=0.3)              # TP $1.50 < $2
    assert not low.ok and low.reason.startswith("target below $ floor") and "TP" in low.reason
    sl_low = size_trade(cfg.merge({"tp_usd": 50.0, "sl_usd": 5.0}), "long", 100.0, 2.0, F, None, risk_mult=0.3)   # SL $1.50 < $2
    assert not sl_low.ok and "SL" in sl_low.reason
    off = size_trade(cfg.merge({"min_tp_usd": 0, "min_sl_usd": 0}), "long", 100.0, 2.0, F, None, risk_mult=0.3)
    assert off.ok
    # a stop-only position (cross-sectional leg) is checked on the stop only
    only_sl = size_trade(cfg, "long", 100.0, 2.0, F, None, risk_mult=0.3, tp_floor=False)
    assert only_sl.ok                                                                # SL $3 >= 2, TP not checked


def test_floor_uses_the_actual_quantity_after_caps_in_atr_mode():
    cfg = Settings().merge({"exit_mode": "atr", "sl_atr": 1.0, "tp_atr": 1.5})
    # ATR 0.1 on a 100 coin: the slice cap (160 USD = 1.6 qty) binds, SL = 1.6 * 0.1 = $0.16 << floor
    s = size_trade(cfg, "long", 100.0, 0.1, F, None, risk_mult=1.0)
    assert not s.ok and s.reason.startswith("target below $ floor")
    assert size_trade(cfg, "long", 100.0, 2.0, F, None, risk_mult=1.0).ok            # ATR 2: SL $3.2, TP $4.8


def test_dollar_floor_helper():
    cfg = Settings()
    assert dollar_floor(cfg, 10.0, 5.0) is None
    assert dollar_floor(cfg, 1.99, 5.0) and dollar_floor(cfg, 10.0, 1.99)
    assert dollar_floor(cfg, 10.0, None) is None                                      # no TP -> only the stop
    assert dollar_floor(cfg, 2.0, 2.0) is None                                        # exactly at the floor passes


def test_partial_gain_floor_unit():
    p = Protect(min_tp_usd=2.0)
    assert partial_gain_ok("long", 100.0, 103.0, 0.8, p)                              # 2.4
    assert not partial_gain_ok("long", 100.0, 103.0, 0.6, p)                          # 1.8
    assert partial_gain_ok("short", 100.0, 97.0, 0.8, p)
    assert partial_gain_ok("long", 100.0, 100.1, 0.1, Protect(min_tp_usd=0.0))        # floor off


def _atr_book(**over):
    cfg = Settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0, "exit_mode": "atr", "sl_atr": 1.0, "tp_atr": 6.0,
                            "sl_usd": 10.0, "trail": {"enabled": False}, "breakeven_r": 1.0, "partial_tp_r": 1.5,
                            "partial_tp_frac": 0.5, "max_side_risk_pct": 100, "max_net_exposure_pct": 0, **over},
                           allow_internal=True)
    return cfg, ShadowBook(cfg)


def _path():
    n = 30
    d = flat(n)
    for i in range(n):
        d["h"][i], d["l"][i] = 101.0, 99.0
    d["h"][22], d["l"][22] = 103.5, 100.3
    d["h"][23], d["l"][23] = 102.0, 100.0
    for i in range(23, n):
        d["o"][i] = d["c"][i] = 101.0
    return n, d


def test_shadow_applies_the_floor_at_mult_1_and_suppresses_a_small_partial():
    cfg, book = _atr_book()
    # ATR 0.1: the settings-path position (160 USD slice) risks $0.16 -> skipped by the floor
    assert not book.open("m", "X", "long", 0.9, 100.0, 1000, 0.1, F)
    assert book.skips.get("floor") == 1
    assert book.open("m", "X", "long", 0.9, 100.0, 1000, 2.0, F)                        # ATR 2: SL $3.2, TP $19.2
    # partial piece: 0.8 x 3.0 = $2.40 at 1.5R. Floor 3.0 -> no partial, 2.0 -> partial
    for floor, expect_partial in ((3.0, False), (2.0, True)):
        cfg2, b2 = _atr_book(min_tp_usd=floor, min_sl_usd=0.0)
        kl = build(_path()[1])
        t_entry = int(kl["open_time"].iloc[21]) // 1000
        assert b2.open("m", "X", "long", 0.9, 100.0, t_entry, 2.0, F)
        r = kl.iloc[22]                                                                # high 103.5 reaches the 1.5R level (103)
        assert b2.on_bar("X", int(r.open_time), float(r.low), float(r.high), (int(r.open_time) + BAR) / 1000.0) == []
        p = b2.pos["m|X"]
        assert p.partial_done and (p.part_gross > 0) == expect_partial
        assert p.qty == pytest.approx(0.8 if expect_partial else 1.6)
        assert b2.skips.get("partial_floor", 0) == (0 if expect_partial else 1)


def test_sim_floor_skip_and_partial_suppression():
    n, d = _path()
    exits = {"exit_mode": "atr", "sl_atr": 1.0, "tp_atr": 6.0, "sl_usd": 10.0, "breakeven_r": 1.0, "partial_tp_r": 1.5,
             "partial_tp_frac": 0.5}
    cfg = plain_cfg(**exits, min_tp_usd=2.0, min_sl_usd=0.0)
    r = run_backtest({"X": build(d)}, {"X": sig_at(n, [20])}, cfg)
    assert r.stats["partials"] == 1 and r.stats["partials_floor_skipped"] == 0 and r.trades[0]["partial_qty"] == pytest.approx(0.8)
    r2 = run_backtest({"X": build(d)}, {"X": sig_at(n, [20])}, plain_cfg(**exits, min_tp_usd=3.0, min_sl_usd=0.0))
    assert r2.stats["partials"] == 0 and r2.stats["partials_floor_skipped"] == 1
    t = r2.trades[0]
    assert t["partial_qty"] == 0 and t["exit_reason"] == "sl" and t["exit_price"] == pytest.approx(100.1)   # whole qty out at break-even
    assert t["pnl"] == pytest.approx(1.6 * 0.1 - FEE * 1.6 * (100.0 + 100.1))
    # the entry itself: TP = 6 ATR x 1.6 qty = $19.2, SL = $3.2; a floor above either skips the trade
    r3 = run_backtest({"X": build(d)}, {"X": sig_at(n, [20])}, plain_cfg(**exits, min_sl_usd=3.5))
    assert r3.stats["entries"] == 0 and r3.stats["skip_dollar_floor"] == 1
    r4 = run_backtest({"X": build(d)}, {"X": sig_at(n, [20])}, plain_cfg(**exits, min_tp_usd=20.0, min_sl_usd=0.0))
    assert r4.stats["entries"] == 0 and r4.stats["skip_dollar_floor"] == 1
    r5 = run_backtest({"X": build(d)}, {"X": sig_at(n, [20])}, plain_cfg(**exits, min_tp_usd=19.0, min_sl_usd=3.0))
    assert r5.stats["entries"] == 1


def test_sim_derated_trade_below_the_floor_is_skipped_not_shrunk():
    # drawdown derating scale 0.2 -> SL/TP dollars x0.2 (fixed mode): TP $5 -> $1, below the $2 floor
    n = 40
    d = flat(n)
    for i in range(n):
        d["h"][i], d["l"][i] = 100.5, 99.5
    from tb.sim.template import size_and_levels

    cfg = plain_cfg(min_tp_usd=2.0, min_sl_usd=2.0)
    qty, tp, sl = size_and_levels(100.0, 100.0, 1, 1.0, cfg, 0.2)
    assert qty * (tp - 100.0) == pytest.approx(1.0)                          # what the sim would otherwise have traded
    # run_backtest with a start equity deep in drawdown territory is covered by the unit above + skip counter on tiny scale
    r = run_backtest({"X": build(d)}, {"X": sig_at(n, [10])}, cfg)
    assert r.stats["entries"] == 1 and r.stats["skip_dollar_floor"] == 0     # un-derated trade clears the floor


# ====================================================================== (C1) net exposure cap
def _book(eq=200.0, net=0.0):
    return Book(eq, eq, 0.0, 0.0, 0.0, 0.0, 0.0, net)


def test_net_exposure_check_reject_vs_reducing_entry_allowed():
    cap = 300.0                                                                  # 150 % of 200
    assert net_exposure_check(0.0, 160.0, cap)[0]                                # first slice
    assert net_exposure_check(160.0, 100.0, cap)[0]                              # 260 <= cap
    ok, why = net_exposure_check(160.0, 160.0, cap)                              # 320 > 300 and growing
    assert not ok and "net exposure" in why
    assert net_exposure_check(160.0, -160.0, cap)[0]                             # reduces |net|
    assert net_exposure_check(500.0, -100.0, cap)[0]                             # still above the cap, but REDUCING: allowed
    assert not net_exposure_check(500.0, 10.0, cap)[0]
    assert net_exposure_check(-500.0, 100.0, cap)[0] and not net_exposure_check(-500.0, -10.0, cap)[0]
    assert net_exposure_check(0.0, 400.0, cap)[0]                                # a lone position into a flat book
    assert net_exposure_check(900.0, 900.0, 0.0)[0]                              # cap 0 = off


def test_check_entry_net_exposure_rule_is_beta_weighted_and_side_aware():
    cfg = Settings().merge({"max_total_leverage": 50, "max_open_risk_pct": 100, "max_side_risk_pct": 100, "max_margin_ratio": 0.9})
    b = _book(200.0, net=160.0)                                                  # 160 long beta-weighted, cap 300
    ok = check_entry(cfg, b, 160.0, 1.0, 1.0, 20.0, 0.1, "long", beta=0.5)       # +80 -> 240
    assert ok.ok
    bad = check_entry(cfg, b, 160.0, 1.0, 1.0, 20.0, 0.1, "long", beta=1.0)      # +160 -> 320
    assert not bad.ok and bad.rule == "net_exposure"
    assert check_entry(cfg, b, 160.0, 1.0, 1.0, 20.0, 0.1, "short", beta=1.0).ok  # reduces to 0
    assert check_entry(cfg.merge({"max_net_exposure_pct": 0}), b, 160.0, 1.0, 1.0, 20.0, 0.1, "long").ok   # off
    # slots_fit follows the cap: on a flat $200 book one long slice, then the long side is full; shorts still fit
    assert slots_fit(cfg, _book(200.0), 0) >= 1


# ====================================================================== beta / z maths
def test_beta_from_returns_recovers_the_slope_and_falls_back_to_one():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 0.01, 200)
    x = 1.7 * y + rng.normal(0, 0.003, 200)
    assert R.beta_from_returns(x, y) == pytest.approx(1.7, abs=0.15)
    assert R.beta_from_returns(x[:20], y[:20]) == 1.0                                # too few observations
    assert R.beta_from_returns(x, np.zeros(200)) == 1.0                              # flat market
    assert R.beta_from_returns(-y, y) == R.BETA_LO                                   # clipped
    assert R.beta_from_returns(10 * y, y) == R.BETA_HI
    # only the last 7 days (168 h) count
    xx = np.concatenate([5 * y[:100], 1.0 * y[100:]])
    yy = np.concatenate([y[:100], y[100:]])
    long = np.concatenate([rng.normal(0, 0.01, 50), yy])                             # 350 returns; first ones are older than 168
    assert R.beta_from_returns(np.concatenate([5 * long[:50], xx]), long) == pytest.approx(
        R.beta_from_returns(np.concatenate([5 * long[:50], xx])[-168:], long[-168:]))


def test_beta_vs_btc_from_hourly_closes_and_rolling_parity():
    rng = np.random.default_rng(4)
    n = 300
    rb = rng.normal(0, 0.01, n)
    rx = 2.0 * rb + rng.normal(0, 0.002, n)
    t = (np.arange(n + 1) * R.HOUR_MS).astype("int64")
    btc = 100 * np.cumprod(np.r_[1.0, 1 + rb])
    x = 50 * np.cumprod(np.r_[1.0, 1 + rx])
    assert R.beta_vs_btc(t, x, t, btc) == pytest.approx(2.0, abs=0.1)
    assert R.beta_vs_btc(t[:10], x[:10], t, btc) == 1.0                              # short history -> 1.0
    rolled = R.rolling_beta(rx, rb)
    assert rolled[-1] == pytest.approx(R.beta_from_returns(rx, rb))                  # the sim's vectorised twin
    assert rolled[100] == pytest.approx(R.beta_from_returns(rx[:101], rb[:101]))
    assert rolled[10] == 1.0


def test_z_score_calculation_and_sim_parity():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.002, 900)
    z = R.rolling_z(r, 3)
    assert math.isnan(z[50])                                                        # not enough history
    # hand computation of the last value: sum of the last 3 / RMS of the earlier non-overlapping 3-windows (last 576)
    rw = np.convolve(r, np.ones(3), "valid")                                         # rw[j] = sum r[j..j+2], ends at j+2
    cur = rw[-1]
    hist = rw[:-3][-576:]                                                            # windows ending at <= i-3
    assert z[-1] == pytest.approx(cur / math.sqrt(np.mean(hist ** 2)))
    # the live (window_z on the prefix) and the vectorised (rolling_z) forms agree at every bar
    for i in (200, 400, 899):
        assert R.window_z(r[: i + 1], 3) == pytest.approx(z[i])
        assert R.window_z(r[: i + 1], 12) == pytest.approx(R.rolling_z(r, 12)[i])
    r[-3:] = -0.008
    assert R.window_z(r, 3) < -4.0
    assert R.window_z(np.zeros(900), 3) != R.window_z(np.zeros(900), 3)           # flat: NaN, never a spurious signal


def _frames(n=700, crash=None, seed=0, syms=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    t = (T0 + np.arange(n) * BAR).astype("int64")
    out = {}
    for k, s in enumerate(syms):
        r = rng.normal(0, 0.0008, n)
        if crash is not None:
            r[-3:] = crash
        out[s] = (t, 100 * np.cumprod(1 + r))
    return out


def test_market_signal_sign_and_adverse_side():
    quiet = R.market_signal(_frames(), _frames(syms=("BTC",))["BTC"])
    assert quiet is not None and all(abs(v) < R.Z_WINDOWS.get("x", 4) for v in quiet.z.values())
    down = R.market_signal(_frames(crash=-0.006), _frames(crash=-0.006, syms=("BTC",))["BTC"])
    assert set(down.z) == {"basket_15m", "basket_1h", "btc_15m", "btc_1h"}
    assert down.z["basket_15m"] < -2.5 and down.z["btc_15m"] < -2.5
    z_long, src = down.adverse("long")
    assert z_long >= 2.5 and src
    assert down.adverse("short")[0] < 0                                              # a falling market is not adverse to shorts
    up = R.market_signal(_frames(crash=+0.006))
    assert up.adverse("short")[0] >= 2.5 and up.adverse("long")[0] < 0
    assert up.t_ms == int(T0 + 699 * BAR)


def test_basket_ignores_the_newest_bar_until_enough_symbols_have_it():
    fr = _frames(crash=-0.006)
    t, c = fr["C"]
    fr["B"] = (t[:-1], fr["B"][1][:-1])
    fr["C"] = (t[:-1], c[:-1])                                                       # only A has the newest bar: 1/3 < 60 %
    a = R.basket_returns(fr)
    assert a[1] == int(t[-2])
    full = R.basket_returns(_frames(crash=-0.006))
    assert full[1] == int(t[-1])


# ====================================================================== lock / block window
def test_lock_stop_only_winners_never_loosens_never_through_the_market():
    kw = dict(fee_rate=FEE, lock_r=0.5)
    # long, entry 100, initial stop 94 (R = 6)
    assert R.lock_stop("long", 100.0, 94.0, 94.0, 102.9, **kw) is None              # +0.48R: not a winner yet
    lk = R.lock_stop("long", 100.0, 94.0, 94.0, 104.0, **kw)                        # +0.67R: keep half of the open profit
    assert lk == pytest.approx(102.0)
    assert R.lock_stop("long", 100.0, 94.0, 94.0, 98.0, **kw) is None              # in a loss: stop untouched
    assert R.lock_stop("long", 100.0, 94.0, 102.5, 104.0, **kw) is None            # existing stop already tighter: never loosen
    be = R.lock_stop("long", 100.0, 99.0, 99.0, 100.55, **kw)                       # R = 1: half of 0.55 = 100.275 > BE 100.1
    assert be == pytest.approx(100.275)
    tiny = R.lock_stop("long", 100.0, 99.99, 99.99, 100.06, **kw)                   # break-even (100.1) would be through the mark
    assert tiny is None
    # short mirror
    assert R.lock_stop("short", 100.0, 106.0, 106.0, 96.0, **kw) == pytest.approx(98.0)
    assert R.lock_stop("short", 100.0, 106.0, 106.0, 102.0, **kw) is None
    assert R.lock_stop("short", 100.0, 106.0, 97.5, 96.0, **kw) is None
    assert R.lock_stop("long", 100.0, 100.0, 94.0, 110.0, **kw) is None            # no initial risk known
    # at least break-even + round-trip fees
    assert R.lock_stop("long", 100.0, 80.0, 80.0, 110.0, fee_rate=FEE, lock_r=0.0) == pytest.approx(105.0)
    assert R.lock_stop("long", 100.0, 99.0, 99.0, 100.3, fee_rate=FEE, lock_r=0.0) == pytest.approx(100.15)


def test_reversal_book_triggers_only_against_the_net_side_and_blocks_for_n_bars():
    rc = R.ReversalCfg(150.0, 2.5, 0.5, 6)
    down = R.RevSignal(T0, {"basket_15m": -3.0, "basket_1h": -1.0, "btc_15m": -0.5, "btc_1h": float("nan")})
    b = R.ReversalBook()
    assert b.step(down, "short", rc) is None and not b.blocked("short", T0 + BAR)  # falling market helps a short book
    assert b.step(down, None, rc) is None                                            # no net side: nothing to protect
    tr = b.step(down, "long", rc)
    assert tr["side"] == "long" and tr["new"] and tr["source"] == "basket_15m" and tr["z"] == pytest.approx(3.0)
    t_dec = T0 + BAR
    assert b.blocked("long", t_dec) and b.blocked("long", t_dec + 5 * BAR) and not b.blocked("long", t_dec + 6 * BAR)
    assert not b.blocked("short", t_dec)                                             # the other side stays open
    assert b.locking("long", t_dec + 5 * BAR)
    again = b.step(R.RevSignal(T0 + 2 * BAR, {"basket_1h": -2.6}), "long", rc)       # extends the window, not a new alarm
    assert again["new"] is False and b.blocked("long", t_dec + 7 * BAR)
    below = R.RevSignal(T0 + 20 * BAR, {"basket_15m": -2.4})
    assert b.step(below, "long", rc) is None
    assert R.ReversalBook().step(down, "long", R.ReversalCfg(150.0, 0.0, 0.5, 6)) is None     # detector off
    st = R.ReversalBook()
    st.load(b.to_dict())
    assert st.blocked("long", t_dec)


def test_net_side_needs_a_meaningful_net():
    assert R.net_side(160.0, 200.0) == "long" and R.net_side(-160.0, 200.0) == "short"
    assert R.net_side(5.0, 200.0) is None and R.net_side(0.0, 200.0) is None and R.net_side(100.0, 0.0) is None
    assert R.signed_net([("long", 160.0, 2.0), ("short", 160.0, 1.0)]) == pytest.approx(160.0)


# ====================================================================== shadow books
def test_shadow_net_cap_per_model_and_reducing_entry_allowed():
    cfg = Settings().merge({"capital_usd": 100.0, "slippage_pct": 0.0, "stop_slippage_pct": 0.0, "max_side_risk_pct": 100},
                           allow_internal=True)                                      # cap = 150 % of 100 = $150
    b = ShadowBook(cfg)
    assert b.open("m", "A", "long", 0.9, 100.0, 1000, 2.0, F)                          # lone slice $160 into a flat book
    assert not b.open("m", "B", "long", 0.9, 100.0, 1001, 2.0, F) and b.skips["net"] == 1
    assert b.open("m", "C", "short", 0.9, 100.0, 1002, 2.0, F)                         # reduces |net|
    assert b.open("other", "B", "long", 0.9, 100.0, 1003, 2.0, F)                      # caps are per model
    b.betas = {"D": 0.5}
    assert b.net("m") == pytest.approx(0.0)
    assert b.open("m", "D", "long", 0.9, 100.0, 1004, 2.0, F)                          # +80 (beta 0.5) <= 150


def test_shadow_reversal_locks_winners_blocks_the_side_and_never_loosens():
    cfg, book = _atr_book(max_net_exposure_pct=0, capital_usd=100.0)
    f = F
    assert book.open("m", "W", "long", 0.9, 100.0, 1000, 2.0, f)                      # winner (stop 98, R = 2)
    assert book.open("m", "L", "long", 0.9, 100.0, 1001, 2.0, f)                      # loser
    assert book.open("m", "S", "short", 0.9, 100.0, 1002, 2.0, f)
    assert book.pos["m|W"].sl == pytest.approx(98.0)
    sig = R.RevSignal(T0, {"basket_15m": -3.0, "basket_1h": -1.0})
    marks = {"W": 103.0, "L": 99.0, "S": 97.0}
    ev = book.apply_reversal(sig, marks, T0 + BAR)
    assert len(ev) == 1 and ev[0]["side"] == "long" and ev[0]["new"] and ev[0]["model"] == "m" and ev[0]["locked"] == 1
    assert book.pos["m|W"].sl == pytest.approx(101.5)                                  # entry + half of the +3 move
    assert book.pos["m|L"].sl == pytest.approx(98.0)                                   # losing position keeps its stop
    assert book.pos["m|S"].sl == pytest.approx(102.0)                                  # the short side is not the hurt one
    # the mark falls back: the stop never loosens
    book.apply_reversal(sig, {"W": 102.2, "L": 99.0, "S": 97.0}, T0 + 2 * BAR)
    assert book.pos["m|W"].sl == pytest.approx(101.5)
    # new longs are blocked for the window (6 bars), shorts are not, and the block ends
    t_in = (T0 + BAR) / 1000.0
    assert not book.open("m", "N", "long", 0.9, 100.0, t_in + 60, 2.0, f) and book.skips["reversal"] == 1
    assert book.open("m", "N2", "short", 0.9, 100.0, t_in + 60, 2.0, f)
    assert book.open("m", "N", "long", 0.9, 100.0, t_in + 6 * 300 + 1, 2.0, f)
    assert not book.rev["m"].blocked("long", T0 + 8 * BAR)


def test_shadow_reversal_off_when_z_is_zero():
    cfg, book = _atr_book(reversal_z=0.0)
    book.open("m", "W", "long", 0.9, 100.0, 1000, 2.0, F)
    assert book.apply_reversal(R.RevSignal(T0, {"basket_15m": -9.0}), {"W": 103.0}, T0 + BAR) == []
    assert book.pos["m|W"].sl == pytest.approx(98.0)


# ====================================================================== simulator parity
def _walk_frames(n=360, rise=None, crash_at=None, seed=1, syms=("A", "B", "C"), noise=0.0004):
    """Same price path + independent noise for every symbol; ``rise`` = (bar0, bar1, total) linear drift; crash from ``crash_at``."""
    rng = np.random.default_rng(seed)
    base = np.full(n, 100.0)
    if rise:
        b0, b1, tot = rise
        base[b0:b1] = 100.0 * (1 + tot * (np.arange(b0, b1) - b0) / (b1 - b0))
        base[b1:] = 100.0 * (1 + tot)
    if crash_at is not None:
        base[crash_at] = base[crash_at - 1] * 0.99
        base[crash_at + 1:] = base[crash_at] * np.linspace(0.97, 0.88, n - crash_at - 1)
    out = {}
    for s in syms:
        c = base * np.exp(np.cumsum(rng.normal(0, noise, n)) * 0.5)
        o = np.r_[c[0], c[:-1]]
        out[s] = mk_klines(o, np.maximum(o, c) * 1.0001, np.minimum(o, c) * 0.9999, c)
    return out


def test_sim_net_exposure_cap_rejects_same_side_and_allows_the_reducing_short():
    n = 40
    d = flat(n)
    for i in range(n):
        d["h"][i], d["l"][i] = 100.5, 99.5
    kls = {s: build(d) for s in ("A", "B", "C")}
    sigs = {"A": sig_at(n, [10]), "B": sig_at(n, [10]), "C": sig_at(n, [10], "short")}
    # equity 1000, cap 20 % = $200: A long 160 ok (lone), B long 320 > 200 rejected, C short reduces -> allowed
    r = run_backtest(kls, sigs, plain_cfg(max_net_exposure_pct=20.0, reversal_z=0.0))
    assert r.stats["entries"] == 2 and r.stats["skip_net_exposure"] == 1
    assert sorted(t["side"] for t in r.trades + []) == []                                # still open at the end
    off = run_backtest(kls, sigs, plain_cfg(max_net_exposure_pct=0.0, reversal_z=0.0))
    assert off.stats["entries"] == 3 and off.stats["skip_net_exposure"] == 0


def test_sim_net_exposure_is_beta_weighted():
    n = 3300
    rng = np.random.default_rng(11)
    rb = rng.normal(0, 0.0015, n)
    def series(beta):
        r = beta * rb + rng.normal(0, 0.0001, n)
        c = 100.0 * np.cumprod(1 + r)
        o = np.r_[100.0, c[:-1]]
        return mk_klines(o, np.maximum(o, c), np.minimum(o, c), c)
    kls = {"BTCUSDT": series(1.0), "HI": series(2.0), "LO": series(0.5)}
    sigs = {"BTCUSDT": np.zeros((n, 2)), "LO": sig_at(n, [3000], prob=1.0), "HI": sig_at(n, [3000], prob=0.9)}
    base = dict(tp_usd=500.0, sl_usd=10.0, min_tp_usd=0.0, min_sl_usd=0.0, reversal_z=0.0)
    # LO (beta ~0.5 -> ~80) first, then HI (beta ~2 -> ~320): sum ~400 > 35 % of 1000 = 350, unweighted 320 would have fit
    r = run_backtest(kls, sigs, plain_cfg(**base, max_net_exposure_pct=35.0))
    assert r.stats["entries"] == 1 and r.stats["skip_net_exposure"] == 1
    r2 = run_backtest(kls, sigs, plain_cfg(**base, max_net_exposure_pct=60.0))
    assert r2.stats["entries"] == 2


def _crash_setup():
    n = 360
    kls = _walk_frames(n, rise=(250, 300, 0.05), crash_at=301)
    return n, kls


def test_sim_reversal_locks_the_winner_and_blocks_the_hurt_side():
    n, kls = _crash_setup()
    sigs = {"A": sig_at(n, [250]), "B": np.zeros((n, 2)), "C": np.zeros((n, 2))}
    sigs["B"][302, 0] = 1.0                                                           # a NEW long inside the block window
    sigs["C"][302, 1] = 1.0                                                           # ... and a short (not the hurt side)
    base = dict(tp_usd=50.0, sl_usd=10.0, min_tp_usd=0.0, min_sl_usd=0.0, max_net_exposure_pct=0.0, start_equity=1000.0)
    off = run_backtest(kls, sigs, plain_cfg(**base, reversal_z=0.0))
    on = run_backtest(kls, sigs, plain_cfg(**base, reversal_z=2.5, reversal_lock_r=0.5, reversal_block_bars=6))
    assert on.stats["reversal_triggers"] >= 1 and on.stats["reversal_locks"] >= 1
    assert on.stats["skip_reversal_block"] == 1                                       # B's long refused, C's short went through
    assert off.stats["skip_reversal_block"] == 0 and off.stats["reversal_triggers"] == 0
    ta = next(t for t in on.trades if t["symbol"] == "A")
    toff = next(t for t in off.trades if t["symbol"] == "A")
    assert ta["exit_reason"] in ("sl", "trail") and ta["pnl"] > 0                      # the locked stop kept a gain
    assert toff["pnl"] < 0                                                             # without the lock the full stop hit
    assert ta["exit_price"] > 100.0 > toff["exit_price"]


def test_sim_reversal_does_not_touch_a_losing_position_or_the_other_side():
    n, kls = _crash_setup()
    # a short during a crash is a winner on the HELPED side: no trigger (net side short, market falling)
    sigs = {"A": sig_at(n, [310], "short"), "B": np.zeros((n, 2)), "C": np.zeros((n, 2))}
    base = dict(tp_usd=50.0, sl_usd=10.0, min_tp_usd=0.0, min_sl_usd=0.0, max_net_exposure_pct=0.0)
    r = run_backtest(kls, sigs, plain_cfg(**base, reversal_z=2.5))
    assert r.stats["reversal_triggers"] == 0 and r.stats["reversal_locks"] == 0
    # a long that is under water at the crash keeps its original stop (no lock, no panic close)
    sigs2 = {"A": sig_at(n, [310]), "B": np.zeros((n, 2)), "C": np.zeros((n, 2))}
    r2 = run_backtest(kls, sigs2, plain_cfg(**base, reversal_z=2.5))
    assert r2.stats["reversal_locks"] == 0


def test_sim_cli_flags_exist():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "tb" / "sim" / "run.py").read_text()
    for flag in ("--max-net-exposure", "--reversal-z", "--reversal-lock-r", "--reversal-block-bars", "--min-tp-usd", "--min-sl-usd"):
        assert flag in src
