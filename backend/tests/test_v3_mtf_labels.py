"""tb.labels additions: exit_mode "pct", atr_tf, rows (decision cadence), cooldown_select. Hand-computed."""
import numpy as np
import pandas as pd
import pytest

from tb.labels import (OUT_SL, OUT_TIMEOUT, OUT_TP, cooldown_select, exit_atr, full_cfg, label_trades,
                       simulate_trades)

BAR = 300_000
T0 = 1_700_006_400_000 - (1_700_006_400_000 % 86_400_000)      # UTC midnight


def frame(o, h, l, c):
    n = len(o)
    return pd.DataFrame({"open_time": np.arange(n) * BAR + T0, "open": o, "high": h, "low": l, "close": c,
                         "volume": 1.0, "quote_volume": 100.0, "trades": 1, "taker_buy_volume": 0.5,
                         "taker_buy_quote_volume": 50.0}).astype(
        {"open": float, "high": float, "low": float, "close": float})


def flat(n, px=100.0):
    return frame([px] * n, [px] * n, [px] * n, [px] * n)


PCT0 = dict(exit_mode="pct", tp_pct=3.0, sl_pct=6.0, slippage_pct=0.0, stop_slippage_pct=0.0, fee_taker=0.0005)
H = 6


def test_pct_long_tp():
    df = flat(20)
    df.loc[4, "high"] = 103.5                                   # signal 0: entry open[1]=100 -> TP 103.00 hit at bar 4
    sim = simulate_trades(df, "long", PCT0, H)
    assert sim["label"][0] == 1 and sim["outcome"][0] == OUT_TP and sim["exit_off"][0] == 4
    assert sim["notional"][0] == pytest.approx(160.0)           # slice 20 USD x 8
    # qty 1.6; gross 1.6*3 = 4.8; fees .0005*1.6*(100+103)
    assert sim["pnl_usd"][0] == pytest.approx(4.8 - 0.0005 * 1.6 * 203.0, abs=1e-6)


def test_pct_long_sl_and_levels_just_missed():
    df = flat(20)
    df.loc[3, "low"] = 94.1                                     # SL = 94.00: not touched
    df.loc[3, "high"] = 102.9                                   # TP = 103.00: not touched
    sim = simulate_trades(df, "long", PCT0, H)
    assert sim["outcome"][0] == OUT_TIMEOUT and sim["label"][0] == 0 and sim["exit_off"][0] == H
    df.loc[3, "low"] = 94.0                                     # touches the stop exactly
    sim = simulate_trades(df, "long", PCT0, H)
    assert sim["outcome"][0] == OUT_SL and sim["exit_off"][0] == 3
    assert sim["pnl_usd"][0] == pytest.approx(-9.6 - 0.0005 * 1.6 * (100 + 94), abs=1e-6)   # -6 % of 160


def test_pct_short_and_same_bar_sl_first():
    df = flat(20)
    df.loc[3, "low"] = 96.9                                     # short TP = 97.00 touched
    sim = simulate_trades(df, "short", PCT0, H)
    assert sim["label"][0] == 1 and sim["exit_off"][0] == 3
    assert sim["pnl_usd"][0] == pytest.approx(4.8 - 0.0005 * 1.6 * (100 + 97), abs=1e-6)
    df.loc[3, "high"] = 106.1                                   # SL = 106.00 also touched in the same bar
    sim = simulate_trades(df, "short", PCT0, H)
    assert sim["outcome"][0] == OUT_SL                          # pessimistic tie-break


def test_slippage_applied_to_pct_entry():
    df = flat(20)
    cfg = dict(PCT0, slippage_pct=0.001)
    df.loc[4, "high"] = 104.0
    sim = simulate_trades(df, "long", cfg, H)
    fill = 100.1                                                # long entry slips up
    tp = fill * 1.03
    qty = 160.0 / fill
    assert sim["pnl_usd"][0] == pytest.approx(qty * (tp - fill) - 0.0005 * qty * (fill + tp), abs=1e-6)


def test_rows_subset_matches_full_and_defaults_unchanged():
    rng = np.random.default_rng(0)
    n = 400
    c = 100 * np.exp(np.cumsum(0.004 * rng.standard_normal(n)))
    o = np.r_[c[0], c[:-1]]
    df = frame(o, np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998, c)
    cfg = dict(PCT0, slippage_pct=0.0002, stop_slippage_pct=0.0005)
    full = simulate_trades(df, "long", cfg, 50)
    rows = np.arange(5, 300, 12)
    sub = simulate_trades(df, "long", cfg, 50, rows=rows)
    for k in full:
        a, b = full[k][rows], sub[k][rows]
        assert np.array_equal(a, b, equal_nan=True), k
    other = np.setdiff1d(np.arange(n), rows)
    assert np.isnan(sub["label"][other]).all() and (sub["outcome"][other] == -1).all()
    # out-of-range rows are ignored, label_trades passes rows through
    lab = label_trades(df, "long", cfg, 50, rows=np.array([-3, 10, 349, 5000]))
    assert np.isfinite(lab[10]) and np.isfinite(lab[349]) and np.isnan(lab).sum() == n - 2   # 349 = last row with a full horizon
    # default config still the original fixed template
    assert full_cfg(None)["exit_mode"] == "fixed" and full_cfg(None)["atr_tf"] == "5m"
    with pytest.raises(ValueError):
        full_cfg({"exit_mode": "nope"})
    with pytest.raises(ValueError):
        full_cfg({"exit_mode": "atr", "atr_tf": "7m"})


def hourly_flat(n_hours, rng_half=1.0, px=100.0):
    """5m bars where every bar spans px +- rng_half -> every hourly true range = 2*rng_half, ATR14(1h) = 2."""
    n = 12 * n_hours
    return frame([px] * n, [px + rng_half] * n, [px - rng_half] * n, [px] * n)


def test_atr_tf_1h_levels_and_notional():
    df = hourly_flat(40)
    cfg = dict(exit_mode="atr", atr_tf="1h", tp_atr=3.0, sl_atr=2.0, sl_usd=1.0, slippage_pct=0.0,
               stop_slippage_pct=0.0, fee_taker=0.0005)
    a = exit_atr(df, cfg)
    assert a[-1] == pytest.approx(2.0)
    s = 12 * 20 + 3                                              # signal with 20 closed hours behind it
    i_hit = s + 5
    df.loc[i_hit, "high"] = 106.5                                # entry 100 -> TP = 100 + 3*2 = 106.0
    sim = simulate_trades(df, "long", cfg, 12, rows=np.array([s]))
    assert sim["outcome"][s] == OUT_TP and sim["exit_off"][s] == 5
    # notional = sl_usd / (sl_dist/entry) = 1 / (4/100) = 25 USD (below the 160 slice cap)
    assert sim["notional"][s] == pytest.approx(25.0)
    qty = 25.0 / 100.0
    assert sim["pnl_usd"][s] == pytest.approx(qty * 6.0 - 0.0005 * qty * (100 + 106), abs=1e-6)
    # the stop is 2*ATR = 4 below: low <= 96 -> SL
    df2 = hourly_flat(40)
    df2.loc[i_hit, "low"] = 95.9
    assert simulate_trades(df2, "long", cfg, 12, rows=np.array([s]))["outcome"][s] == OUT_SL


def test_atr_tf_ignores_the_forming_hour():
    df = hourly_flat(40)
    cfg = dict(exit_mode="atr", atr_tf="1h", tp_atr=3.0, sl_atr=2.0)
    base = exit_atr(df, cfg)
    df2 = df.copy()
    # blow up the ranges of hour 30 (bars 360..371): visible only from the close of its last bar (index 371)
    df2.loc[360:371, "high"] = 120.0
    a2 = exit_atr(df2, cfg)
    assert np.array_equal(a2[:371], base[:371], equal_nan=True)
    assert a2[371] > base[371] + 0.5
    # a 4h ATR needs 14 + 1 closed 4h bars: undefined early, defined later
    df4 = hourly_flat(24 * 5)
    a4 = exit_atr(df4, dict(exit_mode="atr", atr_tf="4h"))
    assert np.isnan(a4[12 * 4 * 14 - 2]) and a4[-1] == pytest.approx(2.0)


def test_atr_5m_default_is_unchanged():
    rng = np.random.default_rng(1)
    n = 500
    c = 100 * np.exp(np.cumsum(0.003 * rng.standard_normal(n)))
    o = np.r_[c[0], c[:-1]]
    df = frame(o, np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998, c)
    from tb.features import atr
    cfg = full_cfg(dict(exit_mode="atr"))
    assert np.array_equal(exit_atr(df, cfg), atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), 14),
                          equal_nan=True)


def test_cooldown_select_hand_example():
    idx = np.array([0, 3, 8, 9, 20, 22, 40])
    off = np.array([5, 5, 5, 5, 10, 5, 5])
    # trade@0 open until bar 5 -> bar 3 blocked; bar 8 ok (>=5), opens until 13 -> 9 blocked;
    # bar 20 ok until 30 -> 22 blocked; bar 40 ok
    assert list(cooldown_select(idx, off, 0)) == [True, False, True, False, True, False, True]
    # with a 4-bar cooldown: trade@0 blocks until 9 -> 8 blocked, 9 taken (until 14+4=18); 20 ok(>=18) until 34 -> 22 blocked; 40 ok
    assert list(cooldown_select(idx, off, 4)) == [True, False, False, True, True, False, True]
    assert list(cooldown_select(np.array([], dtype=int), np.array([]), 3)) == []
    # exit_off 0 is treated as 1 bar
    assert list(cooldown_select(np.array([0, 1, 2]), np.array([0, 0, 0]), 0)) == [True, True, True]
