"""Hand-computed + brute-force cross-checks for tb.labels."""
import numpy as np
import pandas as pd
import pytest

from tb.labels import (OUT_SL, OUT_TIMEOUT, OUT_TP, breakeven_from_cfg, breakeven_win_rate, label_trades,
                       simulate_trades)

BAR = 300_000


def frame(o, h, l, c):
    n = len(o)
    return pd.DataFrame({"open_time": np.arange(n) * BAR + 1_700_000_000_000, "open": o, "high": h, "low": l,
                         "close": c, "volume": 1.0, "quote_volume": 100.0, "trades": 1,
                         "taker_buy_volume": 0.5, "taker_buy_quote_volume": 50.0}).astype(
        {"open": float, "high": float, "low": float, "close": float})


def flat(n, px=100.0):
    return frame([px] * n, [px] * n, [px] * n, [px] * n)


def set_bar(df, i, high=None, low=None):
    if high is not None:
        df.loc[i, "high"] = high
    if low is not None:
        df.loc[i, "low"] = low
    return df


FIXED0 = dict(exit_mode="fixed", slippage_pct=0.0, stop_slippage_pct=0.0, fee_taker=0.0005)
H = 4


def test_fixed_long_tp_levels_and_pnl():
    # notional 160; TP = +5 USD => +3.125% => 103.125; SL = -10 USD => -6.25% => 93.75
    df = flat(12)
    set_bar(df, 3, high=103.2)         # signal bar 0 -> entry open[1]=100 -> TP touched at bar 3 (offset 3)
    sim = simulate_trades(df, "long", FIXED0, H)
    assert sim["label"][0] == 1 and sim["outcome"][0] == OUT_TP and sim["exit_off"][0] == 3
    # qty 1.6, gross 5.0, fees .0005*1.6*(100+103.125)=0.1625
    assert sim["pnl_usd"][0] == pytest.approx(5.0 - 0.1625, abs=1e-4)
    # signal bar 1: entry open[2]=100, TP touched at bar 3 -> offset 2
    assert sim["exit_off"][1] == 2 and sim["label"][1] == 1


def test_fixed_not_quite_tp():
    df = flat(12)
    set_bar(df, 3, high=103.1)         # below 103.125
    assert label_trades(df, "long", FIXED0, H)[0] == 0       # timeout
    sim = simulate_trades(df, "long", FIXED0, H)
    assert sim["outcome"][0] == OUT_TIMEOUT
    # timeout: exit at close[0+4]=100 -> gross 0, fees .0005*1.6*200 = 0.16
    assert sim["pnl_usd"][0] == pytest.approx(-0.16, abs=1e-5)


def test_fixed_sl_and_pnl():
    df = flat(12)
    set_bar(df, 2, low=93.7)           # SL 93.75
    sim = simulate_trades(df, "long", FIXED0, H)
    assert sim["label"][0] == 0 and sim["outcome"][0] == OUT_SL and sim["exit_off"][0] == 2
    assert sim["pnl_usd"][0] == pytest.approx(-10.0 - 0.0005 * 1.6 * (100 + 93.75), abs=1e-4)


def test_same_bar_tp_and_sl_is_sl_first():
    df = flat(12)
    set_bar(df, 2, high=110.0, low=90.0)
    sim = simulate_trades(df, "long", FIXED0, H)
    assert sim["label"][0] == 0 and sim["outcome"][0] == OUT_SL and sim["exit_off"][0] == 2
    # TP in an earlier bar than the SL still wins
    df = flat(12)
    set_bar(df, 2, high=110.0)
    set_bar(df, 3, low=90.0)
    assert label_trades(df, "long", FIXED0, H)[0] == 1


def test_entry_is_next_bar_open_not_signal_close():
    df = flat(12)
    set_bar(df, 0, high=120.0, low=80.0)         # the signal bar's own range must not count
    assert label_trades(df, "long", FIXED0, H)[0] == 0
    df = flat(12)
    df.loc[1, ["open", "high", "low", "close"]] = [200.0, 200.0, 200.0, 200.0]   # entry at 200
    # later bars sit at 100 -> well below the long's SL (187.5): SL on bar 2 (offset 2)
    sim = simulate_trades(df, "long", FIXED0, H)
    assert sim["outcome"][0] == OUT_SL and sim["exit_off"][0] == 2


def test_short_mirror():
    df = flat(12)
    set_bar(df, 3, low=96.8)           # TP for short = 96.875
    sim = simulate_trades(df, "short", FIXED0, H)
    assert sim["label"][0] == 1 and sim["exit_off"][0] == 3
    assert sim["pnl_usd"][0] == pytest.approx(5.0 - 0.0005 * 1.6 * (100 + 96.875), abs=1e-4)
    df = flat(12)
    set_bar(df, 2, high=106.3)         # SL for short = 106.25
    assert label_trades(df, "short", FIXED0, H)[0] == 0
    assert simulate_trades(df, "short", FIXED0, H)["outcome"][0] == OUT_SL


def test_nan_when_not_enough_future():
    df = flat(12)
    lab = label_trades(df, "long", FIXED0, H)
    assert np.isnan(lab[12 - H:]).all()          # i + H > N-1 -> NaN
    assert not np.isnan(lab[: 12 - H]).any()
    assert np.isnan(label_trades(flat(3), "long", FIXED0, H)).all()


def test_atr_mode_levels():
    # constant 99..101 range around 100 => TR = 2 => ATR14 = 2 (after warm-up)
    n = 60
    df = frame([100.0] * n, [101.0] * n, [99.0] * n, [100.0] * n)
    cfg = dict(exit_mode="atr", tp_atr=1.5, sl_atr=1.0, slippage_pct=0.0, stop_slippage_pct=0.0, fee_taker=0.0005)
    lab = label_trades(df, "long", cfg, 10)
    assert np.isnan(lab[:13]).all() and not np.isnan(lab[13])    # ATR needs 14 bars
    # TP = 103, SL = 98.  Bar 25 reaches 103.2 -> signal 20 (entry open[21]) wins at offset 5
    df.loc[25, "high"] = 103.2
    sim = simulate_trades(df, "long", cfg, 10)
    assert sim["label"][20] == 1 and sim["exit_off"][20] == 5
    # notional = min(160, 10 / (2/100)) = 160 ; qty 1.6; gross 4.8; fees .0005*1.6*(203)
    assert sim["pnl_usd"][20] == pytest.approx(1.6 * 3 - 0.0005 * 1.6 * 203, abs=1e-4)
    assert sim["notional"][20] == pytest.approx(160.0)
    # SL 98: low 97.9 at bar 24 (TP at 25 is later) -> SL
    df.loc[24, "low"] = 97.9
    sim = simulate_trades(df, "long", cfg, 10)
    assert sim["label"][20] == 0 and sim["outcome"][20] == OUT_SL and sim["exit_off"][20] == 4
    assert sim["pnl_usd"][20] == pytest.approx(-1.6 * 2 - 0.0005 * 1.6 * 198, abs=1e-4)


def test_atr_position_sizing_cap():
    # huge ATR% (ATR=10 on price 100 -> SL dist 10%): notional = 10/0.10 = 100 < 160
    n = 40
    df = frame([100.0] * n, [105.0] * n, [95.0] * n, [100.0] * n)
    cfg = dict(exit_mode="atr", slippage_pct=0.0, stop_slippage_pct=0.0)
    sim = simulate_trades(df, "long", cfg, 5)
    assert sim["notional"][20] == pytest.approx(100.0)


def test_slippage_applied():
    df = flat(12)
    set_bar(df, 2, low=93.0)
    cfg = dict(exit_mode="fixed", slippage_pct=0.001, stop_slippage_pct=0.002, fee_taker=0.0)
    sim = simulate_trades(df, "long", cfg, H)
    fill = 100 * 1.001
    exit_px = fill * (1 - 10 / 160) * (1 - 0.003)      # levels measured from the slipped fill
    qty = 160 / fill
    assert sim["pnl_usd"][0] == pytest.approx(qty * (exit_px - fill), abs=1e-4)


def test_gap_through_stop_fills_at_open():
    df = flat(12)
    df.loc[3, ["open", "high", "low", "close"]] = [90.0, 91.0, 89.0, 90.0]      # gaps below SL 93.75
    cfg = dict(exit_mode="fixed", slippage_pct=0.0, stop_slippage_pct=0.0, fee_taker=0.0)
    sim = simulate_trades(df, "long", cfg, H)
    assert sim["outcome"][0] == OUT_SL and sim["exit_off"][0] == 3
    assert sim["pnl_usd"][0] == pytest.approx(1.6 * (90.0 - 100.0), abs=1e-4)


def brute(df, side, cfg, H):
    from tb.features import atr
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(h, l, c, 14)
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(n):
        if i + H > n - 1:
            continue
        s = 1 if side == "long" else -1
        e = o[i + 1] * (1 + s * 0.0002)        # default slippage_pct on the entry fill
        if cfg["exit_mode"] == "fixed":
            td, sd = 5 / 160 * e, 10 / 160 * e
        else:
            if not np.isfinite(a[i]):
                continue
            td, sd = 1.5 * a[i], 1.0 * a[i]
        tp, sl = e + s * td, e - s * sd
        res = 0.0
        for j in range(i + 1, i + H + 1):
            th = h[j] >= tp if s > 0 else l[j] <= tp
            sh = l[j] <= sl if s > 0 else h[j] >= sl
            if sh:
                res = 0.0
                break
            if th:
                res = 1.0
                break
        out[i] = res
    return out


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["fixed", "atr"])
def test_matches_bruteforce_on_random_walk(side, mode):
    rng = np.random.default_rng(5)
    n = 700
    r = 0.004 * rng.standard_normal(n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * (1 + 0.003 * rng.random(n))
    l = np.minimum(o, c) * (1 - 0.003 * rng.random(n))
    df = frame(o, h, l, c)
    cfg = dict(exit_mode=mode)
    Hh = 60
    got = label_trades(df, side, cfg, Hh)
    want = brute(df, side, cfg, Hh)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.array_equal(got[~np.isnan(got)], want[~np.isnan(want)].astype("float32"))
    assert 0 < np.nanmean(got) < 1


def test_breakeven_numbers():
    be = breakeven_from_cfg(dict(exit_mode="fixed"))
    assert 0.67 < be < 0.69          # the "~68 %" in the architecture doc
    # atr mode: the notional is capped at the 160 slice, so for a 5m ATR of ~0.5 % the SL loses < 1 USD
    # and fixed fees dominate; the "~42 %" in the doc is the zero-fee limit (1/(1+1.5) = 40 %).
    b05, b1, b3 = (breakeven_from_cfg(dict(exit_mode="atr"), a) for a in (0.5, 1.0, 3.0))
    assert b05 > b1 > b3 > 0.40 and b05 > 0.5
    assert breakeven_from_cfg(dict(exit_mode="atr", fee_taker=0, slippage_pct=0, stop_slippage_pct=0)) == pytest.approx(0.4)


def test_breakeven_from_simulation():
    rng = np.random.default_rng(1)
    n = 3000
    c = 100 * np.exp(np.cumsum(0.003 * rng.standard_normal(n)))
    o = np.concatenate([[100.0], c[:-1]])
    df = frame(o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c)
    sim = simulate_trades(df, "long", dict(exit_mode="fixed"), 100)
    be = breakeven_win_rate(sim)
    assert 0.3 < be < 0.75      # timeouts (pnl ~ 0) are label-0 and pull it down
    oc = sim["outcome"]
    pn = sim["pnl_usd"]
    w, lo = np.nanmean(pn[oc == OUT_TP]), np.nanmean(pn[(oc != OUT_TP) & (oc != -1)])
    assert be * w + (1 - be) * lo == pytest.approx(0, abs=1e-4)
