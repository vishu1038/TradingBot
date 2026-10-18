"""Loader alignment helpers + quality report / plausibility checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

from test_sim_helpers import BAR, T0, mk_klines
from tb.data import frames, loader, quality
from tb.data.paths import METRIC_COLS, kline_path, metrics_path, funding_path


def _write_symbol(root, sym, kl, me=None, fu=None):
    frames.write_parquet_atomic(frames.normalize_klines(kl), kline_path(sym, root))
    if me is not None:
        frames.write_parquet_atomic(frames.normalize_metrics(me), metrics_path(sym, root))
    if fu is not None:
        frames.write_parquet_atomic(frames.normalize_funding(fu), funding_path(sym, root))


def _metrics(n, t0=T0, oi=1000.0):
    d = {"create_time": t0 + np.arange(n) * BAR, "sum_open_interest": oi + np.arange(n),
         "sum_open_interest_value": 2 * (oi + np.arange(n))}
    for c in METRIC_COLS[3:]:
        d[c] = np.full(n, 1.5)
    return pd.DataFrame(d)


def test_load_and_symbols_available(tmp_path):
    kl = mk_klines(np.linspace(100, 101, 50))
    _write_symbol(tmp_path, "AAAUSDT", kl, _metrics(50), pd.DataFrame({"calc_time": [T0], "funding_rate": [0.0001]}))
    _write_symbol(tmp_path, "BBBUSDT", kl)
    assert loader.symbols_available(tmp_path) == ["AAAUSDT", "BBBUSDT"]
    assert loader.symbols_available(tmp_path, "metrics") == ["AAAUSDT"]
    k, m, f = loader.load("AAAUSDT", tmp_path)
    assert len(k) == 50 and len(m) == 50 and len(f) == 1
    k, m, f = loader.load("BBBUSDT", tmp_path)       # missing metrics/funding -> empty typed frames
    assert len(m) == 0 and list(m.columns) == METRIC_COLS and len(f) == 0
    with pytest.raises(FileNotFoundError):
        loader.load("ZZZUSDT", tmp_path)
    assert loader.symbols_available(tmp_path / "nope") == []


def test_align_metrics_no_lookahead_and_dead_oi():
    kl = mk_klines(np.linspace(100, 101, 20))
    me = _metrics(20)
    me = me.iloc[[0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].reset_index(drop=True)   # hole rows 5..9
    me.loc[2, ["sum_open_interest", "sum_open_interest_value"]] = 0.0                                # dead sample
    al = loader.align_metrics(kl, me, tolerance_bars=2)
    assert len(al) == len(kl)
    assert al["sum_open_interest"].iloc[0] == 1000.0
    assert np.isnan(al["sum_open_interest"].iloc[2]) and al["has_metrics"].iloc[2] == 0.0    # dead -> NaN
    # bar 4 uses the sample stamped at its own open_time, never a later one
    assert al["sum_open_interest"].iloc[4] == 1004.0
    # hole: forward fill at most 2 bars, then NaN
    assert al["sum_open_interest"].iloc[6] == 1004.0 and np.isnan(al["sum_open_interest"].iloc[8])
    assert al["has_metrics"].iloc[10] == 1.0
    empty = loader.align_metrics(kl, frames.empty_metrics())
    assert empty["has_metrics"].sum() == 0 and len(empty) == 20


def test_align_funding_asof():
    kl = mk_klines(np.full(10, 100.0))
    fu = pd.DataFrame({"calc_time": [T0 + 3 * BAR, T0 + 7 * BAR], "funding_rate": [0.001, -0.002]})
    s = loader.align_funding(kl, fu)
    assert np.isnan(s.iloc[2]) and s.iloc[3] == 0.001 and s.iloc[6] == 0.001 and s.iloc[7] == -0.002


def test_bar_fault_rules():
    n = 60
    o = np.full(n, 100.0)
    v = np.ones(n)
    kl = mk_klines(o, v=v)
    assert not quality.fault_mask(kl).any()
    # +30 % bar
    c = o.copy(); c[30] = 130.0
    k2 = mk_klines(o, np.maximum(o, c), np.minimum(o, c), c)
    m = quality.bar_fault_masks(k2)
    assert m["big_move"][30] and m["big_move"][31]           # up-move and the move back down
    # 10 % move on a calm market is NOT a fault (floor 15 %)
    c = o.copy(); c[30] = 110.0
    assert not quality.bar_fault_masks(mk_klines(o, np.maximum(o, c), np.minimum(o, c), c))["big_move"].any()
    # high < low, non-positive, zero volume
    h = o.copy(); l = o.copy(); h[10] = 99.0; l[10] = 101.0
    assert quality.bar_fault_masks(mk_klines(o, h, l))["bad_ohlc"][10]
    o3 = o.copy(); o3[11] = 0.0
    assert quality.bar_fault_masks(mk_klines(o3))["bad_ohlc"][11]
    v3 = v.copy(); v3[12] = 0
    assert quality.bar_fault_masks(mk_klines(o, v=v3))["zero_volume"][12]


def test_big_move_scales_with_atr_on_volatile_symbols():
    n = 200
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))       # ~3 % per bar: ATR is large
    o = np.concatenate([[100.0], c[:-1]])
    kl = mk_klines(o, np.maximum(o, c) * 1.02, np.minimum(o, c) * 0.98, c)
    pc = np.concatenate([[np.nan], c[:-1]])
    big = quality.bar_fault_masks(kl)["big_move"]
    ret = np.abs(c / pc - 1)
    assert (ret[big] > 0.15).all()                            # never flags below the 15 % floor
    assert big.sum() <= (np.nan_to_num(ret) > 0.15).sum()     # 8 x ATR can only make it stricter


def test_debounce_matures_after_n_and_clears_after_m():
    f = np.zeros(30, bool); f[5:7] = True                      # 2 bad bars: never matures
    assert not quality.debounce(f, 3, 5).any()
    f = np.zeros(30, bool); f[5:8] = True                      # 3 bad bars: matures at bar 7
    d = quality.debounce(f, 3, 5)
    assert d[7] and not d[6] and d[11] and not d[12]            # clears on the 5th good bar (8..12 -> idx 12)
    # a good bar in the middle resets the bad-run counter
    f = np.zeros(30, bool); f[[5, 6, 8, 9]] = True
    assert not quality.debounce(f, 3, 5).any()


def test_quality_report_flags_gaps_and_bad_bars(tmp_path):
    n = 100
    o = np.full(n, 100.0)
    kl = mk_klines(o)
    kl = kl.drop(index=range(40, 45)).reset_index(drop=True)       # 5-bar gap
    kl.loc[10, "volume"] = 0.0
    me = _metrics(100)
    me.loc[3, "sum_open_interest"] = 0.0
    _write_symbol(tmp_path, "QQQUSDT", kl, me, pd.DataFrame({"calc_time": [T0, T0 + 8 * 3_600_000],
                                                             "funding_rate": [0.0001, 0.0002]}))
    rep = quality.quality_report("QQQUSDT", tmp_path, now_ms=T0 + 200 * BAR)
    assert rep["rows"] == 95 and rep["gaps"] == 1 and rep["missing_bars"] == 5 and rep["largest_gap_bars"] == 5
    assert rep["zero_volume"] == 1 and rep["big_move"] == 0 and rep["bad_ohlc"] == 0
    assert rep["metrics_dead_oi_rows"] == 1 and rep["funding_rows"] == 2
    assert rep["ok"] is False and rep["gap_examples"][0]["missing"] == 5
    clean = mk_klines(o)
    _write_symbol(tmp_path, "OKUSDT", clean)
    assert quality.quality_report("OKUSDT", tmp_path)["gaps"] == 0
