"""Bar plausibility checks + archive quality report.

The per-bar rules mirror the live engine's data plausibility layer (docs/ARCHITECTURE.md
"Safety & robustness layer" #2): a bar is implausible when
  * single-bar |close/prev_close - 1| > max(15%, 8 x ATR14 / prev_close)
  * high < low, any OHLC price non-positive / NaN
  * zero volume
(the 'stale timestamp' and 'mark vs last' rules need live context and do not apply to history).
A fault must repeat `fault_mature_n` consecutive bars to mature and clears after `fault_recover_m`
consecutive good bars (hysteresis): `debounce()`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import loader
from .paths import BAR_MS, archive_root, depth_meta_path

MAX_RET_FLOOR = 0.15
ATR_MULT = 8.0


def _atr(h, l, c, n=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def bar_fault_masks(kl: pd.DataFrame, atr: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Boolean mask per fault kind, aligned to kl rows."""
    o = kl["open"].to_numpy("float64")
    h = kl["high"].to_numpy("float64")
    l = kl["low"].to_numpy("float64")
    c = kl["close"].to_numpy("float64")
    v = kl["volume"].to_numpy("float64")
    n = len(kl)
    if atr is None:
        atr = _atr(h, l, c) if n else np.array([])
    pc = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.abs(c / pc - 1.0)
        # ATR known at the previous bar (the live check runs before the new bar is accepted)
        prev_atr = np.concatenate([[np.nan], atr[:-1]])
        lim = np.maximum(MAX_RET_FLOOR, ATR_MULT * prev_atr / pc)
    big = np.nan_to_num(ret > lim, nan=False)
    # ATR not yet defined -> fall back to the 15% floor
    big = np.where(np.isnan(lim), np.nan_to_num(ret > MAX_RET_FLOOR, nan=False), big)
    bad_ohlc = (h < l) | ~((o > 0) & (h > 0) & (l > 0) & (c > 0))
    zero_vol = ~(v > 0)
    return {"big_move": big, "bad_ohlc": bad_ohlc, "zero_volume": zero_vol}


def fault_mask(kl: pd.DataFrame, atr: np.ndarray | None = None) -> np.ndarray:
    m = bar_fault_masks(kl, atr)
    return m["big_move"] | m["bad_ohlc"] | m["zero_volume"]


def debounce(fault: np.ndarray, mature_n: int = 3, recover_m: int = 5) -> np.ndarray:
    """Matured-fault state per bar (True = symbol excluded from entries after this bar).
    Matures on the bar completing `mature_n` consecutive faulty bars; clears on the bar completing
    `recover_m` consecutive good bars."""
    n = len(fault)
    out = np.zeros(n, dtype=bool)
    if not fault.any():
        return out
    bad_run = good_run = 0
    state = False
    idx = np.nonzero(fault)[0]
    # walk only the neighbourhood of faults; between faults state decays via recover_m good bars
    last = 0
    state_start = 0
    i = 0
    for k in range(n):
        if fault[k]:
            bad_run += 1
            good_run = 0
            if not state and bad_run >= mature_n:
                state = True
        else:
            good_run += 1
            bad_run = 0
            if state and good_run >= recover_m:
                state = False
        out[k] = state
        if not state and k > idx[-1]:
            break
    return out


def quality_report(symbol: str, root: Path | None = None, now_ms: int | None = None) -> dict:
    """Flag implausible bars and gaps in a symbol's archive.

    Returns counts plus up to 20 example rows (timestamp, kind) per kind; `ok` is True when there
    are no implausible bars, no duplicate/unsorted timestamps and no gap > 1 bar.
    """
    root = root or archive_root()
    kl, me, fu = loader.load(symbol, root)
    t = kl["open_time"].to_numpy()
    rep: dict = {"symbol": symbol, "rows": int(len(kl))}
    d = np.diff(t)
    rep["unsorted_or_duplicate"] = int((d <= 0).sum())
    off_grid = int((t % BAR_MS != 0).sum())
    rep["off_grid_timestamps"] = off_grid
    gaps = d > BAR_MS
    rep["gaps"] = int(gaps.sum())
    rep["missing_bars"] = int((d[gaps] // BAR_MS - 1).sum()) if gaps.any() else 0
    rep["largest_gap_bars"] = int(d.max() // BAR_MS - 1) if len(d) else 0
    rep["gap_examples"] = [{"after": int(t[i]), "missing": int(d[i] // BAR_MS - 1)}
                           for i in np.nonzero(gaps)[0][:20]]
    masks = bar_fault_masks(kl)
    for k, m in masks.items():
        rep[k] = int(m.sum())
        rep[k + "_examples"] = [int(x) for x in t[np.nonzero(m)[0][:20]]]
    deb = debounce(fault_mask(kl))
    rep["matured_fault_bars"] = int(deb.sum())
    if len(t):
        import time
        now = int(time.time() * 1000) if now_ms is None else now_ms
        rep["last_bar_age_bars"] = int((now - t[-1]) // BAR_MS) - 1
    # metrics / funding sanity
    if len(me):
        rep["metrics_rows"] = int(len(me))
        rep["metrics_dead_oi_rows"] = int(((me["sum_open_interest"] <= 0) |
                                           (me["sum_open_interest_value"] <= 0)).sum())
        dm = np.diff(me["create_time"].to_numpy())
        rep["metrics_missing_rows"] = int((dm[dm > BAR_MS] // BAR_MS - 1).sum())
        rep["metrics_coverage"] = float(len(me) / max(1, len(kl)))
    else:
        rep["metrics_rows"] = 0
    if len(fu):
        df_ = np.diff(fu["calc_time"].to_numpy())
        rep["funding_rows"] = int(len(fu))
        rep["funding_gaps_over_9h"] = int((df_ > 9 * 3_600_000).sum())
        rep["funding_abs_max"] = float(fu["funding_rate"].abs().max())
    else:
        rep["funding_rows"] = 0
    # depth (order book) sanity
    dp = loader.load_depth(symbol, root)
    rep["depth_rows"] = int(len(dp))
    if len(dp):
        dt_ = dp["open_time"].to_numpy()
        lo, hi = int(dt_[0]), int(dt_[-1])
        in_range = int(((t >= lo) & (t <= hi)).sum())
        rep["depth_first"], rep["depth_last"] = lo, hi
        rep["depth_coverage"] = float(len(dp) / max(1, in_range))      # share of klines bars (in range) with a snapshot
        dd = np.diff(dt_)
        rep["depth_gaps_over_1h"] = int((dd > 12 * BAR_MS).sum())
        rep["depth_unsorted_or_duplicate"] = int((dd <= 0).sum())
        rep["depth_low_snap_bars"] = int((dp["snap_count"] < 3).sum())        # normal bar holds ~10 snapshots
        bn, an = dp["bid_notional_1"].to_numpy("float64"), dp["ask_notional_1"].to_numpy("float64")
        rep["depth_nonpositive_rows"] = int((np.nan_to_num(bn, nan=1.0) <= 0).sum() + (np.nan_to_num(an, nan=1.0) <= 0).sum())
        rep["depth_missing_1pct_rows"] = int((np.isnan(bn) | np.isnan(an)).sum())
        # nested bands must be monotone: notional within 5 % >= within 1 % (same side)
        b5, a5 = dp["bid_notional_5"].to_numpy("float64"), dp["ask_notional_5"].to_numpy("float64")
        with np.errstate(invalid="ignore"):
            rep["depth_nonmonotone_rows"] = int(((b5 < bn * 0.999) | (a5 < an * 0.999)).sum())
        meta = depth_meta_path(symbol, root)
        if meta.exists():
            import json
            m = json.loads(meta.read_text())
            rep["depth_missing_days"] = len(m.get("missing_days", []))
            rep["depth_has_02_band"] = m.get("has_02_band")
    rep["ok"] = bool(rep["unsorted_or_duplicate"] == 0 and rep["bad_ohlc"] == 0
                     and rep["big_move"] == 0 and rep["gaps"] == 0)
    return rep
