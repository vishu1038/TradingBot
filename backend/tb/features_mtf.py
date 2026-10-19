"""Multi-timeframe ("user style") features built by RESAMPLING the 5m archive (numpy / pandas only).

The user decided trades from hour/day context and used the 5m chart only to time the entry.  This module
turns the 5m frames into one flat feature vector per decision bar:

* ``5m``   a compact set of timing features (the base ``tb.features`` columns that describe the last hour);
* ``1h`` / ``4h`` / ``1d``  per-timeframe trend / returns / ATR% / range position / EMA distance / volume
  z-score / last-bar shape, built from CLOSED bars only;
* BTC context per timeframe (returns, ATR%, correlation) and the coin's strength relative to BTC;
* derivatives over hours: open interest, long/short ratios, funding level / z-score / 24 h cumulative;
* time of day / week;
* optional group ``kalman`` (``tb.features_kalman``): state-estimation features per timeframe.

Design rules
------------
* **No look-ahead.**  For the 5m decision bar ``i`` (close time ``c_i = open_i + 5 min``) a timeframe bar is
  used only if its END time is ``<= c_i`` (``htf_index``).  Slow exchange snapshots are joined as-of the bar
  OPEN (one bar late), funding as-of the bar close - exactly like ``tb.features``.
* **Finite memory = train/live parity.**  Every higher-timeframe feature is a function of a bounded number of
  closed bars (SMA ATR instead of Wilder, a truncated exponential mean instead of a recursive EMA, windowed
  Kalman).  The value at a bar therefore does NOT depend on how much older history exists, so a live engine
  that fetches ``history_bars_5m`` bars reproduces the training value (``warmup_bars_5m`` documents the need).
* One builder, used by BOTH training and the live engine: ``build_for_model(meta, bundle)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import features as ft
from .features import BAR_MS, CLIP_RAW, _asof, _ln, _ms, _roll_z

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
DEFAULT_TFS = ("1h", "4h", "1d")
DAY_MS = 86_400_000

# per timeframe: name prefix, return lookbacks (bars), range-position windows, EMA spans, volume z window,
# long ATR window (volatility ratio), BTC return lookbacks, Kalman window (bars)
TF_SPEC: dict[str, dict] = {
    "1h": dict(p="h1", rets=(1, 4, 12, 24, 72), rng=(24, 72), ema=(9, 21, 55), volz=168, atr_long=168,
               btc_rets=(4, 24), kal_L=96, corr=72),
    "4h": dict(p="h4", rets=(1, 3, 6, 12, 30), rng=(30,), ema=(9, 21), volz=42, atr_long=42,
               btc_rets=(3, 12), kal_L=60, corr=30),
    "1d": dict(p="d1", rets=(1, 3, 7, 14), rng=(14,), ema=(5, 10), volz=20, atr_long=21,
               btc_rets=(1, 7), kal_L=30, corr=14),
}
ATR_N = 14

# base 5m columns used as "entry timing" context
TIMING_5M = ["ret_1", "ret_3", "ret_6", "ret_12", "atr_pct", "atr_ratio", "range_pos_48", "dist_ema_9",
             "dist_ema_21", "dist_vwap_288", "body_ratio", "upper_wick", "lower_wick", "rel_vol_z",
             "taker_buy_ratio", "taker_buy_chg"]
DERIV_NAMES = ["oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "oi_val_chg_4h", "oi_price_conf_4h", "top_ls_sum",
               "global_ls", "top_ls_chg_4h", "global_ls_chg_4h", "top_ls_chg_24h", "taker_ls_vol_1h",
               "funding_bps", "funding_z_7d", "funding_sum_24h", "has_metrics"]
TIME_NAMES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
FUND_Z_WIN = 2016           # 7 days of 5m bars
METRIC_MAX_STALE_MS = 30 * 60_000


@dataclass
class FrameBundle:
    """The 5m frames of one symbol (archive schema).  ``btc_kl`` = BTCUSDT 5m klines.  Anything with the same
    attribute names (or a dict with those keys) is accepted by ``build_for_model``."""
    symbol: str
    kl: pd.DataFrame
    metrics: pd.DataFrame | None = None
    funding: pd.DataFrame | None = None
    depth: pd.DataFrame | None = None
    btc_kl: pd.DataFrame | None = None


def _get(bundle: Any, *names: str):
    for n in names:
        if isinstance(bundle, dict):
            if bundle.get(n) is not None:
                return bundle[n]
        elif getattr(bundle, n, None) is not None:
            return getattr(bundle, n)
    return None


# --------------------------------------------------------------------------------------------
# names / warm-up
# --------------------------------------------------------------------------------------------
def _tf_feature_names(tf: str) -> list[str]:
    s = TF_SPEC[tf]
    p = s["p"]
    out = [f"{p}_ret_{n}" for n in s["rets"]] + [f"{p}_atr_pct", f"{p}_atr_ratio"]
    out += [f"{p}_range_pos_{w}" for w in s["rng"]] + [f"{p}_dist_ema_{n}" for n in s["ema"]]
    out += [f"{p}_vol_z", f"{p}_taker_buy", f"{p}_taker_chg", f"{p}_body_ratio", f"{p}_upper_wick",
            f"{p}_lower_wick", f"{p}_range_atr"]
    return out


def _btc_feature_names(tf: str) -> list[str]:
    s = TF_SPEC[tf]
    p = s["p"]
    out = [f"btc_{p}_ret_{n}" for n in s["btc_rets"]] + [f"btc_{p}_atr_pct", f"btc_{p}_corr"]
    out += [f"{p}_rel_str_{n}" for n in s["btc_rets"]]
    return out


def base_names(timeframes=DEFAULT_TFS) -> list[str]:
    out = [f"m5_{n}" for n in TIMING_5M]
    for tf in timeframes:
        out += _tf_feature_names(tf)
    for tf in timeframes:
        out += _btc_feature_names(tf)
    return out + DERIV_NAMES + TIME_NAMES


def feature_names(timeframes=DEFAULT_TFS, groups=()) -> list[str]:
    out = base_names(timeframes)
    for g in groups:
        if g == "kalman":
            from . import features_kalman
            out += features_kalman.names(timeframes)
        else:
            raise KeyError(f"unknown mtf feature group {g!r}")
    return out


def htf_timeframes(timeframes) -> tuple[str, ...]:
    """The higher timeframes of a model's ``timeframes`` list (``"5m"`` is implicit and always present)."""
    tfs = tuple(t for t in timeframes if t != "5m")
    for t in tfs:
        if t not in TF_MS:
            raise ValueError(f"unsupported timeframe {t!r} (have 5m + {sorted(TF_MS)})")
    return tfs or DEFAULT_TFS


def bars_needed(tf: str, kalman: bool = True) -> int:
    """Closed bars of timeframe ``tf`` that the features of ONE decision bar read."""
    s = TF_SPEC[tf]
    n = max(max(s["rets"]) + 1, max(s["rng"]), 2 * max(s["ema"]), s["volz"], s["atr_long"] + 1, ATR_N + 1,
            s["corr"] + 1, max(s["btc_rets"]) + 1)
    if kalman:
        n = max(n, s["kal_L"] + 1)
    return int(n)


def warmup_bars_5m(timeframes=DEFAULT_TFS, groups=()) -> int:
    """5m bars of history from which a decision bar's features are exact (and the live engine's minimum
    ``history_bars_5m``): the longest timeframe lookback + one partial bar of slack + the 5m/derivative warm-up."""
    kal = "kalman" in groups
    htf = max((bars_needed(tf, kal) + 1) * (TF_MS[tf] // BAR_MS) for tf in timeframes)
    return int(max(htf, ft.WARMUP + 2 * ft.WINDOW, FUND_Z_WIN + 300))


def history_bars_5m(timeframes=DEFAULT_TFS, groups=()) -> int:
    """Recommended history to fetch live (``warmup_bars_5m`` rounded up with a small safety margin)."""
    w = warmup_bars_5m(timeframes, groups) + 288
    return int(np.ceil(w / 100.0) * 100)


# --------------------------------------------------------------------------------------------
# resampling (closed bars only)
# --------------------------------------------------------------------------------------------
def resample_ohlcv(kl: pd.DataFrame, tf: str) -> dict[str, np.ndarray]:
    """5m klines (regular grid, see ``ft.regularize``) -> COMPLETE bars of ``tf``.

    Returns arrays ``end_ms`` (bar end = the moment it closes), open, high, low, close, volume, quote_volume,
    taker_buy_volume.  A bar is kept only if all ``tf/5m`` 5m bars are present, so the leading partial bar and
    the still-forming trailing bar never appear."""
    k = TF_MS[tf] // BAR_MS
    t = _ms(kl["open_time"])
    if len(t) == 0:
        z = np.zeros(0)
        return dict(end_ms=np.zeros(0, "int64"), open=z, high=z, low=z, close=z, volume=z, quote_volume=z,
                    taker_buy_volume=z)
    g = t // TF_MS[tf]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    counts = np.diff(np.r_[starts, len(t)])
    full = counts == k
    st, cn = starts[full], counts[full]

    def col(c):
        return kl[c].to_numpy("float64") if c in kl else np.zeros(len(t))

    o, h, l, c = col("open"), col("high"), col("low"), col("close")
    out = dict(end_ms=(g[st] + 1) * TF_MS[tf], open=o[st], high=np.maximum.reduceat(h, starts)[full],
               low=np.minimum.reduceat(l, starts)[full], close=c[st + cn - 1])
    for name in ("volume", "quote_volume", "taker_buy_volume"):
        out[name] = np.add.reduceat(col(name), starts)[full]
    return out


def htf_index(close_ms: np.ndarray, end_ms: np.ndarray) -> np.ndarray:
    """Index of the last timeframe bar that is CLOSED at ``close_ms`` (end <= close); -1 if none."""
    return np.searchsorted(end_ms, np.asarray(close_ms), side="right") - 1


def decision_mask(t_open: np.ndarray, every_bars: int) -> np.ndarray:
    """True for 5m bars whose CLOSE falls on a multiple of ``every_bars`` 5m bars (12 -> every closed 1h bar)."""
    c = np.asarray(t_open, "int64") + BAR_MS
    return (c % (int(every_bars) * BAR_MS)) == 0


# --------------------------------------------------------------------------------------------
# finite-window helpers
# --------------------------------------------------------------------------------------------
def wema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted mean over the last ``2*span`` values (weights (1-a)^k, a = 2/(span+1)), NaN
    before that.  Finite memory (unlike a recursive EMA), so it is history independent."""
    x = np.asarray(x, "float64")
    L = 2 * int(span)
    out = np.full(len(x), np.nan)
    if len(x) < L:
        return out
    a = 2.0 / (span + 1.0)
    w = (1.0 - a) ** np.arange(L - 1, -1, -1)            # oldest ... newest
    out[L - 1:] = sliding_window_view(x, L) @ (w / w.sum())
    return out


def _roll_sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def true_range(h, l, c) -> np.ndarray:
    pc = np.r_[np.nan, c[:-1]]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))      # NaN on bar 0


def sma_atr(h, l, c, n: int = ATR_N) -> np.ndarray:
    return _roll_sma(true_range(np.asarray(h, "float64"), np.asarray(l, "float64"), np.asarray(c, "float64")), n)


def htf_atr_5m(kl: pd.DataFrame, tf: str, n: int = ATR_N) -> np.ndarray:
    """ATR(n) (price units, SMA of true range) of the last CLOSED ``tf`` bar, as-of every 5m row's close.
    Used by the labels / live exits for ``exit_cfg.atr_tf``."""
    bars = resample_ohlcv(kl, tf)
    a = sma_atr(bars["high"], bars["low"], bars["close"], n)
    j = htf_index(_ms(kl["open_time"]) + BAR_MS, bars["end_ms"])
    out = np.full(len(kl), np.nan)
    ok = j >= 0
    out[ok] = a[j[ok]]
    return out


def _logret_pct(c: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    if len(c) > n:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n:] = 100.0 * np.log(c[n:] / c[:-n])
    return out


def _tf_block(bars: dict, tf: str) -> dict[str, np.ndarray]:
    """Per-bar features of one timeframe (arrays aligned with ``bars``)."""
    s = TF_SPEC[tf]
    p = s["p"]
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    vol, tbv = bars["volume"], bars["taker_buy_volume"]
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in s["rets"]:
            out[f"{p}_ret_{n}"] = _logret_pct(c, n)
        tr = true_range(h, l, c)
        a = _roll_sma(tr, ATR_N)
        out[f"{p}_atr_pct"] = 100.0 * a / c
        out[f"{p}_atr_ratio"] = a / _roll_sma(tr, s["atr_long"])
        for w in s["rng"]:
            hi = pd.Series(h).rolling(w, min_periods=w).max().to_numpy()
            lo = pd.Series(l).rolling(w, min_periods=w).min().to_numpy()
            out[f"{p}_range_pos_{w}"] = np.where(hi - lo > 0, (c - lo) / (hi - lo), np.nan)
        for n in s["ema"]:
            out[f"{p}_dist_ema_{n}"] = (c - wema(c, n)) / a
        lv = pd.Series(np.log1p(vol))
        out[f"{p}_vol_z"] = _roll_z(lv, s["volz"]).to_numpy()
        tbr = np.where(vol > 0, tbv / np.where(vol > 0, vol, 1.0), np.nan)
        out[f"{p}_taker_buy"] = tbr
        out[f"{p}_taker_chg"] = _roll_sma(tbr, 3) - _roll_sma(tbr, s["volz"])
        rng = np.where(h - l > 0, h - l, np.nan)
        out[f"{p}_body_ratio"] = (c - o) / rng
        out[f"{p}_upper_wick"] = (h - np.maximum(o, c)) / rng
        out[f"{p}_lower_wick"] = (np.minimum(o, c) - l) / rng
        out[f"{p}_range_atr"] = (h - l) / a
    return out


def _align_btc(btc_kl: pd.DataFrame | None, t_open: np.ndarray) -> pd.DataFrame | None:
    """BTC 5m frame reindexed (forward-filled) onto the coin's 5m grid."""
    if btc_kl is None or not len(btc_kl):
        return None
    b = btc_kl.drop_duplicates("open_time").copy()
    b["open_time"] = _ms(b["open_time"])
    b = b.set_index("open_time").sort_index()
    cols = [c for c in ("open", "high", "low", "close") if c in b]
    bg = b[cols].reindex(np.arange(b.index[0], b.index[-1] + 1, BAR_MS, dtype="int64")).ffill().reindex(t_open)
    for c in ("volume", "quote_volume", "taker_buy_volume"):
        bg[c] = 0.0
    bg = bg.reset_index(drop=True)
    bg.insert(0, "open_time", t_open)
    return bg


def _derivs(kl_t_open: np.ndarray, c5: pd.Series, ret48: pd.Series, metrics, funding) -> dict[str, np.ndarray]:
    n = len(kl_t_open)
    t_close = kl_t_open + BAR_MS
    cols: dict[str, np.ndarray | pd.Series] = {}
    mcols = ("sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
             "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio")
    has = np.zeros(n, dtype=bool)
    if metrics is not None and len(metrics):
        m = metrics.sort_values("create_time")
        al = _asof(_ms(m["create_time"]), m[list(mcols)].to_numpy("float64"), kl_t_open, METRIC_MAX_STALE_MS)
        has = np.isfinite(al[:, 0])
    else:
        al = np.full((n, len(mcols)), np.nan)
    oi, oiv, _topc, tops, glob, tlv = (pd.Series(al[:, j]) for j in range(6))
    oi_ = oi.where(oi > 0)
    oiv_ = oiv.where(oiv > 0)
    cols["oi_chg_1h"] = 100.0 * np.log(oi_ / oi_.shift(12))
    cols["oi_chg_4h"] = 100.0 * np.log(oi_ / oi_.shift(48))
    cols["oi_chg_24h"] = 100.0 * np.log(oi_ / oi_.shift(288))
    cols["oi_val_chg_4h"] = 100.0 * np.log(oiv_ / oiv_.shift(48))
    cols["oi_price_conf_4h"] = cols["oi_chg_4h"] * np.sign(ret48)
    ls_s, ls_g = _ln(tops), _ln(glob)
    cols["top_ls_sum"] = ls_s
    cols["global_ls"] = ls_g
    cols["top_ls_chg_4h"] = ls_s - ls_s.shift(48)
    cols["global_ls_chg_4h"] = ls_g - ls_g.shift(48)
    cols["top_ls_chg_24h"] = ls_s - ls_s.shift(288)
    cols["taker_ls_vol_1h"] = _ln(tlv).rolling(12, min_periods=6).mean()
    fr = np.full(n, np.nan)
    fsum = np.full(n, np.nan)
    if funding is not None and len(funding):
        f = funding.sort_values("calc_time")
        ft_ = _ms(f["calc_time"])
        fv = f["funding_rate"].to_numpy("float64")
        fr = _asof(ft_, fv[:, None], t_close, None)[:, 0]
        cum = np.r_[0.0, np.cumsum(fv)]
        hi = np.searchsorted(ft_, t_close, side="right")
        lo = np.searchsorted(ft_, t_close - DAY_MS, side="right")
        fsum = 1e4 * (cum[hi] - cum[lo])
        fsum[hi == 0] = np.nan
    frs = pd.Series(fr * 1e4)
    cols["funding_bps"] = frs
    cols["funding_z_7d"] = _roll_z(frs, FUND_Z_WIN)
    cols["funding_sum_24h"] = fsum
    cols["has_metrics"] = has.astype("float64")
    return {k: np.asarray(v, "float64") for k, v in cols.items()}


# --------------------------------------------------------------------------------------------
# main builder
# --------------------------------------------------------------------------------------------
def build_mtf_features(kl: pd.DataFrame, metrics: pd.DataFrame | None, funding: pd.DataFrame | None,
                       btc_kl: pd.DataFrame | None, timeframes=DEFAULT_TFS, groups=(),
                       rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(X [R, F] float32, t_open [N] int64)``.  ``kl`` is regularised onto the 5m grid first, so the rows of
    ``X`` correspond to ``t_open[rows]`` (all rows when ``rows`` is None).  Row ``i`` uses only data that is
    public at the close of 5m bar ``i``.  Non-finite values become 0 (the leading warm-up rows are therefore
    meaningless: drop rows before ``warmup_bars_5m``)."""
    timeframes = tuple(timeframes)
    names = feature_names(timeframes, groups)
    kl = ft.regularize(kl)
    n = len(kl)
    rows = np.arange(n) if rows is None else np.asarray(rows, "int64")
    if n == 0:
        return np.zeros((len(rows), len(names)), "float32"), np.zeros(0, "int64")
    t_open = _ms(kl["open_time"])
    t_close = t_open + BAR_MS
    rc = t_close[rows]
    cols: dict[str, np.ndarray] = {}

    # ---- 5m entry-timing features (the base set; recursive filters converge within the 576-bar warm-up) ----
    base = ft.build_features(kl, metrics, funding, btc_kl)
    for nm in TIMING_5M:
        cols[f"m5_{nm}"] = base[rows, ft.FEATURE_NAMES.index(nm)].astype("float64")

    # ---- per-timeframe blocks + BTC ----
    btc = _align_btc(btc_kl, t_open)
    for tf in timeframes:
        bars = resample_ohlcv(kl, tf)
        j = htf_index(rc, bars["end_ms"])
        ok = j >= 0
        blk = _tf_block(bars, tf)
        for nm, arr in blk.items():
            v = np.full(len(rows), np.nan)
            v[ok] = arr[j[ok]]
            cols[nm] = v
        s = TF_SPEC[tf]
        p = s["p"]
        if btc is not None:
            bb = resample_ohlcv(btc, tf)
            # the coin's and BTC's bars share the same boundaries/end times only when both are complete; align by end
            jb = htf_index(rc, bb["end_ms"])
            okb = jb >= 0
            bc, bh, bl = bb["close"], bb["high"], bb["low"]
            for n_ in s["btc_rets"]:
                r = _logret_pct(bc, n_)
                v = np.full(len(rows), np.nan)
                v[okb] = r[jb[okb]]
                cols[f"btc_{p}_ret_{n_}"] = v
                cols[f"{p}_rel_str_{n_}"] = cols[f"{p}_ret_{n_}"] - v
            ba = 100.0 * sma_atr(bh, bl, bc) / bc
            v = np.full(len(rows), np.nan)
            v[okb] = ba[jb[okb]]
            cols[f"btc_{p}_atr_pct"] = v
            # rolling correlation of 1-bar log returns (coin vs BTC) on the timeframe grid
            cb = pd.Series(np.r_[np.nan, np.diff(np.log(bars["close"]))], index=bars["end_ms"])
            bbs = pd.Series(np.r_[np.nan, np.diff(np.log(bc))], index=bb["end_ms"]).reindex(cb.index)
            cr = cb.rolling(s["corr"], min_periods=s["corr"]).corr(bbs).to_numpy()
            v = np.full(len(rows), np.nan)
            v[ok] = cr[j[ok]]
            cols[f"btc_{p}_corr"] = v
        else:
            for nm in _btc_feature_names(tf):
                cols[nm] = np.full(len(rows), np.nan)

    # ---- derivatives over hours ----
    c5 = pd.Series(kl["close"].to_numpy("float64"))
    ret48 = 100.0 * np.log(c5 / c5.shift(48))
    dv = _derivs(t_open, c5, ret48, metrics, funding)
    for nm in DERIV_NAMES:
        cols[nm] = dv[nm][rows]

    # ---- time (decision bar close, UTC) ----
    hod = ((rc // 1000) % 86400) / 86400.0
    dow = ((rc // DAY_MS) + 3) % 7
    cols["hour_sin"] = np.sin(2 * np.pi * hod)
    cols["hour_cos"] = np.cos(2 * np.pi * hod)
    cols["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    cols["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    if "kalman" in groups:
        from . import features_kalman
        kcols = features_kalman.compute(kl, timeframes, rows=rows)
        cols.update(kcols)

    X = np.zeros((len(rows), len(names)), dtype="float32")
    for k, nm in enumerate(names):
        v = np.asarray(cols[nm], "float64")
        X[:, k] = np.clip(np.where(np.isfinite(v), v, 0.0), -CLIP_RAW, CLIP_RAW).astype("float32")
    return X, t_open


def build_for_model(meta: dict, bundle: Any, rows: np.ndarray | None = None, return_index: bool = False):
    """THE feature builder for a ``kind: "mtf"|"gbm"`` model - training and the live engine both call this.

    ``meta`` is the model json (``timeframes`` incl. "5m", ``feature_groups``, optional ``feature_names`` which
    must match).  ``bundle`` is a ``FrameBundle`` (or dict / object with ``kl``, ``metrics``, ``funding`` and
    ``btc_kl``; ``klines`` / ``btc`` are accepted aliases) holding at least ``meta["history_bars_5m"]`` bars.

    Returns ``X [N, F] float32`` aligned row-for-row with the regularised 5m frame (the live engine uses the
    last row; ``X[-1]`` is the vector for the most recent closed 5m bar), or ``(X, t_open)`` with
    ``return_index=True``."""
    tfs = htf_timeframes(meta.get("timeframes") or ["5m", *DEFAULT_TFS])
    groups = tuple(meta.get("feature_groups") or ())
    kl = _get(bundle, "kl", "klines")
    if kl is None:
        raise ValueError("bundle has no 5m klines")
    X, t_open = build_mtf_features(kl, _get(bundle, "metrics"), _get(bundle, "funding"),
                                   _get(bundle, "btc_kl", "btc"), tfs, groups, rows)
    want = meta.get("feature_names")
    if want is not None and list(want) != feature_names(tfs, groups):
        raise ValueError("model feature_names differ from tb.features_mtf (retrain the model)")
    return (X, t_open) if return_index else X
