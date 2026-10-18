"""Feature engineering for the 5m entry model (numpy / pandas only).

Contract (docs/ARCHITECTURE.md "Features contract"):

    FEATURE_NAMES : stable ordered list of F feature names
    WINDOW        : bars of history the model sees (64)
    build_features(kl, metrics, funding, btc_kl) -> [N, F] float32

Design rules
------------
* **No look-ahead.** Row ``i`` is computed from data whose *close time* is <= the close of bar
  ``i`` (``open_time[i] + 5 min``).  Every rolling / ewm op here is causal.  Derivative metrics
  (OI, long/short ratios) are joined as-of ``create_time <= open_time[i]`` - i.e. we use the last
  snapshot that existed when the bar *opened* (one bar of lag).  This is deliberately pessimistic:
  the exchange publishes a snapshot stamped ``t`` shortly after ``t``, so being one bar late
  guarantees the live engine can reproduce exactly the same value.  Funding is joined as-of
  ``calc_time <= close_time[i]`` (the settled rate is public at settlement).
* **Scale-free and fp16-safe.**  Returns / ATR% / OI changes are expressed in *percent*,
  funding in *basis points*, everything else is a ratio / z-score / ATR-unit distance, so
  magnitudes sit in roughly [1e-2, 1e2] and survive float16 storage.  Output is clipped to
  +-CLIP_RAW.
* Missing derivative data -> 0 plus ``has_metrics = 0``.  Warm-up rows (the first WARMUP bars)
  contain zero-filled partial windows: **drop them** (use ``valid_from``), and in live use make
  sure at least ``WARMUP + WINDOW`` bars of history are fetched.
* Inputs must be on a regular 5m grid (use ``regularize`` if the archive has holes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MS = 300_000
WINDOW = 64
CLIP_RAW = 100.0
FUND_Z_WIN = 576            # 2 days of bars
LONG_WIN = 288              # 24 h
WARMUP = FUND_Z_WIN         # rows [0, WARMUP) are not fully warmed up
METRIC_MAX_STALE_MS = 30 * 60_000

_RET_H = (1, 3, 6, 12, 36, 72, 288)
_EMA_N = (9, 21, 55)

FEATURE_NAMES: list[str] = (
    [f"ret_{h}" for h in _RET_H]
    + ["atr_pct", "atr_ratio", "range_pos_48", "range_pos_288"]
    + [f"dist_ema_{n}" for n in _EMA_N] + ["dist_vwap_288"]
    + ["body_ratio", "upper_wick", "lower_wick", "body_atr", "range_atr"]
    + ["rel_vol_z", "vol_ratio_12_288", "taker_buy_ratio", "taker_buy_chg", "trades_z"]
    + ["oi_chg_3", "oi_chg_12", "oi_chg_48", "oi_val_chg_12", "oi_price_conf_12"]
    + ["funding_bps", "funding_z"]
    + ["top_ls_count", "top_ls_sum", "global_ls", "taker_ls_vol", "top_ls_chg_12", "global_ls_chg_12"]
    + ["has_metrics"]
    + ["btc_ret_3", "btc_ret_12", "btc_ret_48", "btc_atr_pct", "corr_btc_96", "rel_str_12"]
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
)
F = len(FEATURE_NAMES)
_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

# Feature groups (used for occlusion-based reasons / ablations).
FEATURE_GROUPS: dict[str, list[str]] = {
    "price": FEATURE_NAMES[0:20],
    "volume": FEATURE_NAMES[20:25],
    "derivatives": FEATURE_NAMES[25:39],
    "market": FEATURE_NAMES[39:45],
    "time": FEATURE_NAMES[45:49],
}

# Human labels: name -> (template, scale-free hint).  Template receives the RAW feature value.
FEATURE_LABELS: dict[str, str] = {
    "ret_1": "5m return {:+.2f}%", "ret_3": "15m return {:+.2f}%", "ret_6": "30m return {:+.2f}%",
    "ret_12": "1h return {:+.2f}%", "ret_36": "3h return {:+.2f}%", "ret_72": "6h return {:+.2f}%",
    "ret_288": "24h return {:+.2f}%",
    "atr_pct": "ATR {:.2f}% of price", "atr_ratio": "volatility {:.2f}x its 24h average",
    "range_pos_48": "at {:.0%} of 4h range", "range_pos_288": "at {:.0%} of 24h range",
    "dist_ema_9": "{:+.2f} ATR from EMA9", "dist_ema_21": "{:+.2f} ATR from EMA21",
    "dist_ema_55": "{:+.2f} ATR from EMA55", "dist_vwap_288": "{:+.2f} ATR from 24h VWAP",
    "body_ratio": "candle body {:+.0%} of range", "upper_wick": "upper wick {:.0%} of range",
    "lower_wick": "lower wick {:.0%} of range", "body_atr": "candle body {:+.2f} ATR",
    "range_atr": "candle range {:.2f} ATR",
    "rel_vol_z": "volume z-score {:+.1f}", "vol_ratio_12_288": "1h volume {:+.2f} (log) vs 24h",
    "taker_buy_ratio": "taker buy share {:.0%}", "taker_buy_chg": "taker buy share {:+.1%} vs 3h",
    "trades_z": "trade-count z-score {:+.1f}",
    "oi_chg_3": "OI {:+.1f}% in 15m", "oi_chg_12": "OI {:+.1f}% in 1h", "oi_chg_48": "OI {:+.1f}% in 4h",
    "oi_val_chg_12": "OI value {:+.2f}% in 1h",
    "oi_price_conf_12": "OI-price confirmation {:+.2f} (1h)",
    "funding_bps": "funding {:+.2f} bps", "funding_z": "funding z-score {:+.1f}",
    "top_ls_count": "top-trader L/S (accounts) ln {:+.2f}", "top_ls_sum": "top-trader L/S (positions) ln {:+.2f}",
    "global_ls": "global L/S ln {:+.2f}", "taker_ls_vol": "taker L/S volume ln {:+.2f}",
    "top_ls_chg_12": "top-trader L/S {:+.3f} in 1h", "global_ls_chg_12": "global L/S {:+.3f} in 1h",
    "has_metrics": "derivatives data {:.0f}",
    "btc_ret_3": "BTC 15m {:+.2f}%", "btc_ret_12": "BTC 1h {:+.2f}%", "btc_ret_48": "BTC 4h {:+.2f}%",
    "btc_atr_pct": "BTC ATR {:.2f}%", "corr_btc_96": "BTC correlation {:+.2f}",
    "rel_str_12": "{:+.2f}% vs BTC over 1h",
    "hour_sin": "time of day (sin) {:+.2f}", "hour_cos": "time of day (cos) {:+.2f}",
    "dow_sin": "weekday (sin) {:+.2f}", "dow_cos": "weekday (cos) {:+.2f}",
}
assert set(FEATURE_LABELS) == set(FEATURE_NAMES)


# --------------------------------------------------------------------------------------------
# optional extra feature groups (hook; nothing registered by default)
# --------------------------------------------------------------------------------------------
# A group is (names, fn, labels).  ``fn(kl, t_open_ms, extra_data) -> [N, len(names)]`` must obey the same
# no-look-ahead rule as the base features (use data with close time <= bar close; the base features
# join slow exchange snapshots as-of the bar OPEN, i.e. one bar late - do the same).  A model trained with
# groups records them in its json as ``feature_groups`` and ``feature_names``; ``model_np`` then validates
# against ``feature_names(groups)``, so the group must be registered before ``load``.  Example (order-book
# depth from data.binance.vision bookDepth, NOT implemented yet):
#     register_feature_group("depth", ["depth_imb_1pct", ...], build_depth, {"depth_imb_1pct": "bid/ask depth {:+.2f}"})
_EXTRA: dict[str, tuple] = {}


def register_feature_group(name: str, names: list[str], fn, labels: dict[str, str] | None = None) -> None:
    if name in _EXTRA:
        raise ValueError(f"feature group {name!r} already registered")
    dup = set(names) & set(FEATURE_NAMES)
    if dup:
        raise ValueError(f"feature names collide with the base set: {sorted(dup)}")
    _EXTRA[name] = (list(names), fn, dict(labels or {}))
    FEATURE_LABELS.update(labels or {})


def feature_names(groups=()) -> list[str]:
    """Base FEATURE_NAMES followed by the names of the given extra groups."""
    out = list(FEATURE_NAMES)
    for g in groups:
        if g not in _EXTRA:
            raise KeyError(f"feature group {g!r} is not registered")
        out += _EXTRA[g][0]
    return out


def feature_label(name: str, value: float) -> str:
    try:
        return FEATURE_LABELS[name].format(value)
    except Exception:  # pragma: no cover - defensive
        return f"{name} {value:+.3g}"


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _ms(s) -> np.ndarray:
    """Time column -> int64 epoch ms."""
    s = pd.Series(s)
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, utc=True)
        return ((dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy("int64")
    return pd.to_numeric(s).to_numpy("int64")


def regularize(kl: pd.DataFrame) -> pd.DataFrame:
    """Reindex klines onto the full 5m grid: forward-fill OHLC from the previous close with zero
    volume for missing bars.  No-op (copy) when the grid is already complete."""
    t = _ms(kl["open_time"])
    if len(t) == 0:
        return kl.copy()
    grid = np.arange(t[0], t[-1] + 1, BAR_MS, dtype="int64")
    if len(grid) == len(t) and np.array_equal(grid, t):
        return kl.reset_index(drop=True).copy()
    df = kl.copy()
    df["open_time"] = t
    df = df.drop_duplicates("open_time").set_index("open_time").reindex(grid)
    close = df["close"].ffill()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].where(df[c].notna(), close)
    for c in df.columns:
        if c not in ("open", "high", "low", "close"):
            df[c] = df[c].fillna(0.0)
    df.index.name = "open_time"
    return df.reset_index()


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder ATR (RMA of true range, alpha = 1/n, seeded after n bars). NaN for the first n-1 bars.
    Value at bar i uses bars <= i only.  Shared with labels.py (the exit template's ATR14)."""
    high = np.asarray(high, "float64")
    low = np.asarray(low, "float64")
    close = np.asarray(close, "float64")
    prev = np.concatenate([[close[0]], close[:-1]]) if len(close) else close
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    s = pd.Series(tr)
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def _asof(src_t: np.ndarray, src_v: np.ndarray, query_t: np.ndarray, max_stale: int | None) -> np.ndarray:
    """Last src_v with src_t <= query_t, NaN where none / older than max_stale ms.  src_v [M,K]."""
    out = np.full((len(query_t), src_v.shape[1]), np.nan)
    if len(src_t) == 0:
        return out
    j = np.searchsorted(src_t, query_t, side="right") - 1
    ok = j >= 0
    if max_stale is not None:
        ok &= (query_t - src_t[np.clip(j, 0, None)]) <= max_stale
    out[ok] = src_v[j[ok]]
    return out


def _roll_z(x: pd.Series, win: int) -> pd.Series:
    r = x.rolling(win, min_periods=win)
    sd = r.std()
    return ((x - r.mean()) / sd.where(sd > 1e-12))


def _logret(c: pd.Series, h: int) -> pd.Series:
    return 100.0 * np.log(c / c.shift(h))


def _ln(x: pd.Series) -> pd.Series:
    return np.log(x.where(x > 0))


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def build_features(kl: pd.DataFrame, metrics: pd.DataFrame | None, funding: pd.DataFrame | None,
                   btc_kl: pd.DataFrame | None, groups: tuple[str, ...] | list[str] = (),
                   extra_data: dict | None = None) -> np.ndarray:
    """Return [N, F (+ extras)] float32 aligned row-for-row with ``kl``.  See module docstring.

    ``groups`` selects optional registered feature groups (see ``register_feature_group``); their
    columns are appended after the base F in the order given, and ``feature_names(groups)`` lists them.
    With the default ``groups=()`` the output is exactly the base contract."""
    n = len(kl)
    out = np.zeros((n, F), dtype="float32")
    if n == 0:
        return out if not groups else np.zeros((0, len(feature_names(groups))), dtype="float32")
    t_open = _ms(kl["open_time"])
    t_close = t_open + BAR_MS
    o = kl["open"].astype("float64").reset_index(drop=True)
    h = kl["high"].astype("float64").reset_index(drop=True)
    l = kl["low"].astype("float64").reset_index(drop=True)
    c = kl["close"].astype("float64").reset_index(drop=True)
    vol = kl["volume"].astype("float64").reset_index(drop=True)
    qv = kl["quote_volume"].astype("float64").reset_index(drop=True)
    trd = kl["trades"].astype("float64").reset_index(drop=True)
    tbv = kl["taker_buy_volume"].astype("float64").reset_index(drop=True)

    cols: dict[str, pd.Series | np.ndarray] = {}

    # ---- price ----
    for k in _RET_H:
        cols[f"ret_{k}"] = _logret(c, k)
    a14 = pd.Series(atr(h.to_numpy(), l.to_numpy(), c.to_numpy(), 14))
    cols["atr_pct"] = 100.0 * a14 / c
    cols["atr_ratio"] = a14 / a14.rolling(LONG_WIN, min_periods=LONG_WIN).mean()
    for w, nm in ((48, "range_pos_48"), (288, "range_pos_288")):
        hi = h.rolling(w, min_periods=w).max()
        lo = l.rolling(w, min_periods=w).min()
        cols[nm] = (c - lo) / (hi - lo).where(hi - lo > 0)
    for m in _EMA_N:
        cols[f"dist_ema_{m}"] = (c - c.ewm(span=m, adjust=False, min_periods=m).mean()) / a14
    vwap = qv.rolling(LONG_WIN, min_periods=LONG_WIN).sum() / vol.rolling(LONG_WIN, min_periods=LONG_WIN).sum().where(lambda s: s > 0)
    cols["dist_vwap_288"] = (c - vwap) / a14
    rng = (h - l).where(h - l > 0)
    cols["body_ratio"] = (c - o) / rng
    cols["upper_wick"] = (h - np.maximum(o, c)) / rng
    cols["lower_wick"] = (np.minimum(o, c) - l) / rng
    cols["body_atr"] = (c - o) / a14
    cols["range_atr"] = (h - l) / a14

    # ---- volume ----
    cols["rel_vol_z"] = _roll_z(np.log1p(vol), LONG_WIN)
    v12 = vol.rolling(12, min_periods=12).mean()
    v288 = vol.rolling(LONG_WIN, min_periods=LONG_WIN).mean()
    cols["vol_ratio_12_288"] = np.log(v12.where(v12 > 0) / v288.where(v288 > 0))
    tbr = tbv / vol.where(vol > 0)
    cols["taker_buy_ratio"] = tbr
    cols["taker_buy_chg"] = tbr.rolling(3, min_periods=2).mean() - tbr.rolling(36, min_periods=18).mean()
    cols["trades_z"] = _roll_z(np.log1p(trd), LONG_WIN)

    # ---- derivatives (metrics, as-of with one bar of lag) ----
    has = np.zeros(n, dtype=bool)
    mcols = ("sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
             "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio")
    if metrics is not None and len(metrics):
        m = metrics.sort_values("create_time")
        mv = m[list(mcols)].to_numpy("float64")
        mt = _ms(m["create_time"])
        al = _asof(mt, mv, t_open, METRIC_MAX_STALE_MS)
        has = np.isfinite(al[:, 0])
    else:
        al = np.full((n, len(mcols)), np.nan)
    oi, oiv, top_c, top_s, glob, tlv = (pd.Series(al[:, j]) for j in range(6))
    for k, nm in ((3, "oi_chg_3"), (12, "oi_chg_12"), (48, "oi_chg_48")):
        cols[nm] = 100.0 * np.log(oi.where(oi > 0) / oi.where(oi > 0).shift(k))
    cols["oi_val_chg_12"] = 100.0 * np.log(oiv.where(oiv > 0) / oiv.where(oiv > 0).shift(12))
    cols["oi_price_conf_12"] = cols["oi_chg_12"] * np.sign(cols["ret_12"])
    fr = np.full(n, np.nan)
    if funding is not None and len(funding):
        f = funding.sort_values("calc_time")
        fv = _asof(_ms(f["calc_time"]), f[["funding_rate"]].to_numpy("float64"), t_close, None)
        fr = fv[:, 0]
    frs = pd.Series(fr * 1e4)
    cols["funding_bps"] = frs
    cols["funding_z"] = _roll_z(frs, FUND_Z_WIN)
    cols["top_ls_count"] = _ln(top_c)
    cols["top_ls_sum"] = _ln(top_s)
    cols["global_ls"] = _ln(glob)
    cols["taker_ls_vol"] = _ln(tlv)
    cols["top_ls_chg_12"] = cols["top_ls_sum"] - cols["top_ls_sum"].shift(12)
    cols["global_ls_chg_12"] = cols["global_ls"] - cols["global_ls"].shift(12)
    cols["has_metrics"] = pd.Series(has.astype("float64"))

    # ---- market context (BTC), aligned by open_time ----
    bn = np.full((n, 3), np.nan)    # btc close, high, low aligned
    if btc_kl is not None and len(btc_kl):
        b = btc_kl.drop_duplicates("open_time").copy()
        b["open_time"] = _ms(b["open_time"])
        b = b.set_index("open_time").sort_index()
        # regularise BTC on its own grid, then reindex to ours (ffill <= 3 bars)
        bg = b[["high", "low", "close"]].reindex(
            np.arange(b.index[0], b.index[-1] + 1, BAR_MS, dtype="int64")).ffill()
        bg = bg.reindex(t_open)
        bn = bg[["close", "high", "low"]].to_numpy("float64")
    bc = pd.Series(bn[:, 0])
    bh = pd.Series(bn[:, 1])
    bl = pd.Series(bn[:, 2])
    for k in (3, 12, 48):
        cols[f"btc_ret_{k}"] = _logret(bc, k)
    cols["btc_atr_pct"] = 100.0 * pd.Series(
        atr(bh.fillna(bc).to_numpy(), bl.fillna(bc).to_numpy(), bc.ffill().bfill().to_numpy(), 14)) / bc
    r1 = np.log(c / c.shift(1))
    b1 = np.log(bc / bc.shift(1))
    cols["corr_btc_96"] = r1.rolling(96, min_periods=96).corr(b1)
    cols["rel_str_12"] = cols["ret_12"] - cols["btc_ret_12"]

    # ---- time (bar open, UTC) ----
    sec = (t_open // 1000) % 86400
    hod = sec / 86400.0
    dow = ((t_open // 86_400_000) + 3) % 7            # 1970-01-01 was a Thursday -> Mon=0
    cols["hour_sin"] = np.sin(2 * np.pi * hod)
    cols["hour_cos"] = np.cos(2 * np.pi * hod)
    cols["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    cols["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    for nm in FEATURE_NAMES:
        v = np.asarray(cols[nm], dtype="float64")
        v = np.where(np.isfinite(v), v, 0.0)
        out[:, _IDX[nm]] = np.clip(v, -CLIP_RAW, CLIP_RAW).astype("float32")
    if groups:
        parts = [out]
        for g in groups:
            names, fn = _EXTRA[g][:2]
            v = np.asarray(fn(kl, t_open, extra_data or {}), dtype="float64").reshape(n, len(names))
            parts.append(np.clip(np.where(np.isfinite(v), v, 0.0), -CLIP_RAW, CLIP_RAW).astype("float32"))
        out = np.concatenate(parts, axis=1)
    return out


def valid_from(n_rows: int | None = None) -> int:
    """First row index whose features are fully warmed up (see WARMUP)."""
    return WARMUP


# --------------------------------------------------------------------------------------------
# windows + normalisation
# --------------------------------------------------------------------------------------------
def windows(feats: np.ndarray, idx) -> np.ndarray:
    """[len(idx), WINDOW, F]: rows idx-WINDOW+1 .. idx (inclusive) for each idx.  idx >= WINDOW-1."""
    idx = np.asarray(idx, dtype="int64")
    if len(idx) and (idx.min() < WINDOW - 1 or idx.max() >= len(feats)):
        raise IndexError("window index out of range (need idx >= WINDOW-1 and < len(feats))")
    ix = idx[:, None] + np.arange(-WINDOW + 1, 1, dtype="int64")[None, :]
    return feats[ix]


def robust_norm_stats(x: np.ndarray, max_rows: int = 2_000_000) -> dict[str, np.ndarray]:
    """Per-feature robust centre/scale from 2-D [rows, F] data: median and 1.4826*MAD, falling back
    to the std (then 1.0) for degenerate columns (e.g. the has_metrics flag).  Returns float32."""
    x = np.asarray(x)
    if len(x) > max_rows:
        x = x[np.linspace(0, len(x) - 1, max_rows).astype("int64")]
    x = x.astype("float32")
    med = np.median(x, axis=0)
    mad = 1.4826 * np.median(np.abs(x - med), axis=0)
    sd = x.std(axis=0)
    scale = np.where(mad > 1e-6, mad, np.where(sd > 1e-6, sd, 1.0))
    return {"mean": med.astype("float32"), "std": scale.astype("float32")}


NORM_CLIP = 5.0


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float = NORM_CLIP) -> np.ndarray:
    """(x - mean) / std, clipped to +-clip, float32.  Broadcasts over the last axis."""
    z = (np.asarray(x, "float32") - mean.astype("float32")) / std.astype("float32")
    return np.clip(z, -clip, clip).astype("float32", copy=False)
