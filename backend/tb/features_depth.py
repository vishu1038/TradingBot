"""Optional ``depth`` feature group: order-book depth bands (data.binance.vision bookDepth, 5m-aggregated).

Importing this module registers the group with ``tb.features`` (hook ``register_feature_group``); a model
trained with it lists ``"feature_groups": ["depth"]`` in its json and ``model_np.load`` imports this module.
Use it as::

    x = build_features(kl, metrics, funding, btc, groups=["depth"], extra_data={"depth": depth_df})

Input frame (``normalize_depth`` accepts either shape)
    * wide : ``open_time`` (5m bucket START, ms) + cumulative notional columns per band, named like
             ``bid_0.2 ... bid_5`` and ``ask_0.2 ... ask_5`` (names containing a side word and the band
             percentage are matched by regex, ``notional`` columns are preferred over ``depth`` columns);
    * long : raw bookDepth rows ``timestamp, percentage (-5..-0.2 bids, 0.2..5 asks), depth, notional``
             (30 s snapshots) - averaged per 5m bucket.
    A band value is the cumulative USD notional resting within pct % of mid on that side.

Timing: the bucket stamped T covers [T, T+5m) and is complete at the close of the bar opened at T.  To
stay on the safe side the join uses bucket ``T - DEPTH_LAG_BARS * 5m`` for the bar opened at T (default
1 bar late, like the derivatives metrics) - set DEPTH_LAG_BARS = 0 once the aggregation key is confirmed
to be the bucket start.  Bars without a fresh bucket (stale > DEPTH_MAX_STALE_BARS) get zeros and
``has_depth = 0``.

Features (18): per-band log imbalance ln(bid/ask) at 0.2/1/2/3/4/5 %; thinning ln(total / its 24 h mean) at
0.2/1/5 %; change of the 1 % imbalance and of the 1 % total over 15 m / 1 h; wall concentration (largest
shell notional / 5 % notional) per side and its 1 h change (wall growth / vanishing); coverage flag.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .features import BAR_MS, register_feature_group

BANDS = (0.2, 1.0, 2.0, 3.0, 4.0, 5.0)
DEPTH_LAG_BARS = 1
DEPTH_MAX_STALE_BARS = 3
_LONG = 288

NAMES = (
    [f"depth_imb_{b:g}" for b in BANDS]
    + [f"depth_thin_{b:g}" for b in (0.2, 1.0, 5.0)]
    + ["depth_imb1_chg_3", "depth_imb1_chg_12", "depth_tot1_chg_3", "depth_tot1_chg_12"]
    + ["bid_wall_conc", "ask_wall_conc", "bid_wall_chg_12", "ask_wall_chg_12"]
    + ["has_depth"]
)
LABELS = {
    **{f"depth_imb_{b:g}": f"book imbalance ln {{:+.2f}} within {b:g}%" for b in BANDS},
    **{f"depth_thin_{b:g}": f"depth within {b:g}% {{:+.2f}} (log) vs 24h avg" for b in (0.2, 1.0, 5.0)},
    "depth_imb1_chg_3": "book imbalance {:+.2f} in 15m", "depth_imb1_chg_12": "book imbalance {:+.2f} in 1h",
    "depth_tot1_chg_3": "1% depth {:+.2f} (log) in 15m", "depth_tot1_chg_12": "1% depth {:+.2f} (log) in 1h",
    "bid_wall_conc": "bid wall {:.0%} of depth", "ask_wall_conc": "ask wall {:.0%} of depth",
    "bid_wall_chg_12": "bid wall {:+.0%} in 1h", "ask_wall_chg_12": "ask wall {:+.0%} in 1h",
    "has_depth": "order-book data {:.0f}",
}

_COLS = [f"{s}_{b:g}" for s in ("bid", "ask") for b in BANDS]


def normalize_depth(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """-> frame indexed by int64 5m bucket start (ms) with the 12 columns ``bid_*`` / ``ask_*`` (float64)."""
    if df is None or not len(df):
        return None
    cols = list(df.columns)
    if "percentage" in cols:                                    # raw long format
        t = pd.to_numeric(df["timestamp"] if "timestamp" in cols else df["open_time"])
        if not pd.api.types.is_numeric_dtype(df.get("timestamp", t)):
            t = (pd.to_datetime(df["timestamp"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
        pct = pd.to_numeric(df["percentage"])
        val = pd.to_numeric(df["notional" if "notional" in cols else "depth"])
        d = pd.DataFrame({"bucket": (t.astype("int64") // BAR_MS) * BAR_MS, "side": np.where(pct < 0, "bid", "ask"),
                          "band": pct.abs().round(3), "v": val})
        w = d.groupby(["bucket", "side", "band"])["v"].mean().unstack(["side", "band"])
        out = pd.DataFrame(index=w.index)
        for s in ("bid", "ask"):
            for b in BANDS:
                out[f"{s}_{b:g}"] = w[(s, round(b, 3))] if (s, round(b, 3)) in w.columns else np.nan
        out.index = out.index.astype("int64")
        return out.sort_index()
    key = "open_time" if "open_time" in cols else ("time" if "time" in cols else cols[0])
    out = pd.DataFrame(index=pd.to_numeric(df[key]).astype("int64"))
    found: dict[tuple, list] = {}
    for c in cols:
        name = str(c).lower()
        if c == key:
            continue
        side = "bid" if "bid" in name else ("ask" if "ask" in name else None)
        nums = re.findall(r"\d+(?:[._p]\d+)?", name)
        if side is None or not nums:
            continue
        band = float(nums[-1].replace("_", ".").replace("p", "."))
        found.setdefault((side, band), []).append(c)
    for s_ in ("bid", "ask"):
        for b_ in BANDS:
            cand = found.get((s_, b_), [])
            pref = [c for c in cand if "notional" in str(c).lower()] or cand
            out[f"{s_}_{b_:g}"] = pd.to_numeric(df[pref[0]]).to_numpy("float64") if pref else np.nan
    return out[~out.index.duplicated(keep="last")].sort_index()


def _ln(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.where((a > 0) & (b > 0), a, np.nan) / np.where((a > 0) & (b > 0), b, np.nan))


def build_depth(kl: pd.DataFrame, t_open: np.ndarray, extra: dict) -> np.ndarray:
    """-> [N, len(NAMES)] (NaN allowed; build_features zero-fills and clips)."""
    n = len(t_open)
    out = np.full((n, len(NAMES)), np.nan)
    dep = normalize_depth(extra.get("depth"))
    if dep is None:
        out[:, -1] = 0.0
        return out
    key = t_open - DEPTH_LAG_BARS * BAR_MS
    # exact bucket if present, else the latest bucket within DEPTH_MAX_STALE_BARS (forward fill, causal)
    idx = dep.index.to_numpy()
    j = np.searchsorted(idx, key, side="right") - 1
    ok = (j >= 0) & ((key - idx[np.clip(j, 0, None)]) <= DEPTH_MAX_STALE_BARS * BAR_MS)
    vals = np.full((n, len(_COLS)), np.nan)
    vals[ok] = dep[_COLS].to_numpy("float64")[j[ok]]
    bid = {b: vals[:, k] for k, b in enumerate(BANDS)}
    ask = {b: vals[:, len(BANDS) + k] for k, b in enumerate(BANDS)}
    has = np.isfinite(vals).all(axis=1) & ok
    col = {nm: k for k, nm in enumerate(NAMES)}
    for b in BANDS:
        out[:, col[f"depth_imb_{b:g}"]] = _ln(bid[b], ask[b])
    tot = {b: bid[b] + ask[b] for b in BANDS}
    for b in (0.2, 1.0, 5.0):
        s = pd.Series(tot[b])
        m = s.rolling(_LONG, min_periods=_LONG // 4).mean()
        out[:, col[f"depth_thin_{b:g}"]] = _ln(tot[b], m.to_numpy())
    imb1 = pd.Series(out[:, col["depth_imb_1"]])
    t1 = pd.Series(np.log(np.where(tot[1.0] > 0, tot[1.0], np.nan)))
    for k in (3, 12):
        out[:, col[f"depth_imb1_chg_{k}"]] = (imb1 - imb1.shift(k)).to_numpy()
        out[:, col[f"depth_tot1_chg_{k}"]] = (t1 - t1.shift(k)).to_numpy()
    for side, arr in (("bid", bid), ("ask", ask)):
        cum = np.stack([arr[b] for b in BANDS], axis=1)
        shells = np.diff(cum, axis=1, prepend=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            conc = np.nanmax(np.where(np.isfinite(shells), shells, -np.inf), axis=1) / np.where(arr[5.0] > 0, arr[5.0], np.nan)
        conc = np.where(np.isfinite(conc) & (conc >= 0), conc, np.nan)
        c = pd.Series(conc)
        out[:, col[f"{side}_wall_conc"]] = conc
        out[:, col[f"{side}_wall_chg_12"]] = (c - c.shift(12)).to_numpy()
    out[:, col["has_depth"]] = has.astype("float64")
    out[~has, : len(NAMES) - 1] = np.nan                    # no fresh book -> zeros after fill
    return out


register_feature_group("depth", NAMES, build_depth, LABELS)
