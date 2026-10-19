"""Cross-sectional (experiment A) data layer: hourly panel, per-coin features, cross-sectional transforms, labels.

numpy / pandas only (imports under 3.14 and 3.12; scipy is imported lazily by the label helper only).

Time convention (one convention everywhere - sim, labels, live):

* ``Panel`` holds ``T`` consecutive UTC hours starting at ``t0`` (hour aligned) for ``S`` symbols; hour ``k`` is
  the bar ``[t0 + k h, t0 + (k+1) h)``.  Hourly bars are aggregated from the 5m archive (bar needs >= 6 of its
  12 five-minute bars, else NaN).  The global 5m grid (``o5/h5/l5/c5``, shape [S, T*12]) is kept for execution.
* A **decision at hour k** is taken on the CLOSE of hour k (= the open of 5m bar ``12 (k+1)``) using only
  data of hours <= k.  Entry prices are the 5m bar after the signal bar (``12 (k+1) + ...``); the forward label
  of row ``k`` is ``ln(open[k+1+h] / open[k+1])`` (exit at the open h hours later).
* Slow exchange snapshots (OI, long/short ratios) are the LAST 5m snapshot of hour k (stamped <= the last 5m bar's
  open: the same one-bar lag as ``tb.features``); funding is the last settled rate with ``calc_time`` <= the
  end of hour k.
* No survivorship look-ahead: ``tradable_mask`` admits a coin at decision k only when it has >= ``MIN_AGE_H``
  hours of valid bars behind it and its TRAILING 7-day average daily quote volume is above the floor.
* Every feature is a finite rolling window (max look-back ``LOOKBACK_H`` hours), so a live process that supplies
  ``HISTORY_H`` hours of 5m history reproduces the offline value exactly (tested).

Features (per coin) -> ``compute_raw_features``; the model input is built by ``xs_transform``: for each base
feature its cross-sectional robust z-score and rank across the tradable universe at that time, a few market
context columns and the ``has_metrics`` flag.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

BAR_MS = 300_000
HOUR_MS = 3_600_000
SUB = 12                                  # 5m bars per hour
MIN_BARS_PER_HOUR = 6
FFILL_LIMIT = 3                           # hours a missing bar / snapshot is carried forward inside features

LOOKBACK_H = 337                          # longest feature look-back (beta 168 h, then 168 h of residuals, +1)
MIN_AGE_H = 360                           # hours of valid bars required before a coin is tradable
AGE_WINDOW_OK = 340                       # ... counted as >= this many valid bars in the last MIN_AGE_H hours
LIQ_WINDOW_H = 168
MIN_DAILY_QV_USD = 20_000_000.0           # trailing 7d average daily quote volume floor (USD)
HISTORY_H = 408                           # hours of history a live caller must supply (>= MIN_AGE_H + margin)
HISTORY_BARS_5M = HISTORY_H * SUB

# ----------------------------------------------------------------------------------------------------------
BASE_FEATURES: list[str] = (
    [f"ret_{h}" for h in (1, 3, 6, 12, 24, 48, 72, 168)]
    + ["rv_24", "rv_72", "rv_168", "rv_ratio_24_168", "rng_24", "max_ret_168", "min_ret_168",
       "range_pos_24", "range_pos_168"]
    + ["log_qv_24", "qv_chg", "qv_1_24", "taker_buy_24", "taker_buy_chg", "trades_chg", "amihud_24"]
    + ["oi_chg_1", "oi_chg_4", "oi_chg_24", "oi_chg_72", "oi_price_conf_24", "oi_to_vol"]
    + ["funding_bps", "funding_avg_72", "funding_z"]
    + ["top_ls", "glob_ls", "taker_ls", "top_ls_chg_24", "glob_ls_chg_24"]
    + ["beta_168", "corr_btc_168", "resid_mom_24", "resid_mom_72", "resid_mom_168", "idio_vol_168"]
)
CTX_FEATURES = ["ctx_btc_ret_24", "ctx_btc_ret_168", "ctx_btc_rv_24", "ctx_mkt_ret_24", "ctx_mkt_disp_24"]
EXTRA_FEATURES = ["has_metrics"]
RAW_NAMES: list[str] = BASE_FEATURES + EXTRA_FEATURES
_RIDX = {n: i for i, n in enumerate(RAW_NAMES)}
MODEL_FEATURES: list[str] = ([f"{n}_z" for n in BASE_FEATURES] + [f"{n}_r" for n in BASE_FEATURES]
                             + CTX_FEATURES + EXTRA_FEATURES)
F_RAW = len(RAW_NAMES)
F_MODEL = len(MODEL_FEATURES)
Z_CLIP = 4.0
CTX_SCALE = 5.0

FEATURE_GROUPS: dict[str, list[str]] = {
    "momentum": [n for n in BASE_FEATURES if n.startswith("ret_")] + ["max_ret_168", "min_ret_168", "range_pos_24",
                                                                     "range_pos_168"],
    "volatility": ["rv_24", "rv_72", "rv_168", "rv_ratio_24_168", "rng_24", "idio_vol_168"],
    "volume": ["log_qv_24", "qv_chg", "qv_1_24", "taker_buy_24", "taker_buy_chg", "trades_chg", "amihud_24"],
    "open_interest": ["oi_chg_1", "oi_chg_4", "oi_chg_24", "oi_chg_72", "oi_price_conf_24", "oi_to_vol"],
    "funding": ["funding_bps", "funding_avg_72", "funding_z"],
    "positioning": ["top_ls", "glob_ls", "taker_ls", "top_ls_chg_24", "glob_ls_chg_24"],
    "btc_residual": ["beta_168", "corr_btc_168", "resid_mom_24", "resid_mom_72", "resid_mom_168"],
}

FEATURE_LABELS: dict[str, str] = {
    "ret_1": "1h return {:+.2f}%", "ret_3": "3h return {:+.2f}%", "ret_6": "6h return {:+.2f}%",
    "ret_12": "12h return {:+.2f}%", "ret_24": "24h return {:+.2f}%", "ret_48": "2d return {:+.2f}%",
    "ret_72": "3d return {:+.2f}%", "ret_168": "7d return {:+.2f}%",
    "resid_mom_24": "24h return vs BTC {:+.2f}%", "resid_mom_72": "3d return vs BTC {:+.2f}%",
    "resid_mom_168": "7d return vs BTC {:+.2f}%", "rv_24": "24h volatility {:.2f}%/h",
    "rv_168": "7d volatility {:.2f}%/h", "funding_bps": "funding {:+.2f} bps", "qv_chg": "volume {:+.2f} (log) vs 7d",
    "oi_chg_24": "OI {:+.1f}% in 24h", "oi_price_conf_24": "OI-price confirmation {:+.2f}",
    "beta_168": "BTC beta {:.2f}", "log_qv_24": "24h volume 10^{:.1f} USD",
}


def feature_label(name: str, value: float) -> str:
    base = name[:-2] if name.endswith(("_z", "_r")) else name
    suffix = " (rank)" if name.endswith("_r") else " (z)" if name.endswith("_z") else ""
    try:
        return FEATURE_LABELS[base].format(value) + suffix
    except Exception:
        return f"{name} {value:+.3g}"


# ----------------------------------------------------------------------------------------------------------
# panel
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Panel:
    syms: list[str]
    t0: int
    T: int
    o5: np.ndarray                    # [S, T*12] float32 (memmap or ndarray), NaN where the bar is missing
    h5: np.ndarray
    l5: np.ndarray
    c5: np.ndarray
    open: np.ndarray                  # [T, S] hourly
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    qv: np.ndarray
    tbq: np.ndarray
    ntr: np.ndarray
    oi: np.ndarray
    oiv: np.ndarray
    top_ls: np.ndarray
    glob_ls: np.ndarray
    tk_ls: np.ndarray
    fund_rate: np.ndarray             # last settled funding rate as of the end of hour k (ffilled)
    fund_ev: np.ndarray               # funding settled inside hour k (0 otherwise)
    nbar: np.ndarray                  # [T, S] int8: number of 5m bars present in the hour
    meta: dict = field(default_factory=dict)

    @property
    def S(self) -> int:
        return len(self.syms)

    def hour_ms(self) -> np.ndarray:
        return self.t0 + np.arange(self.T, dtype="int64") * HOUR_MS

    def decision_ms(self, k) -> np.ndarray | int:
        """Wall-clock time of the decision at the close of hour k."""
        return self.t0 + (np.asarray(k, dtype="int64") + 1) * HOUR_MS

    def index(self, sym: str) -> int:
        return self.syms.index(sym)


_HOURLY_FIELDS = ("open", "high", "low", "close", "qv", "tbq", "ntr", "oi", "oiv", "top_ls", "glob_ls", "tk_ls",
                  "fund_rate", "fund_ev", "nbar")


def _ms_col(s) -> np.ndarray:
    s = pd.Series(s)
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, utc=True)
        return ((dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy("int64")
    return pd.to_numeric(s).to_numpy("int64")


def _unpack(b: Any) -> tuple:
    """A 'FrameBundle' (attrs klines/metrics/funding), a dict with those keys, or a (kl, metrics, funding) tuple."""
    if isinstance(b, (tuple, list)):
        kl, m, f = (list(b) + [None, None])[:3]
        return kl, m, f
    if isinstance(b, Mapping):
        return b.get("klines"), b.get("metrics"), b.get("funding")
    return getattr(b, "klines", None), getattr(b, "metrics", None), getattr(b, "funding", None)


def _last_per_bin(bins: np.ndarray, vals: np.ndarray, T: int) -> np.ndarray:
    """vals [M, K] sorted by time with hour bin index per row -> [T, K] last row per bin (NaN elsewhere)."""
    out = np.full((T, vals.shape[1]), np.nan)
    ok = (bins >= 0) & (bins < T)
    if not ok.any():
        return out
    b, v = bins[ok], vals[ok]
    out[b] = v          # later rows overwrite earlier ones (rows are time sorted)
    return out


def _coin_arrays(kl: pd.DataFrame | None, m: pd.DataFrame | None, f: pd.DataFrame | None, t0: int, T: int,
                 o5, h5, l5, c5) -> dict:
    """Fill the 5m rows ``o5/h5/l5/c5`` (views of length T*12) and return the hourly arrays of one coin."""
    N5 = T * SUB
    nan = lambda: np.full(T, np.nan)  # noqa: E731
    res = {k: nan() for k in _HOURLY_FIELDS}
    res["fund_ev"] = np.zeros(T)
    res["nbar"] = np.zeros(T, dtype="int8")
    o5[:] = np.nan; h5[:] = np.nan; l5[:] = np.nan; c5[:] = np.nan
    if kl is None or len(kl) == 0:
        return res
    t = _ms_col(kl["open_time"])
    bi = (t - t0) // BAR_MS
    sel = (bi >= 0) & (bi < N5) & ((t - t0) % BAR_MS == 0)
    if not sel.any():
        return res
    k2 = kl.loc[sel].reset_index(drop=True)
    bi = bi[sel]
    k2 = k2.assign(_bi=bi).drop_duplicates("_bi", keep="last")
    bi = k2["_bi"].to_numpy("int64")
    o5[bi] = k2["open"].to_numpy("float32"); h5[bi] = k2["high"].to_numpy("float32")
    l5[bi] = k2["low"].to_numpy("float32"); c5[bi] = k2["close"].to_numpy("float32")
    fin = np.isfinite(c5) & np.isfinite(o5) & np.isfinite(h5) & np.isfinite(l5) & (c5 > 0)
    o5[~fin] = np.nan; h5[~fin] = np.nan; l5[~fin] = np.nan; c5[~fin] = np.nan
    fr = fin.reshape(T, SUB)
    n = fr.sum(1)
    good = n >= MIN_BARS_PER_HOUR
    res["nbar"] = n.astype("int8")
    first = np.argmax(fr, axis=1)
    last = SUB - 1 - np.argmax(fr[:, ::-1], axis=1)
    ar = np.arange(T)
    o2, h2, l2, c2 = (a.reshape(T, SUB) for a in (o5, h5, l5, c5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res["open"] = np.where(good, o2[ar, first], np.nan).astype("float64")
        res["close"] = np.where(good, c2[ar, last], np.nan).astype("float64")
        res["high"] = np.where(good, np.nanmax(np.where(fr, h2, -np.inf), axis=1), np.nan).astype("float64")
        res["low"] = np.where(good, np.nanmin(np.where(fr, l2, np.inf), axis=1), np.nan).astype("float64")
    for src, dst in (("quote_volume", "qv"), ("taker_buy_quote_volume", "tbq"), ("trades", "ntr")):
        if src not in k2.columns:
            continue
        a = np.zeros(N5)
        a[bi] = np.nan_to_num(k2[src].to_numpy("float64"))
        res[dst] = np.where(good, a.reshape(T, SUB).sum(1), np.nan)
    if m is not None and len(m):
        m = m.sort_values("create_time")
        hb = (_ms_col(m["create_time"]) - t0) // HOUR_MS
        cols = ("sum_open_interest", "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio")
        vals = m[[c for c in cols]].to_numpy("float64")
        lp = _last_per_bin(hb, vals, T)
        for j, dst in enumerate(("oi", "oiv", "top_ls", "glob_ls", "tk_ls")):
            res[dst] = lp[:, j]
    if f is not None and len(f):
        f = f.sort_values("calc_time")
        ft = _ms_col(f["calc_time"])
        fv = f["funding_rate"].to_numpy("float64")
        fb = (ft - t0) // HOUR_MS
        ev = np.zeros(T)
        last_rate = np.full(T, np.nan)
        inr = (fb >= 0) & (fb < T)
        np.add.at(ev, fb[inr], np.nan_to_num(fv[inr]))
        last_rate[fb[inr]] = fv[inr]
        before = np.nonzero(fb < 0)[0]
        if len(before):                                         # the settlement before the window seeds the ffill
            last_rate[0] = fv[before[-1]] if np.isnan(last_rate[0]) else last_rate[0]
        res["fund_ev"] = ev
        res["fund_rate"] = pd.Series(last_rate).ffill().to_numpy()
    return res


def _grid(frames: Mapping[str, Any], t0: int | None, T: int | None, complete_hours_only: bool) -> tuple[int, int]:
    firsts, lasts = [], []
    for b in frames.values():
        kl = _unpack(b)[0]
        if kl is None or len(kl) == 0:
            continue
        t = _ms_col(kl["open_time"])
        firsts.append(int(t.min())); lasts.append(int(t.max()))
    if not firsts:
        raise ValueError("no klines in any frame")
    if t0 is None:
        t0 = min(firsts) // HOUR_MS * HOUR_MS
    if T is None:
        end = max(lasts) + BAR_MS                                  # end of the latest bar
        T = int((end - t0) // HOUR_MS) if complete_hours_only else int(-(-(end - t0) // HOUR_MS))
    return int(t0), int(T)


def build_panel(frames: Mapping[str, Any], t0: int | None = None, T: int | None = None,
                cache_dir: str | Path | None = None, symbols: list[str] | None = None, progress=None) -> Panel:
    """Aggregate per-symbol 5m frames into a Panel.

    ``frames[sym]`` = (klines, metrics|None, funding|None) or an object/dict with ``klines``/``metrics``/``funding``.
    Only COMPLETE hours are used (the hour containing the latest bar is dropped unless it is complete)."""
    syms = sorted(symbols if symbols is not None else frames)
    t0, T = _grid(frames, t0, T, True)
    S, N5 = len(syms), T * SUB
    if cache_dir is not None:
        cd = Path(cache_dir)
        cd.mkdir(parents=True, exist_ok=True)
        mm = lambda nm: np.lib.format.open_memmap(cd / f"{nm}5.npy", mode="w+", dtype="float32", shape=(S, N5))  # noqa: E731
        o5, h5, l5, c5 = mm("o"), mm("h"), mm("l"), mm("c")
    else:
        o5, h5, l5, c5 = (np.empty((S, N5), dtype="float32") for _ in range(4))
    H = {k: np.full((T, S), np.nan) for k in _HOURLY_FIELDS}
    H["fund_ev"][:] = 0.0
    H["nbar"] = np.zeros((T, S), dtype="int8")
    for j, s in enumerate(syms):
        kl, m, f = _unpack(frames[s]) if s in frames else (None, None, None)
        r = _coin_arrays(kl, m, f, t0, T, o5[j], h5[j], l5[j], c5[j])
        for k in _HOURLY_FIELDS:
            H[k][:, j] = r[k]
        if progress and (j % 25 == 0 or j == S - 1):
            progress(f"[panel] {j + 1}/{S} {s}")
    if cache_dir is not None:
        for a in (o5, h5, l5, c5):
            a.flush()
        o5, h5, l5, c5 = (np.load(cd / f"{nm}5.npy", mmap_mode="r") for nm in "ohlc")
    return Panel(syms=syms, t0=t0, T=T, o5=o5, h5=h5, l5=l5, c5=c5, meta={"built_from": "frames"},
                 **{k: (H[k].astype("float32") if k in ("qv", "tbq", "ntr", "oi", "oiv", "top_ls", "glob_ls", "tk_ls")
                        else H[k]) for k in _HOURLY_FIELDS})


def _archive_key(root: Path, syms: list[str]) -> str:
    h = hashlib.sha1(json.dumps([syms, str(root), "v1", MIN_BARS_PER_HOUR]).encode())
    for s in syms:
        for sub in (f"klines/5m/{s}.parquet", f"metrics/{s}.parquet", f"funding/{s}.parquet"):
            p = root / sub
            h.update(f"{sub}:{p.stat().st_size if p.exists() else -1}".encode())
    return h.hexdigest()[:16]


def build_panel_from_archive(root: str | Path, cache_root: str | Path, symbols: list[str] | None = None,
                             progress=print, min_days: float = 20.0) -> Panel:
    """Panel for the parquet archive (cached under ``cache_root/xsec_panel_<key>``).  Streams one coin at a time."""
    root = Path(root)
    allsyms = sorted(p.stem for p in (root / "klines" / "5m").glob("*.parquet"))
    syms = [s for s in (symbols or allsyms) if s in allsyms]
    key = _archive_key(root, syms)
    cd = Path(cache_root) / f"xsec_panel_{key}"
    if (cd / "meta.json").exists():
        return load_panel(cd)
    # grid from the first / last open_time of every coin (reads only that column)
    firsts, lasts = [], []
    for s in syms:
        t = pd.read_parquet(root / "klines" / "5m" / f"{s}.parquet", columns=["open_time"])["open_time"].to_numpy("int64")
        firsts.append(int(t.min())); lasts.append(int(t.max()))
    t0 = min(firsts) // HOUR_MS * HOUR_MS
    T = int((max(lasts) + BAR_MS - t0) // HOUR_MS)
    S, N5 = len(syms), T * SUB
    cd.mkdir(parents=True, exist_ok=True)
    mm = lambda nm: np.lib.format.open_memmap(cd / f"{nm}5.npy", mode="w+", dtype="float32", shape=(S, N5))  # noqa: E731
    o5, h5, l5, c5 = mm("o"), mm("h"), mm("l"), mm("c")
    H = {k: np.full((T, S), np.nan) for k in _HOURLY_FIELDS}
    H["fund_ev"][:] = 0.0
    H["nbar"] = np.zeros((T, S), dtype="int8")
    for j, s in enumerate(syms):
        rd = lambda sub: pd.read_parquet(root / sub / f"{s}.parquet") if (root / sub / f"{s}.parquet").exists() else None  # noqa: E731
        kl = pd.read_parquet(root / "klines" / "5m" / f"{s}.parquet")
        m = rd("metrics"); f = rd("funding")
        r = _coin_arrays(kl, m, f, t0, T, o5[j], h5[j], l5[j], c5[j])
        for k in _HOURLY_FIELDS:
            H[k][:, j] = r[k]
        if progress and (j % 20 == 0 or j == S - 1):
            progress(f"[panel] {j + 1}/{S} {s}")
    for a in (o5, h5, l5, c5):
        a.flush()
    np.savez(cd / "hourly.npz", **{k: (H[k].astype("float32") if k in ("qv", "tbq", "ntr", "oi", "oiv", "top_ls", "glob_ls", "tk_ls")
                                       else H[k]) for k in _HOURLY_FIELDS})
    (cd / "meta.json").write_text(json.dumps({"syms": syms, "t0": t0, "T": T, "key": key}))
    return load_panel(cd)


def load_panel(cd: str | Path) -> Panel:
    cd = Path(cd)
    meta = json.loads((cd / "meta.json").read_text())
    with np.load(cd / "hourly.npz") as z:
        H = {k: z[k] for k in z.files}
    o5, h5, l5, c5 = (np.load(cd / f"{nm}5.npy", mmap_mode="r") for nm in "ohlc")
    return Panel(syms=meta["syms"], t0=meta["t0"], T=meta["T"], o5=o5, h5=h5, l5=l5, c5=c5, meta=meta, **H)


# ----------------------------------------------------------------------------------------------------------
# tradable universe (trailing information only)
# ----------------------------------------------------------------------------------------------------------
def tradable_mask(panel: Panel, min_daily_qv_usd: float = MIN_DAILY_QV_USD, min_age_h: int = MIN_AGE_H,
                  age_ok: int = AGE_WINDOW_OK, liq_window_h: int = LIQ_WINDOW_H) -> np.ndarray:
    """[T, S] bool: coin may be ranked/traded at the decision on the close of hour k.

    valid bar k  AND  >= ``age_ok`` valid bars in the last ``min_age_h`` hours (listing age)  AND  trailing
    ``liq_window_h``-hour mean daily quote volume >= floor.  Uses data of hours <= k only."""
    valid = np.isfinite(panel.close) & (panel.nbar >= MIN_BARS_PER_HOUR)
    v = pd.DataFrame(valid.astype("float64"))
    age = v.rolling(min_age_h, min_periods=min_age_h).sum().to_numpy() >= age_ok
    qv = pd.DataFrame(np.nan_to_num(panel.qv.astype("float64")))
    daily = qv.rolling(liq_window_h, min_periods=liq_window_h).mean().to_numpy() * 24.0
    liq = np.nan_to_num(daily) >= min_daily_qv_usd
    return valid & age & liq


# ----------------------------------------------------------------------------------------------------------
# per-coin raw features
# ----------------------------------------------------------------------------------------------------------
def _R(df: pd.DataFrame, w: int):
    return df.rolling(w, min_periods=w)


def compute_raw_features(panel: Panel, btc_sym: str = "BTCUSDT") -> np.ndarray:
    """[T, S, F_RAW] float32 (NaN where not computable).  Row k uses hours <= k only."""
    T, S = panel.T, panel.S
    ffl = lambda a, lim=FFILL_LIMIT: pd.DataFrame(a.astype("float64")).ffill(limit=lim)  # noqa: E731
    c = ffl(panel.close)
    hi = ffl(panel.high)
    lo = ffl(panel.low)
    pos = lambda df: df.where(df > 0)  # noqa: E731
    lc = np.log(pos(c))
    lr = lc.diff()                                              # hourly log return
    out: dict[str, pd.DataFrame | np.ndarray] = {}
    for h in (1, 3, 6, 12, 24, 48, 72, 168):
        out[f"ret_{h}"] = 100.0 * (lc - lc.shift(h))
    for w in (24, 72, 168):
        out[f"rv_{w}"] = 100.0 * _R(lr, w).std()
    out["rv_ratio_24_168"] = np.log(out["rv_24"].where(out["rv_24"] > 0) / out["rv_168"].where(out["rv_168"] > 0))
    out["rng_24"] = 100.0 * _R((hi - lo) / c, 24).mean()
    out["max_ret_168"] = 100.0 * _R(lr, 168).max()
    out["min_ret_168"] = 100.0 * _R(lr, 168).min()
    for w in (24, 168):
        mx, mn = _R(hi, w).max(), _R(lo, w).min()
        out[f"range_pos_{w}"] = (c - mn) / (mx - mn).where(mx - mn > 0)
    qv = pd.DataFrame(np.nan_to_num(panel.qv.astype("float64")))
    tbq = pd.DataFrame(np.nan_to_num(panel.tbq.astype("float64")))
    ntr = pd.DataFrame(np.nan_to_num(panel.ntr.astype("float64")))
    q24, q168 = _R(qv, 24).sum(), _R(qv, 168).sum()
    out["log_qv_24"] = np.log10(q24 + 1.0)
    out["qv_chg"] = np.log((q24 + 1.0) / (q168 / 7.0 + 1.0))
    out["qv_1_24"] = np.log((qv + 1.0) * 24.0 / (q24 + 1.0))
    tb24, tb168 = _R(tbq, 24).sum() / q24.where(q24 > 0), _R(tbq, 168).sum() / q168.where(q168 > 0)
    out["taker_buy_24"] = tb24
    out["taker_buy_chg"] = tb24 - tb168
    out["trades_chg"] = np.log((_R(ntr, 24).sum() + 1.0) / (_R(ntr, 168).sum() / 7.0 + 1.0))
    out["amihud_24"] = np.log1p(_R(lr.abs() * 100.0 / (qv / 1e6 + 1e-3), 24).mean())
    oiv = ffl(panel.oiv)
    loiv = np.log(pos(oiv))
    for h in (1, 4, 24, 72):
        out[f"oi_chg_{h}"] = 100.0 * (loiv - loiv.shift(h))
    out["oi_price_conf_24"] = out["oi_chg_24"] * np.sign(out["ret_24"])
    out["oi_to_vol"] = np.log(pos(oiv) / (q24 + 1.0))
    fr = pd.DataFrame(panel.fund_rate.astype("float64")) * 1e4              # bps (already forward filled)
    out["funding_bps"] = fr
    out["funding_avg_72"] = _R(fr, 72).mean()
    sd = _R(fr, 336).std()
    out["funding_z"] = (fr - _R(fr, 336).mean()) / sd.where(sd > 1e-9)
    out["top_ls"] = np.log(pos(ffl(panel.top_ls)))
    out["glob_ls"] = np.log(pos(ffl(panel.glob_ls)))
    out["taker_ls"] = np.log(pos(ffl(panel.tk_ls)))
    out["top_ls_chg_24"] = out["top_ls"] - out["top_ls"].shift(24)
    out["glob_ls_chg_24"] = out["glob_ls"] - out["glob_ls"].shift(24)
    # BTC residual block
    nanf = pd.DataFrame(np.full((T, S), np.nan))
    if btc_sym in panel.syms:
        bj = panel.syms.index(btc_sym)
        rb = lr.iloc[:, bj]
        beta = _R(lr, 168).cov(rb).div(_R(rb, 168).var(), axis=0)
        resid = lr - beta.mul(rb, axis=0)
        out["beta_168"] = beta
        out["corr_btc_168"] = _R(lr, 168).corr(rb)
        out["resid_mom_24"] = 100.0 * _R(resid, 24).sum()
        out["resid_mom_72"] = 100.0 * _R(resid, 72).sum()
        out["resid_mom_168"] = 100.0 * _R(resid, 168).sum()
        out["idio_vol_168"] = 100.0 * _R(resid, 168).std()
        for n in ("beta_168", "corr_btc_168", "resid_mom_24", "resid_mom_72", "resid_mom_168", "idio_vol_168"):
            out[n].iloc[:, bj] = np.nan                         # BTC vs itself is degenerate
    else:
        for n in ("beta_168", "corr_btc_168", "resid_mom_24", "resid_mom_72", "resid_mom_168", "idio_vol_168"):
            out[n] = nanf
    has = np.isfinite(ffl(panel.oiv).to_numpy())
    cube = np.full((T, S, F_RAW), np.nan, dtype="float32")
    for n in BASE_FEATURES:
        v = np.asarray(out[n], dtype="float64")
        cube[:, :, _RIDX[n]] = np.where(np.isfinite(v), v, np.nan)
    cube[:, :, _RIDX["has_metrics"]] = has.astype("float32")
    return cube


# ----------------------------------------------------------------------------------------------------------
# cross-sectional transform
# ----------------------------------------------------------------------------------------------------------
def _xs_rank_z(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v [n, S] (NaN = not in the universe / unknown) -> (rank in [-0.5, 0.5], robust z clipped) with 0 for NaN."""
    fin = np.isfinite(v)
    cnt = fin.sum(1)
    rk = pd.DataFrame(v).rank(axis=1, method="average").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rk - 0.5) / cnt[:, None] - 0.5
    r = np.where(fin, r, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(v, axis=1)
        mad = np.nanmedian(np.abs(v - med[:, None]), axis=1) * 1.4826
        sd = np.nanstd(v, axis=1)
    scale = np.where(mad > 1e-9, mad, sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (v - med[:, None]) / scale[:, None]
    z = np.where(fin & np.isfinite(z), np.clip(z, -Z_CLIP, Z_CLIP), 0.0)
    return r.astype("float32"), z.astype("float32")


def xs_transform(raw: np.ndarray, mask: np.ndarray, btc_idx: int | None = None) -> np.ndarray:
    """raw [n, S, F_RAW] + tradable mask [n, S] -> model inputs [n, S, F_MODEL] float32 (zeros where not tradable).

    Cross-sectional statistics only use tradable coins at that decision time."""
    n, S, _ = raw.shape
    X = np.zeros((n, S, F_MODEL), dtype="float32")
    nb = len(BASE_FEATURES)
    for j, name in enumerate(BASE_FEATURES):
        v = np.where(mask, raw[:, :, _RIDX[name]], np.nan).astype("float64")
        r, z = _xs_rank_z(v)
        X[:, :, j] = z
        X[:, :, nb + j] = r
    # context
    ctx0 = 2 * nb
    r24 = np.where(mask, raw[:, :, _RIDX["ret_24"]], np.nan).astype("float64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mk = np.nanmean(r24, axis=1)
        dp = np.nanstd(r24, axis=1)
    cx = np.zeros((n, len(CTX_FEATURES)))
    if btc_idx is not None:
        cx[:, 0] = raw[:, btc_idx, _RIDX["ret_24"]]
        cx[:, 1] = raw[:, btc_idx, _RIDX["ret_168"]]
        cx[:, 2] = raw[:, btc_idx, _RIDX["rv_24"]]
    cx[:, 3] = mk
    cx[:, 4] = dp
    cx = np.clip(np.nan_to_num(cx) / CTX_SCALE, -10, 10)
    X[:, :, ctx0:ctx0 + len(CTX_FEATURES)] = cx[:, None, :].astype("float32")
    X[:, :, ctx0 + len(CTX_FEATURES)] = np.nan_to_num(raw[:, :, _RIDX["has_metrics"]])
    X[~mask] = 0.0
    return X


def btc_index(panel_syms: list[str], btc_sym: str = "BTCUSDT") -> int | None:
    return panel_syms.index(btc_sym) if btc_sym in panel_syms else None


# ----------------------------------------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------------------------------------
def forward_returns(panel: Panel, horizon_h: int, kind: str = "simple") -> np.ndarray:
    """[T, S] return from the open of hour k+1 to the open of hour k+1+h in percent (``kind`` "simple" = b/a - 1, the
    quantity a position earns, or "log"); NaN when either open is missing / beyond the data."""
    T = panel.T
    out = np.full((T, panel.S), np.nan)
    h = int(horizon_h)
    if T > h + 1:
        a = panel.open[1:T - h]
        b = panel.open[1 + h:T]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:T - h - 1] = 100.0 * (np.log(b / a) if kind == "log" else (b / a - 1.0))
    out[~np.isfinite(out)] = np.nan
    return out


def risk_scaled(fwd: np.ndarray, sigma_day_frac: np.ndarray) -> np.ndarray:
    """Forward return in units of the coin's trailing daily volatility (what an equal-risk leg earns per unit of risk):
    ``fwd [%] / (100 * sigma_day)``; NaN where the volatility is unknown."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = fwd / (100.0 * sigma_day_frac)
    out[~np.isfinite(out)] = np.nan
    return out


def relative_returns(fwd: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """fwd minus the universe mean at the same decision time (universe = tradable coins with a known outcome)."""
    f = np.where(mask & np.isfinite(fwd), fwd, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(f, axis=1)
    return f - mu[:, None]


def rank_target(rel: np.ndarray, gauss: bool = True) -> np.ndarray:
    """Per-row cross-sectional rank of ``rel`` (NaN stays NaN): gaussianised (default) or uniform in [-0.5, 0.5]."""
    fin = np.isfinite(rel)
    cnt = fin.sum(1)
    rk = pd.DataFrame(rel).rank(axis=1, method="average").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        p = (rk - 0.5) / cnt[:, None]
    if gauss:
        from scipy.special import ndtri
        out = ndtri(np.clip(p, 1e-6, 1 - 1e-6))
    else:
        out = p - 0.5
    return np.where(fin, out, np.nan).astype("float32")


# ----------------------------------------------------------------------------------------------------------
# baselines' scores (same universe and layout as model scores)
# ----------------------------------------------------------------------------------------------------------
def momentum_scores(cube: np.ndarray, mask: np.ndarray, name: str = "ret_168", sign: float = 1.0) -> np.ndarray:
    """Plain rule: rank by past return (default 7 d).  ``sign=-1`` gives the reversal rule."""
    s = sign * cube[:, :, _RIDX[name]].astype("float64")
    return np.where(mask & np.isfinite(s), s, np.nan)


def random_scores(mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(mask, rng.random(mask.shape), np.nan)


def raw_index(name: str) -> int:
    return _RIDX[name]
