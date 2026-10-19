"""Triple-barrier style trade labels that implement the shared trade template exactly.

Template (docs/ARCHITECTURE.md "Trade template"):

* signal on the *close* of bar ``i``; the position is entered at the **open of bar i+1**
  (``open[i+1]``), no look-ahead;
* ``exit_mode="fixed"``: TP at +``tp_usd`` PnL and SL at -``sl_usd`` PnL on the slice notional
  (``slice_margin_usd * leverage``), i.e. price moves of ``tp_usd/notional`` and ``sl_usd/notional``;
* ``exit_mode="atr"``: TP = entry +- ``tp_atr`` * ATR14, SL = entry -+ ``sl_atr`` * ATR14 with ATR14
  the Wilder ATR *at the signal bar i* (``features.atr``).  Position size is chosen so that the SL
  loses ~``sl_usd`` (capped by the slice notional);
* ``exit_mode="pct"``: TP = fill * (1 +- ``tp_pct``/100), SL = fill * (1 -+ ``sl_pct``/100), notional = the slice notional;
* ``atr_tf`` (atr mode): timeframe of the ATR - ``"5m"`` (default, ATR14 of the signal bar) or ``"1h"`` / ``"4h"`` /
  ``"1d"``: SMA-ATR14 of the last CLOSED bar of that timeframe at the signal bar's close (``features_mtf.htf_atr_5m``);
* both barriers inside the same 5m candle -> **SL first** (pessimistic);
* the trade lives for at most ``horizon_bars`` bars (bars i+1 .. i+horizon).  A trade that hits
  neither barrier is a **timeout: label 0**, closed at the close of the last bar;
* rows without ``horizon_bars`` bars of future (or without ATR in atr mode) are **NaN**.

``label_trades`` returns the barrier label (1 = TP first, 0 = SL first / timeout).  Fill model mirrors
``tb.sim.template``: entry fill = open[i+1] with adverse ``slippage_pct``; TP/SL levels are GROSS
price levels measured from that fill; TP is a resting limit (fills at its price); SL is a stop-market
(fills at the stop, or the bar open if the bar gaps through it) with ``slippage_pct +
stop_slippage_pct`` adverse; timeout closes at the last bar's close with ``slippage_pct``; taker fee
on both sides.  Funding is not modelled here (at most 3 settlements in a 24 h horizon, second-order).

numpy only; vectorised over signal bars, looping over the (<= horizon) offsets.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .features import atr as _atr

DEFAULT_EXIT_CFG: dict = {
    "exit_mode": "fixed",       # fixed | atr
    "slice_margin_usd": 20.0,
    "leverage": 8,
    "tp_usd": 5.0,
    "sl_usd": 10.0,
    "tp_atr": 1.5,
    "sl_atr": 1.0,
    "tp_pct": 3.0,              # exit_mode "pct": price-move targets in percent
    "sl_pct": 6.0,
    "atr_tf": "5m",             # timeframe of the ATR for exit_mode "atr": 5m | 1h | 4h | 1d
    "fee_taker": 0.0005,
    "slippage_pct": 0.0002,
    "stop_slippage_pct": 0.0005,
}

OUT_TP, OUT_SL, OUT_TIMEOUT, OUT_NA = 1, 0, 2, -1


def full_cfg(exit_cfg: dict | None) -> dict:
    cfg = dict(DEFAULT_EXIT_CFG)
    if exit_cfg:
        cfg.update({k: v for k, v in exit_cfg.items() if v is not None})
        if "mode" in exit_cfg and "exit_mode" not in exit_cfg:
            cfg["exit_mode"] = exit_cfg["mode"]
        for alias, key in (("slippage", "slippage_pct"), ("stop_slippage_extra", "stop_slippage_pct")):
            if alias in exit_cfg and key not in exit_cfg:
                cfg[key] = exit_cfg[alias]
    if cfg["exit_mode"] not in ("fixed", "atr", "pct"):
        raise ValueError(f"exit_mode must be fixed|atr|pct, got {cfg['exit_mode']!r}")
    if cfg["atr_tf"] not in ("5m", "1h", "4h", "1d"):
        raise ValueError(f"atr_tf must be 5m|1h|4h|1d, got {cfg['atr_tf']!r}")
    return cfg


def exit_atr(kl: pd.DataFrame, cfg: dict) -> np.ndarray:
    """ATR (price units) per 5m signal bar used to size ATR exits: Wilder ATR14 of the 5m bars (``atr_tf="5m"``) or
    the SMA-ATR14 of the last closed ``atr_tf`` bar (see ``features_mtf.htf_atr_5m``)."""
    if cfg.get("atr_tf", "5m") == "5m":
        return _atr(kl["high"].to_numpy("float64"), kl["low"].to_numpy("float64"), kl["close"].to_numpy("float64"), 14)
    from .features_mtf import htf_atr_5m
    return htf_atr_5m(kl, cfg["atr_tf"], 14)


def simulate_trades(kl: pd.DataFrame, side: Literal["long", "short"], exit_cfg: dict | None = None,
                    horizon_bars: int = 288, rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Resolve a hypothetical trade for every signal bar (or only for ``rows`` - indices of signal bars - which
    makes decision-cadence datasets cheap; all other entries of the returned arrays stay n/a).

    Returns arrays of length N:
      label    float32  1 TP first, 0 SL first / timeout, NaN not enough future
      outcome  int8     1 TP, 0 SL, 2 timeout, -1 n/a
      exit_off int32    bars after the signal bar at which the trade closed (1..horizon), 0 n/a
      pnl_usd  float32  net PnL in USD (fees + slippage), NaN n/a
      notional float32  position notional in USD, NaN n/a
    """
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    cfg = full_cfg(exit_cfg)
    n = len(kl)
    o = kl["open"].to_numpy("float64")
    hi = kl["high"].to_numpy("float64")
    lo = kl["low"].to_numpy("float64")
    cl = kl["close"].to_numpy("float64")
    H = int(horizon_bars)
    sgn = 1.0 if side == "long" else -1.0

    label = np.full(n, np.nan, dtype="float32")
    outcome = np.full(n, OUT_NA, dtype="int8")
    exit_off = np.zeros(n, dtype="int32")
    pnl = np.full(n, np.nan, dtype="float32")
    notional = np.full(n, np.nan, dtype="float32")
    m = n - H                       # signal rows with a full horizon of future: i in [0, m)
    if m <= 0:
        return dict(label=label, outcome=outcome, exit_off=exit_off, pnl_usd=pnl, notional=notional)
    sig = np.arange(m) if rows is None else np.unique(np.asarray(rows, dtype="int64"))
    sig = sig[(sig >= 0) & (sig < m)]
    ns = len(sig)
    if ns == 0:
        return dict(label=label, outcome=outcome, exit_off=exit_off, pnl_usd=pnl, notional=notional)

    slip = cfg["slippage_pct"]
    sslip = cfg["stop_slippage_pct"]
    entry = o[sig + 1] * (1.0 + sgn * slip)     # entry fill: open[i+1] with adverse slippage
    slice_notional = float(cfg["slice_margin_usd"]) * float(cfg["leverage"])
    if cfg["exit_mode"] == "fixed":
        tp_d = np.full(ns, cfg["tp_usd"] / slice_notional) * entry
        sl_d = np.full(ns, cfg["sl_usd"] / slice_notional) * entry
        notl = np.full(ns, slice_notional)
        ok = np.isfinite(entry) & (entry > 0)
    elif cfg["exit_mode"] == "pct":
        tp_d = cfg["tp_pct"] / 100.0 * entry
        sl_d = cfg["sl_pct"] / 100.0 * entry
        notl = np.full(ns, slice_notional)
        ok = np.isfinite(entry) & (entry > 0)
    else:
        a = exit_atr(kl, cfg)[sig]
        tp_d = cfg["tp_atr"] * a
        sl_d = cfg["sl_atr"] * a
        ok = np.isfinite(entry) & (entry > 0) & np.isfinite(a) & (a > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            notl = np.minimum(slice_notional, cfg["sl_usd"] / (sl_d / entry))
    tp = entry + sgn * tp_d
    sl = entry - sgn * sl_d
    ok &= sl > 0

    tp_k = np.zeros(ns, dtype="int32")    # first offset where TP touched (0 = not yet)
    sl_k = np.zeros(ns, dtype="int32")
    active = np.nonzero(ok)[0]
    for k in range(1, H + 1):
        if len(active) == 0:
            break
        hk = hi[sig[active] + k]
        lk = lo[sig[active] + k]
        if sgn > 0:
            t_hit = hk >= tp[active]
            s_hit = lk <= sl[active]
        else:
            t_hit = lk <= tp[active]
            s_hit = hk >= sl[active]
        done = t_hit | s_hit
        if done.any():
            d = active[done]
            # SL wins ties inside the same candle
            sl_k[d[s_hit[done]]] = k
            tp_k[d[t_hit[done] & ~s_hit[done]]] = k
            active = active[~done]
    resolved_tp = tp_k > 0
    resolved_sl = sl_k > 0

    out_code = np.full(ns, OUT_NA, dtype="int8")
    out_code[ok] = OUT_TIMEOUT
    out_code[resolved_tp] = OUT_TP
    out_code[resolved_sl] = OUT_SL
    k_exit = np.where(resolved_tp, tp_k, np.where(resolved_sl, sl_k, H)).astype("int32")

    # exit price (before exit slippage)
    to_close = cl[sig + H]
    sl_open = o[sig + np.maximum(sl_k, 1)]                   # bar open of the stop bar (gap-through)
    sl_base = np.minimum(sl, sl_open) if sgn > 0 else np.maximum(sl, sl_open)
    exit_px = np.where(resolved_tp, tp,
                       np.where(resolved_sl, sl_base * (1.0 - sgn * (slip + sslip)),
                                to_close * (1.0 - sgn * slip)))
    fill = entry
    with np.errstate(divide="ignore", invalid="ignore"):
        qty = notl / fill
        gross = sgn * qty * (exit_px - fill)
        fees = cfg["fee_taker"] * qty * (fill + exit_px)
    net = gross - fees

    lab = np.where(resolved_tp, 1.0, 0.0)
    label[sig] = np.where(ok, lab, np.nan)
    outcome[sig] = out_code
    exit_off[sig] = np.where(ok, k_exit, 0)
    pnl[sig] = np.where(ok, net, np.nan)
    notional[sig] = np.where(ok, notl, np.nan)
    return dict(label=label, outcome=outcome, exit_off=exit_off, pnl_usd=pnl, notional=notional)


def label_trades(kl: pd.DataFrame, side: Literal["long", "short"], exit_cfg: dict | None = None,
                 horizon_bars: int = 288, rows: np.ndarray | None = None) -> np.ndarray:
    """[N] float32: 1 TP first, 0 SL first / timeout, NaN if not enough future (see module doc)."""
    return simulate_trades(kl, side, exit_cfg, horizon_bars, rows)["label"]


# --------------------------------------------------------------------------------------------
# decision cadence / cooldown (multi-timeframe "user style" trading)
# --------------------------------------------------------------------------------------------
def cooldown_select(signal_idx: np.ndarray, exit_off: np.ndarray, cooldown_bars: int = 0) -> np.ndarray:
    """Greedy per-coin cooldown on ONE coin's time-ordered signals.

    ``signal_idx`` are bar indices (ascending) of candidate entries, ``exit_off[k]`` the bars the k-th trade stays
    open (``simulate_trades(...)["exit_off"][signal_idx]``, >= 1).  A candidate is taken if no earlier TAKEN trade is
    still open at its bar or within ``cooldown_bars`` bars after that trade's exit (trade k occupies the entry bar
    ``i`` up to bar ``i + exit_off`` and the cooldown extends the block to ``i + exit_off + cooldown_bars``).
    Returns a boolean mask over ``signal_idx``."""
    idx = np.asarray(signal_idx, dtype="int64")
    off = np.maximum(np.asarray(exit_off, dtype="int64"), 1)
    take = np.zeros(len(idx), dtype=bool)
    blocked_until = -1
    for k in range(len(idx)):
        if idx[k] >= blocked_until:
            take[k] = True
            blocked_until = int(idx[k] + off[k] + cooldown_bars)
    return take


def breakeven_win_rate(sim: dict[str, np.ndarray]) -> float:
    """Win rate at which expectancy is zero for this exit config, from the simulated population:
    p* = -l / (w - l) with w = mean net PnL of TP outcomes and l = mean net PnL of every other
    resolved outcome (SL and timeout, i.e. all label-0 trades)."""
    oc = sim["outcome"]
    pn = sim["pnl_usd"].astype("float64")
    win = oc == OUT_TP
    lose = (oc != OUT_TP) & (oc != OUT_NA)
    if not (win.any() and lose.any()):
        return float("nan")
    w, lo = pn[win].mean(), pn[lose].mean()
    if w <= lo:
        return float("nan")
    return float(-lo / (w - lo))


def breakeven_from_cfg(exit_cfg: dict | None = None, typical_atr_pct: float = 0.5) -> float:
    """Analytic break-even win rate (wins exit at TP, losses at SL, no timeouts).  fixed: from the
    usd amounts; atr: needs a typical ATR% to size the (slice-capped) notional (default 0.5 %).
    Levels are measured from the slipped fill, so entry slippage does not change the payoffs."""
    cfg = full_cfg(exit_cfg)
    fee = cfg["fee_taker"]
    stop_slip = cfg["slippage_pct"] + cfg["stop_slippage_pct"]
    cap = cfg["slice_margin_usd"] * cfg["leverage"]
    if cfg["exit_mode"] == "fixed":
        notional, gw, gl = cap, cfg["tp_usd"], cfg["sl_usd"]
    else:
        p = typical_atr_pct / 100.0
        notional = min(cap, cfg["sl_usd"] / (cfg["sl_atr"] * p))
        gw, gl = notional * cfg["tp_atr"] * p, notional * cfg["sl_atr"] * p
    w = gw - 2 * fee * notional
    lo = gl + 2 * fee * notional + stop_slip * notional
    return float(lo / (w + lo))


# --------------------------------------------------------------------------------------------
# auxiliary (multi-task) targets - all use only data AFTER the signal bar close, like the labels
# --------------------------------------------------------------------------------------------
AUX_NAMES = ("fvol", "up_exc", "dn_exc")
AUX_VOL_BARS = 36           # forward realised-volatility window (3 h)
AUX_EXC_BARS = 288          # excursion window (24 h)


def aux_targets(kl: pd.DataFrame, vol_bars: int = AUX_VOL_BARS, exc_bars: int = AUX_EXC_BARS) -> dict[str, np.ndarray]:
    """Per signal bar i (float32, NaN where the future or the ATR is missing):

    fvol    ln( sqrt(mean(r_t^2, t=i+1..i+vol_bars)) / (ATR14_i / close_i) )   forward realised vol vs current ATR%
    up_exc  ln(1 + max(high[i+1..i+exc_bars] - open[i+1], 0) / ATR14_i)        max favourable excursion of a long
    dn_exc  ln(1 + max(open[i+1] - min(low[i+1..i+exc_bars]), 0) / ATR14_i)    max adverse excursion of a long
                                                                               (= the favourable one of a short)
    r_t = ln(close_t / close_{t-1}).  The excursion window is anchored at the entry bar (open[i+1])."""
    n = len(kl)
    o = kl["open"].to_numpy("float64")
    hi = kl["high"].to_numpy("float64")
    lo = kl["low"].to_numpy("float64")
    cl = kl["close"].to_numpy("float64")
    a = _atr(hi, lo, cl, 14)
    out = {k: np.full(n, np.nan, dtype="float32") for k in AUX_NAMES}
    if n < 2:
        return out
    r = np.diff(np.log(cl), prepend=np.nan)
    r2 = np.nan_to_num(r * r)
    cs = np.concatenate([[0.0], np.cumsum(r2)])               # cs[k] = sum r2[:k]
    i = np.arange(n)
    ok = i + vol_bars < n
    fwd = np.full(n, np.nan)
    fwd[ok] = (cs[i[ok] + vol_bars + 1] - cs[i[ok] + 1]) / vol_bars     # r2[i+1 .. i+vol_bars]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["fvol"][:] = np.log(np.sqrt(np.maximum(fwd, 1e-18)) / (a / cl))
        s_hi = pd.Series(hi[::-1]).rolling(exc_bars, min_periods=exc_bars).max().to_numpy()[::-1]   # max hi[j..j+exc-1]
        s_lo = pd.Series(lo[::-1]).rolling(exc_bars, min_periods=exc_bars).min().to_numpy()[::-1]
        fmax = np.concatenate([s_hi[1:], [np.nan]])           # window starting at i+1
        fmin = np.concatenate([s_lo[1:], [np.nan]])
        ent = np.concatenate([o[1:], [np.nan]])
        out["up_exc"][:] = np.log1p(np.maximum(fmax - ent, 0.0) / a)
        out["dn_exc"][:] = np.log1p(np.maximum(ent - fmin, 0.0) / a)
    for k in AUX_NAMES:
        out[k][~np.isfinite(out[k])] = np.nan
    return out
