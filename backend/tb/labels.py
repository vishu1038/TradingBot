"""Triple-barrier style trade labels that implement the shared trade template exactly.

Template (docs/ARCHITECTURE.md "Trade template"):

* signal on the *close* of bar ``i``; the position is entered at the **open of bar i+1**
  (``open[i+1]``), no look-ahead;
* ``exit_mode="fixed"``: TP at +``tp_usd`` PnL and SL at -``sl_usd`` PnL on the slice notional
  (``slice_margin_usd * leverage``), i.e. price moves of ``tp_usd/notional`` and ``sl_usd/notional``;
* ``exit_mode="atr"``: TP = entry +- ``tp_atr`` * ATR14, SL = entry -+ ``sl_atr`` * ATR14 with ATR14
  the Wilder ATR *at the signal bar i* (``features.atr``).  Position size is chosen so that the SL
  loses ~``sl_usd`` (capped by the slice notional);
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
    if cfg["exit_mode"] not in ("fixed", "atr"):
        raise ValueError(f"exit_mode must be fixed|atr, got {cfg['exit_mode']!r}")
    return cfg


def simulate_trades(kl: pd.DataFrame, side: Literal["long", "short"], exit_cfg: dict | None = None,
                    horizon_bars: int = 288) -> dict[str, np.ndarray]:
    """Resolve a hypothetical trade for every signal bar.

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

    slip = cfg["slippage_pct"]
    sslip = cfg["stop_slippage_pct"]
    entry = o[1:m + 1] * (1.0 + sgn * slip)     # entry fill: open[i+1] with adverse slippage
    slice_notional = float(cfg["slice_margin_usd"]) * float(cfg["leverage"])
    if cfg["exit_mode"] == "fixed":
        tp_d = np.full(m, cfg["tp_usd"] / slice_notional) * entry
        sl_d = np.full(m, cfg["sl_usd"] / slice_notional) * entry
        notl = np.full(m, slice_notional)
        ok = np.isfinite(entry) & (entry > 0)
    else:
        a = _atr(hi, lo, cl, 14)[:m]
        tp_d = cfg["tp_atr"] * a
        sl_d = cfg["sl_atr"] * a
        ok = np.isfinite(entry) & (entry > 0) & np.isfinite(a) & (a > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            notl = np.minimum(slice_notional, cfg["sl_usd"] / (sl_d / entry))
    tp = entry + sgn * tp_d
    sl = entry - sgn * sl_d
    ok &= sl > 0

    tp_k = np.zeros(m, dtype="int32")     # first offset where TP touched (0 = not yet)
    sl_k = np.zeros(m, dtype="int32")
    active = np.nonzero(ok)[0]
    for k in range(1, H + 1):
        if len(active) == 0:
            break
        hk = hi[active + k]
        lk = lo[active + k]
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

    out_code = np.full(m, OUT_NA, dtype="int8")
    out_code[ok] = OUT_TIMEOUT
    out_code[resolved_tp] = OUT_TP
    out_code[resolved_sl] = OUT_SL
    k_exit = np.where(resolved_tp, tp_k, np.where(resolved_sl, sl_k, H)).astype("int32")

    # exit price (before exit slippage)
    to_close = cl[np.arange(m) + H]
    sl_open = o[np.arange(m) + np.maximum(sl_k, 1)]          # bar open of the stop bar (gap-through)
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
    label[:m] = np.where(ok, lab, np.nan)
    outcome[:m] = out_code
    exit_off[:m] = np.where(ok, k_exit, 0)
    pnl[:m] = np.where(ok, net, np.nan)
    notional[:m] = np.where(ok, notl, np.nan)
    return dict(label=label, outcome=outcome, exit_off=exit_off, pnl_usd=pnl, notional=notional)


def label_trades(kl: pd.DataFrame, side: Literal["long", "short"], exit_cfg: dict | None = None,
                 horizon_bars: int = 288) -> np.ndarray:
    """[N] float32: 1 TP first, 0 SL first / timeout, NaN if not enough future (see module doc)."""
    return simulate_trades(kl, side, exit_cfg, horizon_bars)["label"]


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
