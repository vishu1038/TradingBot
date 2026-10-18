"""Position sizing, stop/trail maths, kill switch, correlation cap, stop invariant. Pure functions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

from .config import CorrCap, Settings

if TYPE_CHECKING:  # keep tb.risk import-light (the backtester imports it; avoids a cycle with the exchange pkg)
    from .exchange.base import ConditionalOrder, SymbolFilters


# ----------------------------------------------------------------------------- drawdown derating
def derate_multiplier(dd_pct: float, kill_dd_pct: float) -> float:
    """Risk multiplier from account drawdown: 1.0 until 40% of the kill limit, linear to 0.5 at 80%,
    linear to 0 at 100% (kill 25: 10% -> 1.0, 20% -> 0.5, 25% -> 0; kill 50: 20/40/50).
    Shared with the backtester (import from tb.risk)."""
    if kill_dd_pct <= 0:
        return 0.0
    f = max(0.0, dd_pct) / kill_dd_pct
    if f <= 0.4:
        return 1.0
    if f <= 0.8:
        return 1.0 - 0.5 * (f - 0.4) / 0.4
    if f < 1.0:
        return 0.5 * (1.0 - f) / 0.2
    return 0.0


# ----------------------------------------------------------------------------- indicators
def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    high, low, close = (np.asarray(x, dtype="float64") for x in (high, low, close))
    prev = np.concatenate([[close[0]], close[:-1]])
    return np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])


def atr14(high: Sequence[float], low: Sequence[float], close: Sequence[float], n: int = 14) -> float | None:
    """Wilder ATR (RMA of true range, alpha=1/n) at the last bar, identical to ``features.atr`` /
    ``labels`` so the engine's exit template matches the model's training labels. None if short."""
    if len(close) < n + 1:
        return None
    tr = true_range(np.asarray(high), np.asarray(low), np.asarray(close))
    v = float(pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().iloc[-1])
    return v if math.isfinite(v) and v > 0 else None


def atr_pct(high: Sequence[float], low: Sequence[float], close: Sequence[float], bars: int = 72) -> float | None:
    """Mean true range over the last ``bars`` (6h of 5m) as % of the last close."""
    if len(close) < bars + 1:
        bars = len(close) - 1
    if bars < 14:
        return None
    tr = true_range(np.asarray(high), np.asarray(low), np.asarray(close))
    return float(np.mean(tr[-bars:]) / float(close[-1]) * 100.0)


# ----------------------------------------------------------------------------- liquidation / sizing
def liq_price(side: str, entry: float, leverage: float, mmr: float = 0.005) -> float:
    """Isolated liquidation price, same (Binance) form as ``tb.sim.template.liq_price``:
    long entry*(1-1/L)/(1-mmr), short entry*(1+1/L)/(1+mmr) (margin = notional/L, no fees)."""
    if side == "long":
        return entry * (1.0 - 1.0 / leverage) / (1.0 - mmr)
    return entry * (1.0 + 1.0 / leverage) / (1.0 + mmr)


@dataclass
class Sizing:
    ok: bool
    reason: str = ""
    qty: float = 0.0
    notional: float = 0.0
    margin: float = 0.0
    entry: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    risk_usd: float = 0.0
    reward_usd: float = 0.0
    atr: float | None = None
    liq_price: float = 0.0


def size_trade(cfg: Settings, side: str, entry: float, atr: float | None, f: SymbolFilters,
               available: float | None = None, risk_mult: float = 1.0) -> Sizing:
    """Qty + SL/TP prices for the trade template. ``side`` is 'long' | 'short'.

    ``risk_mult`` (drawdown derating / drift / first-live-trades) scales the position: the USD risk
    becomes sl_usd * risk_mult while the stop distance in price terms is unchanged. A result below
    the exchange min notional is rejected ("skip the trade")."""
    if risk_mult <= 0:
        return Sizing(False, "risk multiplier is 0 (derated to zero)")
    if side not in ("long", "short"):
        return Sizing(False, f"bad side {side!r}")
    if not (entry > 0 and math.isfinite(entry)):
        return Sizing(False, "bad entry price")
    d = 1.0 if side == "long" else -1.0
    target_notional = cfg.slice_margin_usd * cfg.leverage
    cap_qty = target_notional / entry

    if cfg.exit_mode == "fixed":
        qty = f.round_qty(cap_qty * risk_mult)
        if qty <= 0:
            return Sizing(False, "qty rounds to 0")
        sl_dist = cfg.sl_usd * risk_mult / qty
        tp_dist = cfg.tp_usd * risk_mult / qty
        used_atr = atr
    elif cfg.exit_mode == "atr":
        if atr is None or atr <= 0:
            return Sizing(False, "atr unavailable")
        sl_dist = cfg.sl_atr * atr
        tp_dist = cfg.tp_atr * atr
        qty = f.round_qty(min(cfg.sl_usd * risk_mult / sl_dist, cap_qty))
        if qty <= 0:
            return Sizing(False, "qty rounds to 0")
        used_atr = atr
    else:
        return Sizing(False, f"unknown exit_mode {cfg.exit_mode!r}")

    sl = f.round_price(entry - d * sl_dist)
    tp = f.round_price(entry + d * tp_dist)
    if sl <= 0:
        return Sizing(False, "stop price would be <= 0")
    if (entry - sl) * d <= 0 or (tp - entry) * d <= 0:
        return Sizing(False, "stop/target collapse onto entry after tick rounding")
    notional = qty * entry
    margin = notional / cfg.leverage
    liq = liq_price(side, entry, cfg.leverage, cfg.maint_margin_rate)
    if qty < f.min_qty:
        return Sizing(False, f"qty {qty} < min_qty {f.min_qty}")
    if notional < f.min_notional:
        return Sizing(False, f"notional {notional:.2f} < min_notional {f.min_notional}")
    if abs(entry - sl) > 0.8 * abs(entry - liq):
        return Sizing(False, "stop is too close to (or beyond) the liquidation price")
    if available is not None and margin * 1.01 > available:
        return Sizing(False, f"insufficient available margin ({available:.2f} < {margin:.2f})")
    return Sizing(True, "", qty, notional, margin, entry, sl, tp, qty * abs(entry - sl), qty * abs(tp - entry),
                  used_atr, liq)


def levels(cfg: Settings, side: str, entry: float, qty: float, atr: float | None,
           f: SymbolFilters, risk_mult: float = 1.0) -> tuple[float, float] | None:
    """(sl_price, tp_price) for an actual fill. With ``risk_mult`` < 1 the template's USD risk/reward
    shrink proportionally in fixed mode (the qty is already scaled), in atr mode distances are unchanged."""
    if qty <= 0 or entry <= 0:
        return None
    d = 1.0 if side == "long" else -1.0
    if cfg.exit_mode == "fixed":
        sl_dist, tp_dist = cfg.sl_usd * risk_mult / qty, cfg.tp_usd * risk_mult / qty
    else:
        if not atr or atr <= 0:
            return None
        sl_dist, tp_dist = cfg.sl_atr * atr, cfg.tp_atr * atr
    sl, tp = f.round_price(entry - d * sl_dist), f.round_price(entry + d * tp_dist)
    if sl <= 0 or (entry - sl) * d <= 0 or (tp - entry) * d <= 0:
        return None
    return sl, tp


def max_positions(equity: float, cfg: Settings) -> int:
    """Nominal slot count: floor(equity / slice margin), capped by the setting."""
    return max(0, min(cfg.max_positions, int(math.floor(equity / cfg.slice_margin_usd + 1e-9))))


def effective_slots(cfg: Settings, n_open: int, available: float) -> int:
    """Slots the account can really fill: open positions + whole slices that fit in the FREE margin
    including each entry's taker fee (100 USD -> 4, not 5: the 5th slice is short by the fees)."""
    per = cfg.slice_margin_usd + cfg.notional_usd * cfg.fee_taker
    return max(n_open, min(cfg.max_positions, n_open + int(math.floor(max(0.0, available) / per + 1e-9))))


def derated_below_min_notional(cfg: Settings, dd_pct: float, kill_dd_pct: float, min_notional: float) -> bool:
    """True when drawdown derating shrinks even a full slice below the exchange minimum notional, i.e. the
    bot can no longer open a valid trade yet the kill (dd >= limit) has not tripped: the guard HALTs."""
    return cfg.notional_usd * derate_multiplier(dd_pct, kill_dd_pct) < min_notional


# ----------------------------------------------------------------------------- trailing
def trail_update(side: str, entry: float, init_sl: float, cur_sl: float, best_price: float,
                 atr: float | None, active: bool, activate_r: float, distance_atr: float) -> tuple[bool, float]:
    """Return (trail_active, stop). The stop only ever tightens (never loosens)."""
    if not atr or atr <= 0:
        return active, cur_sl
    d = 1.0 if side == "long" else -1.0
    risk = abs(entry - init_sl)
    if risk <= 0:
        return active, cur_sl
    favourable = (best_price - entry) * d
    if not active and favourable < activate_r * risk:
        return False, cur_sl
    cand = best_price - d * distance_atr * atr
    new = max(cur_sl, cand) if side == "long" else min(cur_sl, cand)
    return True, new


# ----------------------------------------------------------------------------- kill switch
def drawdown_pct(equity: float, peak: float) -> float:
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - equity) / peak * 100.0)


def kill_tripped(equity: float, peak: float, limit_pct: float) -> bool:
    return drawdown_pct(equity, peak) >= limit_pct


# ----------------------------------------------------------------------------- correlation cap
def return_corr(a: pd.Series, b: pd.Series, bars: int = 288, min_overlap: int = 100) -> float | None:
    """Correlation of 5m returns over the last ``bars`` aligned bars (index = open_time)."""
    j = pd.concat([a.pct_change(), b.pct_change()], axis=1, join="inner").dropna().tail(bars)
    if len(j) < min_overlap:
        return None
    x, y = j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlation_gate(cand_side: str, cand_close: pd.Series,
                     open_positions: Sequence[tuple[str, str, pd.Series]], cap: CorrCap,
                     bars: int = 288) -> tuple[bool, list[tuple[str, float]]]:
    """False when >= cap.max_correlated same-direction positions have corr > cap.max_corr."""
    hot: list[tuple[str, float]] = []
    for sym, side, closes in open_positions:
        if side != cand_side:
            continue
        c = return_corr(cand_close, closes, bars)
        if c is not None and c > cap.max_corr:
            hot.append((sym, c))
    return len(hot) < cap.max_correlated, hot


# ----------------------------------------------------------------------------- stop invariant
def positions_without_stop(positions: Sequence[dict], orders: Sequence[ConditionalOrder]) -> list[str]:
    """Symbols of open positions that lack a live reduce-only SL on the closing side covering the qty."""
    bad = []
    for p in positions:
        want = "SELL" if p["side"] == "long" else "BUY"
        ok = any(o.symbol == p["symbol"] and o.kind == "SL" and o.side == want and o.reduce_only
                 and o.qty >= p["qty"] * 0.999 for o in orders)
        if not ok:
            bad.append(p["symbol"])
    return bad
