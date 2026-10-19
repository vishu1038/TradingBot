"""Profit protection maths shared by the engine, the shadow simulator and the backtester (pure functions, no I/O).

Why it exists (incident 2026-10-19): +32 % equity that was mostly OPEN profit on correlated longs was erased in 8 minutes.

* break-even: once the favourable move reaches ``breakeven_r`` x the initial risk |entry - initial SL| the stop moves to
  entry +- round-trip fees (never loosens);
* trailing: the existing ``tb.risk.trail_update`` (ratchet only), combined with the break-even stop by ``next_stop``;
* partial take-profit: at ``partial_tp_r`` x initial risk close ``partial_tp_frac`` of the quantity (once per position);
  the remainder's stop moves to at least break-even (``partial_level`` / ``partial_split``); skipped when the piece would gain
  less than ``min_tp_usd`` at its level (``partial_gain_ok``);
* losing-exit cooldown reasons (``LOSS_EXIT_REASONS``), shared by the live gate and the sims.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .risk import trail_update

MIN_STEP = 0.001                 # a stop is only moved when it improves by >= 0.1 % of the entry price
LOSS_EXIT_REASONS = ("sl", "trail", "kill", "timeout", "liquidation")


def is_loss_exit(reason: str, net_pnl: float) -> bool:
    """A losing exit that starts the per-coin ``sl_cooldown_bars`` (partial take-profits never do)."""
    return reason in LOSS_EXIT_REASONS and net_pnl <= 0.0


@dataclass(frozen=True)
class Protect:
    breakeven_r: float = 1.0          # 0 = off
    trail_enabled: bool = True
    trail_activate_r: float = 1.5
    trail_distance_atr: float = 1.0
    partial_tp_r: float = 1.5         # 0 = off
    partial_tp_frac: float = 0.5
    fee_rate: float = 0.0005          # taker fee per side (fraction)
    min_tp_usd: float = 0.0           # dollar floor: the partial piece must gain at least this at its level (0 = off)

    @classmethod
    def from_settings(cls, cfg: Any) -> "Protect":
        return cls(cfg.breakeven_r, cfg.trail.enabled, cfg.trail.activate_r, cfg.trail.distance_atr,
                   cfg.partial_tp_r, cfg.partial_tp_frac, cfg.fee_taker, cfg.min_tp_usd)

    @classmethod
    def from_sim(cls, cfg: Any) -> "Protect":
        return cls(cfg.breakeven_r, cfg.trail_enabled, cfg.trail_activate_r, cfg.trail_distance_atr,
                   cfg.partial_tp_r, cfg.partial_tp_frac, cfg.fee_taker, getattr(cfg, "min_tp_usd", 0.0))

    @property
    def partial_on(self) -> bool:
        return self.partial_tp_r > 0 and 0.0 < self.partial_tp_frac < 1.0


def breakeven_price(side: str, entry: float, fee_rate: float) -> float:
    """Entry +- round-trip fees (both legs at the taker rate): exiting there is ~flat after costs."""
    return entry * (1.0 + 2.0 * fee_rate) if side == "long" else entry * (1.0 - 2.0 * fee_rate)


def tighter(side: str, a: float, b: float) -> float:
    return max(a, b) if side == "long" else min(a, b)


def initial_risk(entry: float, init_sl: float) -> float:
    return abs(entry - init_sl)


def next_stop(side: str, entry: float, init_sl: float, cur_sl: float, best: float, atr: float | None,
              trail_active: bool, p: Protect) -> tuple[bool, float, str | None]:
    """(trail_active, stop, why): the stop after break-even and trailing, never looser than ``cur_sl``.
    ``why`` is None (unchanged), "breakeven" or "trail" (the rule that set the new level)."""
    risk = initial_risk(entry, init_sl)
    if risk <= 0:
        return trail_active, cur_sl, None
    d = 1.0 if side == "long" else -1.0
    stop, why, active = cur_sl, None, trail_active
    if p.breakeven_r > 0 and (best - entry) * d >= p.breakeven_r * risk - 1e-12:
        be = breakeven_price(side, entry, p.fee_rate)
        if (be - stop) * d > 0:
            stop, why = be, "breakeven"
    if p.trail_enabled and atr and atr > 0:
        act, cand = trail_update(side, entry, init_sl, stop, best, atr, active, p.trail_activate_r, p.trail_distance_atr)
        active = active or act
        if (cand - stop) * d > 0:
            stop, why = cand, "trail"
    return active, stop, why


def partial_level(side: str, entry: float, init_sl: float, tp: float | None, p: Protect) -> float | None:
    """Price at which the partial take-profit fires, or None when it is off / pointless: a TP at (or within 0.1 % of)
    that level closes the whole position anyway, so there is nothing to scale out of."""
    if not p.partial_on:
        return None
    risk = initial_risk(entry, init_sl)
    if risk <= 0:
        return None
    d = 1.0 if side == "long" else -1.0
    lvl = entry + d * p.partial_tp_r * risk
    if tp is not None and (tp - lvl) * d <= MIN_STEP * entry:
        return None
    return lvl


def partial_split(qty: float, price: float, frac: float, f: Any) -> tuple[float, float] | None:
    """(close_qty, remaining_qty) honouring the lot step; None when either part is below min qty / min notional
    (then the partial take-profit is skipped and the position keeps running whole)."""
    q = float(f.round_qty(qty * frac))
    rem = float(f.round_qty(float(Decimal(repr(float(qty))) - Decimal(repr(q)))))
    if q < f.min_qty or rem < f.min_qty or q <= 0 or rem <= 0:
        return None
    if q * price < f.min_notional or rem * price < f.min_notional:
        return None
    return q, rem


def partial_gain_ok(side: str, entry: float, level: float, close_qty: float, p: Protect) -> bool:
    """Dollar floor of the partial take-profit: the piece that would be closed must gain >= ``min_tp_usd`` at the partial level
    (gross price move x quantity); otherwise there is NO partial and the position runs to its full TP (user 2026-10-19)."""
    if p.min_tp_usd <= 0:
        return True
    d = 1.0 if side == "long" else -1.0
    return (level - entry) * d * close_qty >= p.min_tp_usd * (1.0 - 1e-6)


def stop_loss_usd(side: str, qty: float, entry: float, sl: float, fee: float, slip: float, stop_slip: float,
                  entry_fee: bool = False) -> float:
    """USD lost if the stop fills (same maths as ``tb.risk.sl_loss`` with the three rates spelled out)."""
    d = 1.0 if side == "long" else -1.0
    loss = (entry - sl) * d * qty + qty * sl * (fee + slip + stop_slip)
    if entry_fee:
        loss += qty * entry * fee
    return max(0.0, loss)
