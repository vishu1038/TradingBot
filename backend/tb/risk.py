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
def liq_price(side: str, entry: float, leverage: float, mmr: float = 0.005, cum_ratio: float = 0.0) -> float:
    """Isolated liquidation price, Binance form (identical to ``tb.sim.template.liq_price`` when cum == 0):
    long entry*(1-1/L-cum/notional)/(1-mmr), short entry*(1+1/L+cum/notional)/(1+mmr)
    (margin = notional/L, no fees; ``cum_ratio`` = maintenance amount / notional of the bracket)."""
    if side == "long":
        return entry * (1.0 - 1.0 / leverage - cum_ratio) / (1.0 - mmr)
    return entry * (1.0 + 1.0 / leverage + cum_ratio) / (1.0 + mmr)


# ----------------------------------------------------------------------------- brackets / maintenance margin
@dataclass(frozen=True)
class Bracket:
    """One Binance leverage-bracket tier: maintenance margin = notional * mmr - cum for floor <= notional < cap;
    ``max_lev`` is the tier's maximum leverage (initialLeverage)."""
    floor: float
    cap: float
    mmr: float
    cum: float
    max_lev: int = 125


def parse_brackets(raw) -> dict:
    """``GET /fapi/v1/leverageBracket`` (a list, or one dict when ``symbol`` was given) -> {symbol: [Bracket]}."""
    rows = raw if isinstance(raw, list) else [raw]
    out: dict = {}
    for r in rows:
        tiers = sorted((Bracket(float(b.get("notionalFloor", 0)), float(b.get("notionalCap", float("inf"))),
                                float(b["maintMarginRatio"]), float(b.get("cum", 0)), int(b.get("initialLeverage", 125)))
                        for b in r.get("brackets", [])), key=lambda t: t.floor)
        if tiers:
            out[r["symbol"]] = tiers
    return out


def tier_for(notional: float, brackets) -> Bracket | None:
    if not brackets:
        return None
    for t in brackets:
        if t.floor <= notional < t.cap:
            return t
    return brackets[-1]


def maintenance_margin(notional: float, brackets=None, default_mmr: float = 0.01) -> float:
    """Maintenance margin of a position of ``notional`` USD from its bracket tier (``default_mmr`` if unknown)."""
    t = tier_for(notional, brackets)
    if t is None:
        return notional * default_mmr
    return max(0.0, notional * t.mmr - t.cum)


LIQ_STOP_RATIO = 1.5       # liquidation must be at least 1.5x the stop distance away from entry


def choose_leverage(side: str, entry: float, sl: float, notional: float, max_leverage: int = 20, brackets=None,
                    default_mmr: float = 0.01, ratio: float = LIQ_STOP_RATIO) -> int | None:
    """Highest integer leverage L (<= max_leverage and <= the bracket's maxLeverage at this notional) whose isolated
    liquidation price is at least ``ratio`` x |entry - SL| away from entry. None if even 1x does not satisfy it.
    The same function is used by the engine, the paper exchange and the backtester."""
    stop_dist = abs(entry - sl)
    t = tier_for(notional, brackets)
    mmr = t.mmr if t else default_mmr
    cum_ratio = (t.cum / notional) if (t and notional > 0) else 0.0
    cap = int(max_leverage) if not t else min(int(max_leverage), int(t.max_lev))
    for lev in range(max(1, cap), 0, -1):
        if abs(entry - liq_price(side, entry, lev, mmr, cum_ratio)) >= ratio * stop_dist:
            return lev
    return None


FLOOR_REASON = "target below $ floor"


def dollar_floor(cfg: Settings, risk_usd: float, reward_usd: float | None) -> str | None:
    """None when the trade clears the dollar floor, else a human reason. ``risk_usd`` / ``reward_usd`` are the USD moves to the
    stop-loss / take-profit for the ACTUAL quantity (after derating, caps and tick rounding); ``reward_usd`` None = the position has
    no take-profit (stop only), so only the stop is checked. A trade is skipped, never shrunk below the floor (user 2026-10-19)."""
    return dollar_floor_usd(cfg.min_tp_usd, cfg.min_sl_usd, risk_usd, reward_usd)


def dollar_floor_usd(min_tp_usd: float, min_sl_usd: float, risk_usd: float, reward_usd: float | None) -> str | None:
    """``dollar_floor`` with the two floors spelled out (the backtester has no ``Settings``)."""
    if min_sl_usd > 0 and risk_usd < min_sl_usd * (1.0 - 1e-6):
        return f"{FLOOR_REASON}: SL ${risk_usd:.2f} < ${min_sl_usd:g}"
    if reward_usd is not None and min_tp_usd > 0 and reward_usd < min_tp_usd * (1.0 - 1e-6):
        return f"{FLOOR_REASON}: TP ${reward_usd:.2f} < ${min_tp_usd:g}"
    return None


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
    leverage: int = 1
    mm: float = 0.0


def size_trade(cfg: Settings, side: str, entry: float, atr: float | None, f: SymbolFilters,
               available: float | None = None, risk_mult: float = 1.0, brackets=None, tp_floor: bool = True) -> Sizing:
    """Qty + SL/TP prices for the trade template. ``side`` is 'long' | 'short'.

    ``risk_mult`` (drawdown derating / drift / first-live-trades) scales the position: the USD risk
    becomes sl_usd * risk_mult while the stop distance in price terms is unchanged. A result below
    the exchange min notional is rejected ("skip the trade"). Dollar floor: after ALL sizing (derating, caps, lot rounding) the
    USD move to the stop / take-profit must be >= ``min_sl_usd`` / ``min_tp_usd``, else the trade is skipped, never shrunk below
    it (``tp_floor=False`` for a stop-only position, e.g. a cross-sectional leg: only the stop is checked)."""
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
    if qty < f.min_qty:
        return Sizing(False, f"qty {qty} < min_qty {f.min_qty}")
    if notional < f.min_notional:
        return Sizing(False, f"notional {notional:.2f} < min_notional {f.min_notional}")
    # per-trade leverage: the highest L whose liquidation is >= 1.5x the stop distance away (isolated margin)
    lev = choose_leverage(side, entry, sl, notional, cfg.max_leverage, brackets, cfg.maint_margin_rate)
    if lev is None:
        return Sizing(False, "stop is too close to (or beyond) the liquidation price even at 1x")
    margin = notional / lev
    t = tier_for(notional, brackets)
    liq = liq_price(side, entry, lev, t.mmr if t else cfg.maint_margin_rate, (t.cum / notional) if t else 0.0)
    if available is not None and margin * 1.01 > available:
        return Sizing(False, f"insufficient available margin ({available:.2f} < {margin:.2f})")
    why = dollar_floor(cfg, qty * abs(entry - sl), qty * abs(tp - entry) if tp_floor else None)
    if why:
        return Sizing(False, why)
    return Sizing(True, "", qty, notional, margin, entry, sl, tp, qty * abs(entry - sl), qty * abs(tp - entry),
                  used_atr, liq, lev, maintenance_margin(notional, brackets, cfg.maint_margin_rate))


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
    """Nominal slot count: floor(equity / slice margin); capped by ``max_positions`` when that is set."""
    n = int(math.floor(equity / cfg.slice_margin_usd + 1e-9))
    return max(0, min(cfg.max_positions, n) if cfg.max_positions is not None else n)


# ----------------------------------------------------------------------------- portfolio rules for a new entry
def sl_loss(cfg: Settings, qty: float, entry: float, sl: float, side: str, *, entry_fee: bool = False) -> float:
    """USD lost if the stop fills: price loss + exit fee + stop slippage (+ the entry fee if still to be paid).
    Never negative (a trailed stop in profit risks nothing)."""
    d = 1.0 if side == "long" else -1.0
    price_loss = (entry - sl) * d * qty
    costs = qty * sl * (cfg.fee_taker + cfg.slippage_pct + cfg.stop_slippage_pct)
    if entry_fee:
        costs += qty * entry * cfg.fee_taker
    return max(0.0, price_loss + costs)


@dataclass
class Book:
    """Aggregates of the open positions + account state used by ``check_entry`` (all in USD)."""
    equity: float
    available: float
    open_notional: float = 0.0
    open_mm: float = 0.0
    open_risk: float = 0.0
    open_risk_long: float = 0.0        # share of ``open_risk`` held by long positions
    open_risk_short: float = 0.0       # ... and by shorts (side_risk rule)
    open_net: float = 0.0              # signed beta-weighted notional: sum(+1 long / -1 short x notional x beta vs BTC) (net_exposure rule)


@dataclass
class EntryCheck:
    ok: bool
    rule: str = ""          # total_leverage | margin_ratio | open_risk | side_risk | net_exposure | free_margin | "" (ok)
    detail: str = ""


def net_exposure_check(net_before: float, new_signed: float, cap_usd: float) -> tuple[bool, str]:
    """Beta-weighted net exposure cap. ``net_before`` = signed beta-weighted notional of the open book, ``new_signed`` = that of the
    entry (+ long / - short). Rejected only when |net after| > cap AND |net after| > |net before| AND |net after| > |new|:
    an entry that REDUCES |net| is always allowed, and a lone position into a flat book is never blocked by the cap alone
    (otherwise a $160 slice on a $100 stake could never trade at the 150 % default). cap_usd <= 0 = off."""
    if cap_usd <= 0:
        return True, ""
    after = net_before + new_signed
    if abs(after) <= cap_usd + 1e-9 or abs(after) <= abs(net_before) + 1e-9 or abs(after) <= abs(new_signed) + 1e-9:
        return True, ""
    return False, (f"net exposure ${after:+.0f} (beta-weighted) would exceed the cap ${cap_usd:.0f} "
                   f"(book ${net_before:+.0f}, entry ${new_signed:+.0f})")


def check_entry(cfg: Settings, book: Book, new_notional: float, new_mm: float, new_risk: float, margin: float,
                fee: float, side: str | None = None, beta: float = 1.0) -> EntryCheck:
    """A new position is allowed only if, INCLUDING it:
    (1) sum(notional)/equity <= max_total_leverage, (2) sum(maintenance margin)/equity <= max_margin_ratio,
    (3) sum(loss if every SL fills)/equity <= max_open_risk_pct %, (4) free margin >= its margin + entry fee,
    (5) with ``side``: the stop-out risk of ONE side (longs or shorts) <= max_side_risk_pct % of the open-risk BUDGET
    (max_open_risk_pct % x equity), so one correlated direction cannot use the whole budget,
    (6) with ``side``: the beta-weighted NET exposure |sum(side x notional x beta)| <= max_net_exposure_pct % x equity
    (``net_exposure_check``; entries that reduce |net| are always allowed)."""
    eq = book.equity
    if eq <= 0:
        return EntryCheck(False, "total_leverage", "no equity")
    lev = (book.open_notional + new_notional) / eq
    if lev > cfg.max_total_leverage + 1e-9:
        return EntryCheck(False, "total_leverage", f"total leverage {lev:.2f}x > max {cfg.max_total_leverage:g}x")
    ratio = (book.open_mm + new_mm) / eq
    if ratio > cfg.max_margin_ratio + 1e-9:
        return EntryCheck(False, "margin_ratio", f"maintenance margin ratio {ratio:.2f} > max {cfg.max_margin_ratio:g}")
    risk = (book.open_risk + new_risk) / eq * 100.0
    if risk > cfg.max_open_risk_pct + 1e-9:
        return EntryCheck(False, "open_risk", f"open stop-out risk {risk:.1f}% > max {cfg.max_open_risk_pct:g}%")
    if side in ("long", "short"):
        budget = cfg.max_open_risk_pct / 100.0 * eq
        cap = cfg.max_side_risk_pct / 100.0 * budget
        side_risk = (book.open_risk_long if side == "long" else book.open_risk_short) + new_risk
        if side_risk > cap + 1e-9:
            return EntryCheck(False, "side_risk",
                              f"{side} stop-out risk ${side_risk:.2f} > {cfg.max_side_risk_pct:g}% of the open-risk budget "
                              f"${budget:.2f} (max ${cap:.2f})")
    if side in ("long", "short") and cfg.max_net_exposure_pct > 0:
        signed = (1.0 if side == "long" else -1.0) * new_notional * beta
        ok, why = net_exposure_check(book.open_net, signed, cfg.max_net_exposure_pct / 100.0 * eq)
        if not ok:
            return EntryCheck(False, "net_exposure", why)
    if book.available + 1e-9 < margin + fee:
        return EntryCheck(False, "free_margin", f"free margin {book.available:.2f} < slice margin + fee {margin + fee:.2f}")
    return EntryCheck(True)


def slots_fit(cfg: Settings, book: Book, n_open: int, brackets=None) -> int:
    """How many MORE template slices (slice notional, fixed-mode stop, auto leverage) ``check_entry`` would accept
    right now (also honouring the optional ``max_positions`` cap)."""
    notional = cfg.notional_usd
    entry = 100.0
    sl = entry * (1.0 - cfg.sl_usd / notional)
    lev = choose_leverage("long", entry, sl, notional, cfg.max_leverage, brackets, cfg.maint_margin_rate) or 1
    margin = notional / lev
    fee = notional * cfg.fee_taker
    mm = maintenance_margin(notional, brackets, cfg.maint_margin_rate)
    risk = sl_loss(cfg, notional / entry, entry, sl, "long", entry_fee=True)
    best = 0
    for side in ("long", "short"):          # a slice fits on whichever side still has room
        n = 0
        b = Book(book.equity, book.available, book.open_notional, book.open_mm, book.open_risk,
                 book.open_risk_long, book.open_risk_short, book.open_net)
        while n < 100:
            if cfg.max_positions is not None and n_open + n >= cfg.max_positions:
                break
            if not check_entry(cfg, b, notional, mm, risk, margin, fee, side).ok:
                break
            n += 1
            b.open_notional += notional
            b.open_mm += mm
            b.open_risk += risk
            b.open_net += notional if side == "long" else -notional
            if side == "long":
                b.open_risk_long += risk
            else:
                b.open_risk_short += risk
            b.available -= margin + fee
        best = max(best, n)
    return best


def effective_slots(cfg: Settings, n_open: int, available: float) -> int:
    """Free-margin-only slot count (kept for callers without equity): open + whole slices (margin + entry fee)
    that fit in ``available``, capped by ``max_positions`` when set."""
    per = cfg.slice_margin_usd + cfg.notional_usd * cfg.fee_taker
    n = n_open + int(math.floor(max(0.0, available) / per + 1e-9))
    if cfg.max_positions is not None:
        n = min(cfg.max_positions, n)
    return max(n_open, n)


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


def dd_reference(start: float, peak: float, basis: str = "start") -> float:
    """Equity the drawdown is measured from: the starting stake (default) or the high-water mark.

    "start" = loss of the user's money (kill at -25 % means 25 % of the stake is gone); "peak" also
    counts giving back open/realised profit. Falls back to the peak when no start is known."""
    if basis == "start" and start and start > 0:
        return start
    return peak


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
    """Symbols of open positions that lack a live SL on the closing side covering the qty (a closePosition stop always does)."""
    bad = []
    for p in positions:
        want = "SELL" if p["side"] == "long" else "BUY"
        ok = any(o.symbol == p["symbol"] and o.kind == "SL" and o.side == want and o.covers(p["qty"]) for o in orders)
        if not ok:
            bad.append(p["symbol"])
    return bad
