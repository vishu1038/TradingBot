"""Emergency-path primitives shared by the guard and the engine (imports neither of them).

Incident 2026-10-19 (testnet): two trailed stops triggered but the venue's market fills failed, so the positions stayed
open with no stop, one coin ran past its liquidation price, and the guard's account-wide FLATTEN plus the engine's invariant
loops hammered the venue every few seconds for hours. The rules below are what both processes now follow:

* ``Backoff``: per key (symbol + action) exponential 5 s -> 10 s -> ... -> 5 min, reset on success. A failure is logged /
  alerted once per backoff step, never once per loop.
* ``liquidation_reason``: a position whose mark is through its liquidation price (or whose isolated margin is gone) is
  already in the venue's liquidation engine: no orders are sent for it, it is alerted once and re-checked each cycle; when it
  disappears the engine records the trade with ``exit_reason = "liquidation"``.
* ``close_flat`` (alias ``emergency_close``): NO reduce-only orders anywhere (user rule 2026-10-20). Under the symbol's lock:
  re-read the position (the exchange is the truth; flat -> done; liquidating -> nothing is sent), optionally cancel its
  TP/SL first, send a NORMAL market order for exactly the reported quantity in the closing direction, then verify. If the
  position came back as the OPPOSITE side (a stop filled between the read and the order: the normal order opened a reverse
  position) it is flattened the same way and the outcome carries ``flipped=True`` so the caller can alert.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .exchange.base import ExchangeError, NetworkError, PositionInfo, RateLimitError

log = logging.getLogger("tb.emergency")

BACKOFF_BASE_S = 5.0
BACKOFF_MAX_S = 300.0
KEY_LIQ = "liquidating"          # kv: {symbol: {"since", "reason", "side", "qty", "entry", "liq", "mark"}}


# ----------------------------------------------------------------------------- backoff
class Backoff:
    """Per-key exponential backoff. ``fail`` returns the delay until the next attempt (5, 10, 20 ... capped);
    ``ready`` says whether that delay has passed; ``reset`` forgets the key after a success."""

    def __init__(self, base: float = BACKOFF_BASE_S, cap: float = BACKOFF_MAX_S):
        self.base = base
        self.cap = cap
        self._fails: dict[Any, int] = {}
        self._next: dict[Any, float] = {}

    def ready(self, key: Any, now: float) -> bool:
        return now >= self._next.get(key, 0.0)

    def fail(self, key: Any, now: float) -> float:
        n = self._fails.get(key, 0)
        delay = min(self.cap, self.base * (2 ** n))
        self._fails[key] = n + 1
        self._next[key] = now + delay
        return delay

    def reset(self, key: Any) -> bool:
        had = key in self._fails
        self._fails.pop(key, None)
        self._next.pop(key, None)
        return had

    def failures(self, key: Any) -> int:
        return self._fails.get(key, 0)

    def wait_left(self, key: Any, now: float) -> float:
        return max(0.0, self._next.get(key, 0.0) - now)

    def keys(self) -> list[Any]:
        return list(self._fails)


# ----------------------------------------------------------------------------- liquidation
def liquidation_reason(p: PositionInfo, mark: float | None = None) -> str | None:
    """Why ``p`` counts as "in liquidation" (None = it does not): mark through the liquidation price (long: mark <= liq,
    short: mark >= liq) or an isolated position whose margin is exhausted (isolatedMargin <= 0)."""
    m = p.mark_price if p.mark_price and p.mark_price > 0 else (mark or 0.0)
    if p.liq_price and p.liq_price > 0 and m > 0:
        if p.side == "long" and m <= p.liq_price:
            return f"mark {m:g} <= liquidation price {p.liq_price:g}"
        if p.side == "short" and m >= p.liq_price:
            return f"mark {m:g} >= liquidation price {p.liq_price:g}"
    if p.isolated and p.qty > 0 and p.margin <= 0:
        return f"isolated margin {p.margin:g} <= 0"
    return None


def liq_map(store: Any) -> dict[str, dict]:
    got = store.kv_get(KEY_LIQ, {})
    return dict(got) if isinstance(got, dict) else {}


def liq_mark(store: Any, p: PositionInfo, why: str, now: float, alerts: Any = None, mode: str = "") -> bool:
    """Record ``p`` as liquidating (shared by guard and engine through the kv). Returns True the FIRST time (the caller
    alerts / logs); later calls only refresh the snapshot."""
    cur = liq_map(store)
    new = p.symbol not in cur
    since = cur[p.symbol]["since"] if not new else now
    cur[p.symbol] = {"since": since, "reason": why, "side": p.side, "qty": p.qty, "entry": p.entry_price,
                     "liq": p.liq_price, "mark": p.mark_price, "margin": p.margin}
    store.kv_set(KEY_LIQ, cur)
    if new and alerts is not None:
        try:
            alerts.liquidation(p.symbol, p.side, p.qty, p.entry_price, p.mark_price, p.liq_price, why, mode)
        except Exception:
            log.exception("liquidation alert failed")
    return new


def liq_clear(store: Any, symbol: str) -> dict | None:
    cur = liq_map(store)
    gone = cur.pop(symbol, None)
    if gone is not None:
        store.kv_set(KEY_LIQ, cur)
    return gone


# ----------------------------------------------------------------------------- emergency close
@dataclass
class CloseOutcome:
    ok: bool
    detail: str = ""
    flat: bool = False                 # the exchange showed no position (already closed / closed now)
    liquidation: bool = False          # the venue is liquidating it: nothing was sent
    method: str = ""                   # market | market+flip_undone
    flipped: bool = False              # a reverse position appeared (race with a stop) and was flattened: callers alert
    result: Any = None                 # OrderResult of the closing order
    error: ExchangeError | None = field(default=None, repr=False)

    def __bool__(self) -> bool:
        return self.ok


def transient(e: BaseException) -> bool:
    return isinstance(e, (NetworkError, RateLimitError))


def _find(ex: Any, symbol: str) -> PositionInfo | None:
    return next((p for p in ex.positions() if p.symbol == symbol), None)


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def symbol_lock(symbol: str) -> threading.Lock:
    """One lock per symbol: two close paths of this process (engine invariant, kill, time stop, mirror ...) cannot interleave their
    read -> order -> verify sequences. (Another process, the guard, is covered by the post-order verification instead.)"""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(symbol, threading.Lock())


def cancel_conditionals(ex: Any, symbol: str, say: Callable[[str], None] | None = None) -> int:
    """Cancel every conditional (TP / SL) order of ``symbol`` (legacy and algo ids alike: ``cancel_order`` routes on the id
    prefix). Best effort: a failure is reported, never raised. Returns how many were cancelled."""
    n = 0
    try:
        orders = ex.open_conditional_orders(symbol)
    except ExchangeError as e:
        if say:
            say(f"cannot list the conditional orders of {symbol}: {e}")
        return 0
    for o in orders:
        if o.symbol != symbol:
            continue
        try:
            ex.cancel_order(symbol, o.order_id)
            n += 1
        except ExchangeError as e:
            if say:
                say(f"cannot cancel {o.kind} {o.order_id} on {symbol}: {e}")
    return n


def close_flat(ex: Any, symbol: str, *, cancel_first: bool = False,
               say: Callable[[str], None] | None = None) -> CloseOutcome:
    """Close ``symbol`` on the exchange with a NORMAL market order. Never raises ``ExchangeError``.

    Under the symbol lock: 1. re-read the position (flat -> done; liquidating -> no order is sent); 2. cancel its TP/SL first
    (``cancel_first``: removes the race with a stop that fills at the same moment); 3. market order for exactly the reported
    quantity in the closing direction; 4. verify: the exchange must show it gone. A reverse position (opposite side) is
    flattened the same way and reported through ``flipped``; an order that was accepted but left the position open is a failure.
    """
    say = say or (lambda m: log.warning("%s", m))
    with symbol_lock(symbol):
        try:
            pos = _find(ex, symbol)
        except ExchangeError as e:
            return CloseOutcome(False, f"cannot read the position: {e}", error=e)
        if pos is None:
            return CloseOutcome(True, "already flat", flat=True)
        why = liquidation_reason(pos)
        if why:
            return CloseOutcome(False, f"in liquidation ({why}); no order sent", liquidation=True)
        if cancel_first:
            cancel_conditionals(ex, symbol, say)
        side = "SELL" if pos.side == "long" else "BUY"
        try:
            res = ex.market_order(symbol, side, pos.qty)
        except ExchangeError as e:
            return CloseOutcome(False, f"close rejected: {e}", error=e)
        method = "market"
        flipped = False
        try:
            left = _find(ex, symbol)
        except ExchangeError as e:
            return CloseOutcome(True, f"closed ({method}); verification unavailable: {e}", method=method, result=res)
        if left is not None and left.side != pos.side:       # the stop filled first: our normal order opened a reverse position
            flipped = True
            say(f"{symbol}: the close opened a reverse {left.side} {left.qty} (a stop filled in between); flattening it")
            try:
                res = ex.market_order(symbol, "SELL" if left.side == "long" else "BUY", left.qty)
            except ExchangeError as e:
                return CloseOutcome(False, f"reverse position {left.side} {left.qty} could not be flattened: {e}", method=method,
                                    flipped=True, error=e)
            method = "market+flip_undone"
            try:
                left = _find(ex, symbol)
            except ExchangeError as e:
                return CloseOutcome(True, f"closed ({method}); verification unavailable: {e}", method=method, flipped=True,
                                    result=res)
        if left is None:
            try:
                ex.cancel_all(symbol)                         # leftover TP / SL of the closed position
            except ExchangeError as e:
                say(f"{symbol}: closed but leftover orders could not be cancelled: {e}")
            return CloseOutcome(True, f"closed ({method})", flat=True, method=method, flipped=flipped, result=res)
        return CloseOutcome(False, f"order accepted but {left.qty} {left.side} still open ({method})", method=method,
                            flipped=flipped, result=res)


def emergency_close(ex: Any, symbol: str, *, cancel_first: bool = True,
                    say: Callable[[str], None] | None = None) -> CloseOutcome:
    """Guard / engine emergency path: ``close_flat`` with the TP/SL cancelled first by default."""
    return close_flat(ex, symbol, cancel_first=cancel_first, say=say)
