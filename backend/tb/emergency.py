"""Emergency-path primitives shared by the guard and the engine (imports neither of them).

Incident 2026-10-19 (testnet): two trailed stops triggered but the venue's market fills failed, so the positions stayed
open with no stop; every later close was rejected with -2022 (reduce-only), one coin ran past its liquidation price, and the
guard's account-wide FLATTEN plus the engine's invariant loops hammered the venue every few seconds for hours. The rules
below are what both processes now follow:

* ``Backoff``: per key (symbol + action) exponential 5 s -> 10 s -> ... -> 5 min, reset on success. A failure is logged /
  alerted once per backoff step, never once per loop.
* ``liquidation_reason``: a position whose mark is through its liquidation price (or whose isolated margin is gone) is
  already in the venue's liquidation engine: no orders are sent for it, it is alerted once and re-checked each cycle; when it
  disappears the engine records the trade with ``exit_reason = "liquidation"``.
* ``emergency_close``: cancel the symbol's reduce-only conditional orders (they can make the exchange reject the closing
  order with -2022), close reduce-only, and if the venue still rejects with -2022 while it shows the position (and it is not
  liquidating) send ONE plain MARKET order of exactly the quantity it reports (one-way mode: the opposite side with that
  quantity can only close, never flip).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .exchange.base import ExchangeError, NetworkError, PositionInfo, RateLimitError

log = logging.getLogger("tb.emergency")

BACKOFF_BASE_S = 5.0
BACKOFF_MAX_S = 300.0
KEY_LIQ = "liquidating"          # kv: {symbol: {"since", "reason", "side", "qty", "entry", "liq", "mark"}}
REDUCE_REJECTED = -2022          # "ReduceOnly Order is rejected"


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
    method: str = ""                   # reduce_only | cancel_retry | plain_market
    result: Any = None                 # OrderResult of the closing order
    error: ExchangeError | None = field(default=None, repr=False)

    def __bool__(self) -> bool:
        return self.ok


def reduce_rejected(e: ExchangeError) -> bool:
    return e.code == REDUCE_REJECTED or "reduceonly" in str(e).lower().replace(" ", "")


def transient(e: BaseException) -> bool:
    return isinstance(e, (NetworkError, RateLimitError))


def _find(ex: Any, symbol: str) -> PositionInfo | None:
    return next((p for p in ex.positions() if p.symbol == symbol), None)


def cancel_reduce_only(ex: Any, symbol: str, say: Callable[[str], None] | None = None) -> int:
    """Cancel every reduce-only conditional order of ``symbol`` (legacy and algo ids alike: ``cancel_order`` routes on the
    id prefix). Best effort: a failure is reported, never raised. Returns how many were cancelled."""
    n = 0
    try:
        orders = ex.open_conditional_orders(symbol)
    except ExchangeError as e:
        if say:
            say(f"cannot list the conditional orders of {symbol}: {e}")
        return 0
    for o in orders:
        if o.symbol != symbol or not o.reduce_only:
            continue
        try:
            ex.cancel_order(symbol, o.order_id)
            n += 1
        except ExchangeError as e:
            if say:
                say(f"cannot cancel {o.kind} {o.order_id} on {symbol}: {e}")
    return n


def emergency_close(ex: Any, symbol: str, *, cancel_first: bool = True,
                    say: Callable[[str], None] | None = None) -> CloseOutcome:
    """Close ``symbol`` on the exchange, robust against the -2022 rejection. Never raises ``ExchangeError``.

    1. re-read the position (the exchange is the truth): flat -> done; liquidating -> no order is sent;
    2. cancel its reduce-only conditional orders (``cancel_first``; otherwise only if the plain close is rejected);
    3. reduce-only market close of the exchange-reported quantity;
    4. on -2022: re-read; still open and not liquidating -> ONE non-reduce-only MARKET of exactly the reported quantity;
    5. verify (an order the venue accepted but never filled does not count as closed).
    """
    say = say or (lambda m: log.warning("%s", m))
    try:
        pos = _find(ex, symbol)
    except ExchangeError as e:
        return CloseOutcome(False, f"cannot read the position: {e}", error=e)
    if pos is None:
        return CloseOutcome(True, "already flat", flat=True)
    why = liquidation_reason(pos)
    if why:
        return CloseOutcome(False, f"in liquidation ({why}); no order sent", liquidation=True)
    side = "SELL" if pos.side == "long" else "BUY"
    cancelled = False
    if cancel_first:
        cancelled = True
        cancel_reduce_only(ex, symbol, say)
    res = None
    method = "reduce_only"
    err: ExchangeError | None = None
    try:
        res = ex.market_order(symbol, side, pos.qty, reduce_only=True)
    except ExchangeError as e:
        err = e
        if not reduce_rejected(e):
            return CloseOutcome(False, f"close rejected: {e}", error=e)
    if err is not None and not cancelled:
        # the exchange can reject a reduce-only close while reduce-only stops already cover the whole position
        cancelled = True
        cancel_reduce_only(ex, symbol, say)
        try:
            res = ex.market_order(symbol, side, pos.qty, reduce_only=True)
            err, method = None, "cancel_retry"
        except ExchangeError as e:
            err = e
            if not reduce_rejected(e):
                return CloseOutcome(False, f"close rejected: {e}", error=e)
    if err is not None:
        try:
            pos2 = _find(ex, symbol)               # fresh read immediately before the non-reduce-only order
        except ExchangeError as e:
            return CloseOutcome(False, f"close rejected ({err}); cannot re-read the position: {e}", error=e)
        if pos2 is None:
            return CloseOutcome(True, f"flat after -2022 ({err})", flat=True)
        why = liquidation_reason(pos2)
        if why:
            return CloseOutcome(False, f"-2022 and in liquidation ({why}); no order sent", liquidation=True, error=err)
        side2 = "SELL" if pos2.side == "long" else "BUY"
        say(f"{symbol}: reduce-only close rejected ({err}) while the exchange shows {pos2.side} {pos2.qty}; "
            f"sending one plain MARKET {side2} {pos2.qty}")
        try:
            res = ex.market_order(symbol, side2, pos2.qty, reduce_only=False)
            method, pos = "plain_market", pos2
        except ExchangeError as e:
            return CloseOutcome(False, f"plain market close failed: {e}", error=e)
    # verify: the exchange must show the position gone
    try:
        left = _find(ex, symbol)
    except ExchangeError as e:
        return CloseOutcome(True, f"closed ({method}); verification unavailable: {e}", method=method, result=res)
    if left is None:
        try:
            ex.cancel_all(symbol)
        except ExchangeError as e:
            say(f"{symbol}: closed but leftover orders could not be cancelled: {e}")
        return CloseOutcome(True, f"closed ({method})", flat=True, method=method, result=res)
    if left.side != pos.side:                      # (a plain market order that raced with a stop: flipped) undo at once
        say(f"{symbol}: the plain close flipped the position to {left.side} {left.qty}; closing the remainder reduce-only")
        try:
            res = ex.market_order(symbol, "SELL" if left.side == "long" else "BUY", left.qty, reduce_only=True)
        except ExchangeError as e:
            return CloseOutcome(False, f"flipped position could not be closed: {e}", method=method, error=e)
        return CloseOutcome(True, f"closed ({method}) after undoing a flip", method=method, result=res)
    return CloseOutcome(False, f"order accepted but {left.qty} still open ({method})", method=method, result=res)
