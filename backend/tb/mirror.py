"""Testnet mirror: a one-way, best-effort follower that replays every order action of the PAPER book on the Binance futures
testnet at MINIMUM size, purely to exercise the real order path (precision, filters, algo endpoint, closePosition TP/SL, fills,
latency) while the paper book stays the master and the only scoreboard.

Hard rules (docs/ARCHITECTURE.md "Testnet mirror"):

* The paper book never waits for, depends on or is changed by the mirror. The engine only calls the ``on_*`` hooks below; each
  hook copies what it needs, appends to an in-memory queue and returns (no I/O, never raises). All venue I/O runs on a
  dedicated thread pool inside the mirror's own per-symbol asyncio tasks, every call under a timeout, every exception caught.
* Mirror errors never touch the paper safety state, the kill switch, the guard (FLATTEN / HALT) or the drawdown. The mirror
  has its own per-symbol ``emergency.Backoff``.
* The mirror manages ONLY the venue positions it opened (kv ``mirror_map``: paper position id <-> venue symbol / qty / order
  ids). Symbols not TRADING on the venue and venue positions it does not own are skipped, never touched.
* Every order-path venue call is a row in ``mirror_events`` (reads only when they fail), classified ``ok`` / ``our_bug`` /
  ``venue_fault`` / ``account_unusable`` / ``skipped`` by ``classify`` (table: ``CLASS_BY_CODE``).
* An unusable testnet account (margin balance <= 0, canTrade false, available balance below a min-size entry) switches the
  mirror to ``account_unusable``: nothing is sent, ``skipped`` rows are recorded, ONE alert goes out, and it resumes by itself.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from functools import partial, wraps
from typing import Any, Callable

from .emergency import Backoff, close_flat
from .exchange.base import (ExchangeError, NetworkError, OrderNotFilled, RateLimitError, SymbolFilters, opposite,
                            order_side)

log = logging.getLogger("tb.mirror")

KEY_MAP = "mirror_map"
KEY_STATE = "mirror_state"

OK, OUR_BUG, VENUE_FAULT, ACCOUNT_UNUSABLE, SKIPPED = "ok", "our_bug", "venue_fault", "account_unusable", "skipped"
CLASSES = (OK, OUR_BUG, VENUE_FAULT, ACCOUNT_UNUSABLE, SKIPPED)

NOTIONAL_MARGIN = 1.10        # min-size entry: qty * price >= min_notional * 1.10 (a market order is checked against the mark)
CALL_TIMEOUT_S = 15.0         # per venue call (the HTTP client itself times out at 10 s)
RECONCILE_S = 60.0
ACCOUNT_POLL_S = 30.0
ACCOUNT_MAX_AGE_S = 60.0
ENTERING_GRACE_S = 120.0      # an "entering" mapping this young belongs to a task that is still running
MAX_TRIES = 6                 # deferred retries of one queued action (close / entry); the reconciler is the safety net
MAX_QUEUE = 500
ALERT_EVERY_S = 3600.0        # our_bug alerts: one per (code, symbol) per hour
STOP_FAIL_CLOSE = 3           # a mirror position whose stop could not be placed this many times is closed
KEEP_DAYS = 30
KEY_RETRY = "mirror_retry"    # kv: unix time of the last user "retry mirror" (POST /api/mirror/retry)
REJECT_STREAK = 3             # account-class order rejects in a row ...
REJECT_SYMBOLS = 2            # ... on at least this many distinct symbols => account_unusable (balance looks fine, orders fail)
REJECT_CODES = (-2019, -2018, -2022)
REJECT_ACTIONS = ("entry", "entry_limit", "stop", "take_profit")

# ----------------------------------------------------------------------------- classification
# code -> class. A code that is not listed falls back to: HTTP >= 500 / network -> venue_fault, anything else the venue
# rejected -> our_bug (a request the venue refused is our problem until proven otherwise). -2022 is context dependent
# (see ``classify``).
CLASS_BY_CODE: dict[int, str] = {
    # ---- our_bug: request / validation problems on our side
    -1013: OUR_BUG,    # filter failure (lot size, price filter, min notional, percent price ...)
    -1021: OUR_BUG,    # timestamp outside recvWindow (our clock; the client re-syncs once)
    -1022: OUR_BUG,    # invalid signature
    -1100: OUR_BUG, -1101: OUR_BUG,   # illegal characters / too many parameters
    -1102: OUR_BUG,    # mandatory parameter missing or malformed
    -1103: OUR_BUG, -1104: OUR_BUG, -1105: OUR_BUG,
    -1106: OUR_BUG,    # parameter sent when not required
    -1111: OUR_BUG,    # precision over the maximum for this asset
    -1112: OUR_BUG, -1114: OUR_BUG, -1115: OUR_BUG, -1116: OUR_BUG, -1117: OUR_BUG, -1118: OUR_BUG, -1119: OUR_BUG,
    -1120: OUR_BUG, -1121: OUR_BUG,   # invalid symbol
    -1125: OUR_BUG, -1127: OUR_BUG, -1128: OUR_BUG, -1130: OUR_BUG,
    -2011: OUR_BUG,    # unknown order sent (cancel of an id we think exists)
    -2013: OUR_BUG,    # order does not exist
    -2021: OUR_BUG,    # stop would trigger immediately (we computed a level through the mark)
    -2027: OUR_BUG,    # exceeded the maximum position at this leverage
    -4003: OUR_BUG, -4004: OUR_BUG, -4005: OUR_BUG, -4006: OUR_BUG,   # quantity <= 0 / above max / min
    -4014: OUR_BUG,    # price not a multiple of the tick size
    -4015: OUR_BUG, -4016: OUR_BUG,
    -4028: OUR_BUG,    # leverage not valid
    -4061: OUR_BUG,    # order position side does not match the account's position mode
    -4164: OUR_BUG,    # order notional below the minimum (not reduce-only)
    # ---- venue_fault: the venue / transport misbehaved
    -1000: VENUE_FAULT,   # unknown error
    -1001: VENUE_FAULT,   # internal error / disconnected
    -1002: VENUE_FAULT,
    -1003: VENUE_FAULT,   # too many requests
    -1006: VENUE_FAULT, -1007: VENUE_FAULT,   # execution status unknown / timeout
    -1008: VENUE_FAULT,   # server overloaded
    -1015: VENUE_FAULT,   # too many new orders
    -1016: VENUE_FAULT,   # service shutting down
    -1020: VENUE_FAULT,
    -4141: VENUE_FAULT,   # symbol is closed
    # ---- account_unusable: the testnet account cannot trade
    -2014: ACCOUNT_UNUSABLE, -2015: ACCOUNT_UNUSABLE,   # API key format / key, IP or permission rejected
    -2018: ACCOUNT_UNUSABLE,   # balance insufficient
    -2019: ACCOUNT_UNUSABLE,   # margin insufficient
    -2028: ACCOUNT_UNUSABLE,   # leverage not valid for the balance
    # ---- expected answers (not failures)
    -5022: OK,   # post-only order would have taken liquidity: the maker flow re-prices
    -4046: OK,   # no need to change margin type
}
REDUCE_REJECTED = -2022


class VenueTimeout(NetworkError):
    """A venue call that did not answer within the mirror's timeout (the order's fate is unknown)."""


def classify(exc: BaseException | None, *, account_unusable: bool = False) -> tuple[str, str | None]:
    """``(class, error_code)`` of a venue call outcome. ``exc`` None = success."""
    if exc is None:
        return OK, None
    if isinstance(exc, VenueTimeout):
        return VENUE_FAULT, "timeout"
    if isinstance(exc, OrderNotFilled):
        return VENUE_FAULT, "not_filled"
    if isinstance(exc, RateLimitError):
        return VENUE_FAULT, str(exc.code) if exc.code is not None else "rate_limit"
    if isinstance(exc, ExchangeError):
        code, status = exc.code, exc.status
        if code == REDUCE_REJECTED:
            # -2022 "ReduceOnly Order is rejected": we never send reduceOnly, so it can only be the venue's answer when our own
            # state was wrong (an order against a position that is not there) - our bug; while the account is unusable (ghost
            # positions: every order gets it) it says nothing about our request
            return (VENUE_FAULT if account_unusable else OUR_BUG), str(code)
        if code is not None and code in CLASS_BY_CODE:
            return CLASS_BY_CODE[code], str(code)
        if isinstance(exc, NetworkError) or (status is not None and status >= 500):
            return VENUE_FAULT, str(code) if code is not None else (f"http_{status}" if status else "network")
        if code is not None:
            return OUR_BUG, str(code)
        low = str(exc).lower()
        if "below min_qty" in low or "qty" in low:
            return OUR_BUG, "min_qty"
        return OUR_BUG, "error"
    return OUR_BUG, type(exc).__name__          # an exception in OUR code


# ----------------------------------------------------------------------------- sizing
def min_order_qty(f: SymbolFilters, price: float, margin: float = NOTIONAL_MARGIN) -> float | None:
    """Smallest quantity that satisfies the lot step, min qty AND min notional with ``margin`` to spare (None = impossible)."""
    if not price or price <= 0:
        return None
    step = Decimal(repr(float(f.step_size))) if f.step_size > 0 else Decimal("0.001")
    px = Decimal(repr(float(price)))
    need = max(Decimal(repr(float(f.min_qty))), Decimal(repr(float(f.min_notional * margin))) / px)
    q = (need / step).to_integral_value(rounding=ROUND_CEILING) * step
    for _ in range(4):                                    # (float noise in the step arithmetic: never end up below the floor)
        if float(q) * float(price) >= f.min_notional * margin and float(q) >= f.min_qty:
            break
        q += step
    if f.max_qty and float(q) > f.max_qty:
        return None
    return float(q)


def floor_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return float((Decimal(repr(float(qty))) / Decimal(repr(float(step)))).to_integral_value(rounding=ROUND_DOWN)
                 * Decimal(repr(float(step))))


# ----------------------------------------------------------------------------- queue item
@dataclass
class Item:
    kind: str                         # open | stop | partial | close
    pid: int
    symbol: str
    data: dict
    t: float
    not_before: float = 0.0
    tries: int = 0


def _safe(fn: Callable) -> Callable:
    """Engine-facing hooks never raise."""
    @wraps(fn)
    def inner(self, *a, **kw):
        try:
            return fn(self, *a, **kw)
        except Exception:
            log.exception("mirror hook %s failed (ignored: the paper book is unaffected)", fn.__name__)
            return None
    return inner


class Mirror:
    def __init__(self, store: Any, alerts: Any, cfg_fn: Callable[[], Any], *, mode: str = "paper_local",
                 clock: Callable[[], float] = time.time, venue: Any = None,
                 venue_factory: Callable[[], Any] | None = None, sleep: Callable = asyncio.sleep,
                 call_timeout_s: float = CALL_TIMEOUT_S, poll_s: float = 1.0):
        self.store = store
        self.alerts = alerts
        self.cfg_fn = cfg_fn
        self.mode = mode
        self.clock = clock
        self.venue = venue
        self.venue_factory = venue_factory or default_venue
        self.sleep = sleep
        self.call_timeout_s = call_timeout_s
        self.poll_s = poll_s
        self.backoff = Backoff()
        self.queues: dict[str, deque[Item]] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.bg: dict[str, asyncio.Task] = {}
        self.dropped = 0
        self._pool: ThreadPoolExecutor | None = None
        self._map: dict[str, dict] | None = None
        self._venue_err = ""
        self._venue_try_at = 0.0
        self._acct_lock: asyncio.Lock | None = None
        self._acct_loop: Any = None
        self.account: dict[str, Any] = {"state": "unknown", "since": 0.0, "t": 0.0, "reasons": [], "margin_balance": None,
                                        "available": None, "wallet": None, "can_trade": None}
        self._need: dict[str, float] = {}                 # symbol -> margin its last min-size entry needed
        self._alert_seen: dict[tuple, float] = {}
        self._next_recon = 0.0
        self._next_poll = 0.0
        self._state_at = 0.0
        self._pruned_at = 0.0
        self._foreign: list[str] = []
        self._opening: set[int] = set()                   # paper ids whose entry is queued / running (their later hooks must queue too)
        self._rejects: deque[tuple[float, str]] = deque(maxlen=REJECT_STREAK)
        self._built = False                               # the venue was built by venue_factory (so a retry may rebuild it)
        self._retry_seen = 0.0
        self._retry_at = 0.0
        self.stopped = False

    # ================================================================== small helpers
    def cfg(self) -> Any:
        return self.cfg_fn()

    @property
    def enabled(self) -> bool:
        try:
            c = self.cfg()
            return self.mode == "paper_local" and bool(c.mirror_enabled) and c.mirror_venue == "binance_demo"
        except Exception:
            return False

    @property
    def map(self) -> dict[str, dict]:
        if self._map is None:
            try:
                got = self.store.kv_get(KEY_MAP, {})
                self._map = {str(k): dict(v) for k, v in got.items() if isinstance(v, dict)} if isinstance(got, dict) else {}
            except Exception:
                log.exception("mirror map unreadable; starting empty")
                self._map = {}
        return self._map

    def save_map(self) -> None:
        try:
            self.store.kv_set(KEY_MAP, self.map)
        except Exception:
            log.exception("mirror map not saved")

    @property
    def pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tb-mirror")
        return self._pool

    def owned_symbols(self) -> set[str]:
        return {m["symbol"] for m in self.map.values() if m.get("state") in ("entering", "open", "closing")}

    def _paper_open(self, pid: int) -> bool:
        return self.store.one("SELECT 1 FROM positions WHERE id=?", (pid,)) is not None

    def _active(self) -> bool:
        """Anything to do: enabled, or still owning venue positions that must be wound down."""
        return self.enabled or any(m.get("state") in ("entering", "open", "closing") for m in self.map.values())

    # ================================================================== engine hooks (sync, never block, never raise)
    @_safe
    def on_open(self, pid: int, pos: dict, via_maker: bool = False) -> None:
        """A paper position was opened (entry filled, stop/TP recorded)."""
        if not self.enabled:
            return
        d = {k: pos.get(k) for k in ("symbol", "side", "entry_price", "qty", "leverage", "sl_price", "tp_price")}
        d["via_maker"] = bool(via_maker)
        self._opening.add(int(pid))
        self._enqueue(Item("open", int(pid), str(pos["symbol"]), d, self.clock()))

    @_safe
    def on_stop_move(self, pid: int, symbol: str, new_sl: float) -> None:
        """The paper stop moved (break-even / trailing)."""
        if self._owns(pid):
            self._enqueue(Item("stop", int(pid), str(symbol), {"sl_price": float(new_sl)}, self.clock()))

    @_safe
    def on_partial(self, pid: int, symbol: str, frac: float, sl_price: float | None, tp_price: float | None) -> None:
        """The paper book closed ``frac`` of the position (partial take-profit); stop / TP now cover the remainder."""
        if self._owns(pid):
            self._enqueue(Item("partial", int(pid), str(symbol), {"frac": float(frac), "sl_price": sl_price,
                                                                  "tp_price": tp_price}, self.clock()))

    @_safe
    def on_close(self, pid: int, symbol: str, reason: str, emergency: bool = False) -> None:
        """The paper position closed (any reason: tp / sl / trail / kill / manual / timeout / liquidation / ...)."""
        if self._owns(pid):
            self._enqueue(Item("close", int(pid), str(symbol), {"reason": str(reason), "emergency": bool(emergency)},
                               self.clock()))

    def _owns(self, pid: int) -> bool:
        return str(pid) in self.map or int(pid) in self._opening

    def _enqueue(self, it: Item) -> None:
        if sum(len(q) for q in self.queues.values()) >= MAX_QUEUE:
            self.dropped += 1
            log.error("mirror queue full (%d); dropping %s %s (the reconciler repairs the divergence)", MAX_QUEUE, it.kind, it.symbol)
            self._opening.discard(it.pid) if it.kind == "open" else None
            return
        self.queues.setdefault(it.symbol, deque()).append(it)

    # ================================================================== recording / alerting
    def _record(self, symbol: str | None, action: str, pid: int | None, oid: str | None, ok: bool, code: str | None,
                msg: str | None, lat: int | None, cls: str) -> None:
        try:
            self.store.execute(
                "INSERT INTO mirror_events(t,symbol,action,paper_position_id,venue_order_id,ok,error_code,error_msg,"
                "latency_ms,cls) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (self.clock(), symbol, action, pid, None if oid is None else str(oid), 1 if ok else 0, code,
                 (msg or None) and str(msg)[:300], lat, cls))
        except Exception:
            log.exception("mirror event not recorded")
        try:
            self._route_alert(symbol, action, code, msg, cls)
        except Exception:
            log.exception("mirror alert failed")

    def _route_alert(self, symbol: str | None, action: str, code: str | None, msg: str | None, cls: str) -> None:
        if action == "divergence":
            self.alerts.mirror_note("divergence", symbol or "-", code or "")
        elif cls == OUR_BUG:
            key = (code, symbol)
            now = self.clock()
            if now - self._alert_seen.get(key, -1e18) >= ALERT_EVERY_S:
                self._alert_seen[key] = now
                self.alerts.mirror_bug(symbol or "-", action, code or "", str(msg or ""))
        elif cls == VENUE_FAULT:
            self.alerts.mirror_note("venue_fault", symbol or "-", code or "")

    def _skip(self, symbol: str | None, action: str, pid: int | None, code: str, msg: str) -> None:
        self._record(symbol, action, pid, None, False, code, msg, None, SKIPPED)

    def _diverge(self, symbol: str, pid: int | None, kind: str, msg: str, cls: str = OUR_BUG) -> None:
        log.warning("mirror divergence %s %s (paper position %s): %s", kind, symbol, pid, msg)
        self._record(symbol, "divergence", pid, None, False, kind, msg, None, cls)

    # ================================================================== venue plumbing
    async def _venue(self) -> Any:
        """The venue client (built lazily; a missing key is retried every 60 s). None = unavailable."""
        if self.venue is not None:
            return self.venue
        now = self.clock()
        if now - self._venue_try_at < 60.0 and self._venue_err:
            return None
        self._venue_try_at = now
        try:
            self.venue = self.venue_factory()
            self._built = True
            self._venue_err = ""
        except Exception as e:
            first = self._venue_err != str(e)
            self._venue_err = str(e)
            if first:
                log.error("testnet mirror has no venue: %s", e)
                self.alerts.mirror_unusable(["no testnet venue: " + str(e)])
            return None
        return self.venue

    async def _blocking(self, fn: Callable, *args: Any, timeout: float | None = None) -> Any:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(loop.run_in_executor(self.pool, partial(fn, *args)), timeout or self.call_timeout_s)

    async def _call(self, action: str, symbol: str | None, pid: int | None, fn: Callable, *args: Any,
                    record: str = "always", timeout: float | None = None) -> tuple[Any, BaseException | None, str]:
        """One venue call: timeout, exception capture, classification, a mirror_events row (``record``: "always" for order
        path calls, "fail" for reads), backoff bookkeeping. Returns ``(result, exception, class)``; never raises."""
        t0 = time.monotonic()
        res, exc = None, None
        try:
            res = await self._blocking(fn, *args, timeout=timeout)
        except TimeoutError:
            exc = VenueTimeout(f"{action} timed out after {timeout or self.call_timeout_s:g}s")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            exc = e
        lat = int((time.monotonic() - t0) * 1000)
        cls, code = classify(exc, account_unusable=self.account.get("state") == "unusable")
        oid = None
        if exc is None:
            oid = getattr(res, "order_id", None) or (res if isinstance(res, str) else None)
        if exc is not None or record == "always":
            self._record(symbol, action, pid, oid, cls == OK, code, None if exc is None else str(exc), lat, cls)
        if symbol and exc is not None and cls in (OUR_BUG, VENUE_FAULT):
            delay = self.backoff.fail(symbol, self.clock())
            log.warning("mirror %s %s failed [%s %s]: %s (symbol backoff %.0fs)", action, symbol, cls, code, exc, delay)
        elif symbol and exc is None:
            self.backoff.reset(symbol)
        if cls == ACCOUNT_UNUSABLE or (exc is not None and getattr(exc, "code", None) in (-2018, -2019)):
            self._next_poll = 0.0                         # the venue says the account cannot trade: re-check it now
        if action in REJECT_ACTIONS:
            self._note_order(symbol, exc)
        return res, exc, cls

    def _note_order(self, symbol: str | None, exc: BaseException | None) -> None:
        """Order outcomes on clean symbols are evidence about the account that its own balance fields can hide (a ghost-position
        testnet account rejects every order with -2019 / -2022 even with a positive margin balance)."""
        if exc is None:
            self._rejects.clear()
            return
        if getattr(exc, "code", None) not in REJECT_CODES or not symbol:
            return
        self._rejects.append((self.clock(), symbol))
        if len(self._rejects) >= REJECT_STREAK and len({s for _, s in self._rejects}) >= REJECT_SYMBOLS \
                and self.account.get("state") != "unusable":
            syms = sorted({s for _, s in self._rejects})
            why = (f"{REJECT_STREAK} order rejects in a row ({'/'.join(str(c) for c in REJECT_CODES)}) on {', '.join(syms)} "
                   "although the account endpoint looks healthy")
            self.account["sticky"] = {"why": why, "wallet": self.account.get("wallet"), "margin": self.account.get("margin_balance")}
            self._set_account("unusable", [why], None)

    # ================================================================== account health
    def _need_margin(self, leverage: int = 8) -> float:
        """Margin a min-size entry needs (the largest seen, else a 5 USD floor-notional at the paper leverage)."""
        seen = max(self._need.values()) if self._need else 5.0 * NOTIONAL_MARGIN / max(1, leverage) * 1.2
        return max(seen, 0.1)

    def _set_account(self, state: str, reasons: list[str], info: dict | None) -> None:
        old = self.account.get("state")
        now = self.clock()
        self.account.update({"state": state, "reasons": reasons, "t": now})
        if info is not None:
            self.account.update(info)
        if state != old:
            self.account["since"] = now
            if state == "unusable":
                log.warning("testnet account unusable (%s): the mirror stops sending orders until it recovers", "; ".join(reasons))
                self.alerts.mirror_unusable(reasons)
            elif state == "ok" and old == "unusable":
                log.info("testnet account usable again: the mirror resumes")
                self.alerts.mirror_recovered()
        self._write_state(force=True)

    async def poll_account(self) -> str:
        """Poll ``/fapi/v2/account`` and set the account state. Returns the state."""
        loop = asyncio.get_running_loop()
        if self._acct_lock is None or self._acct_loop is not loop:
            self._acct_lock, self._acct_loop = asyncio.Lock(), loop
        async with self._acct_lock:
            v = await self._venue()
            if v is None:
                self._set_account("unavailable", [self._venue_err or "no venue"], None)
                return "unavailable"
            self._next_poll = self.clock() + ACCOUNT_POLL_S
            info, exc, _cls = await self._call("account", None, None, v.account_info, record="fail")
            if exc is not None or not isinstance(info, dict):
                if self.account.get("state") in ("unknown", "unavailable"):
                    self._set_account("unknown", [f"account unreadable: {exc}"], None)
                else:
                    self.account["t"] = self.clock()          # keep the last verdict; the next poll retries
                return self.account["state"]
            try:
                margin = float(info.get("totalMarginBalance", 0) or 0)
                avail = float(info.get("availableBalance", 0) or 0)
                wallet = float(info.get("totalWalletBalance", 0) or 0)
                can = info.get("canTrade", True)
            except (TypeError, ValueError):
                self._set_account("unknown", ["account payload unreadable"], None)
                return "unknown"
            reasons: list[str] = []
            if margin <= 0:
                reasons.append(f"totalMarginBalance {margin:g} <= 0")
            if can is False:
                reasons.append("canTrade is false")
            need = self._need_margin(int(getattr(self.cfg(), "leverage", 8) or 8))
            if avail < need:
                reasons.append(f"availableBalance {avail:g} < {need:g} needed for a min-size entry")
            sticky = self.account.get("sticky")
            if sticky and not reasons:
                # unusable because ORDERS were rejected: the balance fields cannot clear that. It clears when the account
                # itself changes (fresh / funded testnet account: wallet or margin moved) or on a user retry.
                w0, m0 = sticky.get("wallet"), sticky.get("margin")
                same = (w0 is not None and m0 is not None and abs(wallet - w0) <= 0.01 * max(1.0, abs(w0))
                        and abs(margin - m0) <= 0.01 * max(1.0, abs(m0)))
                if same:
                    reasons.append(sticky["why"])
                else:
                    self.account.pop("sticky", None)
                    self._rejects.clear()
            self._set_account("unusable" if reasons else "ok", reasons,
                              {"margin_balance": margin, "available": avail, "wallet": wallet, "can_trade": can})
            return self.account["state"]

    async def retry(self) -> str:
        """User-triggered "retry mirror": forget the order-reject verdict and the symbol backoffs, rebuild the venue client when
        the keys changed (fresh testnet account), and poll the account now."""
        self.account.pop("sticky", None)
        self._rejects.clear()
        self.backoff = Backoff()
        if self._built:
            try:
                fresh = self.venue_factory()
                if getattr(fresh, "_key", None) != getattr(self.venue, "_key", None):
                    log.info("mirror retry: the testnet keys changed; using the new client")
                self.venue, self._venue_err = fresh, ""
            except Exception as e:
                log.warning("mirror retry: venue not rebuilt (%s); keeping the current client", e)
        log.info("mirror retry requested: re-checking the testnet account")
        self._set_account("unknown", ["retry requested"], None)
        st = await self.poll_account()
        self._next_recon = 0.0
        return st

    async def _ensure_account(self) -> str:
        if self.clock() - float(self.account.get("t") or 0.0) > ACCOUNT_MAX_AGE_S or self.account["state"] == "unknown":
            await self.poll_account()
        return self.account["state"]

    # ================================================================== processing
    async def cycle(self) -> None:
        """One scheduler pass (called every ~1 s by the engine). Starts work; never awaits it; never raises."""
        try:
            now = self.clock()
            for sym, q in list(self.queues.items()):
                if not q:
                    self.queues.pop(sym, None)
                    continue
                if sym not in self.tasks and q[0].not_before <= now:
                    self.tasks[sym] = asyncio.create_task(self._symbol_task(sym))
            if self._active():
                if now >= self._next_poll and "poll" not in self.bg:
                    self._spawn("poll", self.poll_account())
                if now >= self._next_recon and "recon" not in self.bg:
                    self._next_recon = now + RECONCILE_S
                    self._spawn("recon", self.reconcile())
            if now - self._retry_at >= 2.0:                # "retry mirror" from the API (a kv flag: no engine command needed)
                self._retry_at = now
                asked = float(self.store.kv_get(KEY_RETRY, 0) or 0)
                if asked > self._retry_seen:
                    self._retry_seen = asked
                    if "retry" not in self.bg:
                        self._spawn("retry", self.retry())
            if now - self._pruned_at > 3600:
                self._pruned_at = now
                self.store.execute("DELETE FROM mirror_events WHERE t<?", (now - KEEP_DAYS * 86400,))
            self._write_state()
        except Exception:
            log.exception("mirror cycle failed (ignored)")

    def _spawn(self, name: str, coro: Any) -> None:
        async def run() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("mirror %s failed (ignored)", name)
            finally:
                self.bg.pop(name, None)
        self.bg[name] = asyncio.create_task(run())

    async def _symbol_task(self, sym: str) -> None:
        try:
            while True:
                q = self.queues.get(sym)
                if not q or q[0].not_before > self.clock():
                    break
                it = q.popleft()
                await self._handle(it)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("mirror task for %s failed (ignored)", sym)
        finally:
            self.tasks.pop(sym, None)
            if not self.queues.get(sym):
                self.queues.pop(sym, None)

    async def drain(self, max_s: float = 30.0, reconcile: bool = False) -> None:
        """Run until the queue is empty and nothing is in flight (tests / shutdown). Optionally one reconcile at the end."""
        end = time.monotonic() + max_s
        while time.monotonic() < end:
            await self.cycle()
            busy = bool(self.tasks) or bool(self.bg) or any(q and q[0].not_before <= self.clock() for q in self.queues.values())
            if not busy:
                break
            await asyncio.sleep(0.005)
        if reconcile:
            await self.reconcile()

    async def stop(self) -> None:
        self.stopped = True
        for t in [*self.tasks.values(), *self.bg.values()]:
            t.cancel()
        await asyncio.gather(*self.tasks.values(), *self.bg.values(), return_exceptions=True)
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    async def _handle(self, it: Item) -> None:
        try:
            if it.kind == "open":
                try:
                    await self._do_open(it)
                finally:
                    self._opening.discard(it.pid)
            elif it.kind == "stop":
                await self._do_stop(it)
            elif it.kind == "partial":
                await self._do_partial(it)
            elif it.kind == "close":
                await self._do_close(it)
        except asyncio.CancelledError:
            raise
        except Exception as e:                           # an exception in OUR code is a bug worth an alert
            log.exception("mirror %s %s crashed", it.kind, it.symbol)
            cls, code = classify(e)
            self._record(it.symbol, it.kind, it.pid, None, False, "internal_error", f"{type(e).__name__}: {e}", None, OUR_BUG)

    def _defer(self, it: Item, delay: float) -> bool:
        """Put ``it`` back at the head of its symbol queue, ``delay`` s from now (False after MAX_TRIES)."""
        it.tries += 1
        if it.tries >= MAX_TRIES:
            return False
        it.not_before = self.clock() + delay
        self.queues.setdefault(it.symbol, deque()).appendleft(it)
        return True

    # ------------------------------------------------------------------ entry
    async def _gate(self, sym: str, pid: int, action: str) -> Any:
        """Venue + account gate. Returns the venue, or None after recording why nothing was sent."""
        v = await self._venue()
        if v is None:
            self._skip(sym, action, pid, "no_venue", self._venue_err or "no testnet venue")
            return None
        st = await self._ensure_account()
        if st != "ok":
            self._skip(sym, action, pid, "account_unusable" if st == "unusable" else f"account_{st}",
                       "; ".join(self.account.get("reasons") or []) or f"account state {st}")
            return None
        return v

    async def _do_open(self, it: Item) -> None:
        pid, sym, d = it.pid, it.symbol, it.data
        key = str(pid)
        base = {"symbol": sym, "side": d["side"], "paper_entry": float(d["entry_price"]), "state": "skipped", "t": it.t}
        if not self._paper_open(pid):
            self._skip(sym, "entry", pid, "paper_closed", "the paper position closed before the mirror entry ran")
            return
        v = await self._gate(sym, pid, "entry")
        if v is None:
            self.map[key] = base
            self.save_map()
            return
        if not self.backoff.ready(sym, self.clock()):
            self._skip(sym, "entry", pid, "backoff", f"symbol in backoff for {self.backoff.wait_left(sym, self.clock()):.0f}s")
            self.map[key] = {**base, "state": "failed"}
            self.save_map()
            return
        info, exc, cls = await self._call("read_exchange_info", sym, pid, v.exchange_info, record="fail")
        if exc is not None:
            self.map[key] = {**base, "state": "failed"}
            self.save_map()
            return
        si = info.get(sym)
        if si is None or getattr(si, "status", "TRADING") != "TRADING":
            self._skip(sym, "entry", pid, "not_trading", f"{sym} is not TRADING on the venue")
            self.map[key] = base
            self.save_map()
            return
        vps, exc, cls = await self._call("read_positions", sym, pid, v.positions, record="fail")
        if exc is not None:
            self.map[key] = {**base, "state": "failed"}
            self.save_map()
            return
        other = [m for k, m in self.map.items() if k != key and m["symbol"] == sym and m.get("state") in ("entering", "open", "closing")]
        if other or any(p.symbol == sym for p in vps):
            self._skip(sym, "entry", pid, "foreign_position", f"{sym} already has a venue position the mirror does not own")
            self.map[key] = base
            self.save_map()
            return
        prem, exc, cls = await self._call("read_mark", sym, pid, v.premium, sym, record="fail")
        mark = float(prem[0].mark_price) if exc is None and prem else 0.0
        if exc is not None or mark <= 0:
            self.map[key] = {**base, "state": "failed"}
            self.save_map()
            return
        f = si.filters
        qty = min_order_qty(f, mark)
        if qty is None:
            self._skip(sym, "entry", pid, "no_min_size", f"no valid minimum size for {sym} at {mark:g}")
            self.map[key] = base
            self.save_map()
            return
        lev = max(1, int(d.get("leverage") or 1))
        self._need[sym] = qty * mark / lev * 1.2
        if len(self._need) > 50:
            self._need.pop(next(iter(self._need)))
        if float(self.account.get("available") or 0.0) < self._need[sym]:
            await self.poll_account()                     # re-evaluate with this entry's real need
            if self.account["state"] != "ok":
                self._skip(sym, "entry", pid, "account_unusable", "; ".join(self.account.get("reasons") or []))
                self.map[key] = base
                self.save_map()
                return
        # ---- ownership is persisted BEFORE the order: a timed-out entry is adopted / closed by the reconciler
        side = d["side"]
        m = {**base, "state": "entering", "qty": 0.0, "req_qty": qty, "v_entry": mark, "sl_id": None, "tp_id": None,
             "sl_applied": None, "tp_applied": None, "sl_paper": d.get("sl_price"), "tp_paper": d.get("tp_price"),
             "stop_fail": 0, "orphans": [], "t_enter": self.clock()}
        self.map[key] = m
        self.save_map()
        _, exc, _cls = await self._call("set_isolated", sym, pid, v.set_isolated, sym)
        if exc is not None and getattr(exc, "code", None) != -4046:
            return self._entry_failed(key, "set_isolated failed")
        lev_ok = False
        for attempt in (lev, max(1, lev // 2), 1):
            _, exc, _cls = await self._call("set_leverage", sym, pid, v.set_leverage, sym, attempt)
            if exc is None:
                lev_ok = True
                break
        if not lev_ok:
            return self._entry_failed(key, "set_leverage failed")
        got = await (self._enter_maker(v, sym, side, qty, pid, f) if d.get("via_maker") else self._enter_market(v, sym, side, qty, pid))
        if got is None:
            return None
        if got == "unknown":                              # timed out: the reconciler adopts or abandons it
            self._next_recon = min(self._next_recon, self.clock() + 5.0)
            return None
        fq, avg = got
        m.update(state="open", qty=fq, v_entry=avg or mark)
        self.save_map()
        await self._place_protection(v, key, f, close_side_of(side))

    def _entry_failed(self, key: str, why: str) -> None:
        m = self.map.get(key)
        if m:
            m["state"] = "failed"
            m["why"] = why
            self.save_map()

    async def _enter_market(self, v: Any, sym: str, side: str, qty: float, pid: int) -> Any:
        res, exc, cls = await self._call("entry", sym, pid, v.market_order, sym, order_side(side), qty)
        key = str(pid)
        if exc is None:
            return float(res.qty or qty), float(res.avg_price or 0.0)
        if isinstance(exc, VenueTimeout):
            return "unknown"
        self._entry_failed(key, f"entry rejected: {exc}")
        return None

    async def _enter_maker(self, v: Any, sym: str, side: str, qty: float, pid: int, f: SymbolFilters) -> Any:
        """Post-only GTX limit at the venue's best bid / ask, re-priced up to ``maker_reprices`` times; the first fill ends the
        attempt (like the paper flow). No fill: market per ``maker_on_timeout`` == taker, else the entry is abandoned."""
        cfg = self.cfg()
        key = str(pid)
        long = side == "long"
        osd = order_side(side)
        terminal = ("FILLED", "CANCELED", "CANCELLED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH")
        polls = max(1, math.ceil(cfg.maker_timeout_s / self.poll_s))
        got_q = got_n = 0.0
        unknown = False
        for attempt in range(int(cfg.maker_reprices) + 1):
            if not self._paper_open(pid):
                break
            ba, exc, _c = await self._call("read_book", sym, pid, v.best_bid_ask, sym, record="fail")
            if exc is not None:
                await self.sleep(self.poll_s)
                continue
            px = f.round_price(ba[0] if long else ba[1])
            rq = f.round_qty(qty - got_q)
            if rq < f.min_qty or rq <= 0:
                break
            res, exc, cls = await self._call("entry_limit", sym, pid, v.limit_order, sym, osd, rq, px, True)
            if exc is not None:
                if getattr(exc, "code", None) == -5022:
                    await self.sleep(0.5)
                    continue
                if isinstance(exc, VenueTimeout):
                    unknown = True
                break
            cur, oid = res, res.order_id
            for _ in range(polls):
                if cur.filled > 0 or cur.status in terminal:
                    break
                await self.sleep(self.poll_s)
                if not self._paper_open(pid):
                    break
                nxt, exc, _c = await self._call("order_status", sym, pid, v.get_order, sym, oid, record="fail")
                if exc is None:
                    cur = nxt
            if cur.status != "FILLED":
                await self._call("cancel_entry", sym, pid, v.cancel_order, sym, oid)
                fin, exc, _c = await self._call("order_status", sym, pid, v.get_order, sym, oid, record="fail")
                if exc is None:
                    cur = fin
                else:
                    unknown = True
            got_q += cur.filled
            got_n += cur.filled * (cur.avg_price or px)
            if cur.filled > 0 or unknown:
                break
        if got_q > 0:
            return got_q, got_n / got_q
        if unknown:
            return "unknown"
        if cfg.maker_on_timeout == "taker" and self._paper_open(pid):
            return await self._enter_market(v, sym, side, qty, pid)
        self._skip(sym, "entry", pid, "maker_no_fill", "no maker fill; entry abandoned (maker_on_timeout=skip)")
        self._entry_failed(key, "maker entry did not fill")
        return None

    async def _place_protection(self, v: Any, key: str, f: SymbolFilters, close_side: str) -> None:
        """TP / SL of the mirror position as closePosition orders (no quantity, no reduce-only), at the paper levels scaled to the
        venue's entry price - placed right after the entry fill, like the Binance UI's TP/SL checkbox."""
        m = self.map[key]
        sym, pid = m["symbol"], int(key)
        if m.get("sl_paper"):
            ok = await self._set_stop(v, key, f, close_side, float(m["sl_paper"]), old_id=None)
            if not ok:
                return
        if m.get("tp_paper"):
            px = f.round_price(m["v_entry"] * float(m["tp_paper"]) / m["paper_entry"])
            oid, exc, _c = await self._call("take_profit", sym, pid, v.place_take_profit, sym, close_side, px)
            if exc is None:
                m["tp_id"], m["tp_applied"] = oid, float(m["tp_paper"])
                self.save_map()

    def vprice(self, m: dict, paper_px: float, f: SymbolFilters) -> float:
        return f.round_price(m["v_entry"] * float(paper_px) / m["paper_entry"])

    async def _set_stop(self, v: Any, key: str, f: SymbolFilters, close_side: str, paper_px: float,
                        old_id: str | None, action: str = "stop") -> bool:
        """Place the closePosition stop at ``paper_px`` (scaled). Moving one is CANCEL then PLACE (the venue keeps one closePosition
        stop per direction, -4130): if the cancel fails the old stop is still live and nothing else happens; if the placement fails
        the old level is restored. A first stop the venue refuses (-2021), or a stop that failed ``STOP_FAIL_CLOSE`` times,
        closes the mirror position, like the engine does."""
        m = self.map[key]
        sym, pid = m["symbol"], int(key)
        px = self.vprice(m, paper_px, f)
        old_px = self.vprice(m, m["sl_applied"], f) if m.get("sl_applied") is not None else None
        if old_id:
            if not await self._cancel_tracked(v, key, old_id, "cancel_stop"):
                return False
            m["sl_id"] = None
            self.save_map()
        oid, exc, cls = await self._call(action, sym, pid, v.place_stop, sym, close_side, px)
        if exc is not None:
            m["stop_fail"] = int(m.get("stop_fail", 0)) + 1
            self.save_map()
            if old_id and old_px is not None:
                rid, e2, _c = await self._call("stop_restore", sym, pid, v.place_stop, sym, close_side, old_px)
                if e2 is None:
                    m["sl_id"] = rid
                    self.save_map()
            first_refused = action == "stop" and getattr(exc, "code", None) == -2021
            if first_refused or m["stop_fail"] >= STOP_FAIL_CLOSE:
                self._diverge(sym, pid, "stop_failed", f"stop @ {px:g} refused/failed ({exc}); closing the mirror position")
                await self._close_venue(v, key, "stop_failed", emergency=True)
            return False
        m["stop_fail"] = 0
        m["sl_id"], m["sl_applied"] = oid, float(paper_px)
        self.save_map()
        return True

    async def _cancel_tracked(self, v: Any, key: str, order_id: str, action: str) -> bool:
        """Cancel an order the mirror believes exists. The client swallows 'unknown order', so look first: an id we track that
        the venue does not list is an our_bug (our state was wrong) and counts as gone. False = the cancel failed (still live)."""
        m = self.map[key]
        sym, pid = m["symbol"], int(key)
        orders, exc, _c = await self._call("read_orders", sym, pid, v.open_conditional_orders, sym, record="fail")
        if exc is None and order_id not in {o.order_id for o in orders}:
            self._record(sym, action, pid, order_id, False, "-2011",
                         f"unknown order: tracked {order_id} is not open on the venue", None, OUR_BUG)
            return True
        _, exc, _c = await self._call(action, sym, pid, v.cancel_order, sym, order_id)
        if exc is not None:
            m.setdefault("orphans", []).append(order_id)
            self.save_map()
            return False
        return True

    # ------------------------------------------------------------------ stop moves / partial
    async def _do_stop(self, it: Item) -> None:
        key = str(it.pid)
        m = self.map.get(key)
        if not m or m.get("state") != "open":
            return
        m["sl_paper"] = it.data["sl_price"]
        self.save_map()
        await self._sync_stop(m, key, it)

    async def _sync_stop(self, m: dict, key: str, it: Item | None = None) -> bool:
        sym, pid = m["symbol"], int(key)
        v = await self._gate(sym, pid, "stop_move")
        if v is None:
            return False
        if not self.backoff.ready(sym, self.clock()):
            self._skip(sym, "stop_move", pid, "backoff", "symbol in backoff; the reconciler re-syncs the stop")
            return False
        f = v.filters(sym) if hasattr(v, "filters") else SymbolFilters()
        return await self._set_stop(v, key, f, close_side_of(m["side"]), float(m["sl_paper"]), old_id=m.get("sl_id"),
                                    action="stop_move")

    async def _do_partial(self, it: Item) -> None:
        key, sym, pid = str(it.pid), it.symbol, it.pid
        m = self.map.get(key)
        if not m or m.get("state") != "open":
            return
        v = await self._gate(sym, pid, "partial_close")
        if v is None:
            return
        if not self.backoff.ready(sym, self.clock()):
            self._skip(sym, "partial_close", pid, "backoff", "symbol in backoff")
            return
        f = v.filters(sym)
        qty = float(m["qty"])
        q_close = floor_step(qty * float(it.data["frac"]), f.step_size)
        rest = floor_step(qty - q_close, f.step_size)
        if q_close < f.min_qty or rest < f.min_qty or q_close <= 0 or rest <= 0:
            self._skip(sym, "partial_close", pid, "too_small", f"mirror qty {qty:g} cannot be split (lot step {f.step_size:g})")
            return
        # a NORMAL market order for the partial quantity (no reduce-only); the closePosition TP / SL cover the remainder
        res, exc, cls = await self._call("partial_close", sym, pid, v.market_order, sym, close_side_of(m["side"]), q_close)
        if exc is not None:
            return
        m["qty"] = float(rest)
        self.save_map()
        sl = it.data.get("sl_price")
        if sl and (m.get("sl_applied") is None or abs(float(sl) - float(m["sl_applied"])) > 1e-9 * max(1.0, abs(float(sl)))):
            m["sl_paper"] = float(sl)                          # (the paper book tightened the stop to break-even with the partial)
            await self._set_stop(v, key, f, close_side_of(m["side"]), float(sl), old_id=m.get("sl_id"), action="stop_move")

    # ------------------------------------------------------------------ closing
    async def _do_close(self, it: Item) -> None:
        key, sym, pid = str(it.pid), it.symbol, it.pid
        m = self.map.get(key)
        if not m:
            return
        st = m.get("state")
        if st in ("skipped", "failed", "closed_early"):
            self.map.pop(key, None)
            self.save_map()
            return
        if st == "entering":                              # the entry's fate is unknown: the reconciler decides
            self._next_recon = min(self._next_recon, self.clock() + 5.0)
            return
        v = await self._gate(sym, pid, "close")
        if v is None:
            return                                         # (nothing sent; the reconciler closes it once the account is usable)
        if not self.backoff.ready(sym, self.clock()):
            if not self._defer(it, self.backoff.wait_left(sym, self.clock()) + 0.5):
                self._skip(sym, "close", pid, "gave_up", "close retries exhausted; the reconciler closes the orphan")
            return
        ok = await self._close_venue(v, key, it.data.get("reason", ""), emergency=bool(it.data.get("emergency")))
        if not ok and key in self.map:
            if not self._defer(it, max(5.0, self.backoff.wait_left(sym, self.clock()))):
                log.error("mirror close of %s gave up after %d tries; the reconciler will retry", sym, MAX_TRIES)

    async def _close_venue(self, v: Any, key: str, reason: str, emergency: bool = False, action: str = "close") -> bool:
        """Close the mirror position with a NORMAL market order (no reduce-only): ``emergency.close_flat`` re-reads the position
        under the symbol lock, sends exactly the quantity the venue reports, verifies it is flat and flattens a reverse position a
        racing stop may have left (recorded as an our_bug ``close_flip``). Returns True when the venue is flat."""
        m = self.map.get(key)
        if not m:
            return True
        sym, pid = m["symbol"], int(key)
        m["state"] = "closing"
        self.save_map()
        cancel_first = emergency or action == "orphan_close"
        t0 = time.monotonic()
        try:
            out = await self._blocking(partial(close_flat, v, sym, cancel_first=cancel_first, say=lambda s: log.warning("%s", s)),
                                       timeout=45.0)
        except TimeoutError:
            self._record(sym, action, pid, None, False, "timeout", "close timed out", int((time.monotonic() - t0) * 1000),
                         VENUE_FAULT)
            self.backoff.fail(sym, self.clock())
            return False
        except Exception as e:
            cls, code = classify(e, account_unusable=self.account.get("state") == "unusable")
            self._record(sym, action, pid, None, False, code, str(e), int((time.monotonic() - t0) * 1000), cls)
            self.backoff.fail(sym, self.clock())
            return False
        lat = int((time.monotonic() - t0) * 1000)
        if out.flipped:
            self._record(sym, "close_flip", pid, None, False, "reverse_position",
                         f"the close opened a reverse position (a stop filled in between); flattened: {out.detail}", None, OUR_BUG)
        if out.liquidation:
            if not m.get("liq_logged"):
                m["liq_logged"] = True
                self.save_map()
                self._diverge(sym, pid, "liquidation", out.detail, VENUE_FAULT)
            return False
        if out.ok:
            # always tidy up: a leftover closePosition TP / SL would close the NEXT position on this symbol at stale levels
            await self._call("cancel_orders", sym, pid, v.cancel_all, sym)
            oid = getattr(out.result, "order_id", None)
            self._record(sym, action, pid, oid, True, None, f"{out.detail} (reason={reason})", lat, OK)
            self.backoff.reset(sym)
            self.map.pop(key, None)
            self.save_map()
            return True
        cls, code = classify(out.error, account_unusable=self.account.get("state") == "unusable")
        if out.error is None:
            cls, code = VENUE_FAULT, "close_unverified"
        self._record(sym, action, pid, None, False, code, out.detail, lat, cls)
        self.backoff.fail(sym, self.clock())
        return False

    # ================================================================== reconciler
    async def reconcile(self) -> None:
        """Compare the paper book with the mirror-owned venue positions and repair the venue side only (never the paper book)."""
        self._next_recon = self.clock() + RECONCILE_S
        if not self._active() or not self.map and not self.enabled:
            return
        v = await self._venue()
        if v is None or await self._ensure_account() != "ok":
            return
        vps, exc, _c = await self._call("read_positions", None, None, v.positions, record="fail")
        if exc is not None:
            return
        by_sym = {p.symbol: p for p in vps}
        paper = {str(r["id"]): r for r in self.store.query("SELECT id,symbol,side,sl_price,tp_price FROM positions")}
        owned = set()
        now = self.clock()
        for key, m in list(self.map.items()):
            sym, pid = m["symbol"], int(key)
            if sym in self.tasks or self.queues.get(sym):
                continue                                   # an action for this symbol is in flight / queued
            twin = paper.get(key)
            st = m.get("state")
            if st in ("skipped", "closed_early"):
                if twin is None:
                    self.map.pop(key, None)
                continue
            if st == "failed":
                if twin is None:
                    self.map.pop(key, None)
                elif not m.get("div_logged"):
                    m["div_logged"] = True
                    self._diverge(sym, pid, "entry_failed", f"paper position is open but the mirror entry failed ({m.get('why', '')}); not entering late")
                continue
            owned.add(sym)
            vp = by_sym.get(sym)
            if st == "entering" and now - float(m.get("t_enter") or 0) < ENTERING_GRACE_S:
                continue
            if not self.backoff.ready(sym, now):
                continue
            if twin is None:                               # a venue position without a paper twin
                if vp is not None or st == "closing":
                    self._diverge(sym, pid, "orphan_position", f"no paper twin; closing the {m.get('side')} {m.get('qty')} venue position")
                    await self._close_venue(v, key, "orphan", emergency=True, action="orphan_close")
                else:
                    await self._call("cancel_orders", sym, pid, v.cancel_all, sym)
                    self.map.pop(key, None)
                continue
            if st == "entering":                           # a timed-out entry: adopt what the venue shows, else abandon
                if vp is not None and vp.side == m["side"]:
                    m.update(state="open", qty=float(vp.qty), v_entry=float(vp.entry_price) or m["v_entry"])
                    self.save_map()
                    log.warning("mirror adopted the venue position of timed-out entry %s", sym)
                    f = v.filters(sym)
                    await self._place_protection(v, key, f, close_side_of(m["side"]))
                else:
                    m["state"], m["why"] = "failed", "entry timed out and the venue shows no position"
                    self.save_map()
                    m["div_logged"] = True
                    self._diverge(sym, pid, "entry_failed", "entry timed out and the venue shows no position; not entering late")
                continue
            if st == "closing":
                if vp is not None:
                    await self._close_venue(v, key, "retry", emergency=False)
                continue
            # ---- state "open" with a paper twin
            if vp is None:
                self._diverge(sym, pid, "venue_closed_early", "the venue position is gone while the paper position is open "
                              "(venue stop/TP filled or liquidated)", VENUE_FAULT)
                await self._call("cancel_orders", sym, pid, v.cancel_all, sym)
                m["state"] = "closed_early"
                self.save_map()
                continue
            if vp.side != m["side"]:
                self._diverge(sym, pid, "side_mismatch", f"venue shows {vp.side}, mirror opened {m['side']}; closing it")
                await self._close_venue(v, key, "side_mismatch", emergency=True, action="orphan_close")
                continue
            f = v.filters(sym)
            if abs(vp.qty - float(m["qty"])) > f.step_size / 2:
                self._diverge(sym, pid, "qty_mismatch", f"venue qty {vp.qty:g} != mirror qty {m['qty']:g}; following the venue")
                m["qty"] = float(vp.qty)
                self.save_map()
            await self._reconcile_orders(v, key, m, vp, f, twin)
        self._foreign = sorted(s for s in by_sym if s not in owned)
        self._write_state(force=True)

    async def _reconcile_orders(self, v: Any, key: str, m: dict, vp: Any, f: SymbolFilters, twin: Any) -> None:
        sym, pid = m["symbol"], int(key)
        orders, exc, _c = await self._call("read_orders", sym, pid, v.open_conditional_orders, sym, record="fail")
        if exc is not None:
            return
        ids = {o.order_id for o in orders}
        known = {x for x in (m.get("sl_id"), m.get("tp_id")) if x}
        for o in orders:                                    # stale TP / SL orders of this (owned) symbol the mirror no longer tracks
            if o.order_id not in known:
                await self._call("cancel_stale", sym, pid, v.cancel_order, sym, o.order_id)
        for oid in list(m.get("orphans") or []):
            _, e2, _c = await self._call("cancel_stale", sym, pid, v.cancel_order, sym, oid)
            if e2 is None:
                m["orphans"].remove(oid)
        m["sl_paper"] = float(twin["sl_price"])             # the paper book's current stop is the target
        cs = close_side_of(m["side"])
        if not m.get("sl_id") or m["sl_id"] not in ids:
            level = self.vprice(m, m["sl_paper"], f)
            mark = float(vp.mark_price or 0)
            through = mark and ((mark <= level) if m["side"] == "long" else (mark >= level))
            if through:
                self._diverge(sym, pid, "stop_not_filled", f"mark {mark:g} is through the stop level {level:g} but the position "
                              "is still open (stop triggered, not filled); closing", VENUE_FAULT)
                await self._close_venue(v, key, "stop_not_filled", emergency=True)
                return
            self._diverge(sym, pid, "stop_missing", f"no live stop on the venue; re-placing @ {level:g}")
            await self._set_stop(v, key, f, cs, m["sl_paper"], old_id=None, action="stop_replace")
            return
        if m.get("sl_applied") is not None and abs(float(m["sl_paper"]) - float(m["sl_applied"])) > 1e-9 * max(1.0, abs(float(m["sl_paper"]))):
            await self._set_stop(v, key, f, cs, m["sl_paper"], old_id=m["sl_id"], action="stop_move")

    # ================================================================== state for the API
    def _write_state(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._state_at < 5.0:
            return
        self._state_at = now
        try:
            c = self.cfg()
            self.store.kv_set(KEY_STATE, {
                "enabled": self.enabled, "venue": getattr(c, "mirror_venue", "binance_demo"), "t": now,
                "account": {**self.account}, "queue": sum(len(q) for q in self.queues.values()),
                "running": len(self.tasks), "dropped": self.dropped, "foreign_positions": self._foreign,
                "backoff": {str(k): round(self.backoff.wait_left(k, now), 1) for k in self.backoff.keys()}})
        except Exception:
            log.exception("mirror state not written")


def close_side_of(pos_side: str) -> str:
    return opposite(order_side(pos_side))


def default_venue() -> Any:
    """The testnet client: existing BinanceExchange against TESTNET_BASE with the keys from ``tb.config.load_secrets``."""
    import json

    from .config import ROOT, TESTNET_BASE, load_secrets
    from .exchange.binance import BinanceExchange

    sec = load_secrets()
    key, secret = sec.api_key, sec.api_secret
    try:      # config.json is read fresh each time: new testnet keys work after a "retry mirror", no restart (env copies are stale)
        b = json.loads((ROOT / "config.json").read_text()).get("binance", {})
        if str(b.get("api_key", "")).strip() and str(b.get("api_secret", "")).strip():
            key, secret = str(b["api_key"]).strip(), str(b["api_secret"]).strip()
    except Exception:
        pass
    if not (key and secret):
        raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET (testnet keys) are not set in env, .env or config.json")
    return BinanceExchange(TESTNET_BASE, key, secret, mode="binance_demo")


# ----------------------------------------------------------------------------- read side (API process)
def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, math.ceil(p / 100.0 * len(s)) - 1))
    return float(s[k])


def _window(rows: list[Any]) -> dict:
    """Aggregates over event rows: success rate over venue calls that were attempted, counts, latency percentiles."""
    calls = [r for r in rows if r["cls"] != SKIPPED and r["action"] != "divergence"]
    by_cls: dict[str, int] = {}
    by_code: dict[str, int] = {}
    for r in rows:
        by_cls[r["cls"]] = by_cls.get(r["cls"], 0) + 1
        if r["error_code"] and r["action"] != "divergence":
            by_code[r["error_code"]] = by_code.get(r["error_code"], 0) + 1
    lats = [float(r["latency_ms"]) for r in calls if r["latency_ms"] is not None]
    return {"calls": len(calls), "success_rate": (sum(1 for r in calls if r["cls"] == OK) / len(calls)) if calls else None,
            "by_cls": by_cls, "by_error_code": by_code, "latency_p50_ms": _pct(lats, 50), "latency_p95_ms": _pct(lats, 95),
            "divergences": sum(1 for r in rows if r["action"] == "divergence")}


def summary(store: Any, cfg: Any, now: float | None = None) -> dict:
    """``GET /api/mirror``: everything the Execution panel shows, read from SQLite only."""
    now = now or time.time()
    cols = "id,t,symbol,action,paper_position_id,venue_order_id,ok,error_code,error_msg,latency_ms,cls"
    r24 = store.query(f"SELECT {cols} FROM mirror_events WHERE t>=? ORDER BY id", (now - 86400,))
    r1 = [r for r in r24 if r["t"] >= now - 3600]
    w1, w24 = _window(r1), _window(r24)
    st = store.kv_get(KEY_STATE, {}) or {}
    mp = store.kv_get(KEY_MAP, {}) or {}
    paper = {str(r["id"]) for r in store.query("SELECT id FROM positions")}
    opened = [{"paper_position_id": int(k), "symbol": m.get("symbol"), "side": m.get("side"), "qty": m.get("qty"),
               "state": m.get("state"), "venue_entry": m.get("v_entry"), "sl_paper": m.get("sl_paper"),
               "has_stop": bool(m.get("sl_id")), "has_tp": bool(m.get("tp_id")), "paper_open": k in paper}
              for k, m in mp.items() if isinstance(m, dict) and m.get("state") in ("entering", "open", "closing")]
    div_kinds: dict[str, int] = {}
    for r in r24:
        if r["action"] == "divergence":
            div_kinds[r["error_code"] or "?"] = div_kinds.get(r["error_code"] or "?", 0) + 1
    last = store.query(f"SELECT {cols} FROM mirror_events ORDER BY id DESC LIMIT 50")
    acct = st.get("account") or {}
    return {
        "enabled": bool(getattr(cfg, "mirror_enabled", False)), "venue": getattr(cfg, "mirror_venue", "binance_demo"),
        "size": getattr(cfg, "mirror_size", "min"),
        "account_state": acct.get("state") or ("disabled" if not getattr(cfg, "mirror_enabled", False) else "unknown"),
        "account": {"state": acct.get("state"), "reasons": acct.get("reasons") or [], "since": acct.get("since"),
                    "margin_balance": acct.get("margin_balance"), "available": acct.get("available"),
                    "wallet": acct.get("wallet"), "can_trade": acct.get("can_trade"), "checked_at": acct.get("t")},
        "success_rate_1h": w1["success_rate"], "success_rate_24h": w24["success_rate"],
        "latency_p50_ms": w24["latency_p50_ms"], "latency_p95_ms": w24["latency_p95_ms"],
        "counts": {"by_cls": w24["by_cls"], "by_error_code": w24["by_error_code"]},
        "counts_1h": {"by_cls": w1["by_cls"], "by_error_code": w1["by_error_code"]},
        "calls_1h": w1["calls"], "calls_24h": w24["calls"],
        "divergences": {"1h": w1["divergences"], "24h": w24["divergences"], "by_kind_24h": div_kinds},
        "open_positions": opened, "foreign_positions": st.get("foreign_positions") or [],
        "queue": st.get("queue", 0), "state_at": st.get("t"), "retry_requested_at": store.kv_get(KEY_RETRY),
        "events": [{k: r[k] for k in r.keys()} for r in last],
    }
