"""Independent safety monitor (``python -m tb.guard``).

A separate OS process with its own exchange client; it never imports ``tb.engine``. Every
``guard_interval_s`` (5 s) it checks

* engine heartbeat age (stale > ``heartbeat_stale_s`` with open positions -> make sure stops exist, DEGRADED)
* every open position has a live exchange-side stop (demo/live: places a missing one itself; if that
  fails it flattens the position; paper_local: acts through the commands table)
* account drawdown vs the kill limit (FLATTEN, then HALTED)
* exchange-wide faults (debounced): DEGRADED, FLATTEN only if the stops can not be confirmed

and publishes ``safety_guard`` (+ ``guard_heartbeat``) in the store. The engine obeys it before
every entry. NORMAL -> DEGRADED (no entries, keep stops) -> FLATTEN (close all) -> HALTED (sticky,
cleared only by a manual ``reset_kill``, which the engine signals with ``safety_reset_request``).
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Callable

from . import __version__
from . import capital as cap
from .alerts import AlertDispatcher
from .config import Settings, load_secrets, resolve_mode, trading_base_url, db_path, LiveNotAllowed
from .exchange.base import Exchange, ExchangeError, order_side
from .risk import derated_below_min_notional, drawdown_pct, positions_without_stop
from .safety import DEGRADED, FLATTEN, HALTED, NORMAL, Debouncer

log = logging.getLogger("tb.guard")

KEY_STATE = "safety_guard"
KEY_BEAT = "guard_heartbeat"
KEY_RESET = "safety_reset_request"


class Guard:
    def __init__(self, store, mode: str, ex: Exchange | None, alerts: AlertDispatcher | None = None,
                 clock: Callable[[], float] = time.time, flatten_after_s: float = 60.0):
        self.store = store
        self.mode = mode
        self.ex = ex
        self.alerts = alerts
        self.clock = clock
        self.flatten_after_s = flatten_after_s
        cfg = store.get_settings()
        self.hb_deb = Debouncer(1, cfg.fault_recover_m)          # stale heartbeat: immediate, hysteretic clear
        self.exch_deb = Debouncer(cfg.fault_mature_n, cfg.fault_recover_m)
        self.nostop_deb = Debouncer(2, 1)
        self.state = NORMAL
        self.halted_since: float | None = None
        self.flatten_reason = ""
        self.peak = 0.0
        self._last_stops_ok = clock()
        self._exch_since: float | None = None
        self._cmd_sent: dict[Any, float] = {}
        self._last_published: tuple | None = None

    # ------------------------------------------------------------------ helpers
    def _say(self, level: int, msg: str) -> None:
        log.log(level, msg)
        try:
            self.store.add_log(logging.getLevelName(level), f"[guard] {msg}")
        except Exception:
            pass

    def _equity(self, cfg: Settings) -> tuple[float | None, float]:
        """(equity, peak) from the exchange (demo/live) or the engine's status (paper)."""
        if self.ex is not None and self.mode != "paper_local":
            b = self.ex.balance()
            eq = cap.bot_equity(self.store, cfg, self.mode, b.equity, b.upnl)       # the bot's stake, not the wallet
            if eq is None:                      # stake not initialised yet (engine has not run): no basis to judge
                return None, 0.0
            basis = self.store.kv_get(cap.BASIS_KEY)
            if basis != getattr(self, "_basis", basis):         # capital_usd was changed: old peaks are on another basis
                self.peak = 0.0
            self._basis = basis
            self.peak = max(self.peak, eq, float(self.store.kv_get("peak_equity", 0.0) or 0.0))
            return eq, self.peak
        st = self.store.kv_get("status", {}) or {}
        eq = st.get("equity")
        peak = float(self.store.kv_get("peak_equity", 0.0) or 0.0)
        return (float(eq) if eq is not None else None), peak

    # ------------------------------------------------------------------ actions
    def _flatten_exchange(self, why: str) -> bool:
        """Close every exchange position (reduce-only market) and cancel stray orders. True if flat."""
        assert self.ex is not None
        ok = True
        try:
            for p in self.ex.positions():
                try:
                    self.ex.market_order(p.symbol, "SELL" if p.side == "long" else "BUY", p.qty, reduce_only=True)
                    self.ex.cancel_all(p.symbol)
                    self._say(logging.WARNING, f"flattened {p.symbol} ({why})")
                except ExchangeError as e:
                    ok = False
                    self._say(logging.ERROR, f"flatten {p.symbol} failed: {e}")
        except ExchangeError as e:
            self._say(logging.ERROR, f"flatten: cannot list positions: {e}")
            return False
        return ok

    def _place_missing_stop(self, sym: str, side: str, qty: float, entry: float, cfg: Settings) -> bool:
        assert self.ex is not None
        row = next((r for r in self.store.get_positions() if r["symbol"] == sym), None)
        d = 1.0 if side == "long" else -1.0
        px = float(row["sl_price"]) if row else entry - d * cfg.sl_usd / qty
        f = self.ex.filters(sym)
        try:
            oid = self.ex.place_stop(sym, "SELL" if side == "long" else "BUY", px, qty)
            self._say(logging.WARNING, f"placed missing stop for {sym} @ {px:g} ({oid})")
            if row:
                self.store.update_position(row["id"], sl_price=f.round_price(px), sl_order_id=oid)
            return True
        except ExchangeError as e:
            self._say(logging.ERROR, f"cannot place stop for {sym}: {e}; closing the position")
            try:
                self.ex.market_order(sym, "SELL" if side == "long" else "BUY", qty, reduce_only=True)
                self.ex.cancel_all(sym)
            except ExchangeError as e2:
                self._say(logging.ERROR, f"emergency close of {sym} failed: {e2}")
            return False

    def _command_once(self, key: Any, action: str, payload: dict, every_s: float = 30.0) -> None:
        now = self.clock()
        if now - self._cmd_sent.get(key, 0.0) >= every_s:
            self._cmd_sent[key] = now
            self.store.add_command(action, payload)
            self._say(logging.WARNING, f"queued command {action} {payload}")

    # ------------------------------------------------------------------ one cycle
    def step(self) -> dict:
        now = self.clock()
        cfg = self.store.get_settings()
        self.hb_deb.m = self.exch_deb.m = cfg.fault_recover_m
        self.exch_deb.n = cfg.fault_mature_n
        reasons: list[str] = []
        desired = NORMAL
        db_positions = self.store.get_positions()

        # ---- reset request (manual reset_kill) clears HALTED
        req = self.store.kv_get(KEY_RESET)
        if self.state == HALTED and req and self.halted_since is not None and float(req) > self.halted_since:
            self._say(logging.WARNING, "reset requested: HALTED -> NORMAL")
            self.state = NORMAL
            self.halted_since = None
            self.peak = 0.0
        sticky = self.state in (FLATTEN, HALTED)

        # ---- engine heartbeat
        hb = self.store.heartbeat()
        age = (now - hb["t"]) if hb else None
        stale = hb is None or age > cfg.heartbeat_stale_s
        hb_active = self.hb_deb.observe("heartbeat", bool(stale and (db_positions or hb is not None)), now)
        engine_stale = bool(stale)
        if hb_active and db_positions:
            desired = max(desired, DEGRADED, key=_rank)
            reasons.append(f"engine heartbeat stale {age:.0f}s with {len(db_positions)} open position(s)"
                           if age is not None else "engine heartbeat missing with open positions")

        # ---- exchange check: positions + stops
        exchange_error = False
        stops_confirmed = False
        eq: float | None = None
        peak = 0.0
        if self.ex is not None and self.mode != "paper_local":
            try:
                xpos = self.ex.positions()
                orders = self.ex.open_conditional_orders()
                stops_confirmed = True
                self._last_stops_ok = now
                bad = positions_without_stop(
                    [{"symbol": p.symbol, "side": p.side, "qty": p.qty} for p in xpos], orders)
                for p in xpos:
                    missing = self.nostop_deb.observe(("nostop", p.symbol), p.symbol in bad, now)
                    if p.symbol in bad and (missing or engine_stale):
                        desired = max(desired, DEGRADED, key=_rank)
                        reasons.append(f"stop missing on {p.symbol}")
                        if not self._place_missing_stop(p.symbol, p.side, p.qty, p.entry_price, cfg):
                            desired = FLATTEN
                            self.flatten_reason = f"cannot place stop on {p.symbol}"
                eq, peak = self._equity(cfg)
            except ExchangeError as e:
                exchange_error = True
                self._say(logging.WARNING, f"exchange check failed: {e}")
        else:
            # paper_local: the simulated stops live in the engine; verify through its persisted state
            ps = self.store.kv_get("paper_state") or {}
            conds = (ps.get("conds") or {}).values()
            for r in db_positions:
                has = any(c["symbol"] == r["symbol"] and c["kind"] == "SL" for c in conds)
                if not has and not engine_stale:
                    if self.nostop_deb.observe(("nostop", r["symbol"]), True, now):
                        desired = max(desired, DEGRADED, key=_rank)
                        reasons.append(f"stop missing on {r['symbol']}")
                        self._command_once(("close", r["id"]), "close_position", {"position_id": r["id"]})
                else:
                    self.nostop_deb.observe(("nostop", r["symbol"]), False, now)
            eq, peak = self._equity(cfg)
        exch_active = self.exch_deb.observe("exchange", exchange_error, now)
        if exch_active:
            self._exch_since = self._exch_since or now
            desired = max(desired, DEGRADED, key=_rank)
            reasons.append("exchange-wide fault (REST errors)")
            if db_positions and not stops_confirmed and now - self._last_stops_ok > self.flatten_after_s \
                    and now - self._exch_since > self.flatten_after_s:
                desired = FLATTEN
                self.flatten_reason = "exchange fault and stops cannot be confirmed"
        else:
            self._exch_since = None

        # ---- drawdown vs kill limit
        limit = cfg.kill_limit(self.mode)
        if eq is not None and peak > 0:
            dd = drawdown_pct(eq, peak)
            if dd >= limit:
                desired = FLATTEN
                self.flatten_reason = f"drawdown {dd:.1f}% >= kill limit {limit:.0f}%"
            elif derated_below_min_notional(cfg, dd, limit, float(self.store.kv_get("min_notional_floor", 5.0) or 5.0)):
                # zombie gap: derating (0 at the limit) makes every trade fall below the exchange minimum while
                # the kill (dd >= limit) has not tripped -> nothing can ever trade again: HALT, manual reset_kill
                desired = max(desired, HALTED, key=_rank)
                self.flatten_reason = "derated below min notional"
                reasons.append(f"derated below min notional (drawdown {dd:.1f}% of {limit:.0f}% limit)")
        st = self.store.kv_get("status", {}) or {}
        if (st.get("kill") or {}).get("tripped"):
            desired = max(desired, HALTED if not db_positions else FLATTEN, key=_rank)
            self.flatten_reason = self.flatten_reason or "engine kill switch tripped"

        # ---- state machine (FLATTEN/HALTED are sticky)
        if sticky:
            desired = self.state if self.state == HALTED else max(self.state, desired, key=_rank)
        if desired == FLATTEN or self.state == FLATTEN:
            reasons.append(self.flatten_reason or "flatten")
            if self.ex is not None and self.mode != "paper_local":
                flat = self._flatten_exchange(self.flatten_reason)
                try:
                    flat = flat and not self.ex.positions()
                except ExchangeError:
                    flat = False
            else:
                if db_positions:
                    self._command_once("kill", "kill", {}, every_s=15.0)
                flat = not db_positions
            desired = HALTED if flat else FLATTEN
        if desired == HALTED:
            if self.halted_since is None:
                self.halted_since = now
            if not reasons:
                reasons.append(self.flatten_reason or "halted: manual reset_kill required")
        self._publish(desired, reasons, now)
        return {"state": self.state, "reasons": reasons}

    def _publish(self, state: str, reasons: list[str], now: float) -> None:
        changed = state != self.state
        prev = self.state
        self.state = state
        self.store.kv_set(KEY_STATE, {"state": state, "reasons": reasons, "t": now, "since": self.halted_since})
        self.store.kv_set(KEY_BEAT, {"t": now, "pid": os.getpid(), "mode": self.mode, "version": __version__})
        sig = (state, tuple(reasons))
        if changed or sig != self._last_published:
            self._last_published = sig
            self.store.add_event("safety", {"state": state, "reasons": reasons, "source": "guard"})
        if changed:
            self._say(logging.WARNING, f"safety state {prev} -> {state}: {'; '.join(reasons) or 'recovered'}")
            if self.alerts and _rank(state) > _rank(prev):
                self.alerts.safety(state, reasons, self.mode)


def _rank(s: str) -> int:
    return (NORMAL, DEGRADED, FLATTEN, HALTED).index(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tb.guard", description="Independent safety monitor")
    ap.add_argument("--mode", choices=["paper_local", "binance_demo", "live"])
    ap.add_argument("--confirm-live", action="store_true")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        mode = resolve_mode(args.mode, live_flag=args.confirm_live)
    except (LiveNotAllowed, ValueError) as e:
        log.error("%s", e)
        return 2
    from .exchange.binance import BinanceExchange
    from .store import Store

    store = Store(db_path())
    ex: Exchange | None = None
    if mode != "paper_local":
        sec = load_secrets()
        if not sec.has_binance:
            log.error("guard: no BINANCE_API_KEY/SECRET in env/.env for mode %s", mode)
            return 2
        ex = BinanceExchange(trading_base_url(mode), sec.api_key, sec.api_secret, mode=mode)
    from .alerts import store_stats
    alerts = AlertDispatcher(stats=store_stats(store))   # every alert carries equity/PnL/drawdown/open
    g = Guard(store, mode, ex, alerts)
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.set())
    log.info("guard started mode=%s", mode)
    while not stop.is_set():
        try:
            g.step()
        except Exception:
            log.exception("guard cycle failed")
        if args.once:
            break
        stop.wait(store.get_settings().guard_interval_s)
    alerts.close(2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
