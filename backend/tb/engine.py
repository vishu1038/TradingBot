"""Trading engine process (``python -m tb.engine``).

asyncio main loop; blocking exchange I/O runs in worker threads and every state-changing trade
operation is serialised by ``self.tlock``. The engine never imports the API and never waits on
it: it writes state/events/heartbeat to SQLite and polls the ``commands`` table.

Safety invariants (docs/ARCHITECTURE.md): exchange-side SL right after entry (else close), every
loop checks that no position lacks a live stop, kill switch on drawdown, correlation cap, and the
safety layer (guard state obeyed before every entry, plausibility checks + debounced faults,
drawdown derating, drift detection, shadow mode).
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import __version__
from .alerts import AlertDispatcher
from .config import (LiveNotAllowed, Settings, db_path, load_secrets, models_dir, resolve_mode, trading_base_url,
                     PROD_BASE)
from .exchange.base import CloseInfo, Exchange, ExchangeError, NetworkError, RateLimitError, opposite, order_side
from .risk import (atr14, correlation_gate, derate_multiplier, drawdown_pct, effective_slots, kill_tripped, levels,
                   positions_without_stop, size_trade, trail_update)
from .safety import (DEGRADED, NORMAL, DriftMonitor, FaultTracker, check_bars, check_tick, merged_safety,
                     slippage_drift)
from .scorer import NullScorer, ModelInfo, Scorer, load_model, reasons_to_dicts, scan_models
from .shadow import Registry, ShadowBook, promotion_check, shadow_stats
from .store import Store
from . import guard as guard_mod

log = logging.getLogger("tb.engine")

BAR_MS = 300_000
CACHE_BARS = 760
FIRST_FETCH = 720
SIGNAL_FRESH_MS = 90_000          # only trade a signal bar that closed < 90 s ago
STOP_CHECK_LIVE_S = 15.0
SYNC_LIVE_S = 5.0
TRAIL_MIN_STEP = 0.0005           # fraction of price a trailing stop must improve by before it is moved


class DbLogHandler(logging.Handler):
    """Mirror INFO+ log records of the ``tb`` logger tree into the logs table and the events feed."""

    def __init__(self, store: Store):
        super().__init__(logging.INFO)
        self.store = store
        self._busy = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            msg = record.getMessage()
            if record.exc_info:
                msg += f" ({record.exc_info[0].__name__}: {record.exc_info[1]})"
            t = time.time()
            self.store.add_log(record.levelname, f"{record.name}: {msg}" if record.name != "tb.engine" else msg, t)
            self.store.add_event("log", {"t": int(t), "level": record.levelname, "msg": msg}, t)
        except Exception:
            pass
        finally:
            self._busy = False


class DerivFeed:
    """Background top-up of metrics/funding frames (archive schema) for the model's derivative features."""

    def __init__(self):
        self.metrics: dict[str, pd.DataFrame] = {}
        self.funding: dict[str, pd.DataFrame] = {}
        self.metrics_at: dict[str, float] = {}
        self.funding_at: dict[str, float] = {}

    def refresh(self, symbol: str) -> None:
        from .data import live

        now = int(time.time() * 1000)
        try:
            m = live.fetch_metrics(symbol, now - 26 * 3600_000, now=now, pace=0.15)
            if len(m):
                self.metrics[symbol] = m
        except Exception as e:
            log.warning("metrics top-up failed for %s: %s", symbol, e)
        self.metrics_at[symbol] = time.time()
        if time.time() - self.funding_at.get(symbol, 0) > 3600:
            try:
                f = live.fetch_funding(symbol, now - 4 * 86_400_000, now)
                if len(f):
                    self.funding[symbol] = f
            except Exception as e:
                log.warning("funding top-up failed for %s: %s", symbol, e)
            self.funding_at[symbol] = time.time()


class Engine:
    def __init__(self, store: Store, data_ex: Exchange, trade_ex: Exchange, mode: str,
                 alerts: AlertDispatcher | None = None, models_path: Any = None,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.data = data_ex
        self.ex = trade_ex
        self.mode = mode
        self.alerts = alerts or AlertDispatcher([])
        self.models_path = models_path or models_dir()
        self.clock = clock
        self.cfg: Settings = store.get_settings()
        self.paper = mode == "paper_local"
        self.stop_event = asyncio.Event()
        self.tlock = asyncio.Lock()
        # market state
        self.kl: dict[str, pd.DataFrame] = {}
        self.last_closed: dict[str, int] = {}
        self.marks: dict[str, float] = {}
        self.rates: dict[str, float] = {}
        self.skew_ms = 0
        self.skew_at = 0.0
        self.universe: list[str] = []
        self.universe_at = 0.0
        self.market_ls: float | None = None
        self.market_ls_at = 0.0
        self.feed = DerivFeed()
        self._lev_set: set[str] = set()
        # account state
        self.equity = 0.0
        self.cash = 0.0
        self.upnl = 0.0
        self.available = 0.0
        self.start_equity = 0.0
        self.peak_equity = 0.0
        self.kill: dict[str, Any] = {"tripped": False, "reason": "", "t": None}
        self.running = self.paper
        self._last_sync = 0.0
        self._last_stop_check = 0.0
        self._last_equity_write = 0.0
        self._last_persist = 0.0
        self._last_day = datetime.now(timezone.utc).date().isoformat()
        # safety layer
        self.faults = FaultTracker(self.cfg.fault_mature_n, self.cfg.fault_recover_m)
        self.drift = DriftMonitor(self.cfg.drift_n)
        self.drift_state = self.drift.update([], [])
        self.slips: deque[float] = deque(maxlen=30)
        self.slip_alarm = False
        self.safety_view: dict[str, Any] = {"state": NORMAL, "reasons": []}
        self._safety_sig: tuple | None = None
        # models / shadow
        self.scorer: Scorer = NullScorer()
        self.model_info: ModelInfo | None = None
        self.challengers: dict[str, tuple[Scorer, ModelInfo]] = {}
        self.registry = Registry.from_dict(store.kv_get("model_registry"))
        self.shadow = ShadowBook(self.cfg)
        self._models_sig: tuple | None = None
        self.stats = {"loops": 0, "errors": 0}

    # ================================================================== helpers
    async def io(self, fn: Callable, *a: Any, **kw: Any) -> Any:
        return await asyncio.to_thread(fn, *a, **kw)

    def now_ms(self) -> int:
        return int(self.clock() * 1000) + self.skew_ms

    def event(self, type_: str, data: Any) -> None:
        try:
            self.store.add_event(type_, data)
        except Exception:
            log.exception("event write failed")

    def watchlist(self) -> list[str]:
        syms = list(self.universe)
        for p in self.store.get_positions():
            if p["symbol"] not in syms:
                syms.append(p["symbol"])
        if "BTCUSDT" not in syms:
            syms.append("BTCUSDT")
        return syms

    def trade_symbols(self) -> set[str]:
        return set(self.universe) | {p["symbol"] for p in self.store.get_positions()}

    # ================================================================== startup
    def startup(self) -> None:
        cfg = self.cfg
        self.skew_ms = 0
        try:
            self.skew_ms = self.data.server_time_ms() - int(self.clock() * 1000)
            self.skew_at = self.clock()
        except ExchangeError as e:
            log.warning("server time unavailable: %s", e)
        if not self.paper:
            self.ex.sync_time() if hasattr(self.ex, "sync_time") else None
            if hasattr(self.ex, "position_mode_hedged") and self.ex.position_mode_hedged():
                raise RuntimeError("account is in hedge (dual-side) position mode; switch to one-way mode")
        # restore persisted state
        st = self.store.kv_get_many(["start_equity", "peak_equity", "kill", "running", "paper_state",
                                     "shadow_open", "universe_at", "live_trades_opened"])
        if self.paper:
            if st.get("paper_state"):
                self.ex.import_state(st["paper_state"])
            else:
                for p in self.store.get_positions():
                    self.ex.adopt(p["symbol"], p["side"], p["qty"], p["entry_price"], p["leverage"],
                                  int(p["opened_at"] * 1000), p.get("entry_fee") or 0.0, p.get("funding") or 0.0)
                    log.warning("paper state missing: re-adopted position %s", p["symbol"])
        self._refresh_account()
        self.start_equity = float(st.get("start_equity") or self.equity)
        self.peak_equity = max(float(st.get("peak_equity") or 0.0), self.equity)
        if st.get("kill"):
            self.kill = st["kill"]
        self.running = bool(st["running"]) if st.get("running") is not None else self.paper
        if st.get("shadow_open"):
            self.shadow.import_state(st["shadow_open"])
        self.store.kv_set("start_equity", self.start_equity)
        self.reload_models(force=True)
        # universe
        rows = self.store.get_universe()
        if rows:
            self.universe = [r["symbol"] for r in rows]
            self.universe_at = float(st.get("universe_at") or 0.0)
        self.reconcile_sync(startup=True)
        self.persist_state(force=True)
        log.info("engine started mode=%s equity=%.2f running=%s model=%s", self.mode, self.equity, self.running,
                 self.scorer.name)

    # ================================================================== models / shadow
    def reload_models(self, force: bool = False) -> None:
        try:
            files = sorted((p.name, p.stat().st_mtime) for p in self.models_path.glob("*.json"))
        except Exception:
            files = []
        sig = (tuple(files), tuple(sorted(self.registry.overrides.items())))
        if sig == self._models_sig and not force:
            return
        self._models_sig = sig
        entries = scan_models(self.models_path, self.registry.overrides)
        champ = next((e for e in entries if e.status == "champion"), None)
        if champ and (self.model_info is None or self.model_info.name != champ.name or force):
            got = load_model(champ)
            if got:
                self.scorer, self.model_info = got
                log.info("champion model: %s v%s", champ.name, self.model_info.version)
            else:
                self.scorer, self.model_info = NullScorer(), None
        elif not champ:
            self.scorer, self.model_info = NullScorer(), None
        wanted = {e.name: e for e in entries if e.status == "challenger"}
        for n in list(self.challengers):
            if n not in wanted:
                del self.challengers[n]
        for n, e in wanted.items():
            if n not in self.challengers:
                got = load_model(e)
                if got:
                    self.challengers[n] = got
                    log.info("challenger model loaded (shadow mode): %s", n)

    def model_status_list(self) -> list[tuple[str, str]]:
        return [(e.name, e.status) for e in scan_models(self.models_path, self.registry.overrides)]

    def handle_promote(self, model: str | None) -> str | None:
        """Returns an error string or None."""
        if not model:
            return "model is required"
        entries = {e.name: e for e in scan_models(self.models_path, self.registry.overrides)}
        e = entries.get(model)
        if not e:
            return f"unknown model {model!r}"
        if e.status == "champion":
            return "already the champion"
        champ = self.model_info.name if self.model_info else None
        chal_trades = self.store.get_shadow_trades(model)
        since = min((t["entry_time"] for t in chal_trades), default=int(self.clock()))
        champ_trades = self.store.get_shadow_trades(champ, since) if champ else []
        ok, why = promotion_check(shadow_stats(chal_trades), shadow_stats(champ_trades), self.drift.level,
                                  self.cfg.shadow_min_trades)
        if not ok:
            return f"promotion refused: {why}"
        self.registry = self.registry.promote(model, champ)
        self.store.kv_set("model_registry", self.registry.to_dict())
        self.reload_models(force=True)
        msg = f"model promoted: {model} (previous champion {champ})"
        log.warning(msg)
        self.alerts.send(f"[{self.mode}] {msg}")
        return None

    def handle_rollback(self, model: str | None) -> str | None:
        champ = self.model_info.name if self.model_info else None
        try:
            if model and self.registry.previous_champion and model != self.registry.previous_champion:
                return f"previous champion is {self.registry.previous_champion!r}, not {model!r}"
            self.registry = self.registry.rollback(champ)
        except ValueError as e:
            return str(e)
        self.store.kv_set("model_registry", self.registry.to_dict())
        self.reload_models(force=True)
        msg = f"model rolled back to {self.model_info.name if self.model_info else 'null'}"
        log.warning(msg)
        self.alerts.send(f"[{self.mode}] {msg}")
        return None

    # ================================================================== account
    def _refresh_account(self) -> None:
        b = self.ex.balance()
        self.cash, self.upnl, self.available = b.wallet, b.upnl, b.available
        self.equity = b.wallet + b.upnl

    def persist_state(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._last_persist < 30:
            return
        self._last_persist = now
        try:
            if self.paper:
                self.store.kv_set("paper_state", self.ex.export_state())
            self.store.kv_set("peak_equity", self.peak_equity)
            self.store.kv_set("kill", self.kill)
            self.store.kv_set("running", self.running)
            self.store.kv_set("shadow_open", self.shadow.export_state())
        except Exception:
            log.exception("persist failed")

    def risk_multiplier(self) -> tuple[float, dict]:
        """derate x drift x slippage x staged-rollout multiplier (+ the parts, for logging)."""
        cfg = self.cfg
        dd = drawdown_pct(self.equity, self.peak_equity)
        derate = derate_multiplier(dd, cfg.kill_limit(self.mode))
        drift = self.drift_state.mult
        slip = 0.5 if self.slip_alarm else 1.0
        stage = 1.0
        if self.mode == "live" and int(self.store.kv_get("live_trades_opened", 0) or 0) < cfg.live_stage_trades:
            stage = cfg.live_stage_mult
        parts = {"derate": derate, "drift": drift, "slippage": slip, "stage": stage, "dd_pct": dd}
        return derate * drift * slip * stage, parts

    def compute_safety(self) -> dict:
        cfg = self.cfg
        guard = self.store.kv_get(guard_mod.KEY_STATE)
        beat = self.store.kv_get(guard_mod.KEY_BEAT)
        age = (self.clock() - float(beat["t"])) if beat else None
        eng_state, reasons = NORMAL, []
        if self.drift.entries_blocked():
            eng_state = DEGRADED
            reasons.append(f"drift alarm (z={self.drift_state.z:.2f})" if self.drift_state.z is not None else "drift alarm")
        if self.faults.exchange_fault():
            eng_state = DEGRADED
            reasons.append("exchange-wide data fault")
        mult, parts = self.risk_multiplier()
        engine_view = {"state": eng_state, "reasons": reasons, "derate_mult": mult,
                       "drift": {"z": self.drift_state.z, "n": self.drift_state.n, "mult": self.drift_state.mult},
                       "faults": self.faults.faults()}
        view = merged_safety(guard, engine_view, age, guard_required=not self.paper, guard_stale_s=60.0)
        view["parts"] = parts
        return view

    # ================================================================== reconcile / closing
    def _db_positions(self) -> list[dict]:
        return self.store.get_positions()

    def reconcile_sync(self, startup: bool = False) -> None:
        """Blocking reconcile used at startup (before the loop runs)."""
        xpos = {p.symbol: p for p in self.ex.positions()}
        for pos in self._db_positions():
            if pos["symbol"] not in xpos:
                info = self.ex.last_close(pos["symbol"], pos["side"], int(pos["opened_at"] * 1000),
                                          pos.get("sl_order_id"), pos.get("tp_order_id"), pos["entry_price"])
                if info is None:
                    px = self.marks.get(pos["symbol"]) or pos["entry_price"]
                    d = 1 if pos["side"] == "long" else -1
                    info = CloseInfo(px, (px - pos["entry_price"]) * pos["qty"] * d, 0.0, 0.0, "manual", int(self.clock()))
                log.warning("reconcile: %s no longer open on the exchange; recording close (%s)", pos["symbol"], info.reason)
                self._finalize(pos, info, info.reason)
        known = {p["symbol"] for p in self._db_positions()}
        for sym, xp in xpos.items():
            if sym in known:
                continue
            log.error("reconcile: exchange position %s %s qty=%s has no record; adopting", sym, xp.side, xp.qty)
            orders = self.ex.open_conditional_orders(sym)
            sl = next((o for o in orders if o.kind == "SL"), None)
            tp = next((o for o in orders if o.kind == "TP"), None)
            if sl is None:
                log.error("reconcile: %s has no stop; closing it", sym)
                try:
                    self.ex.market_order(sym, opposite(order_side(xp.side)), xp.qty, reduce_only=True)
                    self.ex.cancel_all(sym)
                except ExchangeError as e:
                    log.error("reconcile close failed for %s: %s", sym, e)
                continue
            self.store.insert_position({
                "symbol": sym, "side": xp.side, "entry_price": xp.entry_price, "qty": xp.qty,
                "notional": xp.qty * xp.entry_price, "leverage": xp.leverage, "margin": xp.margin or xp.qty * xp.entry_price / xp.leverage,
                "sl_price": sl.trigger_price, "tp_price": tp.trigger_price if tp else None,
                "init_sl_price": sl.trigger_price, "atr": None, "trail_active": False, "best_price": xp.mark_price or xp.entry_price,
                "mark_price": xp.mark_price, "upnl": xp.upnl, "opened_at": int(self.clock()), "prob": None, "reasons": [],
                "sl_order_id": sl.order_id, "tp_order_id": tp.order_id if tp else None, "exit_mode": self.cfg.exit_mode})

    def _finalize(self, pos: dict, info: CloseInfo, reason: str) -> None:
        """Persist a closed position as a trade (sync; no exchange calls)."""
        if reason == "sl" and pos.get("trail_active"):
            reason = "trail"
        t = int(info.closed_at or self.clock())
        net = info.pnl - info.fees + info.funding
        trade = {"position_id": pos["id"], "symbol": pos["symbol"], "side": pos["side"],
                 "entry_price": pos["entry_price"], "exit_price": info.exit_price, "qty": pos["qty"],
                 "opened_at": pos["opened_at"], "closed_at": t, "pnl": info.pnl, "fees": info.fees,
                 "funding": info.funding, "exit_reason": reason, "prob": pos.get("prob")}
        tid = self.store.insert_trade(trade)
        self.store.delete_position(pos["id"])
        self.event("position_closed", {**trade, "id": tid, "net_pnl": net})
        log.info("closed %s %s %s: pnl %+.3f fees %.3f funding %+.3f (%s)", pos["symbol"], pos["side"], pos["qty"],
                 info.pnl, info.fees, info.funding, reason)
        self.alerts.trade_closed(pos["symbol"], pos["side"], pos["entry_price"], info.exit_price, info.pnl,
                                 info.fees - info.funding, reason, self.mode)
        self._update_drift()
        self.persist_state(force=True)

    def _update_drift(self) -> None:
        rows = self.store.query("SELECT prob, pnl, fees, funding FROM trades WHERE prob IS NOT NULL "
                                "ORDER BY id DESC LIMIT ?", (self.cfg.drift_n,))
        rows = list(reversed(rows))
        probs = [r["prob"] for r in rows]
        wins = [(r["pnl"] - r["fees"] + r["funding"]) > 0 for r in rows]
        prev = self.drift.level
        self.drift_state = self.drift.update(probs, wins)
        if self.drift.level != prev:
            msg = f"drift level {prev} -> {self.drift.level} (z={self.drift_state.z}, n={self.drift_state.n})"
            log.warning(msg)
            if self.drift.level > prev:
                self.alerts.send(f"[{self.mode}] {msg}", dedupe_key=f"drift{self.drift.level}")

    async def close_position(self, pos: dict, reason: str) -> bool:
        """Close at market (reduce-only), then clear leftover orders. Caller holds tlock."""
        sym = pos["symbol"]
        try:
            res = await self.io(self.ex.market_order, sym, opposite(order_side(pos["side"])), pos["qty"], True)
        except ExchangeError as e:
            if e.code in (-2022, -4118) or "ReduceOnly" in str(e):     # already flat on the exchange
                log.warning("close %s: nothing to reduce (%s); reconciling", sym, e)
                return False
            log.error("close %s failed: %s", sym, e)
            return False
        info = None
        for _ in range(4):
            try:
                info = await self.io(self.ex.last_close, sym, pos["side"], int(pos["opened_at"] * 1000),
                                     pos.get("sl_order_id"), pos.get("tp_order_id"), pos["entry_price"])
            except ExchangeError as e:
                log.warning("last_close lookup failed for %s: %s", sym, e)
            if info is not None:
                break
            await asyncio.sleep(0.5)
        if info is None:
            d = 1 if pos["side"] == "long" else -1
            px = res.avg_price or self.marks.get(sym) or pos["entry_price"]
            info = CloseInfo(px, (px - pos["entry_price"]) * pos["qty"] * d,
                             (pos.get("entry_fee") or pos["qty"] * pos["entry_price"] * self.cfg.fee_taker)
                             + pos["qty"] * px * self.cfg.fee_taker, pos.get("funding") or 0.0, reason, int(self.clock()))
        try:
            await self.io(self.ex.cancel_all, sym)
        except ExchangeError as e:
            log.warning("cancel_all %s failed: %s", sym, e)
        self._finalize(pos, info, reason)
        return True

    async def close_all(self, reason: str) -> int:
        n = 0
        async with self.tlock:
            for pos in self._db_positions():
                n += 1 if await self.close_position(pos, reason) else 0
        return n

    # ================================================================== opening
    async def open_position(self, sym: str, side: str, sz, prob: float, reasons: list, atr: float | None,
                            ref_price: float, mult: float) -> bool:
        cfg = self.cfg
        ex = self.ex
        osd = order_side(side)
        if sym not in self._lev_set:
            try:
                await self.io(ex.set_isolated, sym)
                await self.io(ex.set_leverage, sym, cfg.leverage)
                self._lev_set.add(sym)
            except ExchangeError as e:
                log.error("cannot set isolated/leverage for %s: %s; skipping entry", sym, e)
                return False
        try:
            res = await self.io(ex.market_order, sym, osd, sz.qty, False)
        except ExchangeError as e:
            log.error("entry order failed for %s: %s", sym, e)
            return False
        fill = res.avg_price or ref_price
        qty = res.qty or sz.qty
        f = await self.io(ex.filters, sym)
        lv = levels(cfg, side, fill, qty, atr, f, risk_mult=mult)
        close_side = opposite(osd)
        sl_id = tp_id = None
        sl = tp = None
        if lv is not None:
            sl, tp = lv
            try:
                sl_id = await self.io(ex.place_stop, sym, close_side, sl, qty)
            except ExchangeError as e:
                log.error("SL placement FAILED for %s (%s); closing the position at market", sym, e)
                self.alerts.engine_error("sl_placement", f"{sym}: {e}")
        if sl_id is None:
            try:
                await self.io(ex.market_order, sym, close_side, qty, True)
                await self.io(ex.cancel_all, sym)
            except ExchangeError as e:
                log.error("EMERGENCY CLOSE of unprotected %s failed: %s", sym, e)
                self.alerts.engine_error("emergency_close", f"{sym}: {e}")
                self.kill = {"tripped": True, "reason": f"unprotected position {sym} could not be closed", "t": int(self.clock())}
            else:
                # record the flat round trip so fees are not lost
                info = None
                try:
                    info = await self.io(ex.last_close, sym, side, int(self.clock() * 1000) - 60_000)
                except ExchangeError:
                    pass
                if info:
                    t = int(self.clock())
                    self.store.insert_trade({"symbol": sym, "side": side, "entry_price": fill, "exit_price": info.exit_price,
                                             "qty": qty, "opened_at": t, "closed_at": t, "pnl": info.pnl, "fees": info.fees,
                                             "funding": info.funding, "exit_reason": "manual", "prob": prob})
            return False
        try:
            tp_id = await self.io(ex.place_take_profit, sym, close_side, tp, qty)
        except ExchangeError as e:
            log.warning("TP placement failed for %s (%s); the engine will close at TP itself", sym, e)
        pos = {"symbol": sym, "side": side, "entry_price": fill, "qty": qty, "notional": qty * fill,
               "leverage": cfg.leverage, "margin": qty * fill / cfg.leverage, "sl_price": sl, "tp_price": tp,
               "init_sl_price": sl, "atr": atr, "trail_active": False, "trail_price": None, "best_price": fill,
               "mark_price": fill, "upnl": 0.0, "opened_at": int(self.clock()), "prob": prob,
               "reasons": reasons, "sl_order_id": sl_id, "tp_order_id": tp_id, "entry_fee": res.fee or 0.0,
               "funding": 0.0, "exit_mode": cfg.exit_mode,
               "liq_price": next((p.liq_price for p in await self.io(ex.positions) if p.symbol == sym), None)}
        pid = self.store.insert_position(pos)
        slip_pct = (fill - ref_price) / ref_price * 100.0 * (1 if side == "long" else -1)
        self.slips.append(slip_pct)
        was = self.slip_alarm
        self.slip_alarm = slippage_drift(list(self.slips), cfg.slippage_pct * 100.0, cfg.slippage_drift_mult)
        if self.slip_alarm and not was:
            msg = f"entry slippage drift: mean {np.mean(self.slips):.3f}% vs assumed {cfg.slippage_pct * 100:.3f}%"
            log.warning(msg)
            self.alerts.send(f"[{self.mode}] {msg}", dedupe_key="slipdrift")
        elif was and not self.slip_alarm:
            log.info("entry slippage back within %.1fx the assumption", cfg.slippage_drift_mult)
        if self.mode == "live":
            self.store.kv_set("live_trades_opened", int(self.store.kv_get("live_trades_opened", 0) or 0) + 1)
        self.event("position_opened", {**pos, "id": pid})
        log.info("opened %s %s qty=%s @ %s SL %s TP %s prob=%.3f mult=%.2f", side, sym, qty, fill, sl, tp, prob, mult)
        self.alerts.trade_opened(sym, side, fill, qty, qty * fill, sl, tp or 0.0, prob, self.mode)
        self.persist_state(force=True)
        await self.io(self._refresh_account)
        return True

    async def try_open(self, sym: str, side: str, prob: float, reasons: list, closed: pd.DataFrame) -> None:
        async with self.tlock:
            cfg = self.cfg
            if not self.entry_allowed(sym):
                return
            positions = self._db_positions()
            if any(p["symbol"] == sym for p in positions):
                return
            if len(positions) >= effective_slots(cfg, len(positions), self.available):
                return
            if self.available < cfg.slice_margin_usd * 0.5:
                log.info("skip %s: not enough available margin (%.2f)", sym, self.available)
                return
            atr = atr14(closed["high"].to_numpy(), closed["low"].to_numpy(), closed["close"].to_numpy())
            ref = self.marks.get(sym) or float(closed["close"].iloc[-1])
            # correlation cap
            others = []
            for p in positions:
                k = self.kl.get(p["symbol"])
                if k is not None:
                    others.append((p["symbol"], p["side"], k.set_index("open_time")["close"]))
            cand = closed.set_index("open_time")["close"]
            ok, hot = correlation_gate(side, cand, others, cfg.corr_cap)
            if not ok:
                log.info("skip %s %s: correlated with %s", side, sym, hot)
                return
            mult, parts = self.risk_multiplier()
            f = await self.io(self.ex.filters, sym)
            sz = size_trade(cfg, side, ref, atr, f, self.available, risk_mult=mult)
            if not sz.ok:
                log.info("skip %s %s: %s (risk mult %.2f)", side, sym, sz.reason, mult)
                return
            await self.open_position(sym, side, sz, prob, reasons, atr, ref, mult)

    def entry_allowed(self, sym: str | None = None) -> bool:
        if not self.running or self.kill.get("tripped"):
            return False
        if self.safety_view.get("state", NORMAL) != NORMAL:
            return False
        if sym and self.faults.excluded(sym):
            return False
        return True

    # ================================================================== kill switch
    async def trip_kill(self, reason: str) -> None:
        if self.kill.get("tripped"):
            return
        self.kill = {"tripped": True, "reason": reason, "t": int(self.clock())}
        log.error("KILL SWITCH: %s", reason)
        self.event("kill", {**self.kill, "dd_limit_pct": self.cfg.kill_limit(self.mode)})
        self.alerts.kill(reason, self.equity, drawdown_pct(self.equity, self.peak_equity), self.mode)
        self.persist_state(force=True)
        await self.close_all("kill")
        await self.io(self._refresh_account)
        self.write_status()

    # ================================================================== commands
    async def commands_cycle(self) -> None:
        for cmd in self.store.pending_commands():
            err = None
            try:
                err = await self.handle_command(cmd["action"], cmd["payload"])
            except Exception as e:
                log.exception("command %s failed", cmd["action"])
                err = f"{type(e).__name__}: {e}"
            self.store.finish_command(cmd["id"], err is None, err)

    async def handle_command(self, action: str, payload: dict) -> str | None:
        log.info("command: %s %s", action, payload or "")
        if action == "start":
            if self.kill.get("tripped"):
                return "kill switch tripped: reset_kill first"
            self.running = True
        elif action == "stop":
            self.running = False
        elif action == "kill":
            await self.trip_kill("manual kill")
        elif action == "reset_kill":
            self.kill = {"tripped": False, "reason": "", "t": None}
            self.peak_equity = self.equity
            self.persist_state(force=True)
            self.write_status()
            self.store.kv_set(guard_mod.KEY_RESET, self.clock())      # clears the guard's HALTED
            self.event("kill", {"tripped": False, "reason": "", "dd_limit_pct": self.cfg.kill_limit(self.mode)})
        elif action == "close_position":
            async with self.tlock:
                pos = self.store.get_position(int(payload.get("position_id", -1)))
                if not pos:
                    return "no such position"
                if not await self.close_position(pos, "manual"):
                    return "close failed (see logs)"
        elif action == "close_all":
            await self.close_all("manual")
        elif action == "promote_model":
            return await self.io(self.handle_promote, payload.get("model"))
        elif action == "rollback_model":
            return await self.io(self.handle_rollback, payload.get("model"))
        else:
            return f"unknown action {action!r}"
        self.persist_state(force=True)
        self.write_status()
        return None

    # ================================================================== prices / positions
    async def prices_cycle(self) -> None:
        cfg = self.cfg
        try:
            prem = await self.io(self.data.premium)
        except (NetworkError, RateLimitError):
            self.faults.observe_exchange(True)
            raise
        self.faults.observe_exchange(False)
        marks = {p.symbol: p.mark_price for p in prem if p.mark_price > 0}
        self.rates = {p.symbol: p.funding_rate for p in prem}
        for s in self.watchlist():
            if s in marks:
                bad = check_tick(marks[s], None)
                if bad:
                    log.warning("implausible mark for %s: %s", s, bad)
                    marks.pop(s)
        self.marks.update(marks)
        watch = {s: marks[s] for s in self.watchlist() if s in marks}
        if self.paper:
            self.ex.tick(watch)
            if self.ex.funding_due(self.clock()):
                paid = self.ex.apply_funding(self.rates, self.clock())
                if paid:
                    log.info("funding applied: %s", {k: round(v, 4) for k, v in paid.items()})
                    for p in self._db_positions():
                        if p["symbol"] in paid:
                            self.store.update_position(p["id"], funding=(p.get("funding") or 0.0) + paid[p["symbol"]])
        # shadow positions follow the same ticks
        if self.shadow.pos:
            for s, m in watch.items():
                for tr in self.shadow.on_price(s, m, self.clock()):
                    self._record_shadow(tr)
        async with self.tlock:
            await self.sync_positions(watch)
        # universe table + ticks
        self.store.update_universe_prices({s: m for s, m in marks.items() if s in self.universe})
        t = int(self.clock())
        for s, m in watch.items():
            self.event("tick", {"symbol": s, "t": t, "price": m})
        # account + kill switch
        await self.io(self._refresh_account)
        self.peak_equity = max(self.peak_equity, self.equity)
        limit = cfg.kill_limit(self.mode)
        if not self.kill.get("tripped") and kill_tripped(self.equity, self.peak_equity, limit):
            await self.trip_kill(f"drawdown {drawdown_pct(self.equity, self.peak_equity):.1f}% >= {limit:.0f}% of peak")
        self.stats["loops"] += 1

    async def sync_positions(self, marks: dict[str, float]) -> None:
        """Reconcile with the exchange, manage stops/trailing/TP, enforce the stop invariant. Holds tlock."""
        cfg = self.cfg
        now = self.clock()
        db = self._db_positions()
        if not db and self.paper:
            return
        live_sync = self.paper or now - self._last_sync >= SYNC_LIVE_S
        xpos = None
        if live_sync:
            self._last_sync = now
            try:
                xpos = {p.symbol: p for p in await self.io(self.ex.positions)}
            except ExchangeError as e:
                log.warning("positions sync failed: %s", e)
                return
            # 1) positions that vanished: closed by an exchange-side order
            for pos in db:
                if pos["symbol"] not in xpos:
                    info = None
                    try:
                        info = await self.io(self.ex.last_close, pos["symbol"], pos["side"], int(pos["opened_at"] * 1000),
                                             pos.get("sl_order_id"), pos.get("tp_order_id"), pos["entry_price"])
                    except ExchangeError as e:
                        log.warning("last_close failed for %s: %s", pos["symbol"], e)
                    if info is None:
                        m = marks.get(pos["symbol"]) or pos["entry_price"]
                        d = 1 if pos["side"] == "long" else -1
                        hit_sl = (m - pos["sl_price"]) * d <= 0
                        info = CloseInfo(pos["sl_price"] if hit_sl else (pos["tp_price"] or m),
                                         (((pos["sl_price"] if hit_sl else (pos["tp_price"] or m)) - pos["entry_price"]) * pos["qty"] * d),
                                         pos["qty"] * m * cfg.fee_taker * 2, 0.0, "sl" if hit_sl else "tp", int(now))
                    try:
                        await self.io(self.ex.cancel_all, pos["symbol"])
                    except ExchangeError:
                        pass
                    self._finalize(pos, info, info.reason)
            db = self._db_positions()
            known = {p["symbol"] for p in db}
            for sym, xp in xpos.items():
                if sym not in known and not self.paper:
                    log.error("untracked exchange position %s; reconciling", sym)
                    await self.io(self.reconcile_sync)
                    db = self._db_positions()
                    break
        # 2) per-position management
        for pos in db:
            sym = pos["symbol"]
            m = marks.get(sym)
            if not m:
                continue
            d = 1 if pos["side"] == "long" else -1
            best = max(pos.get("best_price") or m, m) if d == 1 else min(pos.get("best_price") or m, m)
            upnl = (m - pos["entry_price"]) * pos["qty"] * d
            upd: dict[str, Any] = {"mark_price": m, "upnl": upnl, "best_price": best}
            # fallback exits when the exchange-side order is missing / slow
            if pos.get("tp_price") and not pos.get("tp_order_id") and (m - pos["tp_price"]) * d >= 0:
                await self.close_position({**pos, **upd}, "tp")
                continue
            if not self.paper and (m - pos["sl_price"]) * d < -0.003 * pos["entry_price"]:
                log.error("%s is beyond its stop by >0.3%% and still open; closing at market", sym)
                await self.close_position({**pos, **upd}, "sl")
                continue
            # trailing stop
            if cfg.trail.enabled:
                atr = atr14(*self._hlc(sym)) or pos.get("atr")
                active, new_sl = trail_update(pos["side"], pos["entry_price"], pos["init_sl_price"], pos["sl_price"], best,
                                              atr, bool(pos["trail_active"]), cfg.trail.activate_r, cfg.trail.distance_atr)
                f = await self.io(self.ex.filters, sym)
                new_sl = f.round_price(new_sl)
                if active and (new_sl - pos["sl_price"]) * d > max(f.tick_size, TRAIL_MIN_STEP * pos["entry_price"]):
                    if (new_sl - m) * d >= 0:           # price already through the new stop: exit now
                        await self.close_position({**pos, **upd, "trail_active": True}, "trail")
                        continue
                    try:
                        new_id = await self.io(self.ex.place_stop, sym, opposite(order_side(pos["side"])), new_sl, pos["qty"])
                        old = pos.get("sl_order_id")
                        if old:
                            await self.io(self.ex.cancel_order, sym, old)
                        upd.update(sl_price=new_sl, sl_order_id=new_id, trail_active=True, trail_price=new_sl)
                        self.event("position_updated", {"id": pos["id"], "symbol": sym, "sl_price": new_sl, "trail_active": True,
                                                        "trail_price": new_sl})
                    except ExchangeError as e:
                        log.warning("trail move failed for %s: %s (old stop kept)", sym, e)
                elif active and not pos["trail_active"]:
                    upd.update(trail_active=True)
            self.store.update_position(pos["id"], **upd)
        # 3) invariant: no open position without a live stop
        if self.paper or now - self._last_stop_check >= STOP_CHECK_LIVE_S:
            self._last_stop_check = now
            db = self._db_positions()
            if db:
                try:
                    orders = await self.io(self.ex.open_conditional_orders)
                    xnow = {p.symbol for p in await self.io(self.ex.positions)}
                except ExchangeError as e:
                    log.warning("stop check skipped: %s", e)
                    return
                for sym in positions_without_stop([p for p in db if p["symbol"] in xnow], orders):
                    pos = next(p for p in db if p["symbol"] == sym)
                    log.error("INVARIANT: %s has no live stop; closing it", sym)
                    self.alerts.engine_error("stop_invariant", f"{sym} had no live stop; closed")
                    await self.close_position(pos, "manual")

    def _hlc(self, sym: str):
        k = self.kl.get(sym)
        if k is None or len(k) < 16:
            return [], [], []
        k = k[k["open_time"] + BAR_MS <= self.now_ms()]
        return k["high"].to_numpy(), k["low"].to_numpy(), k["close"].to_numpy()

    # ================================================================== bars / signals
    async def bars_cycle(self) -> None:
        watch = self.watchlist()
        if self.clock() - self.skew_at > 300:
            try:
                self.skew_ms = await self.io(self.data.server_time_ms) - int(self.clock() * 1000)
                self.skew_at = self.clock()
            except ExchangeError:
                pass
        errs = 0
        for sym in watch:
            try:
                await self.update_symbol(sym)
            except RateLimitError:
                raise
            except ExchangeError as e:
                errs += 1
                log.warning("klines %s failed: %s", sym, e)
            except Exception:
                self.stats["errors"] += 1
                log.exception("bar processing failed for %s", sym)
        if watch:
            self.faults.observe_exchange(errs >= max(1, len(watch) // 2))

    async def update_symbol(self, sym: str) -> None:
        cfg = self.cfg
        cache = self.kl.get(sym)
        limit = FIRST_FETCH if cache is None else 6
        df = await self.io(self.data.klines, sym, "5m", limit)
        if df.empty:
            return
        if cache is not None and int(df["open_time"].iloc[0]) > int(cache["open_time"].iloc[-1]) + BAR_MS:
            df = await self.io(self.data.klines, sym, "5m", FIRST_FETCH)      # gap: refetch the whole window
            cache = None
        if cache is None:
            merged = df
        else:
            merged = pd.concat([cache[cache["open_time"] < df["open_time"].iloc[0]], df], ignore_index=True)
        merged = merged.tail(CACHE_BARS).reset_index(drop=True)
        self.kl[sym] = merged
        now_ms = self.now_ms()
        # candles -> store (+ live event)
        first = cache is None
        rows = merged if first else merged.tail(3)
        self.store.upsert_candles(sym, "5m", [(r.open_time // 1000, r.open, r.high, r.low, r.close, r.volume)
                                              for r in rows.itertuples()])
        last = merged.iloc[-1]
        self.event("candle", {"symbol": sym, "tf": "5m", "candle": {"t": int(last.open_time // 1000), "o": last.open, "h": last.high,
                                                                    "l": last.low, "c": last.close, "v": last.volume}})
        if first:
            self.store.prune_candles(sym, "5m", 1500)
        # plausibility
        closed_all = merged[merged["open_time"] + BAR_MS <= now_ms]
        atr = atr14(closed_all["high"].to_numpy(), closed_all["low"].to_numpy(), closed_all["close"].to_numpy())
        bad = check_bars(merged["open_time"].to_numpy(), merged["open"].to_numpy(), merged["high"].to_numpy(),
                         merged["low"].to_numpy(), merged["close"].to_numpy(), merged["volume"].to_numpy(), atr, now_ms,
                         max_ret_pct=cfg.max_bar_return_pct, max_ret_atr=cfg.max_bar_return_atr)
        mark = self.marks.get(sym)
        bad += check_tick(mark, float(last.close), cfg.max_mark_last_diff_pct)
        warn_before = {k for (s, k, _) in self.faults.warnings() if s == sym}
        self.faults.observe_symbol(sym, bad, ("bad_price", "high_low", "zero_volume", "big_move", "stale", "mark_last_diff"),
                                   self.clock())
        if bad:
            if set(bad) - warn_before:
                log.warning("data plausibility fault on %s: %s", sym, ",".join(bad))
            if self.faults.excluded(sym):
                log.warning("%s excluded from entries (fault matured: %s)", sym, ",".join(bad))
            return                      # reject the bar before use
        if closed_all.empty:
            return
        t_last = int(closed_all["open_time"].iloc[-1])
        if t_last <= self.last_closed.get(sym, 0):
            return
        self.last_closed[sym] = t_last
        fresh = now_ms - (t_last + BAR_MS) < SIGNAL_FRESH_MS
        await self.on_new_bar(sym, closed_all, fresh)

    def build_window(self, sym: str, closed: pd.DataFrame) -> np.ndarray | None:
        try:
            from . import features
        except Exception as e:
            log.error("tb.features unavailable (%s); no scoring", e)
            return None
        btc = self.kl.get("BTCUSDT")
        btc_closed = btc[btc["open_time"] + BAR_MS <= self.now_ms()] if btc is not None else closed
        if len(closed) < features.WARMUP + features.WINDOW:
            log.warning("%s: only %d bars of history (need %d); skipping score", sym, len(closed), features.WARMUP + features.WINDOW)
            return None
        feats = features.build_features(closed.reset_index(drop=True), self.feed.metrics.get(sym), self.feed.funding.get(sym),
                                        btc_closed.reset_index(drop=True))
        w = np.asarray(feats[-features.WINDOW:], dtype="float32")
        return np.nan_to_num(w)

    async def on_new_bar(self, sym: str, closed: pd.DataFrame, fresh: bool) -> None:
        cfg = self.cfg
        bar = closed.iloc[-1]
        # paper: replay the completed bar's wicks against simulated stops, shadow too
        if self.paper and any(p["symbol"] == sym for p in self._db_positions()):
            self.ex.check_range(sym, float(bar.low), float(bar.high), int(bar.open_time))
        if self.shadow.pos:
            for tr in self.shadow.on_bar(sym, int(bar.open_time), float(bar.low), float(bar.high), self.clock()):
                self._record_shadow(tr)
        if sym not in self.trade_symbols():
            return
        active = not isinstance(self.scorer, NullScorer)
        prob_l = prob_s = 0.0
        reasons: list = []
        window = None
        if active or self.challengers:
            window = await self.io(self.build_window, sym, closed)
        if window is not None and active:
            sc = await self.io(self.scorer.score, sym, window)
            prob_l, prob_s, reasons = float(sc.prob_long), float(sc.prob_short), reasons_to_dicts(sc.reasons)
        action = "none"
        cand_l = prob_l >= cfg.prob_threshold_long
        cand_s = cfg.short_enabled and prob_s >= cfg.prob_threshold_short
        if cand_l and (not cand_s or prob_l >= prob_s):
            action = "long"
        elif cand_s:
            action = "short"
        t = int(bar.open_time // 1000) + 300
        self.store.add_signal(t, sym, prob_l, prob_s, action, reasons)
        self.event("signal", {"t": t, "symbol": sym, "prob_long": prob_l, "prob_short": prob_s, "action": action,
                              "reasons": reasons})
        # shadow mode: every model (champion + challengers) takes its would-be trades
        if window is not None and self.challengers:
            await self.shadow_signals(sym, closed, window, prob_l, prob_s)
        if action != "none":
            if fresh:
                await self.try_open(sym, action, prob_l if action == "long" else prob_s, reasons, closed)
            else:
                log.info("signal %s %s ignored: bar not fresh", action, sym)

    async def shadow_signals(self, sym: str, closed: pd.DataFrame, window: np.ndarray, champ_l: float, champ_s: float) -> None:
        cfg = self.cfg
        f = await self.io(self.ex.filters, sym)
        atr = atr14(closed["high"].to_numpy(), closed["low"].to_numpy(), closed["close"].to_numpy())
        price = self.marks.get(sym) or float(closed["close"].iloc[-1])
        models: list[tuple[str, float, float]] = []
        if self.model_info:
            models.append((self.model_info.name, champ_l, champ_s))
        for name, (sc, _) in self.challengers.items():
            s = await self.io(sc.score, sym, window)
            models.append((name, float(s.prob_long), float(s.prob_short)))
        for name, pl, ps in models:
            side = None
            if pl >= cfg.prob_threshold_long and (not (cfg.short_enabled and ps >= cfg.prob_threshold_short) or pl >= ps):
                side, prob = "long", pl
            elif cfg.short_enabled and ps >= cfg.prob_threshold_short:
                side, prob = "short", ps
            if side:
                self.shadow.open(name, sym, side, prob, price, self.clock(), atr, f)

    def _record_shadow(self, tr: dict) -> None:
        self.store.add_shadow_trade(tr)
        self.event("shadow_trade", tr)
        log.info("shadow %s %s %s pnl %+.3f (%s)", tr["model"], tr["side"], tr["symbol"], tr["pnl"], tr["exit_reason"])

    # ================================================================== universe
    async def universe_cycle(self) -> None:
        from . import universe as uni

        cfg = self.cfg
        if self.universe and self.clock() - self.universe_at < cfg.universe_refresh_s:
            return
        cur = {r["symbol"]: r["in_universe_since"] for r in self.store.get_universe()}
        rows = await self.io(uni.refresh, self.data, cfg, cur, self.clock())
        if not rows:
            log.warning("universe refresh returned no symbols; keeping the old list")
            self.universe_at = self.clock() - cfg.universe_refresh_s + 120
            return
        self.store.replace_universe(rows)
        self.universe = [r["symbol"] for r in rows]
        self.universe_at = self.clock()
        self.store.kv_set("universe_at", self.universe_at)
        self.event("universe", {"symbols": self.universe, "updated_at": int(self.universe_at)})
        log.info("universe: %s", ", ".join(self.universe))

    async def derivs_cycle(self) -> None:
        if isinstance(self.scorer, NullScorer) and not self.challengers:
            return
        due = [s for s in self.trade_symbols() if self.clock() - self.feed.metrics_at.get(s, 0) > 300]
        if due:
            await self.io(self.feed.refresh, due[0])

    # ================================================================== status / housekeeping
    def market_snapshot(self) -> dict:
        btc = self.kl.get("BTCUSDT")
        out: dict[str, Any] = {"btc_price": self.marks.get("BTCUSDT"), "btc_change_24h_pct": None, "btc_trend": "flat",
                               "market_volatility_pct": None, "avg_funding_rate": None, "long_short_ratio": self.market_ls,
                               "updated_at": int(self.clock())}
        if btc is not None and len(btc) > 60:
            c = btc["close"]
            if len(c) > 288:
                out["btc_change_24h_pct"] = float((c.iloc[-1] / c.iloc[-289] - 1) * 100)
            e20, e50 = c.ewm(span=20).mean().iloc[-1], c.ewm(span=50).mean().iloc[-1]
            out["btc_trend"] = "up" if e20 > e50 * 1.0005 else ("down" if e20 < e50 * 0.9995 else "flat")
            if out["btc_price"] is None:
                out["btc_price"] = float(c.iloc[-1])
        uni = self.store.get_universe()
        atrs = [u["atr_pct"] for u in uni if u.get("atr_pct") is not None]
        fr = [u["funding_rate"] for u in uni if u.get("funding_rate") is not None]
        if atrs:
            out["market_volatility_pct"] = float(np.mean(atrs))
        if fr:
            out["avg_funding_rate"] = float(np.mean(fr))
        return out

    def write_status(self) -> None:
        cfg = self.cfg
        self.store.kv_set("status", {
            "mode": self.mode, "running": self.running, "equity": self.equity, "start_equity": self.start_equity,
            "peak_equity": self.peak_equity, "cash": self.cash, "unrealized_pnl": self.upnl,
            "max_positions": effective_slots(cfg, self.store.one("SELECT COUNT(*) n FROM positions")["n"], self.available),
            "kill": {"tripped": bool(self.kill.get("tripped")), "reason": self.kill.get("reason", ""),
                     "dd_limit_pct": cfg.kill_limit(self.mode)},
            "model": {"name": self.model_info.name if self.model_info else "null",
                      "version": self.model_info.version if self.model_info else "0",
                      "trained_at": self.model_info.trained_at if self.model_info else None,
                      "threshold_long": cfg.prob_threshold_long, "threshold_short": cfg.prob_threshold_short},
            "universe_updated_at": int(self.universe_at) if self.universe_at else None,
            "t": self.clock()})

    async def heartbeat_cycle(self) -> None:
        self.cfg = self.store.get_settings()
        self.faults.deb.n, self.faults.deb.m = self.cfg.fault_mature_n, self.cfg.fault_recover_m
        self.shadow.cfg = self.cfg
        if self.paper:
            self.ex.set_cfg(self.cfg)
        view = self.compute_safety()
        self.safety_view = view
        sig = (view["state"], tuple(view["reasons"]), tuple((f["symbol"], f["kind"]) for f in view["faults"]),
               self.drift_state.level, round(view["derate_mult"], 2))
        if sig != self._safety_sig:
            self._safety_sig = sig
            self.store.kv_set("safety_engine", view)
            self.event("safety", {k: view[k] for k in ("state", "reasons", "derate_mult", "drift", "faults")})
            if view["state"] != NORMAL:
                log.warning("safety state %s: %s", view["state"], "; ".join(view["reasons"]))
        self.write_status()
        self.store.beat(self.mode, __version__, os.getpid(), {
            "running": self.running, "universe": len(self.universe), "loops": self.stats["loops"],
            "errors": self.stats["errors"], "safety": view["state"], "model": self.scorer.name})

    async def housekeeping_cycle(self) -> None:
        now = self.clock()
        self.reload_models()
        if now - self._last_equity_write >= 30:
            self._last_equity_write = now
            self.store.add_equity(int(now), self.equity)
            self.event("equity", {"t": int(now), "equity": self.equity})
        self.event("status", {"equity": self.equity, "running": self.running, "kill": bool(self.kill.get("tripped"))})
        self.store.prune_events(3600)
        self.store.prune_logs(5000)
        self.store.prune_signals(20000)
        self.persist_state()
        self.store.kv_set("market", self.market_snapshot())
        try:     # smallest min-notional among tradable symbols: the guard's "derated below min notional" input
            floors = [(await self.io(self.ex.filters, sy)).min_notional for sy in self.trade_symbols()]
            if floors:
                self.store.kv_set("min_notional_floor", min(floors))
        except ExchangeError as e:
            log.debug("min notional floor unavailable: %s", e)
        if now - self.market_ls_at > 300:
            self.market_ls_at = now
            try:
                self.market_ls = await self.io(self.data.long_short_ratio, "BTCUSDT")
            except ExchangeError as e:
                log.debug("long/short ratio unavailable: %s", e)
        today = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
        if today != self._last_day:
            day0 = int(datetime.fromisoformat(self._last_day).replace(tzinfo=timezone.utc).timestamp())
            s = self.store.trade_stats(day0)
            s_day = self.store.one("SELECT COUNT(*) n, COALESCE(SUM(pnl-fees+funding),0) net, "
                                   "COALESCE(SUM(CASE WHEN pnl-fees+funding>0 THEN 1 ELSE 0 END),0) w FROM trades "
                                   "WHERE closed_at>=? AND closed_at<?", (day0, day0 + 86400))
            self.alerts.daily_summary(date=self._last_day, trades=s_day["n"], wins=s_day["w"], pnl=s_day["net"],
                                      equity=self.equity, mode=self.mode)
            self._last_day = today

    # ================================================================== run
    async def _guarded(self, name: str, interval: float, fn: Callable) -> None:
        backoff = interval
        while not self.stop_event.is_set():
            try:
                await fn()
                backoff = interval
            except RateLimitError as e:
                log.warning("%s: rate limited, sleeping %.0fs", name, e.retry_after)
                backoff = max(interval, e.retry_after)
            except ExchangeError as e:      # transient exchange trouble: no traceback, mild backoff, retry
                log.warning("%s: exchange error: %s", name, e)
                backoff = min(15.0, max(interval, backoff) * 1.5)
            except Exception as e:
                self.stats["errors"] += 1
                backoff = min(60.0, max(interval, backoff) * 2)
                log.exception("%s loop error (retry in %.0fs)", name, backoff)
                self.alerts.engine_error(name, f"{type(e).__name__}: {e}")
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(s, self.stop_event.set)
            except NotImplementedError:
                pass
        await self.io(self.startup)
        tasks = [asyncio.create_task(self._guarded(n, i, f)) for n, i, f in [
            ("heartbeat", 1.0, self.heartbeat_cycle), ("commands", 0.5, self.commands_cycle),
            ("prices", 3.0, self.prices_cycle), ("bars", 4.0, self.bars_cycle),
            ("universe", 5.0, self.universe_cycle), ("housekeeping", 10.0, self.housekeeping_cycle),
            ("derivs", 20.0, self.derivs_cycle)]]
        await self.stop_event.wait()
        log.info("engine stopping (open positions keep their exchange-side stops)")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.persist_state(force=True)
        self.store.beat(self.mode, __version__, os.getpid(), {"stopped": True})


def build_exchanges(mode: str, cfg: Settings):
    from .exchange.binance import BinanceExchange
    from .exchange.paper import PaperExchange

    data = BinanceExchange(PROD_BASE, mode="paper_local")
    if mode == "paper_local":
        return data, PaperExchange(data, cfg)
    sec = load_secrets()
    if not sec.has_binance:
        raise RuntimeError(f"mode {mode} needs BINANCE_API_KEY / BINANCE_API_SECRET in env or .env")
    return data, BinanceExchange(trading_base_url(mode), sec.api_key, sec.api_secret, mode=mode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tb.engine")
    ap.add_argument("--mode", choices=["paper_local", "binance_demo", "live"])
    ap.add_argument("--confirm-live", action="store_true", help="explicit flag required (with ALLOW_LIVE_TRADING=true) for live")
    ap.add_argument("--reset-paper", action="store_true", help="discard saved paper state/equity and start fresh")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    try:
        mode = resolve_mode(args.mode, live_flag=args.confirm_live)
    except (LiveNotAllowed, ValueError) as e:
        log.error("%s", e)
        return 2
    path = db_path()
    lock = open(str(path) + ".engine.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.error("another engine already runs against %s; refusing to start a second one", path)
        return 3
    store = Store(path)
    if args.reset_paper:
        for k in ("paper_state", "start_equity", "peak_equity", "kill", "shadow_open", "live_trades_opened"):
            store.execute("DELETE FROM kv WHERE key=?", (k,))
        store.execute("DELETE FROM positions")
    cfg = store.get_settings()
    if not store.query("SELECT 1 FROM settings LIMIT 1"):
        store.save_settings(cfg)
    try:
        data, trade = build_exchanges(mode, cfg)
    except RuntimeError as e:
        log.error("%s", e)
        return 2
    logging.getLogger("tb").addHandler(DbLogHandler(store))
    alerts = AlertDispatcher()
    eng = Engine(store, data, trade, mode, alerts)
    try:
        asyncio.run(eng.run())
    except KeyboardInterrupt:
        pass
    finally:
        alerts.close(2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
