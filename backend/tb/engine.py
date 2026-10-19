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
from decimal import Decimal
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import __version__
from .alerts import AlertDispatcher
from .config import (LiveNotAllowed, Settings, db_path, load_secrets, models_dir, resolve_mode, trading_base_url,
                     PROD_BASE)
from .entry import EntryCfg, EntryGate, XsecCfg, day_start, is_decision_bar, next_decision_ms, next_rebalance_ms, rebalance_slot
from .exchange.base import CloseInfo, Exchange, ExchangeError, NetworkError, RateLimitError, opposite, order_side
from .exits import SETTINGS_SPEC, ExitSpec, exit_atr, resolve_exit
from .exits import bind as bind_exit
from .protect import (MIN_STEP as PROTECT_MIN_STEP, Protect, breakeven_price, is_loss_exit, next_stop, partial_level,
                      partial_split, tighter)
from .risk import (Book, atr14, check_entry, correlation_gate, dd_reference, derate_multiplier, drawdown_pct, kill_tripped, levels,
                   maintenance_margin, positions_without_stop, size_trade, sl_loss, slots_fit)
from .safety import (DEGRADED, NORMAL, DriftMonitor, FaultTracker, check_bars, check_tick, merged_safety,
                     slippage_drift)
from .scorer import (KIND_PER_SYMBOL, FrameBundle, ModelInfo, NullScorer, Score, Scorer, is_paper_only, load_model,
                     per_symbol_window, reasons_to_dicts, scan_models, score_frames)
from .shadow import Registry, ShadowBook, promotion_check, shadow_stats
from .store import Store
from . import capital as cap
from . import guard as guard_mod

log = logging.getLogger("tb.engine")

BAR_MS = 300_000
CACHE_BARS = 760
FIRST_FETCH = 720
HISTORY_PAGE = 1500               # klines per REST page when backfilling history (weight 10)
HISTORY_WEIGHT_CEILING = 1200     # no history page while the minute's used request weight is above this (limit 2400)
HISTORY_PAGES_PER_CYCLE = 3       # at most this many symbols get a backfill page per bars_cycle
MAKER_POLL_S = 1.0                # order-state poll while a post-only entry rests
XSEC_FRESH_S = 900                # a rebalance slot is only executed within this long after its boundary
SIGNAL_FRESH_MS = 90_000          # only trade a signal bar that closed < 90 s ago
STOP_CHECK_LIVE_S = 15.0
SYNC_LIVE_S = 5.0
DEPTH_MIN_AGE_S = 10.0            # per-symbol order-book refresh period (8 symbols -> <= 0.8 weight-20 calls/s = ~960/min)
DEPTH_WEIGHT_CEILING = 1500       # skip depth fetches while the minute's used weight is above this (limit 2400)
BRACKET_TTL_S = 3600.0
PROTECT_RETRY_S = 30.0            # after a failed stop move / partial take-profit / resize: wait this long before retrying


@dataclass
class MakerFill:
    """What an entry attempt ended with: ``qty`` filled at ``avg_price`` (``mode``: maker | taker fallback)."""
    qty: float
    avg_price: float
    fee: float
    opened_ts: int            # when the first order was placed (every fill of the attempt is at or after it)
    mode: str = "maker"


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

    def refresh(self, symbol: str, lookback_h: float = 26.0) -> None:
        """Top up ``symbol``'s metrics / funding. ``lookback_h`` = how much derivative history the loaded models
        need (hours; REST serves <= 29 days of metrics): the first fetch covers all of it, later ones only the
        tail (overlapping the last 2 h so a late-published row is picked up)."""
        from .data import live

        now = int(time.time() * 1000)
        lb_ms = int(max(26.0, lookback_h) * 3600_000)
        try:
            have = self.metrics.get(symbol)
            start = now - lb_ms
            if have is not None and len(have) and int(have["create_time"].iloc[0]) <= now - lb_ms + 6 * 3600_000:
                start = max(start, int(have["create_time"].iloc[-1]) - 2 * 3600_000)     # incremental top-up
            m = live.fetch_metrics(symbol, start, now=now, pace=0.15 if lb_ms <= 48 * 3600_000 else 0.3)
            if len(m):
                if have is not None and len(have) and start > now - lb_ms:
                    m = pd.concat([have[have["create_time"] < int(m["create_time"].iloc[0])], m], ignore_index=True)
                self.metrics[symbol] = m[m["create_time"] >= now - lb_ms].drop_duplicates("create_time", keep="last") \
                    .reset_index(drop=True)
        except Exception as e:
            log.warning("metrics top-up failed for %s: %s", symbol, e)
        self.metrics_at[symbol] = time.time()
        if time.time() - self.funding_at.get(symbol, 0) > 3600:
            try:
                f = live.fetch_funding(symbol, now - max(4 * 86_400_000, lb_ms + 2 * 86_400_000), now)
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
        self.brackets: dict[str, tuple[float, list | None]] = {}      # symbol -> (fetched_at, tiers or None)
        self.vmarks: dict[str, float] = {}          # ORDER-VENUE mark prices (== prod marks in paper_local)
        self.dropped_venue: list[str] = []
        self.paused: set[str] = set(store.kv_get("paused_symbols", []) or [])
        self.force_universe = False
        self.depth_at: dict[str, float] = {}
        self.depth_stats = {"ok": 0, "err": 0}
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
        self.wallet_equity = 0.0
        self.basis_fresh = False
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
        # V3: history, entry controls, maker entries, cross-sectional book
        self.hist_need = FIRST_FETCH                           # 5m bars of history to keep per symbol
        self.hist_done: set[str] = set()                       # symbols whose REST history is exhausted (new listings)
        self.hist_warned: dict[str, float] = {}
        self.live_gate = EntryGate()                           # cooldown / trades-per-day of the real account
        self.shadow_gates: dict[str, EntryGate] = {}           # ... and of every shadow model
        self.pending: dict[str, dict] = {}                     # symbol -> reservation of an in-flight maker entry
        self.maker_poll_s = MAKER_POLL_S
        self.maker_sleep: Callable = asyncio.sleep             # tests replace it with a no-op
        self.maker_stats = {"entries": 0, "filled": 0, "partial": 0, "timeouts": 0, "gtx_rejects": 0, "reprices": 0,
                            "taker_fallback": 0, "skipped": 0, "aborted": 0}
        self.xsec_slots: dict[str, int] = {}                   # model -> last executed rebalance slot
        self.xsec_book: dict[str, str] = {}                    # symbol -> side of the live cross-sectional legs
        self.time_stops: dict[str, float] = {}                 # position id -> unix time at which it is closed (horizon_bars)
        self.model_gaps: dict[str, float] = {}
        self._spec_cache: dict[tuple, ExitSpec] = {}
        self._backfill_budget = HISTORY_PAGES_PER_CYCLE
        # profit protection, per position id: last stop-move bar, partial take-profit state, booked partial (for the
        # exchange close-out), stale stops still to cancel, failure timestamps. Persisted in kv "protect".
        self.prot: dict[str, dict] = {}

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
                                     "shadow_open", "universe_at", "live_trades_opened", "xsec_book", "xsec_slots",
                                     "maker_stats", "time_stops", "protect"])
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
        if cap.is_active(self.cfg, self.mode):
            if self.basis_fresh or self.store.kv_get(cap.BASIS_KEY) != self.cfg.capital_usd:
                self.rebase_equity("stake set" if self.basis_fresh else "capital_usd changed")
        self.running = bool(st["running"]) if st.get("running") is not None else self.paper
        if st.get("shadow_open"):
            self.shadow.import_state(st["shadow_open"])
        self.store.kv_set("start_equity", self.start_equity)
        self.xsec_book = {str(k): str(v) for k, v in (st.get("xsec_book") or {}).items()}
        self.xsec_slots = {str(k): int(v) for k, v in (st.get("xsec_slots") or {}).items()}
        self.time_stops = {str(k): float(v) for k, v in (st.get("time_stops") or {}).items()}
        self.prot = {str(k): dict(v) for k, v in (st.get("protect") or {}).items() if isinstance(v, dict)}
        for k, v in (st.get("maker_stats") or {}).items():
            if k in self.maker_stats:
                self.maker_stats[k] = int(v)
        self.reload_models(force=True)
        self.seed_live_gate()
        self.cancel_stale_entries()
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
        if champ and self.mode == "live" and is_paper_only(champ):
            log.error("REFUSING to load champion %s in %s mode: its note says 'PAPER TEST ONLY'; running NullScorer",
                      champ.name, self.mode)
            self.scorer, self.model_info = NullScorer(), None
        elif champ and (self.model_info is None or self.model_info.name != champ.name or force):
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
            if self.mode == "live" and is_paper_only(e):
                log.error("REFUSING to load challenger %s in %s mode ('PAPER TEST ONLY')", n, self.mode)
                continue
            if n not in self.challengers:
                got = load_model(e)
                if got:
                    self.challengers[n] = got
                    log.info("challenger model loaded (shadow mode): %s kind=%s history=%s bars", n, got[1].kind,
                             got[1].history_bars_5m or "default")
        self.recompute_history_need()

    # ---- V3 model properties
    def all_models(self) -> list[tuple[str, Scorer, ModelInfo, bool]]:
        """[(name, scorer, info, is_champion)] for the champion (when real) and every challenger."""
        out: list[tuple[str, Scorer, ModelInfo, bool]] = []
        if self.model_info and not isinstance(self.scorer, NullScorer):
            out.append((self.model_info.name, self.scorer, self.model_info, True))
        out += [(n, sc, info, False) for n, (sc, info) in self.challengers.items()]
        return out

    def model_spec(self, info: ModelInfo | None) -> ExitSpec:
        """Exit template of a model: its ``exit_cfg`` for pct / non-5m-ATR exits, else the Settings exits."""
        if info is None or not info.exit_cfg:
            return SETTINGS_SPEC
        key = (info.name, repr(sorted(info.exit_cfg.items(), key=lambda kv: kv[0])), info.kind)
        got = self._spec_cache.get(key)
        if got is None:
            got = self._spec_cache[key] = resolve_exit(info.exit_cfg, info.kind, info.name)
        return got

    def model_entry(self, info: ModelInfo | None) -> EntryCfg:
        return EntryCfg.resolve(self.cfg, info.entry_cfg if info else None)

    def model_thresholds(self, info: ModelInfo | None) -> tuple[float, float]:
        """(long, short) probability thresholds: the Settings for classic per-symbol models (as always), the
        artifact's own ``thresholds`` for the V3 kinds (falling back to the Settings when it has none)."""
        cfg = self.cfg
        if info is None or info.kind == KIND_PER_SYMBOL:
            return cfg.prob_threshold_long, cfg.prob_threshold_short
        return (info.threshold_long if info.threshold_long is not None else cfg.prob_threshold_long,
                info.threshold_short if info.threshold_short is not None else cfg.prob_threshold_short)

    def recompute_history_need(self) -> None:
        need = FIRST_FETCH
        for _, _, info, _ in self.all_models():
            need = max(need, int(info.history_bars_5m or 0) + self.model_spec(info).history_bars())
        if need != self.hist_need:
            log.info("5m history needed per symbol: %d bars (was %d)", need, self.hist_need)
            self.hist_need = need

    def deriv_lookback_h(self) -> float:
        if self.hist_need <= FIRST_FETCH:
            return 26.0
        return min(max(26.0, self.hist_need * 5.0 / 60.0 + 2.0), 28 * 24.0)       # REST serves <= 29 days of metrics

    def seed_live_gate(self) -> None:
        """Cooldown / trades-per-day bookkeeping of the real account from the DB (restart-safe)."""
        g = EntryGate()
        d0 = day_start(self.clock()) - 2 * 86400
        for r in self.store.query("SELECT symbol, opened_at, closed_at, exit_reason, pnl FROM trades "
                                  "WHERE closed_at>=? AND exit_reason!='partial'", (d0,)):
            g.record_entry(r["symbol"], r["opened_at"])
            g.record_exit(r["symbol"], r["closed_at"], loss=is_loss_exit(r["exit_reason"], r["pnl"]))
        for p in self._db_positions():
            g.record_entry(p["symbol"], p["opened_at"])
        self.live_gate = g

    def shadow_gate(self, name: str) -> EntryGate:
        g = self.shadow_gates.get(name)
        if g is None:
            g = self.shadow_gates[name] = EntryGate()
            d0 = day_start(self.clock()) - 2 * 86400
            for t in self.store.get_shadow_trades(name, int(d0)):
                g.record_entry(t["symbol"], t["entry_time"])
                g.record_exit(t["symbol"], t["exit_time"], loss=is_loss_exit(t["exit_reason"], t["pnl"]))
            for p in self.shadow.pos.values():
                if p.model == name:
                    g.record_entry(p.symbol, p.opened_at)
        return g

    def cancel_stale_entries(self) -> None:
        """A resting maker entry left by a crashed / restarted engine could fill later with no stop: cancel it."""
        if self.paper:
            return
        try:
            for sym, oid in self.ex.open_entry_orders():
                self.ex.cancel_order(sym, oid)
                log.warning("cancelled a stale resting entry order %s on %s from before the restart", oid, sym)
        except ExchangeError as e:
            log.warning("could not check for stale entry orders: %s", e)

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
        if self.mode == "live" and is_paper_only(e):
            return f"{model!r} is marked 'PAPER TEST ONLY' and cannot be promoted in {self.mode} mode"
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
        self.wallet_equity = b.wallet + b.upnl
        self.cash, self.upnl, self.available = b.wallet, b.upnl, b.available
        self.equity = self.wallet_equity
        if cap.is_active(self.cfg, self.mode):
            if cap.floor_id(self.store) is None:
                self.basis_fresh = True                       # first run with a stake: peak/start get re-based
            bot = cap.bot_equity(self.store, self.cfg, self.mode, self.wallet_equity, b.upnl, create=True)
            self.equity = bot
            self.cash = bot - b.upnl
            used = sum(p["margin"] for p in self.store.get_positions())
            self.available = max(0.0, min(b.available, bot - used))       # the bot never uses more than its stake

    def rebase_equity(self, why: str) -> None:
        """Start/peak equity restart from the current bot equity (the old figures were on another basis)."""
        old = (self.start_equity, self.peak_equity)
        self.start_equity = self.peak_equity = self.equity
        self.store.kv_set("start_equity", self.start_equity)
        self.store.kv_set("peak_equity", self.peak_equity)
        self.store.kv_set(cap.BASIS_KEY, self.cfg.capital_usd)
        self.store.kv_set(cap.SINCE_KEY, int(self.clock()))   # equity chart restarts on the new basis
        if self.kill.get("tripped") and str(self.kill.get("reason", "")).startswith("drawdown"):
            log.warning("clearing a drawdown kill measured on the old equity basis")
            self.kill = {"tripped": False, "reason": "", "t": None}
        self.basis_fresh = False
        log.warning("equity basis re-set (%s): bot equity %.2f (capital_usd=%s, wallet %.2f); was start %.2f / peak %.2f",
                    why, self.equity, self.cfg.capital_usd, self.wallet_equity, *old)

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
            self.store.kv_set("xsec_book", self.xsec_book)
            self.store.kv_set("xsec_slots", self.xsec_slots)
            self.store.kv_set("time_stops", self.time_stops)
            live_ids = {str(p["id"]) for p in self._db_positions()}
            self.prot = {k: v for k, v in self.prot.items() if k in live_ids}
            self.store.kv_set("protect", self.prot)
            self.store.kv_set("maker_stats", self.maker_stats)
        except Exception:
            log.exception("persist failed")

    def dd(self) -> float:
        """Drawdown % used by the kill switch and derating, on the configured basis (stake or peak)."""
        return drawdown_pct(self.equity, dd_reference(self.start_equity, self.peak_equity, self.cfg.dd_basis))

    def risk_multiplier(self) -> tuple[float, dict]:
        """derate x drift x slippage x staged-rollout multiplier (+ the parts, for logging)."""
        cfg = self.cfg
        dd = self.dd()
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
                info = self._last_close(pos)
                if info is None:
                    px = self.marks.get(pos["symbol"]) or pos["entry_price"]
                    d = 1 if pos["side"] == "long" else -1
                    info = CloseInfo(px, (px - pos["entry_price"]) * pos["qty"] * d, 0.0, 0.0, "manual", int(self.clock()))
                log.warning("reconcile: %s no longer open on the exchange; recording close (%s)", pos["symbol"], info.reason)
                self._finalize(pos, info, info.reason)
        known = {p["symbol"] for p in self._db_positions()}
        for sym, xp in xpos.items():
            if sym in known or sym in self.pending:          # (a pending maker entry's fill is protected by its own flow)
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

    def _last_close(self, pos: dict) -> CloseInfo | None:
        """How the position closed on the exchange; a partial take-profit already booked as its own trade is passed on
        so venues that rebuild the close from fills do not count it twice."""
        booked = self.prot.get(str(pos["id"]), {}).get("booked")
        kw = {"booked": booked} if booked else {}
        return self.ex.last_close(pos["symbol"], pos["side"], int(pos["opened_at"] * 1000), pos.get("sl_order_id"),
                                  pos.get("tp_order_id"), pos["entry_price"], **kw)

    def _finalize(self, pos: dict, info: CloseInfo, reason: str) -> None:
        """Persist a closed position as a trade (sync; no exchange calls)."""
        if reason == "sl" and pos.get("trail_active"):
            reason = "trail"
        t = int(info.closed_at or self.clock())
        net = info.net_pnl                       # trades.pnl is NET; fees and funding are COSTS (>0 = paid)
        trade = {"position_id": pos["id"], "symbol": pos["symbol"], "side": pos["side"],
                 "entry_price": pos["entry_price"], "exit_price": info.exit_price, "qty": pos["qty"],
                 "opened_at": pos["opened_at"], "closed_at": t, "pnl": net, "fees": info.fees,
                 "funding": info.funding_cost, "exit_reason": reason, "prob": pos.get("prob")}
        tid = self.store.insert_trade(trade)
        self.store.delete_position(pos["id"])
        self.live_gate.record_exit(pos["symbol"], t, loss=is_loss_exit(reason, net))
        self.prot.pop(str(pos["id"]), None)
        self.xsec_book.pop(pos["symbol"], None)
        self.time_stops.pop(str(pos["id"]), None)
        self.event("position_closed", {**trade, "id": tid})
        log.info("closed %s %s %s: net pnl %+.3f (gross %+.3f, fees %.3f, funding cost %+.3f) %s", pos["symbol"], pos["side"],
                 pos["qty"], net, info.pnl, info.fees, info.funding_cost, reason)
        self.alerts.trade_closed(pos["symbol"], pos["side"], pos["entry_price"], info.exit_price, net,
                                 info.fees, reason, self.mode)
        self._update_drift()
        self.persist_state(force=True)

    def _update_drift(self) -> None:
        rows = self.store.query("SELECT prob, pnl FROM trades WHERE prob IS NOT NULL AND exit_reason!='partial' "
                                "ORDER BY id DESC LIMIT ?", (self.cfg.drift_n,))
        rows = list(reversed(rows))
        probs = [r["prob"] for r in rows]
        wins = [r["pnl"] > 0 for r in rows]                 # pnl is net
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
                info = await self.io(self._last_close, pos)
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
    async def _prepare_symbol(self, sym: str, want_lev: int) -> int | None:
        """Isolated margin + leverage before an entry (a lower leverage is always safer than the chosen one)."""
        ex = self.ex
        try:
            await self.io(ex.set_isolated, sym)
        except ExchangeError as e:
            log.error("cannot set isolated margin for %s: %s; skipping entry", sym, e)
            return None
        for attempt in (want_lev, max(1, want_lev // 2), 1):
            try:
                await self.io(ex.set_leverage, sym, attempt)
                lev = attempt
                break
            except ExchangeError as e:
                log.warning("set_leverage(%s, %dx) failed: %s", sym, attempt, e)
        else:
            log.error("cannot set leverage for %s; skipping entry", sym)
            return None
        if lev != want_lev:
            log.warning("%s: leverage lowered from %dx to %dx", sym, want_lev, lev)
        return lev

    async def _flatten_fill(self, sym: str, side: str, qty: float, fill: float, prob: float | None) -> bool:
        """Market-close an unprotected / unwanted fill and record the flat round trip (fees are real). True if flat."""
        ex = self.ex
        close_side = opposite(order_side(side))
        try:
            await self.io(ex.market_order, sym, close_side, qty, True)
            await self.io(ex.cancel_all, sym)
        except ExchangeError as e:
            log.error("EMERGENCY CLOSE of unprotected %s failed: %s", sym, e)
            self.alerts.engine_error("emergency_close", f"{sym}: {e}")
            self.kill = {"tripped": True, "reason": f"unprotected position {sym} could not be closed", "t": int(self.clock())}
            return False
        info = None
        try:
            info = await self.io(ex.last_close, sym, side, int(self.clock() * 1000) - 60_000)
        except ExchangeError:
            pass
        if info:
            t = int(self.clock())
            self.store.insert_trade({"symbol": sym, "side": side, "entry_price": fill, "exit_price": info.exit_price,
                                     "qty": qty, "opened_at": t, "closed_at": t, "pnl": info.net_pnl, "fees": info.fees,
                                     "funding": info.funding_cost, "exit_reason": "manual", "prob": prob})
        return True

    async def open_position(self, sym: str, side: str, sz, prob: float, reasons: list, atr: float | None,
                            ref_price: float, mult: float, *, spec: ExitSpec | None = None,
                            fill: "MakerFill | None" = None, lev: int | None = None) -> bool:
        """Enter (market order, or the already-filled ``fill`` of a maker entry), then place the exchange-side stop /
        take-profit. The stop is placed only AFTER the fill; if that fails the position is closed at once."""
        cfg = self.cfg
        ex = self.ex
        spec = spec or SETTINGS_SPEC
        eff = spec.effective(cfg)
        osd = order_side(side)
        if fill is None:
            lev = await self._prepare_symbol(sym, sz.leverage)
            if lev is None:
                return False
            try:
                res = await self.io(ex.market_order, sym, osd, sz.qty, False)
            except ExchangeError as e:
                log.error("entry order failed for %s: %s", sym, e)
                return False
            fill_px = res.avg_price or ref_price
            qty = res.qty or sz.qty
            entry_fee = res.fee or 0.0
            opened_ts = int(self.clock())
            via_maker = False
        else:
            lev = lev or sz.leverage
            fill_px, qty, entry_fee, opened_ts = fill.avg_price, fill.qty, fill.fee, fill.opened_ts
            via_maker = fill.mode == "maker"
            if not self.entry_allowed(sym):          # kill / stop / degraded / paused while the order rested
                log.warning("entry %s %s filled but entries are no longer allowed; closing the fill", side, sym)
                await self._flatten_fill(sym, side, qty, fill_px, prob)
                return False
        f = await self.io(ex.filters, sym)
        lv = levels(eff, side, fill_px, qty, spec.atr_arg(atr, fill_px), f, risk_mult=mult)
        close_side = opposite(osd)
        sl_id = tp_id = None
        sl = tp = None
        if lv is not None:
            sl, tp = lv
            if not spec.tp_enabled:
                tp = None
            if not self.paper:
                # never send a stop computed from one price level that sits on the wrong side of the venue's mark
                try:
                    vm = (await self.io(ex.premium, sym))[0].mark_price
                except ExchangeError:
                    vm = self.vmarks.get(sym)
                d = 1 if side == "long" else -1
                if vm and (vm - sl) * d <= 0:
                    log.error("computed SL %s for %s is on the wrong side of the venue mark %s; closing the fill", sl, sym, vm)
                    lv = None
                elif vm and tp is not None and (tp - vm) * d <= 0:
                    log.warning("computed TP %s for %s is already through the venue mark %s; no TP order, engine exits itself",
                                tp, sym, vm)
        if lv is not None:
            try:
                sl_id = await self.io(ex.place_stop, sym, close_side, sl, qty)
            except ExchangeError as e:
                log.error("SL placement FAILED for %s (%s); closing the position at market", sym, e)
                self.alerts.engine_error("sl_placement", f"{sym}: {e}")
        if sl_id is None:
            await self._flatten_fill(sym, side, qty, fill_px, prob)
            return False
        if tp is not None:
            try:
                tp_id = await self.io(ex.place_take_profit, sym, close_side, tp, qty)
            except ExchangeError as e:
                log.warning("TP placement failed for %s (%s); the engine will close at TP itself", sym, e)
        mode_label = cfg.exit_mode if spec.mode == "settings" else ("pct" if spec.mode == "pct" else f"atr:{spec.atr_tf}")
        pos = {"symbol": sym, "side": side, "entry_price": fill_px, "qty": qty, "notional": qty * fill_px,
               "leverage": lev, "margin": qty * fill_px / lev, "sl_price": sl, "tp_price": tp,
               "init_sl_price": sl, "atr": atr, "trail_active": False, "trail_price": None, "best_price": fill_px,
               "mark_price": fill_px, "upnl": 0.0, "opened_at": opened_ts, "prob": prob,
               "reasons": reasons, "sl_order_id": sl_id, "tp_order_id": tp_id, "entry_fee": entry_fee,
               "funding": 0.0, "exit_mode": mode_label,
               "liq_price": next((p.liq_price for p in await self.io(ex.positions) if p.symbol == sym), None)}
        pid = self.store.insert_position(pos)
        self.live_gate.record_entry(sym, opened_ts)
        if spec.horizon_bars:                         # labels count a timeout as a loss closed at market: enforce the same
            self.time_stops[str(pid)] = opened_ts + spec.horizon_bars * 300.0
        if not via_maker:        # (a maker fill is a price improvement, not a slippage sample)
            slip_pct = (fill_px - ref_price) / ref_price * 100.0 * (1 if side == "long" else -1)
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
        log.info("opened %s %s qty=%s @ %s SL %s TP %s prob=%.3f mult=%.2f%s", side, sym, qty, fill_px, sl, tp, prob, mult,
                 " (maker)" if via_maker else "")
        self.alerts.trade_opened(sym, side, fill_px, qty, qty * fill_px, sl, tp or 0.0, prob, self.mode)
        self.persist_state(force=True)
        await self.io(self._refresh_account)
        return True

    # ---- maker (post-only) entries
    TERMINAL = ("FILLED", "CANCELED", "CANCELLED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH")

    async def _settle_order(self, sym: str, oid: str, cur):
        """Cancel a resting entry and return its FINAL state (a cancel can race a fill: the lookup decides)."""
        ex = self.ex
        try:
            await self.io(ex.cancel_order, sym, oid)
        except ExchangeError as e:
            log.warning("cancel of entry order %s on %s failed: %s", oid, sym, e)
        for i in range(3):
            try:
                return await self.io(ex.get_order, sym, oid)
            except ExchangeError as e:
                log.warning("entry order lookup %s on %s failed: %s", oid, sym, e)
                await self.maker_sleep(0.3)
        try:
            await self.io(ex.cancel_all, sym)           # state unknown: make sure nothing keeps resting
        except ExchangeError:
            pass
        return None

    async def maker_fill(self, sym: str, side: str, sz, f, entry: EntryCfg, ref: float) -> "MakerFill | None":
        """Rest a post-only (GTX) limit at the best bid (long) / ask (short); every ``maker_timeout_s`` re-price it
        (cancel + repost at the new best) up to ``maker_reprices`` times. A rejection with -5022 (it would have taken
        liquidity) just re-prices. The first fill ends the attempt: whatever filled (partial or full) is returned so its
        stop is placed at once and the remainder is cancelled. Nothing filled -> taker order or skip per
        ``maker_on_timeout``. Aborts (cancelling the order) as soon as entries stop being allowed."""
        cfg = self.cfg
        ex = self.ex
        osd = order_side(side)
        long = side == "long"
        ms = self.maker_stats
        ms["entries"] += 1
        want = sz.qty
        got_q = got_n = 0.0
        t0 = int(self.clock())
        polls = max(1, math.ceil(entry.maker_timeout_s / self.maker_poll_s))
        oid: str | None = None
        aborted = uncertain = False
        try:
            for attempt in range(entry.maker_reprices + 1):
                if not self.entry_allowed(sym):
                    aborted = True
                    break
                if attempt:
                    ms["reprices"] += 1
                try:
                    bid, ask = await self.io(ex.best_bid_ask, sym)
                except ExchangeError as e:
                    log.warning("maker entry %s: no best bid/ask (%s)", sym, e)
                    await self.maker_sleep(self.maker_poll_s)
                    continue
                px = f.round_price(bid if long else ask)
                rq = f.round_qty(want - got_q)
                if rq < f.min_qty or rq * px < f.min_notional:
                    break
                try:
                    res = await self.io(ex.limit_order, sym, osd, rq, px, True)
                except ExchangeError as e:
                    if e.code == -5022:                    # post-only would have taken liquidity: the book moved
                        ms["gtx_rejects"] += 1
                        log.info("maker %s %s @ %s rejected (-5022, would take); re-pricing", side, sym, px)
                        await self.maker_sleep(0.5)
                        continue
                    log.error("maker entry order for %s failed: %s", sym, e)
                    break
                oid = res.order_id
                cur = res
                for _ in range(polls):
                    if cur.filled > 0 or cur.status in self.TERMINAL:
                        break
                    await self.maker_sleep(self.maker_poll_s)
                    if not self.entry_allowed(sym):
                        break
                    try:
                        cur = await self.io(ex.get_order, sym, oid)
                    except ExchangeError as e:
                        log.warning("entry order lookup %s on %s failed: %s", oid, sym, e)
                if cur.status != "FILLED":
                    cur = await self._settle_order(sym, oid, cur)
                    if cur is None:
                        uncertain = True
                oid = None
                if cur is not None:
                    got_q += cur.filled
                    got_n += cur.filled * (cur.avg_price or px)
                    if cur.filled > 0:
                        break                              # protect what filled; do not chase the remainder
                elif uncertain:
                    break
                if not self.entry_allowed(sym):
                    aborted = True
                    break
        except Exception:
            log.exception("maker entry for %s failed unexpectedly", sym)
            uncertain = True
        finally:
            if oid is not None:                            # an order may still rest: never leave it
                try:
                    await self.io(ex.cancel_all, sym)
                except Exception:
                    log.exception("could not cancel the resting entry order on %s", sym)
        if got_q <= 0 and uncertain and not self.paper:     # order state unknown: the exchange position is the truth
            try:
                xp = next((p for p in await self.io(ex.positions) if p.symbol == sym and p.side == side), None)
            except ExchangeError:
                xp = None
            if xp is not None and xp.qty > 0:
                log.warning("maker entry %s: order state unknown but the exchange holds %s %s; protecting it", sym, xp.qty, side)
                got_q, got_n = xp.qty, xp.qty * xp.entry_price
        if got_q > 0:
            ms["filled"] += 1
            if got_q < want * 0.999:
                ms["partial"] += 1
            avg = got_n / got_q
            return MakerFill(got_q, avg, got_q * avg * cfg.fee_maker, t0, "maker")
        if aborted:
            ms["aborted"] += 1
            return None
        ms["timeouts"] += 1
        if entry.maker_on_timeout == "taker" and self.entry_allowed(sym):
            try:
                res = await self.io(ex.market_order, sym, osd, want, False)
            except ExchangeError as e:
                log.error("taker fallback for %s failed: %s", sym, e)
                ms["skipped"] += 1
                return None
            ms["taker_fallback"] += 1
            px = res.avg_price or ref
            q = res.qty or want
            return MakerFill(q, px, res.fee or q * px * cfg.fee_taker, t0, "taker")
        ms["skipped"] += 1
        log.info("maker entry %s %s: no fill after %d attempt(s); skipped", side, sym, entry.maker_reprices + 1)
        return None

    async def get_brackets(self, sym: str):
        """Maintenance-margin tiers for ``sym`` (cached ~1 h). None = unknown -> ``cfg.maint_margin_rate`` is used."""
        ent = self.brackets.get(sym)
        if ent and self.clock() - ent[0] < BRACKET_TTL_S:
            return ent[1]
        tiers = None
        try:
            tiers = await self.io(self.ex.leverage_brackets, sym)
        except ExchangeError as e:
            log.warning("leverage brackets unavailable for %s (%s); using default mmr %.3f", sym, e, self.cfg.maint_margin_rate)
        self.brackets[sym] = (self.clock(), tiers)
        return tiers

    def book(self) -> Book:
        """Account aggregates over the open positions AND the in-flight maker entries (cached brackets; no I/O)."""
        n = mm = risk = 0.0
        by_side = {"long": 0.0, "short": 0.0}
        for p in self._db_positions():
            ent = self.brackets.get(p["symbol"])
            n += p["notional"]
            mm += maintenance_margin(p["notional"], ent[1] if ent else None, self.cfg.maint_margin_rate)
            r = sl_loss(self.cfg, p["qty"], p["entry_price"], p["sl_price"], p["side"])
            risk += r
            by_side[p["side"]] = by_side.get(p["side"], 0.0) + r
        reserved = 0.0
        for r in self.pending.values():          # a resting post-only entry already counts against every cap
            n += r["notional"]
            mm += r["mm"]
            risk += r["risk"]
            by_side[r.get("side", "long")] = by_side.get(r.get("side", "long"), 0.0) + r["risk"]
            reserved += r["margin"]
        return Book(self.equity, self.available - reserved, n, mm, risk, by_side["long"], by_side["short"])

    async def try_open(self, sym: str, side: str, prob: float, reasons: list, closed: pd.DataFrame, *,
                       neutral: bool = False, spec: ExitSpec | None = None, entry: EntryCfg | None = None) -> bool:
        """Size, check every cap and open ``sym``. ``neutral`` = a leg of a market-neutral book (the correlation cap is
        not applied; all other caps are). ``spec`` / ``entry`` default to the champion's exit / entry template.
        A maker entry waits for its fill OUTSIDE the trade lock (kill / close_all / stop checks keep running)."""
        cfg = self.cfg
        spec = spec or self.model_spec(self.model_info)
        entry = entry or self.model_entry(self.model_info)
        spec = bind_exit(spec, closed, self.now_ms())
        if spec is None:
            log.info("skip %s %s: not enough hourly history for the volatility-scaled stop", side, sym)
            return False
        async with self.tlock:
            if not self.entry_allowed(sym) or sym in self.pending:
                return False
            if not await self.io(self.venue_tradable, sym):
                log.info("skip %s %s: not TRADING on the order venue (%s)", side, sym, self.mode)
                return False
            positions = self._db_positions()
            if any(p["symbol"] == sym for p in positions):
                return False
            if cfg.max_positions is not None and len(positions) + len(self.pending) >= cfg.max_positions:
                log.info("skip %s %s: blocked by max_positions=%d", side, sym, cfg.max_positions)
                return False
            atr = exit_atr(spec, closed, self.now_ms())
            ref = self.marks.get(sym) or float(closed["close"].iloc[-1])
            if not self.paper:
                # size on the ORDER VENUE's price; scale the prod-derived ATR to the venue's price level
                vref = self.vmarks.get(sym)
                if not vref:
                    log.info("skip %s %s: no order-venue mark price yet", side, sym)
                    return False
                atr = atr * vref / ref if atr else atr
                ref = vref
            # correlation cap (not inside a market-neutral book: its legs hedge each other)
            if not neutral:
                others = []
                for p in positions:
                    k = self.kl.get(p["symbol"])
                    if k is not None:
                        others.append((p["symbol"], p["side"], k.set_index("open_time")["close"]))
                cand = closed.set_index("open_time")["close"]
                ok, hot = correlation_gate(side, cand, others, cfg.corr_cap)
                if not ok:
                    log.info("skip %s %s: correlated with %s", side, sym, hot)
                    return False
            mult, parts = self.risk_multiplier()
            f = await self.io(self.ex.filters, sym)
            brackets = await self.get_brackets(sym)
            bk = self.book()
            eff = spec.effective(cfg)
            sz = size_trade(eff, side, ref, spec.atr_arg(atr, ref), f, bk.available, risk_mult=mult, brackets=brackets)
            if not sz.ok:
                log.info("skip %s %s: %s (risk mult %.2f)", side, sym, sz.reason, mult)
                return False
            fee = sz.notional * cfg.fee_taker
            new_risk = sl_loss(cfg, sz.qty, sz.entry, sz.sl_price, side, entry_fee=True)
            chk = check_entry(cfg, bk, sz.notional, sz.mm, new_risk, sz.margin, fee, side)
            if not chk.ok:
                log.info("skip %s %s: blocked by rule %s (%s)", side, sym, chk.rule, chk.detail)
                return False
            if entry.entry_mode != "maker":
                return await self.open_position(sym, side, sz, prob, reasons, atr, ref, mult, spec=spec)
            lev = await self._prepare_symbol(sym, sz.leverage)
            if lev is None:
                return False
            self.pending[sym] = {"notional": sz.notional, "mm": sz.mm, "risk": new_risk, "margin": sz.margin, "side": side,
                                 "t": self.clock()}
        # ---- maker entry: post-only limit at the best bid / ask, outside the trade lock
        fill = None
        try:
            fill = await self.maker_fill(sym, side, sz, f, entry, ref)
        except Exception:
            log.exception("maker entry %s %s failed", side, sym)
        async with self.tlock:
            try:
                if fill is None:
                    return False
                return await self.open_position(sym, side, sz, prob, reasons, atr, ref, mult, spec=spec, fill=fill, lev=lev)
            finally:
                self.pending.pop(sym, None)

    def venue_tradable(self, sym: str) -> bool:
        """True when the symbol is TRADING on the order venue (always true for the paper venue = prod)."""
        if self.paper:
            return True
        try:
            i = self.ex.exchange_info().get(sym)
        except ExchangeError as e:
            log.warning("venue exchangeInfo unavailable (%s); not trading %s", e, sym)
            return False
        return bool(i and i.status == "TRADING")

    def venue_scale(self, sym: str) -> float:
        """venue mark / prod mark (1.0 in paper): converts prod-derived price distances (ATR) to the venue."""
        if self.paper:
            return 1.0
        v, p = self.vmarks.get(sym), self.marks.get(sym)
        return v / p if v and p else 1.0

    def entry_allowed(self, sym: str | None = None) -> bool:
        if not self.running or self.kill.get("tripped"):
            return False
        if self.safety_view.get("state", NORMAL) != NORMAL:
            return False
        if sym and self.faults.excluded(sym):
            return False
        if sym and sym in self.paused:
            return False
        return True

    # ================================================================== kill switch
    async def trip_kill(self, reason: str) -> None:
        if self.kill.get("tripped"):
            return
        self.kill = {"tripped": True, "reason": reason, "t": int(self.clock())}
        log.error("KILL SWITCH: %s", reason)
        self.event("kill", {**self.kill, "dd_limit_pct": self.cfg.kill_limit(self.mode)})
        self.alerts.kill(reason, self.equity, self.dd(), self.mode)
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
        if action in ("pause_symbol", "unpause_symbol"):
            sym = str(payload.get("symbol") or "").upper().strip()
            if not sym or not sym.isalnum():
                return "symbol is required"
            (self.paused.add if action == "pause_symbol" else self.paused.discard)(sym)
            self.store.kv_set("paused_symbols", sorted(self.paused))
            log.info("%s %s (open trades untouched)", "paused" if action == "pause_symbol" else "unpaused", sym)
            return None
        if action == "refresh_universe":
            self.force_universe = True          # picked up by the universe loop within ~5 s (never blocks commands)
            return None
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
        watch = {s: marks[s] for s in self.watchlist() if s in marks}          # PROD marks: signals, shadow, UI ticks
        if self.paper:
            vwatch = watch                                                    # the simulated venue IS prod
        else:
            vwatch = {}
            try:
                vprem = await self.io(self.ex.premium)                       # order venue's own mark prices
                self.vmarks.update({p.symbol: p.mark_price for p in vprem if p.mark_price > 0})
                vwatch = {s: self.vmarks[s] for s in self.watchlist() if s in self.vmarks}
            except ExchangeError as e:
                log.warning("venue mark prices unavailable: %s (positions are not re-marked this cycle)", e)
        if self.paper:
            self.vmarks.update(marks)
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
            await self.sync_positions(vwatch)
        # universe table + ticks
        self.store.update_universe_prices({s: m for s, m in marks.items() if s in self.universe})
        t = int(self.clock())
        for s, m in watch.items():
            self.event("tick", {"symbol": s, "t": t, "price": m})
        # account + kill switch
        await self.io(self._refresh_account)
        self.peak_equity = max(self.peak_equity, self.equity)
        limit = cfg.kill_limit(self.mode)
        if not self.kill.get("tripped") and self.dd() >= limit:
            await self.trip_kill(f"drawdown {self.dd():.1f}% >= {limit:.0f}% of "
                                 f"{'starting stake' if cfg.dd_basis == 'start' else 'peak'}")
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
                        info = await self.io(self._last_close, pos)
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
                if sym not in known and not self.paper and sym not in self.pending:
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
            # time stop (exit_cfg.horizon_bars): close at market once the horizon has passed
            dl = self.time_stops.get(str(pos["id"]))
            if dl is not None and now >= dl:
                log.info("time stop: %s %s open since %s, horizon reached; closing at market", pos["side"], sym,
                         int(pos["opened_at"]))
                await self.close_position({**pos, **upd}, "timeout")
                continue
            # fallback exits when the exchange-side order is missing / slow
            if pos.get("tp_price") and not pos.get("tp_order_id") and (m - pos["tp_price"]) * d >= 0:
                await self.close_position({**pos, **upd}, "tp")
                continue
            if not self.paper and (m - pos["sl_price"]) * d < -0.003 * pos["entry_price"]:
                log.error("%s is beyond its stop by >0.3%% and still open; closing at market", sym)
                await self.close_position({**pos, **upd}, "sl")
                continue
            # profit protection: break-even / trailing stop, partial take-profit (exchange orders are moved, never dropped)
            if await self._protect(pos, m, best, d, upd, now):
                continue
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

    # ================================================================== profit protection
    def _pos_atr(self, pos: dict, sym: str) -> float | None:
        """ATR the trailing stop of this position uses: the 5m ATR for the Settings exits, the exit timeframe's ATR for
        ``atr:<tf>`` model exits (venue-scaled). None = no trailing (pct exits, cross-sectional legs)."""
        mode = str(pos.get("exit_mode") or "")
        if sym in self.xsec_book:
            return None
        if mode in ("fixed", "atr", ""):
            a_prod = atr14(*self._hlc(sym))
            return a_prod * self.venue_scale(sym) if a_prod else pos.get("atr")
        if mode.startswith("atr:"):
            k = self.kl.get(sym)
            if k is None:
                return None
            now_ms = self.now_ms()
            a = exit_atr(ExitSpec("atr", mode[4:]), k[k["open_time"] + BAR_MS <= now_ms], now_ms)
            return a * self.venue_scale(sym) if a else None
        return None

    async def _protect(self, pos: dict, m: float, best: float, d: int, upd: dict, now: float) -> bool:
        """Break-even + trailing stop and the partial take-profit of one position (caller holds tlock). Mutates ``upd``
        (the DB fields to write). True = the position was closed here. A stop is only moved when it improves by >= 0.1 %
        of the entry price and at most once per 5m bar; the NEW exchange stop is placed before the old one is cancelled
        (``Exchange.replace_conditional``), so a failure keeps the old stop live."""
        sym, side = pos["symbol"], pos["side"]
        pc = Protect.from_settings(self.cfg)
        st = self.prot.setdefault(str(pos["id"]), {})
        entry = pos["entry_price"]
        close_side = opposite(order_side(side))
        bar = self.now_ms() // BAR_MS
        await self._protect_retry(pos, st, upd, now)
        f = await self.io(self.ex.filters, sym)
        sl_cur = float(upd.get("sl_price", pos["sl_price"]))
        # ---- stop: break-even, then trail
        atr = self._pos_atr(pos, sym) if pc.trail_enabled else None
        active, new_sl, why = next_stop(side, entry, pos["init_sl_price"], sl_cur, best, atr, bool(pos["trail_active"]), pc)
        new_sl = f.round_price(new_sl)
        if active and not pos["trail_active"]:
            upd["trail_active"] = True
        if why and (new_sl - sl_cur) * d > max(f.tick_size, PROTECT_MIN_STEP * entry) and st.get("bar") != bar \
                and now - st.get("fail_t", -1e9) >= PROTECT_RETRY_S:
            if (new_sl - m) * d >= 0:               # price already through the new stop: exit now
                await self.close_position({**pos, **upd, "trail_active": active}, "trail" if active else "sl")
                return True
            try:
                new_id, stale = await self.io(self.ex.replace_conditional, sym, "SL", close_side, new_sl, pos["qty"],
                                              pos.get("sl_order_id"))
            except ExchangeError as e:
                st["fail_t"] = now
                log.error("%s stop move %s -> %s FAILED (%s); the old stop stays live, retry in %.0fs", sym, sl_cur, new_sl, e,
                          PROTECT_RETRY_S)
            else:
                st["bar"] = bar
                st.pop("fail_t", None)
                if stale:
                    st.setdefault("orphans", []).append(stale)
                upd.update(sl_price=new_sl, sl_order_id=new_id, trail_active=active)
                if why == "trail":
                    upd["trail_price"] = new_sl
                self.event("position_updated", {"id": pos["id"], "symbol": sym, "sl_price": new_sl, "trail_active": active,
                                                "trail_price": upd.get("trail_price", pos.get("trail_price"))})
                log.info("%s stop -> %s (%s; was %s, price %s)", sym, new_sl, why, sl_cur, m)
        # ---- partial take-profit (once per position; not for cross-sectional legs, which close on the rebalance)
        if not st.get("partial") and sym not in self.xsec_book:
            lvl = partial_level(side, entry, pos["init_sl_price"], pos.get("tp_price"), pc)
            if lvl is not None and (m - lvl) * d >= 0 and now - st.get("pfail_t", -1e9) >= PROTECT_RETRY_S:
                await self._partial_tp(pos, m, d, upd, st, f, pc, now)
        return False

    async def _partial_tp(self, pos: dict, m: float, d: int, upd: dict, st: dict, f, pc: Protect, now: float) -> None:
        """Close ``partial_tp_frac`` of the position with a reduce-only market order, book it as its own trade (exit reason
        "partial", NET pnl: fees and funding are costs), then resize the exchange stop / TP to the remainder."""
        sym, side, entry = pos["symbol"], pos["side"], pos["entry_price"]
        split = partial_split(pos["qty"], m, pc.partial_tp_frac, f)
        if split is None:
            st["partial"] = "skipped"
            log.info("partial take-profit skipped for %s: %s x %s would leave a part below min qty / min notional",
                     sym, pos["qty"], pc.partial_tp_frac)
            return
        q, _ = split
        try:
            res = await self.io(self.ex.market_order, sym, opposite(order_side(side)), q, True)
        except ExchangeError as e:
            st["pfail_t"] = now
            log.error("partial take-profit order for %s failed: %s; retry in %.0fs", sym, e, PROTECT_RETRY_S)
            return
        filled = min(float(res.qty or q), pos["qty"])
        px = float(res.avg_price or m)
        frac = filled / pos["qty"]
        entry_fee = float(pos.get("entry_fee") or pos["qty"] * entry * self.cfg.fee_taker)
        funding = float(pos.get("funding") or 0.0)                    # signed cash flow (+ received)
        gross = (px - entry) * filled * d
        fees = entry_fee * frac + filled * px * self.cfg.fee_taker
        info = CloseInfo(px, gross, fees, funding * frac, "partial", int(now))
        net = info.net_pnl                                            # same convention as every trade row
        t = int(now)
        trade = {"position_id": pos["id"], "symbol": sym, "side": side, "entry_price": entry, "exit_price": px,
                 "qty": filled, "opened_at": pos["opened_at"], "closed_at": t, "pnl": net, "fees": fees,
                 "funding": info.funding_cost, "exit_reason": "partial", "prob": pos.get("prob")}
        tid = self.store.insert_trade(trade)
        rem = float(Decimal(repr(float(pos["qty"]))) - Decimal(repr(float(filled))))
        fields = {"qty": rem, "notional": rem * entry, "margin": float(pos["margin"]) * (1 - frac),
                  "entry_fee": entry_fee * (1 - frac), "funding": funding * (1 - frac)}
        self.store.update_position(pos["id"], **fields)
        pos.update(fields)
        upd.update(fields)
        upd["upnl"] = (m - entry) * rem * d
        st["partial"] = True
        st["booked"] = {"order_ids": [res.order_id], "entry_fee": entry_fee * frac, "funding": funding * frac}
        st["resize"] = True
        self.event("trade_partial", {**trade, "id": tid})
        log.info("PARTIAL take-profit %s %s: closed %s of %s @ %s (%.2fR): net pnl %+.3f (gross %+.3f, fees %.3f, funding cost %+.3f)",
                 side, sym, filled, filled + rem, px, pc.partial_tp_r, net, gross, fees, info.funding_cost)
        self.alerts.trade_closed(sym, side, entry, px, net, fees, "partial", self.mode)
        await self._resize_orders(pos, st, upd, now)
        self.persist_state(force=True)

    async def _resize_orders(self, pos: dict, st: dict, upd: dict, now: float) -> None:
        """After a partial close: stop and TP for exactly the remaining quantity, the stop at least at break-even.
        The new orders are placed first; if the new stop cannot be placed the old (larger) one stays live and the
        resize is retried every ``PROTECT_RETRY_S``."""
        sym, side = pos["symbol"], pos["side"]
        close_side = opposite(order_side(side))
        f = await self.io(self.ex.filters, sym)
        qty = float(pos["qty"])
        be = f.round_price(breakeven_price(side, pos["entry_price"], self.cfg.fee_taker))
        sl = tighter(side, float(upd.get("sl_price", pos["sl_price"])), be)
        try:
            new_id, stale = await self.io(self.ex.replace_conditional, sym, "SL", close_side, sl, qty,
                                          upd.get("sl_order_id", pos.get("sl_order_id")))
        except ExchangeError as e:
            st["resize"] = True
            st["retry_t"] = now
            log.error("%s: resizing the stop to the remaining %s FAILED (%s); the old stop stays live, retrying", sym, qty, e)
            return
        if stale:
            st.setdefault("orphans", []).append(stale)
        upd.update(sl_price=sl, sl_order_id=new_id)
        pos.update(sl_price=sl, sl_order_id=new_id)
        tp_id = upd.get("tp_order_id", pos.get("tp_order_id"))
        if tp_id and pos.get("tp_price"):
            try:
                tp_new, tp_stale = await self.io(self.ex.replace_conditional, sym, "TP", close_side, pos["tp_price"], qty, tp_id)
                upd["tp_order_id"] = pos["tp_order_id"] = tp_new
                if tp_stale:
                    st.setdefault("orphans", []).append(tp_stale)
            except ExchangeError as e:
                log.warning("%s: resizing the take-profit failed (%s); dropping it - the engine closes at TP itself", sym, e)
                try:
                    await self.io(self.ex.cancel_order, sym, tp_id)
                except ExchangeError:
                    st.setdefault("orphans", []).append(tp_id)
                upd["tp_order_id"] = pos["tp_order_id"] = None
        st.pop("resize", None)
        self.store.update_position(pos["id"], **{k: upd[k] for k in ("sl_price", "sl_order_id", "tp_order_id") if k in upd})
        self.event("position_updated", {"id": pos["id"], "symbol": sym, "sl_price": sl, "qty": qty})

    async def _protect_retry(self, pos: dict, st: dict, upd: dict, now: float) -> None:
        """Housekeeping of one position: cancel stops left over by a move, retry an unfinished resize."""
        if not (st.get("orphans") or st.get("resize")) or now - st.get("retry_t", -1e9) < PROTECT_RETRY_S:
            return
        st["retry_t"] = now
        left = []
        for oid in st.get("orphans", []):
            try:
                await self.io(self.ex.cancel_order, pos["symbol"], oid)
            except ExchangeError as e:
                log.warning("could not cancel the stale order %s on %s yet: %s", oid, pos["symbol"], e)
                left.append(oid)
        if left:
            st["orphans"] = left
        else:
            st.pop("orphans", None)
        if st.get("resize"):
            await self._resize_orders(pos, st, upd, now)

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
        self._backfill_budget = HISTORY_PAGES_PER_CYCLE
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
        try:
            await self.universe_models_cycle()
        except Exception:
            self.stats["errors"] += 1
            log.exception("cross-sectional cycle failed")

    async def backfill_history(self, sym: str, merged: pd.DataFrame) -> pd.DataFrame:
        """Prepend ONE page of older 5m bars (REST ``endTime`` paging) when the cache holds fewer bars than the
        loaded models need. Budgeted: a few symbols per cycle and only while the minute's request weight is low."""
        if self._backfill_budget <= 0 or getattr(self.data, "used_weight", 0) > HISTORY_WEIGHT_CEILING:
            return merged
        self._backfill_budget -= 1
        missing = self.hist_need + 40 - len(merged)
        n = int(min(HISTORY_PAGE, max(missing + 5, 100)))
        first = int(merged["open_time"].iloc[0])
        older = await self.io(self.data.klines, sym, "5m", n, None, first - 1)
        older = older[older["open_time"] < first] if len(older) else older
        if older.empty:
            self.hist_done.add(sym)                      # (a young listing has no older bars)
            log.info("%s: no older 5m history available (have %d bars)", sym, len(merged))
            return merged
        if len(older) < min(n, missing) * 0.5:
            self.hist_done.add(sym)
        return pd.concat([older, merged], ignore_index=True).drop_duplicates("open_time", keep="last") \
            .sort_values("open_time").reset_index(drop=True)

    async def update_symbol(self, sym: str) -> None:
        cfg = self.cfg
        cache = self.kl.get(sym)
        first_n = int(min(HISTORY_PAGE, max(FIRST_FETCH, self.hist_need)))
        limit = first_n if cache is None else 6
        df = await self.io(self.data.klines, sym, "5m", limit)
        if df.empty:
            return
        if cache is not None and int(df["open_time"].iloc[0]) > int(cache["open_time"].iloc[-1]) + BAR_MS:
            df = await self.io(self.data.klines, sym, "5m", first_n)      # gap: refetch the whole window
            cache = None
        if cache is None:
            merged = df
        else:
            merged = pd.concat([cache[cache["open_time"] < df["open_time"].iloc[0]], df], ignore_index=True)
        cache_bars = max(CACHE_BARS, self.hist_need + 40)
        if len(merged) < self.hist_need and sym not in self.hist_done:
            merged = await self.backfill_history(sym, merged)
        merged = merged.tail(cache_bars).reset_index(drop=True)
        self.kl[sym] = merged
        now_ms = self.now_ms()
        # candles -> store (+ live event)
        first = cache is None
        rows = merged.tail(1500) if first else merged.tail(3)
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

    # ---- model scoring (any kind)
    def bundle_for(self, sym: str, closed: pd.DataFrame) -> FrameBundle:
        """The ``FrameBundle`` (scorer interface v2) for ``sym``: closed 5m bars + derivatives (+ depth when a model uses it)."""
        btc = self.kl.get("BTCUSDT")
        btc_closed = btc[btc["open_time"] + BAR_MS <= self.now_ms()] if btc is not None else closed
        depth = None
        if any("depth" in (info.meta.get("feature_groups") or ()) for _, _, info, _ in self.all_models()):
            try:
                from . import orderbook as ob

                depth = ob.depth_frame(sym, self.store)
            except Exception:
                depth = None
        return FrameBundle(sym, closed.reset_index(drop=True), self.feed.metrics.get(sym), self.feed.funding.get(sym), depth,
                           btc_closed.reset_index(drop=True))

    def _model_error(self, name: str, what: str) -> None:
        """One model failing must never take the engine (or the other models) down: log, count, rate-limit."""
        self.stats["errors"] += 1
        if self.clock() - self.model_gaps.get(name, -1e9) > 600:
            self.model_gaps[name] = self.clock()
            log.exception("model %s failed (%s); skipped for this bar", name, what)

    async def score_one(self, scorer: Scorer, info: ModelInfo | None, sym: str, closed: pd.DataFrame,
                        cache: dict) -> Score | None:
        """One model's Score for ``sym`` on its newest closed bar; None = not enough history / unusable inputs / model error.
        The classic per-symbol 5m path builds the ``[T, F]`` window once per bar (``cache``) and calls ``score`` exactly
        as before; the V3 kinds go through ``score_frames`` (``score_universe`` / ``build_for_model``)."""
        kind = info.kind if info else KIND_PER_SYMBOL
        name = info.name if info else getattr(scorer, "name", "?")
        try:
            if kind == KIND_PER_SYMBOL:
                if "window" not in cache:
                    cache["window"] = await self.io(self.build_window, sym, closed)
                w = cache["window"]
                return None if w is None else await self.io(scorer.score, sym, w)
            if kind == "xsec":
                return None                                   # ranks the whole universe at once: see universe_models_cycle
            need = int(info.history_bars_5m or 0) if info else 0
            if need and len(closed) < need:
                if self.clock() - self.hist_warned.get(f"{name}|{sym}", -1e9) > 1800:
                    self.hist_warned[f"{name}|{sym}"] = self.clock()
                    log.info("model %s: %s has %d of the %d 5m bars it needs (history is still loading)", name, sym,
                             len(closed), need)
                return None
            res = await self.io(score_frames, scorer, info, {sym: self.bundle_for(sym, closed)})
            return res.get(sym)
        except Exception:
            self._model_error(name, f"{kind} score {sym}")
            return None

    def pick_side(self, sc: Score | None, info: ModelInfo | None) -> tuple[str | None, float]:
        """('long'|'short'|None, prob) for a Score under the model's thresholds (long wins a tie; shorts need short_enabled)."""
        if sc is None:
            return None, 0.0
        tl, ts = self.model_thresholds(info)
        pl, ps = float(sc.prob_long), float(sc.prob_short)
        cand_l = pl >= tl
        cand_s = self.cfg.short_enabled and ps >= ts
        if cand_l and (not cand_s or pl >= ps):
            return "long", pl
        if cand_s:
            return "short", ps
        return None, 0.0

    async def on_new_bar(self, sym: str, closed: pd.DataFrame, fresh: bool) -> None:
        cfg = self.cfg
        bar = closed.iloc[-1]
        # paper: replay the completed bar's wicks against simulated stops / resting maker entries, shadow too
        if self.paper and (sym in self.pending or any(p["symbol"] == sym for p in self._db_positions())):
            self.ex.check_range(sym, float(bar.low), float(bar.high), int(bar.open_time))
        if self.shadow.pos or self.shadow.pending:
            for tr in self.shadow.on_bar(sym, int(bar.open_time), float(bar.low), float(bar.high), self.clock(), frame=closed,
                                  close=float(bar.close)):
                self._record_shadow(tr)
        if sym not in self.trade_symbols():
            return
        info = self.model_info
        active = not isinstance(self.scorer, NullScorer)
        ent = self.model_entry(info)
        xsec_champ = active and info is not None and info.kind == "xsec"
        decision = is_decision_bar(int(bar.open_time), ent.decision_every_bars)
        cache: dict = {}
        sc: Score | None = None
        prob_l = prob_s = 0.0
        reasons: list = []
        if active and decision and not xsec_champ:
            sc = await self.score_one(self.scorer, info, sym, closed, cache)
            if sc is not None:
                prob_l, prob_s, reasons = float(sc.prob_long), float(sc.prob_short), reasons_to_dicts(sc.reasons)
        action, prob = self.pick_side(sc, info)
        action = action or "none"
        if not active or (decision and not xsec_champ):          # (cadence > 1: signals only at decision points)
            t = int(bar.open_time // 1000) + 300
            self.store.add_signal(t, sym, prob_l, prob_s, action, reasons)
            self.event("signal", {"t": t, "symbol": sym, "prob_long": prob_l, "prob_short": prob_s, "action": action,
                                  "reasons": reasons})
        # shadow mode: every model (champion + challengers) takes its would-be trades with its own entry / exit template
        if self.challengers:
            await self.shadow_signals(sym, closed, bar, cache, sc if (active and decision and not xsec_champ) else None)
        if action != "none":
            if fresh:
                ok, why = self.live_gate.check(sym, self.clock(), ent, len(self._db_positions()) + len(self.pending))
                if ok:
                    await self.try_open(sym, action, prob, reasons, closed)
                else:
                    log.info("signal %s %s skipped: %s", action, sym, why)
            else:
                log.info("signal %s %s ignored: bar not fresh", action, sym)

    async def shadow_signals(self, sym: str, closed: pd.DataFrame, bar, cache: dict, champ_sc: Score | None) -> None:
        cfg = self.cfg
        f = await self.io(self.ex.filters, sym)
        price = self.marks.get(sym) or float(closed["close"].iloc[-1])
        now = self.clock()
        models: list[tuple[str, ModelInfo | None, Score | None, EntryCfg]] = []
        if self.model_info:
            models.append((self.model_info.name, self.model_info, champ_sc, self.model_entry(self.model_info)))
        for name, (sc_obj, info) in self.challengers.items():
            if info.kind == "xsec":
                continue                                       # rebalanced in universe_models_cycle
            ent = self.model_entry(info)
            if not is_decision_bar(int(bar.open_time), ent.decision_every_bars):
                continue
            models.append((name, info, await self.score_one(sc_obj, info, sym, closed, cache), ent))
        for name, info, sc, ent in models:
            side, prob = self.pick_side(sc, info)
            if not side:
                continue
            gate = self.shadow_gate(name)
            n_open = sum(1 for k in (*self.shadow.pos, *self.shadow.pending) if k.startswith(f"{name}|"))
            ok, _ = gate.check(sym, now, ent, n_open)
            if not ok or self.shadow.has(name, sym):
                continue
            spec = bind_exit(self.model_spec(info), closed, self.now_ms())
            if spec is None:
                continue
            atr = exit_atr(spec, closed, self.now_ms())
            maker = ent.entry_mode == "maker"
            if self.shadow.open(name, sym, side, prob, float(closed["close"].iloc[-1]) if maker else price, now, atr, f,
                                spec=spec, entry=ent, decision_open_ms=int(bar.open_time)):
                gate.record_entry(sym, now)

    def _record_shadow(self, tr: dict) -> None:
        self.store.add_shadow_trade(tr)
        self.shadow_gate(tr["model"]).record_exit(tr["symbol"], tr["exit_time"], loss=is_loss_exit(tr["exit_reason"], tr["pnl"]))
        self.event("shadow_trade", tr)
        log.info("shadow %s %s %s pnl %+.3f (%s)", tr["model"], tr["side"], tr["symbol"], tr["pnl"], tr["exit_reason"])

    # ================================================================== cross-sectional (xsec) models
    @staticmethod
    def xsec_targets(scores: dict[str, Score], xc: XsecCfg, short_enabled: bool) -> dict[str, str]:
        """Target legs from a ranking: the ``top_k`` highest ``prob_long`` go long, of the rest the ``bottom_k`` highest
        ``prob_short`` go short (only with ``short_enabled``). Ties break by symbol name (deterministic)."""
        ok = {s: sc for s, sc in scores.items() if math.isfinite(float(sc.prob_long)) and math.isfinite(float(sc.prob_short))
              and (float(sc.prob_long) > 0 or float(sc.prob_short) > 0)}      # (0 / 0 = not tradable for the ranker)
        longs = sorted(ok, key=lambda s: (-float(ok[s].prob_long), s))[:xc.top_k]
        out = {s: "long" for s in longs}
        if short_enabled:
            rest = [s for s in ok if s not in out]
            out.update({s: "short" for s in sorted(rest, key=lambda s: (-float(ok[s].prob_short), s))[:xc.bottom_k]})
        return out

    def universe_frames(self, info: ModelInfo, boundary_ms: int) -> dict[str, FrameBundle]:
        """Bundles of the universe symbols whose newest closed bar is at/after ``boundary_ms`` and long enough."""
        out: dict[str, FrameBundle] = {}
        now_ms = self.now_ms()
        for sym in dict.fromkeys([*self.universe, "BTCUSDT"]):          # (BTC: the market reference of the ranker)
            k = self.kl.get(sym)
            if k is None or self.faults.excluded(sym):
                continue
            closed = k[k["open_time"] + BAR_MS <= now_ms]
            if closed.empty or int(closed["open_time"].iloc[-1]) + BAR_MS < boundary_ms:
                continue
            if info.history_bars_5m and len(closed) < info.history_bars_5m:
                continue
            out[sym] = self.bundle_for(sym, closed)
        return out

    async def universe_models_cycle(self) -> None:
        """Rebalance every cross-sectional model (champion for real, challengers in shadow) at its slot boundary."""
        for name, scorer, info, is_champ in self.all_models():
            if info.kind != "xsec":
                continue
            try:
                await self.xsec_rebalance(name, scorer, info, is_champ)
            except RateLimitError:
                raise
            except Exception:
                self._model_error(name, "xsec rebalance")

    async def xsec_rebalance(self, name: str, scorer: Scorer, info: ModelInfo, is_champ: bool) -> None:
        xc = XsecCfg.resolve(info.entry_cfg, info.meta)
        now_ms = self.now_ms()
        slot = rebalance_slot(now_ms, xc.rebalance_every_h)
        last = self.xsec_slots.get(name)
        if last is None:                 # first time this model is seen: wait for the next boundary
            self.xsec_slots[name] = slot
            log.info("xsec model %s: first rebalance at slot boundary in %.0f min", name,
                     (next_rebalance_ms(now_ms, xc.rebalance_every_h) - now_ms) / 60000)
            return
        if slot <= last or not self.universe:
            return
        boundary_ms = slot * xc.rebalance_every_h * 3_600_000
        if now_ms - boundary_ms > XSEC_FRESH_S * 1000:
            self.xsec_slots[name] = slot
            log.warning("xsec model %s: missed the rebalance at slot %d (inputs not ready within %d s)", name, slot,
                        XSEC_FRESH_S)
            return
        frames = self.universe_frames(info, boundary_ms)
        if sum(1 for s in frames if s in self.universe) < max(2, math.ceil(0.8 * len(self.universe))):
            return                        # bars still arriving: retried on the next bars cycle until the slot goes stale
        self.xsec_slots[name] = slot
        scores = await self.io(score_frames, scorer, info, frames)
        scores = {s: sc for s, sc in scores.items() if s in self.universe and s in frames}
        targets = self.xsec_targets(scores, xc, self.cfg.short_enabled)
        t_dec = boundary_ms // 1000
        if is_champ:
            for s, sc in scores.items():
                self.store.add_signal(t_dec, s, float(sc.prob_long), float(sc.prob_short), targets.get(s, "none"),
                                      reasons_to_dicts(sc.reasons))
        log.info("xsec %s rebalance %s: long %s | short %s", name, "LIVE" if is_champ else "shadow",
                 [s for s, v in targets.items() if v == "long"], [s for s, v in targets.items() if v == "short"])
        if is_champ:
            await self.xsec_execute(info, frames, scores, targets)
        else:
            await self.xsec_shadow(name, info, frames, scores, targets)
        self.persist_state(force=True)

    async def xsec_execute(self, info: ModelInfo, frames: dict[str, FrameBundle], scores: dict[str, Score],
                           targets: dict[str, str]) -> None:
        """Move the real book to the target legs: close legs no longer targeted, open the new ones under ALL the usual
        caps (open-risk, total leverage, margin ratio, free margin, kill, safety state, derating); each leg gets its
        exchange-side stop. The correlation cap is not applied inside the neutral book."""
        if not self.running or self.kill.get("tripped"):
            log.warning("xsec rebalance skipped: engine stopped or kill switch tripped")
            return
        spec = self.model_spec(info)
        ent = self.model_entry(info)
        gate_cfg = replace(ent, cooldown_bars=0)
        async with self.tlock:
            for sym in list(self.xsec_book):
                pos = next((p for p in self._db_positions() if p["symbol"] == sym), None)
                if pos is None:
                    self.xsec_book.pop(sym, None)
                elif targets.get(sym) != pos["side"]:
                    log.info("xsec: closing %s %s (no longer targeted)", pos["side"], sym)
                    await self.close_position(pos, "rebalance")
        longs = [s for s, v in targets.items() if v == "long"]
        shorts = [s for s, v in targets.items() if v == "short"]
        legs: list[tuple[str, str]] = []
        for i in range(max(len(longs), len(shorts))):          # alternate so a binding cap cannot starve one side
            if i < len(longs):
                legs.append((longs[i], "long"))
            if i < len(shorts):
                legs.append((shorts[i], "short"))
        for sym, side in legs:
            if self.xsec_book.get(sym) == side:
                continue
            if any(p["symbol"] == sym for p in self._db_positions()):
                log.info("xsec: %s already has a position outside the book; leg skipped", sym)
                continue
            ok, why = self.live_gate.check(sym, self.clock(), gate_cfg)
            if not ok:
                log.info("xsec: leg %s %s skipped: %s", side, sym, why)
                continue
            sc = scores[sym]
            prob = float(sc.prob_long if side == "long" else sc.prob_short)
            if await self.try_open(sym, side, prob, reasons_to_dicts(sc.reasons), frames[sym].kl, neutral=True, spec=spec,
                                   entry=ent):
                self.xsec_book[sym] = side
        self.store.kv_set("xsec_book", self.xsec_book)

    async def xsec_shadow(self, name: str, info: ModelInfo, frames: dict[str, FrameBundle], scores: dict[str, Score],
                          targets: dict[str, str]) -> None:
        spec = self.model_spec(info)
        ent = self.model_entry(info)
        gate = self.shadow_gate(name)
        now = self.clock()
        for key, p in [(k, p) for k, p in self.shadow.pos.items() if p.model == name]:
            if targets.get(p.symbol) != p.side:
                px = self.marks.get(p.symbol) or (float(frames[p.symbol].kl["close"].iloc[-1]) if p.symbol in frames else p.entry)
                tr = self.shadow.close_leg(name, p.symbol, px, now, "rebalance")
                if tr:
                    self._record_shadow(tr)
        for sym, side in targets.items():
            if self.shadow.has(name, sym):
                continue
            ok, _ = gate.check(sym, now, replace(ent, cooldown_bars=0))
            if not ok:
                continue
            closed = frames[sym].kl
            spec = bind_exit(self.model_spec(info), closed, self.now_ms())
            if spec is None:
                continue
            f = await self.io(self.ex.filters, sym)
            sc = scores[sym]
            px = self.marks.get(sym) or float(closed["close"].iloc[-1])
            maker = ent.entry_mode == "maker"
            if self.shadow.open(name, sym, side, float(sc.prob_long if side == "long" else sc.prob_short),
                                float(closed["close"].iloc[-1]) if maker else px, now,
                                exit_atr(spec, closed, self.now_ms()), f, spec=spec, entry=ent,
                                decision_open_ms=int(closed["open_time"].iloc[-1])):
                gate.record_entry(sym, now)

    # ================================================================== universe
    async def universe_cycle(self) -> None:
        from . import universe as uni

        cfg = self.cfg
        forced, self.force_universe = self.force_universe, False
        if not forced and self.universe and self.clock() - self.universe_at < cfg.universe_refresh_s:
            return
        cur = {r["symbol"]: r["in_universe_since"] for r in self.store.get_universe()}
        venue_info = None
        dropped: list[str] = []
        if not self.paper:
            try:
                venue_info = await self.io(self.ex.exchange_info)
            except ExchangeError as e:
                log.warning("order-venue exchangeInfo unavailable (%s); keeping the current universe", e)
                self.universe_at = self.clock() - cfg.universe_refresh_s + 60
                return
        rows = await self.io(uni.refresh, self.data, cfg, cur, self.clock(), venue_info, dropped)
        if dropped:
            self.dropped_venue = dropped
            log.warning("universe: dropped %s (hot on prod, not TRADING on the %s venue)", ", ".join(dropped), self.mode)
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

    async def depth_cycle(self) -> None:
        """Refresh one symbol's order book per call (the one refreshed longest ago, at most every ~10 s).
        Best effort: failures are logged at debug level and never touch trading state or fault counters."""
        from . import orderbook as ob

        syms = [s for s in self.universe + [p["symbol"] for p in self.store.get_positions()] if s]
        syms = list(dict.fromkeys(syms))
        if not syms:
            return
        now = self.clock()
        sym = min(syms, key=lambda s: self.depth_at.get(s, 0.0))
        if now - self.depth_at.get(sym, 0.0) < DEPTH_MIN_AGE_S:
            return
        used = getattr(self.data, "used_weight", 0)
        if used > DEPTH_WEIGHT_CEILING:                      # keep headroom for klines/premium/orders
            return
        self.depth_at[sym] = now
        try:
            bids, asks = await self.io(self.data.depth, sym, 1000)
        except RateLimitError:
            raise
        except ExchangeError as e:
            self.depth_stats["err"] += 1
            log.debug("depth %s failed: %s", sym, e)
            return
        snap = ob.compute_snapshot(bids, asks, sym, now)
        if snap is None:
            return
        self.depth_stats["ok"] += 1
        self.store.set_orderbook(sym, snap)
        ob.record_snapshot(self.store, snap)
        self.event("orderbook", snap)

    async def derivs_cycle(self) -> None:
        if isinstance(self.scorer, NullScorer) and not self.challengers:
            return
        due = [s for s in self.trade_symbols() if self.clock() - self.feed.metrics_at.get(s, 0) > 300]
        if due:
            await self.io(self.feed.refresh, due[0], self.deriv_lookback_h())

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

    def capacity(self) -> dict:
        """Portfolio usage + how many template slices still fit (open + fit = effective ``max_positions``)."""
        cfg = self.cfg
        b = self.book()
        n_open = len(self._db_positions())
        eq = self.equity if self.equity > 0 else float("nan")
        ref = next((e[1] for e in self.brackets.values() if e[1]), None)
        return {"max_positions": n_open + slots_fit(cfg, b, n_open, ref),
                "total_leverage": b.open_notional / eq if eq == eq else 0.0, "max_total_leverage": cfg.max_total_leverage,
                "margin_ratio": b.open_mm / eq if eq == eq else 0.0, "max_margin_ratio": cfg.max_margin_ratio,
                "open_risk_pct": b.open_risk / eq * 100.0 if eq == eq else 0.0, "max_open_risk_pct": cfg.max_open_risk_pct}

    def models_status(self) -> list[dict]:
        """Loaded models for the status page: kind, role, history need, decision cadence, entry / exit template."""
        out = []
        now_ms = self.now_ms()
        for name, _, info, is_champ in self.all_models():
            ent = self.model_entry(info)
            spec = self.model_spec(info)
            row: dict[str, Any] = {"name": name, "kind": info.kind, "role": "champion" if is_champ else "challenger",
                                   "history_bars_5m": info.history_bars_5m, "timeframes": list(info.timeframes),
                                   "entry_mode": ent.entry_mode, "decision_every_bars": ent.decision_every_bars,
                                   "cooldown_bars": ent.cooldown_bars, "max_trades_per_day": ent.max_trades_per_day,
                                   "exit_mode": spec.mode if spec.mode != "atr" else f"atr:{spec.atr_tf}"}
            if info.kind == "xsec":
                xc = XsecCfg.resolve(info.entry_cfg, info.meta)
                row.update(rebalance_every_h=xc.rebalance_every_h, top_k=xc.top_k, bottom_k=xc.bottom_k,
                           next_decision_t=next_rebalance_ms(now_ms, xc.rebalance_every_h) / 1000.0)
            else:
                row["next_decision_t"] = next_decision_ms(now_ms, ent.decision_every_bars) / 1000.0
            sh = self.shadow.stats.get(name)
            if sh:
                row["shadow_maker"] = dict(sh)
            out.append(row)
        return out

    def decisions_status(self) -> dict:
        """Champion decision cadence / entry limits for the status page."""
        info = self.model_info
        active = info is not None and not isinstance(self.scorer, NullScorer)
        ent = self.model_entry(info if active else None)
        now_ms = self.now_ms()
        nxt = None
        if active:
            assert info is not None
            nxt = (next_rebalance_ms(now_ms, XsecCfg.resolve(info.entry_cfg, info.meta).rebalance_every_h) if info.kind == "xsec"
                   else next_decision_ms(now_ms, ent.decision_every_bars)) / 1000.0
        return {"kind": info.kind if active and info else None, "next_decision_t": nxt,
                "decision_every_bars": ent.decision_every_bars, "cooldown_bars": ent.cooldown_bars,
                "trades_today": self.live_gate.trades_today(self.clock()), "max_trades_per_day": ent.max_trades_per_day,
                "entry_mode": ent.entry_mode, "history_bars_5m": self.hist_need,
                "history_ready": sum(1 for s in self.universe if len(self.kl.get(s, ())) >= self.hist_need)}

    def maker_status(self) -> dict:
        m = dict(self.maker_stats)
        m["fill_rate"] = (m["filled"] / m["entries"]) if m["entries"] else None
        m["in_flight"] = sorted(self.pending)
        return m

    def write_status(self) -> None:
        cfg = self.cfg
        self.store.kv_set("status", {
            "models": self.models_status(), "decisions": self.decisions_status(), "maker": self.maker_status(),
            "xsec_legs": dict(self.xsec_book),
            "mode": self.mode, "running": self.running, "equity": self.equity, "wallet_equity": self.wallet_equity,
            "capital_usd": self.cfg.capital_usd if cap.is_active(self.cfg, self.mode) else None,
            "start_equity": self.start_equity,
            "peak_equity": self.peak_equity, "cash": self.cash, "unrealized_pnl": self.upnl,
            **self.capacity(),
            "kill": {"tripped": bool(self.kill.get("tripped")), "reason": self.kill.get("reason", ""),
                     "dd_limit_pct": cfg.kill_limit(self.mode)},
            "model": {"name": self.model_info.name if self.model_info else "null",
                      "version": self.model_info.version if self.model_info else "0",
                      "trained_at": self.model_info.trained_at if self.model_info else None,
                      "kind": self.model_info.kind if self.model_info else None,
                      "threshold_long": self.model_thresholds(self.model_info)[0],
                      "threshold_short": self.model_thresholds(self.model_info)[1]},
            "universe_updated_at": int(self.universe_at) if self.universe_at else None,
            "t": self.clock()})

    async def heartbeat_cycle(self) -> None:
        new_cfg = self.store.get_settings()
        changed = new_cfg.capital_usd != self.cfg.capital_usd
        self.cfg = new_cfg
        if changed and not self.paper:
            await self.io(self._refresh_account)
            self.rebase_equity("capital_usd changed")
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
        for sy in {p["symbol"] for p in self._db_positions()}:          # keep open positions' tiers fresh (status MM)
            await self.get_brackets(sy)
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
            s_day = self.store.one("SELECT COUNT(CASE WHEN exit_reason!='partial' THEN 1 END) n, COALESCE(SUM(pnl),0) net, "
                                   "COALESCE(SUM(CASE WHEN pnl>0 AND exit_reason!='partial' THEN 1 ELSE 0 END),0) w FROM trades "
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
            ("derivs", 20.0, self.derivs_cycle), ("depth", 1.0, self.depth_cycle)]]
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
    from .alerts import store_stats
    alerts = AlertDispatcher(stats=store_stats(store))   # every alert carries equity/PnL/drawdown/open
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
