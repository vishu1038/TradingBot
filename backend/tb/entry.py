"""Entry controls (V3): decision cadence, per-coin cooldown, max trades per day, maker/taker mode.

Resolved per model from its ``entry_cfg`` with the engine ``Settings`` as defaults; pure logic (no I/O).
Decision points are aligned to the UTC grid so every process / restart agrees on them: with
``decision_every_bars = N`` a model decides on the 5m bar that CLOSES at a multiple of N*5 minutes
(N = 12 -> every full hour, N = 48 -> every 4 hours).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .config import Settings

BAR_MS = 300_000


def _int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f != int(f):
        return default
    return int(f) if lo <= int(f) <= hi else default


@dataclass(frozen=True)
class EntryCfg:
    decision_every_bars: int = 1
    cooldown_bars: int = 0
    max_trades_per_day: int | None = None
    entry_mode: str = "taker"
    maker_timeout_s: int = 60
    maker_reprices: int = 2
    maker_on_timeout: str = "skip"
    sl_cooldown_bars: int = 0           # per coin: no new entry for N bars after a LOSING sl/trail/kill/timeout/liquidation exit
    max_open: int | None = None         # entry_cfg only: concurrent positions this model may hold (null = unlimited)

    @classmethod
    def resolve(cls, cfg: Settings, entry_cfg: Mapping[str, Any] | None = None) -> "EntryCfg":
        """Settings defaults overlaid with the model's ``entry_cfg`` (per key; invalid values are ignored)."""
        e = dict(entry_cfg or {})
        if "max_trades_per_day" in e:
            mtd = e["max_trades_per_day"]
            mtd = None if mtd is None else (_int(mtd, 0, 1, 1000) or cfg.max_trades_per_day)
        else:
            mtd = cfg.max_trades_per_day
        mode = e.get("entry_mode", cfg.entry_mode)
        on_to = e.get("maker_on_timeout", cfg.maker_on_timeout)
        return cls(
            decision_every_bars=_int(e.get("decision_every_bars"), cfg.decision_every_bars, 1, 4032),
            cooldown_bars=_int(e.get("cooldown_bars"), cfg.cooldown_bars, 0, 100_000),
            max_trades_per_day=mtd,
            entry_mode=mode if mode in ("taker", "maker") else cfg.entry_mode,
            maker_timeout_s=_int(e.get("maker_timeout_s"), cfg.maker_timeout_s, 5, 900),
            maker_reprices=_int(e.get("maker_reprices"), cfg.maker_reprices, 0, 10),
            maker_on_timeout=on_to if on_to in ("skip", "taker") else cfg.maker_on_timeout,
            sl_cooldown_bars=_int(e.get("sl_cooldown_bars"), cfg.sl_cooldown_bars, 0, 100_000),
            max_open=_int(e.get("max_open"), 0, 1, 1000) or None)


@dataclass(frozen=True)
class XsecCfg:
    rebalance_every_h: int = 24
    top_k: int = 3
    bottom_k: int = 3
    horizon_h: int = 24

    @classmethod
    def resolve(cls, entry_cfg: Mapping[str, Any] | None, meta: Mapping[str, Any] | None = None) -> "XsecCfg":
        """From the artifact: ``xsec_cfg`` (or the top-level fields) of the model json, else ``entry_cfg``."""
        m = dict(meta or {})
        e = {**dict(entry_cfg or {}), **{k: m[k] for k in ("rebalance_every_h", "top_k", "bottom_k", "horizon_h") if k in m},
             **dict(m.get("xsec_cfg") or {})}
        d = cls()
        return cls(_int(e.get("rebalance_every_h"), d.rebalance_every_h, 1, 24 * 14), _int(e.get("top_k"), d.top_k, 0, 50),
                   _int(e.get("bottom_k"), d.bottom_k, 0, 50), _int(e.get("horizon_h"), d.horizon_h, 1, 24 * 30))


def is_decision_bar(open_time_ms: int, every: int) -> bool:
    """True when the 5m bar opened at ``open_time_ms`` closes on a multiple of ``every`` bars (UTC grid)."""
    return every <= 1 or ((int(open_time_ms) // BAR_MS) + 1) % every == 0


def next_decision_ms(now_ms: int, every: int) -> int:
    """Close time (ms) of the next decision bar strictly after ``now_ms``."""
    step = max(1, int(every)) * BAR_MS
    return (int(now_ms) // step + 1) * step


def rebalance_slot(now_ms: int, every_h: int) -> int:
    return int(now_ms) // (max(1, int(every_h)) * 3_600_000)


def next_rebalance_ms(now_ms: int, every_h: int) -> int:
    step = max(1, int(every_h)) * 3_600_000
    return (int(now_ms) // step + 1) * step


def day_start(t: float) -> float:
    d = datetime.fromtimestamp(t, timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


class EntryGate:
    """Per-coin cooldown + entries-per-day bookkeeping for ONE book (the live account, or one shadow model).

    ``cooldown_bars`` counts from the coin's last entry or last exit, whichever is later."""

    def __init__(self) -> None:
        self.last: dict[str, float] = {}      # symbol -> last entry / exit time (s)
        self.last_loss: dict[str, float] = {}  # symbol -> last LOSING stop-type exit time (s): sl_cooldown_bars
        self.entries: list[float] = []        # entry times (s), pruned to the current UTC day

    def seed(self, entries: Iterable[tuple[str, float]] = (), exits: Iterable[tuple] = ()) -> None:
        """``exits``: (symbol, time) or (symbol, time, was_a_losing_exit)."""
        for sym, t in entries:
            self.record_entry(sym, t)
        for ex in exits:
            self.record_exit(ex[0], ex[1], loss=bool(ex[2]) if len(ex) > 2 else False)

    def record_entry(self, sym: str, t: float) -> None:
        self.last[sym] = max(self.last.get(sym, 0.0), float(t))
        self.entries.append(float(t))

    def record_exit(self, sym: str, t: float, loss: bool = False) -> None:
        self.last[sym] = max(self.last.get(sym, 0.0), float(t))
        if loss:
            self.last_loss[sym] = max(self.last_loss.get(sym, 0.0), float(t))

    def trades_today(self, now: float) -> int:
        d0 = day_start(now)
        self.entries = [t for t in self.entries if t >= d0]
        return len(self.entries)

    def check(self, sym: str, now: float, cfg: EntryCfg, n_open: int = 0) -> tuple[bool, str]:
        """``n_open`` = positions (and in-flight entries) the book holds right now, for ``cfg.max_open``."""
        if cfg.max_open is not None and n_open >= cfg.max_open:
            return False, f"max_open {cfg.max_open} positions held"
        if cfg.max_trades_per_day is not None and self.trades_today(now) >= cfg.max_trades_per_day:
            return False, f"max_trades_per_day {cfg.max_trades_per_day} reached"
        if cfg.cooldown_bars > 0:
            left = self.last.get(sym, 0.0) + cfg.cooldown_bars * BAR_MS / 1000.0 - now
            if left > 0:
                return False, f"{sym} cooling down ({left / 60:.0f} min left of {cfg.cooldown_bars} bars)"
        if cfg.sl_cooldown_bars > 0 and sym in self.last_loss:
            left = self.last_loss[sym] + cfg.sl_cooldown_bars * BAR_MS / 1000.0 - now
            if left > 0:
                return False, (f"{sym} cooling down after a losing exit ({left / 60:.0f} min left of "
                               f"{cfg.sl_cooldown_bars} bars)")
        return True, ""
