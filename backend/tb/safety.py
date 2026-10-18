"""Safety & robustness primitives shared by the engine and the guard (pure, unit-tested).

* safety states NORMAL < DEGRADED < FLATTEN < HALTED and how several sources are merged
* ``Debouncer``: a fault matures after N consecutive bad cycles, clears after M good ones
* ``FaultTracker``: per-symbol / exchange-wide faults on top of Debouncer
* data plausibility checks (reject a bar / tick before it is used)
* ``DriftMonitor``: z-score of realised vs predicted win rate over the last N closed trades
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

BAR_MS = 300_000

NORMAL, DEGRADED, FLATTEN, HALTED = "NORMAL", "DEGRADED", "FLATTEN", "HALTED"
STATES = (NORMAL, DEGRADED, FLATTEN, HALTED)
_RANK = {s: i for i, s in enumerate(STATES)}


def worst(*states: str) -> str:
    """Most severe of the given safety states (unknown strings are ignored)."""
    best = NORMAL
    for s in states:
        if s in _RANK and _RANK[s] > _RANK[best]:
            best = s
    return best


def blocks_entries(state: str) -> bool:
    return state != NORMAL


# ----------------------------------------------------------------------------- debounce
@dataclass
class _Fault:
    bad: int = 0
    good: int = 0
    active: bool = False
    since: float | None = None
    count: int = 0           # total bad cycles observed while active or maturing


class Debouncer:
    """A fault matures after ``mature_n`` consecutive bad cycles and clears after ``recover_m``
    consecutive good cycles (hysteresis: a single good cycle does not clear an active fault, a
    single bad cycle does not mature one)."""

    def __init__(self, mature_n: int = 3, recover_m: int = 5):
        self.n = mature_n
        self.m = recover_m
        self._f: dict[Any, _Fault] = {}

    def observe(self, key: Any, bad: bool, now: float | None = None) -> bool:
        """Record one cycle; returns whether the fault is currently active (matured)."""
        f = self._f.setdefault(key, _Fault())
        if bad:
            f.bad += 1
            f.good = 0
            f.count += 1
            if not f.active and f.bad >= self.n:
                f.active = True
                f.since = now if now is not None else time.time()
        else:
            f.good += 1
            f.bad = 0
            if f.active and f.good >= self.m:
                f.active = False
                f.since = None
                f.count = 0
            elif not f.active:
                f.count = 0
        return f.active

    def active(self, key: Any) -> bool:
        f = self._f.get(key)
        return bool(f and f.active)

    def pending(self, key: Any) -> int:
        """Consecutive bad cycles so far (for the 'warn' stage before maturation)."""
        f = self._f.get(key)
        return f.bad if f else 0

    def active_items(self) -> list[tuple[Any, _Fault]]:
        return [(k, f) for k, f in self._f.items() if f.active]


class FaultTracker:
    """Per-symbol faults (warn -> exclude from entries) and exchange-wide faults."""

    EXCHANGE = "*"

    def __init__(self, mature_n: int = 3, recover_m: int = 5):
        self.deb = Debouncer(mature_n, recover_m)

    def observe_symbol(self, symbol: str, kinds_bad: Sequence[str], all_kinds: Sequence[str] | None = None,
                       now: float | None = None) -> None:
        """Feed one cycle for ``symbol``: kinds in ``kinds_bad`` are bad, every other known kind is good."""
        kinds = set(all_kinds or ()) | set(kinds_bad)
        for k in kinds:
            self.deb.observe((symbol, k), k in kinds_bad, now)

    def observe_exchange(self, bad: bool, now: float | None = None) -> bool:
        return self.deb.observe((self.EXCHANGE, "exchange"), bad, now)

    def excluded(self, symbol: str) -> bool:
        return any(self.deb.active((s, k)) for (s, k), _ in self.deb.active_items() if s == symbol)

    def exchange_fault(self) -> bool:
        return self.deb.active((self.EXCHANGE, "exchange"))

    def warnings(self) -> list[tuple[str, str, int]]:
        """(symbol, kind, consecutive_bad) for faults seen but not yet matured."""
        out = []
        for (s, k), f in self.deb._f.items():
            if f.bad and not f.active:
                out.append((s, k, f.bad))
        return out

    def faults(self) -> list[dict]:
        """Active faults in API shape: [{symbol, kind, count, since}]."""
        return [{"symbol": s if s != self.EXCHANGE else "*", "kind": k, "count": f.count,
                 "since": int(f.since) if f.since else None}
                for (s, k), f in self.deb.active_items()]


# ----------------------------------------------------------------------------- plausibility
def check_bars(open_time: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
               v: np.ndarray, atr: float | None, now_ms: int, *, max_ret_pct: float = 15.0,
               max_ret_atr: float = 8.0, stale_bars: int = 2) -> list[str]:
    """Plausibility faults of the most recent bars (last 2 are inspected). Empty list = good.

    kinds: bad_price (non-positive / non-finite), high_low (high < low), zero_volume, big_move
    (|return| > max(15%, 8xATR14/price)), stale (newest bar older than ``stale_bars`` bars)."""
    faults: list[str] = []
    n = len(c)
    if n == 0:
        return ["stale"]
    last = slice(max(0, n - 2), n)
    vals = np.concatenate([o[last], h[last], l[last], c[last]]).astype("float64")
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        faults.append("bad_price")
    if np.any(h[last] < l[last]):
        faults.append("high_low")
    # the forming candle can legitimately have no volume yet; judge the last *closed* bar
    closed_idx = n - 1
    if int(open_time[-1]) + BAR_MS > now_ms and n >= 2:
        closed_idx = n - 2
    if v[closed_idx] <= 0:
        faults.append("zero_volume")
    if n >= 2 and c[n - 2] > 0:
        ret = abs(float(c[n - 1]) / float(c[n - 2]) - 1.0) * 100.0
        lim = max(max_ret_pct, (max_ret_atr * atr / float(c[n - 2]) * 100.0) if atr else 0.0)
        if ret > lim:
            faults.append("big_move")
    if now_ms - int(open_time[-1]) > (stale_bars + 1) * BAR_MS:
        faults.append("stale")
    return faults


def check_tick(mark: float | None, last: float | None, max_diff_pct: float = 1.0) -> list[str]:
    out = []
    for p in (mark, last):
        if p is not None and (not math.isfinite(p) or p <= 0):
            out.append("bad_price")
            return out
    if mark and last and abs(mark - last) / last * 100.0 > max_diff_pct:
        out.append("mark_last_diff")
    return out


# ----------------------------------------------------------------------------- drift
def drift_z(probs: Sequence[float], wins: Sequence[bool | int]) -> float | None:
    """(realised - predicted) / sqrt(p(1-p)/n) over the given trades; None when empty."""
    n = len(probs)
    if n == 0 or n != len(wins):
        return None
    p = float(np.clip(np.mean(probs), 0.01, 0.99))
    realised = float(np.mean([1.0 if w else 0.0 for w in wins]))
    return (realised - p) / math.sqrt(p * (1 - p) / n)


@dataclass
class DriftState:
    z: float | None = None
    n: int = 0
    level: int = 0           # 0 ok, 1 size x0.5 (z < -2), 2 entries blocked (z < -3)
    mult: float = 1.0


class DriftMonitor:
    """Hysteresis: the level rises as z falls below -2 / -3 and only clears once z > -1."""

    WARN, ALARM, CLEAR = -2.0, -3.0, -1.0

    def __init__(self, n: int = 30):
        self.n = n
        self.level = 0

    def update(self, probs: Sequence[float], wins: Sequence[bool | int]) -> DriftState:
        probs = list(probs)[-self.n:]
        wins = list(wins)[-self.n:]
        if len(probs) < self.n:
            # not enough evidence: keep the current level but never raise it
            return DriftState(None, len(probs), self.level, self._mult())
        z = drift_z(probs, wins)
        assert z is not None
        if z < self.ALARM:
            self.level = 2
        elif z < self.WARN:
            self.level = max(self.level, 1)
        elif z > self.CLEAR:
            self.level = 0
        return DriftState(z, len(probs), self.level, self._mult())

    def _mult(self) -> float:
        return {0: 1.0, 1: 0.5, 2: 0.5}[self.level]

    def entries_blocked(self) -> bool:
        return self.level >= 2


def slippage_drift(slips_pct: Sequence[float], assumed_pct: float, mult: float = 2.0) -> bool:
    """True when the mean measured adverse slippage exceeds ``mult`` x the sim assumption."""
    if len(slips_pct) < 5 or assumed_pct <= 0:
        return False
    return float(np.mean(slips_pct)) > mult * assumed_pct


# ----------------------------------------------------------------------------- merged safety view
def merged_safety(guard: Mapping[str, Any] | None, engine: Mapping[str, Any] | None,
                  guard_age_s: float | None = None, guard_required: bool = False,
                  guard_stale_s: float = 60.0) -> dict:
    """Combine the guard's and the engine's own safety views. The result is what the engine obeys
    before an entry and what ``/api/status.safety`` reports."""
    g = guard or {}
    e = engine or {}
    state = worst(g.get("state", NORMAL), e.get("state", NORMAL))
    reasons = list(g.get("reasons", [])) + list(e.get("reasons", []))
    if guard_required and (guard_age_s is None or guard_age_s > guard_stale_s):
        state = worst(state, DEGRADED)
        reasons.append("guard process not running" if guard_age_s is None else f"guard heartbeat stale {guard_age_s:.0f}s")
    drift = e.get("drift") or {"z": None, "n": 0, "mult": 1.0}
    return {"state": state, "reasons": reasons, "derate_mult": float(e.get("derate_mult", 1.0)),
            "drift": {"z": drift.get("z"), "n": drift.get("n", 0), "mult": drift.get("mult", 1.0)},
            "faults": list(e.get("faults", []))}
