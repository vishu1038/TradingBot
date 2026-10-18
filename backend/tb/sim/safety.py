"""Safety & robustness layer as the backtester applies it (mirrors the live engine).

* derate_multiplier  - drawdown derating curve; imported from tb.risk (the single source of truth),
                       with an identical local fallback if tb.risk does not provide it.
* DriftMonitor       - last-N closed trades: z of realised win rate vs mean predicted probability.
* symbol fault state - bar plausibility (tb.data.quality) + debouncing, per symbol.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from tb.data import quality

try:  # single source of truth lives in tb.risk
    from tb.risk import derate_multiplier as _risk_derate  # type: ignore[attr-defined]
    DERATE_SOURCE = "tb.risk"
except Exception:  # noqa: BLE001 - ImportError/AttributeError, or tb.risk failing to import
    _risk_derate = None
    DERATE_SOURCE = "local-fallback"


def _fallback_derate(dd_pct: float, kill_dd_pct: float) -> float:
    """1.0 until 40% of the kill limit, linear to 0.5 at 80%, linear to 0 at 100%."""
    if kill_dd_pct <= 0:
        return 1.0
    f = max(0.0, dd_pct) / kill_dd_pct
    if f <= 0.4:
        return 1.0
    if f <= 0.8:
        return 1.0 - (f - 0.4) / 0.4 * 0.5
    if f < 1.0:
        return 0.5 - (f - 0.8) / 0.2 * 0.5
    return 0.0


def derate_multiplier(dd_pct: float, kill_dd_pct: float) -> float:
    if _risk_derate is not None:
        return float(_risk_derate(dd_pct, kill_dd_pct))
    return _fallback_derate(dd_pct, kill_dd_pct)


class DriftMonitor:
    """Drift ("slip") detection over the last `n` closed trades.

    predicted = mean model probability of the taken side, realised = win rate,
    z = (realised - predicted) / sqrt(p (1 - p) / n)   with p = predicted (clipped to [.01, .99]).
    state 0 normal (x1.0) | 1 half size (z < -2) | 2 blocked, no entries (z < -3).
    Hysteresis: from state 2 it needs z > -2 to drop to 1, and z > -1 to return to 0.
    Needs a full window of n trades before it can fire.
    """

    def __init__(self, n: int = 30):
        self.n = n
        self.buf: deque[tuple[float, bool]] = deque(maxlen=n)
        self.state = 0
        self.z: float | None = None
        self.fired_half = 0       # transitions into state 1
        self.fired_block = 0      # transitions into state 2
        self.bars_half = 0
        self.bars_blocked = 0

    def add(self, prob: float, win: bool) -> None:
        self.buf.append((float(prob), bool(win)))
        if len(self.buf) < self.n:
            self.z = None
            return
        pred = float(np.mean([p for p, _ in self.buf]))
        real = float(np.mean([w for _, w in self.buf]))
        p = min(max(pred, 0.01), 0.99)
        self.z = (real - pred) / math.sqrt(p * (1 - p) / self.n)
        prev = self.state
        z = self.z
        if z < -3:
            self.state = 2
        elif z < -2:
            self.state = 2 if prev == 2 else 1  # blocked stays blocked until z > -2
        elif z > -1:
            self.state = 0
        else:  # -2 <= z <= -1: hysteresis band
            self.state = 1 if prev in (1, 2) else 0
        if self.state == 1 and prev != 1:
            self.fired_half += 1
        if self.state == 2 and prev != 2:
            self.fired_block += 1

    @property
    def multiplier(self) -> float:
        return {0: 1.0, 1: 0.5, 2: 0.0}[self.state]

    @property
    def blocked(self) -> bool:
        return self.state == 2


def symbol_fault_state(kl, atr: np.ndarray, mature_n: int, recover_m: int) -> tuple[np.ndarray, dict]:
    """(excluded[N] bool: symbol excluded from entries as of bar i's close, counters)."""
    f = quality.fault_mask(kl, atr)
    exc = quality.debounce(f, mature_n, recover_m)
    # episodes = rising edges of the matured state
    ep = int(((exc[1:] & ~exc[:-1]).sum()) + (1 if len(exc) and exc[0] else 0)) if len(exc) else 0
    return exc, {"fault_bars": int(f.sum()), "episodes": ep, "excluded_bars": int(exc.sum())}
