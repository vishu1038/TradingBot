"""Adaptive decision cadence (V3, ``entry_cfg.decision_mode = "adaptive"``): pure logic, no I/O.

A model is SCORED on every closed 5m bar but may only ENTER on a *decision point* of that coin:

* REGULAR: ``regular_gap`` bars after the coin's previous regular decision, where
  ``regular_gap = clamp(round(base_gap_bars * median_vol / current_vol), min_regular_gap_bars, max_regular_gap_bars)``
  (``current_vol`` = std of the last 12 5m log-returns, ``median_vol`` = the median of that statistic over the trailing
  7 days): a volatile coin is looked at often, a quiet one rarely. Normal probability threshold.
* EVENT, between regular looks, on the coin's newest closed 5m bar: range >= ``ev_range_atr`` x 5m ATR14, volume z-score
  (48 bars) >= ``ev_volume_z``, close beyond the prior closed 1h bar's high / low (``ev_breakout``), open interest moved
  >= ``ev_oi_jump_pct`` % over 3 bars, or the side's probability rose >= ``ev_prob_jump`` since the coin's previous
  decision point. An event entry additionally needs ``prob >= threshold + ev_extra_conf``.

``min_gap_bars`` bounds how soon after ANY decision point the same coin may have another one. The live engine and the shadow
simulator call the very same ``AdaptiveState.step`` (one state per book), so they cannot diverge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import _ms

BAR_MS = 300_000
HOUR_MS = 3_600_000
VOL_WINDOW = 12                 # bars of the realised-volatility statistic
VOL_HISTORY = 7 * 288           # trailing 7 days of that statistic for the median
MIN_VOL_SAMPLES = 288           # fewer than a day of windows -> base gap (median not trustworthy)
OI_STALE_MS = 30 * 60_000
TAIL = 400                      # bars the event detectors look at (>= 24 bars of the prior hour + 48 volume + ATR)


# ----------------------------------------------------------------------------- regular gap
def regular_gap(closed: pd.DataFrame, base: int, lo: int, hi: int) -> tuple[int, float | None]:
    """(gap in 5m bars, vol ratio median/current or None when it could not be measured)."""
    base, lo, hi = int(base), int(lo), max(int(lo), int(hi))
    c = closed["close"].to_numpy("float64")[-(VOL_HISTORY + VOL_WINDOW + 1):]
    if len(c) < VOL_WINDOW + MIN_VOL_SAMPLES:
        return min(max(base, lo), hi), None
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(c))
    if not np.all(np.isfinite(r)):
        return min(max(base, lo), hi), None
    sd = np.lib.stride_tricks.sliding_window_view(r, VOL_WINDOW).std(axis=1)
    cur, med = float(sd[-1]), float(np.median(sd))
    if not (med > 0 and math.isfinite(med) and math.isfinite(cur)):
        return min(max(base, lo), hi), None
    if cur <= 0:
        return hi, float("inf")
    ratio = med / cur
    return min(max(int(math.floor(base * ratio + 0.5)), lo), hi), ratio


# ----------------------------------------------------------------------------- events
@dataclass(frozen=True)
class Events:
    """Events on one bar: ``common`` (side-agnostic) and per side (breakout)."""
    common: dict[str, float] = field(default_factory=dict)
    long: dict[str, float] = field(default_factory=dict)
    short: dict[str, float] = field(default_factory=dict)

    def for_side(self, side: str) -> dict[str, float]:
        return {**self.common, **(self.long if side == "long" else self.short)}

    def __bool__(self) -> bool:
        return bool(self.common or self.long or self.short)


def bar_events(closed: pd.DataFrame, metrics: pd.DataFrame | None, ent: Any) -> Events:
    """Events on the newest closed 5m bar of ``closed`` (all information is available at that bar's close)."""
    common: dict[str, float] = {}
    lg: dict[str, float] = {}
    sh: dict[str, float] = {}
    k = closed.tail(TAIL)
    n = len(k)
    if n < 16:
        return Events()
    o_t = k["open_time"].to_numpy("int64")
    h = k["high"].to_numpy("float64")
    l = k["low"].to_numpy("float64")
    c = k["close"].to_numpy("float64")
    v = k["volume"].to_numpy("float64")
    t0 = int(o_t[-1])
    if ent.ev_range_atr > 0:
        pc = c[-15:-1]
        tr = np.maximum(h[-15:-1] - l[-15:-1], np.maximum(np.abs(h[-15:-1] - np.concatenate([[c[-16]], pc[:-1]])),
                                                          np.abs(l[-15:-1] - np.concatenate([[c[-16]], pc[:-1]]))))
        atr = float(tr.mean())
        if atr > 0 and math.isfinite(atr) and (h[-1] - l[-1]) >= ent.ev_range_atr * atr:
            common["range_atr"] = float((h[-1] - l[-1]) / atr)
    if ent.ev_volume_z > 0 and n >= 49:
        pv = v[-49:-1]
        sd = float(pv.std())
        if sd > 0 and math.isfinite(sd):
            z = float((v[-1] - pv.mean()) / sd)
            if z >= ent.ev_volume_z:
                common["volume_z"] = z
    if ent.ev_breakout:
        hs = (t0 // HOUR_MS) * HOUR_MS                       # start of the hour this bar belongs to
        ph = (o_t >= hs - HOUR_MS) & (o_t < hs)
        if int(ph.sum()) == 12:                              # the prior 1h bar must be complete
            if c[-1] > float(h[ph].max()):
                lg["breakout"] = 1.0
            if c[-1] < float(l[ph].min()):
                sh["breakout"] = 1.0
    if ent.ev_oi_jump_pct > 0 and metrics is not None and len(metrics) and "sum_open_interest" in metrics:
        ct = _ms(metrics["create_time"])
        oi = metrics["sum_open_interest"].to_numpy("float64")
        order = np.argsort(ct, kind="stable")
        ct, oi = ct[order], oi[order]
        i1 = int(np.searchsorted(ct, t0, side="right")) - 1
        i0 = int(np.searchsorted(ct, t0 - 3 * BAR_MS, side="right")) - 1
        if i1 >= 0 and i0 >= 0 and i1 != i0 and t0 - ct[i1] <= OI_STALE_MS and oi[i0] > 0 and math.isfinite(oi[i1]):
            ch = abs(float(oi[i1] / oi[i0] - 1.0)) * 100.0
            if ch >= ent.ev_oi_jump_pct:
                common["oi_jump_pct"] = ch
    return Events(common, lg, sh)


def describe(ev: dict[str, float]) -> str:
    return ", ".join("breakout" if k == "breakout" else f"{k}={v:.3g}" for k, v in ev.items())


# ----------------------------------------------------------------------------- decisions
@dataclass
class Decision:
    kind: str | None = None            # "regular" | "event" | None (not a decision point)
    side: str | None = None            # entry side, None = looked, did not qualify
    prob: float = 0.0
    reason: str = ""
    events: dict[str, float] = field(default_factory=dict)
    gap: int | None = None             # the coin's regular gap (bars) when a regular decision was made
    gap_ratio: float | None = None


@dataclass
class CoinState:
    last_regular_ms: int
    last_decision_ms: int
    last_prob: dict[str, float | None] = field(default_factory=lambda: {"long": None, "short": None})


class AdaptiveState:
    """Per-book (the live champion, or one shadow model) adaptive-cadence bookkeeping, per coin."""

    def __init__(self) -> None:
        self.coin: dict[str, CoinState] = {}
        self.last: dict[str, Any] | None = None      # newest decision point of any coin
        self.counts: dict[str, int] = {"regular": 0, "event": 0, "entries_regular": 0, "entries_event": 0}

    def step(self, sym: str, closed: pd.DataFrame, metrics: pd.DataFrame | None, pl: float, ps: float, ent: Any,
             thr_l: float, thr_s: float, short_enabled: bool = True) -> Decision:
        """Is the newest closed bar of ``sym`` a decision point, and which side (if any) would it enter?  Mutates the
        coin's state only when it IS a decision point."""
        bar = int(closed["open_time"].iloc[-1])
        cs = self.coin.get(sym)
        if cs is None:                                # first sight (start / restart): the previous UTC-grid close of the base gap
            step = max(1, ent.base_gap_bars) * BAR_MS     # counts as the last regular look (a bar CLOSING on the grid is one)
            g0 = (bar + BAR_MS - 1) // step * step - BAR_MS
            cs = self.coin[sym] = CoinState(g0, g0)
        if (bar - cs.last_decision_ms) // BAR_MS < ent.min_gap_bars:
            return Decision()
        if (bar - cs.last_regular_ms) // BAR_MS >= ent.min_regular_gap_bars:
            gap, ratio = regular_gap(closed, ent.base_gap_bars, ent.min_regular_gap_bars, ent.max_regular_gap_bars)
            if (bar - cs.last_regular_ms) // BAR_MS >= gap:
                side, p = _pick(pl, ps, thr_l, thr_s, short_enabled)
                self._commit(sym, cs, bar, pl, ps, "regular")
                rs = f"regular gap={gap} bars" + (f" (vol x{ratio:.2f})" if ratio is not None and math.isfinite(ratio) else "")
                self._note(sym, bar, "regular", side, p, rs)
                return Decision("regular", side, p, rs, {}, gap, ratio)
        ev_c = bar_events(closed, metrics, ent)
        per = {"long": dict(ev_c.for_side("long")), "short": dict(ev_c.for_side("short"))}
        for sd, p in (("long", pl), ("short", ps)):
            last = cs.last_prob.get(sd)
            if ent.ev_prob_jump > 0 and last is not None and p - last >= ent.ev_prob_jump:
                per[sd]["prob_jump"] = float(p - last)
        if not per["long"] and not per["short"]:
            return Decision()
        best: tuple[str, float] | None = None
        for sd, p, thr in (("long", pl, thr_l), ("short", ps, thr_s)):
            if sd == "short" and not short_enabled:
                continue
            if per[sd] and p >= thr + ent.ev_extra_conf and (best is None or p > best[1]):
                best = (sd, p)
        self._commit(sym, cs, bar, pl, ps, "event")
        if best is None:
            ev_all = {**per["long"], **{k: v for k, v in per["short"].items() if k not in per["long"]}}
            self._note(sym, bar, "event", None, 0.0, "event " + describe(ev_all) + " (below threshold + extra)")
            return Decision("event", None, 0.0, "event " + describe(ev_all), ev_all)
        reason = "event " + describe(per[best[0]])
        self._note(sym, bar, "event", best[0], best[1], reason)
        return Decision("event", best[0], best[1], reason, per[best[0]])

    def _commit(self, sym: str, cs: CoinState, bar: int, pl: float, ps: float, kind: str) -> None:
        cs.last_decision_ms = bar
        if kind == "regular":
            cs.last_regular_ms = bar
        cs.last_prob = {"long": float(pl), "short": float(ps)}
        self.counts[kind] += 1

    def _note(self, sym: str, bar: int, kind: str, side: str | None, prob: float, reason: str) -> None:
        self.last = {"t": (bar + BAR_MS) / 1000.0, "symbol": sym, "kind": kind, "action": side or "none",
                     "prob": prob, "reason": reason}

    def entered(self, kind: str) -> None:
        self.counts["entries_" + kind] += 1

    def summary(self) -> dict[str, Any]:
        return {"last": dict(self.last) if self.last else None, **self.counts}


def _pick(pl: float, ps: float, tl: float, ts: float, short_enabled: bool) -> tuple[str | None, float]:
    cand_l = pl >= tl
    cand_s = short_enabled and ps >= ts
    if cand_l and (not cand_s or pl >= ps):
        return "long", pl
    if cand_s:
        return "short", ps
    return None, 0.0
