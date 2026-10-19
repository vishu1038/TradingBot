"""Shared helpers for the sim tests (no tests here)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tb.sim.template import SimConfig

T0 = 1_767_225_600_000  # 2026-01-01 00:00:00 UTC
BAR = 300_000


def mk_klines(o, h=None, l=None, c=None, v=None, t0: int = T0) -> pd.DataFrame:
    o = np.asarray(o, dtype="float64")
    n = len(o)
    h = o if h is None else np.asarray(h, dtype="float64")
    l = o if l is None else np.asarray(l, dtype="float64")
    c = o if c is None else np.asarray(c, dtype="float64")
    v = np.ones(n) if v is None else np.asarray(v, dtype="float64")
    return pd.DataFrame({
        "open_time": t0 + np.arange(n, dtype="int64") * BAR, "open": o, "high": h, "low": l, "close": c,
        "volume": v, "quote_volume": v * c, "trades": np.ones(n, dtype="int64"),
        "taker_buy_volume": v / 2, "taker_buy_quote_volume": v * c / 2})


def flat(n: int, price: float = 100.0, t0: int = T0) -> dict:
    """Mutable column lists for a flat market, to be edited bar by bar."""
    return {"o": [price] * n, "h": [price] * n, "l": [price] * n, "c": [price] * n, "t0": t0}


def build(d: dict) -> pd.DataFrame:
    return mk_klines(d["o"], d["h"], d["l"], d["c"], t0=d["t0"])


def sig_at(n: int, rows, side: str = "long", prob: float = 1.0) -> np.ndarray:
    a = np.zeros((n, 2))
    for r in rows:
        a[r, 0 if side == "long" else 1] = prob
    return a


def plain_cfg(**kw) -> SimConfig:
    """No slippage, no safety layer, kill off: isolates the template arithmetic."""
    base = dict(start_equity=1000.0, slippage=0.0, stop_slippage_extra=0.0, kill_dd_pct=100.0,
                derate_enabled=False, plausibility_enabled=False, drift_enabled=False,
                # profit protection OFF: these tests hand-compute the bare template (see test_profit_protection.py)
                trail_enabled=False, breakeven_r=0.0, partial_tp_r=0.0, sl_cooldown_bars=0, max_side_risk_pct=100.0)
    base.update(kw)
    return SimConfig(**base)
