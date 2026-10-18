"""Synthetic archive-format frames for tests and smoke runs (no network, no real data).

``make_symbol`` returns (klines, metrics, funding) with the exact archive schema.  The price process is
a random walk with volatility clustering and an optional weak, *planted* predictability so that the
training smoke test can confirm the pipeline learns something when something is there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MS = 300_000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % 86_400_000)    # a UTC midnight


def make_symbol(n: int = 3000, seed: int = 0, start_price: float = 100.0, t0: int = T0,
                planted_edge: float = 0.0, market: np.ndarray | None = None,
                with_metrics: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    rng = np.random.default_rng(seed)
    vol = np.empty(n)
    v = 0.002
    for i in range(n):
        v = 0.0004 + 0.9 * v + 0.0001 * abs(rng.standard_normal())
        vol[i] = v
    # a slow hidden "momentum" state; the planted edge makes the next-bar return depend on it
    state = np.zeros(n)
    for i in range(1, n):
        state[i] = 0.97 * state[i - 1] + 0.25 * rng.standard_normal()
    r = vol * rng.standard_normal(n)
    if planted_edge:
        r[1:] += planted_edge * vol[1:] * np.tanh(state[:-1])
    if market is not None:
        r = r + 0.6 * market[:n]
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = vol * np.abs(rng.standard_normal((2, n))) * 0.8
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    base_v = np.exp(rng.normal(8, 0.4, n)) * (1 + 40 * vol)
    tbr = np.clip(0.5 + 0.2 * np.tanh(state) * 0.3 + 0.05 * rng.standard_normal(n), 0.05, 0.95)
    t = t0 + np.arange(n, dtype="int64") * BAR_MS
    kl = pd.DataFrame({
        "open_time": t, "open": open_, "high": high, "low": low, "close": close,
        "volume": base_v, "quote_volume": base_v * (open_ + close) / 2,
        "trades": (base_v / 3).astype("int64") + 1,
        "taker_buy_volume": base_v * tbr, "taker_buy_quote_volume": base_v * tbr * close,
    })
    metrics = funding = None
    if with_metrics:
        oi = 1e6 * np.exp(np.cumsum(0.002 * rng.standard_normal(n)))
        metrics = pd.DataFrame({
            "create_time": t + BAR_MS,           # snapshot stamped at the END of the 5m period
            "sum_open_interest": oi,
            "sum_open_interest_value": oi * close,
            "count_toptrader_long_short_ratio": np.exp(0.2 * np.tanh(state) + 0.05 * rng.standard_normal(n)),
            "sum_toptrader_long_short_ratio": np.exp(0.2 * np.tanh(state) + 0.05 * rng.standard_normal(n)),
            "count_long_short_ratio": np.exp(0.1 * rng.standard_normal(n)),
            "sum_taker_long_short_vol_ratio": np.exp(0.15 * rng.standard_normal(n)),
        })
        ft = np.arange(t0, t[-1] + 1, 8 * 3_600_000, dtype="int64")
        funding = pd.DataFrame({"calc_time": ft, "funding_rate": 1e-4 + 2e-5 * rng.standard_normal(len(ft))})
    return kl, metrics, funding


def make_market(n: int, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.001 * rng.standard_normal(n)
