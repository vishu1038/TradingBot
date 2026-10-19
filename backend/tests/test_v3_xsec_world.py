"""Synthetic 5m worlds for the cross-sectional tests (helpers only, no tests)."""
import numpy as np
import pandas as pd

HOUR = 3_600_000
BAR = 300_000
T0 = 1_700_000_000_000 // HOUR * HOUR


def make_world(S=12, days=30, seed=0, drift=None, vol=0.002, flat_noise=None, qv=5e7, crash=None,
               funding_rate=None, with_metrics=False, t0=T0, start_offsets=None, qv_scale=None, sym_prefix="C"):
    """{sym: (klines, metrics|None, funding|None)} on a regular 5m grid starting at t0 (hour aligned).

    drift       per-5m-bar mean log return per coin (len S) - lets tests plant a cross-sectional edge
    flat_noise  if given the price is iid noise around 100 (no drift, no trend): a cost-measurement world
    crash       {coin_index: (bar_index, factor)} multiply price from that bar on
    start_offsets  {coin_index: first bar index} - the coin does not exist before (late listing)
    qv_scale    per-coin multiplier of the quote volume (illiquid coins)"""
    rng = np.random.default_rng(seed)
    n = days * 288
    drift = np.zeros(S) if drift is None else np.asarray(drift, dtype="float64")
    out = {}
    t = t0 + np.arange(n, dtype="int64") * BAR
    for i in range(S):
        if flat_noise is not None:
            c = 100.0 * (1.0 + flat_noise * rng.standard_normal(n))
        else:
            c = 100.0 * np.exp(np.cumsum(rng.normal(drift[i], vol, n)))
        if crash and i in crash:
            b, f = crash[i]
            c = c.copy()
            c[b:] *= f
        o = np.concatenate([[100.0], c[:-1]])
        if crash and i in crash:
            o = o.copy()
            o[crash[i][0]] *= crash[i][1]                    # the crash is a GAP: the bar opens at the new price
        wig = np.abs(rng.normal(0, vol / 3 if flat_noise is None else flat_noise / 3, n))
        h = np.maximum(o, c) * (1 + wig)
        l = np.minimum(o, c) * (1 - wig)
        v = rng.uniform(0.5, 1.5, n) * (qv / 100.0) * (1.0 if qv_scale is None else qv_scale[i])
        kl = pd.DataFrame({"open_time": t, "open": o, "high": h, "low": l, "close": c, "volume": v / c,
                           "quote_volume": v, "trades": rng.integers(50, 150, n),
                           "taker_buy_volume": v / c / 2, "taker_buy_quote_volume": v / 2})
        if start_offsets and i in start_offsets:
            kl = kl.iloc[start_offsets[i]:].reset_index(drop=True)
        m = None
        if with_metrics:
            oi = 1e6 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
            m = pd.DataFrame({"create_time": t, "sum_open_interest": oi / c, "sum_open_interest_value": oi,
                              "count_toptrader_long_short_ratio": 1 + 0.1 * rng.standard_normal(n),
                              "sum_toptrader_long_short_ratio": 1 + 0.1 * rng.standard_normal(n),
                              "count_long_short_ratio": 1 + 0.1 * rng.standard_normal(n),
                              "sum_taker_long_short_vol_ratio": 1 + 0.1 * rng.standard_normal(n)})
            if start_offsets and i in start_offsets:
                m = m.iloc[start_offsets[i]:].reset_index(drop=True)
        f = None
        if funding_rate is not None:
            ft = np.arange(t0, t0 + n * BAR, 8 * HOUR, dtype="int64")
            f = pd.DataFrame({"calc_time": ft, "funding_rate": np.full(len(ft), float(funding_rate))})
        out["BTCUSDT" if i == 0 else f"{sym_prefix}{i:02d}USDT"] = (kl, m, f)
    return out
