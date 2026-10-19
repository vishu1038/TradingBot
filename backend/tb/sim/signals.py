"""Signal sources for the backtester + built-in baselines.

A signal source for one symbol is (prob_long, prob_short) arrays aligned to that symbol's kline rows:
row i is the decision made at the CLOSE of bar i (the sim enters at the open of bar i+1).

Accepted specs (see `resolve`):
  * callable(symbol, klines) -> DataFrame[prob_long, prob_short] | ndarray[N,2] | (pl, ps)
  * dict symbol -> DataFrame (index = open_time OR rows aligned to klines) / ndarray[N,2] / tuple
  * a single DataFrame/ndarray (applied to every symbol)

A callable may carry `carries_probabilities = False` (the random / every-N baselines do): their
"probability" is just a 0/1 trigger, so drift detection (which compares predicted vs realised win
rate) is switched off for them.
"""
from __future__ import annotations

import zlib
from typing import Callable

import numpy as np
import pandas as pd


def _as_arrays(x, kl: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(kl)
    if isinstance(x, tuple):
        pl, ps = x
        pl = np.asarray(pl, dtype="float64")
        ps = np.zeros(n) if ps is None else np.asarray(ps, dtype="float64")
    elif isinstance(x, pd.DataFrame):
        if "open_time" in x.columns:
            x = x.set_index("open_time")
        if len(x) == n and not np.array_equal(x.index.to_numpy(), kl["open_time"].to_numpy()):
            # positional
            pl = x["prob_long"].to_numpy("float64")
            ps = x["prob_short"].to_numpy("float64") if "prob_short" in x else np.zeros(n)
        else:
            x = x.reindex(kl["open_time"].to_numpy())
            pl = x["prob_long"].to_numpy("float64")
            ps = x["prob_short"].to_numpy("float64") if "prob_short" in x else np.zeros(n)
    else:
        a = np.asarray(x, dtype="float64")
        if a.ndim == 1:
            pl, ps = a, np.zeros(n)
        else:
            pl, ps = a[:, 0], a[:, 1]
    if len(pl) != n or len(ps) != n:
        raise ValueError(f"signal length {len(pl)} != klines {n}")
    return np.nan_to_num(pl, nan=0.0), np.nan_to_num(ps, nan=0.0)


def resolve(spec, symbol: str, kl: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if callable(spec):
        return _as_arrays(spec(symbol, kl), kl)
    if isinstance(spec, dict):
        return _as_arrays(spec[symbol], kl)
    return _as_arrays(spec, kl)


def carries_probabilities(spec) -> bool:
    return bool(getattr(spec, "carries_probabilities", True))


def random_entries(p: float, seed: int = 0, side: str = "long") -> Callable:
    """Baseline: every bar independently triggers an entry with probability `p` (same exits as the
    model).  side: 'long' | 'short' | 'both' (side drawn 50/50).  Deterministic per (seed, symbol).
    Emits prob 1.0 (use thr <= 1)."""
    if side not in ("long", "short", "both"):
        raise ValueError(side)

    def gen(symbol: str, kl: pd.DataFrame):
        rng = np.random.default_rng([seed, zlib.crc32(symbol.encode())])
        n = len(kl)
        trig = rng.random(n) < p
        pick_long = np.ones(n, bool) if side == "long" else np.zeros(n, bool) if side == "short" \
            else rng.random(n) < 0.5
        pl = np.where(trig & pick_long, 1.0, 0.0)
        ps = np.where(trig & ~pick_long, 1.0, 0.0)
        return pl, ps

    gen.carries_probabilities = False  # type: ignore[attr-defined]
    return gen


def always_long_every_n(n: int, offset: int = 0) -> Callable:
    """Baseline: a long trigger on every n-th bar (rows with (i - offset) % n == 0)."""
    if n < 1:
        raise ValueError("n >= 1")

    def gen(symbol: str, kl: pd.DataFrame):
        i = np.arange(len(kl))
        pl = ((i - offset) % n == 0).astype("float64")
        return pl, np.zeros(len(kl))

    gen.carries_probabilities = False  # type: ignore[attr-defined]
    return gen


def every_n_entries(n: int, side: str = "both", offset: int = 0) -> Callable:
    """Baseline: a trigger on every n-th bar. ``side`` 'long' | 'short' | 'both' (both alternates long / short on successive
    triggers of a symbol - no built-in lean to either side, user 2026-10-19). Deterministic per symbol."""
    if n < 1:
        raise ValueError("n >= 1")
    if side not in ("long", "short", "both"):
        raise ValueError(side)

    def gen(symbol: str, kl: pd.DataFrame):
        i = np.arange(len(kl))
        trig = ((i - offset) % n == 0) & (i >= offset)
        k = np.cumsum(trig) - 1                                  # index of the trigger
        pick_long = np.ones(len(kl), bool) if side == "long" else np.zeros(len(kl), bool) if side == "short" else (k % 2 == 0)
        return np.where(trig & pick_long, 1.0, 0.0), np.where(trig & ~pick_long, 1.0, 0.0)

    gen.carries_probabilities = False  # type: ignore[attr-defined]
    return gen
