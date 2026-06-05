"""Baseline EMA-crossover strategy (with anti-whipsaw guards).

Deliberately simple: it exists to validate the whole pipeline (data -> features ->
backtest -> metrics) end-to-end before any AI is introduced, and to bootstrap each live
engine until its RL policy is trained. If this can't be made to trade sensibly in the
backtester, no ML/RL result should be trusted either.

A naive "long when fast EMA > slow EMA, else short" flips on every marginal crossover and
bleeds fees/slippage in choppy markets (whipsaw). This version keeps the same idea but adds
three guards that each attack churn directly:

  * **Volatility-scaled deadband** — only commit to a side once the EMAs separate by more
    than ``band_mult * ATR``. Near a crossover the gap is noise, so we don't act on it.
  * **Hysteresis** — inside the deadband we *hold* the previous position instead of going
    flat, so price wandering around the crossover doesn't generate a stream of in/out trades.
  * **Trend filter** — an optional long-horizon EMA gates direction: no longs below the
    trend line, no shorts above it. OFF by default: on 1m data a swept backtest showed it
    *adds* churn (price oscillating around the trend line forces flat/re-enter cycles), so
    it hurt every symbol. Left in as an opt-in for slower timeframes.

Set ``band_mult=0.0`` to recover the original naive crossover behaviour. The defaults
(``band_mult=1.0``) were chosen by sweeping real Binance 1m data across BTC/ETH/SOL/DOGE:
they cut summed loss from ~-91% to ~-9% and trade count by ~80% vs the naive crossover,
purely by not acting on noise-level EMA gaps. (This is cost discipline, not an edge — the
RL policy still hot-swaps in once trained; this only governs the bootstrap/fallback.)
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from strategies.features import ema, atr


class EmaCrossStrategy(Strategy):
    name = "ema_cross"

    def __init__(self, fast: int = 12, slow: int = 26, allow_short: bool = True, *,
                 band_mult: float = 1.0, atr_period: int = 14, trend_filter: int = 0):
        """
        fast, slow   : EMA spans (fast must be < slow).
        allow_short  : if False, the strategy is long/flat only.
        band_mult    : deadband width as a multiple of ATR. The fast/slow EMA gap must
                       exceed this before a side is taken; 0.0 disables the deadband.
        atr_period   : ATR look-back used to scale the deadband to current volatility.
        trend_filter : span of a long-horizon EMA that gates direction (longs only above it,
                       shorts only below). 0 disables the filter.
        """
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short
        self.band_mult = float(band_mult)
        self.atr_period = int(atr_period)
        self.trend_filter = int(trend_filter)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        fast_ema = ema(close, self.fast)
        slow_ema = ema(close, self.slow)
        gap = fast_ema - slow_ema

        # Volatility-scaled deadband (in price units). ATR needs high/low; fall back to a
        # rolling std of close if they're absent so the strategy still runs on close-only data.
        if self.band_mult > 0:
            if {"high", "low"}.issubset(df.columns):
                vol = atr(df["high"], df["low"], close, self.atr_period)
            else:
                vol = close.rolling(self.atr_period).std()
            band = self.band_mult * vol.fillna(0.0)
        else:
            band = pd.Series(0.0, index=df.index)

        # Raw side only at a *confident* crossover; NaN elsewhere means "hold previous"
        # (hysteresis), so chop around the crossover doesn't churn in and out.
        raw = pd.Series(np.nan, index=df.index)
        raw[gap > band] = 1.0
        raw[gap < -band] = -1.0 if self.allow_short else 0.0
        desired = raw.ffill().fillna(0.0)

        # Trend filter: drop any position that fights the long-horizon EMA (flat instead).
        if self.trend_filter > 0:
            trend = ema(close, self.trend_filter)
            desired = desired.where(~((desired > 0) & (close <= trend)), 0.0)
            desired = desired.where(~((desired < 0) & (close >= trend)), 0.0)

        # Trade on the *next* bar to avoid look-ahead: act on the signal we could actually
        # have seen at the close of the current bar.
        return desired.shift(1).fillna(0).astype(int)
