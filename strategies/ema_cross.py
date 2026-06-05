"""Baseline EMA-crossover strategy.

Deliberately simple: it exists to validate the whole pipeline (data -> features ->
backtest -> metrics) end-to-end before any AI is introduced. If this can't be made to
trade sensibly in the backtester, no ML/RL result should be trusted either.

Long when the fast EMA is above the slow EMA, short (or flat) otherwise.
"""

import pandas as pd

from strategies.base import Strategy
from strategies.features import ema


class EmaCrossStrategy(Strategy):
    name = "ema_cross"

    def __init__(self, fast: int = 12, slow: int = 26, allow_short: bool = True):
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast_ema = ema(df["close"], self.fast)
        slow_ema = ema(df["close"], self.slow)

        signal = pd.Series(0, index=df.index)
        signal[fast_ema > slow_ema] = 1
        signal[fast_ema < slow_ema] = -1 if self.allow_short else 0

        # Trade on the *next* bar to avoid look-ahead: act on the signal we could
        # actually have seen at the close of the current bar.
        return signal.shift(1).fillna(0).astype(int)
