"""Strategy abstraction.

Every strategy — rule-based, ML, or (later) RL — produces a *target position* per bar:

    +1  fully long
     0  flat
    -1  fully short

`generate_signals(df)` returns a pandas Series of these values aligned to `df`'s index.
The backtester applies fees/slippage and the risk manager scales the raw target into an
actual position size, so strategies stay decoupled from execution and money management.

This vectorized interface fits rule-based and supervised-ML strategies cleanly. The
event-driven RL agent (Phase 3) will implement the same `generate_signals` contract by
stepping its policy bar-by-bar, so the backtester does not need to change.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return target position in {-1, 0, +1} for each row of `df`."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Strategy {self.name}>"
