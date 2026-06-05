"""Vectorized backtester with realistic costs.

The single most important component for trustworthy results: if costs aren't modeled, every
strategy looks profitable. Given a price series and a target-position Series from a strategy
(values in {-1, 0, +1}), it simulates an equity curve where:

  * Position changes incur a proportional transaction cost (fee + slippage) on the traded
    notional. Costs are charged whenever the position changes, including flips.
  * Bar PnL = position_held_during_bar * bar_return * equity, i.e. we act on the position we
    entered at the *previous* bar's close (strategies already shift, so no double counting).
  * `risk_fraction` scales raw {-1,0,1} targets into actual exposure (0.0-1.0 of equity).

This is a fast first-pass simulator. A higher-fidelity event-driven version (partial fills,
stop-losses intrabar) belongs in Phase 3 alongside the live executor.
"""

import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series           # indexed like the input, in account currency
    returns: pd.Series                # per-bar strategy returns (net of costs)
    positions: pd.Series              # actual position held each bar (signed fraction)
    trades: int                       # number of position changes
    initial_capital: float
    final_equity: float
    total_cost: float
    meta: dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_capital - 1.0


class Backtester:
    def __init__(self, fee: float = 0.0004, slippage: float = 0.0002,
                 initial_capital: float = 10_000.0, risk_fraction: float = 1.0):
        """
        fee          : per-side proportional fee (Binance futures taker ~0.04%).
        slippage     : assumed per-side slippage, proportional to notional.
        risk_fraction: fraction of equity exposed at a full +/-1 signal.
        """
        self.cost_per_turn = fee + slippage
        self.initial_capital = initial_capital
        self.risk_fraction = risk_fraction

    def run(self, df: pd.DataFrame, target_position: pd.Series) -> BacktestResult:
        if "close" not in df:
            raise ValueError("df must contain a 'close' column")

        close = df["close"].astype(float).reset_index(drop=True)
        pos = (target_position.reindex(df.index).fillna(0).astype(float)
               .reset_index(drop=True) * self.risk_fraction)

        bar_ret = close.pct_change().fillna(0.0)

        # Cost charged on the change in exposure between bars.
        turnover = pos.diff().abs().fillna(pos.abs())
        cost = turnover * self.cost_per_turn

        # Strategy return for a bar = position carried into the bar * that bar's return,
        # minus the cost of any rebalance executed at the bar's open.
        strat_ret = pos.shift(1).fillna(0.0) * bar_ret - cost

        equity = (1.0 + strat_ret).cumprod() * self.initial_capital
        trades = int((pos.diff().fillna(pos).abs() > 1e-9).sum())

        idx = df.index
        return BacktestResult(
            equity_curve=pd.Series(equity.values, index=idx),
            returns=pd.Series(strat_ret.values, index=idx),
            positions=pd.Series(pos.values, index=idx),
            trades=trades,
            initial_capital=self.initial_capital,
            final_equity=float(equity.iloc[-1]) if len(equity) else self.initial_capital,
            total_cost=float((cost * self.initial_capital).sum()),
            meta={"cost_per_turn": self.cost_per_turn, "risk_fraction": self.risk_fraction},
        )

    def walk_forward(self, df: pd.DataFrame, strategy, train_fn: typing.Callable,
                     n_splits: int = 4) -> typing.List[BacktestResult]:
        """Chronological walk-forward: for each split, `train_fn(strategy, train_df)` then
        backtest on the next out-of-sample block. Returns one BacktestResult per test block.

        `train_fn(strategy, train_df)` should fit the strategy in place (rule-based
        strategies can pass a no-op). This is the only honest way to estimate ML/RL edge.
        """
        results = []
        fold = len(df) // (n_splits + 1)
        if fold < 2:
            raise ValueError("Not enough data for the requested number of splits")

        for i in range(n_splits):
            train_end = fold * (i + 1)
            test_end = fold * (i + 2)
            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]
            train_fn(strategy, train_df)
            signals = strategy.generate_signals(test_df)
            results.append(self.run(test_df, signals))
        return results
