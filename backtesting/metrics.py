"""Performance metrics for a backtest equity curve / return series.

These are the numbers that define "is the bot making money" — interpret them on
out-of-sample (walk-forward) results only. A great in-sample Sharpe means nothing.
"""

import numpy as np
import pandas as pd

# Bars per year, used to annualize. Override per timeframe when known.
_PERIODS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "4h": 2_190, "1d": 365,
}


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction (e.g. -0.23 = -23%)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if downside.std() == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(periods_per_year))


def summarize(result, timeframe: str = "1h") -> dict:
    """Build a metrics dict from a BacktestResult. `result` duck-types on
    .equity_curve, .returns, .total_return, .trades, .total_cost."""
    ppy = _PERIODS_PER_YEAR.get(timeframe, 8_760)
    returns = result.returns
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = float(wins / (wins + losses)) if (wins + losses) else 0.0

    return {
        "total_return_pct": round(result.total_return * 100, 2),
        "final_equity": round(result.final_equity, 2),
        "sharpe": round(sharpe_ratio(returns, ppy), 2),
        "sortino": round(sortino_ratio(returns, ppy), 2),
        "max_drawdown_pct": round(max_drawdown(result.equity_curve) * 100, 2),
        "trades": result.trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "total_cost": round(result.total_cost, 2),
    }


def print_summary(result, timeframe: str = "1h", title: str = "Backtest") -> dict:
    s = summarize(result, timeframe)
    print(f"\n=== {title} ({timeframe}) ===")
    for k, v in s.items():
        print(f"  {k:>18}: {v}")
    return s
