"""End-to-end example: data -> features -> backtest -> metrics.

Runs the baseline EMA strategy and the supervised-ML strategy through the backtester and
prints out-of-sample metrics. Demonstrates the full Phase 1 + 2 scaffold.

Modes
-----
* With API keys configured (.env): downloads & caches real testnet candles via DataManager.
* Without keys: falls back to a synthetic random-walk series so the pipeline runs offline.

    python run_backtest.py                 # synthetic data unless keys are set
    python run_backtest.py BTCUSDT 1h      # real data for a Binance symbol/timeframe
"""

import logging
import sys

import numpy as np
import pandas as pd

from config import CONFIG
from backtesting.engine import Backtester
from backtesting.metrics import print_summary
from strategies.ema_cross import EmaCrossStrategy
from strategies.ml_strategy import MLStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
logger = logging.getLogger()


def synthetic_ohlcv(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Geometric random walk with mild autocorrelation — just enough structure to exercise
    the pipeline. NOT a market model; real edges must be validated on real data."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 0.01, n)
    drift = 0.3 * pd.Series(shocks).rolling(5).mean().fillna(0).values  # weak momentum
    log_ret = shocks + drift
    close = 30_000 * np.exp(np.cumsum(log_ret))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(10, 100, n)
    ts = (1_600_000_000_000 + np.arange(n) * 3_600_000).astype("int64")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": volume})


def load_real(symbol: str, timeframe: str, limit: int = 4000) -> pd.DataFrame:
    from connectors.binance_futures import BinanceFuturesClient
    from data.database import Database
    from data.data_manager import DataManager

    client = BinanceFuturesClient(CONFIG.binance_public_key, CONFIG.binance_secret_key,
                                  CONFIG.use_testnet)
    dm = DataManager(Database(CONFIG.database_path))
    return dm.get_candles(client, symbol, timeframe, limit=limit)


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else None
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1h"

    if symbol and CONFIG.binance_public_key:
        logger.info("Loading real data for %s %s", symbol, timeframe)
        df = load_real(symbol, timeframe)
    else:
        logger.info("No symbol/keys provided -> using synthetic data (pipeline demo only)")
        df, timeframe = synthetic_ohlcv(), "1h"

    if len(df) < 500:
        logger.error("Not enough candles (%s) to backtest", len(df))
        return

    bt = Backtester(fee=0.0004, slippage=0.0002, initial_capital=10_000, risk_fraction=0.5)

    # --- Baseline: EMA crossover (no training needed) ---
    ema = EmaCrossStrategy(fast=12, slow=26)
    ema_result = bt.run(df, ema.generate_signals(df))
    print_summary(ema_result, timeframe, title="EMA crossover (full series)")

    # --- Supervised ML: chronological 70/30 split (NO shuffle) ---
    split = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    ml = MLStrategy(horizon=1, deadband=0.0005, confidence=0.55)
    stats = ml.train(train_df)
    logger.info("ML train stats: %s", stats)
    ml_result = bt.run(test_df, ml.generate_signals(test_df))
    print_summary(ml_result, timeframe, title="ML GBM (out-of-sample 30%)")

    # --- Walk-forward (the honest evaluation) ---
    wf = bt.walk_forward(df, MLStrategy(horizon=1, deadband=0.0005, confidence=0.55),
                         train_fn=lambda s, tr: s.train(tr), n_splits=4)
    avg_ret = np.mean([r.total_return for r in wf]) * 100
    logger.info("ML walk-forward avg out-of-sample return across 4 folds: %.2f%%", avg_ret)
    print("\nNote: on synthetic random-walk data, near-zero/negative net return after costs "
          "is EXPECTED and correct — there is no real edge to find. Validate on real data.")


if __name__ == "__main__":
    main()
