"""A drop-in fake exchange connector that *replays* historical candles bar-by-bar.

Why this exists
---------------
The live/paper `TradingEngine` is written against a connector interface
(`.contracts`, `.prices`, `get_historical_candles`, `get_bid_ask`). On a network that
blocks crypto exchanges — or simply when you want a fast, deterministic, watchable demo —
there is no real feed to drive it. `ReplayClient` satisfies the same interface but serves
candles from an in-memory DataFrame, revealing ONE new bar each time the engine polls.

The effect: the engine "lives through" history at whatever pace you set, executing paper
trades you can watch arrive in the dashboard in real time. Because it implements the same
methods as `BinanceFuturesClient`, the engine cannot tell the difference, and swapping back
to the real connector (once you have testnet keys + network access) is a one-line change.

`place_order` is intentionally **not** implemented for real execution — the engine only
calls it in live mode, and replay is a paper-only feed. If called it raises, which is the
correct, loud failure for "you tried to place a real order against fake data".
"""

from __future__ import annotations

import logging
import threading
import typing

import numpy as np
import pandas as pd

from models import Candle, Contract

logger = logging.getLogger(__name__)


def make_synthetic_ohlcv(n: int = 3000, seed: int = 7, start_price: float = 30_000.0,
                         start_ts_ms: int = 1_600_000_000_000,
                         tf_ms: int = 3_600_000) -> pd.DataFrame:
    """Geometric random walk with mild momentum — same shape used elsewhere in the repo.

    NOT a market model. It exists only to exercise the pipeline and make the dashboard
    show live activity. Any 'profit' on this data is overfitting, not edge.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 0.01, n)
    drift = 0.3 * pd.Series(shocks).rolling(5).mean().fillna(0).values
    log_ret = shocks + drift
    close = start_price * np.exp(np.cumsum(log_ret))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(10, 100, n)
    ts = (start_ts_ms + np.arange(n) * tf_ms).astype("int64")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": volume})


def _make_contract(symbol: str, price_decimals: int = 2,
                   qty_decimals: int = 3) -> Contract:
    """Build a Contract using the connector's own binance-shaped constructor."""
    return Contract(
        {
            "symbol": symbol,
            "baseAsset": symbol.replace("USDT", "") or "BTC",
            "quoteAsset": "USDT",
            "pricePrecision": price_decimals,
            "quantityPrecision": qty_decimals,
        },
        "binance",
    )


class ReplayClient:
    """Replays a candle DataFrame one bar per `get_historical_candles` call.

    Parameters
    ----------
    df : DataFrame with columns [timestamp, open, high, low, close, volume].
    symbol : the single tradable symbol this feed serves.
    warmup : number of bars revealed immediately on the first poll, so strategies that
        need history (EMA/ML/RL windows) have enough data before the first decision.
    spread_bps : half-spread in basis points used to synthesize a bid/ask around close.
    """

    def __init__(self, df: typing.Optional[pd.DataFrame] = None, symbol: str = "BTCUSDT",
                 warmup: int = 200, spread_bps: float = 1.0):
        self._df = (df if df is not None else make_synthetic_ohlcv()).reset_index(drop=True)
        if len(self._df) <= warmup + 1:
            raise ValueError(f"replay df too short ({len(self._df)}) for warmup={warmup}")
        self.symbol = symbol
        self._warmup = int(warmup)
        self._spread = float(spread_bps) / 10_000.0

        # Engine-facing attributes mirroring the real connectors.
        self.contracts: typing.Dict[str, Contract] = {symbol: _make_contract(symbol)}
        self.prices: typing.Dict[str, typing.Dict[str, float]] = {}
        self.logs: typing.List[dict] = []

        self._cursor = self._warmup        # index of the most recently revealed bar
        self._lock = threading.Lock()
        self._set_price(self._cursor)
        logger.info("ReplayClient ready: %d bars, warmup=%d, symbol=%s",
                    len(self._df), self._warmup, symbol)

    # ---------------------------------------------------------------- progress
    @property
    def progress(self) -> typing.Tuple[int, int]:
        """(current_bar, total_bars) — for a dashboard progress indicator."""
        return self._cursor, len(self._df) - 1

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._df) - 1

    # --------------------------------------------------------- connector iface
    def get_historical_candles(self, contract: Contract, interval: str) -> typing.List[Candle]:
        """Return all bars revealed so far, then advance the cursor by one.

        The engine calls this once per iteration. Revealing the new bar *after* building
        the returned list means the strategy decides on bar `t` and the next poll reveals
        `t+1` — there is no peeking at the future.
        """
        with self._lock:
            end = self._cursor + 1                      # inclusive of current cursor
            window = self._df.iloc[:end]
            candles = [
                Candle([int(r.timestamp), float(r.open), float(r.high),
                        float(r.low), float(r.close), float(r.volume)], interval, "binance")
                for r in window.itertuples(index=False)
            ]
            if self._cursor < len(self._df) - 1:
                self._cursor += 1
                self._set_price(self._cursor)
            return candles

    def get_bid_ask(self, contract: Contract) -> typing.Dict[str, float]:
        return self.prices.get(contract.symbol, {})

    def place_order(self, *args, **kwargs):  # pragma: no cover - must never run on replay
        raise NotImplementedError(
            "ReplayClient is a paper-only feed; live order placement is not supported."
        )

    def cancel_order(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("ReplayClient does not place or cancel real orders.")

    # ------------------------------------------------------------------ helpers
    def _set_price(self, idx: int) -> None:
        close = float(self._df["close"].iloc[idx])
        half = close * self._spread
        self.prices[self.symbol] = {"bid": close - half, "ask": close + half}
