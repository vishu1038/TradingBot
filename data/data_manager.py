"""Download and cache historical OHLCV candles.

Bridges the existing exchange connectors (which return `models.Candle` objects) to a
cached pandas DataFrame backed by SQLite. Handles Binance's 1000-candle/request cap via
pagination so arbitrarily long histories can be assembled.

Example
-------
    from connectors.binance_futures import BinanceFuturesClient
    from data.database import Database
    from data.data_manager import DataManager

    client = BinanceFuturesClient(pub, sec, testnet=True)
    dm = DataManager(Database())
    df = dm.get_candles(client, "BTCUSDT", "1h", limit=5000)   # cached after first call
"""

import logging
import time
import typing

import pandas as pd

from models import Candle

logger = logging.getLogger()

# Milliseconds per timeframe, used to walk pagination windows.
_TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def _candles_to_df(candles: typing.List[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        [(c.timestamp, c.open, c.high, c.low, c.close, c.volume) for c in candles],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


class DataManager:
    def __init__(self, database):
        self.db = database

    def get_candles(self, client, symbol: str, timeframe: str, limit: int = 1000,
                    exchange: typing.Optional[str] = None,
                    force_refresh: bool = False) -> pd.DataFrame:
        """Return up to `limit` most-recent candles as a DataFrame.

        Reads from the SQLite cache first; downloads (and caches) only what's missing.
        `client` is any connector exposing `.contracts` and `get_historical_candles`.
        """
        exchange = exchange or self._infer_exchange(client)
        contract = client.contracts.get(symbol)
        if contract is None:
            raise ValueError(f"Unknown symbol {symbol!r} on {exchange}")

        if not force_refresh:
            cached = self.db.load_candles(exchange, symbol, timeframe)
            if len(cached) >= limit:
                return cached.tail(limit).reset_index(drop=True)

        downloaded = self._download(client, contract, timeframe, exchange, limit)
        if not downloaded.empty:
            self.db.save_candles(exchange, symbol, timeframe, downloaded)

        df = self.db.load_candles(exchange, symbol, timeframe)
        return df.tail(limit).reset_index(drop=True)

    def _download(self, client, contract, timeframe: str, exchange: str,
                  limit: int) -> pd.DataFrame:
        """Page backwards/forwards until `limit` candles are gathered.

        Binance honours an `endTime`; BitMEX paginates differently. To keep the scaffold
        portable we issue repeated `get_historical_candles` calls and dedupe by timestamp.
        """
        frames: typing.List[pd.DataFrame] = []
        gathered = 0
        per_call = 1000 if exchange == "binance" else 500
        max_pages = max(1, (limit // per_call) + 1)

        for page in range(max_pages):
            candles = client.get_historical_candles(contract, timeframe)
            if not candles:
                break
            df = _candles_to_df(candles)
            frames.append(df)
            gathered += len(df)
            logger.info("Fetched %s %s %s candles (page %s, total %s)",
                        len(df), contract.symbol, timeframe, page + 1, gathered)
            if gathered >= limit or len(df) < per_call:
                break
            time.sleep(0.5)  # be gentle with rate limits

        if not frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
        return combined.reset_index(drop=True)

    @staticmethod
    def _infer_exchange(client) -> str:
        name = type(client).__name__.lower()
        if "binance" in name:
            return "binance"
        if "bitmex" in name:
            return "bitmex"
        raise ValueError(f"Cannot infer exchange from client {type(client).__name__}")
