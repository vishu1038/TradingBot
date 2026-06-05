"""SQLite persistence for OHLCV candles, trades, and equity curves.

A thin, dependency-free wrapper around the stdlib `sqlite3`. Pandas is used only for
the candle read/write convenience helpers, which are the hot path for backtesting.

Schema
------
candles      : cached historical OHLCV, unique per (exchange, symbol, timeframe, timestamp)
trades       : every (paper or live) trade taken, for the GUI trades table and analysis
equity_curve : periodic account-equity snapshots per run, for performance metrics/charts
"""

import os
import sqlite3
import typing

import pandas as pd


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    exchange   TEXT    NOT NULL,
    symbol     TEXT    NOT NULL,
    timeframe  TEXT    NOT NULL,
    timestamp  INTEGER NOT NULL,   -- ms epoch, bar open time
    open       REAL    NOT NULL,
    high       REAL    NOT NULL,
    low        REAL    NOT NULL,
    close      REAL    NOT NULL,
    volume     REAL    NOT NULL,
    PRIMARY KEY (exchange, symbol, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT,
    timestamp  INTEGER NOT NULL,
    exchange   TEXT,
    symbol     TEXT,
    strategy   TEXT,
    side       TEXT,
    quantity   REAL,
    price      REAL,
    pnl        REAL,
    status     TEXT
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT    NOT NULL,
    timestamp  INTEGER NOT NULL,
    equity     REAL    NOT NULL
);
"""


class Database:
    def __init__(self, path: str = "data/tradingbot.db"):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # --- Candles -----------------------------------------------------------
    def save_candles(self, exchange: str, symbol: str, timeframe: str,
                     df: pd.DataFrame) -> int:
        """Upsert candles. `df` must have columns
        [timestamp, open, high, low, close, volume]. Returns rows written."""
        if df.empty:
            return 0
        rows = [
            (exchange, symbol, timeframe, int(r.timestamp),
             float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume))
            for r in df.itertuples(index=False)
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO candles "
            "(exchange, symbol, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def load_candles(self, exchange: str, symbol: str, timeframe: str,
                     start_ms: typing.Optional[int] = None,
                     end_ms: typing.Optional[int] = None) -> pd.DataFrame:
        query = ("SELECT timestamp, open, high, low, close, volume FROM candles "
                 "WHERE exchange = ? AND symbol = ? AND timeframe = ?")
        params: list = [exchange, symbol, timeframe]
        if start_ms is not None:
            query += " AND timestamp >= ?"
            params.append(int(start_ms))
        if end_ms is not None:
            query += " AND timestamp <= ?"
            params.append(int(end_ms))
        query += " ORDER BY timestamp ASC"
        return pd.read_sql_query(query, self.conn, params=params)

    def last_candle_timestamp(self, exchange: str, symbol: str,
                              timeframe: str) -> typing.Optional[int]:
        cur = self.conn.execute(
            "SELECT MAX(timestamp) AS ts FROM candles "
            "WHERE exchange = ? AND symbol = ? AND timeframe = ?",
            (exchange, symbol, timeframe),
        )
        row = cur.fetchone()
        return row["ts"] if row and row["ts"] is not None else None

    # --- Trades ------------------------------------------------------------
    def save_trade(self, trade: typing.Dict) -> None:
        self.conn.execute(
            "INSERT INTO trades "
            "(run_id, timestamp, exchange, symbol, strategy, side, quantity, price, pnl, status) "
            "VALUES (:run_id, :timestamp, :exchange, :symbol, :strategy, :side, "
            ":quantity, :price, :pnl, :status)",
            {k: trade.get(k) for k in
             ("run_id", "timestamp", "exchange", "symbol", "strategy",
              "side", "quantity", "price", "pnl", "status")},
        )
        self.conn.commit()

    # --- Equity ------------------------------------------------------------
    def save_equity_point(self, run_id: str, timestamp: int, equity: float) -> None:
        self.conn.execute(
            "INSERT INTO equity_curve (run_id, timestamp, equity) VALUES (?, ?, ?)",
            (run_id, int(timestamp), float(equity)),
        )
        self.conn.commit()

    def load_equity_curve(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT timestamp, equity FROM equity_curve WHERE run_id = ? ORDER BY timestamp ASC",
            self.conn, params=[run_id],
        )

    def close(self) -> None:
        self.conn.close()
