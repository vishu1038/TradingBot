"""Filesystem locations and parquet column schemas shared by archive/live/loader."""
from __future__ import annotations

import os
from pathlib import Path

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume",
              "trades", "taker_buy_volume", "taker_buy_quote_volume"]
METRIC_COLS = ["create_time", "sum_open_interest", "sum_open_interest_value",
               "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
               "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
FUNDING_COLS = ["calc_time", "funding_rate"]

KLINE_INT_COLS = ["open_time", "trades"]
BAR_MS = 300_000          # 5m
DAY_MS = 86_400_000


def repo_root() -> Path:
    # backend/tb/data/paths.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def archive_root() -> Path:
    env = os.environ.get("TB_ARCHIVE_DIR")
    return Path(env) if env else repo_root() / "data" / "archive"


def var_dir() -> Path:
    env = os.environ.get("TB_VAR_DIR")
    return Path(env) if env else repo_root() / "var"


def kline_path(symbol: str, root: Path | None = None) -> Path:
    return (root or archive_root()) / "klines" / "5m" / f"{symbol}.parquet"


def metrics_path(symbol: str, root: Path | None = None) -> Path:
    return (root or archive_root()) / "metrics" / f"{symbol}.parquet"


def funding_path(symbol: str, root: Path | None = None) -> Path:
    return (root or archive_root()) / "funding" / f"{symbol}.parquet"
