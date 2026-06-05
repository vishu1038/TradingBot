"""Technical-indicator feature engineering.

Hand-rolled in pandas so the scaffold runs with no extra dependency beyond pandas/numpy.
Swap in `pandas-ta` later for a wider catalogue; the column names here are the contract the
ML strategy and backtester rely on.

`add_indicators(df)` expects an OHLCV DataFrame (columns: open, high, low, close, volume)
and returns a copy with indicator columns appended. Rows with NaNs from look-back windows
are the caller's responsibility to drop (use `add_indicators(df).dropna()`).
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid           # normalized band width
    pctb = (close - lower) / (upper - lower)  # position within the band [0,1]
    return width, pctb


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append a standard indicator set. Returns a new DataFrame."""
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]

    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["volatility_20"] = out["ret_1"].rolling(20).std()

    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)
    out["ema_spread"] = (out["ema_12"] - out["ema_26"]) / close

    out["rsi_14"] = rsi(close, 14)

    macd_line, macd_signal, macd_hist = macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    out["atr_14"] = atr(high, low, close, 14)
    out["atr_pct"] = out["atr_14"] / close

    out["bb_width"], out["bb_pctb"] = bollinger(close, 20, 2.0)

    return out


# Columns the ML strategy feeds to the model. Keep in sync with add_indicators.
FEATURE_COLUMNS = [
    "ret_1", "ret_5", "volatility_20",
    "ema_spread", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "atr_pct", "bb_width", "bb_pctb",
]
