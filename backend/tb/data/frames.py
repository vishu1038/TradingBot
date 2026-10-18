"""Normalisation / merge / parquet IO for the three archive frames."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import FUNDING_COLS, KLINE_COLS, METRIC_COLS

_KLINE_FLOAT = ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_volume", "taker_buy_quote_volume"]
_METRIC_FLOAT = METRIC_COLS[1:]


def empty_klines() -> pd.DataFrame:
    return normalize_klines(pd.DataFrame({c: [] for c in KLINE_COLS}))


def empty_metrics() -> pd.DataFrame:
    return normalize_metrics(pd.DataFrame({c: [] for c in METRIC_COLS}))


def empty_funding() -> pd.DataFrame:
    return normalize_funding(pd.DataFrame({c: [] for c in FUNDING_COLS}))


def _to_ms(s: pd.Series) -> pd.Series:
    """Epoch (ms, or us for very large values) or datetime-string series -> int64 ms (UTC)."""
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().all():
            s = num
        else:
            dt = pd.to_datetime(s, utc=True, errors="coerce")
            # pandas 3 may hold ns/us/ms resolution: normalise through ns difference
            ms = (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
            return ms.astype("int64")
    s = pd.to_numeric(s, errors="coerce")
    if len(s) and float(s.abs().max()) > 1e14:  # microseconds
        s = s // 1000
    return s.astype("int64")


def normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "open_time": _to_ms(df["open_time"]) if len(df) else pd.Series([], dtype="int64"),
    })
    for c in _KLINE_FLOAT:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float64").to_numpy() if len(df) \
            else pd.Series([], dtype="float64")
    out["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype("int64").to_numpy() \
        if len(df) else pd.Series([], dtype="int64")
    return out[KLINE_COLS].reset_index(drop=True)


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "create_time": _to_ms(df["create_time"]) if len(df) else pd.Series([], dtype="int64"),
    })
    for c in _METRIC_FLOAT:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float64").to_numpy() if len(df) \
            else pd.Series([], dtype="float64")
    return out[METRIC_COLS].reset_index(drop=True)


def normalize_funding(df: pd.DataFrame) -> pd.DataFrame:
    """calc_time floored to whole seconds: archive stamps carry a +1 ms artefact (…00:00:00.001),
    the REST endpoint carries other sub-second jitter; flooring makes the two sources dedupe."""
    out = pd.DataFrame({
        "calc_time": (_to_ms(df["calc_time"]) // 1000 * 1000) if len(df) else pd.Series([], dtype="int64"),
        "funding_rate": pd.to_numeric(df["funding_rate"], errors="coerce").astype("float64").to_numpy()
        if len(df) else pd.Series([], dtype="float64"),
    })
    return out[FUNDING_COLS].reset_index(drop=True)


def merge_frames(old: pd.DataFrame | None, new: pd.DataFrame | None, key: str) -> pd.DataFrame:
    """Concat, dedupe on `key` (new wins), sort ascending, reset index."""
    parts = [p for p in (old, new) if p is not None and len(p)]
    if not parts:
        return (old if old is not None else new).iloc[0:0].copy()
    cat = pd.concat(parts, ignore_index=True)
    cat = cat.drop_duplicates(subset=key, keep="last").sort_values(key, kind="stable")
    return cat.reset_index(drop=True)


def read_parquet(path: Path, empty) -> pd.DataFrame:
    if not Path(path).exists():
        return empty()
    return pd.read_parquet(path)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    os.replace(tmp, path)


def times_in_range(times: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Sorted int64 array slice with lo <= t < hi."""
    a = np.searchsorted(times, lo, side="left")
    b = np.searchsorted(times, hi, side="left")
    return times[a:b]
