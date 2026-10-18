"""REST top-up of the archive frames from https://fapi.binance.com (public endpoints, no keys).

Everything here returns frames in exactly the archive schema (see tb.data.paths) so
`top_up(symbol)` can merge them into the parquet files.

Metrics mapping (VERIFIED by overlaying REST output on archive rows for 2026-10-10, BTCUSDT):

  archive column                    REST source                                   timestamp
  --------------------------------  --------------------------------------------  ---------------
  sum_open_interest                 /futures/data/openInterestHist  sumOpenInterest        ts - 5m
  sum_open_interest_value           /futures/data/openInterestHist  sumOpenInterestValue   ts - 5m
  count_toptrader_long_short_ratio  /futures/data/topLongShortAccountRatio  longShortRatio ts - 5m
  sum_toptrader_long_short_ratio    /futures/data/topLongShortPositionRatio longShortRatio ts - 5m
  count_long_short_ratio            /futures/data/globalLongShortAccountRatio longShortRatio ts - 5m
  sum_taker_long_short_vol_ratio    /futures/data/takerlongshortRatio buySellRatio         ts (same)

Known mismatches (not fixable from REST):
  * REST stamps OI / ratio samples at the END of the 5m period, the archive at its START, hence the
    -5 min shift above (taker ratio is already bar-open-stamped and equals taker_buy/taker_sell of
    that bar's kline).  After the shift REST values equal the archive's to REST precision.
  * REST ratios are rounded to 4 decimals (archive: 8); OI to 3 decimals.
  * REST only serves the latest 30 days; anything older than that cannot be topped up.
  * Rows whose end-of-period timestamp is not yet in the past are dropped (no partial periods).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from . import frames
from .paths import (BAR_MS, DAY_MS, FUNDING_COLS, KLINE_COLS, METRIC_COLS, archive_root,
                    funding_path, kline_path, metrics_path)

BASE = "https://fapi.binance.com"
METRICS_HISTORY_DAYS = 29  # REST keeps ~30 days; stay inside it

# (archive column, endpoint, json field, timestamp shift ms)
_METRIC_SOURCES = [
    ("sum_open_interest", "openInterestHist", "sumOpenInterest", -BAR_MS),
    ("sum_open_interest_value", "openInterestHist", "sumOpenInterestValue", -BAR_MS),
    ("count_toptrader_long_short_ratio", "topLongShortAccountRatio", "longShortRatio", -BAR_MS),
    ("sum_toptrader_long_short_ratio", "topLongShortPositionRatio", "longShortRatio", -BAR_MS),
    ("count_long_short_ratio", "globalLongShortAccountRatio", "longShortRatio", -BAR_MS),
    ("sum_taker_long_short_vol_ratio", "takerlongshortRatio", "buySellRatio", 0),
]

_session = requests.Session()
_session.headers["User-Agent"] = "tb-live/1.0"

Getter = Callable[[str, dict], list | dict]


def now_ms() -> int:
    return int(time.time() * 1000)


def rest_get(path: str, params: dict, retries: int = 6, timeout: float = 20.0):
    """GET JSON with backoff; honours 429/418 Retry-After. Errors are raised, never swallowed."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = _session.get(BASE + path, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                wait = float(r.headers.get("Retry-After", 5 * (attempt + 1)))
                time.sleep(min(wait, 120))
                last = RuntimeError(f"HTTP {r.status_code} (rate limited) {path}")
                continue
            if 400 <= r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code} {path} {params}: {r.text[:200]}")
            last = RuntimeError(f"HTTP {r.status_code} {path}")
        except requests.RequestException as e:
            last = e
        time.sleep(min(20.0, 1.0 * 2 ** attempt))
    assert last is not None
    raise last


# --------------------------------------------------------------------------- klines
def parse_klines(rows: list) -> pd.DataFrame:
    if not rows:
        return frames.empty_klines()
    df = pd.DataFrame({
        "open_time": [r[0] for r in rows], "open": [r[1] for r in rows], "high": [r[2] for r in rows],
        "low": [r[3] for r in rows], "close": [r[4] for r in rows], "volume": [r[5] for r in rows],
        "quote_volume": [r[7] for r in rows], "trades": [r[8] for r in rows],
        "taker_buy_volume": [r[9] for r in rows], "taker_buy_quote_volume": [r[10] for r in rows],
    })
    return frames.normalize_klines(df)


def fetch_klines(symbol: str, start_ms: int, end_ms: int | None = None, *, get: Getter = rest_get,
                 now: int | None = None, pace: float = 0.05) -> pd.DataFrame:
    """Closed 5m bars with open_time in [start_ms, end_ms]; pages 1500 at a time."""
    now = now_ms() if now is None else now
    last_closed_open = (now // BAR_MS) * BAR_MS - BAR_MS  # open_time of the last fully closed bar
    end = last_closed_open if end_ms is None else min(end_ms, last_closed_open)
    cur = (start_ms // BAR_MS) * BAR_MS if start_ms % BAR_MS == 0 else (start_ms // BAR_MS + 1) * BAR_MS
    out: list[pd.DataFrame] = []
    while cur <= end:
        rows = get("/fapi/v1/klines", {"symbol": symbol, "interval": "5m", "startTime": cur,
                                       "endTime": end + BAR_MS - 1, "limit": 1500})
        if not rows:
            break
        out.append(parse_klines(rows))
        nxt = int(rows[-1][0]) + BAR_MS
        if nxt <= cur or len(rows) < 1500:
            break
        cur = nxt
        time.sleep(pace)
    if not out:
        return frames.empty_klines()
    df = frames.merge_frames(None, pd.concat(out, ignore_index=True), "open_time")
    return df[df["open_time"] <= end].reset_index(drop=True)


# --------------------------------------------------------------------------- funding
def parse_funding(rows: list) -> pd.DataFrame:
    if not rows:
        return frames.empty_funding()
    return frames.normalize_funding(pd.DataFrame({
        "calc_time": [r["fundingTime"] for r in rows],
        "funding_rate": [r["fundingRate"] for r in rows]}))


def fetch_funding(symbol: str, start_ms: int, end_ms: int | None = None, *, get: Getter = rest_get,
                  pace: float = 0.1) -> pd.DataFrame:
    out = []
    cur = start_ms
    while True:
        p = {"symbol": symbol, "startTime": cur, "limit": 1000}
        if end_ms is not None:
            p["endTime"] = end_ms
        rows = get("/fapi/v1/fundingRate", p)
        if not rows:
            break
        out.append(parse_funding(rows))
        nxt = int(rows[-1]["fundingTime"]) + 1
        if len(rows) < 1000 or nxt <= cur:
            break
        cur = nxt
        time.sleep(pace)
    if not out:
        return frames.empty_funding()
    return frames.merge_frames(None, pd.concat(out, ignore_index=True), "calc_time")


# --------------------------------------------------------------------------- metrics
def parse_metric_endpoint(rows: list, column: str, field: str, shift: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({"create_time": pd.Series([], dtype="int64"),
                             column: pd.Series([], dtype="float64")})
    return pd.DataFrame({
        "create_time": np.array([int(r["timestamp"]) for r in rows], dtype="int64") + shift,
        column: pd.to_numeric(pd.Series([r[field] for r in rows]), errors="coerce").to_numpy("float64"),
    })


def fetch_metrics(symbol: str, start_ms: int, end_ms: int | None = None, *, get: Getter = rest_get,
                  now: int | None = None, pace: float = 0.4) -> pd.DataFrame:
    """Archive-schema metrics rows (create_time = period START) for the last <=29 days."""
    now = now_ms() if now is None else now
    floor = now - METRICS_HISTORY_DAYS * DAY_MS
    start_ms = max(start_ms, floor)
    # period end must be in the past: row stamped T is complete at T + 5m
    last_ct = (now // BAR_MS) * BAR_MS - BAR_MS
    end = last_ct if end_ms is None else min(end_ms, last_ct)
    rows_by_endpoint: dict[str, list] = {}
    shift_of = {ep: sh for _, ep, _, sh in _METRIC_SOURCES}
    for endpoint, shift in shift_of.items():
        cur = start_ms - shift  # request in REST timestamp space
        rows_all: list = []
        while cur <= end - shift:
            rows = get(f"/futures/data/{endpoint}", {"symbol": symbol, "period": "5m", "limit": 500,
                                                     "startTime": cur, "endTime": end - shift})
            if not rows:
                break
            rows_all += rows
            nxt = int(rows[-1]["timestamp"]) + BAR_MS
            if len(rows) < 500 or nxt <= cur:
                break
            cur = nxt
            time.sleep(pace)
        rows_by_endpoint[endpoint] = rows_all
        time.sleep(pace)
    frames_by_col = {col: [parse_metric_endpoint(rows_by_endpoint[ep], col, field, sh)]
                     for col, ep, field, sh in _METRIC_SOURCES}
    merged: pd.DataFrame | None = None
    for col, parts in frames_by_col.items():
        f = parts[0].drop_duplicates("create_time", keep="last")
        merged = f if merged is None else merged.merge(f, on="create_time", how="outer")
    assert merged is not None
    merged = merged[(merged["create_time"] >= start_ms) & (merged["create_time"] <= end)]
    merged = merged.dropna(subset=METRIC_COLS[1:], how="all").sort_values("create_time")
    if merged.empty:
        return frames.empty_metrics()
    return frames.normalize_metrics(merged.reset_index(drop=True))


# --------------------------------------------------------------------------- top_up
def _last(df: pd.DataFrame, col: str) -> int | None:
    return int(df[col].iloc[-1]) if len(df) else None


def top_up(symbol: str, root: Path | None = None, *, kinds=("klines", "metrics", "funding"),
           now: int | None = None, get: Getter = rest_get, default_lookback_days: int = 3) -> dict:
    """Fetch everything newer than the archive's last row and merge it into the parquet files.

    Returns {kind: rows_added}. A kind with no existing parquet is seeded from the last
    `default_lookback_days` days (use the archive downloader for history).
    """
    root = root or archive_root()
    now = now_ms() if now is None else now
    added: dict[str, int] = {}
    if "klines" in kinds:
        p = kline_path(symbol, root)
        old = frames.read_parquet(p, frames.empty_klines)
        last = _last(old, "open_time")
        start = last + BAR_MS if last is not None else now - default_lookback_days * DAY_MS
        new = fetch_klines(symbol, start, get=get, now=now)
        merged = frames.merge_frames(old, new, "open_time")
        added["klines"] = len(merged) - len(old)
        if added["klines"] or not p.exists():
            frames.write_parquet_atomic(merged, p)
    if "metrics" in kinds:
        p = metrics_path(symbol, root)
        old = frames.read_parquet(p, frames.empty_metrics)
        last = _last(old, "create_time")
        start = last + BAR_MS if last is not None else now - default_lookback_days * DAY_MS
        new = fetch_metrics(symbol, start, get=get, now=now)
        merged = frames.merge_frames(old, new, "create_time")
        added["metrics"] = len(merged) - len(old)
        if added["metrics"] or not p.exists():
            frames.write_parquet_atomic(merged, p)
    if "funding" in kinds:
        p = funding_path(symbol, root)
        old = frames.read_parquet(p, frames.empty_funding)
        last = _last(old, "calc_time")
        start = last + 1000 if last is not None else now - 40 * DAY_MS
        new = fetch_funding(symbol, start, now, get=get)
        merged = frames.merge_frames(old, new, "calc_time")
        added["funding"] = len(merged) - len(old)
        if added["funding"] or not p.exists():
            frames.write_parquet_atomic(merged, p)
    return added
