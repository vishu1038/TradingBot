"""Load the archive parquet frames and align them on the 5m kline index."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import frames
from .paths import (BAR_MS, archive_root, funding_path, kline_path, metrics_path)


def symbols_available(root: Path | None = None, kind: str = "klines") -> list[str]:
    """Symbols with a parquet file of `kind` (klines | metrics | funding), sorted."""
    root = root or archive_root()
    d = {"klines": root / "klines" / "5m", "metrics": root / "metrics", "funding": root / "funding"}[kind]
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


def load_klines(symbol: str, root: Path | None = None) -> pd.DataFrame:
    p = kline_path(symbol, root)
    if not p.exists():
        raise FileNotFoundError(f"no klines for {symbol} under {p}")
    return pd.read_parquet(p)


def load(symbol: str, root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(klines, metrics, funding) as stored. Missing metrics/funding -> empty typed frames."""
    kl = load_klines(symbol, root)
    me = frames.read_parquet(metrics_path(symbol, root), frames.empty_metrics)
    fu = frames.read_parquet(funding_path(symbol, root), frames.empty_funding)
    return kl, me, fu


def clean_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Archive quirk: a handful of rows carry sum_open_interest(_value) == 0 (dead sample).
    Those two columns are set to NaN so features never see a fake 100% OI collapse."""
    m = metrics.copy()
    dead = (m["sum_open_interest"] <= 0) | (m["sum_open_interest_value"] <= 0)
    m.loc[dead, ["sum_open_interest", "sum_open_interest_value"]] = np.nan
    return m


def align_metrics(kl: pd.DataFrame, metrics: pd.DataFrame, tolerance_bars: int = 3) -> pd.DataFrame:
    """Metrics re-indexed to the kline rows: row i holds the metrics sample whose
    create_time <= kl.open_time[i] (most recent, at most `tolerance_bars` bars stale, else NaN).

    Look-ahead note: an archive row stamped T is the sample taken at the END of the 5m period that
    opens at T (verified vs REST), i.e. it is known at the close of kline T -- so aligning on
    open_time is usable for a decision taken at that bar's close, and not earlier.
    Returned columns: METRIC_COLS[1:] (+ `has_metrics` 1.0/0.0), same length/order as kl.
    """
    cols = [c for c in metrics.columns if c != "create_time"]
    metrics = clean_metrics(metrics)
    if len(metrics) == 0:
        out = pd.DataFrame({c: np.full(len(kl), np.nan) for c in cols})
        out["has_metrics"] = 0.0
        return out
    left = pd.DataFrame({"open_time": kl["open_time"].to_numpy()})
    right = metrics.sort_values("create_time").rename(columns={"create_time": "open_time"})
    m = pd.merge_asof(left, right, on="open_time", direction="backward",
                      tolerance=tolerance_bars * BAR_MS)
    out = m[cols].copy()
    out["has_metrics"] = out[cols[0]].notna().astype("float64")
    return out


def align_funding(kl: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """Most recent funding_rate known at each bar's open (calc_time <= open_time); NaN before the first."""
    if len(funding) == 0:
        return pd.Series(np.full(len(kl), np.nan), name="funding_rate")
    left = pd.DataFrame({"open_time": kl["open_time"].to_numpy()})
    right = funding.sort_values("calc_time").rename(columns={"calc_time": "open_time"})
    m = pd.merge_asof(left, right, on="open_time", direction="backward")
    return m["funding_rate"]


def regular_index_report(kl: pd.DataFrame) -> dict:
    """Gap summary for a kline frame: number of gaps and total missing bars."""
    t = kl["open_time"].to_numpy()
    d = np.diff(t)
    bad = d > BAR_MS
    return {"rows": int(len(t)), "gaps": int(bad.sum()),
            "missing_bars": int((d[bad] // BAR_MS - 1).sum()) if bad.any() else 0,
            "start": int(t[0]) if len(t) else None, "end": int(t[-1]) if len(t) else None}
