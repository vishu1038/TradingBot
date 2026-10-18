"""Order-book depth bands, identical in definition to the data.binance.vision ``bookDepth`` archive.

Archive rows are ``(timestamp, percentage, depth, notional)`` with ``percentage`` in
{-5,-4,-3,-2,-1,-0.2, 0.2,1,2,3,4,5}; negative = bids. ``depth`` is the cumulative quantity (base
units) of resting orders within ``|percentage|`` % of the mid price on that side, ``notional`` the
cumulative quote value (price * qty). We compute exactly that from a REST depth snapshot:

* mid = (best bid + best ask) / 2
* bid band -p: all bids with price >= mid * (1 - p/100);  ask band +p: asks with price <= mid * (1 + p/100)

REST ``/fapi/v1/depth`` returns at most 1000 levels per side, so for tightly ticked symbols (BTC, ETH)
the wide bands are cut short. ``cover_pct`` records how far from mid the fetched book actually reaches
(the minimum over both sides); a band with ``pct > cover_pct`` is a LOWER BOUND and flagged
``truncated``. Consumers (the model) should mask or discount truncated bands.

``depth_frame(symbol)`` returns the rolling 5m history in a wide shape, one row per 5m bar:
``open_time`` (UTC ms, bar open) + ``depth_<pct>`` / ``notional_<pct>`` for each pct in ``PCTS``
(e.g. ``depth_-0.2``, ``notional_5``) + ``cover_pct``. Aggregation matches the data agent's: per 5m bar
the LAST snapshot at or before the bar's close.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import pandas as pd

BAND_PCTS = (0.2, 1.0, 2.0, 3.0, 4.0, 5.0)
PCTS = tuple([-p for p in reversed(BAND_PCTS)] + list(BAND_PCTS))     # archive percentage values
BAR_MS = 300_000
LADDER = 20
KEEP_BARS = 2016            # one week of 5m bars per symbol


def _fmt_pct(p: float) -> str:
    return f"{p:g}"


def frame_columns() -> list[str]:
    cols = ["open_time"]
    cols += [f"depth_{_fmt_pct(p)}" for p in PCTS]
    cols += [f"notional_{_fmt_pct(p)}" for p in PCTS]
    cols += ["cover_pct"]
    return cols


def compute_snapshot(bids: Sequence[Sequence[float]], asks: Sequence[Sequence[float]], symbol: str = "",
                     t: float | None = None) -> dict[str, Any] | None:
    """bids sorted high->low, asks low->high, each ``(price, qty)``. None if either side is empty."""
    if not bids or not asks:
        return None
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    if best_bid <= 0 or best_ask <= 0 or best_bid > best_ask:
        return None
    mid = (best_bid + best_ask) / 2.0
    cover = min((mid - float(bids[-1][0])) / mid, (float(asks[-1][0]) - mid) / mid) * 100.0
    bands = []
    for p in BAND_PCTS:
        lo, hi = mid * (1 - p / 100.0), mid * (1 + p / 100.0)
        bq = bn = aq = an = 0.0
        for price, qty in bids:
            if price < lo:
                break
            bq += qty
            bn += price * qty
        for price, qty in asks:
            if price > hi:
                break
            aq += qty
            an += price * qty
        bands.append({"pct": p, "bid_depth": bq, "bid_notional": bn, "ask_depth": aq, "ask_notional": an,
                      "truncated": p > cover})
    return {"symbol": symbol, "t": int(t if t is not None else time.time()), "mid": mid,
            "spread": best_ask - best_bid, "spread_pct": (best_ask - best_bid) / mid * 100.0,
            "cover_pct": cover, "bands": bands,
            "bids": [[float(p), float(q)] for p, q in bids[:LADDER]],
            "asks": [[float(p), float(q)] for p, q in asks[:LADDER]]}


def snapshot_to_row(snap: dict[str, Any]) -> dict[str, float]:
    """Archive-shaped wide row (without open_time) for one snapshot."""
    row: dict[str, float] = {}
    for b in snap["bands"]:
        p = _fmt_pct(b["pct"])
        row[f"depth_-{p}"] = b["bid_depth"]
        row[f"notional_-{p}"] = b["bid_notional"]
        row[f"depth_{p}"] = b["ask_depth"]
        row[f"notional_{p}"] = b["ask_notional"]
    row["cover_pct"] = snap["cover_pct"]
    return row


def bar_open(t_s: float) -> int:
    return int(t_s * 1000) // BAR_MS * BAR_MS


def record_snapshot(store, snap: dict[str, Any]) -> None:
    """Fold a snapshot into its 5m bucket. Later snapshots overwrite earlier ones in the same bar, which
    yields 'the last snapshot at or before the bar close'."""
    store.upsert_depth_5m(snap["symbol"], bar_open(snap["t"]), snap["t"], snapshot_to_row(snap))
    store.prune_depth_5m(snap["symbol"], KEEP_BARS)


def depth_frame(symbol: str, store=None, limit: int = KEEP_BARS) -> pd.DataFrame:
    """Rolling 5m depth history for ``symbol`` (oldest first); empty frame with the schema if none."""
    if store is None:
        from .config import db_path
        from .store import Store

        store = Store(db_path())
    rows = store.get_depth_5m(symbol, limit)
    cols = frame_columns()
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="int64" if c == "open_time" else "float64") for c in cols})
    df = pd.DataFrame([{"open_time": r["open_time"], **r["data"]} for r in rows])
    for c in cols:
        if c not in df:
            df[c] = float("nan")
    df["open_time"] = df["open_time"].astype("int64")
    return df[cols].reset_index(drop=True)
