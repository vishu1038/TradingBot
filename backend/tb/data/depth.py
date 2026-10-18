"""Order-book depth from data.binance.vision `daily/bookDepth`, aggregated to 5m bars on the fly.

Source CSV (one file per symbol-day, ~30 s snapshots, header `timestamp,percentage,depth,notional`):
  timestamp  'YYYY-MM-DD HH:MM:SS' UTC
  percentage band, negative = bids, positive = asks; {+-1..+-5} always, {+-0.2} only in newer files
             (written as '-5' in 2023-24 files and '-5.00' later; both handled)
  depth      base-asset quantity resting within that % of the mid
  notional   quote value of the same

Raw 30 s data is never stored.  Per 5m bar (open_time T, window (T, T+5m]) we keep
  * the LAST snapshot with ts <= bar close: `{bid|ask}_{depth|notional}_{0.2,1,2,3,4,5}`
  * the mean over the bar's snapshots of the +-1 % notional: `bid_notional_1_mean`, `ask_notional_1_mean`
  * `snap_count`: number of snapshots that fell in the bar
Only bars with >= 1 snapshot get a row (reindex to the klines with `loader.align_depth`).
A snapshot stamped exactly on a bar boundary belongs to the bar that closes there (known at that close),
except at the very start of the day file, where it is clipped into the day's first bar so a day file never
writes rows outside its own day.  Columns are float32 (7 significant digits; plenty for depth ratios).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import archive, frames
from .paths import BAR_MS, DAY_MS, archive_root, depth_meta_path, depth_path

BANDS = ["0.2", "1", "2", "3", "4", "5"]
_BAND_TENTHS = {"0.2": 2, "1": 10, "2": 20, "3": 30, "4": 40, "5": 50}
DEPTH_COLS = (["open_time"]
              + [f"{s}_{k}_{b}" for b in BANDS for s in ("bid", "ask") for k in ("depth", "notional")]
              + ["bid_notional_1_mean", "ask_notional_1_mean", "snap_count"])
CHECKPOINT_DAYS = 120


def empty_depth() -> pd.DataFrame:
    d = {c: pd.Series([], dtype="float32") for c in DEPTH_COLS}
    d["open_time"] = pd.Series([], dtype="int64")
    d["snap_count"] = pd.Series([], dtype="int16")
    return pd.DataFrame(d)[DEPTH_COLS]


def depth_url(symbol: str, day: str) -> str:
    return f"{archive.BASE}/daily/bookDepth/{symbol}/{symbol}-bookDepth-{day}.zip"


def _ts_ms(col: pd.Series) -> np.ndarray:
    dtv = pd.to_datetime(col, format="%Y-%m-%d %H:%M:%S", utc=True)
    return ((dtv - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy("int64")


def parse_depth_csv(zbytes: bytes) -> pd.DataFrame:
    """-> frame(ts_ms int64, tenths int16 (signed band*10), depth, notional float64). Zip CRC is verified
    by zipfile while reading."""
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        infos = [i for i in z.infolist() if i.filename.lower().endswith(".csv")]
        if not infos:
            raise ValueError("zip has no csv member")
        if infos[0].file_size > archive.MAX_UNCOMPRESSED:
            raise ValueError("csv member too large")
        with z.open(infos[0]) as fh:
            raw = pd.read_csv(fh, dtype={"timestamp": str})
    if "timestamp" not in raw.columns:       # headerless variant
        raise ValueError(f"unexpected depth columns {list(raw.columns)}")
    out = pd.DataFrame({
        "ts": _ts_ms(raw["timestamp"]),
        "tenths": np.rint(pd.to_numeric(raw["percentage"], errors="coerce").to_numpy("float64") * 10).astype("int16"),
        "depth": pd.to_numeric(raw["depth"], errors="coerce").to_numpy("float64"),
        "notional": pd.to_numeric(raw["notional"], errors="coerce").to_numpy("float64"),
    })
    return out


def aggregate_day(snap: pd.DataFrame, day_start_ms: int) -> pd.DataFrame:
    """Aggregate parsed 30 s snapshots (one day) into the 5m depth frame described in the module doc."""
    if len(snap) == 0:
        return empty_depth()
    ts_all = snap["ts"].to_numpy("int64")
    uts, inv = np.unique(ts_all, return_inverse=True)
    n = len(uts)
    cols = [(s, b) for b in BANDS for s in (-1, 1)]       # (sign, band)
    tenth_to_slot = {s * _BAND_TENTHS[b]: i for i, (s, b) in enumerate(cols)}
    slot = np.array([tenth_to_slot.get(int(t), -1) for t in range(-50, 51)], dtype=np.int16)
    tl = snap["tenths"].to_numpy("int64")
    ok = (tl >= -50) & (tl <= 50)
    sl = np.where(ok, slot[np.clip(tl + 50, 0, 100)], -1)
    keep = sl >= 0
    dep = np.full((n, len(cols)), np.nan)
    notl = np.full((n, len(cols)), np.nan)
    dep[inv[keep], sl[keep]] = snap["depth"].to_numpy("float64")[keep]
    notl[inv[keep], sl[keep]] = snap["notional"].to_numpy("float64")[keep]
    bar = np.maximum(((uts - 1) // BAR_MS) * BAR_MS, day_start_ms)
    ub, first = np.unique(bar, return_index=True)       # uts sorted -> bar sorted
    last = np.append(first[1:], n) - 1                   # last snapshot index per bar
    cnt = (np.append(first[1:], n) - first).astype("int16")
    out = {"open_time": ub.astype("int64")}
    for i, (s, b) in enumerate(cols):
        side = "bid" if s < 0 else "ask"
        out[f"{side}_depth_{b}"] = dep[last, i].astype("float32")
        out[f"{side}_notional_{b}"] = notl[last, i].astype("float32")
    for side, s in (("bid", -1), ("ask", 1)):
        j = cols.index((s, "1"))
        v = notl[:, j]
        good = np.isfinite(v)
        sums = np.add.reduceat(np.where(good, v, 0.0), first)
        cnts = np.add.reduceat(good.astype("float64"), first)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{side}_notional_1_mean"] = np.where(cnts > 0, sums / cnts, np.nan).astype("float32")
    out["snap_count"] = cnt
    return pd.DataFrame(out)[DEPTH_COLS]


def _read_meta(symbol: str, root: Path) -> dict:
    p = depth_meta_path(symbol, root)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"missing_days": [], "ok_days": 0}


def _write_meta(symbol: str, root: Path, meta: dict) -> None:
    p = depth_meta_path(symbol, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(meta))
    tmp.replace(p)


def download_depth(symbol: str, months: int, root: Path | None = None, verify: bool = False,
                   today: dt.date | None = None) -> archive.Stats:
    """Fetch + aggregate every missing day from the window start (or the symbol's first kline, whichever
    is later) to yesterday.  Days already present, or recorded as 404, are skipped.  `verify` is accepted for
    signature parity with the other kinds; zip CRCs are always checked and .CHECKSUM files are not fetched
    (saves half the requests on ~160k files)."""
    root = root or archive_root()
    today = today or archive.utc_today()
    full, _cur = archive.window_months(months, today)
    first_day = full[0]
    kpath = root / "klines" / "5m" / f"{symbol}.parquet"
    if kpath.exists():
        t0 = int(pd.read_parquet(kpath, columns=["open_time"])["open_time"].iloc[0])
        first_day = max(first_day, dt.datetime.fromtimestamp(t0 / 1000, dt.timezone.utc).date())
    path = depth_path(symbol, root)
    old = frames.read_parquet(path, empty_depth)
    times = old["open_time"].to_numpy()
    meta = _read_meta(symbol, root)
    missing = set(meta.get("missing_days", []))
    stats = archive.Stats()
    todo = []
    for d in archive.days_between(first_day, today):
        iso = d.isoformat()
        lo = archive.to_ms(d)
        if len(frames.times_in_range(times, lo, lo + DAY_MS)) or iso in missing:
            stats["skipped"] += 1
        else:
            todo.append(d)
    parts: list[pd.DataFrame] = []
    cur = old
    new_rows = 0

    def flush():
        nonlocal parts, cur
        if not parts:
            return
        cur = frames.merge_frames(cur, pd.concat(parts, ignore_index=True), "open_time")
        parts = []
        frames.write_parquet_atomic(cur, path)
        meta["missing_days"] = sorted(missing)
        _write_meta(symbol, root, meta)

    with tempfile.TemporaryDirectory(prefix="tb_dl_"):
        for i, d in enumerate(todo, 1):
            stats["units"] += 1
            iso = d.isoformat()
            try:
                body = archive.http_get(depth_url(symbol, iso))
                if body is None:
                    missing.add(iso)
                    stats["missing"] += 1
                    continue
                part = aggregate_day(parse_depth_csv(body), archive.to_ms(d))
            except Exception as e:  # noqa: BLE001 - counted and logged, never silent
                stats["failed"] += 1
                archive.log(f"  FAIL {symbol} depth {iso}: {type(e).__name__}: {e}")
                continue
            stats["downloaded"] += 1
            new_rows += len(part)
            if len(part):
                parts.append(part)
            if i % CHECKPOINT_DAYS == 0:
                flush()
    flush()
    stats["rows_added"] = len(cur) - len(old)
    meta.update({
        "first_day": (dt.datetime.fromtimestamp(int(cur["open_time"].iloc[0]) / 1000, dt.timezone.utc).date().isoformat()
                      if len(cur) else None),
        "last_day": (dt.datetime.fromtimestamp(int(cur["open_time"].iloc[-1]) / 1000, dt.timezone.utc).date().isoformat()
                     if len(cur) else None),
        "rows": int(len(cur)), "missing_days": sorted(missing),
        "has_02_band": bool(len(cur) and cur["bid_notional_0.2"].notna().any()),
        "window_first_day": first_day.isoformat(), "updated": today.isoformat()})
    _write_meta(symbol, root, meta)
    if not path.exists() and not len(cur):
        frames.write_parquet_atomic(cur, path)       # symbol has no depth at all: empty marker file
    return stats
