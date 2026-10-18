"""data.binance.vision downloader for USDⓈ-M futures (klines 5m, metrics, fundingRate).

URL patterns (verified against the live site):
  monthly klines : data/futures/um/monthly/klines/{SYM}/5m/{SYM}-5m-YYYY-MM.zip
  daily   klines : data/futures/um/daily/klines/{SYM}/5m/{SYM}-5m-YYYY-MM-DD.zip
  daily   metrics: data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-YYYY-MM-DD.zip   (5m rows)
  monthly funding: data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-YYYY-MM.zip
  every zip has a sibling `<zip>.CHECKSUM` ("<sha256>  <name>") which we verify when present.
CSV quirks: klines have a header row in current files (older ones don't) and 12 columns
(the last, `ignore`, and `close_time`/ are dropped); metrics have a header and a datetime-string
`create_time`; fundingRate has a header `calc_time,funding_interval_hours,last_funding_rate`.
Both headerless and headered files are handled.  Downloaded zips are written to a temp dir and
only ever parsed as data (never extracted to disk / executed).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import frames
from .paths import (BAR_MS, DAY_MS, FUNDING_COLS, KLINE_COLS, METRIC_COLS, archive_root,
                    funding_path, kline_path, metrics_path)

BASE = "https://data.binance.vision/data/futures/um"
MAX_UNCOMPRESSED = 300 * 1024 * 1024  # zip-bomb guard

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT",
    "LINKUSDT", "SUIUSDT", "WIFUSDT", "1000PEPEUSDT", "TAOUSDT", "ENAUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SEIUSDT", "TIAUSDT", "FETUSDT", "ORDIUSDT",
    "WLDUSDT", "1000BONKUSDT",
]
KINDS = ("klines", "metrics", "funding")

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------- URLs / dates
def monthly_url(kind: str, symbol: str, ym: str) -> str:
    if kind == "klines":
        return f"{BASE}/monthly/klines/{symbol}/5m/{symbol}-5m-{ym}.zip"
    if kind == "funding":
        return f"{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
    raise ValueError(f"no monthly archive for {kind}")


def daily_url(kind: str, symbol: str, day: str) -> str:
    if kind == "klines":
        return f"{BASE}/daily/klines/{symbol}/5m/{symbol}-5m-{day}.zip"
    if kind == "metrics":
        return f"{BASE}/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"
    raise ValueError(f"no daily archive for {kind}")


def utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def month_start(d: dt.date) -> dt.date:
    return d.replace(day=1)


def add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.year * 12 + (d.month - 1) + n, 12)
    return dt.date(y, m + 1, 1)


def to_ms(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def window_months(months: int, today: dt.date | None = None) -> tuple[list[dt.date], dt.date]:
    """The `months` full calendar months before the current one, plus the current month start."""
    today = today or utc_today()
    cur = month_start(today)
    return [add_months(cur, -i) for i in range(months, 0, -1)], cur


def days_between(a: dt.date, b: dt.date) -> list[dt.date]:
    """a <= d < b"""
    return [a + dt.timedelta(days=i) for i in range((b - a).days)]


# --------------------------------------------------------------------------- HTTP
_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "tb-archive/1.0"
        _tls.s = s
    return s


def http_get(url: str, retries: int = 5, timeout: float = 60.0) -> bytes | None:
    """GET with retries. 404 -> None (missing file is fine). Other persistent errors raise."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = _session().get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.content
            if r.status_code in (403,):  # CDN answers 403 for some non-existent keys
                return None
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:
            last = e
        time.sleep(min(30.0, 1.5 * 2 ** attempt))
    assert last is not None
    raise last


def fetch_zip(url: str, tmpdir: Path, verify: bool = True) -> bytes | None:
    """Download `url` (+ .CHECKSUM) into tmpdir, verify sha256, return the zip bytes (None = 404)."""
    for attempt in range(3):
        body = http_get(url)
        if body is None:
            return None
        zpath = tmpdir / url.rsplit("/", 1)[1]
        zpath.write_bytes(body)
        if not verify:
            return body
        cs = http_get(url + ".CHECKSUM")
        if cs is None:
            return body  # no checksum published
        want = cs.decode("utf-8", "replace").split()[0].strip().lower()
        got = hashlib.sha256(body).hexdigest()
        if want == got:
            return body
        log(f"  checksum mismatch {url} (attempt {attempt + 1}) want={want[:12]} got={got[:12]}")
        time.sleep(1 + attempt)
    raise IOError(f"checksum mismatch persisted for {url}")


# --------------------------------------------------------------------------- parsing
def _read_csv_member(zbytes: bytes) -> pd.DataFrame:
    """First CSV inside the zip as an all-string frame with no header handling (header=None)."""
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        names = [i for i in z.infolist() if i.filename.lower().endswith(".csv")]
        if not names:
            raise ValueError("zip has no csv member")
        info = names[0]
        if info.file_size > MAX_UNCOMPRESSED:
            raise ValueError(f"csv member too large: {info.file_size}")
        with z.open(info) as fh:
            return pd.read_csv(fh, header=None, dtype=str, keep_default_na=False)


def _drop_header(raw: pd.DataFrame, headers: set[str]) -> pd.DataFrame:
    if len(raw) and str(raw.iloc[0, 0]).strip().lower() in headers:
        raw = raw.iloc[1:]
    return raw.reset_index(drop=True)


def parse_klines_csv(zbytes: bytes) -> pd.DataFrame:
    raw = _drop_header(_read_csv_member(zbytes), {"open_time", "open time"})
    if raw.shape[1] < 11:
        raise ValueError(f"klines csv has {raw.shape[1]} columns")
    pos = {"open_time": 0, "open": 1, "high": 2, "low": 3, "close": 4, "volume": 5,
           "quote_volume": 7, "trades": 8, "taker_buy_volume": 9, "taker_buy_quote_volume": 10}
    df = pd.DataFrame({k: raw.iloc[:, v] for k, v in pos.items()})
    return frames.normalize_klines(df)


def parse_metrics_csv(zbytes: bytes) -> pd.DataFrame:
    raw = _drop_header(_read_csv_member(zbytes), {"create_time"})
    if raw.shape[1] < 8:
        raise ValueError(f"metrics csv has {raw.shape[1]} columns")
    # col 1 is `symbol`
    df = pd.DataFrame({"create_time": raw.iloc[:, 0]})
    for i, c in enumerate(METRIC_COLS[1:], start=2):
        df[c] = raw.iloc[:, i].replace("", np.nan)
    return frames.normalize_metrics(df)


def parse_funding_csv(zbytes: bytes) -> pd.DataFrame:
    raw = _drop_header(_read_csv_member(zbytes), {"calc_time"})
    if raw.shape[1] < 2:
        raise ValueError(f"funding csv has {raw.shape[1]} columns")
    # (calc_time, [funding_interval_hours,] last_funding_rate)
    df = pd.DataFrame({"calc_time": raw.iloc[:, 0], "funding_rate": raw.iloc[:, raw.shape[1] - 1]})
    return frames.normalize_funding(df)


# --------------------------------------------------------------------------- planning helpers
def _covered(times: np.ndarray, lo: int, hi: int, min_rows: int | None, slack_ms: int) -> bool:
    """True if [lo, hi) already has data at both its start and end (within slack) and (optionally)
    at least min_rows rows."""
    seg = frames.times_in_range(times, lo, hi)
    if len(seg) == 0:
        return False
    if min_rows is not None and len(seg) >= min_rows:
        return True
    return bool(seg[0] <= lo + slack_ms and seg[-1] >= hi - slack_ms - BAR_MS)


class Stats(dict):
    def __init__(self):
        super().__init__(units=0, skipped=0, downloaded=0, missing=0, failed=0, rows_added=0)


def _run_units(units, fetch_fn, stats: Stats, label: str, progress_every: int = 25):
    """Sequentially fetch each (name, url) unit; return list of frames."""
    out = []
    for i, (name, url) in enumerate(units, 1):
        stats["units"] += 1
        try:
            part = fetch_fn(url)
        except Exception as e:  # noqa: BLE001 - logged, counted, never silent
            stats["failed"] += 1
            log(f"  FAIL {label} {name}: {type(e).__name__}: {e}")
            continue
        if part is None:
            stats["missing"] += 1
        else:
            stats["downloaded"] += 1
            out.append(part)
        if progress_every and i % progress_every == 0:
            log(f"  {label}: {i}/{len(units)} units")
    return out


# --------------------------------------------------------------------------- per-kind jobs
def download_klines(symbol: str, months: int, root: Path | None = None, verify: bool = True,
                    today: dt.date | None = None) -> Stats:
    root = root or archive_root()
    today = today or utc_today()
    full, cur = window_months(months, today)
    path = kline_path(symbol, root)
    old = frames.read_parquet(path, frames.empty_klines)
    times = old["open_time"].to_numpy()
    stats = Stats()
    parts: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="tb_dl_") as td:
        tmp = Path(td)

        def get(url):
            z = fetch_zip(url, tmp, verify)
            return None if z is None else parse_klines_csv(z)

        def day_units(days):
            u = []
            for d in days:
                lo = to_ms(d)
                if _covered(times, lo, lo + DAY_MS, 286, 3 * BAR_MS):
                    stats["skipped"] += 1
                else:
                    u.append((d.isoformat(), daily_url("klines", symbol, d.isoformat())))
            return u

        for m in full:
            nxt = add_months(m, 1)
            lo, hi = to_ms(m), to_ms(nxt)
            if _covered(times, lo, hi, None, 3_600_000):
                stats["skipped"] += 1
                continue
            got = _run_units([(m.strftime("%Y-%m"), monthly_url("klines", symbol, m.strftime("%Y-%m")))],
                             get, stats, f"{symbol} klines", progress_every=0)
            if got:
                parts += got
            else:  # monthly not (yet) published -> daily fallback, only for the latest full month
                if nxt >= cur:
                    parts += _run_units(day_units(days_between(m, nxt)), get, stats,
                                        f"{symbol} klines(d)")
        # current month: dailies up to yesterday (today's file does not exist until tomorrow)
        parts += _run_units(day_units(days_between(cur, today)), get, stats, f"{symbol} klines(d)")
    new = pd.concat(parts, ignore_index=True) if parts else frames.empty_klines()
    merged = frames.merge_frames(old, new, "open_time")
    stats["rows_added"] = len(merged) - len(old)
    if stats["rows_added"] or not path.exists():
        frames.write_parquet_atomic(merged, path)
    return stats


def download_metrics(symbol: str, months: int, root: Path | None = None, verify: bool = True,
                     today: dt.date | None = None) -> Stats:
    root = root or archive_root()
    today = today or utc_today()
    full, cur = window_months(months, today)
    path = metrics_path(symbol, root)
    old = frames.read_parquet(path, frames.empty_metrics)
    times = old["create_time"].to_numpy()
    stats = Stats()
    units = []
    for d in days_between(full[0], today):
        lo = to_ms(d)
        if _covered(times, lo, lo + DAY_MS, 286, 3 * BAR_MS):
            stats["skipped"] += 1
        else:
            units.append((d.isoformat(), daily_url("metrics", symbol, d.isoformat())))
    with tempfile.TemporaryDirectory(prefix="tb_dl_") as td:
        tmp = Path(td)

        def get(url):
            z = fetch_zip(url, tmp, verify)
            return None if z is None else parse_metrics_csv(z)

        parts = _run_units(units, get, stats, f"{symbol} metrics", progress_every=60)
    new = pd.concat(parts, ignore_index=True) if parts else frames.empty_metrics()
    merged = frames.merge_frames(old, new, "create_time")
    stats["rows_added"] = len(merged) - len(old)
    if stats["rows_added"] or not path.exists():
        frames.write_parquet_atomic(merged, path)
    return stats


def download_funding(symbol: str, months: int, root: Path | None = None, verify: bool = True,
                     today: dt.date | None = None) -> Stats:
    """Monthly zips for full months. The current month has no archive file; live.top_up covers it."""
    root = root or archive_root()
    today = today or utc_today()
    full, cur = window_months(months, today)
    path = funding_path(symbol, root)
    old = frames.read_parquet(path, frames.empty_funding)
    times = old["calc_time"].to_numpy()
    stats = Stats()
    units = []
    for m in full:
        lo, hi = to_ms(m), to_ms(add_months(m, 1))
        if _covered(times, lo, hi, None, 9 * 3_600_000):
            stats["skipped"] += 1
        else:
            units.append((m.strftime("%Y-%m"), monthly_url("funding", symbol, m.strftime("%Y-%m"))))
    with tempfile.TemporaryDirectory(prefix="tb_dl_") as td:
        tmp = Path(td)

        def get(url):
            z = fetch_zip(url, tmp, verify)
            return None if z is None else parse_funding_csv(z)

        parts = _run_units(units, get, stats, f"{symbol} funding", progress_every=0)
    new = pd.concat(parts, ignore_index=True) if parts else frames.empty_funding()
    merged = frames.merge_frames(old, new, "calc_time")
    stats["rows_added"] = len(merged) - len(old)
    if stats["rows_added"] or not path.exists():
        frames.write_parquet_atomic(merged, path)
    return stats


def universe_symbols(min_age_months: int = 24, today: dt.date | None = None) -> list[str]:
    """Every USDT-quoted PERPETUAL with status TRADING whose onboardDate is at least
    `min_age_months` months old, from fapi exchangeInfo (sorted)."""
    today = today or utc_today()
    cutoff = to_ms(add_months(month_start(today), -min_age_months).replace(day=min(today.day, 28)))
    r = _session().get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30)
    r.raise_for_status()
    out = []
    for s in r.json()["symbols"]:
        if (s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL"
                and s.get("status") == "TRADING" and int(s.get("onboardDate", 1 << 62)) <= cutoff):
            out.append(s["symbol"])
    return sorted(out)


_JOBS = {"klines": download_klines, "metrics": download_metrics, "funding": download_funding}


def download_all(symbols: list[str], kinds: list[str], months: int = 12, workers: int = 6,
                 root: Path | None = None, verify: bool = True) -> dict:
    """Run (symbol, kind) jobs on a thread pool. Returns {(symbol, kind): Stats | Exception-string}."""
    root = root or archive_root()
    jobs = [(s, k) for k in kinds for s in symbols]
    results: dict = {}
    t0 = time.time()
    log(f"archive: {len(symbols)} symbols x {kinds} x {months} months -> {root}  ({len(jobs)} jobs, "
        f"{workers} workers)")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_JOBS[k], s, months, root, verify): (s, k) for s, k in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            s, k = futs[f]
            try:
                st = f.result()
                results[(s, k)] = st
                log(f"[{n}/{len(jobs)}] {s:14s} {k:8s} +{st['rows_added']:>7d} rows  "
                    f"dl={st['downloaded']} skip={st['skipped']} 404={st['missing']} "
                    f"fail={st['failed']}  ({time.time() - t0:.0f}s)")
            except Exception as e:  # noqa: BLE001
                results[(s, k)] = f"ERROR {type(e).__name__}: {e}"
                log(f"[{n}/{len(jobs)}] {s:14s} {k:8s} ERROR {type(e).__name__}: {e}")
    return results


def summarize(symbols: list[str], root: Path | None = None) -> pd.DataFrame:
    """Rows / date range / gaps per symbol and kind from the parquet files."""
    root = root or archive_root()
    rows = []
    for s in symbols:
        for kind, path, col, step in (("klines", kline_path(s, root), "open_time", BAR_MS),
                                      ("metrics", metrics_path(s, root), "create_time", BAR_MS),
                                      ("funding", funding_path(s, root), "calc_time", None)):
            if not path.exists():
                rows.append(dict(symbol=s, kind=kind, rows=0))
                continue
            t = pd.read_parquet(path, columns=[col])[col].to_numpy()
            gaps = int((np.diff(t) > step).sum()) if step and len(t) > 1 else 0
            missing = int((np.diff(t) // step - 1)[np.diff(t) > step].sum()) if step and len(t) > 1 else 0
            rows.append(dict(
                symbol=s, kind=kind, rows=len(t),
                start=dt.datetime.fromtimestamp(t[0] / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M") if len(t) else "",
                end=dt.datetime.fromtimestamp(t[-1] / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M") if len(t) else "",
                gaps=gaps, missing_bars=missing, mb=round(path.stat().st_size / 1e6, 2)))
    return pd.DataFrame(rows)
