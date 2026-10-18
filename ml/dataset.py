"""Dataset builder: archive parquet -> features (fp16) + trade labels/outcomes for every bar.

Layout of a built ``Dataset`` (all rows of all symbols concatenated, symbol blocks contiguous and
time-ordered; each symbol block starts after the feature warm-up so every stored row is valid):

    X        float16 [N, F]     raw (unnormalised) features
    t        int64   [N]        bar open_time (ms)
    sym      int16   [N]        symbol id (index into ``symbols``)
    pos      int32   [N]        row position inside the symbol block (windows need pos >= WINDOW-1)
    lab[k]   float32 [N]        k in {fixed_long, fixed_short, atr_long, atr_short}: 1 / 0 / NaN
    pnl[k]   float32 [N]        net USD per trade for the same k
    off[k]   int32   [N]        bars until the trade closed (for de-overlapping)

Cached under ``var/ml_cache/<key>/`` as .npy files so big arrays can be np.memmap'ed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from tb import features as ft  # noqa: E402
from tb.labels import full_cfg, simulate_trades  # noqa: E402

# Label horizon: the live template has no time stop (a trade runs to TP/SL), so labels must look far enough
# ahead that "timeout" is rare for the 5/10 USD fixed exit (3.1 % / 6.25 % barriers).  With the contract's
# default of 288 bars (24 h) ~70 % of fixed-exit trades time out and the label degenerates into a
# volatility detector.  4 days keeps timeouts small; the walk-forward purge gap is horizon + window.
HORIZON = 1152
# exit configurations ("modes"): name -> overrides of tb.labels.DEFAULT_EXIT_CFG
EXIT_SETS = {
    "fixed": {"exit_mode": "fixed"},                                     # TP +5 USD / SL -10 USD
    "atr": {"exit_mode": "atr", "tp_atr": 1.5, "sl_atr": 1.0},           # template default
    "atr32": {"exit_mode": "atr", "tp_atr": 3.0, "sl_atr": 2.0},
    "atr63": {"exit_mode": "atr", "tp_atr": 6.0, "sl_atr": 3.0},
}
MODES = tuple(EXIT_SETS)
SIDES = ("long", "short")
KEYS = [f"{m}_{s}" for m in MODES for s in SIDES]


def archive_root() -> Path:
    return Path(os.environ.get("TB_ARCHIVE") or os.environ.get("TB_ARCHIVE_DIR") or ROOT / "data" / "archive")


def var_dir() -> Path:
    return Path(os.environ.get("TB_VAR") or os.environ.get("TB_VAR_DIR") or ROOT / "var")


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _clean_time(df: pd.DataFrame | None, col: str) -> pd.DataFrame | None:
    if df is None or not len(df):
        return None
    df = df.copy()
    t = df[col]
    if not pd.api.types.is_integer_dtype(t):
        if pd.api.types.is_datetime64_any_dtype(t):
            df[col] = ((pd.to_datetime(t, utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
                       // pd.Timedelta(milliseconds=1)).astype("int64")
        else:
            df[col] = pd.to_numeric(t).astype("int64")
    return df.drop_duplicates(col).sort_values(col).reset_index(drop=True)


def load_symbol(sym: str, root: Path | None = None):
    """(klines, metrics|None, funding|None) straight from the archive parquet files."""
    root = Path(root or archive_root())
    kl = _clean_time(_read(root / "klines" / "5m" / f"{sym}.parquet"), "open_time")
    if kl is None:
        raise FileNotFoundError(f"no klines for {sym}")
    m = _clean_time(_read(root / "metrics" / f"{sym}.parquet"), "create_time")
    f = _clean_time(_read(root / "funding" / f"{sym}.parquet"), "calc_time")
    return kl, m, f


def list_symbols(root: Path | None = None) -> list[str]:
    root = Path(root or archive_root())
    return sorted(p.stem for p in (root / "klines" / "5m").glob("*.parquet"))


@dataclass
class Dataset:
    symbols: list[str]
    X: np.ndarray
    t: np.ndarray
    sym: np.ndarray
    pos: np.ndarray
    lab: dict[str, np.ndarray]
    pnl: dict[str, np.ndarray]
    off: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)

    @property
    def sample_ok(self) -> np.ndarray:
        """rows that can be the last bar of a full WINDOW inside their symbol block."""
        return self.pos >= ft.WINDOW - 1


def _cache_key(symbols, root, cfgs, horizon) -> str:
    h = hashlib.sha1()
    h.update(json.dumps([symbols, str(root), cfgs, horizon, ft.FEATURE_NAMES, ft.WARMUP], sort_keys=True,
                        default=str).encode())
    for s in symbols:
        for sub in (f"klines/5m/{s}.parquet", f"metrics/{s}.parquet", f"funding/{s}.parquet"):
            p = Path(root) / sub
            h.update(f"{sub}:{p.stat().st_size if p.exists() else -1}:{int(p.stat().st_mtime) if p.exists() else 0}".encode())
    return h.hexdigest()[:16]


def _build_one(job) -> dict:
    """Features + labels for one symbol (runs in a worker process; numpy/pandas only, no torch)."""
    sym, root, cfgs, horizon = job
    root = Path(root)
    kl, m, f = load_symbol(sym, root)
    btc = load_symbol("BTCUSDT", root)[0] if (root / "klines" / "5m" / "BTCUSDT.parquet").exists() else None
    kl = ft.regularize(kl)
    if len(kl) < ft.WARMUP + 2 * ft.WINDOW + horizon:
        return dict(name=sym, skip=True, bars=len(kl), t=np.zeros(0))
    x = ft.build_features(kl, m, f, btc)
    lab, pnl, off = {}, {}, {}
    for mode in MODES:
        for side in SIDES:
            sim = simulate_trades(kl, side, cfgs[mode], horizon)
            lab[f"{mode}_{side}"] = sim["label"]
            pnl[f"{mode}_{side}"] = sim["pnl_usd"]
            off[f"{mode}_{side}"] = sim["exit_off"]
    sl = slice(ft.WARMUP, None)
    n = len(kl) - ft.WARMUP
    return dict(name=sym, no_metrics=m is None, X=x[sl].astype("float16"), t=kl["open_time"].to_numpy("int64")[sl],
                pos=np.arange(n, dtype="int32"), lab={k: v[sl] for k, v in lab.items()},
                pnl={k: v[sl] for k, v in pnl.items()}, off={k: v[sl] for k, v in off.items()})


def build_dataset(symbols: list[str] | None = None, root: Path | None = None, exit_cfgs: dict | None = None,
                  horizon: int = HORIZON, use_cache: bool = True, mmap: bool = False, log=print) -> Dataset:
    root = Path(root or archive_root())
    symbols = symbols or list_symbols(root)
    cfgs = {m: dict(full_cfg({**EXIT_SETS[m], **(exit_cfgs or {}).get(m, {})})) for m in MODES}
    key = _cache_key(symbols, root, cfgs, horizon)
    cdir = var_dir() / "ml_cache" / key
    if use_cache and (cdir / "meta.json").exists():
        log(f"[dataset] cache hit {cdir}")
        return load_cached(cdir, mmap=mmap)

    t0 = time.time()
    jobs = [(s_, str(root), cfgs, horizon) for s_ in symbols]
    parts_raw = []
    workers = int(os.environ.get("TB_DS_WORKERS", min(3, max(1, (os.cpu_count() or 2) - 2))))
    if workers > 1 and len(jobs) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for r in ex.map(_build_one, jobs):
                parts_raw.append(r)
                log(f"[dataset] {r['name']}: {len(r['t'])} rows ({time.time() - t0:.0f}s)")
    else:
        for j in jobs:
            r = _build_one(j)
            parts_raw.append(r)
            log(f"[dataset] {r['name']}: {len(r['t'])} rows ({time.time() - t0:.0f}s)")
    parts, no_metrics = [], []
    for r in parts_raw:
        if r.get("skip"):
            log(f"[dataset] skip {r['name']}: only {r['bars']} bars")
            continue
        if r["no_metrics"]:
            no_metrics.append(r["name"])
        r["sym"] = np.full(len(r["t"]), len(parts), dtype="int16")
        parts.append(r)
    names = [p["name"] for p in parts]
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    ds = Dataset(
        symbols=names, X=cat("X"), t=cat("t"), sym=cat("sym"), pos=cat("pos"),
        lab={k: np.concatenate([p["lab"][k] for p in parts]) for k in KEYS},
        pnl={k: np.concatenate([p["pnl"][k] for p in parts]) for k in KEYS},
        off={k: np.concatenate([p["off"][k] for p in parts]) for k in KEYS},
        meta=dict(exit_cfgs=cfgs, horizon=horizon, feature_names=ft.FEATURE_NAMES, window=ft.WINDOW,
                  warmup=ft.WARMUP, symbols_without_metrics=no_metrics, built_at=int(time.time())))
    if no_metrics:
        log(f"[dataset] WARNING: no metrics archive for {len(no_metrics)} symbols (derivative features = 0): "
            f"{no_metrics[:6]}...")
    if use_cache:
        save_cached(ds, cdir)
    log(f"[dataset] built {len(ds)} rows x {ft.F} features for {len(names)} symbols in {time.time() - t0:.0f}s")
    return ds


def save_cached(ds: Dataset, cdir: Path) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    for name in ("X", "t", "sym", "pos"):
        np.save(cdir / f"{name}.npy", getattr(ds, name))
    for k in KEYS:
        np.save(cdir / f"lab_{k}.npy", ds.lab[k])
        np.save(cdir / f"pnl_{k}.npy", ds.pnl[k])
        np.save(cdir / f"off_{k}.npy", ds.off[k])
    meta = dict(ds.meta, symbols=ds.symbols)
    (cdir / "meta.json").write_text(json.dumps(meta, default=str))


def load_cached(cdir: Path, mmap: bool = False) -> Dataset:
    mm = "r" if mmap else None
    meta = json.loads((cdir / "meta.json").read_text())
    ld = lambda n: np.load(cdir / f"{n}.npy", mmap_mode=mm)  # noqa: E731
    return Dataset(symbols=meta.pop("symbols"), X=ld("X"), t=np.load(cdir / "t.npy"), sym=np.load(cdir / "sym.npy"),
                   pos=np.load(cdir / "pos.npy"), lab={k: np.load(cdir / f"lab_{k}.npy") for k in KEYS},
                   pnl={k: np.load(cdir / f"pnl_{k}.npy") for k in KEYS},
                   off={k: np.load(cdir / f"off_{k}.npy") for k in KEYS}, meta=meta)


def free_memory_bytes() -> int:
    try:
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 4096
        vals = {}
        for line in out.splitlines():
            if ":" in line and "page size" not in line:
                k, v = line.split(":")
                vals[k.strip()] = int(v.strip().rstrip("."))
        return (vals.get("Pages free", 0) + vals.get("Pages inactive", 0) + vals.get("Pages speculative", 0)) * page
    except Exception:
        return 4 << 30


def ram_budget_bytes() -> int:
    """<= 8 GB, and at most half of the currently free + reclaimable memory."""
    return int(min(8 << 30, 0.5 * free_memory_bytes()))


def synthetic_dataset(n_symbols=4, n=6000, seed=0, planted_edge=0.0, exit_cfgs=None, horizon=HORIZON) -> Dataset:
    """Dataset built from ml/synth.py frames (no archive needed), for smoke tests."""
    sys.path.insert(0, str(ROOT / "ml"))
    from synth import make_market, make_symbol
    cfgs = {m: dict(full_cfg({**EXIT_SETS[m], **(exit_cfgs or {}).get(m, {})})) for m in MODES}
    btc, _, _ = make_symbol(n, seed=1000, start_price=30000.0)
    mk = make_market(n)
    parts = []
    for i in range(n_symbols):
        kl, m, f = make_symbol(n, seed=seed + i, market=mk, planted_edge=planted_edge)
        x = ft.build_features(kl, m, f, btc)
        sl = slice(ft.WARMUP, None)
        k = len(kl) - ft.WARMUP
        lab, pnl, off = {}, {}, {}
        for mode in MODES:
            for side in SIDES:
                sim = simulate_trades(kl, side, cfgs[mode], horizon)
                lab[f"{mode}_{side}"], pnl[f"{mode}_{side}"], off[f"{mode}_{side}"] = (
                    sim["label"][sl], sim["pnl_usd"][sl], sim["exit_off"][sl])
        parts.append(dict(X=x[sl].astype("float16"), t=kl["open_time"].to_numpy()[sl],
                          sym=np.full(k, i, "int16"), pos=np.arange(k, dtype="int32"), lab=lab, pnl=pnl, off=off))
    cat = lambda k_: np.concatenate([p[k_] for p in parts])  # noqa: E731
    return Dataset(symbols=[f"SYN{i}USDT" for i in range(n_symbols)], X=cat("X"), t=cat("t"), sym=cat("sym"),
                   pos=cat("pos"), lab={k_: np.concatenate([p["lab"][k_] for p in parts]) for k_ in KEYS},
                   pnl={k_: np.concatenate([p["pnl"][k_] for p in parts]) for k_ in KEYS},
                   off={k_: np.concatenate([p["off"][k_] for p in parts]) for k_ in KEYS},
                   meta=dict(exit_cfgs=cfgs, horizon=horizon, feature_names=ft.FEATURE_NAMES, window=ft.WINDOW,
                             warmup=ft.WARMUP, synthetic=True))
