"""Dataset builder: archive parquet -> features + trade labels + auxiliary targets, stored as memmaps on disk.

Everything is concatenated over symbols (each symbol block contiguous, time ordered, starting after the
feature warm-up) and cached under ``var/ml_cache/<key>/`` as .npy so it can be opened with
``mmap_mode="r"`` - the matrix for 229 symbols x 24 months is several GB and must never be loaded whole.

    X     float16 [N, F]    raw (unnormalised) features
    t     int64   [N]       bar open_time (ms)          sym int16 [N] symbol id    pos int32 [N] row in block
    lab   int8    [K, N]    1 TP first / 0 SL-first-or-timeout / -1 n.a.      (K = len(MODES) * 2 label keys)
    pnl   float16 [K, N]    net USD per trade           off uint16 [K, N] bars until the trade closed (0 = n.a.)
    aux   float16 [3, N]    fvol, up_exc, dn_exc (tb.labels.aux_targets), NaN where n.a.

Label/aux storage is column-major ([K, N]) so one label column is one contiguous read.  Access goes through
``ds.lab[key]`` etc.: ``col[rows]`` returns float32 (lab: NaN for n.a.) / int32 (off).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from tb import features as ft  # noqa: E402
from tb import features_depth  # noqa: E402,F401  (registers the optional 'depth' feature group)
from tb.labels import AUX_NAMES, aux_targets, full_cfg, simulate_trades  # noqa: E402

# Label horizon: the live template has no time stop, so labels must look far enough ahead that "timeout" is
# rare for the 5/10 USD fixed exit (3.1 % / 6.25 % barriers).  With the contract's default of 288 bars (24 h)
# ~70 % of fixed-exit trades time out and the label degenerates into a volatility detector.  4 days keeps
# timeouts ~10 %; the walk-forward purge gap is horizon + window.
HORIZON = 1152
# exit configurations ("modes"): name -> overrides of tb.labels.DEFAULT_EXIT_CFG
EXIT_SETS = {
    "fixed": {"exit_mode": "fixed"},                                     # TP +5 USD / SL -10 USD
    "atr": {"exit_mode": "atr", "tp_atr": 1.5, "sl_atr": 1.0},           # template default (break-even unreachable)
    "atr32": {"exit_mode": "atr", "tp_atr": 3.0, "sl_atr": 2.0},
    "atr63": {"exit_mode": "atr", "tp_atr": 6.0, "sl_atr": 3.0},
}
DEFAULT_MODES = ("fixed", "atr32", "atr63")
SIDES = ("long", "short")
MODES = DEFAULT_MODES
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


# --------------------------------------------------------------------------------------------
class Col:
    """One column of a [K, N] memmap; ``col[rows]`` -> float32 (lab/pnl/aux) or int32 (off)."""

    def __init__(self, arr, j: int, kind: str):
        self.arr, self.j, self.kind = arr, j, kind

    def __len__(self):
        return self.arr.shape[1]

    def __getitem__(self, sel):
        v = np.asarray(self.arr[self.j][sel])
        if self.kind == "lab":
            out = v.astype("float32")
            out[v < 0] = np.nan
            return out
        if self.kind == "off":
            return v.astype("int32")
        return v.astype("float32")

    def valid(self, sel=slice(None)) -> np.ndarray:
        """bool: label available (lab >= 0)."""
        return np.asarray(self.arr[self.j][sel]) >= 0


class _Cols:
    def __init__(self, arr, names, kind):
        self.arr, self.names, self.kind = arr, list(names), kind

    def __getitem__(self, key: str) -> Col:
        return Col(self.arr, self.names.index(key), self.kind)


@dataclass
class Dataset:
    symbols: list[str]
    X: np.ndarray
    t: np.ndarray
    sym: np.ndarray
    pos: np.ndarray
    lab: _Cols
    pnl: _Cols
    off: _Cols
    aux: _Cols
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)

    @property
    def n_feat(self) -> int:
        return int(self.X.shape[1])

    @property
    def feature_names(self) -> list[str]:
        return list(self.meta.get("feature_names", ft.FEATURE_NAMES))

    @property
    def groups(self) -> list[str]:
        return list(self.meta.get("groups", []))

    def with_cols(self, n: int) -> "Dataset":
        """View on the first ``n`` feature columns (e.g. the base 49, dropping optional groups) - same rows."""
        from dataclasses import replace
        meta = dict(self.meta, feature_names=self.feature_names[:n], groups=[] if n <= len(ft.FEATURE_NAMES) else self.groups)
        return replace(self, X=self.X[:, :n], meta=meta)

    def sample_ok_at(self, rows) -> np.ndarray:
        return np.asarray(self.pos[rows]) >= ft.WINDOW - 1

    def any_label(self, mode: str, rows) -> np.ndarray:
        return self.lab[f"{mode}_long"].valid(rows) | self.lab[f"{mode}_short"].valid(rows)


# --------------------------------------------------------------------------------------------
def _cache_key(symbols, root, cfgs, horizon, groups, start_ms=0) -> str:
    h = hashlib.sha1()
    h.update(json.dumps([symbols, str(root), cfgs, horizon, start_ms, ft.feature_names(groups) if groups else ft.FEATURE_NAMES,
                         list(groups), ft.WARMUP, list(AUX_NAMES), list(MODES)], sort_keys=True, default=str).encode())
    for s in symbols:
        for sub in (f"klines/5m/{s}.parquet", f"metrics/{s}.parquet", f"funding/{s}.parquet", f"depth/{s}.parquet"):
            p = Path(root) / sub
            if sub.startswith("depth") and "depth" not in groups:
                continue
            h.update(f"{sub}:{p.stat().st_size if p.exists() else -1}".encode())
    return h.hexdigest()[:16]


def _build_one(job) -> dict:
    """Features + labels for one symbol (runs in a worker process; numpy/pandas only, no torch).  Saves the
    result as an .npz part file and returns its small header."""
    sym, root, cfgs, horizon, groups, part_path, start_ms = job
    root = Path(root)
    kl, m, f = load_symbol(sym, root)
    btc = load_symbol("BTCUSDT", root)[0] if (root / "klines" / "5m" / "BTCUSDT.parquet").exists() else None
    kl = ft.regularize(kl)
    if len(kl) < ft.WARMUP + 2 * ft.WINDOW + horizon:
        return dict(name=sym, skip=True, bars=len(kl), n=0)
    extra = {}
    if "depth" in groups:
        from tb import features_depth  # noqa: F401  (registers the group)
        extra["depth"] = _read(root / "depth" / f"{sym}.parquet")
    x = ft.build_features(kl, m, f, btc, groups=list(groups), extra_data=extra)
    lab, pnl, off = [], [], []
    for mode in MODES:
        for side in SIDES:
            sim = simulate_trades(kl, side, cfgs[mode], horizon)
            lab.append(np.where(np.isnan(sim["label"]), -1, sim["label"]).astype("int8"))
            pnl.append(sim["pnl_usd"].astype("float16"))
            off.append(sim["exit_off"].astype("uint16"))
    ax = aux_targets(kl)
    aux = np.stack([ax[k] for k in AUX_NAMES]).astype("float16")
    t_all = kl["open_time"].to_numpy("int64")
    first = max(ft.WARMUP, int(np.searchsorted(t_all, start_ms)))      # drop warm-up rows and anything before start_ms
    sl = slice(first, None)
    n = len(kl) - first
    if n <= 0:
        return dict(name=sym, skip=True, bars=len(kl), n=0)
    np.savez(part_path, X=x[sl].astype("float16"), t=kl["open_time"].to_numpy("int64")[sl],
             lab=np.stack(lab)[:, sl], pnl=np.stack(pnl)[:, sl], off=np.stack(off)[:, sl], aux=aux[:, sl])
    return dict(name=sym, skip=False, n=n, no_metrics=m is None,
                has_depth=("depth" in extra and extra["depth"] is not None))


def _assemble(headers: list[dict], part_dir: Path, cdir: Path, meta: dict, log=print) -> None:
    """Concatenate part files into memmapped .npy files in ``cdir`` (one pass, bounded RAM)."""
    ok = [h for h in headers if not h["skip"]]
    total = sum(h["n"] for h in ok)
    nf = len(meta["feature_names"])
    K = len(KEYS)
    cdir.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap
    X = mm(cdir / "X.npy", "w+", np.float16, (total, nf))
    t = mm(cdir / "t.npy", "w+", np.int64, (total,))
    sym = mm(cdir / "sym.npy", "w+", np.int16, (total,))
    pos = mm(cdir / "pos.npy", "w+", np.int32, (total,))
    lab = mm(cdir / "lab.npy", "w+", np.int8, (K, total))
    pnl = mm(cdir / "pnl.npy", "w+", np.float16, (K, total))
    off = mm(cdir / "off.npy", "w+", np.uint16, (K, total))
    aux = mm(cdir / "aux.npy", "w+", np.float16, (len(AUX_NAMES), total))
    r = 0
    for si, h in enumerate(ok):
        with np.load(part_dir / f"{h['name']}.npz") as z:
            n = h["n"]
            X[r:r + n] = z["X"]
            t[r:r + n] = z["t"]
            sym[r:r + n] = si
            pos[r:r + n] = np.arange(n, dtype="int32")
            lab[:, r:r + n] = z["lab"]
            pnl[:, r:r + n] = z["pnl"]
            off[:, r:r + n] = z["off"]
            aux[:, r:r + n] = z["aux"]
        r += n
    for a in (X, t, sym, pos, lab, pnl, off, aux):
        a.flush()
    meta = dict(meta, symbols=[h["name"] for h in ok], rows=int(total),
                symbols_without_metrics=[h["name"] for h in ok if h.get("no_metrics")],
                symbols_without_depth=[h["name"] for h in ok if "depth" in meta["groups"] and not h.get("has_depth")],
                skipped=[h["name"] for h in headers if h["skip"]], keys=KEYS, aux_names=list(AUX_NAMES),
                built_at=int(time.time()))
    (cdir / "meta.json").write_text(json.dumps(meta, default=str))
    log(f"[dataset] assembled {total} rows x {nf} features, {len(ok)} symbols -> {cdir}")


def load_cached(cdir: Path) -> Dataset:
    meta = json.loads((cdir / "meta.json").read_text())
    ld = lambda n: np.load(cdir / f"{n}.npy", mmap_mode="r")  # noqa: E731
    lab, pnl, off, aux = ld("lab"), ld("pnl"), ld("off"), ld("aux")
    keys = meta["keys"]
    return Dataset(symbols=meta["symbols"], X=ld("X"), t=ld("t"), sym=ld("sym"), pos=ld("pos"),
                   lab=_Cols(lab, keys, "lab"), pnl=_Cols(pnl, keys, "pnl"), off=_Cols(off, keys, "off"),
                   aux=_Cols(aux, meta["aux_names"], "aux"), meta=meta)


def build_dataset(symbols: list[str] | None = None, root: Path | None = None, exit_cfgs: dict | None = None,
                  horizon: int = HORIZON, use_cache: bool = True, groups=(), log=print, workers: int | None = None,
                  start_ms: int = 0) -> Dataset:
    root = Path(root or archive_root())
    symbols = symbols or list_symbols(root)
    groups = tuple(groups)
    cfgs = {m: dict(full_cfg({**EXIT_SETS[m], **(exit_cfgs or {}).get(m, {})})) for m in MODES}
    key = _cache_key(symbols, root, cfgs, horizon, groups, start_ms)
    cdir = var_dir() / "ml_cache" / key
    if use_cache and (cdir / "meta.json").exists():
        log(f"[dataset] cache hit {cdir}")
        return load_cached(cdir)
    if "depth" in groups:
        from tb import features_depth  # noqa: F401
    t0 = time.time()
    part_dir = var_dir() / "ml_cache" / f"parts-{key}"
    part_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s_, str(root), cfgs, horizon, groups, str(part_dir / f"{s_}.npz"), start_ms) for s_ in symbols]
    workers = workers or int(os.environ.get("TB_DS_WORKERS", min(3, max(1, (os.cpu_count() or 2) - 2))))
    headers = []
    if workers > 1 and len(jobs) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for i, h in enumerate(ex.map(_build_one, jobs)):
                headers.append(h)
                if i % 10 == 0 or i == len(jobs) - 1:
                    log(f"[dataset] {i + 1}/{len(jobs)} {h['name']} ({time.time() - t0:.0f}s)")
    else:
        for i, j in enumerate(jobs):
            headers.append(_build_one(j))
            if i % 10 == 0 or i == len(jobs) - 1:
                log(f"[dataset] {i + 1}/{len(jobs)} {headers[-1]['name']} ({time.time() - t0:.0f}s)")
    meta = dict(exit_cfgs=cfgs, horizon=horizon, feature_names=ft.feature_names(groups), groups=list(groups),
                window=ft.WINDOW, warmup=ft.WARMUP)
    _assemble(headers, part_dir, cdir, meta, log)
    for p in part_dir.glob("*.npz"):
        p.unlink()
    part_dir.rmdir()
    ds = load_cached(cdir)
    if ds.meta["symbols_without_metrics"]:
        log(f"[dataset] WARNING: no metrics archive for {len(ds.meta['symbols_without_metrics'])} symbols")
    if "depth" in groups:
        log(f"[dataset] {len(ds.meta['symbols_without_depth'])} symbols without depth data (flag has_depth = 0)")
    log(f"[dataset] built in {time.time() - t0:.0f}s")
    return ds


def free_memory_bytes() -> int:
    try:
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        vals = {}
        for line in out.splitlines():
            if ":" in line and "page size" not in line:
                k, v = line.split(":")
                vals[k.strip()] = int(v.strip().rstrip("."))
        return (vals.get("Pages free", 0) + vals.get("Pages inactive", 0) + vals.get("Pages speculative", 0)) * 4096
    except Exception:
        return 4 << 30


def ram_budget_bytes() -> int:
    """<= 8 GB, and at most half of the currently free + reclaimable memory."""
    return int(min(8 << 30, 0.5 * free_memory_bytes()))


def synthetic_dataset(n_symbols=4, n=6000, seed=0, planted_edge=0.0, exit_cfgs=None, horizon=HORIZON,
                      groups=(), depth_cov=1.0) -> Dataset:
    """Dataset from ml/synth.py frames (no archive needed), for smoke tests.  Written to a temp cache dir."""
    sys.path.insert(0, str(ROOT / "ml"))
    from synth import make_depth, make_market, make_symbol
    groups = tuple(groups)
    if "depth" in groups:
        from tb import features_depth  # noqa: F401
    cfgs = {m: dict(full_cfg({**EXIT_SETS[m], **(exit_cfgs or {}).get(m, {})})) for m in MODES}
    cdir = Path(tempfile.mkdtemp(prefix="tb_synth_ds_"))
    part_dir = cdir / "parts"
    part_dir.mkdir()
    btc, _, _ = make_symbol(n, seed=1000, start_price=30000.0)
    mk = make_market(n)
    headers = []
    for i in range(n_symbols):
        kl, m, f = make_symbol(n, seed=seed + i, market=mk, planted_edge=planted_edge)
        extra = {"depth": make_depth(kl, seed + i, cov=depth_cov)} if "depth" in groups else {}
        x = ft.build_features(kl, m, f, btc, groups=list(groups), extra_data=extra)
        lab, pnl, off = [], [], []
        for mode in MODES:
            for side in SIDES:
                sim = simulate_trades(kl, side, cfgs[mode], horizon)
                lab.append(np.where(np.isnan(sim["label"]), -1, sim["label"]).astype("int8"))
                pnl.append(sim["pnl_usd"].astype("float16"))
                off.append(sim["exit_off"].astype("uint16"))
        ax = aux_targets(kl)
        sl = slice(ft.WARMUP, None)
        name = f"SYN{i}USDT"
        np.savez(part_dir / f"{name}.npz", X=x[sl].astype("float16"), t=kl["open_time"].to_numpy("int64")[sl],
                 lab=np.stack(lab)[:, sl], pnl=np.stack(pnl)[:, sl], off=np.stack(off)[:, sl],
                 aux=np.stack([ax[k] for k in AUX_NAMES]).astype("float16")[:, sl])
        headers.append(dict(name=name, skip=False, n=len(kl) - ft.WARMUP, no_metrics=False, has_depth=bool(extra)))
    meta = dict(exit_cfgs=cfgs, horizon=horizon, feature_names=ft.feature_names(groups), groups=list(groups),
                window=ft.WINDOW, warmup=ft.WARMUP, synthetic=True)
    _assemble(headers, part_dir, cdir, meta, lambda *a: None)
    return load_cached(cdir)
