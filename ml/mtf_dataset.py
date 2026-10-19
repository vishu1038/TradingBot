"""Decision-cadence dataset for the multi-timeframe experiments (B / C / E): one row per coin per closed
decision bar (default: every closed 1h bar), features from ``tb.features_mtf.build_for_model``'s builder (the same
function the live engine calls), labels from ``tb.labels`` with 1-4 day horizons.

Cached under ``var/ml_cache/mtf-<key>/`` as memory-mapped .npy files (never loaded whole):

    X    float16 [N, F]     features  = base mtf columns + the kalman columns (``names``; ``cols(kalman)`` selects)
    t    int64   [N]        decision bar open_time (ms); entry is at the next bar's open
    sym  int16   [N]        symbol id          (rows are time ordered inside a symbol block)
    lab  int8    [K, N]     1 TP first / 0 SL-first-or-timeout / -1 n.a.     keys = "<exit>_<side>"
    pnl  float16 [K, N]     net USD (fees + slippage)     off uint16 [K, N] bars until the trade closed
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

import dataset as D  # noqa: E402
from tb import features as ft  # noqa: E402
from tb import features_mtf as fm  # noqa: E402
from tb.labels import full_cfg, simulate_trades  # noqa: E402

BAR_MS = 300_000
SIDES = ("long", "short")
HORIZON = 1152                      # 4 days; the walk-forward purge is horizon + a day of slack
DECISION_EVERY = 12                 # closed 1h bars
TFS = fm.DEFAULT_TFS                # 1h, 4h, 1d (+ the 5m timing features, always)
GROUPS = ("kalman",)

# exit sets (name -> overrides of tb.labels.DEFAULT_EXIT_CFG).  pct = the user's +3 % / -6 % on the slice notional;
# atr1h32 = 3 x / 2 x the ATR14 of the last closed 1h bar (the analogue of the old atr32 on a 5m ATR).
EXITS = {
    "pct36": {"exit_mode": "pct", "tp_pct": 3.0, "sl_pct": 6.0},
    "atr1h32": {"exit_mode": "atr", "atr_tf": "1h", "tp_atr": 3.0, "sl_atr": 2.0},
}
KEYS = [f"{e}_{s}" for e in EXITS for s in SIDES]


def cfg_of(name: str) -> dict:
    return dict(full_cfg(EXITS[name]))


@dataclass
class MtfDataset:
    symbols: list[str]
    X: np.ndarray
    t: np.ndarray
    sym: np.ndarray
    lab: np.ndarray
    pnl: np.ndarray
    off: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)

    @property
    def names(self) -> list[str]:
        return list(self.meta["feature_names"])

    def cols(self, kalman: bool) -> np.ndarray:
        n = self.names
        return np.array([i for i, x in enumerate(n) if kalman or not x.startswith("kal_")], dtype="int64")

    def col_names(self, kalman: bool) -> list[str]:
        n = self.names
        return [n[i] for i in self.cols(kalman)]

    def key_idx(self, exit_name: str, side: str) -> int:
        return self.meta["keys"].index(f"{exit_name}_{side}")

    def y(self, exit_name: str, side: str, rows) -> np.ndarray:
        """float32 labels (NaN where n.a.)."""
        v = np.asarray(self.lab[self.key_idx(exit_name, side)][rows])
        out = v.astype("float32")
        out[v < 0] = np.nan
        return out

    def pnl_of(self, exit_name: str, side: str, rows) -> np.ndarray:
        return np.asarray(self.pnl[self.key_idx(exit_name, side)][rows]).astype("float32")

    def off_of(self, exit_name: str, side: str, rows) -> np.ndarray:
        return np.asarray(self.off[self.key_idx(exit_name, side)][rows]).astype("int32")

    def col(self, name: str, rows) -> np.ndarray:
        return np.asarray(self.X[rows, self.names.index(name)]).astype("float32")


def _build_one(job) -> dict:
    sym, root, horizon, every, part_path, start_ms = job
    root = Path(root)
    kl, m, f = D.load_symbol(sym, root)
    btc = D.load_symbol("BTCUSDT", root)[0] if (root / "klines" / "5m" / "BTCUSDT.parquet").exists() else None
    kl = ft.regularize(kl)
    warm = fm.warmup_bars_5m(TFS, GROUPS)
    if len(kl) < warm + 2 * every + horizon:
        return dict(name=sym, skip=True, bars=len(kl), n=0)
    t_all = kl["open_time"].to_numpy("int64")
    first = max(warm, int(np.searchsorted(t_all, start_ms)))
    dm = fm.decision_mask(t_all, every)
    rows = np.nonzero(dm & (np.arange(len(t_all)) >= first))[0]
    rows = rows[rows < len(kl) - horizon]               # labels need the full horizon (tail rows are n.a. anyway)
    if len(rows) == 0:
        return dict(name=sym, skip=True, bars=len(kl), n=0)
    X, _ = fm.build_mtf_features(kl, m, f, btc, TFS, GROUPS, rows=rows)
    lab, pnl, off = [], [], []
    for e in EXITS:
        cfg = cfg_of(e)
        for s in SIDES:
            sim = simulate_trades(kl, s, cfg, horizon, rows=rows)
            lab.append(np.where(np.isnan(sim["label"][rows]), -1, sim["label"][rows]).astype("int8"))
            pnl.append(sim["pnl_usd"][rows].astype("float16"))
            off.append(sim["exit_off"][rows].astype("uint16"))
    np.savez(part_path, X=X.astype("float16"), t=t_all[rows], lab=np.stack(lab), pnl=np.stack(pnl),
             off=np.stack(off))
    return dict(name=sym, skip=False, n=int(len(rows)), no_metrics=m is None)


def _cache_key(symbols, root, horizon, every, start_ms) -> str:
    h = hashlib.sha1()
    h.update(json.dumps([symbols, str(root), horizon, every, start_ms, fm.feature_names(TFS, GROUPS), list(KEYS),
                         [cfg_of(e) for e in EXITS], fm.warmup_bars_5m(TFS, GROUPS)], sort_keys=True,
                        default=str).encode())
    for s in symbols:
        for sub in (f"klines/5m/{s}.parquet", f"metrics/{s}.parquet", f"funding/{s}.parquet"):
            p = Path(root) / sub
            h.update(f"{sub}:{p.stat().st_size if p.exists() else -1}".encode())
    h.update(f"btc:{(Path(root) / 'klines/5m/BTCUSDT.parquet').stat().st_size if (Path(root) / 'klines/5m/BTCUSDT.parquet').exists() else -1}".encode())
    return h.hexdigest()[:16]


def load_cached(cdir: Path) -> MtfDataset:
    meta = json.loads((cdir / "meta.json").read_text())
    ld = lambda n: np.load(cdir / f"{n}.npy", mmap_mode="r")  # noqa: E731
    return MtfDataset(meta["symbols"], ld("X"), ld("t"), ld("sym"), ld("lab"), ld("pnl"), ld("off"), meta)


def build_dataset(symbols: list[str] | None = None, root: Path | None = None, horizon: int = HORIZON,
                  every: int = DECISION_EVERY, use_cache: bool = True, workers: int | None = None, start_ms: int = 0,
                  log=print) -> MtfDataset:
    root = Path(root or D.archive_root())
    symbols = symbols or D.list_symbols(root)
    key = _cache_key(symbols, root, horizon, every, start_ms)
    cdir = D.var_dir() / "ml_cache" / f"mtf-{key}"
    if use_cache and (cdir / "meta.json").exists():
        log(f"[mtf-dataset] cache hit {cdir}")
        return load_cached(cdir)
    t0 = time.time()
    part_dir = D.var_dir() / "ml_cache" / f"mtf-parts-{key}"
    part_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(root), horizon, every, str(part_dir / f"{s}.npz"), start_ms) for s in symbols]
    workers = workers or int(os.environ.get("TB_DS_WORKERS", 2))
    headers = []
    if workers > 1 and len(jobs) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for i, h in enumerate(ex.map(_build_one, jobs)):
                headers.append(h)
                if i % 10 == 0 or i == len(jobs) - 1:
                    log(f"[mtf-dataset] {i + 1}/{len(jobs)} {h['name']} ({time.time() - t0:.0f}s)")
    else:
        for i, j in enumerate(jobs):
            headers.append(_build_one(j))
            if i % 10 == 0 or i == len(jobs) - 1:
                log(f"[mtf-dataset] {i + 1}/{len(jobs)} {headers[-1]['name']} ({time.time() - t0:.0f}s)")
    ok = [h for h in headers if not h["skip"]]
    total = sum(h["n"] for h in ok)
    names = fm.feature_names(TFS, GROUPS)
    K = len(KEYS)
    cdir.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap
    X = mm(cdir / "X.npy", "w+", np.float16, (total, len(names)))
    t = mm(cdir / "t.npy", "w+", np.int64, (total,))
    sym = mm(cdir / "sym.npy", "w+", np.int16, (total,))
    lab = mm(cdir / "lab.npy", "w+", np.int8, (K, total))
    pnl = mm(cdir / "pnl.npy", "w+", np.float16, (K, total))
    off = mm(cdir / "off.npy", "w+", np.uint16, (K, total))
    r = 0
    for si, h in enumerate(ok):
        with np.load(part_dir / f"{h['name']}.npz") as z:
            n = h["n"]
            X[r:r + n] = z["X"]
            t[r:r + n] = z["t"]
            sym[r:r + n] = si
            lab[:, r:r + n] = z["lab"]
            pnl[:, r:r + n] = z["pnl"]
            off[:, r:r + n] = z["off"]
        r += n
    for a in (X, t, sym, lab, pnl, off):
        a.flush()
    meta = dict(symbols=[h["name"] for h in ok], rows=int(total), keys=KEYS, feature_names=names,
                timeframes=["5m", *TFS], groups=list(GROUPS), exit_cfgs={e: cfg_of(e) for e in EXITS},
                horizon=horizon, decision_every_bars=every, warmup_bars_5m=fm.warmup_bars_5m(TFS, GROUPS),
                history_bars_5m=fm.history_bars_5m(TFS, GROUPS),
                symbols_without_metrics=[h["name"] for h in ok if h.get("no_metrics")],
                skipped=[h["name"] for h in headers if h["skip"]], built_at=int(time.time()))
    (cdir / "meta.json").write_text(json.dumps(meta, default=str))
    for p in part_dir.glob("*.npz"):
        p.unlink()
    part_dir.rmdir()
    log(f"[mtf-dataset] {total} rows x {len(names)} features, {len(ok)} symbols -> {cdir} ({time.time() - t0:.0f}s)")
    return load_cached(cdir)


def synthetic_dataset(n_symbols=4, n=14000, seed=0, horizon=300, every=DECISION_EVERY) -> MtfDataset:
    """Small in-memory-built dataset from ml/synth frames (smoke tests; short horizon so few rows are lost)."""
    import tempfile
    from synth import make_market, make_symbol
    cdir = Path(tempfile.mkdtemp(prefix="tb_synth_mtf_"))
    btc, _, _ = make_symbol(n, seed=1000, start_price=30000.0)
    mk = make_market(n)
    warm = fm.warmup_bars_5m(TFS, GROUPS)
    parts, syms = [], []
    for i in range(n_symbols):
        kl, m, f = make_symbol(n, seed=seed + i, market=mk)
        t_all = kl["open_time"].to_numpy("int64")
        rows = np.nonzero(fm.decision_mask(t_all, every) & (np.arange(n) >= warm))[0]
        rows = rows[rows < n - horizon]
        X, _ = fm.build_mtf_features(kl, m, f, btc, TFS, GROUPS, rows=rows)
        lab, pnl, off = [], [], []
        for e in EXITS:
            cfg = cfg_of(e)
            for s in SIDES:
                sim = simulate_trades(kl, s, cfg, horizon, rows=rows)
                lab.append(np.where(np.isnan(sim["label"][rows]), -1, sim["label"][rows]).astype("int8"))
                pnl.append(sim["pnl_usd"][rows].astype("float16"))
                off.append(sim["exit_off"][rows].astype("uint16"))
        parts.append((X.astype("float16"), t_all[rows], np.stack(lab), np.stack(pnl), np.stack(off)))
        syms.append(f"SYN{i}USDT")
    X = np.concatenate([p[0] for p in parts])
    t = np.concatenate([p[1] for p in parts])
    sym = np.concatenate([np.full(len(p[1]), i, "int16") for i, p in enumerate(parts)])
    lab = np.concatenate([p[2] for p in parts], axis=1)
    pnl = np.concatenate([p[3] for p in parts], axis=1)
    off = np.concatenate([p[4] for p in parts], axis=1)
    meta = dict(symbols=syms, rows=int(len(t)), keys=KEYS, feature_names=fm.feature_names(TFS, GROUPS),
                timeframes=["5m", *TFS], groups=list(GROUPS), exit_cfgs={e: cfg_of(e) for e in EXITS},
                horizon=horizon, decision_every_bars=every, warmup_bars_5m=warm,
                history_bars_5m=fm.history_bars_5m(TFS, GROUPS), synthetic=True)
    return MtfDataset(syms, X, t, sym, lab, pnl, off, meta)
