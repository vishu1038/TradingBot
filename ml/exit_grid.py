"""Exit grid with a dollar floor (every TP and every SL worth >= min_usd) for the V3 decision-cadence dataset.

The cached ``mtf-*`` dataset (features + the two legacy exits ``pct36`` / ``atr1h32``) is re-used as is; this module only
re-labels the SAME decision rows with a small grid of exits (TP:SL ratios chosen by the data) and stores the new
labels next to the cache (``<cache>/exits-<hash>/``).

Exit semantics (the same ``tb.labels`` template: entry at open[i+1] + slippage, both barriers in one candle -> SL first,
4-day horizon, SL = stop-market with extra slippage, taker fees both ways, notional = min(slice notional, sl_usd / SL
fraction), i.e. the SL never loses more than ``sl_usd``):

    mode "atr":  tp_dist = max(tp_atr * ATR14(1h), floor)     sl_dist = max(sl_atr * ATR14(1h), floor)
    mode "pct":  tp_dist = max(tp_pct % , floor)              sl_dist = max(sl_pct % , floor)

``floor`` = (min_usd + round-trip fee on the slice notional) / slice notional  = (2.0 + 0.16) / 160 = 1.35 % of price, so
a winning trade nets >= $2 after fees and a losing trade loses > $2.  TP and SL are floored independently (the natural
reading of min_tp_usd / min_sl_usd): for a low-ATR coin both land on the floor and the realised ratio is 1:1.

    PYTHONPATH=backend:ml nice -n 10 .venv-ml/bin/python ml/exit_grid.py build [--workers 4]
    PYTHONPATH=backend:ml .venv-ml/bin/python ml/exit_grid.py check        # parity with tb.labels when floor = 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

import dataset as D  # noqa: E402
import mtf_dataset as MD  # noqa: E402
from tb import features as ft  # noqa: E402
from tb.labels import DEFAULT_EXIT_CFG, exit_atr, full_cfg, simulate_trades  # noqa: E402

SIDES = MD.SIDES
MIN_USD = 2.0
SLICE_NOTIONAL = DEFAULT_EXIT_CFG["slice_margin_usd"] * DEFAULT_EXIT_CFG["leverage"]        # 160
RT_FEE_USD = 2 * DEFAULT_EXIT_CFG["fee_taker"] * SLICE_NOTIONAL                              # 0.16
FLOOR_FRAC = (MIN_USD + RT_FEE_USD) / SLICE_NOTIONAL                                         # 0.0135
RT_COST_FRAC = 2 * DEFAULT_EXIT_CFG["fee_taker"] + 2 * DEFAULT_EXIT_CFG["slippage_pct"] + DEFAULT_EXIT_CFG["stop_slippage_pct"]
SL_USD = DEFAULT_EXIT_CFG["sl_usd"]                                                          # 10 (max loss of the notional rule)

# name -> (mode, tp, sl): atr = 1h-ATR multiples, pct = percent of price.  8 variants = the whole grid.
GRID = {
    "atr_1to2": ("atr", 1.5, 3.0),        # TP:SL = 1:2   (the user's manual style, volatility scaled)
    "atr_1to1": ("atr", 2.0, 2.0),        # 1:1
    "atr_3to2": ("atr", 3.0, 2.0),      # 1.5:1 (the old atr1h32)
    "atr_2to1": ("atr", 3.0, 1.5),        # 2:1
    "atr_3to1": ("atr", 3.0, 1.0),        # 3:1
    "pct_1to2": ("pct", 3.0, 6.0),        # 1:2   TP +3 % / SL -6 %  (the user's own numbers = the old pct36)
    "pct_1to1": ("pct", 3.0, 3.0),        # 1:1
    "pct_2to1": ("pct", 6.0, 3.0),        # 2:1
}
HORIZON = MD.HORIZON


def exit_cfg_of(name: str) -> dict:
    """The artifact ``exit_cfg`` (engine keys + the two $ floors).  tb.labels / tb.exits ignore ``min_*_usd``."""
    mode, tp, sl = GRID[name]
    cfg = dict(full_cfg({"exit_mode": mode, "tp_atr": tp if mode == "atr" else 1.5, "sl_atr": sl if mode == "atr" else 1.0,
                         "tp_pct": tp if mode == "pct" else 3.0, "sl_pct": sl if mode == "pct" else 6.0,
                         "atr_tf": "1h" if mode == "atr" else "5m"}))
    cfg.update(min_tp_usd=MIN_USD, min_sl_usd=MIN_USD, horizon_bars=HORIZON,
               floor_pct=round(100 * FLOOR_FRAC, 4))
    return cfg


def simulate(kl, side: str, name: str, rows: np.ndarray, atr: np.ndarray | None, floor: float = FLOOR_FRAC,
             horizon: int = HORIZON) -> dict[str, np.ndarray]:
    """Resolve the hypothetical trade of exit ``name`` at every row in ``rows`` (indices of signal bars).
    Returns label (1 TP / 0 SL-or-timeout / NaN n.a.), pnl_usd, exit_off, tp_frac, sl_frac (arrays over ``rows``)."""
    mode, tp_v, sl_v = GRID[name]
    cfg = full_cfg({})
    o, hi, lo, cl = (kl[c].to_numpy("float64") for c in ("open", "high", "low", "close"))
    n, H = len(kl), int(horizon)
    sgn = 1.0 if side == "long" else -1.0
    sig = np.asarray(rows, dtype="int64")
    ns = len(sig)
    nan = np.full(ns, np.nan, dtype="float32")
    ok0 = (sig >= 0) & (sig < n - H)
    ss = np.where(ok0, sig, 0)
    slip, sslip = cfg["slippage_pct"], cfg["stop_slippage_pct"]
    entry = o[ss + 1] * (1.0 + sgn * slip)
    if mode == "atr":
        a = atr[ss]
        tp_d = np.maximum(tp_v * a, floor * entry)
        sl_d = np.maximum(sl_v * a, floor * entry)
        ok = ok0 & np.isfinite(entry) & (entry > 0) & np.isfinite(a) & (a > 0)
    else:
        tp_d = np.maximum(tp_v / 100.0 * entry, floor * entry)
        sl_d = np.maximum(sl_v / 100.0 * entry, floor * entry)
        ok = ok0 & np.isfinite(entry) & (entry > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        notl = np.minimum(SLICE_NOTIONAL, SL_USD / (sl_d / entry))
    tp = entry + sgn * tp_d
    sl = entry - sgn * sl_d
    ok &= sl > 0
    tp_k = np.zeros(ns, dtype="int32")
    sl_k = np.zeros(ns, dtype="int32")
    active = np.nonzero(ok)[0]
    for k in range(1, H + 1):
        if len(active) == 0:
            break
        hk, lk = hi[ss[active] + k], lo[ss[active] + k]
        if sgn > 0:
            t_hit, s_hit = hk >= tp[active], lk <= sl[active]
        else:
            t_hit, s_hit = lk <= tp[active], hk >= sl[active]
        done = t_hit | s_hit
        if done.any():
            d = active[done]
            sl_k[d[s_hit[done]]] = k
            tp_k[d[t_hit[done] & ~s_hit[done]]] = k
            active = active[~done]
    rtp, rsl = tp_k > 0, sl_k > 0
    k_exit = np.where(rtp, tp_k, np.where(rsl, sl_k, H)).astype("int32")
    to_close = cl[ss + H]
    sl_open = o[ss + np.maximum(sl_k, 1)]
    sl_base = np.minimum(sl, sl_open) if sgn > 0 else np.maximum(sl, sl_open)
    exit_px = np.where(rtp, tp, np.where(rsl, sl_base * (1.0 - sgn * (slip + sslip)), to_close * (1.0 - sgn * slip)))
    with np.errstate(divide="ignore", invalid="ignore"):
        qty = notl / entry
        net = sgn * qty * (exit_px - entry) - cfg["fee_taker"] * qty * (entry + exit_px)
    label = np.where(ok, np.where(rtp, 1.0, 0.0), np.nan).astype("float32")
    return dict(label=label, pnl_usd=np.where(ok, net, np.nan).astype("float32"),
                exit_off=np.where(ok, k_exit, 0).astype("int32"),
                tp_frac=np.where(ok, tp_d / entry, np.nan).astype("float32"),
                sl_frac=np.where(ok, sl_d / entry, np.nan).astype("float32"),
                notional=np.where(ok, notl, np.nan).astype("float32"))


# --------------------------------------------------------------------------------------------
def cache_dir(base: Path) -> Path:
    h = hashlib.sha1(json.dumps([GRID, FLOOR_FRAC, SL_USD, HORIZON, MD.SIDES], sort_keys=True).encode()).hexdigest()[:10]
    return base / f"exits-{h}"


def _one(job):
    sym_i, sym, root, ts, out = job
    kl, _, _ = D.load_symbol(sym, Path(root))
    kl = ft.regularize(kl)
    t_all = kl["open_time"].to_numpy("int64")
    rows = np.searchsorted(t_all, ts)
    assert (t_all[rows] == ts).all(), sym
    atr = exit_atr(kl, {"atr_tf": "1h"})
    K = len(GRID) * len(SIDES)
    lab = np.full((K, len(rows)), -1, "int8")
    pnl = np.full((K, len(rows)), np.nan, "float16")
    off = np.zeros((K, len(rows)), "uint16")
    slf = np.full((len(GRID), len(rows)), np.nan, "float16")
    j = 0
    for ei, name in enumerate(GRID):
        for side in SIDES:
            r = simulate(kl, side, name, rows, atr)
            lab[j] = np.where(np.isnan(r["label"]), -1, r["label"]).astype("int8")
            pnl[j] = r["pnl_usd"].astype("float16")
            off[j] = r["exit_off"].astype("uint16")
            slf[ei] = r["sl_frac"].astype("float16")
            j += 1
    np.savez(out, lab=lab, pnl=pnl, off=off, sl=slf)
    return sym_i, sym, int(len(rows))


def build(base: Path, workers: int = 4, log=print) -> Path:
    meta = json.loads((base / "meta.json").read_text())
    syms = meta["symbols"]
    t = np.load(base / "t.npy", mmap_mode="r")
    sy = np.load(base / "sym.npy", mmap_mode="r")
    cdir = cache_dir(base)
    if (cdir / "meta.json").exists():
        log(f"[exit-grid] cache hit {cdir}")
        return cdir
    parts = cdir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    N = len(t)
    # rows of one symbol are a contiguous block (sym is non-decreasing)
    bounds = np.searchsorted(np.asarray(sy), np.arange(len(syms) + 1))
    root = str(D.archive_root())
    jobs = [(i, s, root, np.asarray(t[bounds[i]:bounds[i + 1]]), str(parts / f"{s}.npz")) for i, s in enumerate(syms)]
    t0 = time.time()
    import concurrent.futures as cf
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for k, (i, s, n) in enumerate(ex.map(_one, jobs)):
            if k % 10 == 0 or k == len(jobs) - 1:
                log(f"[exit-grid] {k + 1}/{len(jobs)} {s} n={n} ({time.time() - t0:.0f}s)")
    K, E = len(GRID) * len(SIDES), len(GRID)
    mm = np.lib.format.open_memmap
    lab = mm(cdir / "lab.npy", "w+", np.int8, (K, N))
    pnl = mm(cdir / "pnl.npy", "w+", np.float16, (K, N))
    off = mm(cdir / "off.npy", "w+", np.uint16, (K, N))
    slf = mm(cdir / "sl.npy", "w+", np.float16, (E, N))
    for i, s in enumerate(syms):
        a, b = bounds[i], bounds[i + 1]
        with np.load(parts / f"{s}.npz") as z:
            lab[:, a:b], pnl[:, a:b], off[:, a:b], slf[:, a:b] = z["lab"], z["pnl"], z["off"], z["sl"]
    for x in (lab, pnl, off, slf):
        x.flush()
    keys = [f"{e}_{s}" for e in GRID for s in SIDES]
    (cdir / "meta.json").write_text(json.dumps(dict(keys=keys, grid={k: list(v) for k, v in GRID.items()},
                                                    floor_frac=FLOOR_FRAC, rt_cost_frac=RT_COST_FRAC, horizon=HORIZON,
                                                    built_at=int(time.time()))))
    for p in parts.glob("*.npz"):
        p.unlink()
    parts.rmdir()
    log(f"[exit-grid] done {cdir} ({time.time() - t0:.0f}s)")
    return cdir


def load_dataset(base: Path | None = None, log=print):
    """The cached MtfDataset with its labels replaced by the grid labels (+ ``ds.sl`` stop fraction per exit)."""
    base = Path(base) if base else _default_base()
    cdir = build(base, log=log)
    ds = MD.load_cached(base)
    gm = json.loads((cdir / "meta.json").read_text())
    ld = lambda n: np.load(cdir / f"{n}.npy", mmap_mode="r")  # noqa: E731
    meta = dict(ds.meta)
    meta.update(keys=gm["keys"], exit_cfgs={k: exit_cfg_of(k) for k in GRID}, horizon=HORIZON,
                grid={k: list(v) for k, v in GRID.items()}, floor_frac=gm["floor_frac"], rt_cost_frac=gm["rt_cost_frac"])
    new = MD.MtfDataset(ds.symbols, ds.X, ds.t, ds.sym, ld("lab"), ld("pnl"), ld("off"), meta)
    new.sl = ld("sl")
    return new


def _default_base() -> Path:
    cands = sorted((D.var_dir() / "ml_cache").glob("mtf-*/meta.json"), key=lambda p: p.stat().st_mtime)
    cands = [p for p in cands if json.loads(p.read_text()).get("rows", 0) > 1_000_000]
    return cands[-1].parent


def check():
    """floor = 0 must reproduce tb.labels.simulate_trades (pct36 and atr1h32) bit for bit on real data."""
    for sym in ("SOLUSDT", "DOGEUSDT"):
        kl, _, _ = D.load_symbol(sym)
        kl = ft.regularize(kl)
        rows = np.arange(10000, len(kl) - HORIZON, 12 * 7)
        atr = exit_atr(kl, {"atr_tf": "1h"})
        for name, ref_cfg in (("pct_1to2", {"exit_mode": "pct", "tp_pct": 3.0, "sl_pct": 6.0}),
                              ("atr_3to2", {"exit_mode": "atr", "atr_tf": "1h", "tp_atr": 3.0, "sl_atr": 2.0})):
            for side in SIDES:
                ref = simulate_trades(kl, side, ref_cfg, HORIZON, rows=rows)
                got = simulate(kl, side, name, rows, atr, floor=0.0)
                same = np.array_equal(np.nan_to_num(ref["label"][rows], nan=-1), np.nan_to_num(got["label"], nan=-1))
                dp = np.nanmax(np.abs(ref["pnl_usd"][rows] - got["pnl_usd"]))
                print(f"{sym} {name} {side}: labels equal={same} max|dpnl|={dp:.2e} off equal="
                      f"{np.array_equal(ref['exit_off'][rows], got['exit_off'])}")
                assert same and dp < 1e-3
        for name in GRID:
            r = simulate(kl, "long", name, rows, atr)
            ok = np.isfinite(r["sl_frac"])
            print(f"  {sym} {name}: min tp%={100 * np.nanmin(r['tp_frac']):.2f} min sl%={100 * np.nanmin(r['sl_frac']):.2f} "
                  f"win={np.nanmean(r['label']):.3f} exp={np.nanmean(r['pnl_usd']):.2f} "
                  f"min notional={np.nanmin(r['notional']):.0f}")
    print("floor in USD: gross TP at the floor on 160 notional =", FLOOR_FRAC * SLICE_NOTIONAL)
    print("check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    else:
        build(_default_base(), a.workers)
