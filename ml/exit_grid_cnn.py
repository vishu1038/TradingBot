"""5m CNN on the dollar-floored exit grid (companion of ml/exit_grid_wf.py, which does the GBM).

Re-labels the cached 5m CNN dataset (default: the 25-symbol one used by the retrain) with the chosen exits of
ml/exit_grid.py (1h-ATR multiples or % with the $2-net floor on every TP and SL), then runs the unchanged
ml/walkforward.py machinery (purged expanding folds, validation-only Platt calibration + thresholds, random-entry and
ATR%-regime baselines, locked 60-day holdout evaluated ONCE) on mps.

Differences to walkforward.py (all deliberate):
  * threshold grid = top-fraction of the validation score distribution, the SAME grid for long and short (the old grid
    0.3..0.9 on an absolute probability made long = "trade everything" and short = "never trade");
  * qualification uses the Bonferroni z over (exit variants x sides), min(z, ...) vs the random baseline, and must beat
    the ATR rule; the holdout must confirm (exp > 0, > random, > ATR rule, n >= 20) or the model is PAPER TEST ONLY;
  * exports carry the v3 adaptive ``entry_cfg`` and ``sides_enabled``.

    export PYTHONPATH=backend:ml
    nohup nice -n 10 .venv-ml/bin/python -u ml/exit_grid_cnn.py run --exits atr_2to1,pct_1to2 \\
        > var/ml_logs/exit_grid_cnn.log 2>&1 &
    .venv-ml/bin/python ml/exit_grid_cnn.py smoke        # synthetic end-to-end check (mps)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

import dataset as D  # noqa: E402
import evaluate as E  # noqa: E402
import exit_grid as EG  # noqa: E402
import walkforward as WF  # noqa: E402
from splits import iso, make_splits  # noqa: E402
from tb import features as ft  # noqa: E402

SIDES = D.SIDES
ROOT = Path(__file__).resolve().parents[1]
REF_MODEL = ROOT / "models" / "v3_gbm_pct36_kal_20261019_0607.json"
SYMS25 = ("1000BONKUSDT,1000PEPEUSDT,ADAUSDT,APTUSDT,ARBUSDT,AVAXUSDT,BNBUSDT,BTCUSDT,DOGEUSDT,ENAUSDT,ETHUSDT,FETUSDT,"
          "INJUSDT,LINKUSDT,NEARUSDT,OPUSDT,ORDIUSDT,SEIUSDT,SOLUSDT,SUIUSDT,TAOUSDT,TIAUSDT,WIFUSDT,WLDUSDT,XRPUSDT")
FRACTIONS = (0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.20)
MIN_TRADES = 100


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------------------------
# labels for the CNN rows (every 5m bar)
# --------------------------------------------------------------------------------------------
def _one(job):
    sym, root, ts, exits, out, horizon = job
    kl, _, _ = D.load_symbol(sym, Path(root))
    kl = ft.regularize(kl)
    t_all = kl["open_time"].to_numpy("int64")
    rows = np.searchsorted(t_all, ts)
    assert (t_all[rows] == ts).all(), sym
    from tb.labels import exit_atr
    atr = exit_atr(kl, {"atr_tf": "1h"})
    K = len(exits) * 2
    lab = np.full((K, len(rows)), -1, "int8")
    pnl = np.full((K, len(rows)), np.nan, "float16")
    off = np.zeros((K, len(rows)), "uint16")
    j = 0
    for e in exits:
        for side in SIDES:
            r = EG.simulate(kl, side, e, rows, atr, horizon=horizon)
            lab[j] = np.where(np.isnan(r["label"]), -1, r["label"]).astype("int8")
            pnl[j] = r["pnl_usd"].astype("float16")
            off[j] = r["exit_off"].astype("uint16")
            j += 1
    np.savez(out, lab=lab, pnl=pnl, off=off)
    return sym, int(len(rows))


def relabel(ds, exits: list[str], cache_root: Path | None, workers: int, root: Path, log_=log):
    """Dataset with the label columns replaced by the grid exits (``mode`` names = exit names)."""
    horizon = int(ds.meta.get("horizon", D.HORIZON))
    h = hashlib.sha1(json.dumps([exits, EG.GRID, EG.FLOOR_FRAC, horizon, ds.symbols, len(ds)], sort_keys=True).encode()).hexdigest()[:10]
    cdir = (cache_root or (D.var_dir() / "ml_cache")) / f"exits-cnn-{h}"
    K, N = len(exits) * 2, len(ds)
    if not (cdir / "meta.json").exists():
        cdir.mkdir(parents=True, exist_ok=True)
        parts = cdir / "parts"
        parts.mkdir(exist_ok=True)
        sy = np.asarray(ds.sym)
        bounds = np.searchsorted(sy, np.arange(len(ds.symbols) + 1))
        jobs = [(s, str(root), np.asarray(ds.t[bounds[i]:bounds[i + 1]]), exits, str(parts / f"{s}.npz"), horizon)
                for i, s in enumerate(ds.symbols)]
        t0 = time.time()
        if workers > 1:
            import concurrent.futures as cf
            import multiprocessing as mp
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
                for k, (s, n) in enumerate(ex.map(_one, jobs)):
                    log_(f"[relabel] {k + 1}/{len(jobs)} {s} n={n} ({time.time() - t0:.0f}s)")
        else:
            for k, j in enumerate(jobs):
                s, n = _one(j)
                log_(f"[relabel] {k + 1}/{len(jobs)} {s} n={n} ({time.time() - t0:.0f}s)")
        mm = np.lib.format.open_memmap
        lab = mm(cdir / "lab.npy", "w+", np.int8, (K, N))
        pnl = mm(cdir / "pnl.npy", "w+", np.float16, (K, N))
        off = mm(cdir / "off.npy", "w+", np.uint16, (K, N))
        for i, s in enumerate(ds.symbols):
            a, b = bounds[i], bounds[i + 1]
            with np.load(parts / f"{s}.npz") as z:
                lab[:, a:b], pnl[:, a:b], off[:, a:b] = z["lab"], z["pnl"], z["off"]
        for x in (lab, pnl, off):
            x.flush()
        for p in parts.glob("*.npz"):
            p.unlink()
        parts.rmdir()
        (cdir / "meta.json").write_text(json.dumps(dict(exits=exits, built_at=int(time.time()))))
    ld = lambda n: np.load(cdir / f"{n}.npy", mmap_mode="r")  # noqa: E731
    keys = [f"{e}_{s}" for e in exits for s in SIDES]
    meta = dict(ds.meta)
    meta.update(keys=keys, exit_cfgs={e: EG.exit_cfg_of(e) for e in exits}, horizon=horizon)
    return dataclasses.replace(ds, lab=D._Cols(ld("lab"), keys, "lab"), pnl=D._Cols(ld("pnl"), keys, "pnl"),
                               off=D._Cols(ld("off"), keys, "off"), meta=meta)


# --------------------------------------------------------------------------------------------
# patches: top-fraction threshold grid (same for both sides), population break-even for the report
# --------------------------------------------------------------------------------------------
_orig_trading_table = E.trading_table


def _table(p, y, pnl, t, sym, off, be, thresholds=None, **kw):
    if thresholds is None:
        p = np.asarray(p, "float64")
        thresholds = tuple(sorted(set(float(np.quantile(p, 1.0 - f)) for f in FRACTIONS))) if len(p) else E.THRESHOLDS
    return _orig_trading_table(p, y, pnl, t, sym, off, be, thresholds=thresholds, **kw)


def _be(ds, mode, rows):
    vals = []
    for s in SIDES:
        b = WF.population_breakeven(ds, f"{mode}_{s}", rows)
        if b == b:
            vals.append(b)
    return float(np.mean(vals)) if vals else 0.5


def install_patches():
    E.trading_table = _table
    WF.analytic_breakeven = _be


def bonf(n_tests: int) -> float:
    return float(statistics.NormalDist().inv_cdf(1.0 - 0.05 / max(n_tests, 1)))


def qualify(pooled: dict, rule: dict, z_min: float) -> tuple[bool, str]:
    c = pooled["chosen"]
    why = []
    if c["n"] < MIN_TRADES:
        why.append(f"n={c['n']}<{MIN_TRADES}")
    if c["z_vs_baseline"] is None or c["z_vs_baseline"] < z_min:
        why.append(f"z={c['z_vs_baseline']}<{z_min:.2f}")
    if c["expectancy"] is None or c["expectancy"] <= 0:
        why.append("expectancy<=0")
    if rule["expectancy"] is not None and c["expectancy"] is not None and c["expectancy"] <= rule["expectancy"]:
        why.append("not better than the ATR rule")
    return (not why), "; ".join(why)


def joint_z(res: dict) -> float | None:
    num, var = 0.0, 0.0
    for s in SIDES:
        c = res[f"pooled_{s}"]["chosen"]
        if c["n"] > 1 and c["se"] and np.isfinite(c["baseline_expectancy"]):
            num += c["n"] * (c["expectancy"] - c["baseline_expectancy"])
            var += (c["n"] * c["se"]) ** 2
    return float(num / np.sqrt(var)) if var > 0 else None


# --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "wf", "smoke"])
    ap.add_argument("--exits", default="atr_2to1")
    ap.add_argument("--symbols", default=SYMS25)
    ap.add_argument("--n-variants", type=int, default=len(EG.GRID), help="exit variants the choice was made from (Bonferroni)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--val-days", type=int, default=14)
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--min-train-days", type=int, default=45)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--tag", default="gridcnn")
    a = ap.parse_args(argv)

    import trainer as T
    T.require_mps()
    install_patches()
    var = D.var_dir()
    out_dir = var / "ml" / "exit_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(a.models_dir or ROOT / "models")
    exits = [e for e in a.exits.split(",") if e]
    for e in exits:
        assert e in EG.GRID, e

    if a.cmd == "smoke":
        ds = D.synthetic_dataset(n_symbols=3, n=20000, planted_edge=0.5)
        horizon = int(ds.meta["horizon"])
        # synthetic frames are not in the archive: reuse the original modes under the grid names (plumbing check only)
        keys = [f"{e}_{s}" for e in exits for s in SIDES]
        src = [f"{'fixed'}_{s}" for _ in exits for s in SIDES]
        meta = dict(ds.meta, keys=keys, exit_cfgs={e: EG.exit_cfg_of(e) for e in exits})
        ds = dataclasses.replace(ds, lab=D._Cols(np.stack([ds.lab.arr[ds.lab.names.index(k)] for k in src]), keys, "lab"),
                                 pnl=D._Cols(np.stack([ds.pnl.arr[ds.pnl.names.index(k)] for k in src]), keys, "pnl"),
                                 off=D._Cols(np.stack([ds.off.arr[ds.off.names.index(k)] for k in src]), keys, "off"), meta=meta)
        a.folds, a.test_days, a.val_days, a.holdout_days, a.min_train_days, a.steps = 1, 6, 4, 6, 8, 120
        models_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "exit_grid_cnn_smoke_models"
        out_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "exit_grid_cnn_smoke_out"
        out_dir.mkdir(exist_ok=True)
    else:
        syms = [s for s in a.symbols.split(",") if s]
        base = D.build_dataset(syms, None, use_cache=True, log=log)
        ds = relabel(base, exits, None, a.workers, D.archive_root())
    log(f"dataset: {len(ds)} rows, {len(ds.symbols)} symbols, exits {exits}")
    horizon = int(ds.meta["horizon"])
    splits = make_splits(np.asarray(ds.t[::97]), a.folds, a.test_days, a.val_days, a.holdout_days,
                         purge_bars=horizon + ft.WINDOW, min_train_days=a.min_train_days)
    log("splits: " + json.dumps({"folds": [{k: [iso(x) for x in v] if isinstance(v, tuple) else v for k, v in f.items()}
                                           for f in splits["folds"]],
                                 "final": {k: [iso(x) for x in v] for k, v in splits["final"].items()}}))
    cfg = T.TrainCfg(arch="cnn", steps=a.steps, batch=a.batch, lr=a.lr, patience=a.patience, eval_every=min(a.eval_every, a.steps),
                     pos_weight="auto", seed=0, aux=False, aux_w=0.3)
    results = {}
    t0 = time.time()
    for e in exits:
        f_ = out_dir / f"wf_cnn_{e}_{a.tag}.json"
        if f_.exists() and a.cmd != "smoke":
            results[e] = json.loads(f_.read_text())
            log(f"=== {e}: wf result exists, skipping")
            continue
        res, _ = WF.run_walkforward(ds, "cnn", e, splits, cfg, a.seeds, log)
        results[e] = res
        f_.write_text(json.dumps(res, indent=1, default=float))
        for s in SIDES:
            c = res[f"pooled_{s}"]["chosen"]
            r = res[f"pooled_{s}"]["atr_rule"]
            log(f"### cnn/{e} {s}: n={c['n']} win={c['win_rate']} exp={c['expectancy']} random={c['baseline_expectancy']} "
                f"z={c['z_vs_baseline']} ATR-rule exp={r['expectancy']} [{time.time() - t0:.0f}s]")
    if a.cmd == "wf":
        return
    nt = a.n_variants * 2 + len(exits) * 0           # exit choice was already counted in the n_variants x sides tests
    nt = max(nt, len(exits) * 2)
    z_min = bonf(nt)
    ranked = sorted(results, key=lambda e: -(joint_z(results[e]) if joint_z(results[e]) is not None else -9))
    log(f"Bonferroni z={z_min:.2f} over {nt} tests; joint-z ranking: {[(e, joint_z(results[e])) for e in ranked]}")
    summary = []
    for e in ranked:
        wf = results[e]
        enabled = []
        for s in SIDES:
            q, why = qualify(wf[f"pooled_{s}"], wf[f"pooled_{s}"]["atr_rule"], z_min)
            log(f"  WF {e} {s}: qualifies={q} ({why})")
            if q:
                enabled.append(s)
        nets, info, hres, harrs = WF.run_final(ds, "cnn", e, splits, cfg, a.seeds, log)
        hold = json.loads(json.dumps(hres, default=float))
        edge = []
        for s in enabled:
            h = hold[s]["chosen"]
            rr = hold[s]["atr_rule"]
            if (h["n"] >= 20 and h["expectancy"] and h["expectancy"] > 0 and h["baseline_expectancy"] is not None
                    and h["expectancy"] > h["baseline_expectancy"] and (rr["expectancy"] is None or h["expectancy"] > rr["expectancy"])):
                edge.append(s)
        txt = "; ".join(
            f"{s}: WF n={wf[f'pooled_{s}']['chosen']['n']} exp={wf[f'pooled_{s}']['chosen']['expectancy']:.2f} (random "
            f"{wf[f'pooled_{s}']['chosen']['baseline_expectancy']:.2f}, z={wf[f'pooled_{s}']['chosen']['z_vs_baseline']}), holdout "
            f"n={hold[s]['chosen']['n']} exp={hold[s]['chosen']['expectancy']} (random {hold[s]['chosen']['baseline_expectancy']}, "
            f"ATR rule {hold[s]['atr_rule']['expectancy']})" for s in SIDES
            if wf[f"pooled_{s}"]["chosen"]["n"] and hold[s]["chosen"]["n"] and hold[s]["chosen"]["expectancy"] is not None)
        tail = f"5m CNN, exit {e} (chosen from {a.n_variants} variants, {nt} tests, Bonferroni z={z_min:.2f}); symmetric top-fraction thresholds. {txt}"
        if edge:
            note = f"Beat random and the ATR rule out of sample for {edge} (WF Bonferroni + one-shot holdout); other side disabled. {tail}"
            sides = edge
        else:
            why = "no side beat both baselines on walk-forward" if not enabled else \
                f"beat both baselines in walk-forward for {enabled} but NOT on the holdout"
            note = f"PAPER TEST ONLY - no edge ({why}). {tail}"
            sides = list(SIDES)
        name = WF.export_model(ds, "cnn", e, info, nets, WF.strip_equity(wf), hold, models_dir)
        mp = models_dir / f"{name}.json"
        meta = json.loads(mp.read_text())
        meta.update(note=note, sides_enabled=sides, entry_cfg=json.loads(REF_MODEL.read_text())["entry_cfg"], variant=f"cnn_{e}")
        meta["thresholds"] = {s: (meta["thresholds"][s] if s in sides else 1.01) for s in SIDES}
        mp.write_text(json.dumps(meta, indent=1, default=float))
        (out_dir / f"final_{name}.json").write_text(json.dumps(dict(holdout=hres, note=note, edge_sides=edge), indent=1, default=float))
        summary.append(dict(name=name, exit=e, wf_qualified=enabled, edge_sides=edge, note=note[:200]))
        log(f"exported {name}: edge_sides={edge}")
        T.free_gpu()
    (out_dir / "exported_cnn.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
