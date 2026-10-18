"""Walk-forward training / evaluation / export driver.

    PYTHONPATH=backend nice -n 10 .venv-ml/bin/python ml/walkforward.py run --archs cnn,gru --modes fixed,atr
    ... --synthetic            # smoke test on generated data (no archive needed)

Stages (each also available alone: wf / final):

  1. build the dataset (features + labels for fixed and atr exits) from data/archive
  2. walk-forward: K expanding-window folds; each fold = train | purge | val | purge | test.
     Per fold: early stopping on val; Platt calibration fitted on val; per-side thresholds chosen on
     val ONLY; trading + calibration + drift metrics reported on the (untouched) test block.
  3. choose ONE (arch, mode) by pooled out-of-sample expectancy, retrain it on all dev data
     (train | purge | val), evaluate ONCE on the locked holdout (last --holdout-days), export
     models/{name}.npz/.json (status "challenger") and var/lab/{id}.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

import dataset as D  # noqa: E402
import evaluate as E  # noqa: E402
import trainer as T  # noqa: E402
from splits import BAR_MS, DAY_MS, iso, make_splits, rows_in  # noqa: E402,F401
from tb import features as ft  # noqa: E402
from tb import model_np as mnp  # noqa: E402
from tb.labels import OUT_NA, breakeven_from_cfg, breakeven_win_rate, simulate_trades  # noqa: E402,F401

SIDES = D.SIDES


# --------------------------------------------------------------------------------------------
# one train / calibrate / evaluate cycle
# --------------------------------------------------------------------------------------------
def fit_split(ds, arch, mode, tr_rng, va_rng, cfg: T.TrainCfg, log):
    ok = ds.sample_ok
    any_valid = ok & (np.isfinite(ds.lab[f"{mode}_long"]) | np.isfinite(ds.lab[f"{mode}_short"]))
    tr = rows_in(ds, tr_rng, any_valid)
    va = rows_in(ds, va_rng, any_valid)
    sub = tr[np.linspace(0, len(tr) - 1, min(len(tr), 400_000)).astype("int64")]
    norm = ft.robust_norm_stats(ds.X[sub].astype("float32"))
    net, info = T.train_model(ds, mode, tr, va, norm, cfg, log)
    # calibration + thresholds from VAL only
    lg_val = T.predict_logits(net, ds.X, va, norm)
    cal, thr, val_rows = {}, {}, {}
    for si, side in enumerate(SIDES):
        key = f"{mode}_{side}"
        m = np.isfinite(ds.lab[key][va])
        cal[side] = mnp.fit_calibration(lg_val[m, si], ds.lab[key][va][m], "platt")
        p = mnp.apply_calibration(cal[side], lg_val[m, si])
        sel = va[m]
        be = analytic_breakeven(ds, mode, sel)
        rows = E.trading_table(p, ds.lab[key][sel], ds.pnl[key][sel], ds.t[sel], ds.sym[sel], ds.off[key][sel], be,
                               n_random=2)
        thr[side], pos = E.choose_threshold(rows)
        val_rows[side] = dict(threshold=thr[side], expectancy_positive=pos,
                              table=[{k: r[k] for k in ("threshold", "n", "win_rate", "expectancy")} for r in rows])
    info.update(calibration=cal, thresholds=thr, val=val_rows, norm=norm, n_train=int(len(tr)), n_val=int(len(va)))
    return net, info


ATR_COL = ft.FEATURE_NAMES.index("atr_pct")


def population_breakeven(ds, key, rows):
    """Empirical break-even TP-first rate: given the actual mix of non-TP outcomes (SL and timeouts, whose
    average PnL is smaller in magnitude than an SL), p* = -l / (w - l).  Lower than the analytic figure."""
    y = ds.lab[key][rows]
    out = np.where(y == 1, 1, 0).astype("int8")
    return breakeven_win_rate(dict(outcome=out, pnl_usd=ds.pnl[key][rows]))


def analytic_breakeven(ds, mode, rows):
    """The doc's break-even win rate (wins exit at TP, losses at SL; fees+slippage included).  For atr
    exits it is evaluated at the median 5m ATR% of the sample (the notional is slice-capped, so fees weigh
    more when ATR% is small)."""
    med = float(np.median(ds.X[rows, ATR_COL].astype("float32"))) if len(rows) else 0.5
    return breakeven_from_cfg(ds.meta["exit_cfgs"][mode], max(med, 0.05))


def evaluate_block(ds, net, info, mode, rng_ms, log, tag="test"):
    """Out-of-sample evaluation on a time block with the fold's val-chosen thresholds + calibration."""
    ok = ds.sample_ok
    any_valid = ok & (np.isfinite(ds.lab[f"{mode}_long"]) | np.isfinite(ds.lab[f"{mode}_short"]))
    te = rows_in(ds, rng_ms, any_valid)
    lg = T.predict_logits(net, ds.X, te, info["norm"])
    res = {"n_rows": int(len(te)), "range": [iso(rng_ms[0]), iso(rng_ms[1])]}
    arrays = {}
    for si, side in enumerate(SIDES):
        key = f"{mode}_{side}"
        m = np.isfinite(ds.lab[key][te])
        sel = te[m]
        y = ds.lab[key][sel]
        z = lg[m, si]
        p_raw = mnp.sigmoid(z)
        p_cal = mnp.apply_calibration(info["calibration"][side], z)
        be = analytic_breakeven(ds, mode, sel)
        be_mix = population_breakeven(ds, key, sel)
        table = E.trading_table(p_cal, y, ds.pnl[key][sel], ds.t[sel], ds.sym[sel], ds.off[key][sel], be)
        th = info["thresholds"][side]
        chosen = E.trading_table(p_cal, y, ds.pnl[key][sel], ds.t[sel], ds.sym[sel], ds.off[key][sel], be,
                                 thresholds=(th,), n_random=10)[0]
        # drift behaviour at the chosen threshold: calibrated vs uncalibrated probabilities, same trades
        cand = np.nonzero(p_cal >= th)[0]
        taken = E.select_trades(cand, p_cal, ds.t[sel], ds.sym[sel], ds.off[key][sel])
        drift = dict(
            calibrated=E.drift_firing(p_cal[taken], y[taken]),
            uncalibrated=E.drift_firing(p_raw[taken], y[taken]))
        res[side] = dict(
            auc=E.auc(y, p_raw), calibration=E.calibration_report(y, p_raw, p_cal),
            auc_atr_only=E.auc(y, ds.X[sel, ATR_COL].astype("float32")),
            breakeven_win_rate=be, breakeven_win_rate_mix=be_mix, base_rate=float(y.mean()), table=table, chosen=chosen, drift=drift)
        arrays[side] = dict(p=p_cal.astype("float32"), p_raw=p_raw.astype("float32"), y=y, pnl=ds.pnl[key][sel],
                            t=ds.t[sel], sym=ds.sym[sel], off=ds.off[key][sel], th=th, taken=taken)
        log(f"  [{tag}/{side}] AUC {res[side]['auc']:.3f} (ATR-only {res[side]['auc_atr_only']:.3f}) base "
            f"{y.mean():.3f} BE {be:.3f} (mix {be_mix:.3f}) | thr {th:.2f}: "
            f"n={chosen['n']} win={chosen['win_rate']} exp={chosen['expectancy']} "
            f"(random {chosen['baseline_expectancy']})")
    return res, arrays


def pool(arrs_list, side):
    """Concatenate per-fold OOS arrays for one side (trades keep fold-local selection)."""
    keys = ("p", "p_raw", "y", "pnl", "t", "sym", "off")
    return {k: np.concatenate([a[side][k] for a in arrs_list]) for k in keys}


def pooled_report(arrs_list, folds_info, mode, ds, side="long"):
    """Pooled OOS: (a) fixed threshold grid on all folds, (b) each fold's own val-chosen threshold."""
    P = pool(arrs_list, side)
    key = f"{mode}_{side}"
    be = float(np.mean([f[side]["breakeven_win_rate"] for f in folds_info]))
    grid = E.trading_table(P["p"], P["y"], P["pnl"], P["t"], P["sym"], P["off"], be)
    # (b) concatenation of each fold's chosen-threshold trades
    sel_pnl, sel_y, sel_p, sel_praw, sel_t, sel_end = [], [], [], [], [], []
    for a in arrs_list:
        A = a[side]
        tk = A["taken"]
        sel_pnl.append(A["pnl"][tk]); sel_y.append(A["y"][tk]); sel_p.append(A["p"][tk]); sel_praw.append(A["p_raw"][tk])
        sel_t.append(A["t"][tk]); sel_end.append(A["t"][tk] + np.maximum(A["off"][tk], 1) * BAR_MS)
    pnl = np.concatenate(sel_pnl) if sel_pnl else np.zeros(0)
    y = np.concatenate(sel_y) if sel_y else np.zeros(0)
    n = len(pnl)
    base_exp = float(np.mean([f[side]["chosen"]["baseline_expectancy"] for f in folds_info
                              if f[side]["chosen"]["baseline_expectancy"] is not None] or [np.nan]))
    base_win = float(np.mean([f[side]["chosen"]["baseline_win_rate"] for f in folds_info
                              if f[side]["chosen"]["baseline_win_rate"] is not None] or [np.nan]))
    se = float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    chosen = dict(n=int(n), win_rate=float(y.mean()) if n else None, expectancy=float(pnl.mean()) if n else None,
                  total_pnl=float(pnl.sum()) if n else 0.0, se=se, baseline_expectancy=base_exp,
                  baseline_win_rate=base_win, breakeven_win_rate=be,
                  z_vs_zero=float(pnl.mean() / se) if n > 1 and se else None,
                  z_vs_baseline=float((pnl.mean() - base_exp) / se) if n > 1 and se and np.isfinite(base_exp) else None,
                  equity=dict(t=np.concatenate(sel_end).tolist() if n else [], pnl=pnl.tolist()))
    # drift pooled (windows never straddle folds)
    wins_cal = sum(f[side]["drift"]["calibrated"]["windows"] for f in folds_info)
    drift = {}
    for kind in ("calibrated", "uncalibrated"):
        w = [(f[side]["drift"][kind]["windows"], f[side]["drift"][kind]["frac_z_lt_m2"],
              f[side]["drift"][kind]["frac_z_lt_m3"], f[side]["drift"][kind]["mean_gap"]) for f in folds_info]
        tw = sum(x[0] for x in w)
        drift[kind] = dict(
            windows=int(tw),
            frac_z_lt_m2=float(sum(x[0] * x[1] for x in w if x[1] is not None) / tw) if tw else None,
            frac_z_lt_m3=float(sum(x[0] * x[2] for x in w if x[2] is not None) / tw) if tw else None,
            mean_gap=float(np.mean([x[3] for x in w if x[3] is not None])) if tw else None)
    _ = wins_cal
    return dict(grid=grid, chosen=chosen, drift=drift,
                auc=float(np.nanmean([f[side]["auc"] for f in folds_info])))


def reachability(ds, modes, max_gap):
    """Per exit config: analytic break-even win rate (median ATR%) vs the win rate of random entries over the
    whole archive.  A config whose break-even sits more than ``max_gap`` above what random entries already
    achieve (on both sides) asks the model for an implausible lift and is dropped."""
    out = {}
    ok = ds.sample_ok
    for m in modes:
        r = {}
        for side in SIDES:
            key = f"{m}_{side}"
            rows = np.nonzero(ok & np.isfinite(ds.lab[key]))[0]
            be = analytic_breakeven(ds, m, rows)
            rw = float(ds.lab[key][rows].mean())
            r[side] = dict(breakeven=be, random_win=rw, gap=be - rw, random_expectancy=float(ds.pnl[key][rows].mean()),
                           mean_exit_bars=float(ds.off[key][rows].mean()), timeout_frac=float(np.mean(ds.off[key][rows] >= ds.meta.get("horizon", D.HORIZON))))
        r["drop"] = bool(all(r[sd]["gap"] > max_gap for sd in SIDES))
        out[m] = r
    return out


def strip_equity(d):
    d = json.loads(json.dumps(d, default=float))
    if "chosen" in d and "equity" in d["chosen"]:
        d["chosen"].pop("equity")
    return d


# --------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------
def run_walkforward(ds, arch, mode, splits, cfg: T.TrainCfg, log):
    folds_info, arrs, trains = [], [], []
    for f in splits["folds"]:
        log(f"=== {arch}/{mode} fold {f['fold']}: train<{iso(f['train'][1])} val {iso(f['val'][0])}..{iso(f['val'][1])} "
            f"test {iso(f['test'][0])}..{iso(f['test'][1])}")
        net, info = fit_split(ds, arch, mode, f["train"], f["val"], cfg, log)
        res, a = evaluate_block(ds, net, info, mode, f["test"], log)
        res.update(fold=f["fold"], thresholds=info["thresholds"], calibration=info["calibration"],
                   val=info["val"], train=dict(n_train=info["n_train"], n_val=info["n_val"],
                                               best_val_loss=info["best_val_loss"], steps=info["steps_run"],
                                               ms_per_step=info["ms_per_step"], seconds=info["seconds"]))
        folds_info.append(res)
        arrs.append(a)
        del net
        T.free_gpu()
    out = dict(arch=arch, mode=mode, folds=folds_info)
    for side in SIDES:
        out[f"pooled_{side}"] = pooled_report(arrs, folds_info, mode, ds, side)
    return out, arrs


def pick_best(results: dict) -> tuple[str, str]:
    """(arch, mode) with the best pooled OOS long expectancy at val-chosen thresholds."""
    best, key = -1e9, None
    for k, r in results.items():
        e = r["pooled_long"]["chosen"]["expectancy"]
        if e is not None and r["pooled_long"]["chosen"]["n"] >= 20 and e > best:
            best, key = e, k
    if key is None:                       # nobody traded enough: fall back to the best pooled OOS AUC
        key = max(results, key=lambda k: results[k]["pooled_long"]["auc"])
    return tuple(key.split("/"))


def run_final(ds, arch, mode, splits, cfg, log):
    fs = splits["final"]
    log(f"=== FINAL {arch}/{mode}: train<{iso(fs['train'][1])} val {iso(fs['val'][0])}..{iso(fs['val'][1])} "
        f"HOLDOUT {iso(fs['holdout'][0])}..{iso(fs['holdout'][1])} (evaluated once)")
    net, info = fit_split(ds, arch, mode, fs["train"], fs["val"], cfg, log)
    res, arrs = evaluate_block(ds, net, info, mode, fs["holdout"], log, tag="HOLDOUT")
    weights = T.M.export_state(net)
    return net, info, res, arrs, weights


def export_model(ds, arch, mode, info, weights, wf, holdout, models_dir: Path, cfg, status="challenger"):
    now = datetime.now(timezone.utc)
    name = f"{arch}_{mode}_{now.strftime('%Y%m%d_%H%M')}"
    models_dir.mkdir(parents=True, exist_ok=True)
    parent = None
    for p in sorted(models_dir.glob("*.json")):
        try:
            m = json.loads(p.read_text())
            if m.get("status") == "champion":
                parent = m.get("name")
        except Exception:
            pass
    exit_cfg = ds.meta["exit_cfgs"][mode]
    meta = dict(
        name=name, version=now.strftime("%Y%m%d.%H%M"), created_at=now.isoformat(), status=status, parent=parent,
        arch=arch, arch_cfg=info["arch_cfg"], feature_names=ft.FEATURE_NAMES, window=ft.WINDOW,
        norm=dict(mean=info["norm"]["mean"].tolist(), std=info["norm"]["std"].tolist(), clip=ft.NORM_CLIP),
        calibration=info["calibration"], thresholds=dict(info["thresholds"]), exit_cfg=exit_cfg,
        metrics=dict(walk_forward=wf, holdout=holdout, validation_thresholds=info["val"],
                     train=dict(n_train=info["n_train"], n_val=info["n_val"], best_val_loss=info["best_val_loss"],
                                steps=info["steps_run"], ms_per_step=info["ms_per_step"]),
                     dataset=dict(symbols=ds.symbols, rows=int(len(ds)), horizon=ds.meta.get("horizon"))))
    tmp = models_dir / f".{name}.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **weights)
    os.replace(tmp, models_dir / f"{name}.npz")
    tmpj = models_dir / f".{name}.json.tmp"
    tmpj.write_text(json.dumps(meta, indent=1, default=float))
    os.replace(tmpj, models_dir / f"{name}.json")
    return name


def lab_report(name, ds, arch, mode, wf_all, wf_chosen, holdout, equity, notes, var: Path):
    created = int(time.time())
    ch = wf_chosen["pooled_long"]["chosen"]
    hl = holdout["long"]["chosen"]
    summary = dict(
        model=name, arch=arch, exit_mode=mode, symbols=len(ds.symbols), rows=int(len(ds)),
        wf_oos_trades=ch["n"], wf_oos_win_rate=ch["win_rate"], wf_oos_expectancy_usd=ch["expectancy"],
        wf_random_expectancy_usd=ch["baseline_expectancy"], wf_breakeven_win_rate=ch["breakeven_win_rate"],
        wf_z_vs_baseline=ch["z_vs_baseline"], holdout_trades=hl["n"], holdout_win_rate=hl["win_rate"],
        holdout_expectancy_usd=hl["expectancy"], holdout_random_expectancy_usd=hl["baseline_expectancy"],
        holdout_breakeven_win_rate=hl["breakeven_win_rate"],
        wf_auc_long=wf_chosen["pooled_long"]["auc"], holdout_auc_long=holdout["long"]["auc"],
        drift_calibrated=wf_chosen["pooled_long"]["drift"]["calibrated"],
        drift_uncalibrated=wf_chosen["pooled_long"]["drift"]["uncalibrated"],
        verdict=notes, candidates={k: dict(expectancy=r["pooled_long"]["chosen"]["expectancy"],
                                           win_rate=r["pooled_long"]["chosen"]["win_rate"],
                                           n=r["pooled_long"]["chosen"]["n"]) for k, r in wf_all.items()})
    rep = dict(id=f"train-{name}", created_at=created, kind="train", title=f"Train {name} ({arch}, {mode} exits)",
               summary=summary, equity=equity)
    out = var / "lab"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{rep['id']}.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def build_equity(wf_chosen, holdout_arrs, start=100.0):
    """Equity (flat slices, no compounding) from OOS trades: walk-forward folds then the holdout."""
    eq = wf_chosen["pooled_long"]["chosen"]["equity"]
    t = np.asarray(eq["t"], "int64")
    pnl = np.asarray(eq["pnl"], "float64")
    A = holdout_arrs["long"]
    tk = A["taken"]
    t_h = A["t"][tk] + np.maximum(A["off"][tk], 1) * BAR_MS
    t = np.concatenate([t, t_h])
    pnl = np.concatenate([pnl, A["pnl"][tk]])
    o = np.argsort(t, kind="stable")
    cum = start + np.cumsum(pnl[o])
    pts = [dict(t=int(t[o][i] // 1000), equity=float(cum[i])) for i in range(len(o))]
    if len(pts) > 600:                                    # keep the report small
        sel = np.unique(np.linspace(0, len(pts) - 1, 600).astype(int))
        pts = [pts[i] for i in sel]
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "wf", "final"], help="run = wf + final; wf = walk-forward only (saves var/ml/wf_*.json); final = pick the best saved wf result, train on all dev data, ONE holdout evaluation, export")
    ap.add_argument("--archs", default="cnn")
    ap.add_argument("--modes", default="fixed,atr")
    ap.add_argument("--symbols", default="", help="comma list (default: all in archive)")
    ap.add_argument("--archive", default="")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--val-days", type=int, default=14)
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--gru-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--pos-weight", default="auto", choices=["auto", "none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-be-gap", type=float, default=0.15,
                    help="drop an exit config when break-even minus random-entry win rate exceeds this on BOTH sides")
    ap.add_argument("--force-modes", action="store_true", help="train even configs flagged unreachable")
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="", help="results dir (default var/ml)")
    a = ap.parse_args(argv)

    log = lambda *s: print(*s, flush=True)  # noqa: E731
    T.require_mps()
    var = D.var_dir()
    out_dir = Path(a.out or var / "ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(a.models_dir or os.environ.get("TB_MODELS") or D.ROOT / "models")

    if a.synthetic:
        ds = D.synthetic_dataset(n_symbols=4, n=int(os.environ.get("SYN_N", 60000)), planted_edge=0.5)
    else:
        root = Path(a.archive) if a.archive else None
        syms = [s for s in a.symbols.split(",") if s] or None
        ds = D.build_dataset(syms, root, use_cache=not a.no_cache, log=log)
    log(f"dataset: {len(ds)} rows, {len(ds.symbols)} symbols, X {ds.X.nbytes / 2**20:.0f} MiB fp16 "
        f"(RAM budget {D.ram_budget_bytes() / 2**30:.1f} GiB)")
    splits = make_splits(ds.t, a.folds, a.test_days, a.val_days, a.holdout_days,
                         purge_bars=int(ds.meta.get('horizon', D.HORIZON)) + ft.WINDOW)
    if not splits["folds"]:
        raise SystemExit("not enough history for even one walk-forward fold")
    log("splits: " + json.dumps({"folds": [{k: [iso(x) for x in v] if isinstance(v, tuple) else v
                                            for k, v in f.items()} for f in splits["folds"]],
                                 "final": {k: [iso(x) for x in v] for k, v in splits["final"].items()}}))

    modes = [m for m in a.modes.split(",") if m]
    reach = reachability(ds, modes, a.max_be_gap)
    (out_dir / "reachability.json").write_text(json.dumps(reach, indent=1, default=float))
    log("exit-config reachability (break-even TP-first rate vs the random-entry rate; whole archive):")
    for m, r in reach.items():
        log(f"  {m:6s} " + "  ".join(f"{sd}: BE {r[sd]['breakeven']:.3f} random-win {r[sd]['random_win']:.3f} "
                                       f"gap {r[sd]['gap']:+.3f}" for sd in SIDES) + f"  -> {'DROP' if r['drop'] else 'train'}")
    if a.force_modes:
        for r in reach.values():
            r["drop"] = False
    modes = [m for m in modes if not reach[m]["drop"]]
    if not modes:
        raise SystemExit("every requested exit config has an unreachable break-even (use --force-modes)")
    results, arrays = {}, {}
    t0 = time.time()
    for arch in ([] if a.cmd == "final" else a.archs.split(",")):
        for mode in modes:
            cfg = T.TrainCfg(arch=arch, steps=a.gru_steps if arch == "gru" else a.steps, batch=a.batch, lr=a.lr,
                             patience=a.patience, eval_every=min(a.eval_every, a.gru_steps) if arch == "gru" else a.eval_every,
                             pos_weight=a.pos_weight, seed=a.seed)
            res, arrs = run_walkforward(ds, arch, mode, splits, cfg, log)
            results[f"{arch}/{mode}"] = res
            arrays[f"{arch}/{mode}"] = arrs
            (out_dir / f"wf_{arch}_{mode}.json").write_text(json.dumps(res, indent=1, default=float))
            c = res["pooled_long"]["chosen"]
            log(f"### {arch}/{mode} pooled OOS long: n={c['n']} win={c['win_rate']} exp={c['expectancy']} "
                f"random={c['baseline_expectancy']} BE={c['breakeven_win_rate']:.3f} z0={c['z_vs_zero']} "
                f"zb={c['z_vs_baseline']}  [{time.time() - t0:.0f}s]")
    if a.cmd == "wf":
        return results
    if a.cmd == "final":
        for arch in a.archs.split(","):
            for mode in modes:
                f_ = out_dir / f"wf_{arch}_{mode}.json"
                if f_.exists():
                    results[f"{arch}/{mode}"] = json.loads(f_.read_text())
        if not results:
            raise SystemExit(f"no wf_*.json for archs={a.archs} modes={modes} in {out_dir}")
        log("candidates: " + ", ".join(results))
    arch, mode = pick_best(results)
    log(f"chosen on walk-forward OOS only: {arch}/{mode}")
    cfg = T.TrainCfg(arch=arch, steps=a.gru_steps if arch == "gru" else a.steps, batch=a.batch, lr=a.lr,
                     patience=a.patience, eval_every=a.eval_every, pos_weight=a.pos_weight, seed=a.seed)
    net, info, hres, harrs, weights = run_final(ds, arch, mode, splits, cfg, log)
    wf = strip_equity(results[f"{arch}/{mode}"])
    name = export_model(ds, arch, mode, info, weights, wf, json.loads(json.dumps(hres, default=float)), models_dir, cfg)
    hl = hres["long"]["chosen"]
    chosen = results[f"{arch}/{mode}"]["pooled_long"]["chosen"]
    beats = (chosen["z_vs_baseline"] or 0) > 2 and (chosen["expectancy"] or -1) > 0
    notes = ("walk-forward OOS beats the random baseline" if beats else
             "no statistically meaningful out-of-sample edge over the random-entry baseline") + \
        f"; holdout expectancy {hl['expectancy']} vs random {hl['baseline_expectancy']} (n={hl['n']})"
    equity = build_equity(results[f"{arch}/{mode}"], harrs)
    rep = lab_report(name, ds, arch, mode, results, results[f"{arch}/{mode}"], json.loads(json.dumps(hres, default=float)),
                     equity, notes, var)
    (out_dir / f"final_{name}.json").write_text(json.dumps(dict(holdout=hres, info=dict(
        thresholds=info["thresholds"], calibration=info["calibration"], val=info["val"])), indent=1, default=float))
    log(f"exported models/{name}.npz/.json (status=challenger); lab report {rep['id']}")
    log(json.dumps(rep["summary"], indent=1, default=float))
    return results


if __name__ == "__main__":
    main()
