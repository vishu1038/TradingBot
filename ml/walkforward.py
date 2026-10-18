"""Walk-forward training / evaluation / export driver.

    PYTHONPATH=backend nice -n 10 .venv-ml/bin/python ml/walkforward.py wf --archs cnn --modes fixed,atr32,atr63 \\
        --aux 1 --seeds 5 --folds 8 --test-days 45
    ... --synthetic            # smoke test on generated data (no archive needed)

Stages
  1. dataset (features [+ optional depth group] + labels + aux targets), memmapped on disk, built in a process pool
  2. walk-forward: K expanding-window folds, each train | purge | val | purge | test.  Per fold: N seeds trained
     (early stopping on val), logits averaged, Platt calibration fitted on val, per-side thresholds chosen on val
     ONLY, then trading + calibration + drift metrics on the untouched test block, next to two baselines:
       * random entries with identical exits / position rules / costs
       * the one-line ATR% rule (trade when ATR% is in the top q of the train distribution; q chosen on val)
  3. ``final``: retrain the chosen config on all dev data, evaluate ONCE on the locked holdout, export.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
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
from splits import BAR_MS, DAY_MS, iso, make_splits  # noqa: E402,F401
from tb import features as ft  # noqa: E402
from tb import model_np as mnp  # noqa: E402
from tb.labels import breakeven_from_cfg, breakeven_win_rate  # noqa: E402

SIDES = D.SIDES
TRAIN_ROWS_MAX = 8_000_000
EVAL_ROWS_MAX = 600_000


def thin(rows: np.ndarray, max_rows: int) -> np.ndarray:
    if len(rows) <= max_rows:
        return rows
    return rows[::int(np.ceil(len(rows) / max_rows))]


def atr_col(ds) -> int:
    return ds.feature_names.index("atr_pct")


def analytic_breakeven(ds, mode, rows):
    """The doc's break-even win rate (wins exit at TP, losses at SL; fees+slippage included).  For atr exits
    it is evaluated at the median 5m ATR% of the sample (the notional is slice-capped)."""
    med = float(np.median(np.asarray(ds.X[rows[::max(1, len(rows) // 20000)], atr_col(ds)]).astype("float32"))) if len(rows) else 0.5
    return breakeven_from_cfg(ds.meta["exit_cfgs"][mode], max(med, 0.05))


def population_breakeven(ds, key, rows):
    """Empirical break-even TP-first rate given the actual mix of non-TP outcomes (SL and timeouts)."""
    y = ds.lab[key][rows]
    return breakeven_win_rate(dict(outcome=np.where(y == 1, 1, 0).astype("int8"), pnl_usd=ds.pnl[key][rows]))


# --------------------------------------------------------------------------------------------
# one train / calibrate / threshold cycle
# --------------------------------------------------------------------------------------------
def fit_split(ds, arch, mode, tr_rng, va_rng, cfg: T.TrainCfg, log, n_seeds: int = 1):
    tr = thin(T.select_rows(ds, tr_rng, mode), TRAIN_ROWS_MAX)
    va = thin(T.select_rows(ds, va_rng, mode), EVAL_ROWS_MAX)
    sub = tr[np.linspace(0, len(tr) - 1, min(len(tr), 400_000)).astype("int64")]
    norm = ft.robust_norm_stats(np.asarray(ds.X[sub]).astype("float32"))
    nets, infos = [], []
    for k in range(n_seeds):
        net, info = T.train_model(ds, mode, tr, va, norm, replace(cfg, seed=cfg.seed + k), log)
        nets.append(net)
        infos.append(info)
    lg_val = T.predict_ensemble(nets, ds, va, norm)
    ac = atr_col(ds)
    rule = E.AtrRule(np.asarray(ds.X[sub, ac]).astype("float32"))
    p_rule = rule.score(np.asarray(ds.X[va, ac]).astype("float32"))
    cal, thr, val_rows, rule_info = {}, {}, {}, {}
    for si, side in enumerate(SIDES):
        key = f"{mode}_{side}"
        y_all = ds.lab[key][va]
        m = np.isfinite(y_all)
        sel = va[m]
        y = y_all[m]
        cal[side] = mnp.fit_calibration(lg_val[m, si], y, "platt")
        p = mnp.apply_calibration(cal[side], lg_val[m, si])
        be = analytic_breakeven(ds, mode, sel)
        args = (y, ds.pnl[key][sel], ds.t[sel], ds.sym[sel], ds.off[key][sel], be)
        rows = E.trading_table(p, *args, n_random=2)
        thr[side], pos = E.choose_threshold(rows)
        val_rows[side] = dict(threshold=thr[side], expectancy_positive=pos,
                              table=[{k: r[k] for k in ("threshold", "n", "win_rate", "expectancy")} for r in rows])
        rrows = E.trading_table(p_rule[m], *args, thresholds=E.ATR_QUANTS, n_random=1)
        q, rpos = E.choose_threshold(rrows, default=0.8)
        rule_info[side] = dict(quantile=q, atr_pct_at_q=rule.atr_at(q), expectancy_positive=rpos)
    info = dict(calibration=cal, thresholds=thr, val=val_rows, norm=norm, n_train=int(len(tr)), n_val=int(len(va)),
                best_val_loss=float(np.mean([i["best_val_loss"] for i in infos])), arch_cfg=infos[0]["arch_cfg"],
                steps_run=int(np.mean([i["steps_run"] for i in infos])),
                ms_per_step=float(np.mean([i["ms_per_step"] for i in infos])),
                seconds=float(sum(i["seconds"] for i in infos)), streaming=infos[0]["streaming"], aux=cfg.aux,
                n_seeds=n_seeds, atr_rule=rule_info, rule_ref=rule.ref[::max(1, len(rule.ref) // 20000)])
    return nets, info


def _fold_trades(p, y, pnl, t, sym, off, th, be, n_random=10):
    chosen = E.trading_table(p, y, pnl, t, sym, off, be, thresholds=(th,), n_random=n_random)[0]
    cand = np.nonzero(p >= th)[0]
    taken = E.select_trades(cand, p, t, sym, off)
    return chosen, taken


def evaluate_block(ds, nets, info, mode, rng_ms, log, tag="test"):
    """Out-of-sample evaluation on a time block with the fold's val-chosen thresholds + calibration."""
    te = thin(T.select_rows(ds, rng_ms, mode), EVAL_ROWS_MAX)
    lg_all = T.predict_ensemble(nets, ds, te, info["norm"])
    lg0 = T.predict_logits(nets[0], ds, te, info["norm"])[:, :2] if len(nets) > 1 else lg_all
    ac = atr_col(ds)
    atr_v = np.asarray(ds.X[te, ac]).astype("float32")
    rule = E.AtrRule(info["rule_ref"])
    res = {"n_rows": int(len(te)), "range": [iso(rng_ms[0]), iso(rng_ms[1])]}
    arrays = {}
    for si, side in enumerate(SIDES):
        key = f"{mode}_{side}"
        y_all = ds.lab[key][te]
        m = np.isfinite(y_all)
        sel = te[m]
        y = y_all[m]
        z = lg_all[m, si]
        p_raw = mnp.sigmoid(z)
        p_cal = mnp.apply_calibration(info["calibration"][side], z)
        be = analytic_breakeven(ds, mode, sel)
        be_mix = population_breakeven(ds, key, sel)
        pnl, t, sym, off = ds.pnl[key][sel], ds.t[sel], ds.sym[sel], ds.off[key][sel]
        table = E.trading_table(p_cal, y, pnl, t, sym, off, be)
        th = info["thresholds"][side]
        chosen, taken = _fold_trades(p_cal, y, pnl, t, sym, off, th, be)
        # one-line baseline: ATR% regime rule at the val-chosen quantile
        p_rule = rule.score(atr_v[m])
        q = info["atr_rule"][side]["quantile"]
        rchosen, rtaken = _fold_trades(p_rule, y, pnl, t, sym, off, q, be, n_random=3)
        cand = np.nonzero(p_cal >= th)[0]
        drift = dict(calibrated=E.drift_firing(p_cal[taken], y[taken]), uncalibrated=E.drift_firing(p_raw[taken], y[taken]))
        cal_rep = E.calibration_report(y, p_raw, p_cal)
        res[side] = dict(
            auc=E.auc(y, p_raw), auc_member0=E.auc(y, mnp.sigmoid(lg0[m, si])) if len(nets) > 1 else None,
            auc_atr_only=E.auc(y, atr_v[m]), logloss_cal=cal_rep["logloss_cal"],
            calibration=cal_rep, breakeven_win_rate=be, breakeven_win_rate_mix=be_mix, base_rate=float(y.mean()),
            table=table, chosen=chosen, atr_rule=dict(quantile=q, **rchosen), drift=drift, n_signals=int(len(cand)))
        arrays[side] = dict(p=p_cal.astype("float32"), p_raw=p_raw.astype("float32"), y=y, pnl=pnl, t=t, sym=sym,
                            off=off, th=th, taken=taken, rtaken=rtaken)
        log(f"  [{tag}/{side}] AUC {res[side]['auc']:.3f} (ATR-only {res[side]['auc_atr_only']:.3f}) base "
            f"{y.mean():.3f} BE {be:.3f} | thr {th:.2f}: n={chosen['n']} win={chosen['win_rate']} "
            f"exp={chosen['expectancy']} (random {chosen['baseline_expectancy']}) | ATR-rule q{q:.2f}: "
            f"n={rchosen['n']} exp={rchosen['expectancy']}")
    return res, arrays


def _stats(pnls, ys):
    pnl = np.concatenate(pnls) if pnls else np.zeros(0)
    y = np.concatenate(ys) if ys else np.zeros(0)
    n = len(pnl)
    se = float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(n=int(n), win_rate=float(y.mean()) if n else None, expectancy=float(pnl.mean()) if n else None,
                total_pnl=float(pnl.sum()) if n else 0.0, se=se, z_vs_zero=float(pnl.mean() / se) if n > 1 and se else None)


def pooled_report(arrs_list, folds_info, side="long"):
    """Pooled OOS over folds: model at each fold's val-chosen threshold, ATR rule at its val-chosen quantile,
    and the random-entry baseline averaged over folds."""
    be = float(np.mean([f[side]["breakeven_win_rate"] for f in folds_info]))
    mod = _stats([a[side]["pnl"][a[side]["taken"]] for a in arrs_list], [a[side]["y"][a[side]["taken"]] for a in arrs_list])
    rul = _stats([a[side]["pnl"][a[side]["rtaken"]] for a in arrs_list], [a[side]["y"][a[side]["rtaken"]] for a in arrs_list])
    bl = [f[side]["chosen"] for f in folds_info if f[side]["chosen"]["baseline_expectancy"] is not None]
    base_exp = float(np.mean([c["baseline_expectancy"] for c in bl])) if bl else float("nan")
    base_win = float(np.mean([c["baseline_win_rate"] for c in bl])) if bl else float("nan")
    mod.update(baseline_expectancy=base_exp, baseline_win_rate=base_win, breakeven_win_rate=be,
               z_vs_baseline=float((mod["expectancy"] - base_exp) / mod["se"]) if mod["n"] > 1 and mod["se"] and np.isfinite(base_exp) else None)
    rul.update(z_vs_baseline=float((rul["expectancy"] - base_exp) / rul["se"]) if rul["n"] > 1 and rul["se"] and np.isfinite(base_exp) else None)
    pnl_t = [(a[side]["t"][a[side]["taken"]] + np.maximum(a[side]["off"][a[side]["taken"]], 1) * BAR_MS) for a in arrs_list]
    mod["equity"] = dict(t=np.concatenate(pnl_t).tolist() if pnl_t else [],
                         pnl=np.concatenate([a[side]["pnl"][a[side]["taken"]] for a in arrs_list]).tolist() if arrs_list else [])
    drift = {}
    for kind in ("calibrated", "uncalibrated"):
        w = [(f[side]["drift"][kind]["windows"], f[side]["drift"][kind]["frac_z_lt_m2"], f[side]["drift"][kind]["frac_z_lt_m3"],
              f[side]["drift"][kind]["mean_gap"]) for f in folds_info]
        tw = sum(x[0] for x in w)
        drift[kind] = dict(
            windows=int(tw),
            frac_z_lt_m2=float(sum(x[0] * x[1] for x in w if x[1] is not None) / tw) if tw else None,
            frac_z_lt_m3=float(sum(x[0] * x[2] for x in w if x[2] is not None) / tw) if tw else None,
            mean_gap=float(np.mean([x[3] for x in w if x[3] is not None])) if tw else None)
    aucs = [f[side]["auc"] for f in folds_info]
    return dict(chosen=mod, atr_rule=rul, drift=drift, auc=float(np.nanmean(aucs)),
                auc_atr_only=float(np.nanmean([f[side]["auc_atr_only"] for f in folds_info])),
                logloss_cal=float(np.nanmean([f[side]["logloss_cal"] for f in folds_info])),
                ece_cal=float(np.nanmean([f[side]["calibration"]["ece_cal"] for f in folds_info])),
                ece_raw=float(np.nanmean([f[side]["calibration"]["ece_raw"] for f in folds_info])),
                grid=None)


def run_walkforward(ds, arch, mode, splits, cfg: T.TrainCfg, n_seeds, log):
    folds_info, arrs = [], []
    for f in splits["folds"]:
        log(f"=== {arch}/{mode} aux={int(cfg.aux)} seeds={n_seeds} fold {f['fold']}: train<{iso(f['train'][1])} "
            f"val {iso(f['val'][0])}..{iso(f['val'][1])} test {iso(f['test'][0])}..{iso(f['test'][1])}")
        nets, info = fit_split(ds, arch, mode, f["train"], f["val"], cfg, log, n_seeds)
        res, a = evaluate_block(ds, nets, info, mode, f["test"], log)
        res.update(fold=f["fold"], thresholds=info["thresholds"], calibration=info["calibration"], val=info["val"],
                   atr_rule_val=info["atr_rule"],
                   train=dict(n_train=info["n_train"], n_val=info["n_val"], best_val_loss=info["best_val_loss"],
                              steps=info["steps_run"], ms_per_step=info["ms_per_step"], seconds=info["seconds"],
                              streaming=info["streaming"]))
        folds_info.append(res)
        arrs.append(a)
        del nets
        T.free_gpu()
    out = dict(arch=arch, mode=mode, aux=bool(cfg.aux), seeds=n_seeds, folds=folds_info,
               groups=ds.groups, n_rows=int(len(ds)), n_symbols=len(ds.symbols))
    for side in SIDES:
        out[f"pooled_{side}"] = pooled_report(arrs, folds_info, side)
    return out, arrs


def reachability(ds, modes, max_gap):
    """Per exit config: analytic break-even vs the random-entry win rate over a thinned sample of the archive."""
    out = {}
    N = len(ds)
    step = max(1, N // 3_000_000)
    rows = np.arange(0, N, step)
    rows = rows[ds.sample_ok_at(rows)]
    for m in modes:
        r = {}
        for side in SIDES:
            key = f"{m}_{side}"
            ok = ds.lab[key].valid(rows)
            sel = rows[ok]
            be = analytic_breakeven(ds, m, sel)
            rw = float(ds.lab[key][sel].mean())
            off = ds.off[key][sel]
            r[side] = dict(breakeven=be, random_win=rw, gap=be - rw, random_expectancy=float(ds.pnl[key][sel].mean()),
                           mean_exit_bars=float(off.mean()), timeout_frac=float(np.mean(off >= ds.meta.get("horizon", D.HORIZON))))
        r["drop"] = bool(all(r[sd]["gap"] > max_gap for sd in SIDES))
        out[m] = r
    return out


# --------------------------------------------------------------------------------------------
# final: holdout + export
# --------------------------------------------------------------------------------------------
def run_final(ds, arch, mode, splits, cfg, n_seeds, log):
    fs = splits["final"]
    log(f"=== FINAL {arch}/{mode}: train<{iso(fs['train'][1])} val {iso(fs['val'][0])}..{iso(fs['val'][1])} "
        f"HOLDOUT {iso(fs['holdout'][0])}..{iso(fs['holdout'][1])} (evaluated once)")
    nets, info = fit_split(ds, arch, mode, fs["train"], fs["val"], cfg, log, n_seeds)
    res, arrs = evaluate_block(ds, nets, info, mode, fs["holdout"], log, tag="HOLDOUT")
    return nets, info, res, arrs


def export_model(ds, arch, mode, info, nets, wf, holdout, models_dir: Path, status="challenger"):
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
    weights = {}
    if len(nets) == 1:
        weights = T.M.export_state(nets[0])
    else:
        for k, n in enumerate(nets):
            weights.update({f"e{k}/{kk}": v for kk, v in T.M.export_state(n).items()})
    meta = dict(
        name=name, version=now.strftime("%Y%m%d.%H%M"), created_at=now.isoformat(), status=status, parent=parent,
        arch=arch, arch_cfg=info["arch_cfg"], ensemble=len(nets), feature_names=ds.feature_names,
        feature_groups=ds.groups, window=ft.WINDOW,
        norm=dict(mean=info["norm"]["mean"].tolist(), std=info["norm"]["std"].tolist(), clip=ft.NORM_CLIP),
        calibration=info["calibration"], thresholds=dict(info["thresholds"]), exit_cfg=ds.meta["exit_cfgs"][mode],
        metrics=dict(walk_forward=wf, holdout=holdout, validation_thresholds=info["val"],
                     train=dict(n_train=info["n_train"], n_val=info["n_val"], best_val_loss=info["best_val_loss"],
                                steps=info["steps_run"], ms_per_step=info["ms_per_step"], aux=info["aux"]),
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
        wf_z_vs_baseline=ch["z_vs_baseline"], wf_atr_rule_expectancy_usd=wf_chosen["pooled_long"]["atr_rule"]["expectancy"],
        holdout_trades=hl["n"], holdout_win_rate=hl["win_rate"], holdout_expectancy_usd=hl["expectancy"],
        holdout_random_expectancy_usd=hl["baseline_expectancy"], holdout_breakeven_win_rate=hl["breakeven_win_rate"],
        holdout_atr_rule_expectancy_usd=holdout["long"]["atr_rule"]["expectancy"],
        wf_auc_long=wf_chosen["pooled_long"]["auc"], holdout_auc_long=holdout["long"]["auc"],
        drift_calibrated=wf_chosen["pooled_long"]["drift"]["calibrated"],
        drift_uncalibrated=wf_chosen["pooled_long"]["drift"]["uncalibrated"], verdict=notes,
        candidates={k: dict(expectancy=r["pooled_long"]["chosen"]["expectancy"], win_rate=r["pooled_long"]["chosen"]["win_rate"],
                            n=r["pooled_long"]["chosen"]["n"]) for k, r in wf_all.items()})
    rep = dict(id=f"train-{name}", created_at=created, kind="train", title=f"Train {name} ({arch}, {mode} exits)",
               summary=summary, equity=equity)
    out = var / "lab"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{rep['id']}.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def build_equity(wf_chosen, holdout_arrs, start=100.0):
    eq = wf_chosen["pooled_long"]["chosen"]["equity"]
    t = np.asarray(eq["t"], "int64")
    pnl = np.asarray(eq["pnl"], "float64")
    A = holdout_arrs["long"]
    tk = A["taken"]
    t = np.concatenate([t, A["t"][tk] + np.maximum(A["off"][tk], 1) * BAR_MS])
    pnl = np.concatenate([pnl, A["pnl"][tk]])
    o = np.argsort(t, kind="stable")
    cum = start + np.cumsum(pnl[o])
    pts = [dict(t=int(t[o][i] // 1000), equity=float(cum[i])) for i in range(len(o))]
    if len(pts) > 600:
        pts = [pts[i] for i in np.unique(np.linspace(0, len(pts) - 1, 600).astype(int))]
    return pts


def strip_equity(d):
    d = json.loads(json.dumps(d, default=float))
    for side in SIDES:
        d.get(f"pooled_{side}", {}).get("chosen", {}).pop("equity", None)
    return d


def pick_best(results: dict) -> tuple[str, str, str]:
    """(arch, mode, tag) with the best pooled OOS long expectancy at val-chosen thresholds."""
    best, key = -1e9, None
    for k, r in results.items():
        e = r["pooled_long"]["chosen"]["expectancy"]
        if e is not None and r["pooled_long"]["chosen"]["n"] >= 20 and e > best:
            best, key = e, k
    if key is None:
        key = max(results, key=lambda k: results[k]["pooled_long"]["auc"])
    return tuple(key.split("/"))


def tag_of(aux: int, seeds: int, groups) -> str:
    return f"a{int(aux)}s{seeds}" + ("_" + "+".join(groups) if groups else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "wf", "final"], help="wf = walk-forward only (saves var/ml/wf_*.json); "
                    "final = pick the best saved wf result, train on all dev data, ONE holdout evaluation, export; run = both")
    ap.add_argument("--archs", default="cnn")
    ap.add_argument("--modes", default="fixed,atr32,atr63")
    ap.add_argument("--symbols", default="", help="comma list (default: all in archive)")
    ap.add_argument("--archive", default="")
    ap.add_argument("--groups", default="", help="optional feature groups, e.g. depth")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--val-days", type=int, default=14)
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--min-train-days", type=int, default=45)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--gru-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--pos-weight", default="auto", choices=["auto", "none"])
    ap.add_argument("--aux", type=int, default=0, help="1 = multi-task auxiliary heads")
    ap.add_argument("--aux-w", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, default=1, help="ensemble size (logits averaged)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="result tag (default a<aux>s<seeds>[_groups])")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-be-gap", type=float, default=0.15)
    ap.add_argument("--force-modes", action="store_true")
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
    groups = tuple(g for g in a.groups.split(",") if g)
    tag = a.tag or tag_of(a.aux, a.seeds, groups)

    if a.synthetic:
        ds = D.synthetic_dataset(n_symbols=4, n=int(os.environ.get("SYN_N", 60000)), planted_edge=0.5, groups=groups)
    else:
        root = Path(a.archive) if a.archive else None
        syms = [s for s in a.symbols.split(",") if s] or None
        ds = D.build_dataset(syms, root, use_cache=not a.no_cache, groups=groups, log=log)
    log(f"dataset: {len(ds)} rows, {len(ds.symbols)} symbols, {ds.n_feat} features, X {ds.X.nbytes / 2**30:.2f} GiB fp16 "
        f"({'GPU-resident' if ds.X.nbytes <= T.GPU_MAX_BYTES else 'streamed in GPU chunks'}; RAM budget "
        f"{D.ram_budget_bytes() / 2**30:.1f} GiB)  tag={tag}")
    splits = make_splits(np.asarray(ds.t[::97]), a.folds, a.test_days, a.val_days, a.holdout_days,
                         purge_bars=int(ds.meta.get("horizon", D.HORIZON)) + ft.WINDOW, min_train_days=a.min_train_days)
    if not splits["folds"]:
        raise SystemExit("not enough history for even one walk-forward fold")
    log("splits: " + json.dumps({"folds": [{k: [iso(x) for x in v] if isinstance(v, tuple) else v
                                            for k, v in f.items()} for f in splits["folds"]],
                                 "final": {k: [iso(x) for x in v] for k, v in splits["final"].items()}}))
    modes = [m for m in a.modes.split(",") if m]
    reach = reachability(ds, modes, a.max_be_gap)
    (out_dir / "reachability.json").write_text(json.dumps(reach, indent=1, default=float))
    log("exit-config reachability (break-even TP-first rate vs the random-entry rate):")
    for m, r in reach.items():
        log(f"  {m:6s} " + "  ".join(f"{sd}: BE {r[sd]['breakeven']:.3f} random-win {r[sd]['random_win']:.3f} "
                                       f"gap {r[sd]['gap']:+.3f}" for sd in SIDES) + f"  -> {'DROP' if r['drop'] else 'train'}")
    if not a.force_modes:
        modes = [m for m in modes if not reach[m]["drop"]]
    if not modes:
        raise SystemExit("every requested exit config has an unreachable break-even (use --force-modes)")

    def mk_cfg(arch):
        return T.TrainCfg(arch=arch, steps=a.gru_steps if arch == "gru" else a.steps, batch=a.batch, lr=a.lr,
                          patience=a.patience, eval_every=min(a.eval_every, a.gru_steps) if arch == "gru" else a.eval_every,
                          pos_weight=a.pos_weight, seed=a.seed, aux=bool(a.aux), aux_w=a.aux_w)

    results = {}
    t0 = time.time()
    for arch in ([] if a.cmd == "final" else a.archs.split(",")):
        for mode in modes:
            res, _ = run_walkforward(ds, arch, mode, splits, mk_cfg(arch), a.seeds, log)
            results[f"{arch}/{mode}"] = res
            (out_dir / f"wf_{arch}_{mode}_{tag}.json").write_text(json.dumps(strip_equity(res) if False else res, indent=1, default=float))
            c = res["pooled_long"]["chosen"]
            r_ = res["pooled_long"]["atr_rule"]
            log(f"### {arch}/{mode} [{tag}] pooled OOS long: n={c['n']} win={c['win_rate']} exp={c['expectancy']} "
                f"random={c['baseline_expectancy']} ATR-rule={r_['expectancy']} BE={c['breakeven_win_rate']:.3f} "
                f"zb={c['z_vs_baseline']}  [{time.time() - t0:.0f}s]")
    if a.cmd == "wf":
        return results
    if a.cmd == "final":
        for arch in a.archs.split(","):
            for mode in modes:
                f_ = out_dir / f"wf_{arch}_{mode}_{tag}.json"
                if f_.exists():
                    results[f"{arch}/{mode}"] = json.loads(f_.read_text())
        if not results:
            raise SystemExit(f"no wf_*_{tag}.json for archs={a.archs} modes={modes} in {out_dir}")
    arch, mode = pick_best(results)
    log(f"chosen on walk-forward OOS only: {arch}/{mode}")
    nets, info, hres, harrs = run_final(ds, arch, mode, splits, mk_cfg(arch), a.seeds, log)
    wf = strip_equity(results[f"{arch}/{mode}"])
    hres_j = json.loads(json.dumps(hres, default=float))
    name = export_model(ds, arch, mode, info, nets, wf, hres_j, models_dir)
    hl = hres["long"]["chosen"]
    chosen = results[f"{arch}/{mode}"]["pooled_long"]["chosen"]
    beats = (chosen["z_vs_baseline"] or 0) > 2 and (chosen["expectancy"] or -1) > 0
    notes = ("walk-forward OOS beats the random baseline" if beats else
             "no statistically meaningful out-of-sample edge over the random-entry baseline") + \
        f"; holdout expectancy {hl['expectancy']} vs random {hl['baseline_expectancy']} (n={hl['n']})"
    equity = build_equity(results[f"{arch}/{mode}"], harrs)
    rep = lab_report(name, ds, arch, mode, results, results[f"{arch}/{mode}"], hres_j, equity, notes, var)
    (out_dir / f"final_{name}.json").write_text(json.dumps(dict(holdout=hres, info=dict(
        thresholds=info["thresholds"], calibration=info["calibration"], val=info["val"])), indent=1, default=float))
    log(f"exported models/{name}.npz/.json (status=challenger); lab report {rep['id']}")
    return results


if __name__ == "__main__":
    main()
