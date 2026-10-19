"""Walk-forward / holdout / export driver for the cross-sectional experiment (V3 A + D).

    PYTHONPATH=backend nice -n 10 .venv-ml/bin/python ml/xsec_walkforward.py run --tag v1

Protocol (nothing here may look at a later period than it is allowed to):

 phase 1, per fold  (train | purge | val | purge | test, expanding window; purge >= max label horizon + margin)
    for every (model in {gbm, mlp}) x (horizon h in {12, 24, 48, 72}): fit on train, early-stop on VALIDATION rank-IC,
    save the val and test score matrices, then run the full execution grid on the VALIDATION window only:
    (top_k = bottom_k) x (rebalance_every_h in {h/4, h/2, h}) x (entry taker | maker)  -> 18 backtests per model.
 phase 2, selection (validation only) and evaluation (test windows)
    A) per-fold choice: best val IC picks (model, h); best val net return picks (k, R, entry).
    B) global choice: the same rules on the validation metrics averaged over the folds.  Config B is then run on every
       fold's TEST window; with the same rebalancing/stops/costs we also run N random rankings and the plain 7 d
       momentum rank (and, for information, the reversal rank).  Pooled out-of-sample = compounded fold results.
 phase 3, final: retrain config B on all development data (train | purge | val), evaluate ONCE on the locked holdout
    (last 2 months) against the same baselines, export the artifact (status "challenger").

The multiple-testing count (number of backtests / comparisons looked at on validation) is reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

import xsec_data as D  # noqa: E402
from splits import iso, make_splits  # noqa: E402
from tb import features_xsec as fx  # noqa: E402
from tb import model_xsec as mx  # noqa: E402
from tb.sim import xsec as X  # noqa: E402
from tb.sim.execution import ExecConfig  # noqa: E402

HOUR_MS = fx.HOUR_MS
ENTRIES = ("taker", "maker")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------
# configuration helpers
# --------------------------------------------------------------------------------------------
def rebalance_grid(h: int) -> list[int]:
    return sorted({max(1, h // 4), max(1, h // 2), h})


def exec_cfg(through_pct: float) -> ExecConfig:
    return ExecConfig(through_pct=through_pct, timeout_bars=3, reprices=2, on_timeout="skip")


def sim_cfg(k: int, R: int, entry: str, args) -> X.XsecConfig:
    return X.XsecConfig(rebalance_every_h=R, top_k=k, bottom_k=k, entry_mode=entry, exec=exec_cfg(args.through_pct),
                        kill_dd_pct=args.kill_dd, open_risk_pct=args.open_risk, stop_vol_mult=args.stop_vol_mult)


def small(res: X.XsecResult) -> dict:
    m = res.metrics
    return {"net_return_pct": m["net_return_pct"], "sharpe": m["sharpe"], "max_drawdown_pct": m["max_drawdown_pct"],
            "n_trades": m["n_trades"], "trades_per_day": m["trades_per_day"], "fees_paid": m["fees_paid"],
            "funding_paid": m["funding_paid"], "gross_pnl": m["gross_pnl"], "turnover_x_per_day": m["turnover_x_per_day"],
            "entry_fill_rate": m["entry_fill_rate"], "win_rate": m["win_rate"], "days": m["days"],
            "avg_net_exposure_frac": m["avg_net_exposure_frac"], "killed": res.kill_time is not None,
            "daily": [round(float(x), 6) for x in res.daily_ret]}


def backtest(ctx, score, cfg, k0, k1) -> X.XsecResult:
    return X.run_xsec(ctx.panel, score, cfg, k0, k1)


def pooled(daily_list: list[list[float]]) -> dict:
    r = np.concatenate([np.asarray(d, dtype="float64") for d in daily_list]) if daily_list else np.array([])
    if len(r) == 0:
        return {"net_return_pct": 0.0, "sharpe": float("nan"), "max_drawdown_pct": 0.0, "days": 0}
    eq = np.cumprod(1.0 + r)
    pk = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    sd = r.std(ddof=1)
    return {"net_return_pct": float((eq[-1] - 1) * 100), "sharpe": float(r.mean() / sd * np.sqrt(365)) if sd > 0 else float("nan"),
            "max_drawdown_pct": float(((pk - eq) / pk).max() * 100), "days": int(len(r))}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# --------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------
def train_ranker(model: str, tr: dict, va: dict, S: int, args):
    import xsec_train as TR
    if model == "gbm":
        return TR.train_gbm(tr, va, S, seed=args.seed, max_iter=args.gbm_iter)
    return TR.train_mlp(tr, va, S, seeds=args.mlp_seeds, epochs=args.mlp_epochs)


def save_scores(path: Path, ks: np.ndarray, sc: np.ndarray):
    np.savez_compressed(path, ks=ks, v=sc[ks].astype("float32"))


def load_scores(path: Path, T: int, S: int) -> np.ndarray:
    z = np.load(path)
    out = np.full((T, S), np.nan)
    out[z["ks"]] = z["v"]
    return out


# --------------------------------------------------------------------------------------------
# phase 1
# --------------------------------------------------------------------------------------------
def phase1_fold(ctx, sp, fi, args, out: Path):
    fp = out / f"fold{fi}.json"
    if fp.exists() and args.resume:
        log(f"fold {fi}: cached")
        return json.loads(fp.read_text())
    f = sp["folds"][fi]
    ks_tr = np.arange(*D.hour_range(ctx, *f["train"]))[::args.stride]
    ks_va = np.arange(*D.hour_range(ctx, *f["val"]))
    ks_te = np.arange(*D.hour_range(ctx, *f["test"]))
    log(f"fold {fi}: train {iso(f['train'][0])}..{iso(f['train'][1])} ({len(ks_tr)} hours) val {iso(f['val'][0])}..{iso(f['val'][1])} "
        f"test {iso(f['test'][0])}..{iso(f['test'][1])}")
    base_tr = D.build_rows(ctx, ks_tr, 0, need_label=False)
    base_va = D.build_rows(ctx, ks_va[::2], 0, need_label=False)
    rec = {"fold": fi, "ranges": {k: [iso(v[0]), iso(v[1])] for k, v in f.items() if isinstance(v, tuple)},
           "models": {}, "k_val": [int(ks_va[0]), int(ks_va[-1]) + 1], "k_test": [int(ks_te[0]), int(ks_te[-1]) + 1]}
    for h in args.horizons:
        tgt = D.targets(ctx, h)
        tr = D.with_labels(base_tr, *tgt)
        va = D.with_labels(base_va, *tgt)
        for model in args.models:
            key = f"{model}_h{h}"
            t0 = time.time()
            ranker, info = train_ranker(model, tr, va, ctx.S, args)
            sv = D.score_matrix(ctx, ranker.predict, ks_va)
            st = D.score_matrix(ctx, ranker.predict, ks_te)
            save_scores(out / f"scores_f{fi}_{key}_val.npz", ks_va, sv)
            save_scores(out / f"scores_f{fi}_{key}_test.npz", ks_te, st)
            icv = D.ic_summary(D.ic_matrix(sv, tgt[0], ks_va), h)
            ict = D.ic_summary(D.ic_matrix(st, tgt[0], ks_te), h)
            grid = []
            for k in args.ks:
                for R in rebalance_grid(h):
                    for entry in ENTRIES:
                        r = backtest(ctx, sv, sim_cfg(k, R, entry, args), ks_va[0], ks_va[-1] + 1)
                        grid.append({"k": k, "R": R, "entry": entry, **small(r)})
            info = {kk: vv for kk, vv in info.items() if kk != "curve"}
            rec["models"][key] = {"model": model, "h": h, "val_ic": icv, "test_ic": ict, "info": info, "val_grid": grid}
            log(f"fold {fi} {key}: val IC {icv['ic']:+.4f} (t {icv['t']:+.1f}) test IC {ict['ic']:+.4f} "
                f"({time.time() - t0:.0f}s) best val ret {max(g['net_return_pct'] for g in grid):+.1f}%")
        del tr, va, tgt
    fp.write_text(json.dumps(clean(rec)))
    return rec


# --------------------------------------------------------------------------------------------
# selection (validation data only)
# --------------------------------------------------------------------------------------------
def pick_model(recs: list[dict]) -> tuple[str, float]:
    keys = recs[0]["models"].keys()
    sc = {k: np.nanmean([r["models"][k]["val_ic"]["ic"] for r in recs]) for k in keys}
    best = max(sc, key=lambda k: sc[k] if np.isfinite(sc[k]) else -9)
    return best, sc[best]


def pick_cfg(recs: list[dict], key: str, min_trades: int = 10) -> dict:
    """(k, R, entry) with the highest validation net return (averaged over the given folds)."""
    agg: dict = {}
    for r in recs:
        for g in r["models"][key]["val_grid"]:
            agg.setdefault((g["k"], g["R"], g["entry"]), []).append(g)
    best, bs = None, -1e9
    for cfg, gs in agg.items():
        if np.mean([g["n_trades"] for g in gs]) < min_trades:
            continue
        v = float(np.mean([g["net_return_pct"] for g in gs]))
        if v > bs:
            best, bs = cfg, v
    if best is None:
        best = sorted(agg)[0]
        bs = float(np.mean([g["net_return_pct"] for g in agg[best]]))
    return {"k": best[0], "R": best[1], "entry": best[2], "val_mean_return_pct": bs}


# --------------------------------------------------------------------------------------------
# baselines + evaluation on a (score, window)
# --------------------------------------------------------------------------------------------
def eval_window(ctx, score, cfg, k0, k1, n_random, mom, rev, seed0=1000):
    """model + baselines on one window with identical rebalancing, stops and costs."""
    m = backtest(ctx, score, cfg, k0, k1)
    out = {"model": small(m)}
    out["momentum"] = small(backtest(ctx, mom, cfg, k0, k1))
    out["reversal"] = small(backtest(ctx, rev, cfg, k0, k1))
    rnd = []
    for s in range(n_random):
        rnd.append(small(backtest(ctx, fx.random_scores(ctx.mask, seed0 + s), cfg, k0, k1)))
    out["random"] = rnd
    return out


def summarize_pooled(evals: list[dict]) -> dict:
    """evals: list (per window) of eval_window outputs -> pooled model / baselines / z vs random."""
    res = {"model": pooled([e["model"]["daily"] for e in evals]),
           "momentum": pooled([e["momentum"]["daily"] for e in evals]),
           "reversal": pooled([e["reversal"]["daily"] for e in evals])}
    nrand = len(evals[0]["random"])
    rp = [pooled([e["random"][s]["daily"] for e in evals]) for s in range(nrand)]
    rr = np.array([p["net_return_pct"] for p in rp])
    res["random"] = {"mean_return_pct": float(rr.mean()), "std_return_pct": float(rr.std(ddof=1)) if nrand > 1 else float("nan"),
                     "p95_return_pct": float(np.quantile(rr, 0.95)), "mean_sharpe": float(np.nanmean([p["sharpe"] for p in rp])),
                     "n": nrand}
    res["z_vs_random"] = X.z_vs_random(res["model"]["net_return_pct"], rr)
    res["model_rank_in_random"] = float((rr < res["model"]["net_return_pct"]).mean())
    for key in ("model", "momentum", "reversal"):
        res[key].update({
            "fees_paid": float(sum(e[key]["fees_paid"] for e in evals)),
            "funding_paid": float(sum(e[key]["funding_paid"] for e in evals)),
            "gross_pnl": float(sum(e[key]["gross_pnl"] for e in evals)),
            "trades": int(sum(e[key]["n_trades"] for e in evals)),
            "trades_per_day": float(sum(e[key]["n_trades"] for e in evals) / max(1, sum(e[key]["days"] for e in evals))),
            "turnover_x_per_day": float(np.mean([e[key]["turnover_x_per_day"] for e in evals])),
            "killed": int(sum(bool(e[key]["killed"]) for e in evals)),
        })
    return res


# --------------------------------------------------------------------------------------------
def fmt_row(name, p):
    return (f"  {name:<22} net {p['net_return_pct']:+8.2f}%  Sharpe {p.get('sharpe', float('nan')):+6.2f}  maxDD {p['max_drawdown_pct']:5.1f}%"
            + (f"  trades/day {p.get('trades_per_day', 0):5.1f}  turnover {p.get('turnover_x_per_day', 0):5.2f}x/d  "
               f"fees ${p.get('fees_paid', 0):6.2f}  gross ${p.get('gross_pnl', 0):+7.2f}" if "fees_paid" in p else ""))


def phase2_and_3(ctx, sp, recs, args, out: Path, tag: str):
    folds = sp["folds"]
    T, S = ctx.T, ctx.S
    mom = fx.momentum_scores(ctx.cube, ctx.mask, "ret_168", 1.0)
    rev = fx.momentum_scores(ctx.cube, ctx.mask, "ret_168", -1.0)
    n_models = len(recs[0]["models"])
    grid_n = len(recs[0]["models"][next(iter(recs[0]["models"]))]["val_grid"])
    trials = {"folds": len(recs), "model_horizon_candidates": n_models, "exec_configs_per_candidate": grid_n,
              "val_backtests": len(recs) * n_models * grid_n, "val_ic_comparisons": len(recs) * n_models,
              "note": "per-fold choice = best val IC then best val net return; global choice = the same on the fold-averaged "
                      "validation metrics; random-baseline runs and test/holdout evaluations are not selection trials"}
    # ---- A) per-fold choices
    choices_a = []
    for r in recs:
        key, ic = pick_model([r])
        c = pick_cfg([r], key)
        choices_a.append({"fold": r["fold"], "model": key, "val_ic": ic, **c})
    # ---- B) global choice
    keyB, icB = pick_model(recs)
    cB = pick_cfg(recs, keyB)
    log(f"global choice: {keyB} (mean val IC {icB:+.4f}), k={cB['k']} R={cB['R']} entry={cB['entry']} "
        f"(mean val ret {cB['val_mean_return_pct']:+.1f}%)")
    report = {"tag": tag, "trials": trials, "choices_per_fold": choices_a,
              "global_choice": {"model_key": keyB, "mean_val_ic": icB, **cB},
              "fold_val_ic": {k: [r["models"][k]["val_ic"]["ic"] for r in recs] for k in recs[0]["models"]},
              "fold_test_ic": {k: [r["models"][k]["test_ic"]["ic"] for r in recs] for k in recs[0]["models"]}}
    # ---- evaluation on the TEST windows
    def run_tests(choice_fn, label):
        evals, rows = [], []
        for r in recs:
            ch = choice_fn(r)
            fi = r["fold"]
            sc = load_scores(out / f"scores_f{fi}_{ch['model']}_test.npz", T, S)
            k0, k1 = r["k_test"]
            cfg = sim_cfg(ch["k"], ch["R"], ch["entry"], args)
            ev = eval_window(ctx, sc, cfg, k0, k1, args.n_random, mom, rev)
            # entry-mode contrast on the same scores (information only, not a selection)
            alt = "maker" if ch["entry"] == "taker" else "taker"
            ev["alt_entry"] = {alt: small(backtest(ctx, sc, sim_cfg(ch["k"], ch["R"], alt, args), k0, k1))}
            evals.append(ev)
            rows.append({"fold": fi, **ch, "model_ret": ev["model"]["net_return_pct"], "momentum_ret": ev["momentum"]["net_return_pct"],
                         "random_mean_ret": float(np.mean([x["net_return_pct"] for x in ev["random"]])),
                         "model_sharpe": ev["model"]["sharpe"], "fees": ev["model"]["fees_paid"], "trades": ev["model"]["n_trades"]})
            log(f"  [{label}] fold {fi}: model {ev['model']['net_return_pct']:+.1f}% | momentum {ev['momentum']['net_return_pct']:+.1f}% | "
                f"random {rows[-1]['random_mean_ret']:+.1f}% ({ch['model']} k={ch['k']} R={ch['R']} {ch['entry']})")
        return evals, rows

    evalsA, rowsA = run_tests(lambda r: next(c for c in choices_a if c["fold"] == r["fold"]), "A per-fold")
    evalsB, rowsB = run_tests(lambda r: {"model": keyB, "k": cB["k"], "R": cB["R"], "entry": cB["entry"]}, "B global")
    report["oos_per_fold_choice"] = {"folds": rowsA, "pooled": summarize_pooled(evalsA)}
    report["oos_global_choice"] = {"folds": rowsB, "pooled": summarize_pooled(evalsB)}
    altB = {}
    for ev in evalsB:
        for mode, v in ev["alt_entry"].items():
            altB.setdefault(mode, []).append(v["daily"])
    report["oos_global_choice"]["alt_entry_pooled"] = {m: pooled(d) for m, d in altB.items()}
    # entry-mode contrast pooled for the chosen mode
    report["oos_global_choice"]["entry_mode_contrast"] = {
        cB["entry"]: report["oos_global_choice"]["pooled"]["model"], **report["oos_global_choice"]["alt_entry_pooled"]}

    # ---- phase 3: final retrain + the single holdout evaluation
    if args.no_final:
        report["holdout"] = None
        return report, None
    model, h = keyB.split("_h")[0], int(keyB.split("_h")[1])
    fin = sp["final"]
    ks_tr = np.arange(*D.hour_range(ctx, *fin["train"]))[::args.stride]
    ks_va = np.arange(*D.hour_range(ctx, *fin["val"]))
    ks_ho = np.arange(*D.hour_range(ctx, *fin["holdout"]))
    log(f"final: train {iso(fin['train'][0])}..{iso(fin['train'][1])} val {iso(fin['val'][0])}..{iso(fin['val'][1])} "
        f"HOLDOUT {iso(fin['holdout'][0])}..{iso(fin['holdout'][1])} (touched once)")
    tgt = D.targets(ctx, h)
    tr = D.with_labels(D.build_rows(ctx, ks_tr, 0, need_label=False), *tgt)
    va = D.with_labels(D.build_rows(ctx, ks_va[::2], 0, need_label=False), *tgt)
    ranker, info = train_ranker(model, tr, va, ctx.S, args)
    sv = D.score_matrix(ctx, ranker.predict, ks_va)
    sho = D.score_matrix(ctx, ranker.predict, ks_ho)
    icv = D.ic_summary(D.ic_matrix(sv, tgt[0], ks_va), h)
    # the holdout labels reach h hours past the window: restrict the IC to decisions whose outcome is inside the data
    ic_ho = D.ic_summary(D.ic_matrix(sho, tgt[0], ks_ho), h)
    cfg = sim_cfg(cB["k"], cB["R"], cB["entry"], args)
    k0, k1 = int(ks_ho[0]), int(ks_ho[-1]) + 1
    ev = eval_window(ctx, sho, cfg, k0, k1, args.n_random_holdout, mom, rev)
    hold = summarize_pooled([ev])
    hold["ic"] = ic_ho
    hold["final_val_ic"] = icv
    hold["train_info"] = {kk: vv for kk, vv in info.items() if kk != "curve"}
    hold["ranges"] = {k: [iso(v[0]), iso(v[1])] for k, v in fin.items()}
    hold["model_run"] = {kk: vv for kk, vv in ev["model"].items() if kk != "daily"}
    hold["trades_detail"] = ev["model"]
    report["holdout"] = hold
    log(f"HOLDOUT: model {hold['model']['net_return_pct']:+.2f}% Sharpe {hold['model']['sharpe']:+.2f} | momentum "
        f"{hold['momentum']['net_return_pct']:+.2f}% | random {hold['random']['mean_return_pct']:+.2f}% +- {hold['random']['std_return_pct']:.2f} "
        f"(z {hold['z_vs_random']:+.2f})")
    return report, (ranker, model, h, cB, info, hold, evalsB)


def verdict(report: dict) -> dict:
    """Does the chosen variant beat BOTH baselines out of sample?  (conservative, all must hold)"""
    po = report["oos_global_choice"]["pooled"]
    ho = report.get("holdout")
    oos_ok = (po["model"]["net_return_pct"] > 0 and po["z_vs_random"] >= 2.0
              and po["model"]["net_return_pct"] > po["momentum"]["net_return_pct"])
    ho_ok = None
    if ho:
        ho_ok = (ho["model"]["net_return_pct"] > 0 and ho["model"]["net_return_pct"] > ho["random"]["mean_return_pct"]
                 and ho["model"]["net_return_pct"] > ho["momentum"]["net_return_pct"])
    return {"oos_beats_random_and_momentum": bool(oos_ok), "holdout_beats_random_and_momentum": ho_ok,
            "edge": bool(oos_ok and ho_ok)}


def text_report(report: dict, v: dict) -> str:
    L = []
    t = report["trials"]
    L.append(f"V3 cross-sectional ranking (A) + maker execution (D) - run {report['tag']}")
    L.append(f"multiple testing: {t['val_backtests']} validation backtests ({t['folds']} folds x {t['model_horizon_candidates']} "
             f"model/horizon candidates x {t['exec_configs_per_candidate']} execution configs) and {t['val_ic_comparisons']} val-IC comparisons")
    g = report["global_choice"]
    L.append(f"global choice (validation only): {g['model_key']}  top_k=bottom_k={g['k']}  rebalance every {g['R']} h  entry={g['entry']}  "
             f"(mean val IC {g['mean_val_ic']:+.4f}, mean val net return {g['val_mean_return_pct']:+.1f}%)")
    L.append("")
    L.append("rank-IC (Spearman, mean per hour; val / test per fold):")
    for k in report["fold_val_ic"]:
        v_ = report["fold_val_ic"][k]
        te = report["fold_test_ic"][k]
        L.append(f"  {k:<9} val {np.nanmean([x if x is not None else np.nan for x in v_]):+.4f}  test "
                 f"{np.nanmean([x if x is not None else np.nan for x in te]):+.4f}   test per fold "
                 + " ".join(f"{(x if x is not None else float('nan')):+.3f}" for x in te))
    for label, key in (("OUT-OF-SAMPLE, global config on every fold's test window", "oos_global_choice"),
                       ("OUT-OF-SAMPLE, per-fold validation choice", "oos_per_fold_choice")):
        p = report[key]["pooled"]
        L.append("")
        L.append(f"{label} (pooled over {p['model']['days']} days):")
        L.append(fmt_row("model", p["model"]))
        L.append(fmt_row("momentum 7d rank", p["momentum"]))
        L.append(fmt_row("reversal 7d (info)", p["reversal"]))
        r = p["random"]
        L.append(f"  random rankings (n={r['n']}): net mean {r['mean_return_pct']:+.2f}%  sd {r['std_return_pct']:.2f}  p95 {r['p95_return_pct']:+.2f}%  "
                 f"-> model z = {p['z_vs_random']:+.2f}, beats {100 * p['model_rank_in_random']:.0f}% of random runs")
        if key == "oos_global_choice":
            ec = report[key]["entry_mode_contrast"]
            L.append("  entry-mode contrast on the same scores: " + "; ".join(
                f"{m}: {x['net_return_pct']:+.2f}% (Sharpe {x['sharpe']:+.2f})" for m, x in ec.items()))
    ho = report.get("holdout")
    L.append("")
    if ho:
        L.append(f"LOCKED HOLDOUT {ho['ranges']['holdout'][0]}..{ho['ranges']['holdout'][1]} (evaluated once):")
        L.append(fmt_row("model", {**ho["model"], **{k: ho["model_run"].get(k, 0) for k in ()}}))
        L.append(fmt_row("momentum 7d rank", ho["momentum"]))
        L.append(fmt_row("reversal 7d (info)", ho["reversal"]))
        r = ho["random"]
        L.append(f"  random rankings (n={r['n']}): net mean {r['mean_return_pct']:+.2f}%  sd {r['std_return_pct']:.2f}  p95 {r['p95_return_pct']:+.2f}%  "
                 f"-> model z = {ho['z_vs_random']:+.2f}")
        L.append(f"  holdout rank-IC {ho['ic']['ic']:+.4f} (t {ho['ic']['t']:+.1f}); model run: trades {ho['model_run']['n_trades']}, "
                 f"trades/day {ho['model_run']['trades_per_day']:.1f}, fees ${ho['model_run']['fees_paid']:.2f}, "
                 f"turnover {ho['model_run']['turnover_x_per_day']:.2f}x/day, fill rate {ho['model_run']['entry_fill_rate']:.2f}")
    L.append("")
    L.append(f"VERDICT: OOS beats random AND momentum: {v['oos_beats_random_and_momentum']}; holdout: {v['holdout_beats_random_and_momentum']}; "
             f"edge: {v['edge']}" + ("" if v["edge"] else "  -> exported as PAPER TEST ONLY - no edge"))
    return "\n".join(L)


# --------------------------------------------------------------------------------------------
def export_model(final, report, v, args, models_dir: Path) -> Path | None:
    ranker, model, h, cB, info, hold, _ = final
    stamp = time.strftime("%Y%m%d_%H%M")
    name = f"xsec_{model}_h{h}_k{cB['k']}_r{cB['R']}_{cB['entry']}_{stamp}"
    po = report["oos_global_choice"]["pooled"]
    metrics = {"oos_pooled": {"model": po["model"], "momentum": po["momentum"], "random": po["random"],
                              "z_vs_random": po["z_vs_random"]},
               "holdout": {"model": hold["model"], "momentum": hold["momentum"], "random": hold["random"],
                           "z_vs_random": hold["z_vs_random"], "ic": hold["ic"]},
               "trials": report["trials"], "verdict": v, "tag": report["tag"],
               "fold_test_ic": report["fold_test_ic"].get(f"{model}_h{h}")}
    note = None if v["edge"] else ("PAPER TEST ONLY - no edge: the cross-sectional ranker did not beat BOTH the random-ranking "
                                   "and the 7d-momentum baselines out of sample (see metrics); shadow on testnet only, never live")
    meta = mx.make_meta(name, model, cB["R"], cB["k"], cB["k"], h, cB["entry"], metrics=metrics, note=note,
                        timeout_bars=3, reprices=2, extra={"stride": args.stride, "through_pct": args.through_pct,
                                                           "train_range": hold["ranges"]["train"]})
    meta["entry_cfg"]["maker_through_pct"] = args.through_pct
    meta["exit_cfg"].update({"stop_vol_mult": args.stop_vol_mult})
    base = models_dir / name
    if model == "mlp":
        mx.save_mlp(base, meta, ranker)
    else:
        mx.save_gbm(base, meta, ranker)
    # round trip: the file must load and rank
    sc = mx.load(str(base) + ".json")
    Xr = np.random.default_rng(0).normal(size=(20, fx.F_MODEL)).astype("float32")
    assert np.allclose(sc.ranker.predict(Xr), ranker.predict(Xr), atol=1e-5)
    return Path(str(base) + ".json")


def lab_report(report, v, out_dir: Path, tag: str, evalsB, hold):
    import hashlib
    eq, t0, level = [], None, 100.0
    for ev in evalsB:
        d = np.asarray(ev["model"]["daily"], dtype="float64")
        for x in d:
            level *= (1.0 + x)
            eq.append(level)
    now = int(time.time())
    n = len(eq)
    series = [{"t": now - (n - i) * 86400, "equity": round(float(x), 4)} for i, x in enumerate(eq)]
    po = report["oos_global_choice"]["pooled"]
    rid = "xsec-" + hashlib.sha1(f"{tag}{now}".encode()).hexdigest()[:8]
    g = report["global_choice"]
    summary = {"title": f"xsec {g['model_key']} k={g['k']} R={g['R']}h {g['entry']}", "trials": report["trials"],
               "oos_pooled": {k: po[k] for k in ("model", "momentum", "reversal", "random", "z_vs_random")},
               "holdout": None if not hold else {k: hold[k] for k in ("model", "momentum", "reversal", "random", "z_vs_random", "ic")},
               "verdict": v}
    rep = {"id": rid, "created_at": now, "kind": "backtest", "title": f"V3 xsec ranking ({g['model_key']}) walk-forward {tag}",
           "summary": clean(summary), "equity": series}
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{rid}.json"
    p.write_text(json.dumps(rep, indent=1))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--test-days", type=int, default=45)
    ap.add_argument("--val-days", type=int, default=28)
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--purge-h", type=int, default=96)
    ap.add_argument("--horizons", default="12,24,48,72")
    ap.add_argument("--ks", default="3,5,10")
    ap.add_argument("--models", default="gbm,mlp")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--n-random-holdout", type=int, default=100)
    ap.add_argument("--through-pct", type=float, default=0.0001)
    ap.add_argument("--kill-dd", type=float, default=50.0)
    ap.add_argument("--open-risk", type=float, default=20.0)
    ap.add_argument("--stop-vol-mult", type=float, default=2.0)
    ap.add_argument("--gbm-iter", type=int, default=240)
    ap.add_argument("--mlp-seeds", type=int, default=3)
    ap.add_argument("--mlp-epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-final", action="store_true", help="never touch the holdout (smoke tests)")
    ap.add_argument("--drop-last-days", type=int, default=0,
                    help="dry runs: build the splits as if the data ended this many days earlier (the real holdout stays untouched)")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    a.horizons = [int(x) for x in a.horizons.split(",")]
    a.ks = [int(x) for x in a.ks.split(",")]
    a.models = a.models.split(",")
    out = D.var_dir() / "ml" / f"v3_xsec_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"args {vars(a)}")
    ctx = D.get_context(symbols=[s for s in a.symbols.split(",") if s] or None, log=log)
    log(f"panel: {ctx.S} symbols x {ctx.T} hours ({iso(ctx.panel.t0)} .. {iso(int(ctx.dec_ms[-1]))}); tradable universe "
        f"min/median/max = {ctx.mask.sum(1)[ctx.mask.sum(1) > 0].min()}/{int(np.median(ctx.mask.sum(1)[ctx.mask.sum(1) > 0]))}/{ctx.mask.sum(1).max()}")
    dec = ctx.dec_ms if not a.drop_last_days else ctx.dec_ms[ctx.dec_ms < ctx.dec_ms[-1] - a.drop_last_days * 86_400_000]
    sp = make_splits(dec, n_folds=a.folds, test_days=a.test_days, val_days=a.val_days, holdout_days=a.holdout_days,
                     purge_bars=a.purge_h * 12, min_train_days=120)
    if len(sp["folds"]) < a.folds:
        log(f"WARNING: only {len(sp['folds'])} folds fit")
    (out / "splits.json").write_text(json.dumps(clean({"folds": [{k: [iso(x) for x in v] if isinstance(v, tuple) else v
                                                                   for k, v in f.items()} for f in sp["folds"]],
                                                       "final": {k: [iso(x) for x in v] for k, v in sp["final"].items()}})))
    recs = [phase1_fold(ctx, sp, i, a, out) for i in range(len(sp["folds"]))]
    report, final = phase2_and_3(ctx, sp, recs, a, out, a.tag)
    v = verdict(report)
    report["verdict"] = v
    txt = text_report(report, v)
    (out.parent / f"v3_xsec_{a.tag}_report.txt").write_text(txt)
    (out.parent / f"v3_xsec_{a.tag}.json").write_text(json.dumps(clean(report)))
    exported = None
    if final is not None and not a.no_export:
        mdir = Path(os.environ.get("TB_MODELS") or D.ROOT / "models")
        exported = export_model(final, report, v, a, mdir)
        lp = lab_report(report, v, D.var_dir() / "lab", a.tag, final[6], final[5])
        txt += f"\n\nexported: {exported}\nlab report: {lp}"
        (out.parent / f"v3_xsec_{a.tag}_report.txt").write_text(txt)
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
