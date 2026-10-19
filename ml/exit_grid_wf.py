"""Walk-forward + ONE holdout evaluation + export of the V3 GBM over the dollar-floored exit grid (ml/exit_grid.py).

Re-uses the existing protocol of ml/mtf_walkforward.py unchanged (purged expanding folds, validation-only calibration /
thresholds, random-entry and ATR-regime baselines under identical exits / rules, locked holdout) and changes only:

  * the exit set (8 variants of TP:SL ratio / sizing, every TP and SL >= $2 net of fees);
  * the multiple-testing correction: qualification needs min(z, day-clustered z) vs random >= Bonferroni z over ALL
    variant x side tests (8 x 2 = 16, +4 when the cost-filter flag variants are added);
  * symmetric sides: the same top-fraction threshold search on the validation block for long and short, no priors;
  * an optional cost filter: rows whose round-trip cost exceeds ``cf`` x the stop distance are never candidates.

    export PYTHONPATH=backend:ml
    OMP_NUM_THREADS=2 nice -n 10 .venv-ml/bin/python ml/exit_grid_wf.py wf --only atr_1to2,atr_1to1,...
    OMP_NUM_THREADS=2 nice -n 10 .venv-ml/bin/python ml/exit_grid_wf.py wfcf --exit atr_2to1 --cf 0.10,0.15
    nice -n 10 .venv-ml/bin/python ml/exit_grid_wf.py select          # table + chosen exit(s); NO holdout
    nice -n 10 .venv-ml/bin/python ml/exit_grid_wf.py finish --top 2  # ONE holdout each + export (challenger)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

import dataset as D  # noqa: E402
import exit_grid as EG  # noqa: E402
import mtf_eval as ME  # noqa: E402
import mtf_walkforward as W  # noqa: E402
from mtf_dataset import MtfDataset  # noqa: E402
from splits import iso, make_splits  # noqa: E402

SIDES = W.SIDES
ROOT = Path(__file__).resolve().parents[1]
REF_MODEL = ROOT / "models" / "v3_gbm_pct36_kal_20261019_0607.json"
OUT = D.var_dir() / "ml" / "exit_grid"
MIN_TRADES = W.MIN_TRADES


def log(*a):
    print(*a, flush=True)


def ref_entry_cfg() -> dict:
    """entry_cfg of the current v3 model (adaptive decision cadence + ev_* fields), copied verbatim."""
    return json.loads(REF_MODEL.read_text())["entry_cfg"]


def variant(exit_name: str) -> dict:
    return dict(name=f"gbm_{exit_name}_kal", model="gbm", exit=exit_name, kalman=True)


def base_variants() -> list[dict]:
    return [variant(e) for e in EG.GRID]


def bonf(n_tests: int) -> float:
    return float(statistics.NormalDist().inv_cdf(1.0 - 0.05 / max(n_tests, 1)))


def with_cost_filter(ds: MtfDataset, exit_name: str, cf: float, tag: str) -> str:
    """Add keys ``<tag>_{long,short}`` = the exit's labels with rows masked out where RT cost > cf * stop distance.
    Mutates ``ds`` (lab / pnl / off become in-memory arrays with the extra keys).  Returns the new exit name."""
    if f"{tag}_long" in ds.meta["keys"]:
        return tag
    ei = list(EG.GRID).index(exit_name)
    sl = np.asarray(ds.sl[ei]).astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = (EG.RT_COST_FRAC / sl) <= cf
    keep &= np.isfinite(sl)
    new_lab, new_pnl, new_off = [], [], []
    for s in SIDES:
        k = ds.key_idx(exit_name, s)
        lab = np.asarray(ds.lab[k]).copy()
        lab[~keep] = -1
        new_lab.append(lab)
        new_pnl.append(np.asarray(ds.pnl[k]))
        new_off.append(np.asarray(ds.off[k]))
    ds.lab = np.concatenate([np.asarray(ds.lab), np.stack(new_lab)])
    ds.pnl = np.concatenate([np.asarray(ds.pnl), np.stack(new_pnl)])
    ds.off = np.concatenate([np.asarray(ds.off), np.stack(new_off)])
    ds.meta["keys"] = list(ds.meta["keys"]) + [f"{tag}_{s}" for s in SIDES]
    ds.meta["exit_cfgs"][tag] = dict(ds.meta["exit_cfgs"][exit_name], max_cost_frac_of_sl=cf, base_exit=exit_name)
    valid = np.isfinite(sl)
    log(f"[cost filter] {tag}: cost {100 * EG.RT_COST_FRAC:.3f} % of price; drops {100 * (1 - keep[valid].mean()):.2f} % of rows "
        f"(stop <= {100 * EG.RT_COST_FRAC / cf:.2f} % of price)")
    return tag


def setup(a):
    W.ENTRY_CFG = ref_entry_cfg()
    ds = EG.load_dataset(log=log)
    return ds


def make_splits_for(ds, a):
    return W.make_all_splits(ds, a)


# --------------------------------------------------------------------------------------------
def qualify(p: dict, z_min: float) -> tuple[bool, str]:
    """A side qualifies when n >= 100, expectancy > 0, min(z, day-clustered z) vs random >= z_min and it beats the best
    ATR-regime rule (both baselines)."""
    why = []
    if p["n"] < MIN_TRADES:
        why.append(f"n={p['n']}<{MIN_TRADES}")
    z = [x for x in (p.get("z_vs_random"), p.get("z_vs_random_clustered")) if x is not None]
    zm = min(z) if len(z) == 2 else None
    if zm is None or zm < z_min:
        why.append(f"min(z,zc)={None if zm is None else round(zm, 2)}<{z_min:.2f}")
    if p["expectancy"] is None or p["expectancy"] <= 0:
        why.append("expectancy<=0")
    if not p["beats_rule"]:
        why.append("not better than the ATR rule")
    return (not why), "; ".join(why)


def joint(r: dict) -> dict:
    """Both sides pooled (one exit_cfg serves both sides): excess over the random baseline in $ and its z."""
    num, var, n, exp = 0.0, 0.0, 0, 0.0
    for s in SIDES:
        p = r[f"pooled_{s}"]
        if p["n"] and p["se"] and p["baseline_expectancy"] is not None:
            num += p["n"] * (p["expectancy"] - p["baseline_expectancy"])
            var += (p["n"] * p["se"]) ** 2
            n += p["n"]
            exp += p["n"] * p["expectancy"]
    return dict(n=n, excess_per_trade=num / n if n else None, expectancy=exp / n if n else None,
                z=num / np.sqrt(var) if var > 0 else None)


def load_results(names: list[str] | None = None) -> dict:
    out = {}
    for p in sorted(OUT.glob("v3_mtf_wf_*.json")):
        if p.name.endswith(".partial.json"):
            continue
        r = json.loads(p.read_text())
        out[r["variant"]["name"]] = r
    return out


def n_tests_of(results: dict) -> int:
    return len(results) * len(SIDES)


def table(results: dict, z_min: float) -> str:
    L = [f"{'variant':26s} {'side':5s} {'n':>5s} {'/day':>5s} {'win':>5s} {'exp$':>6s} {'rand$':>6s} {'z':>5s} {'zclu':>5s} "
         f"{'atr1h':>6s} {'atr5m':>6s} {'AUC':>5s} {'BE':>5s} Q"]
    for name, r in sorted(results.items()):
        for s in SIDES:
            p = r[f"pooled_{s}"]
            q, _ = qualify(p, z_min)
            L.append(f"{name:26s} {s:5s} {p['n']:5d} {p['trades_per_day']:5.2f} {W.fmt(p['win_rate'], 2):>5s} "
                     f"{W.fmt(p['expectancy'], 2):>6s} {W.fmt(p['baseline_expectancy'], 2):>6s} {W.fmt(p['z_vs_random'], 2):>5s} "
                     f"{W.fmt(p['z_vs_random_clustered'], 2):>5s} {W.fmt(p['rules']['atr1h']['expectancy'], 2):>6s} "
                     f"{W.fmt(p['rules']['atr5m']['expectancy'], 2):>6s} {p['auc']:5.3f} {W.fmt(p['breakeven_win_rate'], 2):>5s} "
                     f"{'YES' if q else 'no'}")
        j = joint(r)
        L.append(f"{'':26s} both  {j['n']:5d}   exp {W.fmt(j['expectancy'], 2)}  excess/trade {W.fmt(j['excess_per_trade'], 2)}  "
                 f"joint z {W.fmt(j['z'], 2)}")
    return "\n".join(L)


def rank_exits(results: dict) -> list[str]:
    """Walk-forward (OOS on the dev period, never the holdout) ranking of the exit variants: joint z vs random of both
    sides pooled.  Only variants that traded >= MIN_TRADES in total are ranked."""
    cand = []
    for name, r in results.items():
        if r["variant"].get("cf"):
            continue
        j = joint(r)
        if j["z"] is not None and j["n"] >= MIN_TRADES:
            cand.append((j["z"], name))
    return [n for _, n in sorted(cand, reverse=True)]


# --------------------------------------------------------------------------------------------
def run_final(ds, v, splits, wf, models_dir, enabled, nt, z_min, log_):
    """ONE holdout evaluation + export.  ``enabled``: sides whose walk-forward result qualified."""
    fs = splits["final"]
    log_(f"=== FINAL {v['name']} (HOLDOUT {iso(fs['holdout'][0])}..{iso(fs['holdout'][1])}, evaluated once)")
    fitted, info = W.fit_split(ds, v, fs["train"], fs["val"], log_, refit_gbm=True)
    va = W.select_rows(ds, fs["val"], v["exit"], 1)
    calinfo = W.calibrate_and_threshold(ds, v, fitted, fs["train"], fs["val"], va, W.read_x(ds, va, ds.cols(v["kalman"])), log_)
    res, arrs = W.evaluate_block(ds, v, fitted, calinfo, fs["holdout"], "HOLDOUT", log_, n_random=20)
    hold = json.loads(json.dumps(res, default=float))
    edge_sides = []
    for s in enabled:
        h = res[s]["chosen"]
        rules = [res[s]["rules"][k]["expectancy"] for k in W.RULE_FEATURES]
        beats_rules = all(x is None or (h["expectancy"] is not None and h["expectancy"] > x) for x in rules)
        if (h["n"] >= 20 and h["expectancy"] is not None and h["expectancy"] > 0 and h["baseline_expectancy"] is not None
                and h["expectancy"] > h["baseline_expectancy"] and beats_rules):
            edge_sides.append(s)
    txt = "; ".join(
        f"{s}: WF n={wf[f'pooled_{s}']['n']} exp={W.fmt(wf[f'pooled_{s}']['expectancy'], 2)} (random "
        f"{W.fmt(wf[f'pooled_{s}']['baseline_expectancy'], 2)}, z={W.fmt(wf[f'pooled_{s}']['z_vs_random'], 2)}, "
        f"zc={W.fmt(wf[f'pooled_{s}']['z_vs_random_clustered'], 2)}), holdout n={res[s]['chosen']['n']} "
        f"exp={W.fmt(res[s]['chosen']['expectancy'], 2)} (random {W.fmt(res[s]['chosen']['baseline_expectancy'], 2)}, "
        f"atr1h {W.fmt(res[s]['rules']['atr1h']['expectancy'], 2)}, atr5m {W.fmt(res[s]['rules']['atr5m']['expectancy'], 2)})"
        for s in SIDES)
    tail = (f"exit {v['exit']} chosen on walk-forward OOS from {nt // 2} exit variants; {nt} variant x side tests, "
            f"Bonferroni z={z_min:.2f}; symmetric validation-chosen thresholds. {txt}")
    if edge_sides:
        sides = edge_sides
        note = (f"Beat the random and ATR-regime baselines out of sample (walk-forward min(z,zc)>=Bonferroni and the one-shot "
                f"holdout) for {edge_sides}; the other side is disabled - shadow before trusting. {tail}")
    else:
        sides = list(SIDES)
        why = ("no side beat both baselines on walk-forward" if not enabled else
               f"beat both baselines in walk-forward for {enabled} but NOT on the one-shot holdout")
        note = f"{W.PAPER_NOTE} ({why}). {tail}"
    name = W.export_variant(ds, v, calinfo, fitted, hold, W.strip(wf), models_dir, "challenger", note, sides)
    # the ev_* adaptive entry fields (W.ENTRY_CFG was copied from the reference model) are in entry_cfg already
    mp = models_dir / f"{name}.json"
    meta = json.loads(mp.read_text())
    meta["entry_cfg"] = dict(W.ENTRY_CFG, **({"max_cost_frac_of_sl": ds.meta["exit_cfgs"][v["exit"]]["max_cost_frac_of_sl"]}
                                             if "max_cost_frac_of_sl" in ds.meta["exit_cfgs"][v["exit"]] else {}))
    mp.write_text(json.dumps(meta, indent=1, default=float))
    log_(f"exported models/{name}  edge_sides={edge_sides}  note: {note[:200]}")
    return name, res, arrs, calinfo, note, edge_sides


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["wf", "wfcf", "select", "finish"])
    ap.add_argument("--only", default="")
    ap.add_argument("--exit", default="")
    ap.add_argument("--cf", default="0.10,0.15")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-days", type=int, default=45)
    ap.add_argument("--val-days", type=int, default=28)
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--min-train-days", type=int, default=90)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models-dir", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    models_dir = Path(a.models_dir or ROOT / "models")

    if a.cmd == "select":
        res = load_results()
        nt = n_tests_of(res)
        z = bonf(max(nt, 2 * len(EG.GRID)))
        print(f"{len(res)} variants, {nt} tests, Bonferroni z={z:.2f}")
        print(table(res, z))
        print("ranking (joint z vs random, WF OOS):", rank_exits(res))
        return

    ds = setup(a)
    W.variants = base_variants
    W.RULES = dict(ME.DEFAULT_RULES)
    splits = make_splits_for(ds, a)
    log("splits: " + json.dumps({"folds": [{k: [iso(x) for x in vv] if isinstance(vv, tuple) else vv for k, vv in f.items()}
                                           for f in splits["folds"]],
                                 "final": {k: [iso(x) for x in vv] for k, vv in splits["final"].items()}}))

    if a.cmd == "wf":
        only = [x for x in a.only.split(",") if x]
        vs = [v for v in base_variants() if not only or v["exit"] in only]
        for v in vs:
            W.run_wf_variant(ds, v, splits, OUT, log, force=a.force, seeds=a.seeds)
        return

    if a.cmd == "wfcf":
        for cf in [float(x) for x in a.cf.split(",") if x]:
            tag = f"{a.exit}_cf{int(round(cf * 100))}"
            with_cost_filter(ds, a.exit, cf, tag)
            v = dict(variant(tag), cf=cf, base_exit=a.exit)
            W.variants = lambda: base_variants() + [v]
            W.run_wf_variant(ds, v, splits, OUT, log, force=a.force, seeds=a.seeds)
        return

    # ---- finish: ONE holdout evaluation per exported variant ----
    res = load_results()
    nt = n_tests_of(res)
    z_min = bonf(nt)
    order = rank_exits(res)[:a.top]
    log(f"results for {len(res)} variants ({nt} tests, Bonferroni z={z_min:.2f}); exporting top {a.top}: {order}")
    exported = []
    for nm in order:
        r = res[nm]
        v = r["variant"]
        W.variants = base_variants
        enabled = [s for s in SIDES if qualify(r[f"pooled_{s}"], z_min)[0]]
        W.PAPER_NOTE = W.PAPER_NOTE
        name, hres, harrs, calinfo, note, edge = run_final(ds, v, splits, r, models_dir, enabled, nt, z_min, log)
        (OUT / f"final_{name}.json").write_text(json.dumps(
            dict(holdout=hres, thresholds=calinfo["thresholds"], val=calinfo["val"], note=note, wf_qualified_sides=enabled,
                 edge_sides=edge, n_tests=nt, bonferroni_z=z_min), indent=1, default=float))
        exported.append(dict(name=name, variant=nm, wf_qualified=enabled, edge_sides=edge, note=note[:160]))
        W.lab_report(name, f"Train {name} (gbm, {v['exit']} exits with $2 floor)",
                     dict(variant=nm, model=name, note=note), harrs[SIDES[0]]["chosen"], D.var_dir())
    (OUT / "exported.json").write_text(json.dumps(exported, indent=1))
    log(json.dumps(exported, indent=1))


if __name__ == "__main__":
    main()
