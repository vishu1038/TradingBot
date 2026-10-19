"""Walk-forward + final evaluation + export for the multi-timeframe experiments (B: mtf net, C: gradient-boosted
trees, E: Kalman features).  Variants = {mlp, gbm} x {pct36, atr1h32} x {kalman off, on}.

    PYTHONPATH=backend nice -n 10 .venv-ml/bin/python ml/mtf_walkforward.py build                 # dataset (CPU)
    ... mtf_walkforward.py wf --models gbm                       # CPU
    ... mtf_walkforward.py wf --models mlp                       # GPU (mps)
    ... mtf_walkforward.py finish                                # qualify, ONE holdout evaluation each, export, report
    ... mtf_walkforward.py smoke                                 # synthetic end-to-end check

Protocol (the same as ml/walkforward.py, applied to decision-cadence rows)
  * ``|train|purge|val|purge|test|`` expanding folds, purge >= the 4-day label horizon, locked holdout (last 60 days)
    that only ``finish`` touches, ONCE per exported variant;
  * normalisation (median / MAD), early stopping, Platt calibration and the score threshold (top-fraction grid) use
    train / validation data only;
  * trades obey the engine rules + a per-coin cooldown + a trades-per-day cap (``mtf_eval.DEFAULT_RULES``);
  * BOTH baselines under identical exits / rules / costs: random entries (same number of candidates) and the
    ATR%-regime rule (1h-ATR% and 5m-ATR%, threshold chosen on validation the same way);
  * the number of variants / tests is reported; a Bonferroni z is printed next to the raw z.
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
import mtf_dataset as MD  # noqa: E402
import mtf_eval as ME  # noqa: E402
from splits import DAY_MS, iso, make_splits  # noqa: E402
from tb import features as ft  # noqa: E402
from tb import features_mtf as fm  # noqa: E402
from tb import model_np as mnp  # noqa: E402
from tb.labels import breakeven_win_rate  # noqa: E402

SIDES = MD.SIDES
MODELS = ("mlp", "gbm")
PURGE_BARS = MD.HORIZON + 12
RULES = dict(ME.DEFAULT_RULES)
ENTRY_CFG = {"decision_every_bars": MD.DECISION_EVERY, "cooldown_bars": RULES["cooldown_bars"],
             "max_trades_per_day": RULES["max_per_day"], "max_open": RULES["max_pos"], "entry_mode": "taker",
             "maker_timeout_s": None, "maker_reprices": None}
PAPER_NOTE = "PAPER TEST ONLY - no edge"
MIN_TRADES = 100                      # pooled out-of-sample trades needed before a side can qualify
STRIDE = {"mlp": 2, "gbm": 6}         # train on every n-th decision hour (labels overlap for days anyway)
GBM_PARAMS = {"max_iter": 250}
RULE_FEATURES = {"atr1h": "h1_atr_pct", "atr5m": "m5_atr_pct"}


def log(*a):
    print(*a, flush=True)


def variants() -> list[dict]:
    out = []
    for model in MODELS:
        for exit_name in MD.EXITS:
            for kal in (0, 1):
                out.append(dict(name=f"{model}_{exit_name}_{'kal' if kal else 'nokal'}", model=model, exit=exit_name,
                                kalman=bool(kal)))
    return out


# --------------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------------
def select_rows(ds, rng_ms, exit_name, stride: int = 1, block: int = 2_000_000) -> np.ndarray:
    lo, hi = rng_ms
    out = []
    kl, ks = ds.key_idx(exit_name, "long"), ds.key_idx(exit_name, "short")
    for s in range(0, len(ds), block):
        e = min(s + block, len(ds))
        t = np.asarray(ds.t[s:e])
        m = (t >= lo) & (t < hi)
        if stride > 1:
            m &= (((t + 300_000) // (MD.DECISION_EVERY * 300_000)) % stride) == 0
        idx = np.nonzero(m)[0] + s
        if len(idx):
            ok = (np.asarray(ds.lab[kl][idx]) >= 0) | (np.asarray(ds.lab[ks][idx]) >= 0)
            idx = idx[ok]
        out.append(idx)
    return np.concatenate(out) if out else np.zeros(0, dtype="int64")


def read_x(ds, rows, cols) -> np.ndarray:
    return np.asarray(ds.X[rows])[:, cols].astype("float32")


def targets(ds, exit_name, rows):
    Y = np.stack([ds.y(exit_name, s, rows) for s in SIDES], axis=1)
    Mk = np.isfinite(Y).astype("float32")
    return np.nan_to_num(Y, nan=0.0).astype("float32"), Mk


# --------------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------------
class Fitted:
    """A trained model that maps raw feature rows to raw [long, short] logits."""

    def __init__(self, model: str):
        self.model = model
        self.nets: list = []
        self.norm: dict | None = None
        self.clfs: dict = {}
        self.best_iter: dict = {}
        self.info: dict = {}

    def logits(self, X: np.ndarray) -> np.ndarray:
        if self.model == "mlp":
            import mtf_trainer as T
            Xn = T.normalise_np(X, self.norm["mean"], self.norm["std"])
            return sum(T.predict_logits(n, Xn)[:, :2] for n in self.nets) / len(self.nets)
        out = []
        for side in SIDES:
            clf, bi = self.clfs[side], self.best_iter[side]
            got = None
            for k, pr in enumerate(clf.staged_predict_proba(X), 1):
                if k == bi:
                    got = pr[:, 1]
                    break
            out.append(mnp.logit(got))
        return np.stack(out, axis=1)


def _gbm_fit_side(Xtr, ytr, Xva, yva, params, max_iter=None):
    from tb.model_gbm import make_estimator
    p = dict(GBM_PARAMS, **(params or {}))
    if max_iter is not None:
        p["max_iter"] = int(max_iter)
    clf = make_estimator(p)
    clf.fit(Xtr, ytr)
    best_ll, best_i = np.inf, 1
    for k, pr in enumerate(clf.staged_predict_proba(Xva), 1):
        ll = E.logloss(yva, pr[:, 1])
        if ll < best_ll - 1e-6:
            best_ll, best_i = ll, k
    return clf, best_i, best_ll


def fit_split(ds, v: dict, tr_rng, va_rng, log_, refit_gbm: bool = False, seeds: int = 3, params: dict | None = None):
    """Train on ``tr_rng``, early-stop / select on ``va_rng``.  Returns (Fitted, info, rows dict)."""
    cols = ds.cols(v["kalman"])
    tr = select_rows(ds, tr_rng, v["exit"], STRIDE[v["model"]])
    va = select_rows(ds, va_rng, v["exit"], 1)
    Xtr, Xva = read_x(ds, tr, cols), read_x(ds, va, cols)
    Ytr, Mtr = targets(ds, v["exit"], tr)
    Yva, Mva = targets(ds, v["exit"], va)
    f = Fitted(v["model"])
    sub = np.linspace(0, len(Xtr) - 1, min(len(Xtr), 400_000)).astype("int64")
    norm = ft.robust_norm_stats(Xtr[sub])
    f.norm = norm
    info = dict(n_train=int(len(tr)), n_val=int(len(va)), n_features=int(len(cols)), norm=norm)
    if v["model"] == "mlp":
        import mtf_trainer as T
        Xtr_n = T.normalise_np(Xtr, norm["mean"], norm["std"])
        Xva_n = T.normalise_np(Xva, norm["mean"], norm["std"])
        del Xtr
        infos = []
        for k in range(seeds):
            net, inf = T.train_mlp(Xtr_n, Ytr, Mtr, Xva_n, Yva, Mva, T.MlpCfg(seed=k, **(params or {})), log_)
            f.nets.append(net)
            infos.append(inf)
        info.update(best_val_loss=float(np.mean([i["best_val_loss"] for i in infos])), arch_cfg=infos[0]["arch_cfg"],
                    steps_run=int(np.mean([i["steps_run"] for i in infos])), seconds=float(sum(i["seconds"] for i in infos)),
                    seeds=seeds)
    else:
        t0 = time.time()
        for si, side in enumerate(SIDES):
            mt, mv = Mtr[:, si] > 0, Mva[:, si] > 0
            clf, bi, ll = _gbm_fit_side(Xtr[mt], Ytr[mt, si], Xva[mv], Yva[mv, si], params)
            if refit_gbm:
                clf, _, _ = _gbm_fit_side(Xtr[mt], Ytr[mt, si], Xva[mv], Yva[mv, si], params, max_iter=bi)
                f.best_iter[side] = clf.n_iter_
            else:
                f.best_iter[side] = bi
            f.clfs[side] = clf
            log_(f"  [gbm/{v['name']}/{side}] best iter {bi} val logloss {ll:.4f} ({time.time() - t0:.0f}s)")
        info.update(best_iter=dict(f.best_iter), seconds=round(time.time() - t0, 1),
                    params=dict(GBM_PARAMS, **(params or {})))
    f.info = info
    return f, info


def block(ds, exit_name, side, rows):
    """Arrays over the rows with a valid label for this side."""
    y = ds.y(exit_name, side, rows)
    m = np.isfinite(y)
    r = rows[m]
    return dict(m=m, rows=r, y=y[m], pnl=ds.pnl_of(exit_name, side, r), t=np.asarray(ds.t[r]),
                sym=np.asarray(ds.sym[r]), off=ds.off_of(exit_name, side, r))


def calibrate_and_threshold(ds, v, fitted, tr_rng, va_rng, rows_va, X_va, log_):
    """Platt calibration + score threshold + ATR-rule thresholds, all from the VALIDATION block."""
    lg = fitted.logits(X_va)
    cal, thr, val, rules = {}, {}, {}, {}
    # the ATR-regime rules use the train distribution of a volatility feature as reference
    tr = select_rows(ds, tr_rng, v["exit"], 8)
    refs = {k: ME.cdf_scorer(ds.col(c, tr)) for k, c in RULE_FEATURES.items()}
    for si, side in enumerate(SIDES):
        b = block(ds, v["exit"], side, rows_va)
        cal[side] = mnp.fit_calibration(lg[b["m"], si], b["y"], "platt")
        p = mnp.apply_calibration(cal[side], lg[b["m"], si])
        th, pos, table = ME.choose_threshold(p, b["y"], b["pnl"], b["t"], b["sym"], b["off"], RULES)
        thr[side] = th
        val[side] = dict(threshold=th, expectancy_positive=pos, table=table, n=int(len(b["y"])))
        rules[side] = {}
        for k, c in RULE_FEATURES.items():
            ref, cdf = refs[k]
            s_val = cdf(ds.col(c, b["rows"]))
            q, rpos, rtable = ME.choose_threshold(s_val, b["y"], b["pnl"], b["t"], b["sym"], b["off"], RULES)
            rules[side][k] = dict(threshold=q, expectancy_positive=rpos, ref=ref[::max(1, len(ref) // 20000)].tolist())
    return dict(calibration=cal, thresholds=thr, val=val, rules=rules)


def evaluate_block(ds, v, fitted, calinfo, rng_ms, tag, log_, n_random=8):
    """Out-of-sample evaluation on a time block with the validation-chosen calibration / thresholds."""
    te = select_rows(ds, rng_ms, v["exit"], 1)
    cols = ds.cols(v["kalman"])
    lg = fitted.logits(read_x(ds, te, cols))
    res = {"n_rows": int(len(te)), "range": [iso(rng_ms[0]), iso(rng_ms[1])],
           "days": float((rng_ms[1] - rng_ms[0]) / DAY_MS)}
    arrays = {}
    for si, side in enumerate(SIDES):
        b = block(ds, v["exit"], side, te)
        p_raw = mnp.sigmoid(lg[b["m"], si])
        p = mnp.apply_calibration(calinfo["calibration"][side], lg[b["m"], si])
        th = calinfo["thresholds"][side]
        chosen, taken = ME.run_rule(p, th, b["y"], b["pnl"], b["t"], b["sym"], b["off"], RULES, n_random=n_random)
        rule_res, rule_taken = {}, {}
        for k, c in RULE_FEATURES.items():
            ri = calinfo["rules"][side][k]
            ref = np.sort(np.asarray(ri["ref"], "float64"))
            s_te = np.searchsorted(ref, ds.col(c, b["rows"]).astype("float64"), side="right") / max(len(ref), 1)
            rr, rt = ME.run_rule(s_te, ri["threshold"], b["y"], b["pnl"], b["t"], b["sym"], b["off"], RULES,
                                 with_random=False)
            rule_res[k] = rr
            rule_taken[k] = rt
        be = breakeven_win_rate(dict(outcome=np.where(b["y"] == 1, 1, 0).astype("int8"), pnl_usd=b["pnl"]))
        res[side] = dict(auc=E.auc(b["y"], p_raw), base_rate=float(b["y"].mean()), breakeven_win_rate=be,
                         population_expectancy=float(b["pnl"].mean()), threshold=th, chosen=chosen, rules=rule_res,
                         n_signals=chosen["n_signals"])
        arrays[side] = dict(chosen=dict(pnl=b["pnl"][taken], y=b["y"][taken], t=b["t"][taken]),
                            **{k: dict(pnl=b["pnl"][rule_taken[k]], y=b["y"][rule_taken[k]], t=b["t"][rule_taken[k]])
                               for k in RULE_FEATURES})
        c = chosen
        log_(f"  [{tag}/{side}] AUC {res[side]['auc']:.3f} base {b['y'].mean():.3f} BE {be if be is not None else float('nan'):.3f} "
             f"thr {th:.3f}: n={c['n']} win={c['win_rate']} exp={c['expectancy']} (random {c['baseline_expectancy']}) | "
             + " ".join(f"{k}: n={rr['n']} exp={rr['expectancy']}" for k, rr in rule_res.items()))
    return res, arrays


# --------------------------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------------------------
def pool_side(folds: list[dict], arrs: list[dict], side: str) -> dict:
    ch = ME.pooled([a[side] for a in arrs], "chosen")
    days = sum(f["days"] for f in folds)
    n = ch["n"]
    wts = [(f[side]["chosen"]["n"], f[side]["chosen"]["baseline_expectancy"]) for f in folds
           if f[side]["chosen"]["baseline_expectancy"] is not None and f[side]["chosen"]["n"]]
    base = float(sum(w * b for w, b in wts) / sum(w for w, _ in wts)) if wts else None
    out = dict(n=n, win_rate=ch["win_rate"], expectancy=ch["expectancy"], se=ch["se"], total_pnl=ch["total_pnl"],
               trades_per_day=float(n / days) if days else 0.0, test_days=days, baseline_expectancy=base,
               auc=float(np.nanmean([f[side]["auc"] for f in folds])),
               breakeven_win_rate=float(np.nanmean([f[side]["breakeven_win_rate"] if f[side]["breakeven_win_rate"] is not None
                                                    else np.nan for f in folds])),
               population_expectancy=float(np.mean([f[side]["population_expectancy"] for f in folds])),
               base_rate=float(np.mean([f[side]["base_rate"] for f in folds])))
    if n > 1 and ch["se"] and base is not None:
        out["z_vs_random"] = float((ch["expectancy"] - base) / ch["se"])
        out["z_vs_random_clustered"] = ME.clustered_z(ch["_pnl"], ch["_t"], base)
        out["z_vs_zero"] = float(ch["expectancy"] / ch["se"])
    else:
        out["z_vs_random"] = out["z_vs_random_clustered"] = out["z_vs_zero"] = None
    rules = {}
    for k in RULE_FEATURES:
        r = ME.pooled([a[side] for a in arrs], k)
        rules[k] = dict(n=r["n"], win_rate=r["win_rate"], expectancy=r["expectancy"], se=r["se"],
                        trades_per_day=float(r["n"] / days) if days else 0.0)
    out["rules"] = rules
    best_rule = max((r["expectancy"] for r in rules.values() if r["expectancy"] is not None), default=None)
    out["best_rule_expectancy"] = best_rule
    out["beats_rule"] = bool(out["expectancy"] is not None and (best_rule is None or out["expectancy"] > best_rule))
    return out


def qualifies(p: dict, z_min: float = 2.0) -> tuple[bool, str]:
    """Beats BOTH baselines: z vs random >= 2, expectancy > 0, better than the (best) ATR rule, enough trades."""
    why = []
    if p["n"] < MIN_TRADES:
        why.append(f"n={p['n']}<{MIN_TRADES}")
    if p["z_vs_random"] is None or p["z_vs_random"] < z_min:
        why.append(f"z_vs_random={p['z_vs_random']}")
    if p["expectancy"] is None or p["expectancy"] <= 0:
        why.append("expectancy<=0")
    if not p["beats_rule"]:
        why.append("not better than the ATR rule")
    return (not why), "; ".join(why)


# --------------------------------------------------------------------------------------------
# walk-forward driver
# --------------------------------------------------------------------------------------------
def result_paths(out_dir: Path, name: str):
    return out_dir / f"v3_mtf_wf_{name}.json", out_dir / f"v3_mtf_wf_{name}.npz"


def run_wf_variant(ds, v, splits, out_dir: Path, log_, force=False, seeds=3):
    jp, zp = result_paths(out_dir, v["name"])
    if jp.exists() and not force:
        log_(f"=== {v['name']}: result exists, skipping ({jp.name})")
        return json.loads(jp.read_text())
    folds, arrs = [], []
    t0 = time.time()
    for f in splits["folds"]:
        log_(f"=== {v['name']} fold {f['fold']}: train<{iso(f['train'][1])} val {iso(f['val'][0])}..{iso(f['val'][1])} "
             f"test {iso(f['test'][0])}..{iso(f['test'][1])}")
        fitted, info = fit_split(ds, v, f["train"], f["val"], log_, seeds=seeds)
        va = select_rows(ds, f["val"], v["exit"], 1)
        calinfo = calibrate_and_threshold(ds, v, fitted, f["train"], f["val"], va, read_x(ds, va, ds.cols(v["kalman"])), log_)
        res, a = evaluate_block(ds, v, fitted, calinfo, f["test"], f"fold{f['fold']}", log_)
        res.update(fold=f["fold"], thresholds=calinfo["thresholds"], calibration=calinfo["calibration"],
                   val={s: calinfo["val"][s] for s in SIDES},
                   train={k: info[k] for k in info if k not in ("norm",)})
        folds.append(res)
        arrs.append(a)
        if v["model"] == "mlp":
            import mtf_trainer as T
            del fitted
            T.free_gpu()
        (out_dir / f"v3_mtf_wf_{v['name']}.partial.json").write_text(json.dumps(
            dict(variant=v, folds_done=len(folds), folds=folds), indent=1, default=float))
        log_(f"    fold done [{time.time() - t0:.0f}s]")
    out = dict(variant=v, n_variants=len(variants()), n_rows=int(len(ds)), n_symbols=len(ds.symbols), folds=folds,
               rules=RULES, entry_cfg=ENTRY_CFG, horizon_bars=ds.meta["horizon"], seconds=round(time.time() - t0, 1))
    for side in SIDES:
        out[f"pooled_{side}"] = pool_side(folds, arrs, side)
        q, why = qualifies(out[f"pooled_{side}"])
        out[f"pooled_{side}"]["qualifies"] = q
        out[f"pooled_{side}"]["why_not"] = why
    save = {}
    for si, side in enumerate(SIDES):
        for rule in ("chosen", *RULE_FEATURES):
            for key in ("pnl", "t", "y"):
                save[f"{side}_{rule}_{key}"] = np.concatenate([a[side][rule][key] for a in arrs])
    np.savez(zp, **save)
    jp.write_text(json.dumps(out, indent=1, default=float))
    (out_dir / f"v3_mtf_wf_{v['name']}.partial.json").unlink(missing_ok=True)
    for side in SIDES:
        p = out[f"pooled_{side}"]
        log_(f"### {v['name']} {side}: n={p['n']} ({p['trades_per_day']:.2f}/day) win={p['win_rate']} exp={p['expectancy']} "
             f"random={p['baseline_expectancy']} z={p['z_vs_random']} zc={p['z_vs_random_clustered']} "
             f"rules={ {k: r['expectancy'] for k, r in p['rules'].items()} } AUC={p['auc']:.3f} qualifies={p['qualifies']} "
             f"({p['why_not']})")
    return out


# --------------------------------------------------------------------------------------------
# final: ONE holdout evaluation per exported variant + export + report
# --------------------------------------------------------------------------------------------
def pick_export(results: dict[str, dict]) -> tuple[list[dict], bool, dict]:
    """(variants to export, fallback?, multiple-testing info).  Qualifying variants = at least one side beats BOTH
    baselines; otherwise the single best (highest z vs random with >= MIN_TRADES) in paper-only mode."""
    n_tests = len(variants()) * len(SIDES)               # every planned variant x side counts, finished or not
    mt = dict(n_variants=len(variants()), n_variants_finished=len(results), n_tests=n_tests,
              bonferroni_z=ME.bonferroni_z(n_tests))
    q = []
    for name, r in results.items():
        sides = [s for s in SIDES if r[f"pooled_{s}"]["qualifies"]]
        if sides:
            q.append(dict(r["variant"], sides=sides))
    if q:
        return q, False, mt
    best, bz = None, -1e9
    for name, r in results.items():
        for s in SIDES:
            p = r[f"pooled_{s}"]
            z = p["z_vs_random"]
            if p["n"] >= MIN_TRADES and z is not None and z > bz:
                best, bz = dict(r["variant"], sides=list(SIDES)), z
    if best is None:                                   # nothing traded enough: take the highest AUC
        key = max(results, key=lambda k: max(results[k][f"pooled_{s}"]["auc"] for s in SIDES))
        best = dict(results[key]["variant"], sides=list(SIDES))
    return [best], True, mt


def export_variant(ds, v, calinfo, fitted, holdout, wf, models_dir: Path, status: str, note: str, sides: list[str]):
    from tb import model_gbm
    now = datetime.now(timezone.utc)
    kind = "gbm" if v["model"] == "gbm" else "mtf"
    name = f"v3_{v['name']}_{now.strftime('%Y%m%d_%H%M')}"
    parent = None
    for p in sorted(models_dir.glob("*.json")):
        try:
            m = json.loads(p.read_text())
            if m.get("status") == "champion":
                parent = m.get("name")
        except Exception:
            pass
    groups = ["kalman"] if v["kalman"] else []
    tfs = ["5m", *MD.TFS]
    names = ds.col_names(v["kalman"])
    assert names == fm.feature_names(MD.TFS, tuple(groups))
    thr = {s: (calinfo["thresholds"][s] if s in sides else 1.01) for s in SIDES}
    med = fitted.norm["mean"]
    exit_cfg = dict(ds.meta["exit_cfgs"][v["exit"]], horizon_bars=int(ds.meta["horizon"]))
    meta = dict(
        name=name, version=now.strftime("%Y%m%d.%H%M"), created_at=now.isoformat(), status=status, parent=parent,
        kind=kind, arch="mlp" if kind == "mtf" else "hist_gbm", variant=v["name"], timeframes=tfs,
        feature_groups=groups, feature_names=names, window=1, history_bars_5m=fm.history_bars_5m(MD.TFS, tuple(groups)),
        norm=dict(mean=[float(x) for x in med], std=[float(x) for x in fitted.norm["std"]], clip=ft.NORM_CLIP),
        calibration=calinfo["calibration"], thresholds=thr, exit_cfg=exit_cfg, entry_cfg=dict(ENTRY_CFG),
        sides_enabled=sides, note=note,
        metrics=dict(walk_forward=wf, holdout=holdout, validation_thresholds=calinfo["val"],
                     train={k: v_ for k, v_ in fitted.info.items() if k not in ("norm",)},
                     dataset=dict(symbols=len(ds.symbols), rows=int(len(ds)), horizon=ds.meta["horizon"],
                                  decision_every_bars=ds.meta["decision_every_bars"])))
    if kind == "mtf":
        from models import export_state
        meta["arch_cfg"] = dict(fitted.info["arch_cfg"], n_out=2)
        meta["ensemble"] = len(fitted.nets)
        w = {}
        if len(fitted.nets) == 1:
            w = export_state(fitted.nets[0])
        else:
            for k, n in enumerate(fitted.nets):
                w.update({f"e{k}/{kk}": vv for kk, vv in export_state(n).items()})
        models_dir.mkdir(parents=True, exist_ok=True)
        tmp = models_dir / f".{name}.npz.tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, **w)
        os.replace(tmp, models_dir / f"{name}.npz")
        tj = models_dir / f".{name}.json.tmp"
        tj.write_text(json.dumps(meta, indent=1, default=float))
        os.replace(tj, models_dir / f"{name}.json")
    else:
        meta["arch_cfg"] = dict(fitted.info["params"])
        model_gbm.save(models_dir, name, meta, fitted.clfs)
    return name


def run_final(ds, v, splits, wf, models_dir, fallback, mt, log_, status="challenger"):
    fs = splits["final"]
    sides = v["sides"]
    log_(f"=== FINAL {v['name']} sides={sides}: train<{iso(fs['train'][1])} val {iso(fs['val'][0])}..{iso(fs['val'][1])} "
         f"HOLDOUT {iso(fs['holdout'][0])}..{iso(fs['holdout'][1])} (evaluated once)")
    fitted, info = fit_split(ds, v, fs["train"], fs["val"], log_, refit_gbm=True)
    va = select_rows(ds, fs["val"], v["exit"], 1)
    calinfo = calibrate_and_threshold(ds, v, fitted, fs["train"], fs["val"], va, read_x(ds, va, ds.cols(v["kalman"])), log_)
    res, arrs = evaluate_block(ds, v, fitted, calinfo, fs["holdout"], "HOLDOUT", log_, n_random=20)
    hold = json.loads(json.dumps(res, default=float))
    bad = []
    for s in sides:
        h = res[s]["chosen"]
        if h["n"] < 20 or h["expectancy"] is None or h["expectancy"] <= 0 or (
                h["baseline_expectancy"] is not None and h["expectancy"] <= h["baseline_expectancy"]):
            bad.append(s)
    pooled_txt = "; ".join(
        f"{s}: WF n={wf[f'pooled_{s}']['n']} exp={fmt(wf[f'pooled_{s}']['expectancy'], 2)} "
        f"(random {fmt(wf[f'pooled_{s}']['baseline_expectancy'], 2)}, z={fmt(wf[f'pooled_{s}']['z_vs_random'], 2)}), "
        f"holdout n={res[s]['chosen']['n']} exp={fmt(res[s]['chosen']['expectancy'], 2)} "
        f"(random {fmt(res[s]['chosen']['baseline_expectancy'], 2)})" for s in SIDES)
    if fallback:
        note = f"{PAPER_NOTE} (no variant beat both baselines on walk-forward; best of {mt['n_variants']} variants, " \
               f"{mt['n_tests']} tests, Bonferroni z={mt['bonferroni_z']:.2f}). {pooled_txt}"
    elif bad:
        note = f"{PAPER_NOTE} (beat both baselines in walk-forward but NOT on the one-shot holdout for {bad}; " \
               f"{mt['n_variants']} variants / {mt['n_tests']} tests tried). {pooled_txt}"
    else:
        note = f"Beat both baselines in walk-forward (z vs random >= 2) and on the holdout; {mt['n_variants']} variants / " \
               f"{mt['n_tests']} tests tried (Bonferroni z={mt['bonferroni_z']:.2f}) - shadow before trusting. {pooled_txt}"
    name = export_variant(ds, v, calinfo, fitted, hold, strip(wf), models_dir, status, note, sides)
    log_(f"exported models/{name} ({'gbm' if v['model'] == 'gbm' else 'mtf'}, status={status})  note: {note[:160]}")
    if v["model"] == "mlp":
        import mtf_trainer as T
        del fitted
        T.free_gpu()
    return name, res, arrs, calinfo, note


def strip(wf: dict) -> dict:
    return json.loads(json.dumps(wf, default=float))


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def report_text(results: dict, mt: dict, exported: list[dict], fallback: bool, ds_meta: dict) -> str:
    L = []
    L.append(f"V3 multi-timeframe experiments - walk-forward summary ({mt['n_variants']} variants, {mt['n_tests']} side-tests; "
             f"Bonferroni z>={mt['bonferroni_z']:.2f}; qualify = z vs random>=2 & exp>0 & > ATR rule & n>={MIN_TRADES})")
    L.append(f"data: {ds_meta.get('symbols_n')} coins, {ds_meta.get('rows')} decision rows, horizon {ds_meta.get('horizon')} bars, "
             f"decisions every {ds_meta.get('decision_every_bars')} bars; rules {RULES}")
    L.append("")
    L.append(f"{'variant':28s} {'side':5s} {'n':>5s} {'/day':>5s} {'win':>5s} {'exp$':>7s} {'rand$':>7s} {'z':>5s} {'zclu':>5s} "
             f"{'atr1h$':>7s} {'atr5m$':>7s} {'AUC':>5s} {'BE':>5s} Q")
    for name, r in sorted(results.items()):
        for s in SIDES:
            p = r[f"pooled_{s}"]
            L.append(f"{name:28s} {s:5s} {p['n']:5d} {p['trades_per_day']:5.2f} {fmt(p['win_rate'], 2):>5s} {fmt(p['expectancy'], 2):>7s} "
                     f"{fmt(p['baseline_expectancy'], 2):>7s} {fmt(p['z_vs_random'], 2):>5s} {fmt(p['z_vs_random_clustered'], 2):>5s} "
                     f"{fmt(p['rules']['atr1h']['expectancy'], 2):>7s} {fmt(p['rules']['atr5m']['expectancy'], 2):>7s} "
                     f"{p['auc']:5.3f} {fmt(p['breakeven_win_rate'], 2):>5s} {'YES' if p['qualifies'] else 'no'}")
    L.append("")
    if not exported:
        L.append("exported: none")
    for e in exported:
        h = e["holdout"]
        L.append(f"EXPORTED {e['name']}{'  [PAPER TEST ONLY - no edge]' if e['paper'] else ''}")
        for s in e["sides"]:
            c = h[s]["chosen"]
            L.append(f"  holdout {s}: n={c['n']} win={fmt(c['win_rate'], 2)} exp={fmt(c['expectancy'], 2)} random={fmt(c['baseline_expectancy'], 2)} "
                     f"atr1h={fmt(h[s]['rules']['atr1h']['expectancy'], 2)} atr5m={fmt(h[s]['rules']['atr5m']['expectancy'], 2)}")
    q = [n for n, r in results.items() if any(r[f"pooled_{s}"]["qualifies"] for s in SIDES)]
    L.append("")
    L.append("VERDICT: " + (f"{len(q)} variant(s) beat both baselines on walk-forward: {', '.join(q)}" if q else
                            "no variant beats both baselines on walk-forward (no edge); best exported for PAPER TEST ONLY"))
    return "\n".join(L)


def chat_text(results: dict, mt: dict, exported: list[dict], fallback: bool) -> str:
    """<= 24 lines (notify_text posts the last 25): one line per variant (its better side), exports, verdict."""
    L = [f"V3 mtf walk-forward OOS ({mt['n_variants']} variants/{mt['n_tests']} tests, Bonf z>={mt['bonferroni_z']:.1f}); "
         f"exp$ per trade vs random | best ATR rule"]
    for name, r in sorted(results.items()):
        s = max(SIDES, key=lambda sd: (r[f"pooled_{sd}"]["z_vs_random"] if r[f"pooled_{sd}"]["z_vs_random"] is not None else -9))
        p = r[f"pooled_{s}"]
        L.append(f"{name} {s[0].upper()}: n={p['n']} ({p['trades_per_day']:.1f}/d) exp {fmt(p['expectancy'], 2)} vs rnd "
                 f"{fmt(p['baseline_expectancy'], 2)} | rule {fmt(p['best_rule_expectancy'], 2)} z={fmt(p['z_vs_random'], 1)} "
                 f"AUC {p['auc']:.2f} {'QUALIFIES' if p['qualifies'] else ''}")
    for e in exported:
        h = e["holdout"]
        parts = " ".join(f"{s[0].upper()}: n={h[s]['chosen']['n']} exp {fmt(h[s]['chosen']['expectancy'], 2)} (rnd {fmt(h[s]['chosen']['baseline_expectancy'], 2)})"
                         for s in e["sides"])
        L.append(f"EXPORT {e['name']}{' [PAPER ONLY]' if e['paper'] else ''} holdout {parts}")
    q = [n for n, r in results.items() if any(r[f"pooled_{s}"]["qualifies"] for s in SIDES)]
    L.append("VERDICT: " + (f"{len(q)} beat both baselines: {', '.join(q)}" if q else "NO variant beats both baselines (no edge)"))
    return "\n".join(L[-24:])


def lab_report(name, title, summary, arrs_chosen, out_var: Path):
    pnl = np.asarray(arrs_chosen["pnl"], "float64")
    t = np.asarray(arrs_chosen["t"], "int64")
    o = np.argsort(t, kind="stable")
    cum = 100.0 + np.cumsum(pnl[o])
    pts = [dict(t=int(t[o][i] // 1000), equity=float(cum[i])) for i in range(len(o))]
    if len(pts) > 600:
        pts = [pts[i] for i in np.unique(np.linspace(0, len(pts) - 1, 600).astype(int))]
    rep = dict(id=f"train-{name}", created_at=int(time.time()), kind="train", title=title, summary=summary, equity=pts)
    d = out_var / "lab"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rep['id']}.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


# --------------------------------------------------------------------------------------------
def load_ds(a):
    root = Path(a.archive) if a.archive else None
    syms = [s for s in a.symbols.split(",") if s] or None
    return MD.build_dataset(syms, root, use_cache=not a.no_cache, log=log)


def make_all_splits(ds, a):
    return make_splits(np.asarray(ds.t[::97]), a.folds, a.test_days, a.val_days, a.holdout_days,
                       purge_bars=PURGE_BARS, min_train_days=a.min_train_days)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "wf", "finish", "smoke"])
    ap.add_argument("--models", default="mlp,gbm")
    ap.add_argument("--only", default="", help="comma list of variant names")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--archive", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-days", type=int, default=45)
    ap.add_argument("--val-days", type=int, default=28)
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--min-train-days", type=int, default=90)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args(argv)

    var = D.var_dir()
    out_dir = Path(a.out or var / "ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(a.models_dir or os.environ.get("TB_MODELS") or D.ROOT / "models")

    if a.cmd == "smoke":
        return smoke(a, out_dir)

    ds = load_ds(a)
    log(f"dataset: {len(ds)} rows, {len(ds.symbols)} symbols, {len(ds.names)} features "
        f"(base {len(ds.cols(False))} + kalman {len(ds.cols(True)) - len(ds.cols(False))}), horizon {ds.meta['horizon']}")
    if a.cmd == "build":
        return ds
    splits = make_all_splits(ds, a)
    if not splits["folds"]:
        raise SystemExit("not enough history for a walk-forward fold")
    log("splits: " + json.dumps({"folds": [{k: [iso(x) for x in vv] if isinstance(vv, tuple) else vv
                                            for k, vv in f.items()} for f in splits["folds"]],
                                 "final": {k: [iso(x) for x in vv] for k, vv in splits["final"].items()}}))
    want_models = set(a.models.split(","))
    only = set(x for x in a.only.split(",") if x)
    vs = [v for v in variants() if v["model"] in want_models and (not only or v["name"] in only)]

    if a.cmd == "wf":
        if any(v["model"] == "mlp" for v in vs):
            import mtf_trainer as T
            T.require_mps()
        for v in vs:
            run_wf_variant(ds, v, splits, out_dir, log, force=a.force, seeds=a.seeds)
        return

    # ---- finish ----
    results = {}
    for v in variants():
        jp, _ = result_paths(out_dir, v["name"])
        if jp.exists():
            results[v["name"]] = json.loads(jp.read_text())
    if not results:
        raise SystemExit("no v3_mtf_wf_*.json results in " + str(out_dir))
    chosen, fallback, mt = pick_export(results)
    log(f"qualifying variants: {[c['name'] for c in chosen] if not fallback else 'none'}; fallback={fallback}; "
        f"{mt['n_variants']} variants, {mt['n_tests']} tests, Bonferroni z={mt['bonferroni_z']:.2f}")
    exported = []
    best_arrs = None
    if not a.no_export:
        for v in chosen:
            wf = results[v["name"]]
            name, hres, harrs, calinfo, note = run_final(ds, v, splits, wf, models_dir, fallback, mt, log)
            (out_dir / f"v3_mtf_final_{name}.json").write_text(json.dumps(
                dict(holdout=hres, thresholds=calinfo["thresholds"], calibration=calinfo["calibration"],
                     val=calinfo["val"], note=note), indent=1, default=float))
            exported.append(dict(name=name, holdout=hres, sides=v["sides"], paper=PAPER_NOTE in note, variant=v["name"]))
            side0 = v["sides"][0]
            summ = dict(variant=v["name"], model=name, sides=v["sides"], fallback=fallback, note=note,
                        wf=dict((s, {k: wf[f"pooled_{s}"][k] for k in ("n", "win_rate", "expectancy", "baseline_expectancy",
                                                                       "z_vs_random", "z_vs_random_clustered", "trades_per_day",
                                                                       "best_rule_expectancy", "auc")}) for s in SIDES),
                        holdout={s: hres[s]["chosen"] for s in SIDES}, n_variants=mt["n_variants"])
            lab_report(name, f"Train {name} ({v['model']}, {v['exit']} exits{', kalman' if v['kalman'] else ''})", summ,
                       harrs[side0]["chosen"], var)
    ds_meta = dict(symbols_n=len(ds.symbols), rows=int(len(ds)), horizon=ds.meta["horizon"],
                   decision_every_bars=ds.meta["decision_every_bars"])
    txt = report_text(results, mt, exported, fallback, ds_meta)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"v3_mtf_report_{stamp}.txt").write_text(txt)
    (out_dir / "v3_mtf_report_latest.txt").write_text(txt)
    (out_dir / "v3_mtf_chat_latest.txt").write_text(chat_text(results, mt, exported, fallback))
    (out_dir / "v3_mtf_summary.json").write_text(json.dumps(
        dict(multiple_testing=mt, fallback=fallback, exported=[{k: v_ for k, v_ in e.items() if k != "holdout"} for e in exported],
             variants={k: {s: {kk: vv for kk, vv in r[f"pooled_{s}"].items()} for s in SIDES} for k, r in results.items()}),
        indent=1, default=float))
    log(txt)


def smoke(a, out_dir):
    """End-to-end on synthetic data: dataset -> 1 fold for gbm + mlp -> final -> export -> load via model_np."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="tb_mtf_smoke_"))
    ds = MD.synthetic_dataset(n_symbols=4, n=30000, horizon=300)
    log(f"smoke dataset {len(ds)} rows {len(ds.names)} features")
    global PURGE_BARS
    PURGE_BARS = 300 + 12
    t = np.asarray(ds.t)
    span_days = (t.max() - t.min()) / DAY_MS
    log(f"span {span_days:.1f} days")
    sp = make_splits(t[::7], 1, 8, 6, 8, purge_bars=PURGE_BARS, min_train_days=10)
    assert sp["folds"], sp
    for v in [x for x in variants() if x["exit"] == "pct36" and x["kalman"]]:
        if v["model"] == "mlp":
            import mtf_trainer as T
            T.require_mps()
        run_wf_variant(ds, v, sp, tmp, log, force=True, seeds=1)
        v2 = dict(v, sides=list(SIDES))
        wf = json.loads(result_paths(tmp, v["name"])[0].read_text())
        name, hres, harrs, calinfo, note = run_final(ds, v2, sp, wf, tmp / "models", True,
                                                     dict(n_variants=2, n_tests=4, bonferroni_z=2.5), log)
        sc = mnp.load(tmp / "models" / f"{name}.json")
        X = np.asarray(ds.X[:5]).astype("float32")[:, ds.cols(v["kalman"])]
        p = sc.predict(X)
        assert p.shape == (5, 2) and np.isfinite(p).all(), p
        log(f"smoke OK {name} kind={sc.kind} p={p[0]}")
    return tmp


if __name__ == "__main__":
    main()
