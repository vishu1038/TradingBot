"""Mean-reversion family (S4) of the rules research: 4 variants = {chop, any} x {5m, 1h}, pre-registered 2026-10-21.

A SEPARATE module so the 12-variant trend family of ``tb.sim.rules_research`` (VARIANTS, Bonferroni family 12, its tables /
select / holdout / export and its cache files) is untouched.  Everything else is reused from it: the engine-faithful ATR-exit
simulation (``simulate_atr``: TP 1.5 x ATR(1h) closes the WHOLE position, SL 3 x ATR(1h), time stop 1152 bars, $2 floor, caps,
reversals), the segments, the statistics and the baseline draws.

Differences (all by design):
  * candidates = ``rules.rule_signals`` of the S4 variants (no BTC data needed); membership / ATR / pools come from the trend
    run's ``signals.pkl`` pools (so the universe is exactly the same), nothing here looks at trade outcomes;
  * baselines: random (same count, same side mix) and regime-only = random entries only while the coin's own regime is chop
    (variants ``chop``) or simply random (variants ``any`` have no gate, so the two baselines coincide in distribution);
  * Bonferroni over the 4 MR variants.

Cache files (var/rules_cache): signals_mr.pkl, dev_mr_results.pkl, carried_mr.json, holdout_mr_results.pkl.

  python -m tb.sim.rules_research dev_mr [workers]       # (builds signals_mr.pkl if missing) 4 variants x (rule + 2 baselines x 20 seeds)
  python -m tb.sim.rules_research holdout_mr [workers]   # the dev-chosen variant(s) ONCE on the holdout 2026-08-15..10-14 and post 10-14..10-18
  python -m tb.sim.rules_research preview_mr             # dev-only shadow challenger (holdout pending)
  python -m tb.sim.rules_research export_mr              # dev-chosen best variant with the holdout verdict
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rules
from ..data import loader
from . import rules_research as rr
from .model_backtest import BAR, Market, load_settings

CACHE = rr.CACHE
MR_VARIANTS = [rules.RuleCfg(m, "S4", t) for m in rules.MR_MODES for t in rules.TFS]
N_MR = len(MR_VARIANTS)                                  # Bonferroni family size (4)
N_SEEDS = rr.N_SEEDS
MIN_N = 150                                              # dev selection: n >= 150 trades, best z vs random on $/trade
N_CARRY = 2
log = rr.log


def variant(key: str) -> rules.RuleCfg:
    return next(v for v in MR_VARIANTS if v.key == key)


# ----------------------------------------------------------------------------------------------------- build
def build_mr() -> None:
    """S4 signals for every symbol of the trend run's universe (membership + ATR + pools taken from its ``signals.pkl``)."""
    from ..features_mtf import htf_atr_5m

    meta = pickle.load(open(CACHE / "meta.pkl", "rb"))
    sig = pickle.load(open(CACHE / "signals.pkl", "rb"))
    pool = {tf: sig["pool"][tf][["sym", "bar", "atr"]].reset_index(drop=True) for tf in ("5m", "1h")}
    del sig
    rows: dict[str, dict[str, np.ndarray]] = {}
    for tf, df in pool.items():
        codes = df["sym"].cat.codes.to_numpy()
        order = np.argsort(codes, kind="stable")
        cuts = np.searchsorted(codes[order], np.arange(len(df["sym"].cat.categories) + 1))
        cats = list(df["sym"].cat.categories)
        for i, s in enumerate(cats):
            rows.setdefault(s, {})[tf] = order[cuts[i]:cuts[i + 1]]
    chop = {tf: np.zeros(len(df), bool) for tf, df in pool.items()}
    cands: dict[str, list] = {v.key: [] for v in MR_VARIANTS}
    for n_i, s in enumerate(meta["ever"]):
        kl = loader.load_klines(s)
        t = kl["open_time"].to_numpy("int64")
        atr = htf_atr_5m(kl, "1h", 14)
        sigs = {v.key: rules.rule_signals(kl, None, v) for v in MR_VARIANTS}
        reg_chop = sigs["mr_chop_5m"].regime
        for tf in ("5m", "1h"):
            pos = rows.get(s, {}).get(tf)
            if pos is None or not len(pos):
                continue
            bars = pool[tf]["bar"].to_numpy()[pos]
            k = np.searchsorted(t, bars)
            assert (t[k] == bars).all(), s
            chop[tf][pos] = reg_chop[k] == 0
            member = np.zeros(len(t), bool)
            member[k] = True
            for v in MR_VARIANTS:
                if v.tf != tf:
                    continue
                for side, arr in (("long", sigs[v.key].long), ("short", sigs[v.key].short)):
                    idx = np.flatnonzero(arr & member)
                    if len(idx):
                        cands[v.key].append(pd.DataFrame({"sym": s, "bar": t[idx], "side": side, "atr": atr[idx].astype("float32")}))
        if n_i % 25 == 0:
            log("mr signals", n_i, "of", len(meta["ever"]), s)
    for tf in pool:
        pool[tf]["chop"] = chop[tf]
    out = {"cands": {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=["sym", "bar", "side", "atr"]))
                     for k, v in cands.items()}, "pool": pool}
    pickle.dump(out, open(CACHE / "signals_mr.pkl", "wb"), protocol=4)
    for k, df in out["cands"].items():
        log("mr candidates", k, len(df), "long", int((df["side"] == "long").sum()), "short", int((df["side"] == "short").sum()))
    for k, df in pool.items():
        log("mr pool", k, len(df), "chop share", round(float(df["chop"].mean()), 3))


# ----------------------------------------------------------------------------------------------------- workers
_G: dict = {}


def _load_parent() -> None:
    """Loaded ONCE in the parent before the fork, so the workers share the klines copy-on-write."""
    cfg = load_settings()
    meta = pickle.load(open(CACHE / "meta.pkl", "rb"))
    sig = pickle.load(open(CACHE / "signals_mr.pkl", "rb"))
    data = {s: (loader.load_klines(s), rr._funding(s)) for s in meta["ever"]}
    _G.update(cfg=cfg, mkt=Market(data), cands=sig["cands"], pool=sig["pool"])


def cands_for(vkey: str, kind: str, seed: int, period: tuple[int, int], cands=None, pools=None) -> pd.DataFrame:
    v = variant(vkey)
    c = (cands or _G["cands"])[vkey]
    c = c[(c["bar"] >= period[0]) & (c["bar"] + BAR <= period[1])].copy()
    if kind == "rule":
        c["prob"] = np.random.default_rng(12345).random(len(c))          # random tie-break among simultaneous signals
        return c
    pool = rr._pool_slice((pools or _G["pool"])[v.tf], *period)
    if kind == "regime" and v.mode == "chop":
        pool = pool[pool["chop"].to_numpy()]
    rng = np.random.default_rng([seed, 1000 + (0 if kind == "random" else 1), sum(map(ord, vkey))])
    return rr.draw_random(pool[["sym", "bar", "atr"]], int((c["side"] == "long").sum()), int((c["side"] == "short").sum()), rng)


def run_job(job: tuple) -> dict:
    vkey, kind, seed, pname = job
    period = rr.PERIODS[pname]
    t = time.time()
    c = cands_for(vkey, kind, seed, period)
    trades, extra = rr.run_period(c, _G["mkt"], _G["cfg"], period)
    days = (period[1] - period[0]) / 86_400_000
    out = dict(vkey=vkey, kind=kind, seed=seed, period=pname, n_cands=len(c), secs=time.time() - t, **extra,
               summary=rr.summarize(trades, days))
    if kind == "rule":
        out["trades"] = trades
    return out


def run_jobs_mr(jobs: list[tuple], workers: int, out_path: Path) -> list[dict]:
    """Resumable like ``rr.run_jobs`` (jobs already in ``out_path`` are skipped)."""
    res: list[dict] = pickle.load(open(out_path, "rb")) if out_path.exists() else []
    done = {(r["vkey"], r["kind"], r["seed"], r["period"]) for r in res}
    jobs = [j for j in jobs if tuple(j) not in done]
    if done:
        log(f"resuming: {len(done)} job(s) already in {out_path.name}, {len(jobs)} to run")
    if not jobs:
        return res
    _load_parent()
    t = time.time()
    with mp.get_context("fork").Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(run_job, jobs, chunksize=1)):
            res.append(r)
            if i % 10 == 0 or r["kind"] == "rule":
                log(f"job {i + 1}/{len(jobs)} {r['vkey']} {r['kind']} {r['seed']} n={r['summary']['n']} per={r['summary']['per']:.3f} "
                    f"({r['secs']:.0f}s, elapsed {time.time() - t:.0f}s)")
                pickle.dump(res, open(out_path, "wb"))
    pickle.dump(res, open(out_path, "wb"))
    return res


# ----------------------------------------------------------------------------------------------------- reporting
def table_mr(res: list[dict], pname: str, keys: list[str] | None = None) -> pd.DataFrame:
    """``rr.table`` for the MR family: same columns, Bonferroni over 4."""
    rows = []
    for v in MR_VARIANTS:
        if keys is not None and v.key not in keys:
            continue
        rr_ = [r for r in res if r["vkey"] == v.key and r["period"] == pname]
        rule = next((r for r in rr_ if r["kind"] == "rule"), None)
        if rule is None:
            continue
        s = rule["summary"]
        rnd = [r["summary"]["per"] for r in rr_ if r["kind"] == "random"]
        reg = [r["summary"]["per"] for r in rr_ if r["kind"] == "regime"]
        z_r, z_g = rr.zscore(s["per"], rnd), rr.zscore(s["per"], reg)
        p_r, _ = rr.bonferroni(z_r, N_MR)
        p_g, _ = rr.bonferroni(z_g, N_MR)
        rows.append(dict(variant=v.key, n=s["n"], per_day=round(s["per_day"], 2), win=round(s["win"], 3), per=round(s["per"], 3),
                         total=round(s["total"], 1), maxdd=round(s["maxdd"], 1), L_S=f"{s['n_long']}/{s['n_short']}",
                         usd_L=round(s["usd_long"], 1), usd_S=round(s["usd_short"], 1),
                         rand_per=round(float(np.nanmean(rnd)), 3) if rnd else np.nan, z_rand=round(z_r, 2), p_bonf_rand=round(p_r, 4),
                         regime_per=round(float(np.nanmean(reg)), 3) if reg else np.nan, z_regime=round(z_g, 2),
                         p_bonf_regime=round(p_g, 4), kills=rule.get("kills", 0)))
    return pd.DataFrame(rows)


def select_mr(dev: pd.DataFrame) -> list[str]:
    """Pre-registered: n >= 150 trades, ranked by dev z vs random on $/trade (fallback: all variants); the best ``N_CARRY`` go to the holdout."""
    t = dev[dev["n"] >= MIN_N]
    if t.empty:
        t = dev
    t = t.assign(_z=t["z_rand"].fillna(-1e9)).sort_values("_z", ascending=False, kind="stable")
    return t["variant"].tolist()[:N_CARRY]


# ----------------------------------------------------------------------------------------------------- export
def _fmt(r: dict) -> str:
    return (f"n={r['n']} win={r['win']:.2f} ${r['per']:+.2f}/trade total ${r['total']:+.0f} maxDD ${r['maxdd']:.0f} "
            f"L/S {r['L_S']}; random ${r['rand_per']:+.2f} (z={r['z_rand']:.2f}, p_bonf={r['p_bonf_rand']:.3f}), "
            f"regime-only ${r['regime_per']:+.2f} (z={r['z_regime']:.2f}, p_bonf={r['p_bonf_regime']:.3f})") if r else "n/a"


def export_mr(models_dir: Path | None = None, stamp: str | None = None, alpha: float = 0.05, preview: bool = False) -> Path:
    """The dev-chosen best MR variant as a shadow challenger.  ``preview``: dev numbers only ('holdout pending')."""
    models_dir = models_dir or rr.repo_root() / "models"
    stamp = stamp or time.strftime("%Y%m%d_%H%M")
    dev = table_mr(pickle.load(open(CACHE / "dev_mr_results.pkl", "rb")), "dev")
    key = select_mr(dev)[0]
    v = variant(key)
    d = rr._row(dev, key)
    h = q = {}
    if not preview:
        hres = pickle.load(open(CACHE / "holdout_mr_results.pkl", "rb"))
        h, q = rr._row(table_mr(hres, "holdout", [key]), key), rr._row(table_mr(hres, "post", [key]), key)
    clears = (not preview) and bool(h) and h["n"] >= 30 and h["per"] > 0 and h["p_bonf_rand"] < alpha and h["p_bonf_regime"] < alpha
    name = f"rules_{v.key}_{stamp}" + ("_preview" if preview else "")
    common = (f"rule bot {v.key}: {rules.describe(v)}. Exits 1:2 ATR(1h) (TP 1.5, SL 3), the TP closes the whole position. "
              f"Chosen on the DEVELOPMENT period only (best dev z vs random of the {N_MR} mean-reversion variants, n>={MIN_N}). "
              f"Dev 2024-11..2026-08-14: {_fmt(d)}. ")
    if preview:
        note = rr.PAPER_NOTE + "PREVIEW " + common + "HOLDOUT pending (the locked holdout verdict replaces this preview)."
    else:
        verdict = (f"beats both baselines on the locked holdout (Bonferroni over {N_MR} variants)" if clears else
                   "does NOT beat both baselines on the locked holdout with the Bonferroni-corrected z; watch it on paper only")
        note = ("" if clears else rr.PAPER_NOTE) + common + f"HOLDOUT 2026-08-15..10-14: {_fmt(h)}. POST 10-14..10-18: {_fmt(q)}. {verdict}."
    meta = {"name": name, "version": stamp.replace("_", "."), "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "status": "challenger", "kind": "rules", "arch": "rules", "parent": None, "rule_cfg": v.to_dict(),
            "timeframes": ["5m", "1h"], "history_bars_5m": v.history_bars_5m(), "thresholds": {"long": 0.5, "short": 0.5},
            "exit_cfg": {**rr.EXIT_CFG, "tp_closes_fully": True, "min_tp_usd": 2.0, "min_sl_usd": 2.0},
            "entry_cfg": {"decision_every_bars": 12 if v.tf == "1h" else 1, "cooldown_bars": 6, "max_trades_per_day": None,
                          "max_open": None, "entry_mode": "taker", "decision_mode": "fixed"},
            "metrics": {"dev": d, "holdout": h, "post": q, "n_variants": N_MR, "baseline_seeds": N_SEEDS}, "note": note}
    p = models_dir / f"{name}.json"
    tmp = models_dir / f".{name}.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1, default=float))
    tmp.replace(p)
    log("exported", p)
    return p


# ----------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "help"
    workers = min(int(argv[2]) if len(argv) > 2 else 2, 2)                # the trend run owns the other cores
    pd.set_option("display.width", 250, "display.max_columns", 30)
    if cmd == "build_mr":
        build_mr()
    elif cmd == "dev_mr":
        if not (CACHE / "signals_mr.pkl").exists():
            build_mr()
        jobs = [(v.key, "rule", 0, "dev") for v in MR_VARIANTS]
        jobs += [(v.key, k, s, "dev") for v in MR_VARIANTS for k in ("random", "regime") for s in range(N_SEEDS)]
        jobs.sort(key=lambda j: (j[1] != "rule", variant(j[0]).tf != "5m"))
        res = run_jobs_mr(jobs, workers, CACHE / "dev_mr_results.pkl")
        t = table_mr(res, "dev")
        carried = select_mr(t)
        json.dump({"carried": carried, "best": carried[0], "rule": f"dev z vs random on $/trade, n>={MIN_N}, top {N_CARRY}"},
                  open(CACHE / "carried_mr.json", "w"), indent=1)
        print(t.to_string(index=False))
        print("carried:", carried)
    elif cmd == "holdout_mr":
        carried = json.load(open(CACHE / "carried_mr.json"))["carried"]
        jobs = [(k, "rule", 0, p) for k in carried for p in ("holdout", "post")]
        jobs += [(k, kind, s, p) for k in carried for kind in ("random", "regime") for s in range(N_SEEDS) for p in ("holdout", "post")]
        res = run_jobs_mr(jobs, workers, CACHE / "holdout_mr_results.pkl")
        for p in ("holdout", "post"):
            print(p)
            print(table_mr(res, p, carried).to_string(index=False))
    elif cmd == "preview_mr":
        export_mr(preview=True)
    elif cmd == "export_mr":
        export_mr()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
