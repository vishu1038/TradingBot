"""Honest research backtest of the rule variants of ``tb.rules`` (12 = {btc, coin} x {S1, S2, S3} x {5m, 1h}).

Reuses the engine-faithful machinery of ``tb.sim.model_backtest`` (point-in-time hot list, ``Market`` exit lookups, the
engine's sizing / caps / cooldowns) and differs in exactly three places:
  * candidates are rule signals (``tb.rules.rule_signals``), not model probabilities;
  * exits are the ATR template of the rule models: TP 1.5 x ATR14(1h), SL 3 x ATR14(1h), TP closes the WHOLE position, time stop
    1152 bars; ``tb.exits`` / ``tb.risk.size_trade`` do the sizing exactly as the engine does for ``exit_cfg.atr_tf = "1h"``;
  * an opposite signal on a held coin reverses it (as the engine does with ``reverse_enabled``).
Entry = the decision bar's close (+ slippage), like ``model_backtest.simulate``; profit protection is inert for this exit
(``protection_inert`` asserts it from the real ``tb.protect`` functions).

Workflow (no peeking: the holdout is only touched by ``holdout``, after ``select`` froze the carried variants):
  python -m tb.sim.rules_research build     # universe, signals, pools   -> var/rules_cache
  python -m tb.sim.rules_research dev       # 12 variants x (rule + random + regime-only baselines) on the development period
  python -m tb.sim.rules_research select    # freeze the carried variants (best of each mode + top-2 others) -> carried.json
  python -m tb.sim.rules_research holdout   # carried variants ONCE on 2026-08-15..10-14 and 10-14..10-18
"""
from __future__ import annotations

import heapq
import json
import math
import multiprocessing as mp
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rules
from ..data import loader
from ..data.paths import funding_path, repo_root
from ..entry import EntryCfg  # noqa: F401  (documented gate semantics)
from ..exits import ExitSpec
from ..protect import Protect, is_loss_exit, next_stop, partial_level
from ..risk import Book, check_entry, choose_leverage, correlation_gate, derate_multiplier, maintenance_margin, size_trade, sl_loss  # noqa: F401
from . import model_backtest as mb
from .model_backtest import BAR, Market, SimCfg, _stub_filters, build_universe, load_settings, ms

CACHE = repo_root() / "var" / "rules_cache"
T0 = ms("2024-10-01")
START_EFF = ms("2024-11-03")           # regimes need ~30 days of history (BTC EMA240h = 723 hourly taps)
DEV_END = ms("2026-08-15")
HOLD_END = ms("2026-10-14")
POST_END = ms("2026-10-18 16:35")
PERIODS = {"dev": (START_EFF, DEV_END), "holdout": (DEV_END, HOLD_END), "post": (HOLD_END, POST_END)}
N_SEEDS = 20
VARIANTS = [rules.RuleCfg(m, s, t) for m in rules.MODES for s in rules.SETUPS for t in rules.TFS]
EXIT_CFG = {"exit_mode": "atr", "atr_tf": "1h", "tp_atr": 1.5, "sl_atr": 3.0, "horizon_bars": 1152}
SPEC = ExitSpec("atr", "1h", tp_atr=1.5, sl_atr=3.0, horizon_bars=1152)
SEG_MONTHS = 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------- build
def _funding(sym: str) -> pd.DataFrame:
    p = funding_path(sym)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame({"calc_time": np.zeros(0, "int64"), "funding_rate": np.zeros(0)})


def build_universe_chunked(klines: dict[str, pd.DataFrame], funding: dict, cfg, t_end: int, chunk: int = 40_000, ov: int = 300) -> dict:
    """``build_universe`` over the whole archive in time chunks (the dense arrays of one call would be ~4 GB)."""
    grid_n = int((t_end - T0) // BAR) + 1
    mask = {s: np.zeros(grid_n, bool) for s in klines}
    first = {s: int(k["open_time"].iloc[0]) for s, k in klines.items()}
    c0 = 0
    while c0 < grid_n:
        c1 = min(grid_n, c0 + chunk)
        a = max(0, c0 - ov)
        g0, g1 = T0 + a * BAR, T0 + (c1 - 1) * BAR
        panel = {}
        for s, k in klines.items():
            t = k["open_time"].to_numpy()
            i0, i1 = np.searchsorted(t, g0), np.searchsorted(t, g1, side="right")
            if i1 > i0:
                panel[s] = k.iloc[i0:i1]
        m = build_universe(panel, g0, g1, cfg, funding, first, T0, mb.HOUR)
        for s, v in m.items():
            mask[s][c0:c1] = v[c0 - a:]
        log("universe chunk", c0, c1, "of", grid_n)
        c0 = c1
    return mask


def build() -> None:
    """Universe + signals + pools for every symbol; nothing here looks at trade outcomes."""
    from ..features_mtf import htf_atr_5m

    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = load_settings()
    syms = loader.symbols_available()
    log("symbols", len(syms), "universe_size", cfg.universe_size)
    kls = {s: loader.load_klines(s) for s in syms}
    fus = {s: _funding(s) for s in syms}
    t_last = max(int(k["open_time"].iloc[-1]) for k in kls.values())
    mask = build_universe_chunked(kls, fus, cfg, t_last)
    ever = sorted(s for s in syms if mask[s].any())
    log("ever members", len(ever))
    pickle.dump({"ever": ever, "t_last": t_last, "universe_size": int(cfg.universe_size)}, open(CACHE / "meta.pkl", "wb"))
    btc = kls["BTCUSDT"]
    cands: dict[str, list] = {v.key: [] for v in VARIANTS}
    pool = {"5m": [], "1h": []}
    for n_i, s in enumerate(ever):
        kl = kls[s]
        t = kl["open_time"].to_numpy("int64")
        member = mask[s][(t - T0) // BAR] & (t >= START_EFF)
        atr = htf_atr_5m(kl, "1h", 14)
        okatr = np.isfinite(atr) & (atr > 0)
        regs = {}
        for v in VARIANTS:
            sig = rules.rule_signals(kl, btc, v)
            regs[(v.mode, v.tf)] = sig.regime
            for side, arr in (("long", sig.long), ("short", sig.short)):
                idx = np.flatnonzero(arr & member & okatr)
                if len(idx):
                    cands[v.key].append(pd.DataFrame({"sym": s, "bar": t[idx], "side": side, "atr": atr[idx].astype("float32")}))
        for tf in ("5m", "1h"):
            due = np.ones(len(t), bool) if tf == "5m" else ((t + BAR) % rules.HOUR_MS == 0)
            idx = np.flatnonzero(due & member & okatr)
            pool[tf].append(pd.DataFrame({"sym": pd.Categorical([s] * len(idx), categories=ever), "bar": t[idx],
                                          "atr": atr[idx].astype("float32"),
                                          "reg_btc": regs[("btc", tf)][idx], "reg_coin": regs[("coin", tf)][idx]}))
        if n_i % 20 == 0:
            log("signals", n_i, "of", len(ever), s)
    out = {"cands": {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=["sym", "bar", "side", "atr"]))
                     for k, v in cands.items()},
           "pool": {k: pd.concat(v, ignore_index=True).sort_values("bar", kind="stable").reset_index(drop=True) for k, v in pool.items()}}
    pickle.dump(out, open(CACHE / "signals.pkl", "wb"), protocol=4)
    for k, df in out["cands"].items():
        log("candidates", k, len(df))
    for k, df in out["pool"].items():
        log("pool", k, len(df))


# ----------------------------------------------------------------------------------------------------- simulation
def protection_inert(cfg, spec: ExitSpec) -> bool:
    """True when break-even / partial TP / trailing can never fire before the TP for this ATR exit (TP at 0.5 R)."""
    pc = Protect.from_settings(cfg)
    a = 1.0
    for side, d in (("long", 1.0), ("short", -1.0)):
        entry = 100.0
        sl0, tp = entry - d * spec.sl_atr * a, entry + d * spec.tp_atr * a
        if partial_level(side, entry, sl0, tp, pc) is not None:
            return False
        # the most favourable price the position can have seen while still open is just below the TP
        _, _, why = next_stop(side, entry, sl0, sl0, tp - d * 1e-6, a, False, pc)
        if why is not None:
            return False
    return True


def simulate_atr(cands: pd.DataFrame, mkt: Market, cfg, spec: ExitSpec, sc: SimCfg, t_end: int, info: dict | None = None) -> list[dict]:
    """``model_backtest.simulate`` for an ATR exit (sl_atr / tp_atr x ATR of ``cands.atr``) with reversal on an opposite signal.
    Columns of ``cands``: sym, bar, side, prob, atr (``bar`` = open_time of the decision bar)."""
    info = {} if info is None else info
    eff = spec.effective(cfg)
    f = _stub_filters()
    cooldown = 6 * BAR
    slc = cfg.sl_cooldown_bars * BAR
    horizon = int(sc.horizon_bars or spec.horizon_bars or 0) or 10**9
    fee, slip, sslip = cfg.fee_taker, cfg.slippage_pct, cfg.stop_slippage_pct
    kill_dd = cfg.kill_limit("paper_local")
    cl = cfg.corr_cap
    wallet = float(sc.start_equity)
    open_: list[dict] = []
    held: dict[str, dict] = {}
    heap: list[tuple] = []
    by_uid: dict[int, dict] = {}
    last: dict[str, int] = {}
    last_loss: dict[str, int] = {}
    trades: list[dict] = []
    killed, kill_t, uid = False, None, 0
    stats = {"reversals": 0, "skip_floor": 0, "skip_caps": 0, "skip_corr": 0, "skip_size": 0}

    def close_pos(p: dict, t_x: int) -> None:
        nonlocal wallet
        open_.remove(p)
        held.pop(p["sym"], None)
        p["done"] = True
        d = 1.0 if p["side"] == "long" else -1.0
        gross = (p["px"] - p["fill"]) * p["qty"] * d
        exit_fee = p["qty"] * p["px"] * fee
        fund = mkt.funding_paid(p["sym"], p["side"], p["t_in"], t_x, p["notional"]) if sc.funding else 0.0
        net = gross - p["entry_fee"] - exit_fee - fund
        wallet += gross - exit_fee - fund
        last[p["sym"]] = max(last.get(p["sym"], 0), t_x)
        if is_loss_exit(p["reason"], net):
            last_loss[p["sym"]] = max(last_loss.get(p["sym"], 0), t_x)
        trades.append(dict(sym=p["sym"], side=p["side"], t_in=p["t_in"], t_out=t_x, pnl=net, fees=p["entry_fee"] + exit_fee,
                           funding=fund, reason=p["reason"], notional=p["notional"], end=p["end"]))

    def settle(until: int) -> None:
        while heap and heap[0][0] <= until:
            t_x, u = heapq.heappop(heap)
            p = by_uid.pop(u)
            if not p.get("done"):
                close_pos(p, t_x)

    def equity(at: int) -> float:
        e = wallet
        for p in open_:
            m = mkt.mark(p["sym"], at)
            if m is not None:
                e += (m - p["fill"]) * p["qty"] * (1.0 if p["side"] == "long" else -1.0)
        return e

    for r in cands.itertuples(index=False):
        t_dec = int(r.bar) + BAR
        if t_dec > t_end:
            break
        settle(t_dec)
        if killed:
            break
        sym, side = r.sym, r.side
        if cooldown and t_dec - last.get(sym, -10**18) < cooldown:
            continue
        if slc and t_dec - last_loss.get(sym, -10**18) < slc:
            continue
        h = held.get(sym)
        if h is not None and h["side"] == side:
            continue
        eq = equity(t_dec) if sc.caps else sc.start_equity
        if sc.caps and (sc.start_equity - eq) / sc.start_equity * 100.0 >= kill_dd:
            killed, kill_t = True, t_dec
            for p in list(open_):
                m = mkt.mark(p["sym"], t_dec) or p["fill"]
                p.update(px=m * (1 - (1.0 if p["side"] == "long" else -1.0) * slip), reason="kill", end=False)
                close_pos(p, t_dec)
            break
        ref = mkt.mark(sym, t_dec)
        if ref is None:
            continue
        if h is not None:                                     # opposite signal on a held coin: reverse (engine: reverse_enabled)
            dh = 1.0 if h["side"] == "long" else -1.0
            h.update(px=ref * (1 - dh * slip), reason="reverse", end=False)
            close_pos(h, t_dec)
            stats["reversals"] += 1
            eq = equity(t_dec) if sc.caps else sc.start_equity
        mult = 1.0
        if sc.caps:
            dd = max(0.0, (sc.start_equity - eq) / sc.start_equity * 100.0)
            mult = derate_multiplier(dd, kill_dd)
            if mult <= 0:
                continue
            same = [p for p in open_ if p["side"] == side]
            if len(same) >= cl.max_correlated:
                hot = 0
                for p in same:
                    c = mkt.corr(sym, p["sym"], t_dec)
                    if c is not None and c > cl.max_corr:
                        hot += 1
                if hot >= cl.max_correlated:
                    stats["skip_corr"] += 1
                    continue
        sz = size_trade(eff, side, ref, spec.atr_arg(float(r.atr), ref), f, None, risk_mult=mult, brackets=None)
        if not sz.ok:
            stats["skip_floor" if "floor" in sz.reason else "skip_size"] += 1
            continue
        d = 1.0 if side == "long" else -1.0
        fill = ref * (1 + slip) if side == "long" else ref * (1 - slip)
        qty = sz.qty
        sl_p = fill - d * spec.sl_atr * float(r.atr)
        tp_p = fill + d * spec.tp_atr * float(r.atr)
        notional = qty * fill
        efee = notional * fee
        if sc.caps:
            lev = choose_leverage(side, fill, sl_p, notional, cfg.max_leverage, None, cfg.maint_margin_rate)
            if lev is None:
                stats["skip_size"] += 1
                continue
            margin = notional / lev
            used = sum(p["margin"] for p in open_)
            risk_l = sum(p["risk"] for p in open_ if p["side"] == "long")
            risk_s = sum(p["risk"] for p in open_ if p["side"] == "short")
            new_risk = sl_loss(cfg, qty, fill, sl_p, side, entry_fee=True)
            bk = Book(eq, eq - used, sum(p["notional"] for p in open_),
                      sum(maintenance_margin(p["notional"], None, cfg.maint_margin_rate) for p in open_),
                      risk_l + risk_s, risk_l, risk_s)
            if not check_entry(cfg, bk, notional, maintenance_margin(notional, None, cfg.maint_margin_rate), new_risk, margin, efee, side).ok:
                stats["skip_caps"] += 1
                continue
        else:
            margin, new_risk = notional / cfg.leverage, 0.0
        t_x, px, why, end = mkt.exit_of(sym, side, int(r.bar), fill, sl_p, tp_p, horizon, slip, sslip, sc.stop_fill)
        uid += 1
        wallet -= efee
        p = dict(uid=uid, sym=sym, side=side, t_in=t_dec, fill=fill, qty=qty, notional=notional, margin=margin,
                 risk=new_risk - efee, entry_fee=efee, px=px, reason=why, end=end)
        open_.append(p)
        held[sym] = p
        by_uid[uid] = p
        heapq.heappush(heap, (t_x, uid))
        last[sym] = t_dec
    settle(10**18)
    info.update(kill_t=kill_t, **stats)
    return trades


def segments(t0: int, t1: int, months: int = SEG_MONTHS) -> list[tuple[int, int]]:
    """Consecutive blocks of ``months`` calendar months from t0 to t1 (each is simulated from a fresh $200 book)."""
    out, a = [], pd.Timestamp(t0, unit="ms", tz="UTC")
    end = pd.Timestamp(t1, unit="ms", tz="UTC")
    while a < end:
        b = min(end, (a + pd.DateOffset(months=months)).normalize().replace(day=1) if months else end)
        if b <= a:
            b = end
        out.append((int(a.value // 10**6), int(b.value // 10**6)))
        a = b
    return out


def run_period(cands: pd.DataFrame, mkt: Market, cfg, period: tuple[int, int], seg_months: int = SEG_MONTHS) -> tuple[list[dict], dict]:
    """All trades of ``cands`` inside [t0, t1): one fresh-book simulation per segment (a 50 % kill ends only its own segment)."""
    t0, t1 = period
    c = cands[(cands["bar"] >= t0) & (cands["bar"] + BAR <= t1)].sort_values(["bar", "prob"], ascending=[True, False], kind="stable")
    trades, kills = [], 0
    for a, b in segments(t0, t1, seg_months):
        part = c[(c["bar"] >= a) & (c["bar"] < b)]
        if not len(part):
            continue
        inf: dict = {}
        trades += simulate_atr(part, mkt, cfg, SPEC, SimCfg(), b, inf)
        kills += int(inf.get("kill_t") is not None)
    return trades, {"kills": kills}


# ----------------------------------------------------------------------------------------------------- statistics
def summarize(trades: list[dict], days: float, start_equity: float = 200.0) -> dict:
    n = len(trades)
    if not n:
        return dict(n=0, per_day=0.0, win=float("nan"), per=float("nan"), total=0.0, maxdd=0.0, n_long=0, n_short=0, usd_long=0.0,
                    usd_short=0.0, tp=0, sl=0, to=0)
    df = pd.DataFrame(trades).sort_values("t_out")
    pnl = df["pnl"].to_numpy()
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    lg = df["side"] == "long"
    return dict(n=n, per_day=n / days, win=float((pnl > 0).mean()), per=float(pnl.mean()), total=float(pnl.sum()),
                maxdd=float((peak - eq).max()), n_long=int(lg.sum()), n_short=int((~lg).sum()), usd_long=float(pnl[lg.to_numpy()].sum()),
                usd_short=float(pnl[~lg.to_numpy()].sum()), tp=int((df["reason"] == "tp").sum()), sl=int((df["reason"] == "sl").sum()),
                to=int((df["reason"] == "timeout").sum()))


def zscore(x: float, base: list[float]) -> float:
    b = np.asarray([v for v in base if np.isfinite(v)])
    if len(b) < 3 or not np.isfinite(x):
        return float("nan")
    sd = b.std(ddof=1)
    return float((x - b.mean()) / sd) if sd > 0 else float("nan")


def p_one_sided(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0)) if np.isfinite(z) else float("nan")


def bonferroni(z: float, m: int = len(VARIANTS)) -> tuple[float, float]:
    """(corrected one-sided p, 'corrected z' = the z that p/m... i.e. isf(min(1, m*p)))."""
    p = min(1.0, m * p_one_sided(z)) if np.isfinite(z) else float("nan")
    if not np.isfinite(p):
        return p, float("nan")
    from statistics import NormalDist
    return p, NormalDist().inv_cdf(1.0 - min(max(p, 1e-12), 1 - 1e-12))


# ----------------------------------------------------------------------------------------------------- baselines
def _pool_slice(pool: pd.DataFrame, t0: int, t1: int) -> pd.DataFrame:
    b = pool["bar"].to_numpy()
    return pool.iloc[np.searchsorted(b, t0): np.searchsorted(b, t1 - BAR + 1)]


def draw_random(pool: pd.DataFrame, k_long: int, k_short: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for side, k in (("long", k_long), ("short", k_short)):
        if k > 0 and len(pool):
            pick = pool.iloc[rng.choice(len(pool), size=min(k, len(pool)), replace=False)][["sym", "bar", "atr"]].copy()
            pick["side"] = side
            parts.append(pick)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["sym", "bar", "atr", "side"])
    out["sym"] = out["sym"].astype(str)
    out["prob"] = rng.random(len(out))
    return out


def draw_regime_only(pool: pd.DataFrame, mode: str, k_total: int, rng: np.random.Generator) -> pd.DataFrame:
    """Enter in the regime direction (no setup) on a random subset of the bars where the regime is not chop, same total count."""
    reg = pool[f"reg_{mode}"].to_numpy()
    sub = pool[reg != 0]
    if not len(sub) or k_total <= 0:
        return pd.DataFrame(columns=["sym", "bar", "atr", "side", "prob"])
    pick = sub.iloc[rng.choice(len(sub), size=min(k_total, len(sub)), replace=False)].copy()
    pick["side"] = np.where(pick[f"reg_{mode}"].to_numpy() > 0, "long", "short")
    pick = pick[["sym", "bar", "atr", "side"]]
    pick["sym"] = pick["sym"].astype(str)
    pick["prob"] = rng.random(len(pick))
    return pick


# ----------------------------------------------------------------------------------------------------- workers
_G: dict = {}


def _init() -> None:
    cfg = load_settings()
    meta = pickle.load(open(CACHE / "meta.pkl", "rb"))
    sig = pickle.load(open(CACHE / "signals.pkl", "rb"))
    data = {}
    for s in meta["ever"]:
        data[s] = (loader.load_klines(s), _funding(s))
    _G.update(cfg=cfg, mkt=Market(data), cands=sig["cands"], pool=sig["pool"])


def _cands_for(vkey: str, kind: str, seed: int, period: tuple[int, int]) -> pd.DataFrame:
    v = next(x for x in VARIANTS if x.key == vkey)
    c = _G["cands"][vkey]
    c = c[(c["bar"] >= period[0]) & (c["bar"] + BAR <= period[1])].copy()
    if kind == "rule":
        rng = np.random.default_rng(12345)
        c["prob"] = rng.random(len(c))                       # random tie-break among simultaneous signals (no symbol-order bias)
        return c
    pool = _pool_slice(_G["pool"][v.tf], *period)
    rng = np.random.default_rng([seed, 1000 + (0 if kind == "random" else 1), sum(map(ord, vkey))])
    if kind == "random":
        return draw_random(pool, int((c["side"] == "long").sum()), int((c["side"] == "short").sum()), rng)
    return draw_regime_only(pool, v.mode, len(c), rng)


def run_job(job: tuple) -> dict:
    vkey, kind, seed, pname = job
    period = PERIODS[pname]
    t = time.time()
    c = _cands_for(vkey, kind, seed, period)
    trades, extra = run_period(c, _G["mkt"], _G["cfg"], period)
    days = (period[1] - period[0]) / 86_400_000
    out = dict(vkey=vkey, kind=kind, seed=seed, period=pname, n_cands=len(c), secs=time.time() - t, **extra,
               summary=summarize(trades, days))
    if kind == "rule":
        out["trades"] = trades
    return out


def run_jobs(jobs: list[tuple], workers: int, out_path: Path) -> list[dict]:
    """Runs the jobs not already in ``out_path`` (resumable: an interrupted run continues where it stopped)."""
    res: list[dict] = pickle.load(open(out_path, "rb")) if out_path.exists() else []
    done = {(r["vkey"], r["kind"], r["seed"], r["period"]) for r in res}
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    if done:
        log(f"resuming: {len(done)} job(s) already in {out_path.name}, {len(jobs)} to run")
    t = time.time()
    with mp.get_context("fork").Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1)):
            res.append(r)
            if i % 10 == 0 or r["kind"] == "rule":
                log(f"job {i + 1}/{len(jobs)} {r['vkey']} {r['kind']} {r['seed']} n={r['summary']['n']} per={r['summary']['per']:.3f} "
                    f"({r['secs']:.0f}s, elapsed {time.time() - t:.0f}s)")
                pickle.dump(res, open(out_path, "wb"))
    pickle.dump(res, open(out_path, "wb"))
    return res


# ----------------------------------------------------------------------------------------------------- reporting
def table(res: list[dict], pname: str, keys: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        if keys is not None and v.key not in keys:
            continue
        rr = [r for r in res if r["vkey"] == v.key and r["period"] == pname]
        rule = next((r for r in rr if r["kind"] == "rule"), None)
        if rule is None:
            continue
        s = rule["summary"]
        rnd = [r["summary"]["per"] for r in rr if r["kind"] == "random"]
        reg = [r["summary"]["per"] for r in rr if r["kind"] == "regime"]
        z_r, z_g = zscore(s["per"], rnd), zscore(s["per"], reg)
        p_r, zc_r = bonferroni(z_r)
        p_g, zc_g = bonferroni(z_g)
        rows.append(dict(variant=v.key, n=s["n"], per_day=round(s["per_day"], 2), win=round(s["win"], 3), per=round(s["per"], 3),
                         total=round(s["total"], 1), maxdd=round(s["maxdd"], 1), L_S=f"{s['n_long']}/{s['n_short']}",
                         usd_L=round(s["usd_long"], 1), usd_S=round(s["usd_short"], 1),
                         rand_per=round(float(np.nanmean(rnd)), 3) if rnd else np.nan, z_rand=round(z_r, 2), p_bonf_rand=round(p_r, 4),
                         regime_per=round(float(np.nanmean(reg)), 3) if reg else np.nan, z_regime=round(z_g, 2),
                         p_bonf_regime=round(p_g, 4), kills=rule.get("kills", 0)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------- export
PAPER_NOTE = "PAPER TEST ONLY - "


def _row(df: pd.DataFrame, key: str) -> dict:
    r = df[df["variant"] == key]
    return r.iloc[0].to_dict() if len(r) else {}


def export_models(models_dir: Path | None = None, stamp: str | None = None, alpha: float = 0.05) -> list[Path]:
    """One ``rules`` artifact per direction mode (the dev-selected best variant of each: ``carried.json``), labelled with the
    holdout result. 'PAPER TEST ONLY' unless the holdout beats BOTH baselines with the Bonferroni-corrected p < alpha."""
    models_dir = models_dir or repo_root() / "models"
    stamp = stamp or time.strftime("%Y%m%d_%H%M")
    car = json.load(open(CACHE / "carried.json"))
    dev = table(pickle.load(open(CACHE / "dev_results.pkl", "rb")), "dev")
    hres = pickle.load(open(CACHE / "holdout_results.pkl", "rb"))
    hold, post = table(hres, "holdout", car["carried"]), table(hres, "post", car["carried"])
    out = []
    for key in car["best_per_mode"]:
        v = next(x for x in VARIANTS if x.key == key)
        d, h, q = _row(dev, key), _row(hold, key), _row(post, key)
        clears = bool(h) and h["n"] >= 30 and h["per"] > 0 and h["p_bonf_rand"] < alpha and h["p_bonf_regime"] < alpha
        name = f"rules_{v.mode}_{v.setup}_{v.tf}_{stamp}"
        fmt = lambda r: (f"n={r['n']} win={r['win']:.2f} ${r['per']:+.2f}/trade total ${r['total']:+.0f} maxDD ${r['maxdd']:.0f} "
                         f"L/S {r['L_S']}; random ${r['rand_per']:+.2f} (z={r['z_rand']:.2f}, p_bonf={r['p_bonf_rand']:.3f}), "
                         f"regime-only ${r['regime_per']:+.2f} (z={r['z_regime']:.2f}, p_bonf={r['p_bonf_regime']:.3f})") if r else "n/a"
        verdict = ("beats both baselines on the locked holdout (Bonferroni over 12 variants)" if clears else
                   "does NOT beat both baselines on the locked holdout with the Bonferroni-corrected z; watch it on paper only")
        note = ((("" if clears else PAPER_NOTE) +
                 f"rule bot {v.key}: {rules.describe(v)}. Exits 1:2 ATR(1h) (TP 1.5, SL 3), the TP closes the whole position. "
                 f"Dev 2024-11..2026-08-14: {fmt(d)}. HOLDOUT 2026-08-15..10-14: {fmt(h)}. POST 10-14..10-18: {fmt(q)}. {verdict}."))
        meta = {"name": name, "version": stamp.replace("_", "."), "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": "challenger", "kind": "rules", "arch": "rules", "parent": None, "rule_cfg": v.to_dict(),
                "timeframes": ["5m", "1h"], "history_bars_5m": v.history_bars_5m(),
                "thresholds": {"long": 0.5, "short": 0.5},
                "exit_cfg": {**EXIT_CFG, "tp_closes_fully": True, "min_tp_usd": 2.0, "min_sl_usd": 2.0},
                "entry_cfg": {"decision_every_bars": 12 if v.tf == "1h" else 1, "cooldown_bars": 6, "max_trades_per_day": None,
                              "max_open": None, "entry_mode": "taker", "decision_mode": "fixed"},
                "metrics": {"dev": d, "holdout": h, "post": q, "n_variants": len(VARIANTS), "baseline_seeds": N_SEEDS},
                "note": note}
        p = models_dir / f"{name}.json"
        tmp = models_dir / f".{name}.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1, default=float))
        tmp.replace(p)
        out.append(p)
        log("exported", p)
    return out


# ----------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "help"
    workers = int(argv[2]) if len(argv) > 2 else 5
    pd.set_option("display.width", 250, "display.max_columns", 30)
    if cmd == "build":
        build()
    elif cmd == "dev":
        jobs = [(v.key, "rule", 0, "dev") for v in VARIANTS]
        jobs += [(v.key, k, s, "dev") for v in VARIANTS for k in ("random", "regime") for s in range(N_SEEDS)]
        # heaviest first (5m variants), the rule runs of every variant before any baseline
        jobs.sort(key=lambda j: (j[1] != "rule", next(x for x in VARIANTS if x.key == j[0]).tf != "5m"))
        res = run_jobs(jobs, workers, CACHE / "dev_results.pkl")
        print(table(res, "dev").to_string(index=False))
    elif cmd == "select":
        res = pickle.load(open(CACHE / "dev_results.pkl", "rb"))
        t = table(res, "dev")
        t = t[t["n"] >= 150].copy()
        t["mode"] = t["variant"].str.split("_").str[0]
        best = [g.sort_values("z_rand", ascending=False).iloc[0]["variant"] for _, g in t.groupby("mode")]
        rest = t[~t["variant"].isin(best)].sort_values("z_rand", ascending=False)["variant"].tolist()[:2]
        carried = best + rest
        json.dump({"best_per_mode": best, "carried": carried, "rule": "dev z vs random on $/trade, n>=150"}, open(CACHE / "carried.json", "w"), indent=1)
        print(t.sort_values("z_rand", ascending=False).to_string(index=False))
        print("carried:", carried)
    elif cmd == "holdout":
        carried = json.load(open(CACHE / "carried.json"))["carried"]
        jobs = [(k, "rule", 0, p) for k in carried for p in ("holdout", "post")]
        jobs += [(k, kind, s, p) for k in carried for kind in ("random", "regime") for s in range(N_SEEDS) for p in ("holdout", "post")]
        res = run_jobs(jobs, workers, CACHE / "holdout_results.pkl")
        for p in ("holdout", "post"):
            print(p)
            print(table(res, p, carried).to_string(index=False))
    elif cmd == "export":
        export_models()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
