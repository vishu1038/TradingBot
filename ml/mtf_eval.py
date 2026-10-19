"""Evaluation for the multi-timeframe experiments: trade selection with the engine's position rules PLUS a per-coin
cooldown and a trades-per-day cap, thresholds chosen on validation by top-fraction-of-score grid, random-entry and
ATR%-regime baselines under identical exits / rules, pooled statistics with a day-clustered z.

Inputs are arrays over decision rows of ONE side / exit set:
    p    score (calibrated probability of "TP first")      y    label 1/0
    pnl  net USD per trade                                 t    decision bar open_time (ms)
    sym  symbol id                                         off  bars until the trade closed
"""
from __future__ import annotations

import heapq
import statistics

import numpy as np

import evaluate as E

BAR_MS = 300_000
DAY_MS = 86_400_000
FRACTIONS = (0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.20)      # top share of scored rows that would be traded
MAX_POS = 5
DEFAULT_RULES = dict(max_pos=MAX_POS, cooldown_bars=72, max_per_day=8)


def select_trades(idx, p, t, sym, off, max_pos=MAX_POS, cooldown_bars=0, max_per_day=None) -> np.ndarray:
    """Subset of candidate positions ``idx`` that would be taken, time ordered (highest score first within a bar):
    one open trade per symbol (plus ``cooldown_bars`` after its exit), at most ``max_pos`` open overall and at most
    ``max_per_day`` entries per UTC day."""
    idx = np.asarray(idx, dtype="int64")
    if len(idx) == 0:
        return idx
    order = np.lexsort((-p[idx], t[idx]))
    idx = idx[order]
    busy: dict[int, int] = {}
    heap: list[int] = []
    per_day: dict[int, int] = {}
    keep = []
    tt, ss, oo = t[idx], sym[idx], off[idx]
    cd = int(cooldown_bars) * BAR_MS
    for k in range(len(idx)):
        now = int(tt[k])
        while heap and heap[0] <= now:
            heapq.heappop(heap)
        s = int(ss[k])
        if busy.get(s, -1) > now or len(heap) >= max_pos:
            continue
        day = now // DAY_MS
        if max_per_day is not None and per_day.get(day, 0) >= max_per_day:
            continue
        end = now + max(int(oo[k]), 1) * BAR_MS
        busy[s] = end + cd
        heapq.heappush(heap, end)
        per_day[day] = per_day.get(day, 0) + 1
        keep.append(k)
    return idx[np.asarray(keep, dtype="int64")]


def row_stats(sel, y, pnl) -> dict:
    n = len(sel)
    if n == 0:
        return dict(n=0, win_rate=None, expectancy=None, total_pnl=0.0, se=None)
    pn = pnl[sel].astype("float64")
    return dict(n=int(n), win_rate=float(y[sel].mean()), expectancy=float(pn.mean()), total_pnl=float(pn.sum()),
                se=float(pn.std(ddof=1) / np.sqrt(n)) if n > 1 else None)


def run_rule(score, th, y, pnl, t, sym, off, rules, n_random=8, seed=0, with_random=True):
    """Trade every row with ``score >= th`` under the rules; random baseline = the same number of candidates drawn
    uniformly over all rows (``n_random`` draws), same rules / exits.  Returns (stats dict, taken positions)."""
    cand = np.nonzero(score >= th)[0]
    taken = select_trades(cand, score, t, sym, off, **rules)
    r = row_stats(taken, y, pnl)
    r.update(threshold=float(th), n_signals=int(len(cand)))
    bl_e, bl_w, bl_n = [], [], []
    if with_random and len(cand):
        rng = np.random.default_rng(seed)
        allix = np.arange(len(score))
        for _ in range(n_random):
            rc = rng.choice(allix, size=min(len(cand), len(score)), replace=False)
            rs = select_trades(rc, rng.random(len(score)), t, sym, off, **rules)
            q = row_stats(rs, y, pnl)
            if q["n"]:
                bl_e.append(q["expectancy"])
                bl_w.append(q["win_rate"])
                bl_n.append(q["n"])
    r["baseline_expectancy"] = float(np.mean(bl_e)) if bl_e else None
    r["baseline_win_rate"] = float(np.mean(bl_w)) if bl_w else None
    r["baseline_n"] = float(np.mean(bl_n)) if bl_n else None
    return r, taken


def thresholds_by_fraction(score_val: np.ndarray, fractions=FRACTIONS) -> list[float]:
    """Score thresholds such that ``fraction`` of the validation rows is above (top-fraction grid)."""
    s = np.asarray(score_val, "float64")
    return [float(np.quantile(s, 1.0 - f)) for f in fractions]


def choose_threshold(score_val, y, pnl, t, sym, off, rules, fractions=FRACTIONS, min_trades=15):
    """Pick the top-fraction (as a score threshold) with the highest validation expectancy (>= ``min_trades``
    trades).  Returns (threshold, expectancy_positive, table).  Nothing from the test block is used."""
    ths = thresholds_by_fraction(score_val, fractions)
    rows = []
    for f, th in zip(fractions, ths):
        r, _ = run_rule(score_val, th, y, pnl, t, sym, off, rules, with_random=False)
        r["fraction"] = f
        rows.append(r)
    ok = [r for r in rows if r["n"] >= min_trades and r["expectancy"] is not None]
    table = [{k: r[k] for k in ("fraction", "threshold", "n", "win_rate", "expectancy")} for r in rows]
    if not ok:
        return ths[3], False, table                     # default: the 2 % row
    best = max(ok, key=lambda r: r["expectancy"])
    return best["threshold"], bool(best["expectancy"] > 0), table


def choose_rule_threshold(ref_sorted, score_val, y, pnl, t, sym, off, rules, fractions=FRACTIONS, min_trades=15):
    """Same for an ATR-regime rule whose score is the training-CDF of a volatility feature."""
    return choose_threshold(score_val, y, pnl, t, sym, off, rules, fractions, min_trades)


def cdf_scorer(train_values: np.ndarray):
    ref = np.sort(np.asarray(train_values, "float64"))
    return ref, (lambda v: np.searchsorted(ref, np.asarray(v, "float64"), side="right") / max(len(ref), 1))


def pooled(folds: list[dict], rule_key: str = "chosen") -> dict:
    """Pool per-fold trade arrays (``folds[i][rule_key]`` = dict(pnl, y, t) of the TAKEN trades) into overall stats
    with a day-clustered standard error."""
    pnl = np.concatenate([f[rule_key]["pnl"] for f in folds]) if folds else np.zeros(0)
    y = np.concatenate([f[rule_key]["y"] for f in folds]) if folds else np.zeros(0)
    t = np.concatenate([f[rule_key]["t"] for f in folds]) if folds else np.zeros(0, "int64")
    n = len(pnl)
    if n == 0:
        return dict(n=0, win_rate=None, expectancy=None, total_pnl=0.0, se=None, trades_per_day=0.0)
    se = float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(n=int(n), win_rate=float(y.mean()), expectancy=float(pnl.mean()), total_pnl=float(pnl.sum()), se=se,
                _pnl=pnl, _t=t)


def clustered_z(pnl: np.ndarray, t: np.ndarray, ref: float) -> float | None:
    """z of mean(pnl) - ref with the standard error clustered by entry day (trades of one day share the market)."""
    n = len(pnl)
    if n < 10:
        return None
    r = pnl.astype("float64") - ref
    rbar = r.mean()
    day = (np.asarray(t) // DAY_MS).astype("int64")
    _, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=r - rbar)
    se = np.sqrt((s * s).sum()) / n
    return float(rbar / se) if se > 0 else None


def bonferroni_z(n_tests: int, alpha: float = 0.05) -> float:
    return float(statistics.NormalDist().inv_cdf(1.0 - alpha / max(n_tests, 1)))
