"""Evaluation: AUC, calibration, and trading metrics vs. a random-entry baseline with identical exits.

All inputs are numpy arrays over *samples* (bars) of one side/mode:

    p      calibrated probability of "TP first"
    y      label (1/0)             -> win rate
    pnl    net USD per trade (fees + slippage included, from tb.labels.simulate_trades)
    t      bar open_time (ms), sym symbol id, off bars-until-exit

"Trades" are the signals that survive the same position rules the engine uses: one open position per
symbol and at most ``max_pos`` open positions overall (highest probability first within a bar).
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from tb import model_np as mnp  # noqa: E402

BAR_MS = 300_000
THRESHOLDS = (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)   # 0.3/0.4: context for low base rates


def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p, "float64")
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ps = p[order]
    # average ranks for ties
    uniq, inv, cnt = np.unique(ps, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = csum - (cnt - 1) / 2.0
    ranks[order] = avg[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def logloss(y, p, eps=1e-6) -> float:
    p = np.clip(np.asarray(p, "float64"), eps, 1 - eps)
    y = np.asarray(y, "float64")
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def select_trades(idx: np.ndarray, p: np.ndarray, t: np.ndarray, sym: np.ndarray, off: np.ndarray,
                  max_pos: int = 5, dedupe: bool = True) -> np.ndarray:
    """Subset of ``idx`` (positions into the sample arrays) that would actually be taken.
    ``idx`` are candidate signals.  Returns positions (subset of idx), time ordered."""
    if len(idx) == 0:
        return idx
    order = np.lexsort((-p[idx], t[idx]))          # by time, then highest prob first
    idx = idx[order]
    if not dedupe:
        return idx
    busy: dict[int, int] = {}
    heap: list[int] = []
    keep = []
    tt, ss, oo = t[idx], sym[idx], off[idx]
    for k in range(len(idx)):
        now = int(tt[k])
        while heap and heap[0] <= now:
            heapq.heappop(heap)
        s = int(ss[k])
        if busy.get(s, -1) > now:
            continue
        if len(heap) >= max_pos:
            continue
        end = now + max(int(oo[k]), 1) * BAR_MS
        busy[s] = end
        heapq.heappush(heap, end)
        keep.append(k)
    return idx[np.asarray(keep, dtype="int64")]


def trading_row(sel: np.ndarray, y: np.ndarray, pnl: np.ndarray) -> dict:
    n = len(sel)
    if n == 0:
        return dict(n=0, win_rate=None, expectancy=None, total_pnl=0.0, se=None)
    pn = pnl[sel].astype("float64")
    return dict(n=int(n), win_rate=float(y[sel].mean()), expectancy=float(pn.mean()), total_pnl=float(pn.sum()),
                se=float(pn.std(ddof=1) / np.sqrt(n)) if n > 1 else None)


def trading_table(p, y, pnl, t, sym, off, be_win_rate: float, thresholds=THRESHOLDS, max_pos: int = 5,
                  n_random: int = 5, seed: int = 0) -> list[dict]:
    """One row per threshold: model trades vs. random entries with the same count and exits."""
    rng = np.random.default_rng(seed)
    n = len(p)
    all_idx = np.arange(n)
    # population baseline: every bar is a trade (no position rules) -> E[pnl], win rate
    pop = dict(win_rate=float(y.mean()), expectancy=float(pnl.mean())) if n else dict(win_rate=None, expectancy=None)
    rows = []
    for th in thresholds:
        cand = np.nonzero(p >= th)[0]
        sel = select_trades(cand, p, t, sym, off, max_pos)
        m = trading_row(sel, y, pnl)
        m["threshold"] = float(th)
        m["n_signals"] = int(len(cand))
        # random-entry baseline: same number of candidate signals drawn uniformly, same position rules
        bl_win, bl_exp, bl_n = [], [], []
        if len(cand):
            for _ in range(n_random):
                rc = rng.choice(all_idx, size=min(len(cand), n), replace=False)
                rs = select_trades(rc, rng.random(n), t, sym, off, max_pos)
                r = trading_row(rs, y, pnl)
                if r["n"]:
                    bl_win.append(r["win_rate"])
                    bl_exp.append(r["expectancy"])
                    bl_n.append(r["n"])
        m["baseline_win_rate"] = float(np.mean(bl_win)) if bl_win else None
        m["baseline_expectancy"] = float(np.mean(bl_exp)) if bl_exp else None
        m["baseline_n"] = float(np.mean(bl_n)) if bl_n else None
        m["population_win_rate"] = pop["win_rate"]
        m["population_expectancy"] = pop["expectancy"]
        m["breakeven_win_rate"] = be_win_rate
        if m["n"] > 1 and m["se"] and m["baseline_expectancy"] is not None:
            m["z_vs_baseline"] = float((m["expectancy"] - m["baseline_expectancy"]) / m["se"])
            m["z_vs_zero"] = float(m["expectancy"] / m["se"])
        else:
            m["z_vs_baseline"] = m["z_vs_zero"] = None
        rows.append(m)
    return rows


def choose_threshold(rows: list[dict], min_trades: int = 30, default: float = 0.7) -> tuple[float, bool]:
    """Pick the threshold with the highest expectancy (needs >= min_trades de-overlapped trades) from
    rows computed on VALIDATION data.  Returns (threshold, expectancy_positive)."""
    ok = [r for r in rows if r["n"] >= min_trades and r["expectancy"] is not None]
    if not ok:
        return default, False
    best = max(ok, key=lambda r: r["expectancy"])
    return best["threshold"], bool(best["expectancy"] > 0)


def calibration_report(y, p_raw, p_cal) -> dict:
    return dict(
        n=int(len(y)), base_rate=float(np.mean(y)) if len(y) else None,
        auc=auc(y, p_raw), logloss_raw=logloss(y, p_raw), logloss_cal=logloss(y, p_cal),
        ece_raw=mnp.ece(p_raw, y), ece_cal=mnp.ece(p_cal, y),
        mean_pred_raw=float(np.mean(p_raw)), mean_pred_cal=float(np.mean(p_cal)),
        reliability_raw=mnp.reliability_bins(p_raw, y, 10), reliability_cal=mnp.reliability_bins(p_cal, y, 10))


def drift_z(p_taken: np.ndarray, wins: np.ndarray, n: int = 30) -> np.ndarray:
    """Rolling z of the live drift detector over consecutive windows of ``n`` trades (in time order):
    z = (win_rate - mean_pred) / sqrt(mean_pred (1 - mean_pred) / n)."""
    p_taken = np.asarray(p_taken, "float64")
    wins = np.asarray(wins, "float64")
    if len(p_taken) < n:
        return np.zeros(0)
    k = np.ones(n) / n
    pm = np.convolve(p_taken, k, mode="valid")
    wm = np.convolve(wins, k, mode="valid")
    sd = np.sqrt(np.clip(pm * (1 - pm), 1e-6, None) / n)
    return (wm - pm) / sd


def drift_firing(p_taken, wins, n: int = 30) -> dict:
    z = drift_z(p_taken, wins, n)
    if len(z) == 0:
        return dict(windows=0, frac_z_lt_m2=None, frac_z_lt_m3=None, mean_z=None, mean_gap=None)
    return dict(windows=int(len(z)), frac_z_lt_m2=float((z < -2).mean()), frac_z_lt_m3=float((z < -3).mean()),
                mean_z=float(z.mean()),
                mean_gap=float(np.mean(wins) - np.mean(p_taken)))
