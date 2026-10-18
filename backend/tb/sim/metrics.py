"""Backtest metrics from the trade list and the 5m equity curve."""
from __future__ import annotations

import math

import numpy as np

DAY_MS = 86_400_000


def daily_equity(ts: np.ndarray, eq: np.ndarray, start_equity: float) -> np.ndarray:
    """Equity at the last step of each UTC day, prefixed by the start equity."""
    if len(ts) == 0:
        return np.array([start_equity])
    day = ts // DAY_MS
    last = np.nonzero(np.diff(day, append=day[-1] + 1))[0]
    return np.concatenate([[start_equity], eq[last]])


def compute_metrics(trades: list[dict], ts: np.ndarray, eq: np.ndarray, cfg, open_steps: int) -> dict:
    start = float(cfg.start_equity)
    end = float(eq[-1]) if len(eq) else start
    n_steps = len(ts)
    days = (ts[-1] - ts[0] + 300_000) / DAY_MS if n_steps else 0.0
    m: dict = {"start_equity": start, "end_equity": end,
               "total_return_pct": (end / start - 1.0) * 100.0, "days": days}
    if days > 0 and end > 0:
        m["cagr_pct"] = ((end / start) ** (365.0 / days) - 1.0) * 100.0
    else:
        m["cagr_pct"] = float("nan") if end > 0 else -100.0
    if n_steps:
        run_peak = np.maximum.accumulate(np.maximum(eq, start))
        dd = (run_peak - eq) / run_peak
        m["max_drawdown_pct"] = float(dd.max() * 100.0)
    else:
        m["max_drawdown_pct"] = 0.0
    de = daily_equity(ts, eq, start)
    if len(de) > 2:
        r = np.diff(de) / np.where(de[:-1] > 0, de[:-1], np.nan)
        r = r[np.isfinite(r)]
        sd = r.std(ddof=1) if len(r) > 1 else 0.0
        m["sharpe"] = float(r.mean() / sd * math.sqrt(365)) if sd > 0 else float("nan")
        dn = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2))) if len(r) else 0.0
        m["sortino"] = float(r.mean() / dn * math.sqrt(365)) if dn > 0 else float("nan")
    else:
        m["sharpe"] = m["sortino"] = float("nan")
    n = len(trades)
    m["n_trades"] = n
    m["trades_per_day"] = n / days if days > 0 else 0.0
    m["exposure_pct"] = open_steps / n_steps * 100.0 if n_steps else 0.0
    if n:
        pnl = np.array([t["pnl"] for t in trades])
        win = pnl > 0
        m["win_rate"] = float(win.mean())
        m["avg_win"] = float(pnl[win].mean()) if win.any() else 0.0
        m["avg_loss"] = float(pnl[~win].mean()) if (~win).any() else 0.0
        gw, gl = pnl[win].sum(), -pnl[~win].sum()
        m["profit_factor"] = float(gw / gl) if gl > 0 else float("inf")
        m["expectancy"] = float(pnl.mean())
        m["total_pnl"] = float(pnl.sum())
        m["fees_paid"] = float(sum(t["fees"] for t in trades))
        m["funding_paid"] = float(sum(t["funding"] for t in trades))
        m["gross_pnl"] = float(sum(t["gross_pnl"] for t in trades))
        reasons: dict[str, int] = {}
        for t in trades:
            reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
        m["exit_reasons"] = reasons
        m["tp_rate"] = reasons.get("tp", 0) / n
        m["avg_notional"] = float(np.mean([t["notional"] for t in trades]))
        m["avg_bars_held"] = float(np.mean([t["bars_held"] for t in trades]))
        m["long_trades"] = sum(1 for t in trades if t["side"] == "long")
        m["short_trades"] = n - m["long_trades"]
    else:
        for k in ("win_rate", "avg_win", "avg_loss", "expectancy", "total_pnl", "fees_paid",
                  "funding_paid", "gross_pnl", "tp_rate", "avg_notional", "avg_bars_held"):
            m[k] = 0.0
        m["profit_factor"] = float("nan")
        m["exit_reasons"] = {}
        m["long_trades"] = m["short_trades"] = 0
    return m
