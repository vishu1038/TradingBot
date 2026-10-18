"""JSON report writer compatible with the API's `lab_reports` shape.

{id, created_at (unix s), kind: "backtest", title, summary: {...}, equity: [{t (unix s), equity}]}
saved to var/lab/{id}.json (trades go to var/lab/{id}.trades.csv, NOT a .json, so a glob of
var/lab/*.json only sees reports).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tb.data.paths import var_dir

from .backtest import BacktestResult
from .template import breakeven_winrate


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def equity_points(res: BacktestResult, max_points: int = 2000) -> list[dict]:
    ts, eq = res.ts, res.equity
    if len(ts) == 0:
        return []
    idx = np.unique(np.linspace(0, len(ts) - 1, min(max_points, len(ts))).astype(int))
    return [{"t": int(ts[i] // 1000), "equity": round(float(eq[i]), 4)} for i in idx]


def build_report(res: BacktestResult, title: str, extra: dict | None = None,
                 atr_frac: float | None = None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    h = hashlib.sha1(f"{title}|{len(res.trades)}|{res.metrics['end_equity']:.6f}".encode()).hexdigest()[:6]
    rid = f"bt-{int(now)}-{h}"
    try:
        be = breakeven_winrate(res.cfg, atr_frac if res.cfg.exit_mode == "atr" else None)
    except ValueError:
        be = None
    summary = dict(res.metrics)
    summary.update({"symbols": res.symbols, "config": res.cfg.to_dict(), "safety": res.stats,
                    "breakeven": be, "kill_time": (res.kill_time // 1000 if res.kill_time else None)})
    if extra:
        summary.update(extra)
    return {"id": rid, "created_at": int(now), "kind": "backtest", "title": title,
            "summary": _clean(summary), "equity": equity_points(res)}


def write_report(res: BacktestResult, title: str, extra: dict | None = None,
                 atr_frac: float | None = None, out_dir: Path | None = None) -> Path:
    rep = build_report(res, title, extra, atr_frac)
    out = Path(out_dir) if out_dir else var_dir() / "lab"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rep['id']}.json"
    path.write_text(json.dumps(rep, indent=1))
    if res.trades:
        pd.DataFrame(res.trades).to_csv(out / f"{rep['id']}.trades.csv", index=False)
    return path
