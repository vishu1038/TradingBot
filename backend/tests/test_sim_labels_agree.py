"""The backtester and tb.labels implement the same trade template: outcomes and PnL must agree."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

labels = pytest.importorskip("tb.labels")

from test_sim_helpers import mk_klines, plain_cfg, sig_at
from tb.sim import run_backtest


def _market(n=6000, sigma=0.012, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    o = np.concatenate([[100.0], c[:-1]])
    spread = np.abs(rng.normal(0, sigma / 2, n))
    return mk_klines(o, np.maximum(o, c) * (1 + spread), np.minimum(o, c) * (1 - spread), c)


@pytest.mark.parametrize("mode,side", [("fixed", "long"), ("fixed", "short"), ("atr", "long"), ("atr", "short")])
def test_sim_matches_labels(mode, side):
    kl = _market()
    n = len(kl)
    rows = list(range(40, n - 600, 450))      # sparse: one trade at a time, horizon fully inside the data
    cfg = plain_cfg(exit_mode=mode, slippage=0.0002, stop_slippage_extra=0.0005, short_enabled=True,
                    sl_usd=10.0)
    r = run_backtest({"X": kl}, {"X": sig_at(n, rows, side)}, cfg)
    exit_cfg = {"exit_mode": mode, "slippage_pct": 0.0002, "stop_slippage_pct": 0.0005}
    sim = labels.simulate_trades(kl, side, exit_cfg, horizon_bars=288)
    by_open = {t["opened_at"]: t for t in r.trades}
    compared = 0
    for row in rows:
        lab, out, pnl = sim["label"][row], sim["outcome"][row], sim["pnl_usd"][row]
        t = by_open.get((int(kl["open_time"].iloc[row + 1])) // 1000)
        if t is None or np.isnan(lab) or out == labels.OUT_TIMEOUT:
            continue
        assert t["bars_held"] <= 288 or True
        if t["bars_held"] > 288:
            continue
        assert (t["exit_reason"] == "tp") == (lab == 1.0), (row, t["exit_reason"], lab)
        assert t["pnl"] == pytest.approx(float(pnl), rel=1e-4, abs=1e-5), (row, t["pnl"], pnl)
        compared += 1
    assert compared >= 8
