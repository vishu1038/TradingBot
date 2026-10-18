"""Random-entry baseline sanity on a driftless synthetic market, report writer and CLI."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

from test_sim_helpers import BAR, T0, mk_klines, plain_cfg
from tb.data import frames
from tb.data.paths import kline_path
from tb.sim import SimConfig, always_long_every_n, breakeven_winrate, random_entries, run_backtest
from tb.sim import report, run as sim_run


def _gbm(n, seed, sigma=0.002):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, sigma, n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.concatenate([[100.0], c[:-1]])
    wick = np.abs(rng.normal(0, sigma * 0.5, n))
    return mk_klines(o, np.maximum(o, c) * (1 + wick), np.minimum(o, c) * (1 - wick), c)


def test_random_entries_fixed_exit_wins_about_two_thirds_and_loses_after_costs():
    kls = {f"S{i}": _gbm(60_000, i) for i in range(4)}
    cfg = SimConfig(start_equity=1e6, kill_dd_pct=100.0, max_positions=5)
    r = run_backtest(kls, random_entries(0.004, seed=7), cfg)
    m = r.metrics
    assert m["n_trades"] > 250
    assert 0.62 < m["win_rate"] < 0.71                      # zero-cost win rate is 66.7 %
    be = breakeven_winrate(cfg)["breakeven_winrate"]
    assert be > 0.67 and m["win_rate"] < be + 0.02          # fees put break-even above the random rate
    assert m["expectancy"] < 0.5 and m["fees_paid"] > 0
    assert m["tp_rate"] == pytest.approx(m["win_rate"])     # fixed exits: win <=> TP


def test_random_entries_are_deterministic_per_seed_and_symbol():
    kl = _gbm(2000, 1)
    g = random_entries(0.05, seed=3)
    a1, b1 = g("A", kl)
    a2, b2 = g("A", kl)
    a3, _ = g("B", kl)
    assert np.array_equal(a1, a2) and not np.array_equal(a1, a3)
    assert 0.03 < a1.mean() < 0.07 and b1.sum() == 0
    both_l, both_s = random_entries(0.1, seed=1, side="both")("A", kl)
    assert both_l.sum() > 0 and both_s.sum() > 0 and (both_l * both_s).sum() == 0
    l, _ = always_long_every_n(10, offset=3)("A", kl)
    assert l.sum() == 200 and l[3] == 1 and l[13] == 1 and l[4] == 0


def test_report_shape_matches_lab_reports(tmp_path):
    kls = {"S0": _gbm(5000, 0)}
    r = run_backtest(kls, always_long_every_n(40), plain_cfg())
    path = report.write_report(r, "unit test", out_dir=tmp_path)
    rep = json.loads(path.read_text())
    assert set(["id", "created_at", "kind", "title", "summary", "equity"]) <= set(rep)
    assert rep["kind"] == "backtest" and rep["title"] == "unit test" and isinstance(rep["created_at"], int)
    assert rep["equity"] and set(rep["equity"][0]) == {"t", "equity"}
    assert rep["equity"][0]["t"] == T0 // 1000                         # unix SECONDS
    assert len(rep["equity"]) <= 2000
    s = rep["summary"]
    for k in ("total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe", "sortino", "win_rate", "avg_win",
              "avg_loss", "profit_factor", "expectancy", "fees_paid", "funding_paid", "exposure_pct",
              "trades_per_day", "n_trades", "breakeven", "safety", "config"):
        assert k in s, k
    assert (tmp_path / f"{rep['id']}.trades.csv").exists()
    t = pd.read_csv(tmp_path / f"{rep['id']}.trades.csv")
    for k in ("id", "symbol", "side", "entry_price", "exit_price", "qty", "opened_at", "closed_at", "pnl",
              "fees", "funding", "exit_reason", "prob"):
        assert k in t.columns, k


def test_metrics_values():
    # two trades: +TP, -SL on a flat market with scripted bars
    from test_sim_helpers import flat, build, sig_at
    d = flat(40)
    d["h"][3] = 103.2                       # TP for signal at 0
    d["l"][13] = 90.0                       # SL for signal at 10
    r = run_backtest({"X": build(d)}, {"X": sig_at(40, [0, 10])}, plain_cfg())
    m = r.metrics
    assert m["n_trades"] == 2 and m["win_rate"] == 0.5
    assert m["avg_win"] == pytest.approx(5 - 0.1625) and m["avg_loss"] == pytest.approx(-10 - 0.08 - 0.075)
    assert m["profit_factor"] == pytest.approx(m["avg_win"] / -m["avg_loss"])
    assert m["fees_paid"] == pytest.approx(0.1625 + 0.155)
    assert m["expectancy"] == pytest.approx((m["avg_win"] + m["avg_loss"]) / 2)
    assert m["exposure_pct"] == pytest.approx(100.0 * 4 / 40)   # position open at the close of 4 of 40 steps
    assert m["max_drawdown_pct"] > 0 and m["total_return_pct"] < 0


def test_cli_on_a_tiny_archive(tmp_path, monkeypatch, capsys):
    root = tmp_path / "arch"
    for i, s in enumerate(["AAAUSDT", "BBBUSDT"]):
        frames.write_parquet_atomic(frames.normalize_klines(_gbm(4000, i)), kline_path(s, root))
    monkeypatch.setenv("TB_ARCHIVE_DIR", str(root))
    monkeypatch.setenv("TB_VAR_DIR", str(tmp_path / "var"))
    rc = sim_run.main(["--signals", "random", "--p", "0.01", "--exit", "atr", "--trail"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "trades" in out and "report:" in out
    files = list((tmp_path / "var" / "lab").glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["kind"] == "backtest"
