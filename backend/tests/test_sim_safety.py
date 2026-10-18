"""Safety & robustness layer inside the backtester: derating, plausibility, drift."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from test_sim_helpers import BAR, T0, build, flat, mk_klines, plain_cfg, sig_at
from tb.sim import run_backtest, safety
from tb.sim.signals import random_entries


def test_derate_curve_fallback_values():
    f = safety._fallback_derate
    assert f(0, 50) == 1.0 and f(20, 50) == 1.0                  # 40 % of the kill limit
    assert f(30, 50) == pytest.approx(0.75)
    assert f(40, 50) == pytest.approx(0.5)                       # 80 %
    assert f(45, 50) == pytest.approx(0.25)
    assert f(50, 50) == 0.0 and f(70, 50) == 0.0
    assert f(10, 25) == 1.0 and f(20, 25) == pytest.approx(0.5) and f(25, 25) == 0.0   # live numbers


def test_derate_matches_tb_risk_when_present():
    risk = pytest.importorskip("tb.risk")
    if not hasattr(risk, "derate_multiplier"):
        pytest.skip("tb.risk.derate_multiplier not present yet")
    assert safety.DERATE_SOURCE == "tb.risk"
    for kill in (25.0, 50.0):
        for dd in np.linspace(0, kill * 1.2, 13):
            assert safety.derate_multiplier(dd, kill) == pytest.approx(risk.derate_multiplier(dd, kill))
            assert safety._fallback_derate(dd, kill) == pytest.approx(risk.derate_multiplier(dd, kill), abs=1e-9)


def _tp_series(n=12):
    d = flat(n)
    d["h"][3] = 103.2
    return build(d)


def test_derating_scales_size_and_skips_below_min_notional(monkeypatch):
    n = 12
    kl = _tp_series(n)
    cfg = plain_cfg(derate_enabled=True, kill_dd_pct=50.0)
    monkeypatch.setattr(safety, "derate_multiplier", lambda dd, kill: 0.5)
    r = run_backtest({"X": kl}, {"X": sig_at(n, [0])}, cfg)
    t = r.trades[0]
    assert t["qty"] == pytest.approx(0.8) and t["notional"] == pytest.approx(80.0)
    assert t["gross_pnl"] == pytest.approx(2.5)                  # same price levels, half the risk
    assert t["scale"] == 0.5 and r.stats["derated_trades"] == 1
    monkeypatch.setattr(safety, "derate_multiplier", lambda dd, kill: 0.02)   # 3.2 USD notional < 5
    r = run_backtest({"X": kl}, {"X": sig_at(n, [0])}, cfg)
    assert r.trades == [] and r.stats["skip_min_notional"] == 1


def test_derating_kicks_in_after_drawdown():
    # start 100, kill 50: dd must reach 20 % (40 % of the limit) before size shrinks.
    # Lose repeatedly with a constant 100-price market whose low stops every trade out.
    n = 400
    d = flat(n)
    for i in range(n):
        d["l"][i] = 90.0
    kl = build(d)
    sigs = {"X": sig_at(n, range(0, n - 2, 3))}
    r = run_backtest({"X": kl}, sigs, plain_cfg(start_equity=100.0, kill_dd_pct=50.0, derate_enabled=True))
    scales = [t["scale"] for t in r.trades]
    assert scales[0] == 1.0
    assert min(scales) < 1.0 and r.stats["derated_trades"] > 0
    assert all(b <= a + 1e-12 for a, b in zip(scales, scales[1:]))     # only ever shrinks while losing
    assert r.stats["min_derate_mult"] < 0.5
    # without derating the same run loses faster (kill earlier / larger loss)
    r0 = run_backtest({"X": kl}, sigs, plain_cfg(start_equity=100.0, kill_dd_pct=50.0, derate_enabled=False))
    assert len(r.trades) >= len(r0.trades)


def test_plausibility_excludes_symbol_only_when_fault_matures():
    n = 40
    # 3 consecutive zero-volume bars -> matured fault; 1 bad bar -> debounced away
    v_bad = np.ones(n); v_bad[5:8] = 0.0
    kl_bad = mk_klines(flat(n)["o"], v=v_bad)
    v_one = np.ones(n); v_one[5] = 0.0
    kl_one = mk_klines(flat(n)["o"], v=v_one)
    cfg = plain_cfg(plausibility_enabled=True)
    r = run_backtest({"X": kl_bad}, {"X": sig_at(n, [7, 20])}, cfg)
    assert r.stats["skip_fault"] == 1            # row 7: fault matured on bar 7; row 20: recovered (5 good bars)
    assert r.stats["fault_episodes"] == 1 and r.stats["faults_bars"] == 3
    r1 = run_backtest({"X": kl_one}, {"X": sig_at(n, [6])}, cfg)
    assert r1.stats["skip_fault"] == 0 and r1.stats["fault_episodes"] == 0 and r1.stats["faults_bars"] == 1
    # hysteresis: right after the fault ends the symbol stays excluded until 5 good bars passed
    r2 = run_backtest({"X": kl_bad}, {"X": sig_at(n, [9, 12])}, cfg)   # rows 8..12: good bars 1..5
    assert r2.stats["skip_fault"] == 1           # row 9 is excluded (2 good bars), row 12 (5 good) is not


def test_plausibility_big_move_rule():
    n = 60
    d = flat(n)
    for i in (30, 31, 32):                       # three +30 % bars in a row
        d["o"][i] = d["c"][i] = d["h"][i] = d["l"][i] = 100 * 1.3 ** (i - 29)
    kl = build(d)
    r = run_backtest({"X": kl}, {"X": sig_at(n, [33])}, plain_cfg(plausibility_enabled=True))
    assert r.stats["faults_bars"] >= 3 and r.stats["skip_fault"] == 1


def test_drift_monitor_states_and_hysteresis():
    m = safety.DriftMonitor(30)
    for _ in range(29):
        m.add(0.8, False)
    assert m.state == 0 and m.z is None            # needs a full window
    m.add(0.8, False)
    assert m.state == 2 and m.z < -3 and m.multiplier == 0.0 and m.fired_block == 1
    # recover: wins push z up; state steps 2 -> 1 -> 0 with hysteresis
    seen = []
    for _ in range(60):
        m.add(0.8, True)
        seen.append(m.state)
    assert seen[-1] == 0
    assert 1 in seen and seen.index(1) < seen.index(0)
    # half-size band: realised 0.55 vs predicted 0.8 -> z ~ -3.4 ... use 0.6 -> z ~ -2.7
    m2 = safety.DriftMonitor(30)
    for i in range(30):
        m2.add(0.8, i % 5 < 3)                       # 60 % wins
    assert m2.state == 1 and m2.multiplier == 0.5 and -3 < m2.z < -2


def test_drift_blocks_real_entries_and_shadow_trades_continue():
    n = 600
    d = flat(n)
    for i in range(n):
        d["l"][i] = 90.0                             # every entry stops out immediately
    kl = build(d)
    sigs = {"X": sig_at(n, range(0, n - 2, 3), prob=0.9)}
    cfg = plain_cfg(drift_enabled=True, start_equity=1e6)
    r = run_backtest({"X": kl}, sigs, cfg)
    assert len(r.trades) == 30                       # drift fires after the 30th losing trade
    assert r.stats["drift_fired_block"] == 1
    assert r.stats["skip_drift_block"] > 0 and r.stats["shadow_trades"] > 0
    assert r.stats["shadow_wins"] == 0
    # off switch
    r2 = run_backtest({"X": kl}, sigs, plain_cfg(drift_enabled=False, start_equity=1e6))
    assert len(r2.trades) > 100 and r2.stats["drift_fired_block"] == 0


def test_drift_is_off_for_baselines_without_probabilities():
    n = 800
    d = flat(n)
    for i in range(n):
        d["l"][i] = 90.0
    kl = build(d)
    r = run_backtest({"X": kl}, random_entries(0.3, seed=1), plain_cfg(drift_enabled=True, start_equity=1e6))
    assert len(r.trades) > 100 and r.stats["drift_fired_block"] == 0 and r.stats["shadow_trades"] == 0
