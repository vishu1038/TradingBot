"""tb.sim.xsec: market-neutral portfolio simulator on synthetic worlds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tb import features_xsec as fx  # noqa: E402
from tb.sim import xsec as X  # noqa: E402
from tb.sim.execution import ExecConfig  # noqa: E402
from test_v3_xsec_world import BAR, HOUR, make_world  # noqa: E402

K0 = 130          # first decision hour (sigma needs 120 h of history)


def rand_scores(p, seed=0):
    return np.random.default_rng(seed).random((p.T, p.S))


def run(p, score, **kw):
    cfg = X.XsecConfig(**{"rebalance_every_h": 24, "top_k": 3, "bottom_k": 3, **kw})
    return X.run_xsec(p, score, cfg, K0, p.T - 1)


@pytest.fixture(scope="module")
def flat():
    return fx.build_panel(make_world(S=12, days=14, seed=11, flat_noise=1e-4, funding_rate=None))


def test_taker_round_trip_cost_in_a_flat_world(flat):
    r = run(flat, rand_scores(flat))
    assert len(r.trades) >= 20 and r.kill_time is None
    ratio = np.array([t["pnl"] / t["notional"] for t in r.trades])
    # entry slippage 2 bp + exit slippage 2 bp + 2 x 5 bp fee = 14 bp of notional per leg (+- the 1 bp price noise)
    assert -0.0016 < ratio.mean() < -0.0012
    assert all(t["exit_reason"] in ("rebalance", "end") for t in r.trades)
    assert r.metrics["fees_paid"] == pytest.approx(sum(t["fees"] for t in r.trades))
    assert r.metrics["net_return_pct"] < 0
    # PnL convention: pnl = gross - fees - funding
    for t in r.trades[:10]:
        assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["fees"] - t["funding"])
        assert t["fees"] >= 0


def test_maker_is_cheaper_but_does_not_always_fill(flat):
    sc = rand_scores(flat, 1)
    taker = run(flat, sc, entry_mode="taker")
    maker = run(flat, sc, entry_mode="maker")
    assert maker.metrics["fees_paid"] < taker.metrics["fees_paid"] * 0.6
    assert maker.stats["maker_entries"] > 0 and maker.stats["maker_exits"] > 0
    ratio = np.mean([t["pnl"] / t["notional"] for t in maker.trades])
    assert -0.0008 < ratio < -0.0001                                 # ~4 bp of fees, no slippage
    # a stricter fill rule can only reduce fills
    strict = X.run_xsec(flat, sc, X.XsecConfig(rebalance_every_h=24, top_k=3, bottom_k=3, entry_mode="maker",
                                              exec=ExecConfig(through_pct=0.01)), K0, flat.T - 1)
    assert strict.stats["entries"] < maker.stats["entries"] or strict.stats["entry_no_fill"] > 0


def test_low_vol_gate_blocks_trades_when_fees_dominate(flat):
    sc = rand_scores(flat, 4)
    base = run(flat, sc)
    gated = run(flat, sc, gate_min_ratio=3.0)
    assert base.stats["entries"] > 0
    assert gated.stats["entries"] == 0 and gated.stats["skip_gate"] > 0 and not gated.trades
    assert gated.metrics["net_return_pct"] == 0.0


def test_book_is_dollar_neutral(flat):
    r = run(flat, rand_scores(flat, 2))
    assert r.metrics["avg_net_exposure_frac"] < 0.15
    assert r.metrics["long_trades"] > 0 and r.metrics["short_trades"] > 0
    assert 0 < r.metrics["avg_gross_leverage"] <= 20


def test_keep_existing_legs_do_not_trade():
    p = fx.build_panel(make_world(S=12, days=12, seed=12, flat_noise=1e-4))     # 158 h < max_hold_h
    sc = np.tile(np.arange(p.S, dtype="float64"), (p.T, 1))        # the same ranking forever
    r = run(p, sc, rebalance_every_h=12)
    assert r.stats["kept_legs"] > 0
    assert len(r.trades) <= 6 and all(t["exit_reason"] == "end" for t in r.trades)    # held to the end, 6 legs
    # exits at the end are charged exactly one more taker fee
    longs = {t["symbol"] for t in r.trades if t["side"] == "long"}
    assert longs == {p.syms[i] for i in range(p.S - 3, p.S)}


def test_oracle_makes_money_and_beats_random():
    S = 14
    drift = np.linspace(-1.5e-4, 1.5e-4, S)
    p = fx.build_panel(make_world(S=S, days=16, seed=13, drift=drift, vol=0.002))
    oracle = np.tile(drift, (p.T, 1))
    ro = run(p, oracle)
    rr = [run(p, rand_scores(p, s)).metrics["net_return_pct"] for s in range(6)]
    rn = run(p, -oracle)
    assert ro.metrics["net_return_pct"] > 0 and ro.metrics["sharpe"] > 0
    assert ro.metrics["net_return_pct"] > max(rr)
    assert rn.metrics["net_return_pct"] < 0
    z = X.z_vs_random(ro.metrics["net_return_pct"], rr)
    assert z > 2
    # a strong market-wide move does not leak into the market-neutral book: all coins +,  book roughly unaffected
    up = np.full(S, 6e-5)
    p2 = fx.build_panel(make_world(S=S, days=16, seed=14, drift=up, vol=0.002))
    r2 = run(p2, rand_scores(p2, 3))
    assert abs(r2.metrics["net_return_pct"]) < 25               # a +50 %/16 d long-only book would be far above


def test_stop_fills_at_the_gap_and_reports_sl():
    w = make_world(S=12, days=14, seed=15, flat_noise=1e-4, crash={5: (12 * (K0 + 6) + 3, 0.7)})
    p = fx.build_panel(w)
    sc = np.zeros((p.T, p.S))
    sc[:, 5] = 10.0                                              # the crashing coin is always the top pick (long)
    r = run(p, sc, rebalance_every_h=48)
    t = next(t for t in r.trades if t["symbol"] == p.syms[5])
    assert t["side"] == "long" and t["exit_reason"] == "sl"
    assert t["exit_price"] < t["entry_price"] * 0.8              # filled through the gap, not at the stop level
    assert t["pnl"] < 0 and r.stats["stops"] >= 1


def test_kill_switch_stops_trading():
    w = make_world(S=12, days=14, seed=16, flat_noise=1e-4, crash={5: (12 * (K0 + 6) + 3, 0.7)})
    p = fx.build_panel(w)
    sc = np.zeros((p.T, p.S))
    sc[:, 5] = 10.0
    r = run(p, sc, rebalance_every_h=12, kill_dd_pct=15.0)
    assert r.kill_time is not None
    assert r.metrics["max_drawdown_pct"] >= 15.0
    assert all(t["opened_at"] <= r.kill_time // 1000 for t in r.trades)
    assert r.stats["entries"] > 0
    # after the kill the equity is flat
    after = r.equity[r.ts > r.kill_time]
    assert len(after) > 5 and np.ptp(after) == 0


def test_derating_shrinks_new_positions_after_losses():
    w = make_world(S=12, days=14, seed=17, flat_noise=1e-4, crash={5: (12 * (K0 + 6) + 3, 0.7)})
    p = fx.build_panel(w)
    sc = np.zeros((p.T, p.S))
    sc[:, 5] = 10.0
    on = run(p, sc, rebalance_every_h=12, kill_dd_pct=50.0, derate_enabled=True)
    off = run(p, sc, rebalance_every_h=12, kill_dd_pct=50.0, derate_enabled=False)
    assert on.stats["derated_rebalances"] > 0 and on.stats["min_derate_mult"] < 1.0
    assert off.stats["derated_rebalances"] == 0


def test_funding_sign_and_size():
    p = fx.build_panel(make_world(S=12, days=14, seed=18, flat_noise=1e-4, funding_rate=0.001))
    r = run(p, rand_scores(p, 5), rebalance_every_h=24)
    lg = [t for t in r.trades if t["side"] == "long"]
    sh = [t for t in r.trades if t["side"] == "short"]
    assert lg and sh and all(t["funding"] > 0 for t in lg) and all(t["funding"] < 0 for t in sh)
    for t in r.trades:
        n_settle = abs(t["funding"]) / (t["notional"] * 0.001)
        assert abs(n_settle - t["hold_h"] / 8.0) <= 1.01
    off = run(p, rand_scores(p, 5), rebalance_every_h=24, apply_funding=False)
    assert off.metrics["funding_paid"] == 0.0


def test_small_universe_skips_and_bad_inputs(flat):
    sc = rand_scores(flat)
    sc[:, 3:] = np.nan                                            # 3 eligible coins only
    r = run(flat, sc)
    assert r.stats["rebalances"] == 0 and r.stats["rebalances_skipped_small_universe"] > 0 and not r.trades
    with pytest.raises(ValueError):
        X.run_xsec(flat, np.zeros((3, 3)), X.XsecConfig(), K0, flat.T - 1)
    with pytest.raises(ValueError):
        X.run_xsec(flat, sc, X.XsecConfig(), 10, 11)


def test_deterministic(flat):
    a = run(flat, rand_scores(flat, 9), entry_mode="maker")
    b = run(flat, rand_scores(flat, 9), entry_mode="maker")
    assert np.array_equal(a.equity, b.equity) and a.trades == b.trades


def test_pooled_metrics_and_z():
    class R:  # minimal result stand-in
        def __init__(self, d, kill=None):
            self.daily_ret = np.asarray(d)
            self.trades = [{"pnl": 1.0}]
            self.metrics = {"days": len(d), "fees_paid": 1.0, "funding_paid": 0.0, "gross_pnl": 2.0,
                            "turnover_x_per_day": 1.0}
            self.kill_time = kill
    pm = X.pooled_metrics([R([0.01, 0.02, -0.01]), R([0.0, 0.03, 0.01], kill=5)])
    assert pm["net_return_pct"] == pytest.approx(((1.01 * 1.02 * 0.99 * 1.0 * 1.03 * 1.01) - 1) * 100)
    assert pm["kills"] == 1 and pm["n_trades"] == 2 and pm["days"] == 6
    assert X.z_vs_random(10.0, [0.0, 2.0, -2.0, 1.0, -1.0]) > 4
    assert np.isnan(X.z_vs_random(1.0, [1.0, 1.0, 1.0]))
