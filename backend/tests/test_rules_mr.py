"""Mean-reversion family (setup S4, modes chop / any): conditions, no look-ahead, regime gating, window independence,
research == live parity, the unchanged trend family, protection inertness and the MR research harness helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tb import rules as R
from tb.config import Settings
from tb.exits import resolve_exit
from tb.scorer import FrameBundle, load_model, scan_models, score_frames
from tb.sim import rules_research as RR
from tb.sim import rules_research_mr as MR

from test_rules import BAR, T0, artifact, synth

MR_VARIANTS = [R.RuleCfg(m, "S4", tf) for m in R.MR_MODES for tf in R.TFS]


@pytest.fixture(scope="module")
def coin():
    return synth(20_000, seed=5, drift_scale=1e-3)


@pytest.fixture(scope="module")
def sigs(coin):
    return {v.key: R.rule_signals(coin, None, v) for v in MR_VARIANTS}


def test_family_definition_and_trend_family_untouched():
    assert [v.key for v in MR_VARIANTS] == ["mr_chop_5m", "mr_chop_1h", "mr_any_5m", "mr_any_1h"]
    assert [v.key for v in MR.MR_VARIANTS] == [v.key for v in MR_VARIANTS] and MR.N_MR == 4
    assert R.MODES == ("btc", "coin") and R.SETUPS == ("S1", "S2", "S3") and len(RR.VARIANTS) == 12
    assert RR.VARIANTS[0].key == "btc_S1_5m"
    assert R.RuleCfg("btc", "S1", "5m").history_bars_5m() == 8712         # the trend bots' history need did not move
    for bad in (("coin", "S4", "5m"), ("chop", "S1", "5m"), ("any", "S3", "1h")):
        with pytest.raises(ValueError):
            R.RuleCfg(*bad)
    c = R.RuleCfg("chop", "S4", "1h")
    assert R.RuleCfg.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    assert "mean reversion" in R.describe(c) and "chop" in R.describe(c) and "no regime gate" in R.describe(R.RuleCfg("any", "S4", "5m"))


def test_s4_conditions_hold_at_every_signal_5m(coin, sigs):
    c, h, l = (coin[k].to_numpy() for k in ("close", "high", "low"))
    e = R.ema(c, 20)
    atr = R.atr_sma(h, l, c, 14)
    mu, sd = pd.Series(c).rolling(20).mean().to_numpy(), pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    z = (c - mu) / sd
    r = R.rsi(c, 14)
    for key in ("mr_any_5m", "mr_chop_5m"):
        s = sigs[key]
        assert not (s.long & s.short).any()
        assert s.long.sum() > 20 and s.short.sum() > 20, key
        for i in np.flatnonzero(s.long):
            assert c[i] < e[i] - 2 * atr[i] and z[i] <= -2.0 and r[i] < 25.0, (key, int(i))
        for i in np.flatnonzero(s.short):
            assert c[i] > e[i] + 2 * atr[i] and z[i] >= 2.0 and r[i] > 75.0, (key, int(i))
    # the conditions are also SUFFICIENT: every bar meeting them is a signal of the ungated variant
    fin = np.isfinite(e) & np.isfinite(atr) & np.isfinite(z) & np.isfinite(r)
    want_l = fin & (c < e - 2 * atr) & (z <= -2.0) & (r < 25.0)
    assert np.array_equal(want_l, sigs["mr_any_5m"].long)


def test_s4_conditions_hold_at_every_signal_1h(coin, sigs):
    b = R.bars_1h(coin)
    e, atr = R.ema(b.c, 20), R.atr_sma(b.h, b.l, b.c, 14)
    z = (b.c - pd.Series(b.c).rolling(20).mean().to_numpy()) / pd.Series(b.c).rolling(20).std(ddof=0).to_numpy()
    r = R.rsi(b.c, 14)
    s = sigs["mr_any_1h"]
    tc = coin["open_time"].to_numpy() + BAR
    for side in ("long", "short"):
        rows = np.flatnonzero(getattr(s, side))
        assert len(rows) > 10 and (tc[rows] % R.HOUR_MS == 0).all()
        for i in rows:
            j = int(np.searchsorted(b.t_end, tc[i]))
            assert b.t_end[j] == tc[i]
            if side == "long":
                assert b.c[j] < e[j] - 2 * atr[j] and z[j] <= -2.0 and r[j] < 25.0
            else:
                assert b.c[j] > e[j] + 2 * atr[j] and z[j] >= 2.0 and r[j] > 75.0


def test_no_lookahead_prefix_equals_full(coin, sigs):
    cut = 15_000
    for v in MR_VARIANTS:
        pre = R.rule_signals(coin.iloc[:cut + 1].reset_index(drop=True), None, v)
        full = sigs[v.key]
        assert np.array_equal(pre.long, full.long[:cut + 1]), v.key
        assert np.array_equal(pre.short, full.short[:cut + 1]), v.key
        assert np.array_equal(pre.regime, full.regime[:cut + 1]), v.key


def test_chop_gate_uses_the_coins_own_regime_and_never_trades_unknown(coin, sigs):
    b = R.bars_1h(coin)
    reg = R.coin_regime_valid(b.c)
    tc = coin["open_time"].to_numpy() + BAR
    s = sigs["mr_chop_5m"]
    assert (s.regime[:R.ema_taps(200) * 12 - 1] == 2).all()                # young history -> unknown, never chop
    for i in np.flatnonzero(s.long | s.short):
        j = np.searchsorted(b.t_end, tc[i], side="right") - 1
        assert b.t_end[j] <= tc[i] and reg[j] == 0 and s.regime[i] == 0
    any_ = sigs["mr_any_5m"]
    assert not (s.long & ~any_.long).any() and not (s.short & ~any_.short).any()      # chop is a subset of any
    gated_out = (any_.long | any_.short) & ~(s.long | s.short)
    assert gated_out.sum() > 0 and (s.regime[gated_out] != 0).all()                    # exactly the trending / unknown bars are dropped
    assert not any(sigs[k].long[:R.ema_taps(200) * 12 - 1].any() for k in ("mr_chop_5m", "mr_chop_1h"))
    # a coin with too little history: any trades, chop does not
    short = coin.iloc[:3000].reset_index(drop=True)
    assert not R.rule_signals(short, None, R.RuleCfg("chop", "S4", "5m")).long.any()
    assert R.rule_signals(short, None, R.RuleCfg("any", "S4", "5m")).long.sum() > 0
    known = reg != 2
    assert known.sum() > 0 and (reg[known] == R.coin_regime(b.c)[known]).all() and (reg[~known] == 2).all()


def test_window_independence_tail_equals_full(coin, sigs):
    rng = np.random.default_rng(11)
    for v in MR_VARIANTS:
        n = v.history_bars_5m()
        full = sigs[v.key]
        hit = np.flatnonzero(full.long | full.short)
        assert len(hit) > 0
        rows = [i for i in list(rng.choice(hit, size=min(10, len(hit)), replace=False)) + list(rng.integers(n, len(coin), size=12)) if i + 1 >= n]
        for i in rows:
            tail = R.rule_signals(coin.iloc[i + 1 - n:i + 1].reset_index(drop=True), None, v)
            assert (bool(tail.long[-1]), bool(tail.short[-1]), int(tail.regime[-1])) == \
                   (bool(full.long[i]), bool(full.short[i]), int(full.regime[i])), (v.key, int(i))


@pytest.mark.parametrize("v", MR_VARIANTS, ids=lambda v: v.key)
def test_live_scorer_equals_research_arrays(v, coin, sigs, tmp_path):
    artifact(v, tmp_path)
    sc, info = load_model(scan_models(tmp_path)[0])
    assert info.kind == "rules" and info.history_bars_5m == v.history_bars_5m()
    full = sigs[v.key]
    idx = np.flatnonzero(full.long | full.short)
    assert len(idx) > 0
    rng = np.random.default_rng(6)
    pick = list(idx[rng.choice(len(idx), size=min(8, len(idx)), replace=False)]) + list(rng.integers(9000, len(coin) - 1, size=10)) + [len(coin) - 1]
    for i in pick:
        lo = max(0, i + 1 - 12_000)
        fb = FrameBundle("ETHUSDT", coin.iloc[lo:i + 1].reset_index(drop=True), btc_kl=None)       # no BTC data needed for MR
        s = score_frames(sc, info, {"ETHUSDT": fb})["ETHUSDT"]
        want = (1.0, 0.0) if full.long[i] else (0.0, 1.0) if full.short[i] else (0.0, 0.0)
        assert (s.prob_long, s.prob_short) == want, (v.key, int(i))
        if want != (0.0, 0.0):
            assert s.reasons[0].feature == "regime" and "mean reversion" in s.reasons[1].label


def test_exit_and_protection_are_the_framework_ones(tmp_path):
    v = MR_VARIANTS[0]
    spec = resolve_exit(json.loads(artifact(v, tmp_path).read_text())["exit_cfg"], "rules", "x")
    assert spec == RR.SPEC and RR.protection_inert(Settings(), spec)
    assert RR.EXIT_CFG["tp_atr"] == 1.5 and RR.EXIT_CFG["sl_atr"] == 3.0 and RR.EXIT_CFG["horizon_bars"] == 1152


# ------------------------------------------------------------------------------------------ research helpers
def test_mr_baselines_match_counts_sides_and_chop():
    n = 4000
    pool5 = pd.DataFrame({"sym": pd.Categorical(np.where(np.arange(n) % 2 == 0, "A", "B")), "bar": T0 + np.arange(n) * BAR,
                          "atr": np.float32(1.0), "chop": (np.arange(n) % 4 == 0)})
    cands = pd.DataFrame({"sym": ["A"] * 60, "bar": T0 + np.arange(60) * BAR, "side": ["long"] * 40 + ["short"] * 20,
                          "atr": np.float32(1.0)})
    cd = {"mr_chop_5m": cands, "mr_any_5m": cands}
    pools = {"5m": pool5, "1h": pool5}
    per = (T0, T0 + n * BAR)
    r = MR.cands_for("mr_chop_5m", "regime", 3, per, cd, pools)
    assert (r["side"] == "long").sum() == 40 and (r["side"] == "short").sum() == 20
    assert pool5.set_index("bar")["chop"].loc[r["bar"]].all()                          # chop baseline enters only during chop
    r2 = MR.cands_for("mr_any_5m", "regime", 3, per, cd, pools)
    assert not pool5.set_index("bar")["chop"].loc[r2["bar"]].all()                     # any: ungated
    assert len(MR.cands_for("mr_chop_5m", "random", 3, per, cd, pools)) == 60
    assert len(MR.cands_for("mr_chop_5m", "rule", 0, per, cd, pools)) == 60


def test_table_bonferroni_over_four_and_export(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)

    def res(vkey, pname, per, n=200):
        out = [dict(vkey=vkey, kind="rule", seed=0, period=pname, kills=0,
                    summary=dict(n=n, per_day=1.0, win=0.5, per=per, total=per * n, maxdd=10.0, n_long=n // 2, n_short=n // 2,
                                 usd_long=1.0, usd_short=2.0, tp=1, sl=1, to=0))]
        for kind in ("random", "regime"):
            for s in range(20):
                out.append(dict(vkey=vkey, kind=kind, seed=s, period=pname, kills=0, summary=dict(
                    n=n, per_day=1.0, win=0.5, per=float(rng.normal(0.0, 0.1)), total=0.0, maxdd=1.0, n_long=1, n_short=1, usd_long=0.0,
                    usd_short=0.0, tp=0, sl=0, to=0)))
        return out

    dev = sum((res(v.key, "dev", 0.05 * (i + 1)) for i, v in enumerate(MR.MR_VARIANTS)), [])
    t = MR.table_mr(dev, "dev")
    assert list(t["variant"]) == [v.key for v in MR.MR_VARIANTS]
    z = t["z_rand"].iloc[0]
    assert t["p_bonf_rand"].iloc[0] == pytest.approx(min(1.0, 4 * RR.p_one_sided(z)), abs=0.02)
    assert MR.select_mr(t)[0] == "mr_any_1h" and len(MR.select_mr(t)) == 2                      # highest per -> highest z
    hold = res("mr_any_1h", "holdout", 0.9, n=100) + res("mr_any_1h", "post", 0.9, n=20)
    monkeypatch.setattr(MR, "CACHE", tmp_path)
    pickle_dump = __import__("pickle").dump
    pickle_dump(dev, open(tmp_path / "dev_mr_results.pkl", "wb"))
    pickle_dump(hold, open(tmp_path / "holdout_mr_results.pkl", "wb"))
    p = MR.export_mr(tmp_path, "20261021_0000")
    meta = json.loads(p.read_text())
    assert p.name == "rules_mr_any_1h_20261021_0000.json" and meta["kind"] == "rules" and meta["status"] == "challenger"
    assert not meta["note"].startswith(RR.PAPER_NOTE) and "HOLDOUT 2026-08-15" in meta["note"] and meta["metrics"]["n_variants"] == 4
    pv = json.loads(MR.export_mr(tmp_path, "20261021_0001", preview=True).read_text())
    assert pv["name"].endswith("_preview") and pv["note"].startswith(RR.PAPER_NOTE) and "pending" in pv["note"]
    # a holdout that does not clear both baselines is labelled PAPER TEST ONLY
    pickle_dump(dev + res("mr_any_1h", "holdout", -0.5, n=100) + res("mr_any_1h", "post", -0.5, n=20), open(tmp_path / "holdout_mr_results.pkl", "wb"))
    assert json.loads(MR.export_mr(tmp_path, "20261021_0002").read_text())["note"].startswith(RR.PAPER_NOTE)
    assert R.RuleCfg.from_dict(meta["rule_cfg"]) == R.RuleCfg("any", "S4", "1h")
