"""ml/xsec_data.py, ml/xsec_train.py, ml/xsec_walkforward.py helpers (no archive, no holdout; torch only if mps is there)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import xsec_data as D  # noqa: E402
import xsec_train as TR  # noqa: E402
import xsec_walkforward as W  # noqa: E402
from tb import features_xsec as fx  # noqa: E402
from tb import model_xsec as mx  # noqa: E402
from test_v3_xsec_world import make_world  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    w = make_world(S=10, days=24, seed=31, with_metrics=True, funding_rate=0.0001)
    p = fx.build_panel(w)
    cube = fx.compute_raw_features(p)
    mask = fx.tradable_mask(p, 1e5, 100, 90, 48)
    return D.Ctx(p, cube, mask, fx.btc_index(p.syms), np.asarray(p.decision_ms(np.arange(p.T))))


def test_targets_are_demeaned_risk_scaled_simple_returns(ctx):
    rel, y = D.targets(ctx, 24)
    k = 300
    ok = np.isfinite(rel[k])
    assert ok.sum() >= 8 and abs(np.nanmean(rel[k])) < 1e-9
    assert abs(float(np.nanmean(y[k]))) < 1e-5
    # raw (unscaled) variant differs, same universe
    rel2, _ = D.targets(ctx, 24, risk_scale=False)
    assert np.array_equal(np.isfinite(rel2[k]), np.isfinite(rel[k]))
    assert np.isfinite(rel).sum() <= np.isfinite(rel2).sum()          # volatility history is needed for the scaling
    assert not np.allclose(np.nan_to_num(rel), np.nan_to_num(rel2))


def test_build_rows_matches_the_cube(ctx):
    ks = np.arange(300, 330, 3)
    rows = D.build_rows(ctx, ks, 24)
    assert rows["X"].shape[1] == fx.F_MODEL and len(rows["y"]) == len(rows["k"]) == len(rows["s"])
    assert np.isfinite(rows["y"]).all() and set(rows["k"]) <= set(ks)
    i = 7
    k, s = rows["k"][i], rows["s"][i]
    Xk = fx.xs_transform(np.asarray(ctx.cube[[k]]), ctx.mask[[k]], ctx.btc)[0, s]
    assert np.allclose(rows["X"][i], Xk)
    assert ctx.mask[k, s]
    nolab = D.build_rows(ctx, ks, 24, need_label=False)
    assert len(nolab["k"]) >= len(rows["k"])
    lab = D.with_labels(nolab, *D.targets(ctx, 24))
    assert len(lab["k"]) == len(rows["k"]) and np.allclose(lab["y"], rows["y"])


def test_score_matrix_and_ic(ctx):
    ks = np.arange(300, 340)
    j = fx.MODEL_FEATURES.index("ret_24_z")
    sc = D.score_matrix(ctx, lambda X: X[:, j].astype("float64"), ks)
    assert np.isnan(sc[:300]).all() and np.isnan(sc[340:]).all()
    assert np.array_equal(np.isfinite(sc[ks]), ctx.mask[ks])
    rel, _ = D.targets(ctx, 24)
    ic_self = D.ic_matrix(np.where(np.isfinite(rel), rel, np.nan), rel, ks)
    assert np.allclose(ic_self[np.isfinite(ic_self)], 1.0)
    ic_neg = D.ic_matrix(-np.nan_to_num(rel, nan=0.0) + np.where(np.isfinite(rel), 0, np.nan), rel, ks)
    assert np.nanmean(ic_neg) == pytest.approx(-1.0)
    s = D.ic_summary(ic_self, 24)
    assert s["ic"] == pytest.approx(1.0)
    # rows_ic agrees with the matrix version
    rows = D.build_rows(ctx, ks, 24)
    pred = rows["X"][:, j].astype("float64")
    a = TR.rows_ic(pred, rows["rel"], rows["k"], rows["s"], ctx.S)
    b = np.nanmean(D.ic_matrix(sc, rel, ks))
    assert a == pytest.approx(b, abs=1e-9)


def planted_rows(n=6000, S=30, seed=0):
    rng = np.random.default_rng(seed)
    nk = n // S
    k = np.repeat(np.arange(nk), S)
    s = np.tile(np.arange(S), nk)
    X = rng.normal(size=(n, fx.F_MODEL)).astype("float32")
    j = fx.MODEL_FEATURES.index("ret_24_z")
    rel = (0.7 * X[:, j] + rng.normal(size=n)).astype("float32")
    return dict(X=X, y=rel, rel=rel, k=k.astype("int64"), s=s.astype("int64"))


def test_train_gbm_learns_a_planted_signal_and_picks_stages():
    tr, va = planted_rows(seed=1), planted_rows(3000, seed=2)
    ranker, info = TR.train_gbm(tr, va, 30, seed=0, max_iter=40, step=10, min_samples_leaf=50)
    assert isinstance(ranker, mx.GbmRanker) and 10 <= ranker.n_stages <= 40
    assert info["val_ic"] > 0.4
    te = planted_rows(3000, seed=3)
    assert TR.rows_ic(ranker.predict(te["X"]), te["rel"], te["k"], te["s"], 30) > 0.4


def test_train_mlp_on_mps_if_available():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("mps not available (torch CPU math crashes on this machine)")
    tr, va = planted_rows(seed=4), planted_rows(3000, seed=5)
    ranker, info = TR.train_mlp(tr, va, 30, seeds=2, epochs=6, batch=1024)
    assert len(ranker.members) == 2 and info["val_ic"] > 0.3
    te = planted_rows(3000, seed=6)
    assert TR.rows_ic(ranker.predict(te["X"]), te["rel"], te["k"], te["s"], 30) > 0.3


def test_selection_helpers():
    def rec(fold, ic_a, ic_b, grids):
        return {"fold": fold, "models": {
            "gbm_h24": {"val_ic": {"ic": ic_a}, "val_grid": grids["a"]},
            "mlp_h24": {"val_ic": {"ic": ic_b}, "val_grid": grids["b"]}}}

    def g(k, R, entry, ret, n=40):
        return {"k": k, "R": R, "entry": entry, "net_return_pct": ret, "n_trades": n}
    grids = {"a": [g(3, 6, "taker", -5), g(5, 12, "maker", 4), g(5, 24, "maker", 9, n=3)],
             "b": [g(3, 6, "taker", 1)]}
    recs = [rec(0, 0.03, 0.01, grids), rec(1, 0.05, 0.02, grids)]
    key, ic = W.pick_model(recs)
    assert key == "gbm_h24" and ic == pytest.approx(0.04)
    c = W.pick_cfg(recs, key)
    assert (c["k"], c["R"], c["entry"]) == (5, 12, "maker")            # the +9 % config has too few trades
    assert W.rebalance_grid(24) == [6, 12, 24] and W.rebalance_grid(12) == [3, 6, 12]


def test_pooling_and_verdict():
    p = W.pooled([[0.1, -0.05], [0.02]])
    assert p["net_return_pct"] == pytest.approx((1.1 * 0.95 * 1.02 - 1) * 100) and p["days"] == 3

    def ev(m, mom, rnd):
        d = lambda r: {"daily": [r / 100.0], "net_return_pct": r, "fees_paid": 1.0, "funding_paid": 0.0, "gross_pnl": 1.0,  # noqa: E731
                       "n_trades": 5, "days": 1, "turnover_x_per_day": 1.0, "killed": False}
        return {"model": d(m), "momentum": d(mom), "reversal": d(0.0), "random": [d(r) for r in rnd]}
    s = W.summarize_pooled([ev(10.0, 2.0, [0, 1, -1, 2, -2]), ev(5.0, 1.0, [0, 1, -1, 2, -2])])
    assert s["model"]["net_return_pct"] > 15 and s["random"]["n"] == 5 and s["z_vs_random"] > 3
    good = {"oos_global_choice": {"pooled": s}, "holdout": {"model": {"net_return_pct": 3.0},
                                                            "momentum": {"net_return_pct": 1.0},
                                                            "random": {"mean_return_pct": 0.0}}}
    assert W.verdict(good)["edge"] is True
    bad = {"oos_global_choice": {"pooled": s}, "holdout": {"model": {"net_return_pct": -3.0},
                                                          "momentum": {"net_return_pct": 1.0},
                                                          "random": {"mean_return_pct": 0.0}}}
    v = W.verdict(bad)
    assert v["edge"] is False and v["oos_beats_random_and_momentum"] is True
    assert W.clean({"a": np.float64("nan"), "b": [np.int64(3)]}) == {"a": None, "b": [3]}
