"""tb.model_xsec: artifact round trip, Scorer(kind="xsec") semantics, train/live feature parity."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tb import features_xsec as fx  # noqa: E402
from tb import model_xsec as mx  # noqa: E402
from test_v3_xsec_world import BAR, HOUR, make_world  # noqa: E402

F = fx.F_MODEL
J = fx.MODEL_FEATURES.index("ret_24_z")


def rand_mlp(seed=0, members=2):
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(members):
        ms.append({"l0.weight": rng.normal(0, 0.3, (16, F)), "l0.bias": rng.normal(0, 0.1, 16),
                   "l1.weight": rng.normal(0, 0.3, (8, 16)), "l1.bias": rng.normal(0, 0.1, 8),
                   "l2.weight": rng.normal(0, 0.3, (1, 8)), "l2.bias": np.zeros(1)})
    return mx.MlpRanker(ms, np.zeros(F), np.ones(F))


class Linear:
    """Ranker that scores by one feature (ret_24 z-score): easy to reason about."""
    def predict(self, X):
        return np.asarray(X)[:, J].astype("float64")


@pytest.fixture(scope="module")
def world():
    return make_world(S=10, days=20, seed=21, with_metrics=True, funding_rate=0.0001)


def scorer(ranker, **kw):
    meta = mx.make_meta("xsec_test", "mlp", rebalance_every_h=12, top_k=2, bottom_k=2, horizon_h=24,
                        entry_mode="maker", **kw)
    return mx.XsecScorer(meta, ranker)


def test_meta_contract_fields():
    meta = mx.make_meta("xsec_x", "gbm", 6, 3, 4, 24, "maker", note="PAPER TEST ONLY - no edge")
    assert meta["kind"] == "xsec" and meta["status"] == "challenger"
    assert (meta["rebalance_every_h"], meta["top_k"], meta["bottom_k"], meta["horizon_h"]) == (6, 3, 4, 24)
    assert meta["entry_mode"] == "maker" and meta["entry_cfg"]["entry_mode"] == "maker"
    assert meta["entry_cfg"]["decision_every_bars"] == 72 and meta["entry_cfg"]["maker_timeout_s"] == 900
    assert meta["xsec_cfg"] == {"rebalance_every_h": 6, "top_k": 3, "bottom_k": 4, "horizon_h": 24,
                                "entry_mode": "maker"}
    assert meta["history_bars_5m"] == fx.HISTORY_BARS_5M and meta["timeframes"] == ["5m", "1h"]
    assert meta["exit_cfg"]["exit_mode"] == "pct" and "PAPER TEST ONLY" in meta["note"]
    assert meta["feature_names"] == fx.MODEL_FEATURES
    json.dumps(meta)


def test_mlp_artifact_round_trip(tmp_path):
    r = rand_mlp()
    meta = mx.make_meta("xsec_rt", "mlp", 24, 5, 5, 24, "taker")
    mx.save_mlp(tmp_path / "xsec_rt", meta, r)
    s = mx.load(tmp_path / "xsec_rt.json")
    assert s.kind == "xsec" and s.arch == "mlp" and len(s.ranker.members) == 2 and s.status == "challenger"
    X = np.random.default_rng(1).normal(size=(40, F)).astype("float32")
    assert np.allclose(s.ranker.predict(X), r.predict(X), atol=1e-6)
    # loads from the .npz path / bare path too
    assert mx.load(tmp_path / "xsec_rt.npz").name == s.name and mx.load(tmp_path / "xsec_rt").name == s.name


def test_gbm_artifact_round_trip_with_stage_truncation(tmp_path):
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, F)).astype("float32")
    y = (X[:, J] + 0.5 * rng.normal(size=3000)).astype("float32")
    m = HistGradientBoostingRegressor(max_iter=40, max_depth=3, random_state=0).fit(X, y)
    full, part = mx.GbmRanker(m, None), mx.GbmRanker(m, 10)
    assert not np.allclose(full.predict(X[:50]), part.predict(X[:50]))
    mx.save_gbm(tmp_path / "xsec_g", mx.make_meta("xsec_g", "gbm", 24, 5, 5, 24, "taker"), part)
    s = mx.load(tmp_path / "xsec_g")
    assert s.arch == "gbm" and s.ranker.n_stages == 10
    assert np.allclose(s.ranker.predict(X[:50]), part.predict(X[:50]))
    assert np.corrcoef(s.ranker.predict(X), X[:, J])[0, 1] > 0.5


def test_score_universe_semantics(world):
    s = scorer(Linear())
    sc = s.score_universe(world)
    assert set(sc) == set(world)
    pl = np.array([sc[k].prob_long for k in sorted(sc)])
    ps = np.array([sc[k].prob_short for k in sorted(sc)])
    assert np.allclose(pl + ps, 1.0) and pl.min() > 0 and pl.max() < 1
    assert sorted(np.round(pl * 10 - 0.5).astype(int)) == list(range(10))         # percentiles (i + 0.5)/n
    # ordering follows the ranker: highest ret_24 z-score gets the highest prob_long
    inp = s.build_inputs(world)
    z = inp.X[:, J]
    best = inp.syms[int(np.argmax(z))]
    assert sc[best].prob_long == pl.max()
    r = sc[best].reasons
    assert 1 <= len(r) <= 3 and all(hasattr(x, "contribution") for x in r)
    assert r[0].feature == "momentum"                                              # the group that drives this ranker
    assert s.score("BTCUSDT", np.zeros((10, 3))).prob_long == 0.0                  # per-symbol API is a no-op


def test_target_portfolio_and_untradable_symbols(world):
    # an illiquid / young coin is not tradable -> Score(0, 0), never selected
    w = dict(world)
    young = make_world(S=1, days=20, seed=99, start_offsets={0: 19 * 288}, with_metrics=True)["BTCUSDT"]
    w["NEWUSDT"] = young
    s = scorer(Linear())
    sc = s.score_universe(w)
    assert sc["NEWUSDT"].prob_long == 0.0 and sc["NEWUSDT"].prob_short == 0.0
    tp = s.target_portfolio(w)
    assert len(tp["long"]) == 2 and len(tp["short"]) == 2 and not set(tp["long"]) & set(tp["short"])
    assert tp["n_universe"] == len(world) and "NEWUSDT" not in tp["scores"]
    assert tp["decision_ms"] % HOUR == 0
    pl = {k: v.prob_long for k, v in sc.items() if v.prob_long > 0}
    assert tp["long"] == sorted(pl, key=pl.get, reverse=True)[:2]


def test_train_live_feature_parity(world):
    """Live path (history-limited frames, FrameBundle in any of three shapes) == offline pipeline at that hour."""
    s = scorer(Linear())
    as_tuple = {k: v for k, v in world.items()}
    as_dict = {k: {"klines": v[0], "metrics": v[1], "funding": v[2]} for k, v in world.items()}
    as_obj = {k: SimpleNamespace(klines=v[0], metrics=v[1], funding=v[2]) for k, v in world.items()}
    a, b, c = (s.build_inputs(x) for x in (as_tuple, as_dict, as_obj))
    assert np.array_equal(a.X, b.X) and np.array_equal(a.X, c.X) and a.decision_ms == b.decision_ms
    # offline: full history panel, last complete hour
    p = fx.build_panel(world)
    cube = fx.compute_raw_features(p)
    mask = fx.tradable_mask(p)
    k = p.T - 1
    Xoff = fx.xs_transform(cube[k:k + 1], mask[k:k + 1], fx.btc_index(p.syms))[0]
    assert a.decision_ms == int(p.decision_ms(k))
    assert np.array_equal(a.tradable, mask[k])
    assert np.allclose(a.X, Xoff, atol=1e-5)


def test_asof_ignores_the_future_and_incomplete_hours(world):
    s = scorer(Linear())
    cut = 12 * 400 + 7                                    # 7 bars into hour 400
    asof = world["BTCUSDT"][0].open_time.iloc[0] + cut * BAR
    a = s.build_inputs(world, asof_ms=int(asof))
    trunc = {k: (v[0].iloc[:cut], v[1].iloc[:cut], v[2]) for k, v in world.items()}
    b = s.build_inputs(trunc)
    assert a.decision_ms == b.decision_ms == int(world["BTCUSDT"][0].open_time.iloc[0] + 400 * HOUR)
    assert np.array_equal(a.X, b.X)
    # perturbing data after the as-of time changes nothing
    w2 = {k: (v[0].assign(close=np.where(np.arange(len(v[0])) >= cut, 1.0, v[0].close)), v[1], v[2])
          for k, v in world.items()}
    c = s.build_inputs(w2, asof_ms=int(asof))
    assert np.array_equal(a.X, c.X)


def test_wrong_kind_or_features_rejected(tmp_path):
    meta = mx.make_meta("bad", "mlp", 24, 5, 5, 24, "taker")
    with pytest.raises(ValueError):
        mx.XsecScorer(dict(meta, kind="gbm"), Linear())
    with pytest.raises(ValueError):
        mx.XsecScorer(dict(meta, feature_names=["a", "b"]), Linear())
    (tmp_path / "x.json").write_text(json.dumps(dict(meta, arch="transformer")))
    with pytest.raises(ValueError):
        mx.load(tmp_path / "x.json")
