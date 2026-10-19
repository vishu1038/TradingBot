"""kind "mtf" (numpy MLP) and kind "gbm" (sklearn) artifacts: save/load round trip, Scorer interface v2."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from synth import make_market, make_symbol  # noqa: E402

from tb import features_mtf as fm  # noqa: E402
from tb import model_np as mnp  # noqa: E402
from tb.scorer import ModelEntry, load_model  # noqa: E402

TFS = ("1h", "4h", "1d")
GROUPS = ("kalman",)
NAMES = fm.feature_names(TFS, GROUPS)
F = len(NAMES)


def meta_for(name, kind, **kw):
    return dict(name=name, version="1", status="challenger", kind=kind, timeframes=["5m", *TFS],
                feature_groups=list(GROUPS), feature_names=NAMES, history_bars_5m=fm.history_bars_5m(TFS, GROUPS),
                thresholds={"long": 0.6, "short": 0.7}, exit_cfg={"exit_mode": "pct", "tp_pct": 3.0, "sl_pct": 6.0},
                entry_cfg={"decision_every_bars": 12, "cooldown_bars": 72, "max_trades_per_day": 6,
                           "entry_mode": "taker"}, **kw)


def mlp_weights(rng, h1=16, h2=8):
    return {"fc1.weight": rng.standard_normal((h1, F)).astype("float32") * 0.3,
            "fc1.bias": rng.standard_normal(h1).astype("float32") * 0.1,
            "fc2.weight": rng.standard_normal((h2, h1)).astype("float32") * 0.3,
            "fc2.bias": rng.standard_normal(h2).astype("float32") * 0.1,
            "head.weight": rng.standard_normal((2, h2)).astype("float32"),
            "head.bias": np.array([0.1, -0.2], "float32")}


def save_mtf(tmp_path, rng):
    W = mlp_weights(rng)
    meta = meta_for("mtf_test", "mtf", arch="mlp", arch_cfg={"hidden": 16, "hidden2": 8, "n_out": 2},
                    norm=dict(mean=rng.standard_normal(F).tolist(), std=(1 + rng.random(F)).tolist(), clip=5.0),
                    calibration={"long": {"method": "platt", "a": 0.8, "b": -0.1}, "short": {"method": "none"}})
    np.savez(tmp_path / "mtf_test.npz", **W)
    (tmp_path / "mtf_test.json").write_text(json.dumps(meta))
    return W, meta


def test_mlp_forward_matches_manual_and_scorer_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    W, meta = save_mtf(tmp_path, rng)
    x = rng.standard_normal((5, F)).astype("float32")
    silu = lambda v: v / (1 + np.exp(-v))  # noqa: E731
    h = silu(x @ W["fc1.weight"].T + W["fc1.bias"])
    h = silu(h @ W["fc2.weight"].T + W["fc2.bias"])
    want = h @ W["head.weight"].T + W["head.bias"]
    assert np.allclose(mnp.forward("mlp", W, {}, x), want, atol=1e-5)
    assert np.allclose(mnp.forward("mlp", W, {}, x[:, None, :]), want, atol=1e-5)       # [B,T,F] -> last row
    sc = mnp.load(tmp_path / "mtf_test.json")
    assert sc.kind == "mtf" and sc.history_bars_5m == meta["history_bars_5m"] and sc.timeframes == meta["timeframes"]
    assert sc.exit_cfg["exit_mode"] == "pct" and sc.entry_cfg["cooldown_bars"] == 72
    xn = mnp.normalize(x, np.asarray(meta["norm"]["mean"], "float32"), np.asarray(meta["norm"]["std"], "float32"))
    lg = mnp.forward("mlp", W, {}, xn)
    p = sc.predict(x)
    assert np.allclose(p[:, 0], mnp.sigmoid(0.8 * lg[:, 0] - 0.1), atol=1e-6)
    assert np.allclose(p[:, 1], mnp.sigmoid(lg[:, 1]), atol=1e-6)
    s = sc.score("X", x[0])
    assert s.prob_long == pytest.approx(p[0, 0]) and s.prob_short == pytest.approx(p[0, 1])
    assert len(s.reasons) == 5 and all(r.feature in NAMES for r in s.reasons)
    assert sc.score("X", x[:3], with_reasons=False).prob_long == pytest.approx(p[2, 0])    # [T,F] -> last row
    with pytest.raises(ValueError):
        sc.score("X", np.zeros(F + 1))
    # the engine loader reaches it through model_np.load
    got = load_model(ModelEntry("mtf_test", tmp_path / "mtf_test.json", "challenger", None, meta))
    assert got is not None and got[0].kind == "mtf"


def test_feature_name_mismatch_is_rejected(tmp_path):
    rng = np.random.default_rng(0)
    W, meta = save_mtf(tmp_path, rng)
    meta["feature_names"] = NAMES[:-1] + ["something_else"]
    (tmp_path / "mtf_test.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        mnp.load(tmp_path / "mtf_test.json")


def test_gbm_roundtrip_and_scorer(tmp_path):
    pytest.importorskip("sklearn")
    from tb import model_gbm
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3000, F)).astype("float32")
    yl = (X[:, 3] + 0.3 * rng.standard_normal(3000) > 0).astype(int)
    ys = (X[:, 5] + 0.3 * rng.standard_normal(3000) > 0).astype(int)
    models = {}
    for side, y in (("long", yl), ("short", ys)):
        clf = model_gbm.make_estimator(dict(max_iter=40, min_samples_leaf=50))
        clf.fit(X, y)
        models[side] = clf
    meta = meta_for("gbm_test", "gbm", arch="hist_gbm", norm=dict(mean=np.median(X, 0).tolist(), std=[1.0] * F, clip=5),
                    calibration={"long": {"method": "none"}, "short": {"method": "none"}})
    model_gbm.save(tmp_path, "gbm_test", meta, models)
    assert (tmp_path / "gbm_test.pkl").exists() and (tmp_path / "gbm_test.json").exists()
    sc = mnp.load(tmp_path / "gbm_test.json")
    assert sc.kind == "gbm" and sc.history_bars_5m == meta["history_bars_5m"]
    p = sc.predict(X[:50])
    assert np.allclose(p[:, 0], models["long"].predict_proba(X[:50])[:, 1], atol=1e-4)
    assert np.allclose(p[:, 1], models["short"].predict_proba(X[:50])[:, 1], atol=1e-4)
    s = sc.score("X", X[0])
    assert 0 <= s.prob_long <= 1 and len(s.reasons) == 5
    # the two informative features dominate the occlusion reasons for the side with the higher probability
    top = {r.feature for r in s.reasons}
    assert NAMES[3] in top or NAMES[5] in top
    assert json.loads((tmp_path / "gbm_test.json").read_text())["kind"] == "gbm"


def test_score_universe_builds_features_like_training_and_handles_bad_frames(tmp_path):
    rng = np.random.default_rng(0)
    save_mtf(tmp_path, rng)
    sc = mnp.load(tmp_path / "mtf_test.json")
    H = sc.history_bars_5m
    n = H + 400
    btc, _, _ = make_symbol(n, seed=1, start_price=30000.0)
    a, am, af = make_symbol(n, seed=2, market=make_market(n))
    b, bm, bf = make_symbol(n, seed=3, market=make_market(n))
    frames = {"BTCUSDT": fm.FrameBundle("BTCUSDT", btc),
              "AAAUSDT": fm.FrameBundle("AAAUSDT", a.iloc[-H:].reset_index(drop=True), am, af),
              "BBBUSDT": {"kl": b.iloc[-H:].reset_index(drop=True), "metrics": bm, "funding": bf},
              "SHORTUSDT": fm.FrameBundle("SHORTUSDT", a.iloc[-500:].reset_index(drop=True))}
    out = sc.score_universe(frames)
    assert set(out) == set(frames)
    # equals scoring the training-path row by hand
    meta = json.loads((tmp_path / "mtf_test.json").read_text())
    X = fm.build_for_model(meta, fm.FrameBundle("AAAUSDT", a.iloc[-H:].reset_index(drop=True), am, af,
                                                None, btc.iloc[-H:].reset_index(drop=True)))
    want = sc.score("AAAUSDT", X[-1])
    assert out["AAAUSDT"].prob_long == pytest.approx(want.prob_long, abs=1e-6)
    assert out["SHORTUSDT"].prob_long == 0.0 and out["SHORTUSDT"].prob_short == 0.0     # too little history
    assert 0 < out["BBBUSDT"].prob_long < 1
