"""model_np without torch: random-weight artifacts, scorer behaviour, calibration maths."""
import json

import numpy as np
import pytest

from tb import features as ft
from tb import model_np as mnp

F, T = ft.F, ft.WINDOW


def rand_weights(arch, cfg, rng):
    W = {}

    def lin(name, o, i):
        W[name + ".weight"] = (rng.standard_normal((o, i)) / np.sqrt(i)).astype("float32")
        W[name + ".bias"] = (0.1 * rng.standard_normal(o)).astype("float32")

    if arch == "cnn":
        C = cfg["channels"]
        W["inp.weight"] = (rng.standard_normal((C, F, 1)) / np.sqrt(F)).astype("float32")
        W["inp.bias"] = (0.1 * rng.standard_normal(C)).astype("float32")
        for i, _ in enumerate(cfg["dilations"]):
            W[f"blocks.{i}.weight"] = (rng.standard_normal((C, C, cfg["kernel"])) / np.sqrt(3 * C)).astype("float32")
            W[f"blocks.{i}.bias"] = (0.1 * rng.standard_normal(C)).astype("float32")
        lin("head1", cfg["hidden"], 3 * C)
        lin("head2", 2, cfg["hidden"])
    elif arch == "gru":
        H = cfg["hidden"]
        for l in range(cfg["layers"]):
            i = F if l == 0 else H
            W[f"gru.weight_ih_l{l}"] = (rng.standard_normal((3 * H, i)) / np.sqrt(i)).astype("float32")
            W[f"gru.weight_hh_l{l}"] = (rng.standard_normal((3 * H, H)) / np.sqrt(H)).astype("float32")
            W[f"gru.bias_ih_l{l}"] = (0.1 * rng.standard_normal(3 * H)).astype("float32")
            W[f"gru.bias_hh_l{l}"] = (0.1 * rng.standard_normal(3 * H)).astype("float32")
        lin("head1", cfg["head_hidden"], 2 * H)
        lin("head2", 2, cfg["head_hidden"])
    else:
        D = cfg["d_model"]
        lin("embed", D, F)
        W["pos"] = (0.1 * rng.standard_normal((T, D))).astype("float32")
        for i in range(cfg["layers"]):
            p = f"layers.{i}."
            for ln in ("ln1", "ln2"):
                W[p + ln + ".weight"] = np.ones(D, "float32")
                W[p + ln + ".bias"] = np.zeros(D, "float32")
            lin(p + "qkv", 3 * D, D)
            lin(p + "proj", D, D)
            lin(p + "ff1", cfg["ff"], D)
            lin(p + "ff2", D, cfg["ff"])
        W["lnf.weight"] = np.ones(D, "float32")
        W["lnf.bias"] = np.zeros(D, "float32")
        lin("head1", cfg["head_hidden"], 2 * D)
        lin("head2", 2, cfg["head_hidden"])
    return W


CFGS = {
    "cnn": {"channels": 16, "dilations": [1, 2, 4], "kernel": 3, "hidden": 16, "dropout": 0.1},
    "gru": {"hidden": 16, "layers": 2, "dropout": 0.1, "head_hidden": 16},
    "transformer": {"d_model": 16, "heads": 4, "layers": 2, "ff": 32, "dropout": 0.1, "head_hidden": 16,
                    "window": 64},
}


def make_artifact(tmp_path, arch, status="challenger", calibration=None, name=None):
    rng = np.random.default_rng(3)
    W = rand_weights(arch, CFGS[arch], rng)
    name = name or f"m_{arch}"
    meta = {"name": name, "version": "1", "created_at": "2026-10-18T00:00:00Z", "status": status, "parent": None,
            "arch": arch, "arch_cfg": CFGS[arch], "feature_names": ft.FEATURE_NAMES, "window": T,
            "norm": {"mean": [0.0] * F, "std": [1.0] * F, "clip": 5.0},
            "calibration": calibration or {"long": {"method": "none"}, "short": {"method": "none"}},
            "thresholds": {"long": 0.7, "short": 0.8}, "exit_cfg": {"exit_mode": "fixed"}, "metrics": {}}
    np.savez(tmp_path / f"{name}.npz", **W)
    (tmp_path / f"{name}.json").write_text(json.dumps(meta))
    return tmp_path / name


@pytest.mark.parametrize("arch", ["cnn", "gru", "transformer"])
def test_load_score_reasons(tmp_path, arch):
    base = make_artifact(tmp_path, arch)
    for p in (base, base.with_suffix(".json"), base.with_suffix(".npz")):
        sc = mnp.load(p)
    assert sc.name == f"m_{arch}" and sc.status == "challenger"
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((100, F)).astype("float32")
    s = sc.score("BTCUSDT", feats)
    assert 0 < s.prob_long < 1 and 0 < s.prob_short < 1
    assert len(s.reasons) == 5
    assert {r.feature for r in s.reasons} <= set(ft.FEATURE_NAMES)
    assert all(isinstance(r.label, str) and r.label for r in s.reasons)
    assert abs(s.reasons[0].contribution) >= abs(s.reasons[-1].contribution)
    # windowing: only the last 64 rows matter; short history is edge-padded
    s2 = sc.score("X", feats[-64:])
    assert s2.prob_long == pytest.approx(s.prob_long, abs=1e-9)
    assert sc.score("X", feats[:20]).prob_long > 0
    # batch API agrees with the single-window API
    pw = sc.predict_windows(feats[None, -64:])
    assert pw[0, 0] == pytest.approx(s.prob_long, abs=1e-6)
    # occlusion consistency: contribution of the top reason equals the logit change
    top = s.reasons[0]
    j = ft.FEATURE_NAMES.index(top.feature)
    z = ft.normalize(feats[-64:], sc.mean, sc.std)
    z2 = z.copy()
    z2[:, j] = 0
    lg = mnp.forward(arch, sc.W, sc.arch_cfg, np.stack([z, z2]))
    side = 0 if s.prob_long >= s.prob_short else 1
    assert top.contribution == pytest.approx(lg[0, side] - lg[1, side], abs=1e-6)


def test_speed_8_symbols(tmp_path):
    import time
    base = make_artifact(tmp_path, "cnn")
    sc = mnp.load(base)
    feats = np.random.default_rng(1).standard_normal((100, F)).astype("float32")
    t = time.time()
    for _ in range(8):
        sc.score("S", feats)
    assert time.time() - t < 2.0


def test_feature_mismatch_rejected(tmp_path):
    base = make_artifact(tmp_path, "cnn")
    meta = json.loads((tmp_path / "m_cnn.json").read_text())
    meta["feature_names"] = meta["feature_names"][:-1]
    (tmp_path / "m_cnn.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        mnp.load(base)


def test_calibration_applied_in_scorer(tmp_path):
    cal = {"long": {"method": "platt", "a": 0.5, "b": -1.0}, "short": {"method": "none"}}
    base = make_artifact(tmp_path, "cnn", calibration=cal, name="cal")
    sc = mnp.load(base)
    feats = np.random.default_rng(2).standard_normal((70, F)).astype("float32")
    s = sc.score("S", feats, with_reasons=False)
    lg = sc.logits_raw_windows(feats[None, -64:])[0]
    assert s.prob_long == pytest.approx(1 / (1 + np.exp(-(0.5 * lg[0] - 1.0))), abs=1e-9)
    assert s.prob_short == pytest.approx(1 / (1 + np.exp(-lg[1])), abs=1e-9)


def test_platt_recovers_known_miscalibration():
    rng = np.random.default_rng(0)
    z = 3.0 * rng.standard_normal(40000)                    # over-confident raw logits
    true_p = mnp.sigmoid(0.4 * z - 0.3)
    y = (rng.random(len(z)) < true_p).astype(float)
    cal = mnp.fit_platt(z, y)
    assert cal["method"] == "platt" and cal["a"] == pytest.approx(0.4, abs=0.03) and cal["b"] == pytest.approx(-0.3, abs=0.05)
    before = mnp.ece(mnp.sigmoid(z), y)
    after = mnp.ece(mnp.apply_calibration(cal, z), y)
    assert after < before / 3 and after < 0.02


def test_isotonic_monotone_and_reduces_ece():
    rng = np.random.default_rng(1)
    z = 2.0 * rng.standard_normal(30000)
    true_p = 0.2 + 0.5 * mnp.sigmoid(z * 0.7)                # not logistic in z
    y = (rng.random(len(z)) < true_p).astype(float)
    cal = mnp.fit_isotonic(mnp.sigmoid(z), y)
    assert cal["method"] == "isotonic" and np.all(np.diff(cal["y"]) >= 0) and len(cal["x"]) <= 200
    assert mnp.ece(mnp.apply_calibration(cal, z), y) < mnp.ece(mnp.sigmoid(z), y) / 3


def test_degenerate_calibration_is_identity():
    assert mnp.fit_platt(np.zeros(10), np.zeros(10))["method"] == "none"
    assert mnp.fit_platt(np.random.randn(500), np.ones(500))["method"] == "none"
    assert mnp.fit_isotonic(np.random.rand(500), np.zeros(500))["method"] == "none"
    np.testing.assert_allclose(mnp.apply_calibration({"method": "none"}, np.array([0.0, 1.0])), mnp.sigmoid([0, 1]))


def test_reliability_bins():
    p = np.array([0.05, 0.15, 0.15, 0.95])
    y = np.array([0, 1, 0, 1])
    b = mnp.reliability_bins(p, y, 10)
    assert [x["n"] for x in b] == [1, 2, 1]
    assert b[1]["frac_pos"] == 0.5 and b[1]["mean_pred"] == pytest.approx(0.15)
