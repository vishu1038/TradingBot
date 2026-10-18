"""model_np: multi-task outputs (n_out > 2) and ensembles of seeds."""
import json

import numpy as np
import pytest

from test_ml_model_np import CFGS, rand_weights  # noqa: E402

from tb import features as ft  # noqa: E402
from tb import model_np as mnp  # noqa: E402

F, T = ft.F, ft.WINDOW


def widen_head(W, cfg, n_out, rng):
    h = cfg.get("hidden", cfg.get("head_hidden"))
    W["head2.weight"] = (rng.standard_normal((n_out, h)) / np.sqrt(h)).astype("float32")
    W["head2.bias"] = (0.1 * rng.standard_normal(n_out)).astype("float32")
    return W


@pytest.mark.parametrize("arch", ["cnn", "gru", "transformer"])
def test_ensemble_and_aux_outputs(tmp_path, arch):
    rng = np.random.default_rng(4)
    cfg = dict(CFGS[arch], n_out=7)
    members = [widen_head(rand_weights(arch, CFGS[arch], np.random.default_rng(s)), CFGS[arch], 7, rng) for s in (1, 2, 3)]
    flat = {}
    for k, W in enumerate(members):
        flat.update({f"e{k}/{kk}": v for kk, v in W.items()})
    meta = {"name": "ens", "version": "1", "created_at": "x", "status": "challenger", "parent": None, "arch": arch,
            "arch_cfg": cfg, "ensemble": 3, "feature_names": ft.FEATURE_NAMES, "feature_groups": [], "window": T,
            "norm": {"mean": [0.0] * F, "std": [1.0] * F, "clip": 5.0},
            "calibration": {"long": {"method": "platt", "a": 0.7, "b": -0.2}, "short": {"method": "none"}},
            "thresholds": {"long": 0.6, "short": 0.7}, "exit_cfg": {}, "metrics": {}}
    np.savez(tmp_path / "ens.npz", **flat)
    (tmp_path / "ens.json").write_text(json.dumps(meta))
    sc = mnp.load(tmp_path / "ens")
    assert sc.n_members == 3
    x = np.random.default_rng(0).standard_normal((9, T, F)).astype("float32")
    lg = sc.logits_raw_windows(x)
    want = sum(mnp.forward(arch, W, cfg, np.clip(x, -5, 5))[:, :2] for W in members) / 3
    assert lg.shape == (9, 2) and np.abs(lg - want).max() < 1e-6
    assert mnp.forward(arch, members[0], cfg, x).shape == (9, 7)      # raw forward keeps the auxiliary columns
    p = sc.predict_windows(x)
    assert p[:, 0] == pytest.approx(1 / (1 + np.exp(-(0.7 * want[:, 0] - 0.2))), abs=1e-6)
    s = sc.score("S", x[0])
    assert len(s.reasons) == 5 and s.prob_long == pytest.approx(p[0, 0], abs=1e-6)
    with pytest.raises(ValueError):                                   # a missing member must not load silently
        flat.pop("e2/head1.weight")
        for k in [k for k in flat if k.startswith("e2/")]:
            flat.pop(k)
        np.savez(tmp_path / "bad.npz", **flat)
        (tmp_path / "bad.json").write_text(json.dumps(meta))
        mnp.load(tmp_path / "bad")
