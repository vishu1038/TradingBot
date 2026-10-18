"""End-to-end smoke: tiny synthetic dataset -> train on mps -> export -> numpy scorer == torch (run in .venv-ml)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("mps not available", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import dataset as D  # noqa: E402
import trainer as T  # noqa: E402
import walkforward as W  # noqa: E402

from tb import features as ft  # noqa: E402
from tb import model_np as mnp  # noqa: E402


@pytest.fixture(scope="module")
def ds():
    return D.synthetic_dataset(n_symbols=3, n=4200, seed=5, planted_edge=0.5, horizon=100)


def test_dataset_invariants(ds):
    n = len(ds)
    assert ds.X.dtype == np.float16 and ds.X.shape == (n, ft.F)
    assert np.isfinite(ds.X.astype("float32")).all()
    for k in D.KEYS:
        assert len(ds.lab[k]) == len(ds.pnl[k]) == len(ds.off[k]) == n
        # NaN labels exactly where pnl is NaN, only at the tail of each symbol block
        assert np.array_equal(np.isnan(ds.lab[k]), np.isnan(ds.pnl[k]))
    # windows never cross a symbol boundary: sample_ok rows have WINDOW-1 earlier rows of the same symbol
    ok = np.nonzero(ds.sample_ok)[0]
    ix = ok[:, None] + T.OFFS[None, :]
    assert (ds.sym[ix] == ds.sym[ok][:, None]).all()
    assert (np.diff(ds.t[ix], axis=1) == 300_000).all()


@pytest.mark.parametrize("arch", ["cnn", "gru", "transformer"])
def test_train_export_roundtrip(ds, tmp_path, arch):
    t0, t1 = int(ds.t.min()), int(ds.t.max())
    span = t1 - t0
    cfg = T.TrainCfg(arch=arch, steps=30, batch=64, eval_every=15, patience=5, val_max=2000)
    net, info = W.fit_split(ds, arch, "fixed", (t0, t0 + int(0.6 * span)), (t0 + int(0.65 * span), t0 + int(0.8 * span)),
                            cfg, lambda *a: None)
    assert np.isfinite(info["best_val_loss"]) and set(info["thresholds"]) == {"long", "short"}
    assert set(info["calibration"]) == {"long", "short"}
    # a champion already in the dir -> the new model records it as parent and is a challenger
    (tmp_path / "old.json").write_text(json.dumps({"name": "old", "status": "champion"}))
    name = W.export_model(ds, arch, "fixed", info, T.M.export_state(net), {"folds": []}, {}, tmp_path, cfg)
    meta = json.loads((tmp_path / f"{name}.json").read_text())
    assert meta["status"] == "challenger" and meta["parent"] == "old" and meta["arch"] == arch
    assert meta["feature_names"] == ft.FEATURE_NAMES and meta["window"] == ft.WINDOW
    assert set(meta["norm"]) >= {"mean", "std", "clip"} and set(meta["thresholds"]) == {"long", "short"}
    assert meta["exit_cfg"]["exit_mode"] == "fixed" and meta["calibration"]["long"]["method"] in ("platt", "none")
    sc = mnp.load(tmp_path / name)
    rows = np.nonzero(ds.sample_ok)[0][::397][:12]
    w = ds.X[rows[:, None] + T.OFFS[None, :]].astype("float32")
    # torch side: logits -> calibration in numpy; numpy side: whole scorer
    lg_t = T.predict_logits(net, ds.X, rows, info["norm"])
    p_t = np.stack([mnp.apply_calibration(info["calibration"]["long"], lg_t[:, 0]),
                    mnp.apply_calibration(info["calibration"]["short"], lg_t[:, 1])], axis=1)
    p_n = sc.predict_windows(w)
    assert np.abs(p_t - p_n).max() < 1e-4
    s = sc.score("SYN0USDT", w[0])
    assert s.prob_long == pytest.approx(p_n[0, 0], abs=1e-6) and len(s.reasons) == 5
    T.free_gpu()


def test_training_learns_planted_signal():
    ds = D.synthetic_dataset(n_symbols=3, n=9000, seed=11, planted_edge=0.8, horizon=100)
    t0, t1 = int(ds.t.min()), int(ds.t.max())
    span = t1 - t0
    cfg = T.TrainCfg(arch="cnn", steps=300, batch=256, eval_every=100, patience=10, val_max=4000, lr=2e-3)
    net, info = W.fit_split(ds, "cnn", "fixed", (t0, t0 + int(0.55 * span)),
                            (t0 + int(0.6 * span), t0 + int(0.75 * span)), cfg, lambda *a: None)
    res, _ = W.evaluate_block(ds, net, info, "fixed", (t0 + int(0.8 * span), t1 - 100 * 300_000), lambda *a: None)
    assert res["long"]["auc"] > 0.6, res["long"]["auc"]
    T.free_gpu()
