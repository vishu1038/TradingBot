"""End-to-end smoke: tiny synthetic dataset -> train on mps -> export -> numpy scorer == torch (run in .venv-ml)."""
import json
import sys
from dataclasses import replace
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


def spans(ds):
    t0, t1 = int(ds.t.min()), int(ds.t.max())
    s = t1 - t0
    return (t0, t0 + int(0.6 * s)), (t0 + int(0.65 * s), t0 + int(0.8 * s)), (t0 + int(0.85 * s), t1 - 100 * 300_000)


def test_dataset_invariants(ds):
    n = len(ds)
    assert ds.X.dtype == np.float16 and ds.X.shape == (n, ft.F) and ds.n_feat == ft.F
    assert np.isfinite(np.asarray(ds.X[::7]).astype("float32")).all()
    for k in D.KEYS:
        lab, pnl = ds.lab[k], ds.pnl[k]
        assert len(lab) == len(pnl) == len(ds.off[k]) == n
        assert np.array_equal(np.isnan(lab[:]), np.isnan(pnl[:]))      # n.a. exactly where there is no future
    for a in ("fvol", "up_exc", "dn_exc"):
        assert len(ds.aux[a]) == n
    ok = np.nonzero(np.asarray(ds.pos) >= ft.WINDOW - 1)[0]
    ix = ok[:, None] + T.OFFS[None, :]
    assert (np.asarray(ds.sym)[ix] == np.asarray(ds.sym)[ok][:, None]).all()      # windows never cross symbols
    assert (np.diff(np.asarray(ds.t)[ix], axis=1) == 300_000).all()
    rows = T.select_rows(ds, (int(ds.t.min()), int(ds.t.max()) + 1), "fixed")
    assert (ds.pos[rows] >= ft.WINDOW - 1).all() and ds.any_label("fixed", rows).all()


@pytest.mark.parametrize("arch", ["cnn", "gru", "transformer"])
def test_train_export_roundtrip(ds, tmp_path, arch):
    tr, va, _ = spans(ds)
    cfg = T.TrainCfg(arch=arch, steps=24, batch=64, eval_every=12, patience=5, val_max=2000, aux=True)
    nets, info = W.fit_split(ds, arch, "fixed", tr, va, cfg, lambda *a: None, n_seeds=2)
    assert len(nets) == 2 and info["arch_cfg"]["n_out"] == 7
    assert np.isfinite(info["best_val_loss"]) and set(info["thresholds"]) == {"long", "short"}
    (tmp_path / "old.json").write_text(json.dumps({"name": "old", "status": "champion"}))
    name = W.export_model(ds, arch, "fixed", info, nets, {"folds": []}, {}, tmp_path)
    meta = json.loads((tmp_path / f"{name}.json").read_text())
    assert meta["status"] == "challenger" and meta["parent"] == "old" and meta["arch"] == arch and meta["ensemble"] == 2
    assert meta["feature_names"] == ft.FEATURE_NAMES and meta["window"] == ft.WINDOW and meta["feature_groups"] == []
    assert meta["exit_cfg"]["exit_mode"] == "fixed" and meta["calibration"]["long"]["method"] in ("platt", "none")
    sc = mnp.load(tmp_path / name)
    assert sc.n_members == 2
    rows = np.nonzero(np.asarray(ds.pos) >= ft.WINDOW - 1)[0][::397][:12]
    w = np.asarray(ds.X[rows[:, None] + T.OFFS[None, :]]).astype("float32")
    lg_t = T.predict_ensemble(nets, ds, rows, info["norm"])
    p_t = np.stack([mnp.apply_calibration(info["calibration"]["long"], lg_t[:, 0]),
                    mnp.apply_calibration(info["calibration"]["short"], lg_t[:, 1])], axis=1)
    p_n = sc.predict_windows(w)
    assert np.abs(p_t - p_n).max() < 1e-4
    s = sc.score("SYN0USDT", w[0])
    assert s.prob_long == pytest.approx(p_n[0, 0], abs=1e-6) and len(s.reasons) == 5
    T.free_gpu()


def test_streaming_matches_resident_and_trains(ds, monkeypatch):
    tr, va, te = spans(ds)
    cfg = T.TrainCfg(arch="cnn", steps=40, batch=128, eval_every=20, patience=5, val_max=3000, aux=True,
                     chunk_blocks=20, block_len=512, chunk_steps=10)
    rows = T.select_rows(ds, va, "fixed")[:500]
    norm = ft.robust_norm_stats(np.asarray(ds.X[::50]).astype("float32"))
    resident = T.predict_logits(T.M.build("cnn", ds.n_feat, ft.WINDOW, {"n_out": 7})[0].to("mps"), ds, rows, norm)
    monkeypatch.setattr(T, "GPU_MAX_BYTES", 0)                      # force the streaming path
    T._GPU.clear()
    assert T.gpu_data(ds.X) is None
    net, info = T.train_model(ds, "fixed", T.select_rows(ds, tr, "fixed"), T.select_rows(ds, va, "fixed"), norm,
                              cfg, lambda *a: None)
    assert info["streaming"] and info["steps_run"] >= 20 and np.isfinite(info["best_val_loss"])
    # same net, the two inference paths (GPU gather vs host gather) agree
    monkeypatch.setattr(T, "GPU_MAX_BYTES", 2 << 30)
    T._GPU.clear()
    a = T.predict_logits(net, ds, rows, norm)
    monkeypatch.setattr(T, "GPU_MAX_BYTES", 0)
    b = T.predict_logits(net, ds, rows, norm)
    assert np.abs(a - b).max() < 1e-4 and a.shape == (len(rows), 7) and resident.shape == a.shape
    T._GPU.clear()
    T.free_gpu()


def test_aux_target_alignment(ds):
    rows = T.select_rows(ds, (int(ds.t.min()), int(ds.t.max()) + 1), "fixed")[:1000]
    Y, Mk, A = T.label_block(ds, "fixed", rows)
    assert Y.shape == (1000, 2) and A.shape == (1000, 5)
    assert np.allclose(Mk, np.isfinite(ds.lab["fixed_long"][rows])[:, None] * [1, 0] + np.isfinite(ds.lab["fixed_short"][rows])[:, None] * [0, 1])
    off = ds.off["fixed_long"][rows]
    np.testing.assert_allclose(A[off > 0, 3], np.log1p(off[off > 0]), rtol=1e-6)


def test_training_learns_planted_signal():
    ds = D.synthetic_dataset(n_symbols=3, n=9000, seed=11, planted_edge=0.8, horizon=100)
    t0, t1 = int(ds.t.min()), int(ds.t.max())
    span = t1 - t0
    cfg = T.TrainCfg(arch="cnn", steps=300, batch=256, eval_every=100, patience=10, val_max=4000, lr=2e-3,
                     weight_decay=1e-4)
    nets, info = W.fit_split(ds, "cnn", "fixed", (t0, t0 + int(0.55 * span)), (t0 + int(0.6 * span), t0 + int(0.75 * span)),
                             cfg, lambda *a: None)
    res, _ = W.evaluate_block(ds, nets, info, "fixed", (t0 + int(0.8 * span), t1 - 100 * 300_000), lambda *a: None)
    assert res["long"]["auc"] > 0.6, res["long"]["auc"]
    assert "atr_rule" in res["long"] and res["long"]["atr_rule"]["n"] >= 0
    T.free_gpu()
