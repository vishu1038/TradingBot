"""End to end with REAL V3 artifacts (mtf numpy MLP + gbm pickle) and the real loaders: the engine pages the history the
models need, scores through ``score_universe`` and shadows / trades them with their own entry / exit config."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from synth import make_market, make_symbol  # noqa: E402

from conftest import FakeData
from test_engine_core import BAR, mk_engine, run
from test_engine_core import FAST  # noqa: F401
from test_v3_engine_core import DeepData, hour_now
from test_v3_mtf_models import F, NAMES, meta_for, save_mtf
from tb import model_gbm
from tb import engine as E
from tb import scorer as S


def shifted(kl, m, f, end_close_ms):
    off = end_close_ms - (int(kl["open_time"].iloc[-1]) + BAR)
    kl = kl.copy()
    kl["open_time"] += off
    if m is not None:
        m = m.copy()
        m["create_time"] += off
    if f is not None:
        f = f.copy()
        f["calc_time"] += off
    return kl, m, f


def test_real_mtf_champion_and_gbm_challenger_run_through_the_engine(store, tmp_path):
    pytest.importorskip("sklearn")
    d = tmp_path / "models"
    d.mkdir()
    rng = np.random.default_rng(0)
    save_mtf(d, rng)
    meta = json.loads((d / "mtf_test.json").read_text())
    meta.update(status="champion", thresholds={"long": 0.0, "short": 1.1},
                exit_cfg={"exit_mode": "pct", "tp_pct": 3.0, "sl_pct": 6.0},
                entry_cfg={"decision_every_bars": 12, "cooldown_bars": 72, "max_trades_per_day": 6, "entry_mode": "taker"})
    (d / "mtf_test.json").write_text(json.dumps(meta))
    X = rng.standard_normal((2000, F)).astype("float32")
    models = {}
    for side in ("long", "short"):
        clf = model_gbm.make_estimator(dict(max_iter=20, min_samples_leaf=50))
        clf.fit(X, (X[:, 3] + 0.5 * rng.standard_normal(2000) > 0).astype(int))
        models[side] = clf
    gm = meta_for("gbm_test", "gbm", arch="hist_gbm", norm=dict(mean=[0.0] * F, std=[1.0] * F, clip=5),
                  calibration={"long": {"method": "none"}, "short": {"method": "none"}})
    gm["thresholds"] = {"long": 0.0, "short": 1.1}
    model_gbm.save(d, "gbm_test", gm, models)

    eng, data = mk_engine(store, tmp_path, now=hour_now())
    assert eng.model_info.name == "mtf_test" and eng.model_info.kind == "mtf" and set(eng.challengers) == {"gbm_test"}
    H = eng.model_info.history_bars_5m
    assert eng.hist_need == H and H > 9000
    data.__class__ = DeepData
    end = int(eng.clock() * 1000) // BAR * BAR
    mk = make_market(H + 600)
    for i, s in enumerate(list(data.kl)):
        kl, m, f = make_symbol(H + 600, seed=i + 5, start_price=100.0 if s != "BTCUSDT" else 30000.0, market=mk)
        data.kl[s], mm, ff = shifted(kl, m, f, end)
        eng.feed.metrics[s], eng.feed.funding[s] = mm, ff
        data.marks[s] = float(data.kl[s]["close"].iloc[-1])
    for _ in range(9):                                                     # history arrives page by page
        run(eng.bars_cycle())
    assert all(len(eng.kl[s]) >= H for s in ("AAAUSDT", "BBBUSDT"))
    eng.last_closed.clear()                                                # the newest bar again, now with full history
    run(eng.bars_cycle())
    sig = store.get_signals("AAAUSDT")
    assert sig and sig[-1]["prob_long"] > 0 and sig[-1]["reasons"]         # scored by the real MLP via score_universe
    pos = {p["symbol"]: p for p in store.get_positions()}
    assert {"AAAUSDT", "BBBUSDT"} <= set(pos)
    p = pos["AAAUSDT"]
    assert p["exit_mode"] == "pct" and (p["entry_price"] - p["sl_price"]) / p["entry_price"] == pytest.approx(0.06, abs=2e-3)
    assert p["sl_order_id"] and p["tp_order_id"]
    assert eng.shadow.has("gbm_test", "AAAUSDT") and eng.shadow.has("mtf_test", "AAAUSDT")      # the gbm challenger shadows too
    run(eng.heartbeat_cycle())
    st = store.kv_get("status")
    assert {m["kind"] for m in st["models"]} == {"mtf", "gbm"}
    assert st["decisions"]["kind"] == "mtf" and st["decisions"]["history_ready"] >= 2
