"""ml/evaluate.py + ml/splits.py (numpy only)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import evaluate as E  # noqa: E402
import splits as S  # noqa: E402

BAR = 300_000


def test_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert E.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert E.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert E.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5       # ties
    assert E.auc(y, np.array([0.1, 0.6, 0.4, 0.9])) == pytest.approx(0.75)
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    yy = (rng.random(5000) < p).astype(int)
    from itertools import product  # brute-force pair count on a subsample
    a, b = p[yy == 1][:300], p[yy == 0][:300]
    brute = np.mean([(x > z) + 0.5 * (x == z) for x, z in product(a, b)])
    assert E.auc(np.r_[np.ones(300), np.zeros(300)], np.r_[a, b]) == pytest.approx(brute, abs=1e-12)


def test_select_trades_position_rules():
    # one symbol, signals every bar, each trade lasts 3 bars -> every 3rd signal is taken
    t = np.arange(12) * BAR
    sym = np.zeros(12, int)
    off = np.full(12, 3)
    p = np.full(12, 0.9)
    sel = E.select_trades(np.arange(12), p, t, sym, off)
    assert list(sel) == [0, 3, 6, 9]
    # no de-dup returns everything
    assert len(E.select_trades(np.arange(12), p, t, sym, off, dedupe=False)) == 12
    # global cap: 4 symbols signal at the same bar, cap 2 -> the two highest-probability win
    t = np.zeros(4, int)
    sym = np.arange(4)
    p = np.array([0.6, 0.9, 0.8, 0.7])
    sel = E.select_trades(np.arange(4), p, t, sym, np.full(4, 5), max_pos=2)
    assert sorted(sel) == [1, 2]


def test_trading_table_on_planted_signal():
    rng = np.random.default_rng(0)
    n = 20000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(float)             # perfectly calibrated
    pnl = np.where(y == 1, 4.8, -10.3)
    t = np.arange(n) * BAR
    sym = rng.integers(0, 20, n)
    off = np.full(n, 6)
    rows = E.trading_table(p, y, pnl, t, sym, off, 0.68, thresholds=(0.5, 0.9), max_pos=50)
    r90 = rows[1]
    assert r90["win_rate"] > 0.85 and r90["expectancy"] > 2.0 and r90["z_vs_baseline"] > 5
    assert r90["baseline_expectancy"] < 0.5 and r90["baseline_win_rate"] == pytest.approx(0.5, abs=0.05)
    thr, pos = E.choose_threshold(rows, min_trades=10)
    assert thr == 0.9 and pos
    # no signal anywhere -> default, not positive
    thr, pos = E.choose_threshold([dict(threshold=0.7, n=3, expectancy=1.0)], min_trades=30)
    assert (thr, pos) == (0.7, False)


def test_drift_z_and_firing_rates():
    # perfectly calibrated outcomes fire z<-2 rarely; a model that is 15 points over-confident fires a lot
    rng = np.random.default_rng(1)
    n = 6000
    p = np.full(n, 0.7)
    y_ok = (rng.random(n) < 0.7).astype(float)
    y_bad = (rng.random(n) < 0.55).astype(float)
    ok = E.drift_firing(p, y_ok, 30)
    bad = E.drift_firing(p, y_bad, 30)
    assert ok["frac_z_lt_m2"] < 0.06 and bad["frac_z_lt_m2"] > 0.3 and bad["frac_z_lt_m3"] > 0.1 and bad["frac_z_lt_m2"] > 5 * ok["frac_z_lt_m2"]
    z = E.drift_z(np.array([0.7] * 30), np.array([1.0] * 21 + [0.0] * 9), 30)   # 70% realised vs 70%
    assert z[0] == pytest.approx(0.0, abs=1e-9)
    z = E.drift_z(np.array([0.7] * 30), np.array([1.0] * 15 + [0.0] * 15), 30)  # 50% vs 70%
    assert z[0] == pytest.approx((0.5 - 0.7) / np.sqrt(0.7 * 0.3 / 30))
    assert len(E.drift_z(np.ones(10), np.ones(10), 30)) == 0


def test_splits_purge_and_holdout_locked():
    t = np.arange(0, 380 * 288) * BAR + 1_700_000_000_000
    sp = S.make_splits(t, n_folds=4, test_days=30, val_days=14, holdout_days=60)
    purge = sp["purge_bars"] * BAR
    assert purge >= S.HORIZON * BAR
    assert len(sp["folds"]) == 4
    hold_start = sp["final"]["holdout"][0]
    prev_test_end = None
    for f in sp["folds"]:
        assert f["val"][0] - f["train"][1] >= purge
        assert f["test"][0] - f["val"][1] >= purge
        assert f["train"][1] <= f["val"][0] and f["val"][1] <= f["test"][0]
        # nothing a fold touches (incl. the label horizon of its test rows) reaches into the holdout
        assert f["test"][1] + S.HORIZON * BAR <= hold_start
        if prev_test_end:
            assert f["test"][0] == prev_test_end          # contiguous OOS coverage
        prev_test_end = f["test"][1]
        assert f["train"][1] - f["train"][0] >= 45 * S.DAY_MS
    fin = sp["final"]
    assert fin["val"][1] + purge <= hold_start and fin["val"][0] - fin["train"][1] >= purge
    assert hold_start == int(t.max()) + BAR - 60 * S.DAY_MS
    # too little history -> folds dropped rather than starved
    short = S.make_splits(t[: 150 * 288], n_folds=4)
    assert len(short["folds"]) < 4
