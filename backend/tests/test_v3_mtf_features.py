"""tb.features_mtf: resampling, no-look-ahead, closed-bars-only, train/live parity."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from synth import make_market, make_symbol  # noqa: E402

from tb import features as ft  # noqa: E402
from tb import features_mtf as fm  # noqa: E402

BAR = 300_000
N = 12000
GROUPS = ("kalman",)


@pytest.fixture(scope="module")
def data():
    btc, bm, bf = make_symbol(N, seed=1, start_price=30000.0)
    kl, m, f = make_symbol(N, seed=2, market=make_market(N))
    return kl, m, f, btc


@pytest.fixture(scope="module")
def built(data):
    kl, m, f, btc = data
    X, t = fm.build_mtf_features(kl, m, f, btc, fm.DEFAULT_TFS, GROUPS)
    return X, t


def test_names_shape_and_liveness(data, built):
    X, t = built
    names = fm.feature_names(fm.DEFAULT_TFS, GROUPS)
    assert len(names) == len(set(names)) == X.shape[1]
    assert X.dtype == np.float32 and np.isfinite(X).all() and np.abs(X).max() <= ft.CLIP_RAW
    w = fm.warmup_bars_5m(fm.DEFAULT_TFS, GROUPS)
    assert w <= fm.history_bars_5m(fm.DEFAULT_TFS, GROUPS) < N
    dead = [n for n, s in zip(names, X[w:].std(axis=0)) if s == 0 and n != "has_metrics"]
    assert not dead, dead
    assert len(fm.feature_names(("1h",), ("kalman",))) < len(names)


def test_resample_hand_computed():
    # 5m bars start mid-hour (00:20): the first hour is partial (dropped) and so is the trailing partial hour
    t0 = 1_700_006_400_000 - (1_700_006_400_000 % 86_400_000) + 20 * 60_000
    n = 12 * 5 + 7
    t = t0 + np.arange(n) * BAR
    o = np.arange(n, dtype=float) + 100
    kl = pd.DataFrame({"open_time": t, "open": o, "high": o + 2, "low": o - 1, "close": o + 1,
                       "volume": 1.0, "quote_volume": 2.0, "taker_buy_volume": 0.25})
    b = fm.resample_ohlcv(kl, "1h")
    first_full = int(np.flatnonzero((t % 3_600_000) == 0)[0])
    n_full = (n - first_full) // 12
    assert len(b["end_ms"]) == n_full
    i0 = first_full
    assert b["open"][0] == o[i0] and b["close"][0] == o[i0 + 11] + 1
    assert b["high"][0] == o[i0 + 11] + 2 and b["low"][0] == o[i0] - 1
    assert b["volume"][0] == 12.0 and b["taker_buy_volume"][0] == 3.0
    assert b["end_ms"][0] == t[i0] + 3_600_000
    # not closed -> not visible: when the 11th 5m bar of the hour closes the hour is not used yet
    j = fm.htf_index(np.array([t[i0 + 10] + BAR, t[i0 + 11] + BAR]), b["end_ms"])
    assert j[0] == -1 and j[1] == 0


def test_htf_index_exact_close_boundary():
    end = np.array([3_600_000, 7_200_000])
    assert list(fm.htf_index(np.array([3_599_999, 3_600_000, 7_199_999, 7_200_000]), end)) == [-1, 0, 0, 1]


def test_decision_mask():
    t = np.arange(48) * BAR + 3_600_000 * 10            # 10:00 open
    assert list(np.flatnonzero(fm.decision_mask(t, 12))) == [11, 23, 35, 47]     # closes on the hour
    assert fm.decision_mask(t, 1).all()
    assert list(np.flatnonzero(fm.decision_mask(t, 3))) == list(range(2, 48, 3))


def test_htf_columns_constant_within_forming_bar(data, built):
    X, t = built
    names = fm.feature_names(fm.DEFAULT_TFS, GROUPS)
    cols = [k for k, n in enumerate(names) if n.startswith(("h1_", "kal_h1_"))]
    base = 9900
    base += (-int((t[base] // BAR) % 12)) % 12            # first 5m row of an hour
    assert (t[base] % 3_600_000) == 0
    # rows base..base+10 close before the hour ends: identical 1h context
    assert np.array_equal(X[base, cols], X[base + 10, cols])
    # row base+11 closes the hour -> the new bar becomes visible
    assert not np.array_equal(X[base + 10, cols], X[base + 11, cols])


def _mutate_future(kl, m, f, btc, k):
    kl2, btc2 = kl.copy(), btc.copy()
    rng = np.random.default_rng(7)
    for c in ("open", "high", "low", "close"):
        kl2.loc[k + 1:, c] = kl2.loc[k + 1:, c].to_numpy() * rng.uniform(0.5, 2.0, len(kl2) - k - 1)
        btc2.loc[k + 1:, c] = btc2.loc[k + 1:, c].to_numpy() * rng.uniform(0.5, 2.0, len(btc2) - k - 1)
    kl2.loc[k + 1:, "volume"] = 123.0
    t_k = kl["open_time"].iloc[k] + BAR                    # close of bar k
    m2 = m.copy()
    m2.loc[m2["create_time"] > t_k, "sum_open_interest"] *= 3.0
    f2 = f.copy()
    f2.loc[f2["calc_time"] > t_k, "funding_rate"] = 0.01
    return kl2, m2, f2, btc2


@pytest.mark.parametrize("k", [9800, 10555, 11111])
def test_no_lookahead_mutating_future_leaves_past(data, k):
    kl, m, f, btc = data
    X1, _ = fm.build_mtf_features(kl, m, f, btc, fm.DEFAULT_TFS, GROUPS)
    kl2, m2, f2, btc2 = _mutate_future(kl, m, f, btc, k)
    X2, _ = fm.build_mtf_features(kl2, m2, f2, btc2, fm.DEFAULT_TFS, GROUPS)
    assert np.array_equal(X1[:k + 1], X2[:k + 1])
    assert not np.array_equal(X1[k + 1:], X2[k + 1:])      # sanity: the mutation does change the future


def test_train_live_parity_last_row(data, built):
    """The live engine feeds only `history_bars_5m` bars; the vector must equal the training row."""
    kl, m, f, btc = data
    X, t = built
    H = fm.history_bars_5m(fm.DEFAULT_TFS, GROUPS)
    names = fm.feature_names(fm.DEFAULT_TFS, GROUPS)
    meta = dict(timeframes=["5m", *fm.DEFAULT_TFS], feature_groups=list(GROUPS), feature_names=names,
                history_bars_5m=H)
    worst = 0.0
    for i in (H + 3, H + 500, N - 777, N - 1):
        lo = i - H + 1
        b = fm.FrameBundle("X", kl.iloc[lo:i + 1].reset_index(drop=True), m, f, None,
                           btc.iloc[lo:i + 1].reset_index(drop=True))
        live = fm.build_for_model(meta, b)
        assert live.shape == (H, len(names))
        d = np.abs(live[-1].astype("float64") - X[i].astype("float64"))
        worst = max(worst, float(d.max()))
        assert d.max() < 2e-3, (i, [(names[j], d[j]) for j in np.argsort(-d)[:4]])
    print("worst train/live feature diff", worst)


def test_build_for_model_checks_names_and_accepts_dict(data):
    kl, m, f, btc = data
    meta = dict(timeframes=["5m", "1h", "4h"], feature_groups=[])
    X = fm.build_for_model(meta, dict(kl=kl.iloc[-3000:], metrics=m, funding=f, btc_kl=btc.iloc[-3000:]),
                           rows=np.array([2999]))
    assert X.shape == (1, len(fm.feature_names(("1h", "4h"), ())))
    with pytest.raises(ValueError):
        fm.build_for_model(dict(meta, feature_names=["nope"]), dict(kl=kl.iloc[-3000:]))


def test_missing_metrics_and_btc_do_not_crash(data):
    kl, m, f, btc = data
    X, _ = fm.build_mtf_features(kl.iloc[-4000:].reset_index(drop=True), None, None, None, fm.DEFAULT_TFS, ())
    assert np.isfinite(X).all()
    names = fm.feature_names(fm.DEFAULT_TFS, ())
    assert (X[:, names.index("has_metrics")] == 0).all()


def test_wema_is_history_independent_and_weights_recent():
    x = np.arange(100, dtype=float)
    a = fm.wema(x, 10)
    b = fm.wema(x[40:], 10)
    assert np.allclose(a[60:], b[20:]) and np.isnan(a[18]) and np.isfinite(a[19])
    assert x[-20:].mean() < a[-1] < x[-1]                  # between the window mean and the latest value


def test_htf_atr_5m_uses_closed_bars_only():
    t0 = 1_700_006_400_000 - (1_700_006_400_000 % 86_400_000)
    n = 12 * 40
    t = t0 + np.arange(n) * BAR
    c = np.full(n, 100.0)
    kl = pd.DataFrame({"open_time": t, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0,
                       "quote_volume": 1.0, "taker_buy_volume": 0.5})
    a = fm.htf_atr_5m(kl, "1h")
    assert a[-1] == pytest.approx(2.0)                      # TR = high-low = 2 every hour
    assert np.isnan(a[12 * 14 - 2])                        # 14 closed hours (+1 for the previous close) not there yet
    kl2 = kl.copy()
    kl2.loc[n - 6:n - 2, "high"] = 500.0                    # spike inside the last (still forming) half hour
    a2 = fm.htf_atr_5m(kl2, "1h")
    assert a2[n - 2] == pytest.approx(2.0)                  # the hour of the spike is not closed at bar n-2
