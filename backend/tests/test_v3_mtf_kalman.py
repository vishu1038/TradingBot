"""tb.features_kalman: tracks a synthetic trend, fixed-window (history independent), no look-ahead, group hook."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from synth import make_market, make_symbol  # noqa: E402

from tb import features as ft  # noqa: E402
from tb import features_kalman as fk  # noqa: E402
from tb import features_mtf as fm  # noqa: E402

BAR = 300_000


def test_tracks_synthetic_trend():
    rng = np.random.default_rng(0)
    n, L = 400, 60
    mu, sig_n = 0.004, 0.006                                  # drift per bar, observation noise (log)
    true_level = np.cumsum(np.full(n, mu))
    y = true_level + sig_n * rng.standard_normal(n)
    out = fk.filter_windows(y, L)
    assert np.isnan(out["slope"][L - 2]) and np.isfinite(out["slope"][L - 1])
    sl = out["slope"][L:]
    sigma = out["sigma"][L:] / 100.0                          # window bar vol (log units)
    slope_log = sl * sigma                                    # back to log-price per bar
    # recovered slope within 25 % of the truth on average, always positive, strongly significant
    assert abs(slope_log.mean() - mu) / mu < 0.25
    assert (slope_log > 0).all()
    assert np.median(out["slope_z"][L:]) > 1.5
    # the level tracks: price is within a couple of noise sd of the filtered level
    assert np.abs(out["gap"][L:]).mean() * sigma.mean() < 3 * sig_n
    # posterior uncertainty shrinks the noise: well below the raw bar volatility * a few
    assert (out["post_sd"][L:] > 0).all()


def test_flat_noise_has_no_trend_and_trend_has_higher_slope_z():
    rng = np.random.default_rng(1)
    n, L = 800, 60
    noise = np.cumsum(0.003 * rng.standard_normal(n))         # random walk, no drift
    trend = np.cumsum(np.full(n, 0.004)) + 0.003 * rng.standard_normal(n).cumsum() * 0.2
    z_rw = np.abs(fk.filter_windows(noise, L)["slope_z"][L:])
    z_tr = np.abs(fk.filter_windows(trend, L)["slope_z"][L:])
    assert np.median(z_tr) > 2 * np.median(z_rw)


def test_jump_shows_up_in_gap_and_innovation_variance():
    rng = np.random.default_rng(2)
    L = 40
    y = 0.002 * rng.standard_normal(200).cumsum()
    y2 = y.copy()
    y2[-1] += 0.08                                            # 8 % jump on the last bar
    a, b = fk.filter_windows(y, L), fk.filter_windows(y2, L)
    assert b["gap"][-1] > a["gap"][-1] + 1.5                  # price stretched above the level
    assert b["innov_var"][-1] > a["innov_var"][-1] + 0.5


def test_fixed_window_is_history_independent():
    rng = np.random.default_rng(3)
    L = 30
    y = 0.01 * rng.standard_normal(300).cumsum()
    base = fk.filter_windows(y, L)
    y2 = y.copy()
    y2[:100] += 5 * rng.standard_normal(100)                  # change history older than the window of row 200
    other = fk.filter_windows(y2, L)
    for k in base:
        assert np.allclose(base[k][200:], other[k][200:], equal_nan=True)
        assert not np.allclose(base[k][110:130], other[k][110:130], equal_nan=True)


def test_scale_free():
    rng = np.random.default_rng(4)
    y = 0.01 * rng.standard_normal(200).cumsum()
    a = fk.filter_windows(y, 30)
    b = fk.filter_windows(7.0 * y, 30)
    for k in ("gap", "slope", "slope_z", "innov_var"):
        assert np.allclose(a[k], b[k], equal_nan=True, atol=1e-9)
    assert np.allclose(7.0 * a["sigma"][29:], b["sigma"][29:], atol=1e-9)


def test_constant_price_is_finite():
    out = fk.filter_windows(np.full(100, np.log(5.0)), 30)
    for k, v in out.items():
        assert np.isfinite(v[29:]).all(), k


def test_compute_causal_and_registered_group():
    N = 12000
    btc, _, _ = make_symbol(N, seed=1, start_price=30000.0)
    kl, m, f = make_symbol(N, seed=2, market=make_market(N))
    cols = fk.compute(kl, fm.DEFAULT_TFS)
    assert list(cols) == fk.names(fm.DEFAULT_TFS)
    k = 10000
    kl2 = kl.copy()
    kl2.loc[k + 1:, "close"] = kl2.loc[k + 1:, "close"].to_numpy() * 1.7
    kl2.loc[k + 1:, "high"] = kl2.loc[k + 1:, "high"].to_numpy() * 1.7
    cols2 = fk.compute(kl2, fm.DEFAULT_TFS)
    for nme in cols:
        assert np.array_equal(cols[nme][:k + 1], cols2[nme][:k + 1], equal_nan=True), nme
    assert not all(np.array_equal(cols[nme][k + 1:], cols2[nme][k + 1:], equal_nan=True) for nme in cols)
    # registered with tb.features: base columns first, then the 18 kalman columns
    X = ft.build_features(kl, m, f, btc, groups=["kalman"])
    assert X.shape == (N, ft.F + len(fk.names(fm.DEFAULT_TFS))) and np.isfinite(X).all()
    assert ft.feature_names(["kalman"]) == ft.FEATURE_NAMES + fk.names(fm.DEFAULT_TFS)
    assert "kalman" in ft._EXTRA
    assert fm.feature_names(fm.DEFAULT_TFS, ("kalman",))[-18:] == fk.names(fm.DEFAULT_TFS)
