"""tb.features_xsec: panel aggregation, no look-ahead, universe, cross-sectional transform, labels."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tb import features_xsec as fx  # noqa: E402
from test_v3_xsec_world import BAR, HOUR, T0, make_world  # noqa: E402


def test_panel_aggregates_hours_from_5m():
    w = make_world(S=2, days=2, seed=1)
    sym = sorted(w)[0]
    kl = w[sym][0]
    p = fx.build_panel(w)
    assert p.T == 48 and p.S == 2 and p.t0 == T0
    h = 5
    blk = kl.iloc[h * 12:(h + 1) * 12]
    j = p.index(sym)
    assert p.open[h, j] == pytest.approx(blk.open.iloc[0], rel=1e-6)
    assert p.close[h, j] == pytest.approx(blk.close.iloc[-1], rel=1e-6)
    assert p.high[h, j] == pytest.approx(blk.high.max(), rel=1e-6)
    assert p.low[h, j] == pytest.approx(blk.low.min(), rel=1e-6)
    assert p.qv[h, j] == pytest.approx(blk.quote_volume.sum(), rel=1e-5)
    assert p.nbar[h, j] == 12
    assert p.c5[j, 12 * h + 11] == pytest.approx(blk.close.iloc[-1], rel=1e-6)
    assert p.decision_ms(h) == T0 + (h + 1) * HOUR


def test_hour_with_few_bars_is_nan_and_incomplete_last_hour_dropped():
    w = make_world(S=1, days=1, seed=2)
    sym = next(iter(w))
    kl = w[sym][0]
    drop = np.arange(12 * 3 + 2, 12 * 3 + 12)                     # hour 3 keeps only 2 bars
    kl2 = kl.drop(index=drop).iloc[:-5]                           # and the last hour is incomplete (7 bars)
    p = fx.build_panel({sym: (kl2, None, None)})
    assert p.T == 23                                              # only complete hours
    assert np.isnan(p.close[3, 0]) and p.nbar[3, 0] == 2
    assert np.isfinite(p.close[2, 0]) and np.isfinite(p.close[4, 0])


def test_metrics_and_funding_alignment():
    w = make_world(S=2, days=3, seed=3, with_metrics=True, funding_rate=0.0002)
    p = fx.build_panel(w)
    sym = sorted(w)[0]
    m = w[sym][1]
    j = p.index(sym)
    # snapshot of hour 10 = the LAST 5m snapshot inside the hour (stamped at :55)
    assert p.oiv[10, j] == pytest.approx(m.sum_open_interest_value.iloc[10 * 12 + 11], rel=1e-5)
    ev_hours = np.nonzero(p.fund_ev[:, j])[0]
    assert list(ev_hours[:3]) == [0, 8, 16]
    assert p.fund_ev[8, j] == pytest.approx(0.0002)
    assert p.fund_rate[7, j] == pytest.approx(0.0002) and p.fund_rate[20, j] == pytest.approx(0.0002)


def test_features_have_no_lookahead():
    w = make_world(S=6, days=20, seed=4, with_metrics=True, funding_rate=0.0001)
    p = fx.build_panel(w)
    cube = fx.compute_raw_features(p)
    k = 300
    # corrupt EVERYTHING after hour k (5m bar index >= 12 (k+1)) and rebuild
    w2 = {}
    cut = 12 * (k + 1)
    for s, (kl, m, f) in w.items():
        kl2 = kl.astype({"trades": "float64"})
        for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades"):
            kl2.loc[cut:, c] = kl2.loc[cut:, c] * 3.7 + 11
        m2 = m.copy()
        m2.loc[cut:, "sum_open_interest_value"] *= 5
        m2.loc[cut:, "sum_open_interest"] *= 5
        f2 = f.copy()
        f2.loc[f2.calc_time >= T0 + (k + 1) * HOUR, "funding_rate"] = 0.05
        w2[s] = (kl2, m2, f2)
    p2 = fx.build_panel(w2)
    cube2 = fx.compute_raw_features(p2)
    a, b = cube[:k + 1], cube2[:k + 1]
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.all(both_nan | (a == b))
    assert not np.array_equal(np.nan_to_num(cube[k + 5]), np.nan_to_num(cube2[k + 5]))      # the perturbation was effective
    m1, m2_ = fx.tradable_mask(p, 1e5, 100, 90, 48), fx.tradable_mask(p2, 1e5, 100, 90, 48)
    assert np.array_equal(m1[:k + 1], m2_[:k + 1])


def test_truncated_history_reproduces_the_features():
    """Live supplies HISTORY_H hours; the offline value for the same hour must be identical."""
    days = 25
    w = make_world(S=8, days=days, seed=5, with_metrics=True, funding_rate=0.0001)
    pfull = fx.build_panel(w)
    cube = fx.compute_raw_features(pfull)
    k = pfull.T - 1
    T = fx.HISTORY_H
    t0 = pfull.t0 + (pfull.T - T) * HOUR
    w2 = {s: (kl[kl.open_time >= t0], m[m.create_time >= t0], f[f.calc_time >= t0]) for s, (kl, m, f) in w.items()}
    p2 = fx.build_panel(w2, t0=t0, T=T)
    cube2 = fx.compute_raw_features(p2)
    a, b = cube[k], cube2[T - 1]
    assert np.all((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-5, atol=1e-6))
    assert np.isfinite(b).mean() > 0.9                             # actually warmed up at the last row
    mask = fx.tradable_mask(pfull)[k]
    mask2 = fx.tradable_mask(p2)[T - 1]
    assert np.array_equal(mask, mask2) and mask.all()


def test_tradable_universe_listing_age_and_liquidity():
    S = 6
    w = make_world(S=S, days=40, seed=6, start_offsets={0: 20 * 288}, qv_scale=[1, 1, 1, 1, 1, 1e-4])
    p = fx.build_panel(w)
    m = fx.tradable_mask(p)
    j0, jl = 0, 5
    assert not m[:, jl].any()                                      # illiquid coin never enters
    assert m[:, 1].sum() > 0
    first = int(np.argmax(m[:, j0]))
    listing_hour = 20 * 24
    assert first >= listing_hour + fx.AGE_WINDOW_OK - 1            # not before it has ~15 days of history
    assert not m[:listing_hour, j0].any()


def test_xs_transform_properties():
    rng = np.random.default_rng(0)
    n, S = 4, 30
    raw = rng.normal(size=(n, S, fx.F_RAW)).astype("float32")
    raw[:, :, fx.raw_index("has_metrics")] = 1.0
    mask = np.ones((n, S), bool)
    mask[:, -3:] = False
    X = fx.xs_transform(raw, mask, btc_idx=0)
    assert X.shape == (n, S, fx.F_MODEL)
    nb = len(fx.BASE_FEATURES)
    z, r = X[:, :, :nb], X[:, :, nb:2 * nb]
    assert np.all(np.abs(r) <= 0.5) and np.all(np.abs(z) <= fx.Z_CLIP)
    assert np.all(X[~mask] == 0)
    j = fx.BASE_FEATURES.index("ret_24")
    m_ = int(mask[0].sum())
    assert np.allclose(np.sort(r[0, :m_, j]), (np.arange(m_) + 0.5) / m_ - 0.5, atol=1e-6)
    assert abs(np.median(z[0, :m_, j])) < 1e-5
    raw2 = raw.copy()
    raw2[:, :, fx.raw_index("ret_24")] = raw[:, :, fx.raw_index("ret_24")] * 3.0 + 7.0
    X2 = fx.xs_transform(raw2, mask, btc_idx=0)
    assert np.allclose(X2[:, :, j], X[:, :, j], atol=1e-5) and np.allclose(X2[:, :, nb + j], X[:, :, nb + j])
    perm = rng.permutation(S - 3)
    pi = np.concatenate([perm, np.arange(S - 3, S)])
    X3 = fx.xs_transform(raw[:, pi], mask, btc_idx=None)
    assert np.allclose(X3[:, :S - 3, :2 * nb], X[:, pi, :2 * nb][:, :S - 3], atol=1e-5)


def test_xs_ties_get_average_rank():
    raw = np.zeros((1, 10, fx.F_RAW), dtype="float32")
    raw[0, :, fx.raw_index("funding_bps")] = [1] * 6 + [2, 3, 4, 5]
    X = fx.xs_transform(raw, np.ones((1, 10), bool))
    j = fx.BASE_FEATURES.index("funding_bps")
    r = X[0, :6, len(fx.BASE_FEATURES) + j]
    assert np.allclose(r, r[0])                                    # ties share one rank, no order artefacts


def test_forward_returns_alignment_and_relative_rank_target():
    w = make_world(S=5, days=6, seed=7)
    p = fx.build_panel(w)
    h = 24
    f = fx.forward_returns(p, h)
    k, j = 10, 2
    assert f[k, j] == pytest.approx(100 * (p.open[k + 1 + h, j] / p.open[k + 1, j] - 1))
    fl = fx.forward_returns(p, h, "log")
    assert fl[k, j] == pytest.approx(100 * np.log(p.open[k + 1 + h, j] / p.open[k + 1, j]))
    sg = np.full(f.shape, 0.04)
    assert fx.risk_scaled(f, sg)[k, j] == pytest.approx(f[k, j] / 4.0)
    assert np.isnan(f[p.T - h - 1:]).all() and np.isfinite(f[p.T - h - 2]).all()
    mask = np.ones((p.T, p.S), bool)
    rel = fx.relative_returns(f, mask)
    assert np.nanmax(np.abs(np.nanmean(rel, axis=1))) < 1e-9       # demeaned over the universe
    y = fx.rank_target(rel)
    ok = np.isfinite(y[k])
    assert abs(float(y[k][ok].mean())) < 1e-5
    assert np.array_equal(np.argsort(y[k][ok]), np.argsort(rel[k][ok]))
    yu = fx.rank_target(rel, gauss=False)
    assert yu[k].min() > -0.5 and yu[k].max() < 0.5


def test_baseline_score_helpers_respect_mask():
    w = make_world(S=6, days=20, seed=8)
    p = fx.build_panel(w)
    cube = fx.compute_raw_features(p)
    mask = fx.tradable_mask(p, 1e5, 100, 90, 48)
    mo = fx.momentum_scores(cube, mask)
    assert np.array_equal(np.isfinite(mo), mask & np.isfinite(cube[:, :, fx.raw_index("ret_168")]))
    rv = fx.momentum_scores(cube, mask, sign=-1)
    assert np.allclose(np.nan_to_num(rv), -np.nan_to_num(mo))
    r1, r2 = fx.random_scores(mask, 1), fx.random_scores(mask, 1)
    assert np.array_equal(np.nan_to_num(r1), np.nan_to_num(r2)) and np.isfinite(r1).sum() == mask.sum()
