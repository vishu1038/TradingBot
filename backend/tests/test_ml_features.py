"""tb.features: shape/finite/no-look-ahead/alignment/normalisation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from synth import make_market, make_symbol  # noqa: E402

from tb import features as ft  # noqa: E402

N = 1500


@pytest.fixture(scope="module")
def data():
    btc, bm, bf = make_symbol(N, seed=1, start_price=30000.0)
    kl, m, f = make_symbol(N, seed=2, market=make_market(N))
    return kl, m, f, btc


def test_names_and_shape(data):
    kl, m, f, btc = data
    assert len(ft.FEATURE_NAMES) == len(set(ft.FEATURE_NAMES)) == ft.F
    assert 40 <= ft.F <= 55
    assert ft.WINDOW == 64
    x = ft.build_features(kl, m, f, btc)
    assert x.shape == (N, ft.F) and x.dtype == np.float32
    assert np.isfinite(x).all()
    assert np.abs(x).max() <= ft.CLIP_RAW
    # after warm-up nothing is a constant zero column except possibly nothing
    live = x[ft.WARMUP:]
    dead = [n for n, s in zip(ft.FEATURE_NAMES, live.std(axis=0)) if s == 0 and n != 'has_metrics']
    assert not dead, dead
    assert set(ft.FEATURE_GROUPS) and sum(len(v) for v in ft.FEATURE_GROUPS.values()) == ft.F
    assert sum((ft.FEATURE_GROUPS[g] for g in ft.FEATURE_GROUPS), []) == ft.FEATURE_NAMES


def col(x, name):
    return x[:, ft.FEATURE_NAMES.index(name)]


def test_hand_computed_values(data):
    kl, m, f, btc = data
    x = ft.build_features(kl, m, f, btc)
    i = 900
    c = kl["close"].to_numpy()
    assert col(x, "ret_12")[i] == pytest.approx(100 * np.log(c[i] / c[i - 12]), rel=1e-4)
    o, h, l = kl["open"].to_numpy(), kl["high"].to_numpy(), kl["low"].to_numpy()
    assert col(x, "body_ratio")[i] == pytest.approx((c[i] - o[i]) / (h[i] - l[i]), rel=1e-4)
    assert col(x, "range_pos_48")[i] == pytest.approx((c[i] - l[i - 47:i + 1].min()) /
                                                     (h[i - 47:i + 1].max() - l[i - 47:i + 1].min()), rel=1e-4)
    tb = kl["taker_buy_volume"].to_numpy() / kl["volume"].to_numpy()
    assert col(x, "taker_buy_ratio")[i] == pytest.approx(tb[i], rel=1e-4)
    # time features from the bar open (UTC)
    t = int(kl["open_time"].iloc[i])
    hod = (t // 1000 % 86400) / 86400
    assert col(x, "hour_sin")[i] == pytest.approx(np.sin(2 * np.pi * hod), abs=1e-5)
    wd = pd.Timestamp(t, unit="ms").weekday()
    assert col(x, "dow_cos")[i] == pytest.approx(np.cos(2 * np.pi * wd / 7), abs=1e-5)
    # OI change: metrics join is as-of create_time <= open_time -> bar i sees the snapshot stamped
    # open_time[i] which is the row of bar i-1 (create_time = open + 5m)
    mm = m.set_index("create_time")["sum_open_interest"]
    oi_now = mm.loc[int(kl["open_time"].iloc[i])]
    oi_then = mm.loc[int(kl["open_time"].iloc[i - 12])]
    assert col(x, "oi_chg_12")[i] == pytest.approx(100 * np.log(oi_now / oi_then), rel=1e-4)
    # funding is the last settled rate at or before the bar close
    fr = f[f["calc_time"] <= t + 300_000]["funding_rate"].iloc[-1]
    assert col(x, "funding_bps")[i] == pytest.approx(fr * 1e4, rel=1e-4)
    assert col(x, "has_metrics")[i] == 1.0


def test_no_lookahead_truncation_and_mutation(data):
    kl, m, f, btc = data
    full = ft.build_features(kl, m, f, btc)
    for k in (700, 1000, 1333):
        # (a) truncate EVERYTHING at close of bar k-1: rows < k must be identical
        t_cut = int(kl["open_time"].iloc[k - 1]) + 300_000
        kl_k = kl.iloc[:k]
        m_k = m[m["create_time"] <= t_cut]
        f_k = f[f["calc_time"] <= t_cut]
        btc_k = btc[btc["open_time"] < t_cut]
        part = ft.build_features(kl_k, m_k, f_k, btc_k)
        np.testing.assert_array_equal(part, full[:k])
        # (b) corrupt every future input (kl, metrics, funding, btc) after bar k-1: past rows unchanged
        kl2, m2, f2, b2 = kl.copy(), m.copy(), f.copy(), btc.copy()
        kl2["trades"] = kl2["trades"].astype(float)
        b2["trades"] = b2["trades"].astype(float)
        for df, tc in ((kl2, "open_time"), (b2, "open_time")):
            fut = df[tc] >= int(kl["open_time"].iloc[k])
            for cname in ("open", "high", "low", "close", "volume", "quote_volume", "trades",
                          "taker_buy_volume"):
                df.loc[fut, cname] = df.loc[fut, cname] * 7.3 + 11
        m2.loc[m2["create_time"] > t_cut - 300_000, m2.columns[1:]] *= 5.0
        f2.loc[f2["calc_time"] > t_cut, "funding_rate"] = 0.05
        mut = ft.build_features(kl2, m2, f2, b2)
        np.testing.assert_array_equal(mut[:k], full[:k])
        assert not np.array_equal(mut[k:], full[k:])      # and the future did change (test has teeth)


def test_metrics_one_bar_lag_boundary():
    """A snapshot stamped exactly at the bar CLOSE must not be visible in that bar's row."""
    kl, m, f = make_symbol(700, seed=3)
    btc, _, _ = make_symbol(700, seed=4)
    base = ft.build_features(kl, m, f, btc)
    i = 650
    m2 = m.copy()
    bar_close = int(kl["open_time"].iloc[i]) + 300_000
    m2.loc[m2["create_time"] == bar_close, "sum_open_interest"] *= 3.0
    chg = ft.build_features(kl, m2, f, btc)
    np.testing.assert_array_equal(chg[: i + 1], base[: i + 1])
    assert not np.array_equal(chg[i + 1], base[i + 1])


def test_missing_metrics_and_funding_and_btc():
    kl, _, _ = make_symbol(900, seed=7, with_metrics=False)
    x = ft.build_features(kl, None, None, None)
    assert np.isfinite(x).all() and x.shape == (900, ft.F)
    assert (col(x, "has_metrics") == 0).all()
    for nm in ("oi_chg_12", "funding_bps", "global_ls", "btc_ret_12"):
        assert (col(x, nm) == 0).all()
    # stale metrics (gap) -> has_metrics drops to 0 inside the gap
    kl, m, f = make_symbol(900, seed=7)
    m = m[(m["create_time"] < kl["open_time"].iloc[500]) | (m["create_time"] > kl["open_time"].iloc[600])]
    x = ft.build_features(kl, m, f, kl)
    hm = col(x, "has_metrics")
    assert hm[400] == 1 and hm[560] == 0 and hm[700] == 1


def test_regularize_fills_holes():
    kl, _, _ = make_symbol(100, seed=1)
    holey = kl.drop(index=[10, 11, 50]).reset_index(drop=True)
    reg = ft.regularize(holey)
    assert len(reg) == 100 and (np.diff(reg["open_time"]) == 300_000).all()
    assert reg.loc[10, "volume"] == 0 and reg.loc[10, "close"] == holey.loc[9, "close"]


def test_windows_helper_and_norm(data):
    kl, m, f, btc = data
    x = ft.build_features(kl, m, f, btc)
    idx = np.array([ft.WINDOW - 1, 500, 1499])
    w = ft.windows(x, idx)
    assert w.shape == (3, ft.WINDOW, ft.F)
    np.testing.assert_array_equal(w[1], x[500 - ft.WINDOW + 1: 501])
    with pytest.raises(IndexError):
        ft.windows(x, [10])
    st = ft.robust_norm_stats(x[ft.WARMUP:])
    z = ft.normalize(x[ft.WARMUP:], st["mean"], st["std"])
    assert z.dtype == np.float32 and np.abs(z).max() <= ft.NORM_CLIP
    # robust centre/scale: median ~0, MAD-scale ~1 for non-degenerate columns
    assert abs(np.median(z[:, ft.FEATURE_NAMES.index("ret_12")])) < 0.05
    assert (st["std"] > 0).all()


def test_labels_for_features():
    for nm in ft.FEATURE_NAMES:
        assert isinstance(ft.feature_label(nm, 0.5), str)
    assert ft.feature_label("oi_chg_12", 3.2) == "OI +3.2% in 1h"


def test_optional_feature_group_hook(data):
    kl, m, f, btc = data
    base = ft.build_features(kl, m, f, btc)

    def fn(kl_, t_open, extra):
        return np.stack([np.sin(t_open / 1e7), np.full(len(t_open), extra.get("k", 1.0))], axis=1)

    ft.register_feature_group("unit_test_grp", ["ut_a", "ut_b"], fn, {"ut_a": "A {:+.2f}", "ut_b": "B {:+.2f}"})
    try:
        with pytest.raises(ValueError):
            ft.register_feature_group("unit_test_grp", ["x"], fn)
        with pytest.raises(ValueError):
            ft.register_feature_group("clash", ["ret_1"], fn)
        assert ft.feature_names() == ft.FEATURE_NAMES and ft.feature_names(["unit_test_grp"])[-2:] == ["ut_a", "ut_b"]
        x = ft.build_features(kl, m, f, btc, groups=["unit_test_grp"], extra_data={"k": 3.0})
        assert x.shape == (N, ft.F + 2)
        np.testing.assert_array_equal(x[:, :ft.F], base)                  # base columns untouched
        assert (x[:, -1] == 3.0).all()
        assert ft.feature_label("ut_a", 0.5) == "A +0.50"
        with pytest.raises(KeyError):
            ft.feature_names(["nope"])
    finally:
        ft._EXTRA.pop("unit_test_grp", None)
