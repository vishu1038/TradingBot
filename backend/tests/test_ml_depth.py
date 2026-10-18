"""Depth feature group: adapter formats, no-look-ahead, coverage flag, registered through the hook."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from synth import make_symbol  # noqa: E402

from tb import features as ft  # noqa: E402
from tb import features_depth as fd  # noqa: E402

N = 900
BANDS = fd.BANDS


def make_depth(kl, seed=0, long=False):
    rng = np.random.default_rng(seed)
    t = kl["open_time"].to_numpy("int64")
    base = 1e6 * np.exp(np.cumsum(0.01 * rng.standard_normal(len(t))))
    rows = {}
    for s in ("bid", "ask"):
        cum = np.cumsum(np.stack([base * (0.3 + 0.2 * k) * np.exp(0.2 * rng.standard_normal(len(t))) for k in range(6)], axis=1), axis=1)
        for k, b in enumerate(BANDS):
            rows[f"{s}_{b:g}"] = cum[:, k]
    wide = pd.DataFrame({"open_time": t, **rows})
    if not long:
        return wide
    recs = []                                              # 30 s snapshots in the raw long format
    for sec in (0, 30):
        for s, sg in (("bid", -1), ("ask", 1)):
            for b in BANDS:
                recs.append(pd.DataFrame({"timestamp": t + sec * 1000, "percentage": sg * b, "depth": 1.0,
                                          "notional": wide[f"{s}_{b:g}"].to_numpy()}))
    return pd.concat(recs, ignore_index=True)


@pytest.fixture(scope="module")
def data():
    kl, m, f = make_symbol(N, seed=2)
    btc, _, _ = make_symbol(N, seed=3, start_price=30000.0)
    return kl, m, f, btc


def col(x, nm):
    return x[:, ft.feature_names(["depth"]).index(nm)]


def test_registered_and_shapes(data):
    kl, m, f, btc = data
    assert "depth" in ft._EXTRA and ft.feature_names(["depth"])[-2] == "has_depth"
    assert len(fd.NAMES) == 19 and fd.NAMES[-2:] == ["has_depth", "has_depth_02"]
    x = ft.build_features(kl, m, f, btc, groups=["depth"], extra_data={"depth": make_depth(kl)})
    assert x.shape == (N, ft.F + 19) and np.isfinite(x).all()
    assert (col(x, "has_depth")[ft.WARMUP if N > ft.WARMUP else 300:] == 1).all()
    live = x[400:, ft.F:-2]
    assert (live.std(axis=0) > 0).all()


def test_values_and_lag(data):
    kl, m, f, btc = data
    d = make_depth(kl)
    x = ft.build_features(kl, m, f, btc, groups=["depth"], extra_data={"depth": d})
    i = 500
    j = i - fd.DEPTH_LAG_BARS                              # bar i sees the bucket of bar i - lag
    assert col(x, "depth_imb_1")[i] == pytest.approx(np.log(d["bid_1"].iloc[j] / d["ask_1"].iloc[j]), rel=1e-4)
    assert col(x, "depth_imb1_chg_12")[i] == pytest.approx(
        np.log(d["bid_1"].iloc[j] / d["ask_1"].iloc[j]) - np.log(d["bid_1"].iloc[j - 12] / d["ask_1"].iloc[j - 12]), abs=1e-4)
    shells = np.diff([d[f"bid_{b:g}"].iloc[j] for b in BANDS], prepend=0.0)
    assert col(x, "bid_wall_conc")[i] == pytest.approx(shells.max() / d["bid_5"].iloc[j], rel=1e-4)


def test_no_lookahead_and_coverage(data):
    kl, m, f, btc = data
    d = make_depth(kl)
    full = ft.build_features(kl, m, f, btc, groups=["depth"], extra_data={"depth": d})
    k = 600
    d2 = d.copy()
    d2.loc[d2["open_time"] >= kl["open_time"].iloc[k - fd.DEPTH_LAG_BARS], d2.columns[1:]] *= 9.0   # corrupt the future
    mut = ft.build_features(kl, m, f, btc, groups=["depth"], extra_data={"depth": d2})
    np.testing.assert_array_equal(mut[:k], full[:k])
    assert not np.array_equal(mut[k:], full[k:])
    # a hole in the book data -> flag 0 and zeros inside the hole, recovery afterwards
    hole = d[(d["open_time"] < kl["open_time"].iloc[500]) | (d["open_time"] > kl["open_time"].iloc[560])]
    x = ft.build_features(kl, m, f, btc, groups=["depth"], extra_data={"depth": hole})
    hd = col(x, "has_depth")
    assert hd[450] == 1 and hd[540] == 0 and hd[700] == 1
    assert (x[540, ft.F:-2] == 0).all()
    # no depth at all
    x0 = ft.build_features(kl, m, f, btc, groups=["depth"], extra_data={})
    assert (x0[:, ft.F:] == 0).all()


def test_long_format_adapter_matches_wide(data):
    kl = data[0]
    wide = make_depth(kl)
    a = fd.normalize_depth(wide)
    b = fd.normalize_depth(make_depth(kl, long=True))
    pd.testing.assert_frame_equal(a[b.columns].astype("float64"), b.astype("float64"), check_names=False, rtol=1e-9)
    alt = wide.rename(columns={c: c.replace("bid_", "bid_notional_").replace("ask_", "ask_notional_") for c in wide.columns if c != "open_time"})
    pd.testing.assert_frame_equal(fd.normalize_depth(alt)[a.columns], a)


def test_missing_02_band_and_mean_columns(data):
    kl, m, f, btc = data
    d = make_depth(kl)
    d["bid_notional_1_mean"] = 1.0                       # extra archive columns must not be mistaken for bands
    d["ask_notional_1_mean"] = 1.0
    d["snap_count"] = 9
    d.loc[:400, ["bid_0.2", "ask_0.2"]] = np.nan         # the 0.2 % band only exists in newer archive data
    x = ft.build_features(kl, m, f, btc, groups=["depth"], extra_data={"depth": d})
    assert np.isfinite(x).all()
    assert col(x, "has_depth")[300] == 1 and col(x, "has_depth_02")[300] == 0 and col(x, "has_depth_02")[600] == 1
    assert col(x, "depth_imb_0.2")[300] == 0 and col(x, "depth_imb_0.2")[600] != 0
    assert col(x, "depth_imb_1")[300] != 0
    j = 600 - fd.DEPTH_LAG_BARS
    assert col(x, "depth_imb_1")[600] == pytest.approx(np.log(d["bid_1"].iloc[j] / d["ask_1"].iloc[j]), rel=1e-4)
