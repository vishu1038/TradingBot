"""Auxiliary multi-task targets: brute-force check and no-look-ahead (past rows independent of the future)."""
import numpy as np
import pandas as pd
import pytest

from tb.features import atr
from tb.labels import aux_targets


def frame(o, h, l, c):
    n = len(o)
    return pd.DataFrame({"open_time": np.arange(n) * 300_000 + 1_700_000_000_000, "open": o, "high": h, "low": l,
                         "close": c, "volume": 1.0, "quote_volume": 100.0, "trades": 1,
                         "taker_buy_volume": 0.5, "taker_buy_quote_volume": 50.0})


def test_aux_targets_bruteforce_and_causality():
    rng = np.random.default_rng(9)
    n = 500
    c = 100 * np.exp(np.cumsum(0.004 * rng.standard_normal(n)))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * (1 + 0.003 * rng.random(n))
    l = np.minimum(o, c) * (1 - 0.003 * rng.random(n))
    df = frame(o, h, l, c)
    got = aux_targets(df, vol_bars=10, exc_bars=40)
    a = atr(h, l, c, 14)
    r = np.diff(np.log(c))                                  # r[i] = ln c[i+1] / c[i]
    for i in (20, 100, 300, 459):
        want_v = np.log(np.sqrt(np.mean(r[i:i + 10] ** 2)) / (a[i] / c[i]))
        assert got["fvol"][i] == pytest.approx(want_v, abs=1e-4)
        up = np.log1p(max(h[i + 1:i + 41].max() - o[i + 1], 0) / a[i])
        dn = np.log1p(max(o[i + 1] - l[i + 1:i + 41].min(), 0) / a[i])
        assert got["up_exc"][i] == pytest.approx(up, abs=1e-4)
        assert got["dn_exc"][i] == pytest.approx(dn, abs=1e-4)
    assert np.isnan(got["up_exc"][n - 40:]).all() and np.isnan(got["fvol"][n - 10:]).all()
    assert np.isnan(got["up_exc"][:13]).all()               # no ATR yet
    # targets for rows < 250 only look at bars <= 250 + 40: mutate bars >= 300 and they must not change
    df2 = df.copy()
    df2.loc[300:, ["high", "low", "close", "open"]] *= 1.5
    g2 = aux_targets(df2, vol_bars=10, exc_bars=40)
    for k in got:
        np.testing.assert_array_equal(g2[k][:250], got[k][:250])
