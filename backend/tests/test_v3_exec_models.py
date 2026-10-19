"""tb.sim.execution: taker / maker fill model on hand-made bars (numpy only)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tb.sim.execution import (KIND_MAKER, KIND_NONE, KIND_TAKER, ExecConfig, fees_dominate, fill_order,  # noqa: E402
                              fill_orders, low_vol_gate, round_trip_cost)


def bars(rows):
    """rows: [(o, h, l, c), ...] -> four arrays"""
    a = np.array(rows, dtype="float64")
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def test_taker_fills_next_open_with_slippage():
    o, h, l, c = bars([(100, 101, 99, 100), (101, 102, 100, 101), (101, 102, 100, 101)])
    cfg = ExecConfig(mode="taker")
    f = fill_order(o, h, l, c, 0, +1, cfg)
    assert f.filled and f.kind == KIND_TAKER and f.bar == 1
    assert f.price == pytest.approx(101 * (1 + 0.0002))
    assert f.fee_rate == 0.0005 and f.placements == 0
    s = fill_order(o, h, l, c, 0, -1, cfg)
    assert s.price == pytest.approx(101 * (1 - 0.0002))
    # no next bar -> unresolved
    assert not fill_order(o, h, l, c, 2, +1, cfg).filled


def test_maker_buy_needs_a_bar_trading_through_the_limit():
    # signal bar 0 closes at 100 -> buy limit 100.  Bar 1 only TOUCHES 100 (low == 100): no fill.
    o, h, l, c = bars([(100, 101, 99, 100), (100.5, 101, 100.0, 100.5), (100.5, 101, 99.99, 100.5)])
    cfg = ExecConfig(mode="maker", timeout_bars=2, reprices=0)
    f = fill_order(o, h, l, c, 0, +1, cfg)
    assert f.filled and f.kind == KIND_MAKER
    assert f.bar == 2                                 # bar 1 touched only, bar 2 traded through
    assert f.price == 100.0                           # booked exactly at the limit
    assert f.fee_rate == 0.0002 and f.placements == 1


def test_maker_touch_does_not_fill_and_timeout_skips():
    o, h, l, c = bars([(100, 101, 99, 100)] + [(100.5, 101, 100.0, 100.5)] * 6)
    cfg = ExecConfig(mode="maker", timeout_bars=3, reprices=0, on_timeout="skip")
    f = fill_order(o, h, l, c, 0, +1, cfg)
    assert not f.filled and f.kind == KIND_NONE and f.placements == 1
    assert np.isnan(f.price)


def test_maker_gap_through_limit_still_fills_at_limit_not_better():
    o, h, l, c = bars([(100, 101, 99, 100), (98, 99, 97, 98.5)])       # gaps down far through the buy limit
    f = fill_order(o, h, l, c, 0, +1, ExecConfig(mode="maker"))
    assert f.filled and f.price == 100.0                                # not 98


def test_maker_sell_symmetric():
    o, h, l, c = bars([(100, 101, 99, 100), (99.5, 100.0, 99, 99.5), (99.5, 100.01, 99, 99.5)])
    f = fill_order(o, h, l, c, 0, -1, ExecConfig(mode="maker", timeout_bars=2, reprices=0))
    assert f.filled and f.bar == 2 and f.price == 100.0 and f.kind == KIND_MAKER


def test_through_pct_requires_more_penetration():
    o, h, l, c = bars([(100, 101, 99, 100), (100, 101, 99.95, 100), (100, 101, 99.85, 100)])
    cfg = ExecConfig(mode="maker", timeout_bars=2, reprices=0, through_pct=0.001)   # needs low < 99.9
    f = fill_order(o, h, l, c, 0, +1, cfg)
    assert f.filled and f.bar == 2


def test_reprice_chases_the_market_down_for_a_sell():
    # sell limit 100; price drifts DOWN and never reaches 100 -> after the timeout the limit is re-posted
    # at the last close (98.0) and fills in bar 4 when price rallies through it
    o, h, l, c = bars([(100, 100, 99, 100),            # 0 signal
                       (99, 99.5, 98.5, 98.8),         # 1
                       (98.8, 99, 98, 98.2),           # 2
                       (98.2, 98.5, 97.8, 98.0),       # 3 (timeout=3 -> re-post at 98.0 from bar 4)
                       (98.0, 98.4, 97.9, 98.2),       # 4 rally through 98.0
                       (98.2, 98.3, 98.0, 98.1)])
    cfg = ExecConfig(mode="maker", timeout_bars=3, reprices=1)
    f = fill_order(o, h, l, c, 0, -1, cfg)
    assert f.filled and f.kind == KIND_MAKER and f.bar == 4
    assert f.price == 98.0 and f.placements == 2
    # without a reprice the order is dead
    g = fill_order(o, h, l, c, 0, -1, ExecConfig(mode="maker", timeout_bars=3, reprices=0))
    assert not g.filled and g.placements == 1


def test_timeout_fallback_to_taker_uses_next_open():
    o, h, l, c = bars([(100, 101, 99, 100)] + [(100.5, 101, 100.2, 100.6)] * 3 + [(100.8, 101, 100.5, 100.9)])
    cfg = ExecConfig(mode="maker", timeout_bars=3, reprices=0, on_timeout="taker")
    f = fill_order(o, h, l, c, 0, +1, cfg)
    assert f.filled and f.kind == KIND_TAKER and f.bar == 4
    assert f.price == pytest.approx(100.8 * (1 + 0.0002)) and f.fee_rate == 0.0005
    # per-call override (a position that must be closed): skip config but force taker
    cfg2 = ExecConfig(mode="maker", timeout_bars=3, reprices=0, on_timeout="skip")
    assert not fill_order(o, h, l, c, 0, +1, cfg2).filled
    assert fill_order(o, h, l, c, 0, +1, cfg2, on_timeout="taker").filled


def test_data_ends_while_order_is_resting_is_unresolved():
    o, h, l, c = bars([(100, 101, 99, 100), (100.5, 101, 100.0, 100.5)])
    f = fill_order(o, h, l, c, 0, +1, ExecConfig(mode="maker", timeout_bars=3, reprices=1, on_timeout="taker"))
    assert not f.filled


def test_vectorised_matches_slow_reference():
    rng = np.random.default_rng(3)
    n = 600
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0007, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0007, n)))

    def ref(i, side, cfg, ot):
        L, start, pl = c[i], i + 1, 0
        for _ in range(cfg.reprices + 1):
            pl += 1
            for w in range(cfg.timeout_bars):
                b = start + w
                if b >= n:
                    return None
                hit = l[b] < L * (1 - cfg.through_pct) if side > 0 else h[b] > L * (1 + cfg.through_pct)
                if hit:
                    return (KIND_MAKER, b, L, cfg.fee_maker, pl)
            start += cfg.timeout_bars
            if start - 1 >= n:
                return None
            L = c[start - 1]
        if ot == "taker":
            if start >= n:
                return None
            return (KIND_TAKER, start, o[start] * (1 + side * cfg.slippage), cfg.fee_taker, pl)
        return (KIND_NONE, -1, np.nan, 0.0, pl)

    for ot in ("skip", "taker"):
        cfg = ExecConfig(mode="maker", timeout_bars=2, reprices=2, on_timeout=ot, through_pct=0.0003)
        idx = np.arange(0, n - 1, 3)
        side = np.where(rng.random(len(idx)) < 0.5, 1, -1)
        r = fill_orders(o, h, l, c, idx, side, cfg)
        n_maker = 0
        for k, (i, s) in enumerate(zip(idx, side)):
            e = ref(int(i), int(s), cfg, ot)
            if e is None:
                assert not r["filled"][k]
                continue
            kind, b, px, fee, pl = e
            assert int(r["kind"][k]) == kind and int(r["bar"][k]) == b
            assert r["fee_rate"][k] == pytest.approx(fee) and int(r["placements"][k]) == pl
            if kind:
                assert r["price"][k] == pytest.approx(px)
            n_maker += kind == KIND_MAKER
        assert n_maker > 10                                   # the test actually exercised fills


def test_round_trip_cost_and_gate():
    t = ExecConfig(mode="taker")
    m = ExecConfig(mode="maker")
    assert round_trip_cost(t) == pytest.approx(2 * (0.0005 + 0.0002))
    assert round_trip_cost(m) == pytest.approx(2 * 0.0002)
    assert round_trip_cost(m) < round_trip_cost(t)
    assert round_trip_cost(m, entry_maker=1.0, exit_maker=0.0) == pytest.approx(0.0002 + 0.0007)
    # per-bar ATR 0.05 % over 36 bars -> 0.3 % typical move vs 0.14 % taker round trip
    assert low_vol_gate(0.0005, t, 36, min_ratio=2.0)
    assert not low_vol_gate(0.0005, t, 36, min_ratio=3.0)
    assert low_vol_gate(0.0005, m, 36, min_ratio=3.0)                     # cheaper execution passes the same gate
    v = low_vol_gate(np.array([0.0001, 0.001, np.nan]), t, 36, min_ratio=3.0)
    assert list(v) == [False, True, False]
    assert list(fees_dominate(np.array([0.0001, 0.001, np.nan]), t, 36, 3.0)) == [True, False, True]


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ExecConfig(mode="iceberg").validate()
    with pytest.raises(ValueError):
        fill_orders([1.0], [1.0], [1.0], [1.0], [0], 1, ExecConfig(on_timeout="cancel"))
