import numpy as np
import pandas as pd
import pytest

from tb.config import CorrCap, Settings
from tb.exchange.base import ConditionalOrder, SymbolFilters
from tb.risk import (atr14, correlation_gate, derate_multiplier, derated_below_min_notional, drawdown_pct,
                     effective_slots, kill_tripped, levels, liq_price,
                     max_positions, positions_without_stop, return_corr, size_trade, trail_update)

F = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0)


def test_fixed_sizing_matches_template():
    cfg = Settings()                          # 20 margin x 8 lev, TP +5, SL -10
    s = size_trade(cfg, "long", 100.0, None, F)
    assert s.ok
    assert s.notional == pytest.approx(160.0)
    assert s.margin == pytest.approx(20.0)
    assert s.qty == pytest.approx(1.6)
    assert (s.tp_price - 100) / 100 == pytest.approx(0.03125, abs=1e-4)      # +5 on 160 = 3.125%
    assert (100 - s.sl_price) / 100 == pytest.approx(0.0625, abs=1e-4)       # -10 on 160 = 6.25%
    assert s.risk_usd == pytest.approx(10.0, abs=0.01)
    assert s.reward_usd == pytest.approx(5.0, abs=0.02)


def test_fixed_sizing_short_mirrors():
    s = size_trade(Settings(), "short", 100.0, None, F)
    assert s.sl_price == pytest.approx(106.25, abs=0.01)
    assert s.tp_price == pytest.approx(96.875, abs=0.01)


def test_atr_mode_sizes_risk_to_sl_usd():
    cfg = Settings().merge({"exit_mode": "atr", "sl_usd": 2.0})
    s = size_trade(cfg, "long", 100.0, 1.0, F)          # SL = 1 ATR = 1.0 -> qty = 2/1 = 2 (notional 200 > 160 cap)
    assert s.ok and s.qty == pytest.approx(1.6)         # capped by slice notional
    assert s.sl_price == pytest.approx(99.0) and s.tp_price == pytest.approx(101.5)
    cfg = Settings().merge({"exit_mode": "atr", "sl_usd": 1.0})
    s = size_trade(cfg, "long", 100.0, 1.0, F)
    assert s.qty == pytest.approx(1.0) and s.risk_usd == pytest.approx(1.0)


def test_atr_mode_needs_atr():
    assert not size_trade(Settings().merge({"exit_mode": "atr"}), "long", 100.0, None, F).ok


def test_min_notional_and_qty_rejected():
    big_step = SymbolFilters(0.01, 1.0, 1.0, 5.0)          # whole units only
    assert not size_trade(Settings(), "long", 1000.0, None, big_step).ok      # 0.16 units rounds to 0
    assert not size_trade(Settings(), "long", 100.0, None, SymbolFilters(0.01, 0.001, 0.001, 500.0)).ok


def test_stop_beyond_liquidation_rejected():
    cfg = Settings().merge({"sl_usd": 30.0})       # SL at 18.75% but liquidation is ~12% away
    r = size_trade(cfg, "long", 100.0, None, F)
    assert not r.ok and "liquidation" in r.reason


def test_available_margin_check():
    assert not size_trade(Settings(), "long", 100.0, None, F, available=10.0).ok
    assert size_trade(Settings(), "long", 100.0, None, F, available=25.0).ok


def test_liq_price():
    # Binance form, identical to tb.sim.template.liq_price
    assert liq_price("long", 100, 8, 0.005) == pytest.approx(100 * 0.875 / 0.995)
    assert liq_price("short", 100, 8, 0.005) == pytest.approx(100 * 1.125 / 1.005)
    from tb.sim.template import liq_price as sim_liq
    assert liq_price("long", 123.4, 8, 0.005) == pytest.approx(sim_liq(123.4, 1, 8, 0.005))
    assert liq_price("short", 123.4, 8, 0.005) == pytest.approx(sim_liq(123.4, -1, 8, 0.005))


def test_max_positions():
    cfg = Settings()
    assert max_positions(100, cfg) == 5
    assert max_positions(59.9, cfg) == 2
    assert max_positions(500, cfg) == 5          # capped by the setting
    assert max_positions(10, cfg) == 0


def test_effective_slots_account_for_entry_fees():
    cfg = Settings()                                   # 20 margin, 160 notional, 0.05% fee -> 0.08 per entry
    assert effective_slots(cfg, 0, 100.0) == 4         # not 5: the 5th slice is short by the fees
    assert effective_slots(cfg, 0, 100.4) == 5
    assert effective_slots(cfg, 4, 99.68 - 80.0) == 4  # 4 open, 19.68 free -> no 5th
    assert effective_slots(cfg, 2, 60.0) == 4          # 2 open + 2 more slices
    assert effective_slots(cfg, 0, 1000.0) == 5        # capped by max_positions
    assert effective_slots(cfg, 3, 0.0) == 3
    assert effective_slots(cfg.merge({"max_positions": 2}), 3, 100.0) == 3     # never below what is already open


def test_zombie_gap_detected():
    cfg = Settings()
    assert not derated_below_min_notional(cfg, 40.0, 50.0, 5.0)          # 0.5x -> 80 notional: fine
    assert not derated_below_min_notional(cfg, 49.0, 50.0, 5.0)          # 0.1x -> 16 notional: still tradeable
    assert derated_below_min_notional(cfg, 49.9, 50.0, 5.0)              # 0.005x -> 0.8 < 5: nothing can trade
    assert derated_below_min_notional(cfg, 25.0, 25.0, 5.0)
    assert not derated_below_min_notional(cfg, 49.9, 50.0, 0.5)          # a symbol with a tiny minimum still can


# ------------------------------------------------------------------ derating curve
@pytest.mark.parametrize("dd,kill,expect", [(0, 25, 1.0), (10, 25, 1.0), (20, 25, 0.5), (25, 25, 0.0), (30, 25, 0.0),
                                            (20, 50, 1.0), (40, 50, 0.5), (50, 50, 0.0), (15, 25, 0.75), (22.5, 25, 0.25)])
def test_derate_points(dd, kill, expect):
    assert derate_multiplier(dd, kill) == pytest.approx(expect)


def test_derate_monotonic():
    vals = [derate_multiplier(d / 10, 25) for d in range(0, 300)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_derated_size_and_min_notional_skip():
    cfg = Settings()
    half = size_trade(cfg, "long", 100.0, None, F, risk_mult=0.5)
    assert half.qty == pytest.approx(0.8) and half.risk_usd == pytest.approx(5.0, abs=0.01)
    # stop distance in price terms is unchanged by derating
    assert half.sl_price == pytest.approx(size_trade(cfg, "long", 100.0, None, F).sl_price, abs=0.01)
    # derated below the exchange minimum notional -> trade skipped
    assert not size_trade(cfg, "long", 100.0, None, SymbolFilters(0.01, 0.001, 0.001, 100.0), risk_mult=0.5).ok
    assert not size_trade(cfg, "long", 100.0, None, F, risk_mult=0.0).ok


def test_levels_for_fill():
    sl, tp = levels(Settings(), "long", 100.0, 1.6, None, F)
    assert (sl, tp) == (pytest.approx(93.75), pytest.approx(103.13, abs=0.01))
    assert levels(Settings().merge({"exit_mode": "atr"}), "long", 100.0, 1.0, None, F) is None


# ------------------------------------------------------------------ trailing
def test_trail_activates_after_r_multiple():
    # long: entry 100, init SL 90 (R=10); activate at +1R
    active, sl = trail_update("long", 100, 90, 90, 105, 2.0, False, 1.0, 1.0)
    assert (active, sl) == (False, 90)
    active, sl = trail_update("long", 100, 90, 90, 110, 2.0, False, 1.0, 1.0)
    assert active and sl == pytest.approx(108.0)


def test_trail_never_loosens():
    active, sl = trail_update("long", 100, 90, 90, 110, 2.0, False, 1.0, 1.0)
    # a bigger ATR would give a looser candidate: stop must stay
    active, sl2 = trail_update("long", 100, 90, sl, 110, 5.0, active, 1.0, 1.0)
    assert sl2 == sl
    # best price falls back: unchanged
    _, sl3 = trail_update("long", 100, 90, sl, 105, 2.0, True, 1.0, 1.0)
    assert sl3 == sl
    # property: random walk never decreases a long stop / increases a short stop
    rng = np.random.default_rng(0)
    cur_l, cur_s, act_l, act_s = 90.0, 110.0, False, False
    best_l = best_s = 100.0
    for _ in range(500):
        p = 100 + rng.normal(0, 8)
        best_l, best_s = max(best_l, p), min(best_s, p)
        atr = abs(rng.normal(2, 1)) + 0.1
        act_l, new_l = trail_update("long", 100, 90, cur_l, best_l, atr, act_l, 1.0, 1.0)
        act_s, new_s = trail_update("short", 100, 110, cur_s, best_s, atr, act_s, 1.0, 1.0)
        assert new_l >= cur_l and new_s <= cur_s
        cur_l, cur_s = new_l, new_s


def test_trail_short():
    active, sl = trail_update("short", 100, 110, 110, 90, 2.0, False, 1.0, 1.0)
    assert active and sl == pytest.approx(92.0)


def test_trail_without_atr_is_noop():
    assert trail_update("long", 100, 90, 90, 120, None, False, 1.0, 1.0) == (False, 90)


# ------------------------------------------------------------------ kill switch
def test_kill_switch_threshold():
    assert drawdown_pct(75, 100) == pytest.approx(25.0)
    assert not kill_tripped(75.1, 100, 25.0)
    assert kill_tripped(75.0, 100, 25.0)
    assert kill_tripped(50, 100, 50.0)
    assert not kill_tripped(120, 100, 25.0)


# ------------------------------------------------------------------ correlation cap
def _series(seed, base=None, noise=0.0, n=400):
    rng = np.random.default_rng(seed)
    idx = np.arange(n) * 300_000
    if base is None:
        r = rng.normal(0, 0.002, n)
    else:
        r = base + rng.normal(0, noise, n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx), r


def test_corr_gate_blocks_after_two_correlated_same_direction():
    cand, r = _series(1)
    a, _ = _series(2, r, 0.0002)
    b, _ = _series(3, r, 0.0002)
    c, _ = _series(4)                                 # independent
    cap = CorrCap(0.8, 2)
    ok, hot = correlation_gate("long", cand, [("A", "long", a), ("C", "long", c)], cap)
    assert ok and len(hot) == 1
    ok, hot = correlation_gate("long", cand, [("A", "long", a), ("B", "long", b)], cap)
    assert not ok and {h[0] for h in hot} == {"A", "B"}
    # opposite-direction positions do not count
    ok, _ = correlation_gate("long", cand, [("A", "short", a), ("B", "short", b)], cap)
    assert ok


def test_return_corr_needs_overlap():
    a, _ = _series(1, n=50)
    assert return_corr(a, a) is None


# ------------------------------------------------------------------ stop invariant
def test_positions_without_stop():
    pos = [{"symbol": "A", "side": "long", "qty": 1.0}, {"symbol": "B", "side": "short", "qty": 2.0}]
    orders = [ConditionalOrder("1", "A", "SL", "SELL", 90, 1.0, True),
              ConditionalOrder("2", "B", "TP", "BUY", 90, 2.0, True)]       # only a TP -> B unprotected
    assert positions_without_stop(pos, orders) == ["B"]
    orders.append(ConditionalOrder("3", "B", "SL", "SELL", 110, 2.0, True))  # wrong side
    assert positions_without_stop(pos, orders) == ["B"]
    orders.append(ConditionalOrder("4", "B", "SL", "BUY", 110, 1.0, True))   # too small
    assert positions_without_stop(pos, orders) == ["B"]
    orders.append(ConditionalOrder("5", "B", "SL", "BUY", 110, 2.0, True))
    assert positions_without_stop(pos, orders) == []


def test_atr14_matches_wilder_reference():
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 1, 200))
    h, l = c + 1, c - 1
    from tb.features import atr as ref  # data/ML agent's definition
    assert atr14(h, l, c) == pytest.approx(float(ref(h, l, c)[-1]))
