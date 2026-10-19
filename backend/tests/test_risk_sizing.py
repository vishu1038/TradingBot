import numpy as np
import pandas as pd
import pytest

from tb.config import CorrCap, Settings
from tb.exchange.base import ConditionalOrder, SymbolFilters
from tb.risk import (Bracket, Book, atr14, check_entry, choose_leverage, maintenance_margin, parse_brackets,
                     sl_loss, slots_fit, correlation_gate, derate_multiplier, derated_below_min_notional, drawdown_pct,
                     effective_slots, kill_tripped, levels, liq_price,
                     max_positions, positions_without_stop, return_corr, size_trade, trail_update)

F = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0)


def test_fixed_sizing_matches_template():
    cfg = Settings()                          # 20 margin x 8 lev, TP +5, SL -10
    s = size_trade(cfg, "long", 100.0, None, F)
    assert s.ok
    assert s.notional == pytest.approx(160.0)
    assert s.leverage == 9 and s.margin == pytest.approx(160.0 / 9)     # auto leverage, default mmr 1%
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
    # an absurdly wide stop: even 1x cannot keep liquidation 1.5x further away than the stop
    cfg = Settings().merge({"sl_usd": 150.0, "slice_margin_usd": 200.0, "leverage": 1})
    r = size_trade(cfg, "long", 100.0, None, F)
    assert not r.ok and "liquidation" in r.reason
    # a wide-but-feasible stop simply gets a low leverage instead of being rejected
    r = size_trade(Settings().merge({"sl_usd": 30.0}), "long", 100.0, None, F)
    assert r.ok and r.leverage == 3 and abs(100 - r.liq_price) >= 1.5 * abs(100 - r.sl_price)


def test_available_margin_check():
    assert not size_trade(Settings(), "long", 100.0, None, F, available=10.0).ok
    assert size_trade(Settings(), "long", 100.0, None, F, available=25.0).ok


def test_liq_price():
    # Binance form, identical to tb.sim.template.liq_price
    assert liq_price("long", 100, 8, 0.005, 0.0625) == pytest.approx(100 * (0.875 - 0.0625) / 0.995)    # maintenance amount
    assert liq_price("long", 100, 8, 0.005) == pytest.approx(100 * 0.875 / 0.995)
    assert liq_price("short", 100, 8, 0.005) == pytest.approx(100 * 1.125 / 1.005)
    from tb.sim.template import liq_price as sim_liq
    assert liq_price("long", 123.4, 8, 0.005) == pytest.approx(sim_liq(123.4, 1, 8, 0.005))
    assert liq_price("short", 123.4, 8, 0.005) == pytest.approx(sim_liq(123.4, -1, 8, 0.005))


def test_max_positions():
    assert max_positions(100, Settings()) == 5                       # nominal: equity / slice margin, no cap by default
    assert max_positions(59.9, Settings()) == 2
    assert max_positions(500, Settings()) == 25
    assert max_positions(500, Settings().merge({"max_positions": 5})) == 5
    assert max_positions(10, Settings()) == 0


def test_effective_slots_free_margin_only():
    cfg = Settings()                                   # 20 slice margin, 160 notional, 0.05% fee -> 0.08 per entry
    assert effective_slots(cfg, 0, 100.0) == 4         # not 5: the 5th slice is short by the fees
    assert effective_slots(cfg, 0, 100.4) == 5
    assert effective_slots(cfg, 0, 1000.0) == 49
    assert effective_slots(cfg.merge({"max_positions": 5}), 0, 1000.0) == 5
    assert effective_slots(cfg, 3, 0.0) == 3
    assert effective_slots(cfg.merge({"max_positions": 2}), 3, 100.0) == 3     # never below what is already open


# ------------------------------------------------------------------ automatic per-trade leverage
def test_leverage_tight_vs_wide_stop():
    # liquidation must be >= 1.5x the stop distance away; mmr 1%, no bracket data
    tight = choose_leverage("long", 100.0, 99.0, 160.0, max_leverage=50)      # 1% stop
    mid = choose_leverage("long", 100.0, 93.75, 160.0, max_leverage=50)       # 6.25% stop
    wide = choose_leverage("long", 100.0, 88.0, 160.0, max_leverage=50)       # 12% stop
    assert mid == 9 and wide == 5 and tight > mid > wide
    for lev, sl in ((tight, 99.0), (mid, 93.75), (wide, 88.0)):
        assert abs(100 - liq_price("long", 100.0, lev, 0.01)) >= 1.5 * abs(100 - sl) - 1e-9
        assert abs(100 - liq_price("long", 100.0, lev + 1, 0.01)) < 1.5 * abs(100 - sl)      # and it is the highest such L
    assert choose_leverage("short", 100.0, 106.25, 160.0, max_leverage=50) == 9              # symmetric for shorts


def test_leverage_caps_setting_and_bracket():
    assert choose_leverage("long", 100.0, 99.0, 160.0, max_leverage=20) == 20
    assert choose_leverage("long", 100.0, 99.0, 160.0, max_leverage=125) > 20
    assert choose_leverage("long", 100.0, 99.0, 160.0, max_leverage=5) == 5
    br = [Bracket(0, 5000, 0.01, 0.0, 10), Bracket(5000, 50000, 0.025, 75.0, 5)]
    assert choose_leverage("long", 100.0, 99.0, 160.0, max_leverage=50, brackets=br) == 10      # bracket maxLeverage
    assert choose_leverage("long", 100.0, 99.0, 8000.0, max_leverage=50, brackets=br) == 5      # larger notional, lower tier


def test_leverage_high_mmr_coin_gets_less():
    base = choose_leverage("long", 100.0, 93.75, 160.0, 50)
    hot = choose_leverage("long", 100.0, 93.75, 160.0, 50, brackets=[Bracket(0, 1e9, 0.05, 0.0, 50)])
    assert base == 9 and hot == 7 and hot < base
    # the maintenance amount (cum) moves liquidation further away -> allows more leverage
    cum = choose_leverage("long", 100.0, 93.75, 160.0, 50, brackets=[Bracket(0, 1e9, 0.05, 10.0, 50)])
    assert cum > hot


def test_leverage_none_when_even_1x_fails():
    assert choose_leverage("long", 100.0, 30.0, 160.0, 20) is None       # 70% stop: liq at 0 is only 100% away < 105%


def test_size_trade_uses_bracket_leverage_and_mm():
    br = [Bracket(0, 1e9, 0.05, 0.0, 50)]
    s = size_trade(Settings(), "long", 100.0, None, F, brackets=br)
    assert s.leverage == 7 and s.margin == pytest.approx(160 / 7) and s.mm == pytest.approx(160 * 0.05)
    assert s.liq_price == pytest.approx(liq_price("long", 100.0, 7, 0.05))


def test_parse_brackets_and_maintenance_margin():
    raw = [{"symbol": "ETHUSDT", "notionalCoef": 1.0, "brackets": [
        {"bracket": 2, "initialLeverage": 50, "notionalCap": 100000, "notionalFloor": 10000, "maintMarginRatio": 0.01, "cum": 50.0},
        {"bracket": 1, "initialLeverage": 100, "notionalCap": 10000, "notionalFloor": 0, "maintMarginRatio": 0.005, "cum": 0.0}]}]
    b = parse_brackets(raw)["ETHUSDT"]
    assert [(t.floor, t.cap, t.mmr, t.cum, t.max_lev) for t in b] == [(0, 10000, 0.005, 0, 100), (10000, 100000, 0.01, 50, 50)]
    assert maintenance_margin(160.0, b) == pytest.approx(0.8)
    assert maintenance_margin(20000.0, b) == pytest.approx(20000 * 0.01 - 50)
    assert maintenance_margin(10000.0, b) == pytest.approx(10000 * 0.01 - 50)          # floor belongs to the next tier
    assert maintenance_margin(1e9, b) == pytest.approx(1e9 * 0.01 - 50)                 # above the last cap: last tier
    assert maintenance_margin(160.0, None) == pytest.approx(1.6)                        # default mmr 1%
    assert maintenance_margin(160.0, None, 0.02) == pytest.approx(3.2)
    assert parse_brackets({"symbol": "X", "brackets": [{"maintMarginRatio": "0.02", "initialLeverage": 20}]})["X"][0].mmr == 0.02


# ------------------------------------------------------------------ portfolio rules for a new entry
def _book(eq=200.0, avail=200.0, n=0.0, mm=0.0, risk=0.0):
    return Book(eq, avail, n, mm, risk)


def test_check_entry_each_rule_names_itself():
    cfg = Settings()                                                  # 20x total, 0.5 mm ratio, 20% open risk
    ok = check_entry(cfg, _book(), 160, 1.6, 10.2, 17.8, 0.08)
    assert ok.ok and ok.rule == ""
    r = check_entry(cfg, _book(n=3900), 160, 1.6, 10.2, 17.8, 0.08)                      # (3900+160)/200 = 20.3x
    assert not r.ok and r.rule == "total_leverage"
    r = check_entry(cfg, _book(mm=99), 160, 5.0, 10.2, 17.8, 0.08)                       # (99+5)/200 = 0.52
    assert not r.ok and r.rule == "margin_ratio"
    r = check_entry(cfg, _book(risk=30), 160, 1.6, 10.2, 17.8, 0.08)                     # 40.2/200 = 20.1%
    assert not r.ok and r.rule == "open_risk"
    r = check_entry(cfg, _book(avail=17.0), 160, 1.6, 10.2, 17.8, 0.08)
    assert not r.ok and r.rule == "free_margin"
    assert not check_entry(cfg, _book(eq=0.0), 160, 1.6, 10.2, 17.8, 0.08).ok
    # boundaries are inclusive
    assert check_entry(cfg, _book(n=3840), 160, 0, 0, 1, 0).ok                           # exactly 20.0x
    assert check_entry(cfg, _book(risk=29.8), 160, 0, 10.2, 1, 0).ok                     # exactly 20.0%


def test_sl_loss_includes_fees_and_stop_slippage():
    cfg = Settings()
    base = sl_loss(cfg, 1.6, 100.0, 93.75, "long")
    assert base == pytest.approx(10.0 + 1.6 * 93.75 * (0.0005 + 0.0002 + 0.0005))
    assert sl_loss(cfg, 1.6, 100.0, 93.75, "long", entry_fee=True) == pytest.approx(base + 1.6 * 100 * 0.0005)
    assert sl_loss(cfg, 1.6, 100.0, 106.25, "short") == pytest.approx(10.0 + 1.6 * 106.25 * 0.0012)
    assert sl_loss(cfg, 1.6, 100.0, 130.0, "long") >= 0                                  # a trailed stop in profit risks nothing


def test_slots_fit_capacity_on_stakes():
    cfg = Settings().merge({"max_side_risk_pct": 100})                                  # (side cap: test_profit_protection.py)
    # 200 USDT stake: margin 17.8+0.08 per slice -> 11 by margin, 25 by 20x, but open risk 20% = 40 / 10.3 -> 3
    assert slots_fit(cfg, _book(200, 200), 0) == 3
    loose = cfg.merge({"max_open_risk_pct": 100})
    assert slots_fit(loose, _book(200, 200), 0) == 11                                  # margin binds (200 / 17.86)
    assert slots_fit(loose.merge({"max_total_leverage": 8}), _book(200, 200), 0) == 10  # 8x of 200 = 1600 notional
    assert slots_fit(loose.merge({"max_positions": 4}), _book(200, 200), 1) == 3        # optional hard cap
    assert slots_fit(cfg, _book(100, 100), 0) == 1
    hot = [Bracket(0, 1e9, 0.2, 0.0, 20)]                                               # high-mmr coin: MM ratio binds first
    assert slots_fit(loose, _book(200, 200), 0, hot) < 11


def test_liquidation_beyond_stop_property():
    rng = np.random.default_rng(3)
    for _ in range(200):
        side = "long" if rng.random() < 0.5 else "short"
        stop_pct = rng.uniform(0.3, 20.0)
        sl = 100.0 * (1 - stop_pct / 100) if side == "long" else 100.0 * (1 + stop_pct / 100)
        mmr = rng.uniform(0.004, 0.1)
        br = [Bracket(0, 1e9, mmr, 0.0, int(rng.integers(5, 125)))]
        lev = choose_leverage(side, 100.0, sl, 160.0, 50, br)
        assert lev is not None and 1 <= lev <= min(50, br[0].max_lev)
        assert abs(100 - liq_price(side, 100.0, lev, mmr)) >= 1.5 * abs(100 - sl) - 1e-9


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
    orders.append(ConditionalOrder("4", "B", "SL", "BUY", 110, 1.0, False))  # an old-style stop with a quantity that is too small
    assert positions_without_stop(pos, orders) == ["B"]
    orders.append(ConditionalOrder("5", "B", "SL", "BUY", 110, 2.0, False))  # an old-style stop that covers the quantity
    assert positions_without_stop(pos, orders) == []
    orders.pop()
    orders.append(ConditionalOrder("6", "B", "SL", "BUY", 110, 0.0, True))   # closePosition: no quantity, covers whatever is open
    assert positions_without_stop(pos, orders) == []


def test_atr14_matches_wilder_reference():
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 1, 200))
    h, l = c + 1, c - 1
    from tb.features import atr as ref  # data/ML agent's definition
    assert atr14(h, l, c) == pytest.approx(float(ref(h, l, c)[-1]))
