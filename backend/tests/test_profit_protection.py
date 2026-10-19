"""Profit protection: break-even + trailing stop, losing-exit cooldown, one-direction risk cap, partial take-profit.

Same maths in the live engine (paper + exchange venues), the shadow simulator and the backtester (tb.sim)."""
import json
import logging
import time

import numpy as np
import pytest

from conftest import FakeData
from test_engine_core import closed_of, mk_engine, open_long, run
from test_engine_testnet import FakeVenue, mk_demo, open_
from test_exchange_binance import Resp, mk as mk_binance, qs
from test_sim_helpers import BAR, T0, build, flat, plain_cfg, sig_at
from tb import engine as E
from tb.alerts import AlertDispatcher
from tb.config import Settings
from tb.entry import BAR_MS, EntryCfg, EntryGate
from tb.exchange.base import (Balance, CloseInfo, ConditionalOrder, ExchangeError, OrderResult, PositionInfo,
                              SymbolFilters)
from tb.protect import (Protect, breakeven_price, is_loss_exit, next_stop, partial_level, partial_split,
                        stop_loss_usd)
from tb.risk import Book, check_entry
from tb.shadow import ShadowBook
from tb.sim import run_backtest

FEE = 0.0005


# ====================================================================== pure maths
def test_breakeven_price_is_entry_plus_minus_round_trip_fees():
    assert breakeven_price("long", 100.0, FEE) == pytest.approx(100.1)
    assert breakeven_price("short", 100.0, FEE) == pytest.approx(99.9)


def test_next_stop_moves_to_breakeven_at_1r_and_never_loosens():
    p = Protect(breakeven_r=1.0, trail_enabled=False, fee_rate=FEE)
    # long entry 100, initial stop 98 -> R = 2
    act, sl, why = next_stop("long", 100.0, 98.0, 98.0, 101.9, None, False, p)
    assert (sl, why) == (98.0, None)                                  # 0.95R: not yet
    act, sl, why = next_stop("long", 100.0, 98.0, 98.0, 102.0, None, False, p)
    assert (sl, why) == (pytest.approx(100.1), "breakeven")
    # price falls back: best stays, the stop never loosens
    _, sl2, why2 = next_stop("long", 100.0, 98.0, sl, 102.0, None, False, p)
    assert sl2 == sl and why2 is None
    # a better stop (e.g. a previous trail) is kept
    _, sl3, _ = next_stop("long", 100.0, 98.0, 101.5, 102.5, None, False, p)
    assert sl3 == 101.5
    # shorts mirror it
    _, s, w = next_stop("short", 100.0, 102.0, 102.0, 98.0, None, False, p)
    assert (s, w) == (pytest.approx(99.9), "breakeven")
    # off
    _, s, w = next_stop("long", 100.0, 98.0, 98.0, 110.0, None, False, Protect(breakeven_r=0.0, trail_enabled=False))
    assert (s, w) == (98.0, None)


def test_trailing_ratchets_only_up_and_waits_for_activation():
    p = Protect(breakeven_r=0.0, trail_enabled=True, trail_activate_r=1.5, trail_distance_atr=1.0)
    act, sl, why = next_stop("long", 100.0, 98.0, 98.0, 102.9, 1.0, False, p)       # 1.45R: not active
    assert (act, sl, why) == (False, 98.0, None)
    act, sl, why = next_stop("long", 100.0, 98.0, 98.0, 103.0, 1.0, False, p)       # 1.5R: best - 1 ATR
    assert (act, sl, why) == (True, 102.0, "trail")
    act, sl2, why = next_stop("long", 100.0, 98.0, sl, 104.0, 1.0, act, p)
    assert (sl2, why) == (103.0, "trail")
    act, sl3, why = next_stop("long", 100.0, 98.0, sl2, 103.5, 1.0, act, p)         # best only rises through ``best``; a lower
    assert sl3 == sl2 and act                                                       # best would not loosen: ratchet
    _, sl4, _ = next_stop("long", 100.0, 98.0, sl2, 100.5, 1.0, True, p)            # even a bad best cannot loosen
    assert sl4 == sl2


def test_partial_level_and_split():
    p = Protect(partial_tp_r=1.5, partial_tp_frac=0.5)
    assert partial_level("long", 100.0, 98.0, 112.0, p) == pytest.approx(103.0)
    assert partial_level("short", 100.0, 102.0, 88.0, p) == pytest.approx(97.0)
    assert partial_level("long", 100.0, 98.0, 103.0, p) is None                      # TP at the level: nothing to scale out of
    assert partial_level("long", 100.0, 98.0, 103.05, p) is None                     # ... or within 0.1 %
    assert partial_level("long", 100.0, 98.0, None, p) == pytest.approx(103.0)       # stop-only legs: no TP
    assert partial_level("long", 100.0, 98.0, 112.0, Protect(partial_tp_r=0.0)) is None
    f = SymbolFilters(0.01, 0.001, 0.001, 5.0)
    assert partial_split(1.6, 103.0, 0.5, f) == (0.8, 0.8)
    assert partial_split(0.3, 100.0, 0.5, SymbolFilters(0.01, 0.1, 0.1, 5.0)) == (0.1, 0.2)   # lot step; exact remainder
    assert partial_split(0.001, 100.0, 0.5, f) is None                               # half rounds to 0
    assert partial_split(0.003, 100.0, 0.5, f) is None                               # 0.3 USD notional < min notional
    assert partial_split(1.6, 103.0, 0.5, SymbolFilters(0.01, 0.001, 0.001, 100.0)) is None   # 82 USD half < 100 min notional


def test_loss_exit_classification():
    assert is_loss_exit("sl", -1.0) and is_loss_exit("trail", -0.1) and is_loss_exit("kill", 0.0)
    assert is_loss_exit("timeout", -2.0) and is_loss_exit("liquidation", -9.0)
    assert not is_loss_exit("sl", 0.5)                      # a stop that locked in profit is no reason to wait
    assert not is_loss_exit("tp", -1.0) and not is_loss_exit("partial", -1.0) and not is_loss_exit("manual", -1.0)


# ====================================================================== settings
def test_defaults_and_validation():
    s = Settings()
    assert (s.breakeven_r, s.sl_cooldown_bars, s.max_side_risk_pct, s.partial_tp_r, s.partial_tp_frac) == (1.0, 24, 60.0, 1.5, 0.5)
    assert s.trail.enabled and s.trail.activate_r == 1.5 and s.trail.distance_atr == 1.0
    api = s.to_api()
    for k in ("breakeven_r", "sl_cooldown_bars", "max_side_risk_pct", "partial_tp_r", "partial_tp_frac"):
        assert k in api
    ok = s.merge({"breakeven_r": 0, "partial_tp_r": 0, "sl_cooldown_bars": 0, "max_side_risk_pct": 100, "partial_tp_frac": 0.3},
                 allow_internal=False)
    assert ok.breakeven_r == 0 and ok.partial_tp_frac == 0.3
    for bad in ({"breakeven_r": -1}, {"breakeven_r": 21}, {"sl_cooldown_bars": -1}, {"max_side_risk_pct": 0},
                {"max_side_risk_pct": 101}, {"partial_tp_r": -0.5}, {"partial_tp_frac": 0}, {"partial_tp_frac": 1},
                {"sl_cooldown_bars": 1.5}, {"breakeven_r": "x"}):
        with pytest.raises(ValueError):
            s.merge(bad)


# ====================================================================== (3) one-direction cap
def test_side_risk_cap_blocks_the_nth_same_side_trade_and_allows_the_other_side():
    cfg = Settings().merge({"max_open_risk_pct": 35.0})                # budget 35 % of 200 = 70 -> 60 % = 42 per side
    eq = 200.0
    chk = lambda b, side, risk=10.3: check_entry(cfg, b, 160.0, 1.0, risk, 18.0, 0.08, side)
    assert chk(Book(eq, eq, open_risk=30.9, open_risk_long=30.9), "long").ok                    # 41.2 <= 42
    blocked = chk(Book(eq, eq, open_risk=41.2, open_risk_long=41.2), "long")                    # 51.5 > 42 (total 51.5 < 70)
    assert not blocked.ok and blocked.rule == "side_risk"
    assert "long" in blocked.detail and "$42.00" in blocked.detail and "$70.00" in blocked.detail
    assert chk(Book(eq, eq, open_risk=41.2, open_risk_long=41.2), "short").ok                   # the other side is free
    assert chk(Book(eq, eq, open_risk=41.2, open_risk_long=41.2), None).ok                      # no side given: legacy behaviour
    assert chk(Book(eq, eq, open_risk=41.2, open_risk_long=41.2), "long", 0.0).ok               # +0 risk stays under the cap
    # the total rule still comes first when it is the binding one
    b = chk(Book(eq, eq, open_risk=65.0, open_risk_long=10.0, open_risk_short=55.0), "long")
    assert not b.ok and b.rule == "open_risk"
    # a 100 % cap leaves only the total rule
    loose = cfg.merge({"max_side_risk_pct": 100})
    assert check_entry(loose, Book(eq, eq, open_risk=41.2, open_risk_long=41.2), 160.0, 1.0, 10.3, 18.0, 0.08, "long").ok


def test_engine_side_cap_blocks_fifth_long_but_not_a_short(store, tmp_path, caplog):
    syms = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT", "BTCUSDT")
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={s: 100.0 for s in syms}, venue_marks={s: 100.0 for s in syms},
                               settings={"capital_usd": 200.0, "max_open_risk_pct": 35.0, "short_enabled": True,
                                         "corr_cap": {"max_corr": 0.99, "max_correlated": 9}}, wallet=5000.0)
    eng.cfg = store.get_settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)
    eng.universe = list(syms[:6])
    with caplog.at_level(logging.INFO, logger="tb.engine"):
        for s in syms[:5]:
            open_(eng, s)                                              # each stop-out risks ~10.1 USD (+ fees)
        assert [p["symbol"] for p in store.get_positions()] == list(syms[:4])        # 4 x 10.3 = 41.2 <= 42; the 5th long is refused
        assert any("blocked by rule side_risk (long stop-out risk" in r.message for r in caplog.records)
        open_(eng, "FFFUSDT", "short")                                 # the short side has its own 42 USD
    assert sorted(p["symbol"] for p in store.get_positions() if p["side"] == "short") == ["FFFUSDT"]
    b = eng.book()
    assert b.open_risk_long == pytest.approx(40.3, abs=0.5) and b.open_risk_short == pytest.approx(10.1, abs=0.3)
    assert b.open_risk == pytest.approx(b.open_risk_long + b.open_risk_short)


# ====================================================================== (2) cooldown after a losing exit
def test_entry_gate_sl_cooldown_counts_only_losing_exits():
    g = EntryGate()
    cfg = EntryCfg(sl_cooldown_bars=24)
    t = 1_000_000.0
    g.record_exit("AAAUSDT", t, loss=True)
    g.record_exit("BBBUSDT", t, loss=False)
    ok, why = g.check("AAAUSDT", t + 3600, cfg)
    assert not ok and "losing exit" in why and "60 min left" in why
    assert g.check("BBBUSDT", t + 60, cfg)[0] and g.check("CCCUSDT", t + 60, cfg)[0]
    assert g.check("AAAUSDT", t + 24 * BAR_MS / 1000 + 1, cfg)[0]                  # 24 bars = 2 h later
    assert g.check("AAAUSDT", t + 60, EntryCfg(sl_cooldown_bars=0))[0]            # off
    assert EntryCfg.resolve(Settings()).sl_cooldown_bars == 24
    assert EntryCfg.resolve(Settings(), {"sl_cooldown_bars": 6}).sl_cooldown_bars == 6


def test_sl_cooldown_survives_a_restart(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng)
    p = store.get_positions()[0]
    data.marks["AAAUSDT"] = p["sl_price"] * 0.999
    run(eng.prices_cycle())
    t = store.get_trades(1)[0]
    assert t["exit_reason"] == "sl" and t["pnl"] < 0
    assert not eng.live_gate.check("AAAUSDT", t["closed_at"] + 600, EntryCfg.resolve(eng.cfg))[0]
    # a brand-new engine on the same DB rebuilds the cooldown from the trades table
    eng2, _ = mk_engine(store, tmp_path)
    cfg2 = EntryCfg.resolve(eng2.cfg)
    ok, why = eng2.live_gate.check("AAAUSDT", t["closed_at"] + 600, cfg2)
    assert not ok and "losing exit" in why
    assert eng2.live_gate.check("BBBUSDT", t["closed_at"] + 600, cfg2)[0]
    assert eng2.live_gate.check("AAAUSDT", t["closed_at"] + 24 * 300 + 1, cfg2)[0]
    # the signal path honours it: a new bar's signal on AAAUSDT inside the window is skipped, no position opens
    eng2.clock = lambda: t["closed_at"] + 600.0
    open_long(eng2)                                                       # (try_open itself is gate-less; on_new_bar checks)
    store.execute("DELETE FROM positions")
    ok, _ = eng2.live_gate.check("AAAUSDT", eng2.clock(), cfg2)
    assert not ok


def test_a_winning_stop_or_tp_does_not_start_the_cooldown(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng)
    p = store.get_positions()[0]
    data.marks["AAAUSDT"] = p["tp_price"] * 1.001
    run(eng.prices_cycle())
    t = store.get_trades(1)[0]
    assert t["exit_reason"] == "tp"
    assert eng.live_gate.check("AAAUSDT", t["closed_at"] + 60, EntryCfg.resolve(eng.cfg))[0]
    eng2, _ = mk_engine(store, tmp_path)
    assert eng2.live_gate.check("AAAUSDT", t["closed_at"] + 60, EntryCfg.resolve(eng2.cfg))[0]


# ====================================================================== (1) break-even + trailing in the engine (paper)
ATR_EXITS = {"exit_mode": "atr", "sl_atr": 1.0, "tp_atr": 8.0, "partial_tp_r": 0.0, "breakeven_r": 1.0,
             "trail": {"enabled": False, "activate_r": 1.5, "distance_atr": 1.0}}


def pump(eng, data, price, dt=0.0, sym="AAAUSDT"):
    data.marks[sym] = price
    t0 = eng.clock()
    eng.clock = lambda: t0 + dt
    run(eng.prices_cycle())


def _paper(store, tmp_path, **over):
    eng, data = mk_engine(store, tmp_path, settings={**ATR_EXITS, **over})
    eng.cfg = store.get_settings()
    # funding settles on the real 00/08/16 UTC grid; pump() moves the clock forward, so a run shortly before a
    # settlement would book funding mid-test. These tests assert funding == 0: keep the settlement out of reach.
    eng.ex._slot = 1 << 40
    open_long(eng)
    return eng, data, store.get_positions()[0]


def test_engine_moves_stop_to_breakeven_on_the_exchange_and_never_loosens(store, tmp_path):
    eng, data, p = _paper(store, tmp_path)
    entry, risk, sl0 = p["entry_price"], p["entry_price"] - p["sl_price"], p["sl_price"]
    old_id = p["sl_order_id"]
    pump(eng, data, entry + 0.9 * risk)                      # not yet 1R
    assert store.get_positions()[0]["sl_price"] == sl0
    pump(eng, data, entry + 1.05 * risk, dt=300)               # 1R reached -> break-even
    q = store.get_positions()[0]
    be = round(entry * (1 + 2 * FEE), 2)
    assert q["sl_price"] == pytest.approx(be, abs=0.011) and q["sl_order_id"] != old_id and not q["trail_active"]
    sl_orders = [o for o in eng.ex.open_conditional_orders("AAAUSDT") if o.kind == "SL"]
    assert [o.trigger_price for o in sl_orders] == [q["sl_price"]]            # old exchange stop replaced, not duplicated
    assert all(o.qty == pytest.approx(p["qty"]) and o.reduce_only for o in sl_orders)
    pump(eng, data, entry + 0.4 * risk, dt=900)                # pull back: stays at break-even
    assert store.get_positions()[0]["sl_price"] == q["sl_price"]
    pump(eng, data, q["sl_price"] * 0.9995, dt=1200)           # through the break-even stop: exits ~flat, NOT a big loss
    assert store.get_positions() == []
    t = store.get_trades(1)[0]
    assert t["exit_reason"] == "sl" and -0.25 < t["pnl"] < 0.05                # ~ -fees, versus -risk*qty = -0.3..


def test_engine_stop_moves_are_throttled_to_once_per_bar(store, tmp_path):
    eng, data, p = _paper(store, tmp_path, trail={"enabled": True, "activate_r": 1.5, "distance_atr": 1.0}, breakeven_r=1.0)
    entry, atr = p["entry_price"], p["atr"]
    risk = entry - p["sl_price"]
    pump(eng, data, entry + 1.05 * risk)                       # BE move in bar 0
    sl_a = store.get_positions()[0]["sl_price"]
    pump(eng, data, entry + 3.0 * risk)                        # trail would move it again, but not in the same bar
    assert store.get_positions()[0]["sl_price"] == sl_a
    pump(eng, data, entry + 3.0 * risk, dt=301)                # next bar: trail ratchets
    sl_b = store.get_positions()[0]["sl_price"]
    assert sl_b > sl_a and sl_b == pytest.approx(entry + 3.0 * risk - atr, abs=0.02)
    assert store.get_positions()[0]["trail_active"]
    pump(eng, data, entry + 2.6 * risk, dt=700)                # lower price, later bar: never loosens
    assert store.get_positions()[0]["sl_price"] == sl_b
    pump(eng, data, entry + 3.001 * risk, dt=1000)             # a sub-0.1 % improvement is not worth an exchange round trip
    assert store.get_positions()[0]["sl_price"] == sl_b


def test_trailing_off_and_breakeven_off_leave_the_stop_alone(store, tmp_path):
    eng, data, p = _paper(store, tmp_path, breakeven_r=0.0)
    entry, risk = p["entry_price"], p["entry_price"] - p["sl_price"]
    pump(eng, data, entry + 3 * risk)
    assert store.get_positions()[0]["sl_price"] == p["sl_price"]


def test_failed_stop_replacement_keeps_the_old_stop_and_retries(store, tmp_path, caplog):
    prod = {"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0}
    eng, data, venue = mk_demo(store, tmp_path, prod_marks=prod, venue_marks=dict(prod), wallet=100.0,
                               settings={**ATR_EXITS, "max_open_risk_pct": 100.0})
    eng.cfg = store.get_settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)
    open_(eng, "AAAUSDT")
    p = store.get_positions()[0]
    entry, risk = p["entry_price"], p["entry_price"] - p["sl_price"]
    old = [o for o in venue.orders if o.kind == "SL"][0]
    real = venue.place_stop
    venue.place_stop = lambda *a: (_ for _ in ()).throw(ExchangeError("temporarily unavailable", code=-1001))
    venue.marks["AAAUSDT"] = entry + 1.1 * risk
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="tb.engine"):
        run(eng.prices_cycle())
    q = store.get_positions()[0]
    assert q["sl_price"] == p["sl_price"] and q["sl_order_id"] == old.order_id            # DB unchanged
    assert [o.order_id for o in venue.orders if o.kind == "SL"] == [old.order_id]         # the old stop is still live
    assert any("FAILED" in r.message and "old stop stays live" in r.message for r in caplog.records)
    # still protected: the invariant check does not close it
    run(eng.prices_cycle())
    assert len(store.get_positions()) == 1
    # the venue recovers; the retry is rate limited (30 s) and then succeeds
    venue.place_stop = real
    t0 = eng.clock()
    eng.clock = lambda: t0 + 10
    run(eng.prices_cycle())
    assert store.get_positions()[0]["sl_price"] == p["sl_price"]
    eng.clock = lambda: t0 + 45
    run(eng.prices_cycle())
    q = store.get_positions()[0]
    assert q["sl_price"] > p["sl_price"] and [o.trigger_price for o in venue.orders if o.kind == "SL"] == [q["sl_price"]]


def test_replace_conditional_on_binance_algo_places_new_then_cancels_old():
    ex, s = mk_binance(Resp(200, {"algoId": 7001}), Resp(200, {"algoId": 5555}), conditional_mode="algo")
    new, stale = ex.replace_conditional("ETHUSDT", "SL", "SELL", 2300.5, 0.067, old_id="a:5555")
    assert (new, stale) == ("a:7001", None)
    assert [(c["method"], c["url"].split("?")[0].rsplit("/", 1)[-1]) for c in s.calls] == [("POST", "algoOrder"), ("DELETE", "algoOrder")]
    post, delete = qs(s.calls[0]), qs(s.calls[1])
    assert post["type"] == "STOP_MARKET" and post["triggerPrice"] == "2300.5" and post["quantity"] == "0.067"
    assert post["reduceOnly"] == "true" and post["algoType"] == "CONDITIONAL" and post["side"] == "SELL"
    assert delete["algoId"] == "5555"


def test_replace_conditional_legacy_order_ids_use_the_order_endpoint():
    ex, s = mk_binance(Resp(200, {"orderId": 91}), Resp(200, {"orderId": 90}), conditional_mode="legacy")
    new, stale = ex.replace_conditional("ETHUSDT", "TP", "SELL", 2400.0, 0.05, old_id="o:90")
    assert (new, stale) == ("o:91", None)
    assert qs(s.calls[0])["type"] == "TAKE_PROFIT_MARKET" and s.calls[1]["method"] == "DELETE"
    assert qs(s.calls[1])["orderId"] == "90"


def test_replace_conditional_failure_to_place_never_touches_the_old_order():
    bad = Resp(400, {"code": -1111, "msg": "Precision is over the maximum"})
    ex, s = mk_binance(bad, bad, bad, conditional_mode="algo")
    with pytest.raises(ExchangeError):
        ex.replace_conditional("ETHUSDT", "SL", "SELL", 2300.5, 0.067, old_id="a:5555", retries=3, pause_s=0)
    assert [c["method"] for c in s.calls] == ["POST", "POST", "POST"]            # retried, never a DELETE: old stop stays live
    # -2021 (would trigger immediately) is not retried
    ex2, s2 = mk_binance(Resp(400, {"code": -2021, "msg": "Order would immediately trigger."}), conditional_mode="algo")
    with pytest.raises(ExchangeError) as e:
        ex2.replace_conditional("ETHUSDT", "SL", "SELL", 2300.5, 0.067, old_id="a:1", pause_s=0)
    assert e.value.code == -2021 and len(s2.calls) == 1
    # a retry that works on the 2nd attempt still cancels the old stop afterwards
    ex3, s3 = mk_binance(bad, Resp(200, {"algoId": 9}), Resp(200, {}), conditional_mode="algo")
    assert ex3.replace_conditional("ETHUSDT", "SL", "SELL", 2300.5, 0.067, old_id="a:1", pause_s=0) == ("a:9", None)
    assert [c["method"] for c in s3.calls] == ["POST", "POST", "DELETE"]


def test_replace_conditional_reports_an_old_order_that_could_not_be_cancelled():
    ex, s = mk_binance(Resp(200, {"algoId": 12}), Resp(500, {"code": -1000, "msg": "boom"}), conditional_mode="algo")
    new, stale = ex.replace_conditional("ETHUSDT", "SL", "SELL", 2300.5, 0.067, old_id="a:11")
    assert new == "a:12" and stale == "a:11"                       # harmless (reduce-only); the engine retries the cancel


def test_binance_last_close_does_not_count_a_booked_partial_twice():
    fills = [
        {"id": 1, "orderId": 100, "side": "BUY", "qty": "1.6", "price": "100", "realizedPnl": "0", "commission": "0.08",
         "time": 1_000_000},
        {"id": 2, "orderId": 200, "side": "SELL", "qty": "0.8", "price": "103", "realizedPnl": "2.4", "commission": "0.0412",
         "time": 1_100_000},                                           # the partial (booked as its own trade)
        {"id": 3, "orderId": 300, "side": "SELL", "qty": "0.8", "price": "100.1", "realizedPnl": "0.08", "commission": "0.04004",
         "time": 1_200_000},
    ]
    ex, s = mk_binance(Resp(200, fills), Resp(200, [{"income": "-0.2"}]))
    info = ex.last_close("ETHUSDT", "long", 900_000, booked={"order_ids": ["o:200"], "entry_fee": 0.04, "funding": -0.1})
    assert info.exit_price == pytest.approx(100.1) and info.pnl == pytest.approx(0.08)
    assert info.fees == pytest.approx(0.08 + 0.04004 - 0.04)            # entry fee left for the remainder + the final exit fee
    assert info.funding == pytest.approx(-0.1)                         # signed: the partial already took -0.1 of the -0.2
    assert info.net_pnl == pytest.approx(0.08 - 0.08004 - 0.1)


# ====================================================================== (4) partial take-profit
PART = {**ATR_EXITS, "partial_tp_r": 1.5, "partial_tp_frac": 0.5, "breakeven_r": 1.0}


def test_partial_tp_closes_half_books_net_pnl_and_resizes_stop_and_tp(store, tmp_path):
    eng, data, p = _paper(store, tmp_path, **PART)
    entry, risk, qty = p["entry_price"], p["entry_price"] - p["sl_price"], p["qty"]
    half = SymbolFilters(0.01, 0.001, 0.001, 5.0).round_qty(qty * 0.5)       # floored to the lot step
    px = round(entry + 1.6 * risk, 2)
    pump(eng, data, px)
    q = store.get_positions()[0]
    assert q["qty"] == pytest.approx(qty - half) and q["notional"] == pytest.approx((qty - half) * entry)
    rows = store.get_trades(5)
    assert len(rows) == 1 and rows[0]["exit_reason"] == "partial"
    r = rows[0]
    gross = (px - entry) * half
    entry_fee = p["entry_fee"] * half / qty
    assert r["qty"] == pytest.approx(half) and r["exit_price"] == pytest.approx(px) and r["position_id"] == p["id"]
    assert r["fees"] == pytest.approx(entry_fee + half * px * FEE)                   # fees are a cost: entry share + exit fee
    assert r["funding"] == pytest.approx(0.0)
    assert r["pnl"] == pytest.approx(gross - r["fees"])                              # NET
    assert q["entry_fee"] == pytest.approx(p["entry_fee"] - entry_fee)
    # exchange orders: stop and TP resized to the remainder, stop at least at break-even; old ones gone
    conds = eng.ex.open_conditional_orders("AAAUSDT")
    assert sorted(o.kind for o in conds) == ["SL", "TP"] and all(o.qty == pytest.approx(qty - half) for o in conds)
    sl = [o for o in conds if o.kind == "SL"][0]
    assert sl.trigger_price >= round(entry * (1 + 2 * FEE), 2) - 1e-9 and sl.trigger_price == pytest.approx(q["sl_price"])
    assert [o.order_id for o in conds if o.kind == "TP"] == [q["tp_order_id"]]
    # once per position: a later higher price does not scale out again
    pump(eng, data, round(entry + 2.4 * risk, 2), dt=600)
    assert len(store.get_trades(5)) == 1 and store.get_positions()[0]["qty"] == pytest.approx(qty - half)
    # the rest runs into the TP; both rows together equal the wallet change (net pnl convention holds for both parts)
    pump(eng, data, q["tp_price"] * 1.001, dt=900)
    assert store.get_positions() == []
    rows = store.get_trades(5)
    assert [t["exit_reason"] for t in rows] == ["tp", "partial"]
    last = rows[0]
    assert last["qty"] == pytest.approx(qty - half)
    assert last["fees"] == pytest.approx(p["entry_fee"] - entry_fee + (qty - half) * q["tp_price"] * FEE)
    assert last["pnl"] == pytest.approx((q["tp_price"] - entry) * (qty - half) - last["fees"])
    assert sum(t["pnl"] for t in rows) == pytest.approx(eng.ex.balance().wallet - 100.0, abs=1e-6)
    # stats: one position with two rows counts as ONE winning trade
    st = store.trade_stats()
    assert st["n"] == 1 and st["wins"] == 1 and st["net"] == pytest.approx(sum(t["pnl"] for t in rows))


def test_partial_tp_state_survives_a_restart_and_is_not_repeated(store, tmp_path):
    eng, data, p = _paper(store, tmp_path, **PART)
    entry, risk = p["entry_price"], p["entry_price"] - p["sl_price"]
    pump(eng, data, round(entry + 1.6 * risk, 2))
    left = store.get_positions()[0]["qty"]
    assert store.kv_get("protect")[str(p["id"])]["partial"] is True
    eng2, data2 = mk_engine(store, tmp_path)
    assert eng2.prot[str(p["id"])]["partial"] is True
    data2.marks["AAAUSDT"] = round(entry + 2.0 * risk, 2)
    run(eng2.prices_cycle())
    assert len(store.get_trades(5)) == 1 and store.get_positions()[0]["qty"] == pytest.approx(left)


def test_partial_tp_skipped_when_a_part_would_be_below_min_notional(store, tmp_path):
    eng, data = mk_engine(store, tmp_path, settings=PART)
    eng.cfg = store.get_settings()
    from tb.exchange.base import SymbolInfo
    data.exchange_info = lambda: {s: SymbolInfo(s, filters=SymbolFilters(0.01, 0.001, 0.001, 100.0)) for s in data.marks}
    open_long(eng)
    p = store.get_positions()[0]                                   # notional 160 >= 100, but half = 80 < 100
    entry, risk = p["entry_price"], p["entry_price"] - p["sl_price"]
    pump(eng, data, round(entry + 1.6 * risk, 2))
    q = store.get_positions()[0]
    assert q["qty"] == p["qty"] and store.get_trades(5) == []
    assert eng.prot[str(p["id"])]["partial"] == "skipped"
    pump(eng, data, round(entry + 2.0 * risk, 2), dt=600)           # and it is not retried every cycle
    assert store.get_trades(5) == [] and store.get_positions()[0]["qty"] == p["qty"]


def test_partial_tp_off_or_tp_at_the_level_does_nothing(store, tmp_path):
    eng, data, p = _paper(store, tmp_path, **{**PART, "partial_tp_r": 0.0})
    entry, risk = p["entry_price"], p["entry_price"] - p["sl_price"]
    pump(eng, data, round(entry + 2 * risk, 2))
    assert store.get_trades(5) == [] and store.get_positions()[0]["qty"] == p["qty"]


def test_partial_tp_is_pointless_when_the_tp_sits_at_the_level(store, tmp_path):
    # TP exactly where the partial would fire (tp_atr 1.5 x sl_atr 1.0 -> TP at 1.5R): the TP order handles it
    eng2, data2, p2 = _paper(store, tmp_path, **{**PART, "tp_atr": 1.5})
    assert p2["tp_price"] == pytest.approx(p2["entry_price"] + 1.5 * (p2["entry_price"] - p2["sl_price"]), abs=0.02)
    from tb.protect import partial_level as pl
    assert pl("long", p2["entry_price"], p2["init_sl_price"], p2["tp_price"], Protect.from_settings(eng2.cfg)) is None


def test_partial_tp_failure_is_retried_later_without_booking_anything(store, tmp_path):
    eng, data, p = _paper(store, tmp_path, **PART)
    entry, risk = p["entry_price"], p["entry_price"] - p["sl_price"]
    real = eng.ex.market_order
    eng.ex.market_order = lambda *a, **k: (_ for _ in ()).throw(ExchangeError("rejected", code=-2022))
    pump(eng, data, round(entry + 1.6 * risk, 2))
    assert store.get_trades(5) == [] and store.get_positions()[0]["qty"] == p["qty"] and not eng.prot[str(p["id"])].get("partial")
    eng.ex.market_order = real
    pump(eng, data, round(entry + 1.6 * risk, 2), dt=40)
    assert len(store.get_trades(5)) == 1 and store.get_positions()[0]["qty"] == pytest.approx(p["qty"] - SymbolFilters(0.01, 0.001, 0.001, 5.0).round_qty(p["qty"] * 0.5))


class PartialVenue(FakeVenue):
    """FakeVenue whose reduce-only market orders may close only part of the position (like the real exchange)."""

    def market_order(self, symbol, side, qty, reduce_only=False):
        if reduce_only and symbol in self.pos and qty < self.pos[symbol].qty - 1e-9:
            px = self.marks[symbol]
            pos = self.pos[symbol]
            self.log.append(("market", symbol, side, qty, True, px))
            pos.qty = round(pos.qty - qty, 8)
            return OrderResult(f"o:{len(self.log)}", symbol, side, qty, px, "FILLED", qty * px * 0.0005)
        return super().market_order(symbol, side, qty, reduce_only)


def test_partial_tp_on_an_exchange_venue_resizes_orders_and_keeps_old_ones_on_failure(store, tmp_path):
    prod = {"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0}
    eng, data, venue = mk_demo(store, tmp_path, prod_marks=prod, venue_marks=dict(prod), wallet=100.0,
                               settings={**PART, "max_open_risk_pct": 100.0})
    venue.__class__ = PartialVenue
    eng.cfg = store.get_settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)
    open_(eng, "AAAUSDT")
    p = store.get_positions()[0]
    entry, risk = p["entry_price"], p["entry_price"] - p["sl_price"]
    old_sl = [o for o in venue.orders if o.kind == "SL"][0]
    real = venue.place_stop
    calls = {"n": 0}

    def flaky(*a):                                                   # BE move is attempted first and fails, then the resize
        calls["n"] += 1
        raise ExchangeError("temporarily unavailable", code=-1001)

    venue.place_stop = flaky
    venue.marks["AAAUSDT"] = round(entry + 1.6 * risk, 2)
    run(eng.prices_cycle())
    q = store.get_positions()[0]
    assert q["qty"] == pytest.approx(p["qty"] / 2) and venue.pos["AAAUSDT"].qty == pytest.approx(p["qty"] / 2)
    assert [t["exit_reason"] for t in store.get_trades(5)] == ["partial"]
    # the resize failed: the OLD (oversized) stop is still live, so the position never lost protection
    assert [o.order_id for o in venue.orders if o.kind == "SL"] == [old_sl.order_id]
    assert eng.prot[str(p["id"])]["resize"] is True
    run(eng.prices_cycle())                                          # invariant check passes (stop covers the position)
    assert len(store.get_positions()) == 1
    venue.place_stop = real
    t0 = eng.clock()
    eng.clock = lambda: t0 + 40
    run(eng.prices_cycle())
    sls = [o for o in venue.orders if o.kind == "SL"]
    tps = [o for o in venue.orders if o.kind == "TP"]
    assert len(sls) == 1 and sls[0].qty == pytest.approx(p["qty"] / 2) and sls[0].trigger_price >= round(entry * (1 + 2 * FEE), 2) - 1e-9
    assert len(tps) == 1 and tps[0].qty == pytest.approx(p["qty"] / 2)
    assert not eng.prot[str(p["id"])].get("resize")
    # the exchange close-out of the remainder is told about the booked partial
    assert eng.prot[str(p["id"])]["booked"]["order_ids"]


# ====================================================================== paper exchange
def test_paper_exchange_reduce_only_partial_keeps_the_position_and_prorates_costs(cfg):
    from tb.exchange.paper import PaperExchange
    data = FakeData(("AAAUSDT",))
    ex = PaperExchange(data, cfg.merge({"slippage_pct": 0.0}, allow_internal=True), 1000.0)
    ex.marks["AAAUSDT"] = 100.0
    ex.set_leverage("AAAUSDT", 5)
    ex.market_order("AAAUSDT", "BUY", 1.6)
    sl = ex.place_stop("AAAUSDT", "SELL", 98.0, 1.6)
    ex.marks["AAAUSDT"] = 103.0
    w = ex.balance().wallet
    r = ex.market_order("AAAUSDT", "SELL", 0.8, reduce_only=True)
    assert r.qty == pytest.approx(0.8) and ex.positions()[0].qty == pytest.approx(0.8)
    assert r.fee == pytest.approx(0.5 * 1.6 * 100 * FEE + 0.8 * 103 * FEE)                  # entry share + exit fee
    assert ex.balance().wallet - w == pytest.approx(0.8 * 3.0 - 0.8 * 103 * FEE)
    assert ex.positions()[0].margin == pytest.approx(1.6 * 100 / 5 / 2)
    new_id, stale = ex.replace_conditional("AAAUSDT", "SL", "SELL", 100.1, 0.8, sl)
    assert stale is None and [(o.order_id, o.trigger_price, o.qty) for o in ex.open_conditional_orders()] == [(new_id, 100.1, 0.8)]


# ====================================================================== shadow + sim parity on one hand-made path
def _path():
    """ATR14 = 2 from bar 13 on; signal on bar 20 -> entry at the open of bar 21 (100). R = 2 (stop 98), TP 112.
    bar 22: high 103.5 (1.5R level 103 reached, best >= 1R); bar 23: low 100.0 hits the break-even stop 100.1."""
    n = 30
    d = flat(n)
    for i in range(n):
        d["h"][i], d["l"][i] = 101.0, 99.0
    d["h"][22], d["l"][22] = 103.5, 100.3
    d["h"][23], d["l"][23] = 102.0, 100.0
    for i in range(23, n):
        d["o"][i] = d["c"][i] = 101.0
    return n, d


EXITS = {"exit_mode": "atr", "sl_atr": 1.0, "tp_atr": 6.0, "sl_usd": 10.0}
EXPECT_PARTIAL_NET = 0.8 * 3.0 + 0.8 * (100.1 - 100.0) - (FEE * 160 + FEE * 0.8 * 103.0 + FEE * 0.8 * 100.1)


def test_sim_breakeven_plus_partial_hand_computed():
    n, d = _path()
    cfg = plain_cfg(**EXITS, breakeven_r=1.0, partial_tp_r=1.5, partial_tp_frac=0.5)
    r = run_backtest({"X": build(d)}, {"X": sig_at(n, [20])}, cfg)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t["exit_reason"] == "sl" and t["exit_price"] == pytest.approx(100.1) and t["qty"] == pytest.approx(1.6)
    assert t["partial_qty"] == pytest.approx(0.8) and r.stats["partials"] == 1
    assert t["gross_pnl"] == pytest.approx(0.8 * 3.0 + 0.8 * 0.1)
    assert t["fees"] == pytest.approx(FEE * 160 + FEE * 0.8 * 103.0 + FEE * 0.8 * 100.1)
    assert t["pnl"] == pytest.approx(EXPECT_PARTIAL_NET)                      # NET of both parts
    assert r.equity[-1] == pytest.approx(1000 + EXPECT_PARTIAL_NET)           # wallet agrees with the trade row


def test_sim_breakeven_only_and_off_switches():
    n, d = _path()
    rbe = run_backtest({"X": build(d)}, {"X": sig_at(n, [20])}, plain_cfg(**EXITS, breakeven_r=1.0, partial_tp_r=0.0))
    be = rbe.trades[0]
    assert rbe.stats["breakeven_moves"] == 1 and rbe.stats["partials"] == 0
    assert be["exit_reason"] == "sl" and be["exit_price"] == pytest.approx(100.1)
    assert be["pnl"] == pytest.approx(1.6 * 0.1 - FEE * 1.6 * (100.0 + 100.1))          # ~flat, a small fee loss
    off = run_backtest({"X": build(d)}, {"X": sig_at(n, [20])}, plain_cfg(**EXITS, breakeven_r=0.0, partial_tp_r=0.0)).trades
    assert off == []                                                                     # no protection: still open (stop 98 not hit)


def test_sim_trailing_ratchet_uses_the_new_defaults_when_enabled():
    n, d = _path()
    d["h"][22], d["l"][22] = 104.0, 100.3                  # best 104 -> trail 102 (distance 1 ATR) after 1.5R
    d["h"][23], d["l"][23] = 104.0, 103.0
    d["h"][24], d["l"][24] = 103.5, 101.9                  # pierces the ratcheted 102
    d["o"][23] = d["o"][24] = 103.5                        # (no gap-down opens)
    cfg = plain_cfg(**EXITS, breakeven_r=0.0, partial_tp_r=0.0, trail_enabled=True, trail_activate_r=1.5, trail_distance_atr=1.0)
    t = run_backtest({"X": build(d)}, {"X": sig_at(n, [20])}, cfg).trades[0]
    assert t["exit_reason"] == "trail" and t["exit_price"] == pytest.approx(102.0)


def test_shadow_matches_the_sim_on_the_same_path():
    n, d = _path()
    kl = build(d)
    cfg = Settings().merge({**EXITS, "slippage_pct": 0.0, "stop_slippage_pct": 0.0, "trail": {"enabled": False},
                            "breakeven_r": 1.0, "partial_tp_r": 1.5, "partial_tp_frac": 0.5, "max_side_risk_pct": 100},
                           allow_internal=True)
    book = ShadowBook(cfg)
    f = SymbolFilters(0.01, 0.001, 0.001, 5.0)
    t_entry = int(kl["open_time"].iloc[21]) // 1000
    assert book.open("m", "X", "long", 0.9, 100.0, t_entry, 2.0, f)
    p = book.pos["m|X"]
    assert p.qty == pytest.approx(1.6) and p.sl == pytest.approx(98.0) and p.tp == pytest.approx(112.0)
    out = []
    for i in range(22, 25):
        r = kl.iloc[i]
        out += book.on_bar("X", int(r.open_time), float(r.low), float(r.high), (int(r.open_time) + BAR) / 1000.0)
    assert len(out) == 1
    tr = out[0]
    # (the shadow book fills a stop at the bar's adverse extreme - 100.0 here - where the sim fills at the stop 100.1)
    assert tr["exit_reason"] == "sl" and tr["exit_price"] == pytest.approx(100.0) and tr["qty"] == pytest.approx(1.6)
    fees = FEE * 160 + FEE * 0.8 * 103.0 + FEE * 0.8 * 100.0
    assert tr["fees"] == pytest.approx(fees) and tr["pnl"] == pytest.approx(0.8 * 3.0 + 0.0 - fees)
    # off switches give the old behaviour: still open
    book2 = ShadowBook(cfg.merge({"breakeven_r": 0, "partial_tp_r": 0}, allow_internal=True))
    book2.open("m", "X", "long", 0.9, 100.0, t_entry, 2.0, f)
    out2 = []
    for i in range(22, 25):
        r = kl.iloc[i]
        out2 += book2.on_bar("X", int(r.open_time), float(r.low), float(r.high), (int(r.open_time) + BAR) / 1000.0)
    assert out2 == [] and "m|X" in book2.pos
    # persisted state round trip keeps the protection fields
    st = json.loads(json.dumps(book.export_state()))
    b3 = ShadowBook(cfg)
    b3.import_state(st)
    assert b3.pos == book.pos


def test_shadow_side_cap_and_cooldown_gate():
    cfg = Settings().merge({"max_open_risk_pct": 35.0, "capital_usd": 200.0, **{"slippage_pct": 0.0, "stop_slippage_pct": 0.0}},
                           allow_internal=True)
    book = ShadowBook(cfg)
    f = SymbolFilters(0.01, 0.001, 0.001, 5.0)
    opened = [book.open("m", s, "long", 0.9, 100.0, 1000 + i, 2.0, f) for i, s in enumerate("ABCDEF")]
    assert opened == [True, True, True, True, False, False]                  # 4 x ~10.2 under the 42 USD side cap
    assert book.open("m", "G", "short", 0.9, 100.0, 2000, 2.0, f)           # the other side is free
    assert book.open("other-model", "A", "long", 0.9, 100.0, 2000, 2.0, f)  # caps are per model
    # losing exits start the cooldown in the shadow gate, with the same rule as the live book
    g = EntryGate()
    g.record_exit("A", 5000.0, loss=is_loss_exit("sl", -3.0))
    assert not g.check("A", 5600.0, EntryCfg(sl_cooldown_bars=24))[0]


def test_sim_sl_cooldown_blocks_re_entry_after_a_stop_out_and_allows_it_later():
    n = 80
    d = flat(n)
    for i in range(n):
        d["h"][i], d["l"][i] = 100.5, 99.5
    d["l"][22] = 90.0                                        # stops the first entry (bar 21) out: SL 93.75 (fixed mode)
    d["l"][27] = 90.0                                        # ... and the second one (bar 26), when there is one
    sigs = sig_at(n, [20, 25, 60])                             # 25 is inside the 24-bar window, 60 is after it
    off = run_backtest({"X": build(d)}, {"X": sigs}, plain_cfg(sl_cooldown_bars=0))
    on = run_backtest({"X": build(d)}, {"X": sigs}, plain_cfg(sl_cooldown_bars=24))
    assert off.stats["entries"] == 3 and on.stats["entries"] == 2
    assert on.stats["skip_sl_cooldown"] == 1
    assert [t["exit_reason"] for t in on.trades] == ["sl"]


def test_sim_side_cap_blocks_the_second_same_side_entry_but_not_the_other_side():
    n = 40
    d = flat(n)
    for i in range(n):
        d["h"][i], d["l"][i] = 100.5, 99.5
    kls = {s: build(d) for s in ("A", "B", "C")}
    sigs = {"A": sig_at(n, [10]), "B": sig_at(n, [10]), "C": sig_at(n, [10], "short")}
    # equity 100: budget 20 % = 20 USD, side cap 60 % = 12 USD; one fixed-mode trade risks ~10.2 USD
    cfg = plain_cfg(start_equity=100.0, max_side_risk_pct=60.0, max_open_risk_pct=20.0, short_enabled=True)
    r = run_backtest(kls, sigs, cfg)
    assert r.stats["entries"] == 2 and r.stats["skip_side_risk"] == 1                  # A long yes, B long no, C short yes
    off = run_backtest(kls, sigs, plain_cfg(start_equity=100.0, short_enabled=True))
    assert off.stats["entries"] == 3 and off.stats["skip_side_risk"] == 0


def test_stop_loss_usd_matches_the_live_risk_function():
    from tb.risk import sl_loss
    cfg = Settings()
    a = sl_loss(cfg, 1.6, 100.0, 93.75, "long", entry_fee=True)
    b = stop_loss_usd("long", 1.6, 100.0, 93.75, cfg.fee_taker, cfg.slippage_pct, cfg.stop_slippage_pct, entry_fee=True)
    assert a == pytest.approx(b)
