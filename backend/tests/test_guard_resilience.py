"""Incident 2026-10-19 (testnet): stops triggered but the fills failed, every close was rejected with -2022, one coin ran past
its liquidation price and the guard flattened the WHOLE account in a loop. These tests pin the per-symbol behaviour:
backoff, liquidation awareness, the emergency close order, the whitelist universe."""
import dataclasses
import logging
import time

import pytest

from conftest import FakeData
from test_engine_core import run
from test_engine_testnet import FakeVenue, mk_demo, open_
from test_risk_safety import Clock
from tb import emergency as EM
from tb import guard as G
from tb import universe as uni
from tb.config import Settings
from tb.exchange.base import (Balance, ConditionalOrder, ExchangeError, NetworkError, OrderNotFilled, OrderResult,
                              PositionInfo, SymbolFilters, SymbolInfo)
from tb.safety import DEGRADED, FLATTEN, HALTED, NORMAL

from test_exchange_binance import Resp, mk, qs


# ------------------------------------------------------------------ fakes
class Rec:
    """AlertDispatcher stand-in: records what would have gone to Google Chat."""

    def __init__(self):
        self.liq, self.sym, self.safety_calls = [], [], []

    def liquidation(self, symbol, *a, **k):
        self.liq.append(symbol)

    def symbol_problem(self, symbol, what, detail, mode=""):
        self.sym.append((symbol, what))

    def safety(self, state, reasons, mode=""):
        self.safety_calls.append(state)

    def send(self, *a, **k):
        pass

    def engine_error(self, *a, **k):
        pass

    def trade_closed(self, *a, **k):
        pass

    def close(self, *a, **k):
        pass


class Venue:
    """A scriptable order venue (one-way mode). Reproduces the testnet failure modes of the incident."""

    def __init__(self):
        self.pos: dict[str, PositionInfo] = {}
        self.orders: list[ConditionalOrder] = []
        self.calls: list[tuple] = []
        self.reduce_reject: set[str] = set()      # reduce-only market orders rejected with -2022
        self.plain_reject: set[str] = set()       # plain market orders rejected
        self.stops_block_reduce = False           # reduce-only close rejected while a reduce-only stop covers the position
        self.no_stop: set[str] = set()            # place_stop -> -2021 (price already through the level)
        self.fail_positions = False
        self.equity = 100.0
        self.on_reduce_reject = None              # hook called once when a reduce-only close is rejected

    def filters(self, s):
        return SymbolFilters(0.01, 0.001, 0.001, 5)

    def positions(self):
        if self.fail_positions:
            raise NetworkError("down")
        return list(self.pos.values())

    def open_conditional_orders(self, symbol=None):
        return [o for o in self.orders if symbol in (None, o.symbol)]

    def balance(self):
        return Balance(self.equity, self.equity, 0.0)

    def place_stop(self, sym, side, price, qty):
        self.calls.append(("stop", sym, side, price, qty))
        if sym in self.no_stop:
            raise ExchangeError("Order would immediately trigger.", code=-2021)
        o = ConditionalOrder(f"a:{len(self.calls)}", sym, "SL", side, price, qty, True)
        self.orders.append(o)
        return o.order_id

    def cancel_order(self, sym, oid):
        self.calls.append(("cancel", sym, oid))
        self.orders = [o for o in self.orders if o.order_id != oid]

    def cancel_all(self, sym):
        self.calls.append(("cancel_all", sym))
        self.orders = [o for o in self.orders if o.symbol != sym]

    def market_order(self, sym, side, qty, reduce_only=False):
        self.calls.append(("market", sym, side, qty, reduce_only))
        covered = any(o.symbol == sym and o.reduce_only and o.qty >= qty * 0.999 for o in self.orders)
        if reduce_only and (sym in self.reduce_reject or (self.stops_block_reduce and covered)):
            hook, self.on_reduce_reject = self.on_reduce_reject, None
            if hook:
                hook()
            raise ExchangeError("HTTP 400 on POST /fapi/v1/order: code=-2022 ReduceOnly Order is rejected.", code=-2022)
        if not reduce_only and sym in self.plain_reject:
            raise ExchangeError("not filled", code=-9)
        self.pos.pop(sym, None)
        return OrderResult("o:1", sym, side, qty, 1.0, "FILLED")

    # helpers
    def n(self, kind, sym=None, reduce_only=None):
        return [c for c in self.calls if c[0] == kind and (sym is None or c[1] == sym)
                and (reduce_only is None or c[-1] is reduce_only)]


def long_(sym, qty=1.6, mark=100.0, liq=88.0, margin=20.0, entry=100.0):
    return PositionInfo(sym, "long", qty, entry, mark, 0.0, 8, liq, margin)


def short_(sym, qty=1.6, mark=100.0, liq=112.0, margin=20.0, entry=100.0):
    return PositionInfo(sym, "short", qty, entry, mark, 0.0, 8, liq, margin)


def sl_order(sym, side="SELL", price=93.75, qty=1.6, oid="a:9"):
    return ConditionalOrder(oid, sym, "SL", side, price, qty, True)


def db_row(store, sym, side="long", sl=93.75, qty=1.6):
    return store.insert_position({"symbol": sym, "side": side, "entry_price": 100.0, "qty": qty, "notional": 160.0,
                                  "leverage": 8, "margin": 20.0, "sl_price": sl, "tp_price": 103.0 if side == "long" else 97.0,
                                  "init_sl_price": sl, "opened_at": 1, "sl_order_id": "a:9"})


def alive(store, clk):
    store.beat("binance_demo", "2", 1)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))


def cycle(g, store, clk, dt=5.0):
    clk.t += dt
    alive(store, clk)
    return g.step()


# ------------------------------------------------------------------ backoff / liquidation primitives
def test_backoff_doubles_from_5s_to_a_5min_cap_and_resets():
    b = EM.Backoff()
    t = 1000.0
    assert b.ready("k", t)
    delays = [b.fail("k", t) for _ in range(9)]
    assert delays == [5, 10, 20, 40, 80, 160, 300, 300, 300]
    assert not b.ready("k", t + 299) and b.ready("k", t + 300)
    assert b.failures("k") == 9 and b.reset("k") and b.ready("k", t) and b.failures("k") == 0
    assert b.fail("k", t) == 5                                  # restarted from the bottom
    assert not b.reset("never-failed")


def test_liquidation_reason_long_short_margin_and_cross():
    assert EM.liquidation_reason(long_("A", mark=100, liq=88)) is None
    assert EM.liquidation_reason(long_("A", mark=88, liq=88))                      # long: mark <= liq
    assert EM.liquidation_reason(long_("A", mark=87, liq=88))
    assert EM.liquidation_reason(short_("A", mark=112, liq=112))                   # short: mark >= liq
    assert EM.liquidation_reason(short_("A", mark=100, liq=112)) is None
    assert EM.liquidation_reason(long_("A", mark=100, liq=88, margin=-0.5)) == "isolated margin -0.5 <= 0"
    assert EM.liquidation_reason(long_("A", mark=100, liq=88, margin=0.0))
    cross = dataclasses.replace(long_("A", margin=0.0), isolated=False)
    assert EM.liquidation_reason(cross) is None                                    # cross margin reports isolatedMargin 0
    assert EM.liquidation_reason(long_("A", mark=0.0, liq=0.0)) is None            # unknown liq price: not a verdict


# ------------------------------------------------------------------ emergency_close (the order of operations)
def test_emergency_close_cancels_reduce_only_stops_first_then_closes_reduce_only():
    ex = Venue()
    ex.pos["A"] = long_("A")
    ex.orders = [sl_order("A", oid="a:9"), ConditionalOrder("o:5", "A", "TP", "SELL", 110, 1.6, True),
                 ConditionalOrder("a:7", "B", "SL", "SELL", 90, 1.0, True)]
    ex.stops_block_reduce = True                                # the covering stops would make the venue reject the close
    out = EM.emergency_close(ex, "A")
    assert out.ok and out.flat and out.method == "reduce_only"
    kinds = [c[0] for c in ex.calls]
    assert kinds.index("cancel") < kinds.index("market")        # cancelled BEFORE the close
    assert ("cancel", "A", "a:9") in ex.calls and ("cancel", "A", "o:5") in ex.calls   # algo + legacy ids
    assert not [c for c in ex.calls if c[:2] == ("cancel", "B")]                        # only THIS symbol's orders
    assert ex.n("market", "A") == [("market", "A", "SELL", 1.6, True)]


def test_emergency_close_without_cancel_first_cancels_and_retries_on_minus_2022():
    ex = Venue()
    ex.pos["A"] = long_("A")
    ex.orders = [sl_order("A")]
    ex.stops_block_reduce = True
    out = EM.emergency_close(ex, "A", cancel_first=False)
    assert out.ok and out.method == "cancel_retry"
    assert [c[0] for c in ex.calls[:3]] == ["market", "cancel", "market"]


def test_minus_2022_with_position_still_open_falls_back_to_one_plain_market_of_the_exchange_qty():
    ex = Venue()
    ex.pos["A"] = long_("A", qty=1.6)
    ex.reduce_reject.add("A")                                   # cancelling does not help: the venue is jammed

    def shrink():                                               # a partial fill happened meanwhile: re-read, never the old qty
        ex.pos["A"] = long_("A", qty=1.0)
    ex.on_reduce_reject = shrink
    out = EM.emergency_close(ex, "A")
    assert out.ok and out.method == "plain_market"
    plain = ex.n("market", "A", reduce_only=False)
    assert plain == [("market", "A", "SELL", 1.0, False)]       # exactly the exchange-reported quantity, one order


def test_plain_market_fallback_is_attempted_once_then_gives_up():
    ex = Venue()
    ex.pos["A"] = long_("A")
    ex.reduce_reject.add("A")
    ex.plain_reject.add("A")
    out = EM.emergency_close(ex, "A")
    assert not out.ok and not out.liquidation and "plain market close failed" in out.detail
    assert len(ex.n("market", "A", reduce_only=False)) == 1
    assert "A" in ex.pos                                        # still open: the caller backs off


def test_no_plain_market_when_the_position_went_into_liquidation_or_vanished_meanwhile():
    ex = Venue()
    ex.pos["A"] = long_("A")
    ex.reduce_reject.add("A")
    ex.on_reduce_reject = lambda: ex.pos.__setitem__("A", long_("A", mark=85.0, liq=88.0))
    out = EM.emergency_close(ex, "A")
    assert out.liquidation and not out.ok and not ex.n("market", "A", reduce_only=False)
    ex2 = Venue()
    ex2.pos["A"] = long_("A")
    ex2.reduce_reject.add("A")
    ex2.on_reduce_reject = lambda: ex2.pos.pop("A")
    out2 = EM.emergency_close(ex2, "A")
    assert out2.ok and out2.flat and not ex2.n("market", "A", reduce_only=False)


def test_emergency_close_sends_nothing_for_a_liquidating_or_flat_symbol():
    ex = Venue()
    ex.pos["L"] = long_("L", mark=80.0, liq=88.0)
    out = EM.emergency_close(ex, "L")
    assert out.liquidation and not ex.calls
    assert EM.emergency_close(ex, "NOPE").flat and not ex.calls


def test_emergency_close_unfilled_order_is_not_a_close():
    ex = Venue()
    ex.pos["A"] = long_("A")
    orig = ex.market_order
    ex.market_order = lambda sym, side, qty, reduce_only=False: (ex.calls.append(("market", sym, side, qty, reduce_only)),
                                                                 OrderResult("o:1", sym, side, qty, 0.0, "NEW"))[1]
    out = EM.emergency_close(ex, "A")
    assert not out.ok and "still open" in out.detail           # accepted but never filled -> the position is still there


# ------------------------------------------------------------------ guard: per-symbol, not account-wide
def test_one_jammed_coin_degrades_but_never_flattens_the_account_and_backs_off(store):
    clk = Clock()
    ex = Venue()
    alerts = Rec()
    for s in ("AAAUSDT", "BBBUSDT"):
        ex.pos[s] = long_(s)
        db_row(store, s)
    ex.orders = [sl_order("AAAUSDT")]                           # AAA is healthy, BBB's trailed stop triggered and vanished
    ex.pos["BBBUSDT"] = long_("BBBUSDT", mark=92.0)             # mark already through the stop level (93.75)
    ex.reduce_reject.add("BBBUSDT")
    ex.plain_reject.add("BBBUSDT")
    g = G.Guard(store, "binance_demo", ex, alerts, clock=clk)
    states, t0 = [], clk.t
    for _ in range(121):                                        # 10 minutes of 5 s guard cycles
        states.append(cycle(g, store, clk)["state"])
    assert FLATTEN not in states and HALTED not in states
    assert states[-1] == DEGRADED                               # entries blocked while the problem persists ...
    assert not [c for c in ex.calls if c[1] == "AAAUSDT"]       # ... and the healthy coin was never touched
    attempts = len(ex.n("market", "BBBUSDT", reduce_only=True))
    assert 6 <= attempts <= 8, attempts                         # 5 s, 10 s, 20 s ... 5 min: ~7 tries in 10 min, not 120
    assert alerts.sym == [("BBBUSDT", "emergency close")]       # ONE alert per symbol per incident
    msgs = [r["msg"] for r in store.query("SELECT msg FROM logs WHERE msg LIKE '%emergency close on BBBUSDT%'")]
    assert len(msgs) == attempts and any("next try in 300s" in m for m in msgs)    # one log line per backoff step
    assert "stop missing on BBBUSDT" in " ".join(store.kv_get("safety_guard")["reasons"])
    # the venue recovers: the next due attempt closes it, state returns to NORMAL (hysteresis)
    ex.reduce_reject.clear()
    for _ in range(80):
        r = cycle(g, store, clk)
        if "BBBUSDT" not in ex.pos:
            break
    assert "BBBUSDT" not in ex.pos
    store.delete_position(2)
    for _ in range(8):
        r = cycle(g, store, clk)
    assert r["state"] == NORMAL


def test_unplaceable_stop_closes_only_that_coin(store):
    clk = Clock()
    ex = Venue()
    for s in ("AAAUSDT", "BBBUSDT"):
        ex.pos[s] = long_(s)
        db_row(store, s)
    ex.orders = [sl_order("AAAUSDT")]
    ex.no_stop.add("BBBUSDT")                                   # -2021: price already through the level
    g = G.Guard(store, "binance_demo", ex, Rec(), clock=clk)
    cycle(g, store, clk)
    r = cycle(g, store, clk)                                    # (second sighting: the engine may be mid-trail)
    assert "BBBUSDT" not in ex.pos and "AAAUSDT" in ex.pos
    assert r["state"] == DEGRADED
    assert ex.n("market", "BBBUSDT") and not ex.n("market", "AAAUSDT")


def test_stop_beyond_mark_is_not_even_sent_the_coin_is_closed(store):
    clk = Clock()
    ex = Venue()
    ex.pos["BBBUSDT"] = long_("BBBUSDT", mark=90.0)
    db_row(store, "BBBUSDT", sl=93.75)
    g = G.Guard(store, "binance_demo", ex, Rec(), clock=clk)
    cycle(g, store, clk)
    cycle(g, store, clk)
    assert not ex.n("stop") and "BBBUSDT" not in ex.pos


def test_transient_network_error_placing_a_stop_does_not_close_the_coin(store):
    clk = Clock()
    ex = Venue()
    ex.pos["A"] = long_("A")
    db_row(store, "A")
    ex.place_stop = lambda *a: (_ for _ in ()).throw(NetworkError("timeout"))
    g = G.Guard(store, "binance_demo", ex, Rec(), clock=clk)
    cycle(g, store, clk)
    r = cycle(g, store, clk)
    assert r["state"] == DEGRADED and "A" in ex.pos and not ex.n("market")


def test_liquidating_position_gets_no_orders_one_alert_and_is_rechecked(store):
    clk = Clock()
    ex = Venue()
    alerts = Rec()
    ex.pos["LONGUSDT"] = long_("LONGUSDT", mark=87.0, liq=88.0)                 # long: mark <= liq
    ex.pos["SHORTUSDT"] = short_("SHORTUSDT", mark=113.0, liq=112.0)            # short: mark >= liq
    ex.pos["MARGINUSDT"] = long_("MARGINUSDT", mark=100.0, liq=88.0, margin=-3.0)    # isolatedMargin went negative
    for s in ("LONGUSDT", "SHORTUSDT", "MARGINUSDT"):
        db_row(store, s, side="short" if s.startswith("SHORT") else "long")
    g = G.Guard(store, "binance_demo", ex, alerts, clock=clk)
    for _ in range(30):
        r = cycle(g, store, clk)
    assert not ex.calls                                                         # not one order for any of them
    assert sorted(alerts.liq) == ["LONGUSDT", "MARGINUSDT", "SHORTUSDT"]        # alerted ONCE each
    assert r["state"] == NORMAL                                                 # nothing to protect: not a reason to flatten
    assert sorted(store.kv_get("safety_guard")["liquidating"]) == ["LONGUSDT", "MARGINUSDT", "SHORTUSDT"]
    assert sorted(EM.liq_map(store)) == ["LONGUSDT", "MARGINUSDT", "SHORTUSDT"]
    # mark recovers: it is no longer in liquidation and is managed normally again (stop is re-placed)
    ex.pos["LONGUSDT"] = long_("LONGUSDT", mark=100.0, liq=88.0)
    for _ in range(3):
        cycle(g, store, clk)
    assert "LONGUSDT" not in EM.liq_map(store) and ex.n("stop", "LONGUSDT")
    # the venue finally closes the others and the engine has booked them: the guard drops its marks
    for s in ("SHORTUSDT", "MARGINUSDT"):
        ex.pos.pop(s)
        store.delete_position(next(p["id"] for p in store.get_positions() if p["symbol"] == s))
    cycle(g, store, clk)
    assert sorted(EM.liq_map(store)) == []


def test_account_flatten_closes_what_it_can_skips_liquidation_and_halts_once_flat(store):
    clk = Clock()
    ex = Venue()
    for s in ("AAAUSDT", "BBBUSDT", "LIQUSDT"):
        ex.pos[s] = long_(s)
        db_row(store, s)
    ex.orders = [sl_order(s) for s in ("AAAUSDT", "BBBUSDT")]
    ex.pos["LIQUSDT"] = long_("LIQUSDT", mark=80.0, liq=88.0)
    ex.reduce_reject.add("BBBUSDT")
    ex.plain_reject.add("BBBUSDT")
    store.kv_set("peak_equity", 100.0)
    store.kv_set("capital_floor_trade_id", 0)
    alerts = Rec()
    g = G.Guard(store, "binance_demo", ex, alerts, clock=clk)
    cycle(g, store, clk)
    ex.equity = 74.0                                                            # -26%: drawdown kill -> account-wide flatten
    states = [cycle(g, store, clk)["state"] for _ in range(20)]
    assert set(states) == {FLATTEN}                                             # BBB is still open: stays FLATTEN, no HALTED yet
    assert "AAAUSDT" not in ex.pos and "BBBUSDT" in ex.pos                      # AAA closed despite BBB failing
    assert not ex.n("market", "LIQUSDT") and not ex.n("cancel", "LIQUSDT")      # liquidating coin untouched
    assert len(ex.n("market", "BBBUSDT", reduce_only=True)) <= 5                # backed off: 0,5,15,35,75 s in 100 s
    assert alerts.sym == [("BBBUSDT", "flatten")]
    ex.reduce_reject.clear()
    for _ in range(40):
        r = cycle(g, store, clk)
        if r["state"] == HALTED:
            break
    assert r["state"] == HALTED and sorted(ex.pos) == ["LIQUSDT"]               # flat except the venue-owned liquidation


def test_exchange_wide_failure_still_degrades_then_flattens_and_logs_once_per_step(store):
    clk = Clock()
    ex = Venue()
    ex.pos["A"] = long_("A")
    db_row(store, "A")
    ex.orders = [sl_order("A")]
    g = G.Guard(store, "binance_demo", ex, Rec(), clock=clk, flatten_after_s=60)
    cycle(g, store, clk)
    ex.fail_positions = True
    states = [cycle(g, store, clk)["state"] for _ in range(30)]                 # 150 s of outage
    assert states[0] == NORMAL and DEGRADED in states and states[-1] == FLATTEN
    warn = store.query("SELECT msg FROM logs WHERE msg LIKE '%exchange check failed%'")
    assert 0 < len(warn) <= 7                                                   # backoff steps, not 30 lines


def test_paper_guard_unchanged_by_the_liquidation_machinery(store):
    clk = Clock()
    g = G.Guard(store, "paper_local", None, clock=clk)
    store.kv_set("status", {"equity": 100.0, "kill": {"tripped": False}})
    assert g.step()["state"] == NORMAL


# ------------------------------------------------------------------ engine
def _venue_pos(venue, sym, **kw):
    venue.pos[sym] = dataclasses.replace(venue.pos[sym], **kw)


def test_engine_liquidation_no_orders_one_alert_then_recorded_as_liquidation(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0})
    open_(eng, "AAAUSDT")
    p = store.get_positions()[0]
    n_before = len(venue.log)
    _venue_pos(venue, "AAAUSDT", mark_price=80.0, liq_price=85.0)               # marked through the liquidation price
    venue.marks["AAAUSDT"] = 80.0                                               # (and through the stop: would normally close)
    for _ in range(6):
        eng._last_sync = eng._last_stop_check = 0.0
        run(eng.prices_cycle())
    assert eng.liquidating == {"AAAUSDT"}
    assert len(venue.log) == n_before                                           # no close, no stop, no cancel: nothing sent
    assert [e for e in store.get_positions()] and "AAAUSDT" in EM.liq_map(store)
    eng.alerts.flush(2)
    sent = eng.alerts.notifiers[0].msgs
    assert len([m for m in sent if "LIQUIDATION" in m]) == 1                    # alerted once, not per cycle
    # the venue liquidates it: gone from the exchange
    venue.pos.pop("AAAUSDT")
    venue.orders.clear()
    eng._last_sync = 0.0
    run(eng.prices_cycle())
    assert store.get_positions() == []
    t = store.get_trades(1)[0]
    assert t["exit_reason"] == "liquidation" and t["pnl"] < 0
    assert EM.liq_map(store) == {} and eng.liquidating == set()


def test_engine_close_loop_on_a_jammed_venue_backs_off(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0})
    open_(eng, "AAAUSDT")
    p = store.get_positions()[0]
    t = [time.time()]
    eng.clock = lambda: t[0]
    jam = {"n": 0}

    def jammed(symbol, side, qty, reduce_only=False):
        jam["n"] += 1
        venue.log.append(("market", symbol, side, qty, reduce_only, 0))
        raise ExchangeError("HTTP 400 code=-2022 ReduceOnly Order is rejected.", code=-2022)
    venue.market_order = jammed
    venue.orders.clear()                                                        # the trailed stop triggered but did not fill
    venue.marks["AAAUSDT"] = p["sl_price"] * 0.98                               # beyond the stop by > 0.3%
    for _ in range(25):                                                         # 2 minutes of 5 s loops
        t[0] += 5
        eng._last_sync = eng._last_stop_check = 0.0
        run(eng.prices_cycle())
    assert len(store.get_positions()) == 1                                      # still open, nothing booked
    assert 4 <= jam["n"] <= 14, jam["n"]                                        # backed off (was ~25+ per 2 min before)
    eng.alerts.flush(2)
    assert len([m for m in eng.alerts.notifiers[0].msgs if "AAAUSDT" in m and "close" in m.lower()]) == 1


def test_engine_invariant_is_per_coin_replaces_stop_or_closes_only_that_coin(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0})
    open_(eng, "AAAUSDT")
    open_(eng, "BBBUSDT")
    a, b = [store.get_positions()[i] for i in (0, 1)]
    venue.orders = [o for o in venue.orders if o.symbol == "BBBUSDT" and o.kind == "SL"]      # AAA lost its stop
    eng._last_sync = eng._last_stop_check = 0.0
    run(eng.prices_cycle())
    assert {p["symbol"] for p in store.get_positions()} == {"AAAUSDT", "BBBUSDT"}
    assert [o for o in venue.orders if o.symbol == "AAAUSDT" and o.kind == "SL"]               # re-placed, nothing closed
    # BBB loses its stop AND the price is through the level: only BBB is closed
    venue.orders = [o for o in venue.orders if o.symbol == "AAAUSDT"]
    venue.marks["BBBUSDT"] = b["sl_price"] * 0.999
    eng._last_sync = eng._last_stop_check = 0.0
    run(eng.prices_cycle())
    assert [p["symbol"] for p in store.get_positions()] == ["AAAUSDT"]
    assert "AAAUSDT" in venue.pos and "BBBUSDT" not in venue.pos


def test_engine_unprotected_fill_that_cannot_be_closed_does_not_trip_the_kill_switch(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0})
    real = venue.market_order

    def jammed(symbol, side, qty, reduce_only=False):
        if reduce_only:
            raise ExchangeError("HTTP 400 code=-2022 ReduceOnly Order is rejected.", code=-2022)
        return real(symbol, side, qty, reduce_only)
    venue.market_order = jammed
    venue.next_mark = 90.0                                                      # stop can not be placed -> flatten the fill
    open_(eng, "AAAUSDT")
    assert not eng.kill.get("tripped")                                          # one coin never takes the account down


# ------------------------------------------------------------------ universe whitelist
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]


def test_whitelist_universe_is_whitelist_intersect_venue_ignoring_size_and_ranking():
    prod = FakeData(tuple(MAJORS) + ("JUNKUSDT",))
    prod.exchange_info = lambda: {s: SymbolInfo(s, onboard_date_ms=0) for s in prod.marks}
    cfg = Settings().merge({"universe_size": 2, "universe_whitelist": MAJORS})
    venue = {s: SymbolInfo(s) for s in MAJORS + ["JUNKUSDT"]}
    venue["DOGEUSDT"].status = "BREAK"
    del venue["XRPUSDT"]
    dropped: list[str] = []
    rows = uni.refresh(prod, cfg, {"SOLUSDT": 42}, venue_info=venue, dropped=dropped)
    assert [r["symbol"] for r in rows] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]     # size 2 ignored; JUNK not added
    assert sorted(dropped) == ["DOGEUSDT", "XRPUSDT"]
    assert [r["rank"] for r in rows] == [1, 2, 3, 4]
    assert {r["symbol"]: r["in_universe_since"] for r in rows}["SOLUSDT"] == 42
    assert all(r["atr_pct"] and r["last_price"] for r in rows)                             # prod-data signals kept
    none = uni.refresh(prod, Settings().merge({"universe_whitelist": ["NOPEUSDT"]}), {})
    assert none == []
    default = uni.refresh(prod, Settings().merge({"universe_size": 2}), {})                 # null = the ranking, as before
    assert len(default) == 2


def test_engine_whitelist_change_rebuilds_the_universe_and_open_positions_stay_managed(store, tmp_path):
    prod = {"AAAUSDT": 100.0, "BBBUSDT": 100.0, "CCCUSDT": 100.0, "BTCUSDT": 100.0}
    eng, data, venue = mk_demo(store, tmp_path, prod_marks=prod, venue_marks=dict(prod))
    data.exchange_info = lambda: {s: SymbolInfo(s, onboard_date_ms=0) for s in data.marks}
    open_(eng, "CCCUSDT")                                                       # a coin that will fall out of the universe
    eng.universe_at = eng.clock()                                               # a fresh universe: only the whitelist forces a rebuild
    eng.cfg = eng.cfg.merge({"universe_whitelist": ["BTCUSDT", "AAAUSDT"]})
    run(eng.universe_cycle())
    assert eng.universe == ["BTCUSDT", "AAAUSDT"]
    assert store.kv_get("universe_wl") == ["BTCUSDT", "AAAUSDT"]
    assert [r["symbol"] for r in store.get_universe()] == ["BTCUSDT", "AAAUSDT"]
    # CCC is no longer in the universe but its trade runs to its exits: watched, priced, protected, closable
    assert "CCCUSDT" in eng.watchlist() and "CCCUSDT" in eng.trade_symbols()
    venue.marks["CCCUSDT"] = 101.0
    eng._last_sync = eng._last_stop_check = 0.0
    run(eng.prices_cycle())
    p = store.get_positions()[0]
    assert p["symbol"] == "CCCUSDT" and p["mark_price"] == pytest.approx(101.0)          # still priced and managed
    assert [o for o in venue.orders if o.symbol == "CCCUSDT" and o.kind == "SL"]         # and still protected
    # unchanged whitelist + fresh universe: no rebuild
    eng.universe = ["X"]
    run(eng.universe_cycle())
    assert eng.universe == ["X"]
    eng.cfg = eng.cfg.merge({"universe_whitelist": None})                       # cleared: back to the ranking
    run(eng.universe_cycle())
    assert eng.universe != ["X"] and store.kv_get("universe_wl") == []


# ------------------------------------------------------------------ the venue that acks market orders but never fills them
def test_binance_market_order_accepted_but_not_filled_is_an_error_and_is_cancelled(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ex, s = mk(Resp(200, {"orderId": 5, "status": "NEW", "executedQty": "0"}),            # POST: acked
               Resp(200, {"orderId": 5, "status": "NEW", "executedQty": "0"}),            # re-read: still unfilled
               Resp(200, {"orderId": 5, "status": "CANCELED"}))                           # our cancel
    with pytest.raises(OrderNotFilled):
        ex.market_order("ETHUSDT", "SELL", 0.067, reduce_only=True)
    assert [c["method"] for c in s.calls] == ["POST", "GET", "DELETE"]


def test_binance_partial_fill_is_reported_with_the_executed_quantity(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ex, s = mk(Resp(200, {"orderId": 5, "status": "PARTIALLY_FILLED", "executedQty": "0.02", "avgPrice": "10"}))
    r = ex.market_order("ETHUSDT", "SELL", 0.067)
    assert r.qty == pytest.approx(0.02) and r.status == "PARTIALLY_FILLED"
