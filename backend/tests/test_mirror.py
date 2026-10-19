"""Testnet mirror (tb/mirror.py) against a fake venue: accepted / rejected (every class) / timeouts / partial fills /
orphan reconcile / missing stop / account_unusable gating / paper book identical with the mirror on and off."""
import asyncio
import time

import pytest

from conftest import FakeData
from test_engine_core import FAST, Sent, closed_of, mk_engine, open_long, run
from tb import mirror as M
from tb.alerts import AlertDispatcher
from tb.config import Settings
from tb.exchange.base import (ConditionalOrder, ExchangeError, NetworkError, OrderNotFilled, OrderResult, PositionInfo,
                              Premium, SymbolFilters, SymbolInfo)
from tb.store import Store


# ------------------------------------------------------------------ fakes
class FakeVenue:
    """Duck-typed testnet. ``fail[method]`` = exceptions raised by the next calls of that method (FIFO)."""

    def __init__(self, marks=None, status=None):
        self.marks = dict(marks or {"AAAUSDT": 100.0, "BBBUSDT": 50.0})
        self.status = status or {}
        self.pos: dict[str, PositionInfo] = {}
        self.conds: list[ConditionalOrder] = []
        self.calls: list[tuple] = []
        self.fail: dict[str, list] = {}
        self.delay: dict[str, float] = {}
        self.fill_frac = 1.0
        self.limit_fill = 0.0                     # fraction of a resting limit order that fills at once
        self.limits: dict[str, OrderResult] = {}
        self.n = 0
        self.account = {"totalMarginBalance": "1000", "availableBalance": "900", "totalWalletBalance": "1000", "canTrade": True}
        self.min_notional = 5.0

    # -- plumbing
    def _hit(self, name, *args):
        self.calls.append((name, *args))
        d = self.delay.get(name)
        if d:
            time.sleep(d)
        q = self.fail.get(name)
        if q:
            raise q.pop(0)

    def _id(self):
        self.n += 1
        return self.n

    def n_calls(self, name):
        return sum(1 for c in self.calls if c[0] == name)

    # -- reads
    def filters(self, sym):
        return SymbolFilters(0.01, 0.001, 0.001, self.min_notional)

    def exchange_info(self):
        self._hit("exchange_info")
        return {s: SymbolInfo(s, self.status.get(s, "TRADING"), filters=self.filters(s)) for s in self.marks}

    def premium(self, symbol=None):
        self._hit("premium", symbol)
        return [Premium(s, m) for s, m in self.marks.items() if symbol in (None, s)]

    def positions(self):
        self._hit("positions")
        return [PositionInfo(p.symbol, p.side, p.qty, p.entry_price, self.marks.get(p.symbol, p.entry_price), 0.0, 8, 0.0, 10.0)
                for p in self.pos.values()]

    def account_info(self):
        self._hit("account_info")
        return dict(self.account)

    def best_bid_ask(self, symbol):
        self._hit("best_bid_ask", symbol)
        m = self.marks[symbol]
        return m - 0.01, m + 0.01

    def open_conditional_orders(self, symbol=None):
        self._hit("open_conditional_orders", symbol)
        return [o for o in self.conds if symbol in (None, o.symbol)]

    # -- orders
    def set_isolated(self, sym):
        self._hit("set_isolated", sym)

    def set_leverage(self, sym, lev):
        self._hit("set_leverage", sym, lev)

    def market_order(self, sym, side, qty):
        self._hit("market_order", sym, side, qty)
        px = self.marks[sym]
        cur = self.pos.get(sym)
        if cur is not None and (cur.side == "long") == (side == "SELL"):      # a normal opposite order reduces / closes
            q = min(qty, cur.qty)
            cur.qty = round(cur.qty - q, 8)
            if cur.qty <= 0:
                self.pos.pop(sym)
            return OrderResult(f"o:{self._id()}", sym, side, q, px, "FILLED")
        q = round(qty * self.fill_frac, 8)
        self.pos[sym] = PositionInfo(sym, "long" if side == "BUY" else "short", q, px)
        return OrderResult(f"o:{self._id()}", sym, side, q, px, "FILLED" if self.fill_frac >= 1 else "PARTIALLY_FILLED")

    def limit_order(self, sym, side, qty, price, post_only=True, reduce_only=False):
        self._hit("limit_order", sym, side, qty, price)
        ex = round(qty * self.limit_fill, 8)
        r = OrderResult(f"o:{self._id()}", sym, side, qty, price if ex else 0.0, "FILLED" if self.limit_fill >= 1 else
                        ("PARTIALLY_FILLED" if ex else "NEW"), 0.0, ex, price)
        self.limits[r.order_id] = r
        if ex:
            self.pos[sym] = PositionInfo(sym, "long" if side == "BUY" else "short", ex, price)
        return r

    def get_order(self, sym, oid):
        self._hit("get_order", sym, oid)
        return self.limits[oid]

    def _cond(self, kind, sym, side, price, qty):
        self.calls.append((kind, sym, side, price, qty))
        q = self.fail.get(kind)
        if q:
            raise q.pop(0)
        o = ConditionalOrder(f"a:{self._id()}", sym, kind, side, price, qty, True)
        self.conds.append(o)
        return o.order_id

    def place_stop(self, sym, side, price, qty=None):
        return self._cond("SL", sym, side, price, 0.0)

    def place_take_profit(self, sym, side, price, qty=None):
        return self._cond("TP", sym, side, price, 0.0)

    def cancel_order(self, sym, oid):
        self._hit("cancel_order", sym, oid)
        self.conds = [o for o in self.conds if o.order_id != oid]

    def cancel_all(self, sym):
        self._hit("cancel_all", sym)
        self.conds = [o for o in self.conds if o.symbol != sym]


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(tmp_path, venue=None, settings=None, digest_s=0.0, **kw):
    store = Store(tmp_path / "tb.sqlite")
    cfg = Settings().merge({"mirror_enabled": True, **(settings or {})})
    sink = Sent()
    alerts = AlertDispatcher([sink], min_interval=0, digest_s=digest_s)
    clock = Clock()
    v = venue or FakeVenue()
    m = M.Mirror(store, alerts, lambda: cfg, clock=clock, venue=v, sleep=lambda s: asyncio.sleep(0), **kw)
    m._next_recon = m._next_poll = 1e18               # tests drive the account poll / reconciler explicitly
    m.sink, m.clock_, m.venue_, m.cfg_ = sink, clock, v, cfg
    return m, store, v


def paper(store, sym="AAAUSDT", side="long", entry=100.0, qty=1.6, sl=99.0, tp=101.0, lev=8):
    pid = store.insert_position({"symbol": sym, "side": side, "entry_price": entry, "qty": qty, "notional": qty * entry,
                                 "leverage": lev, "margin": qty * entry / lev, "sl_price": sl, "tp_price": tp,
                                 "init_sl_price": sl, "opened_at": 1_000_000, "sl_order_id": "p1", "tp_order_id": "p2"})
    return pid


def open_mirror(m, store, pid, via_maker=False):
    m.on_open(pid, store.get_position(pid), via_maker=via_maker)
    asyncio.run(m.drain())


def events(store, **where):
    q = "SELECT * FROM mirror_events"
    if where:
        q += " WHERE " + " AND ".join(f"{k}=?" for k in where)
    return [dict(r) for r in store.query(q + " ORDER BY id", tuple(where.values()))]


def actions(store):
    return [(e["action"], e["cls"]) for e in events(store)]


def flush(m):
    m.alerts.flush(2)


# ------------------------------------------------------------------ sizing
def test_min_order_qty_respects_step_and_notional():
    f = SymbolFilters(0.01, 0.001, 0.001, 5.0)
    q = M.min_order_qty(f, 100.0)
    assert q == pytest.approx(0.055) and q * 100 >= 5.5          # 5 * 1.10 / 100 = 0.055 -> next lot step
    assert M.min_order_qty(SymbolFilters(0.1, 0.001, 0.001, 100.0), 60000.0) == pytest.approx(0.002)   # BTC-like: 110 / 60000
    assert M.min_order_qty(SymbolFilters(1e-5, 1.0, 1.0, 5.0), 0.01) == 550.0                           # step 1, price 1 cent
    assert M.min_order_qty(SymbolFilters(0.01, 0.001, 0.5, 5.0), 100000.0) == 0.5                       # min_qty dominates
    assert M.min_order_qty(SymbolFilters(0.01, 0.001, 0.001, 5.0, max_qty=0.01), 100.0) is None
    assert M.min_order_qty(f, 0.0) is None


# ------------------------------------------------------------------ classification table
@pytest.mark.parametrize("exc,cls,code", [
    (ExchangeError("precision", code=-1111), M.OUR_BUG, "-1111"),
    (ExchangeError("filter", code=-1013), M.OUR_BUG, "-1013"),
    (ExchangeError("notional", code=-4164), M.OUR_BUG, "-4164"),
    (ExchangeError("param", code=-1102), M.OUR_BUG, "-1102"),
    (ExchangeError("param", code=-1106), M.OUR_BUG, "-1106"),
    (ExchangeError("unknown order", code=-2011), M.OUR_BUG, "-2011"),
    (ExchangeError("would trigger", code=-2021), M.OUR_BUG, "-2021"),
    (ExchangeError("some new 4xx code", code=-9999, status=400), M.OUR_BUG, "-9999"),
    (ExchangeError("qty 0.0001 below min_qty 0.001 for X"), M.OUR_BUG, "min_qty"),
    (ExchangeError("reduce", code=-2022), M.OUR_BUG, "-2022"),                       # account usable: our state was wrong
    (NetworkError("HTTP 503", status=503), M.VENUE_FAULT, "http_503"),
    (NetworkError("boom"), M.VENUE_FAULT, "network"),
    (M.VenueTimeout("slow"), M.VENUE_FAULT, "timeout"),
    (ExchangeError("internal", code=-1001), M.VENUE_FAULT, "-1001"),
    (ExchangeError("HTTP 500", code=None, status=500), M.VENUE_FAULT, "http_500"),
    (OrderNotFilled("accepted, never filled"), M.VENUE_FAULT, "not_filled"),
    (ExchangeError("margin", code=-2019), M.ACCOUNT_UNUSABLE, "-2019"),
    (ExchangeError("balance", code=-2018), M.ACCOUNT_UNUSABLE, "-2018"),
    (ExchangeError("key", code=-2015), M.ACCOUNT_UNUSABLE, "-2015"),
    (ExchangeError("post only", code=-5022), M.OK, "-5022"),                          # expected answer of the maker flow
    (ValueError("bug in our code"), M.OUR_BUG, "ValueError"),
    (None, M.OK, None),
])
def test_classification_table(exc, cls, code):
    assert M.classify(exc) == (cls, code)


def test_reduce_rejected_is_a_venue_fault_while_the_account_is_unusable():
    e = ExchangeError("ReduceOnly Order is rejected", code=-2022)
    assert M.classify(e, account_unusable=True) == (M.VENUE_FAULT, "-2022")
    assert M.classify(e, account_unusable=False) == (M.OUR_BUG, "-2022")
    assert set(M.CLASS_BY_CODE.values()) <= {M.OK, M.OUR_BUG, M.VENUE_FAULT, M.ACCOUNT_UNUSABLE}


# ------------------------------------------------------------------ accepted flow
def test_accepted_entry_places_min_size_with_stop_and_tp(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    assert [c[0] for c in v.calls if c[0] in ("set_isolated", "set_leverage", "market_order", "SL", "TP")] == \
        ["set_isolated", "set_leverage", "market_order", "SL", "TP"]
    mo = next(c for c in v.calls if c[0] == "market_order")
    assert mo[1:] == ("AAAUSDT", "BUY", 0.055)                                         # min size, NOT the paper's 1.6
    sl = next(c for c in v.calls if c[0] == "SL")
    tp = next(c for c in v.calls if c[0] == "TP")
    assert sl[2:] == ("SELL", 99.0, 0.0) and tp[2:] == ("SELL", 101.0, 0.0)             # closePosition (no qty), same-% levels
    ev = events(store)
    assert [e["action"] for e in ev] == ["set_isolated", "set_leverage", "entry", "stop", "take_profit"]
    assert all(e["cls"] == "ok" and e["ok"] == 1 and e["latency_ms"] is not None for e in ev)
    assert next(e for e in ev if e["action"] == "entry")["venue_order_id"].startswith("o:")
    assert next(e for e in ev if e["action"] == "stop")["paper_position_id"] == pid
    mp = store.kv_get("mirror_map")[str(pid)]
    assert mp["state"] == "open" and mp["qty"] == 0.055 and mp["sl_id"] and mp["tp_id"]
    assert M.summary(store, m.cfg_, now=m.clock_())["success_rate_1h"] == 1.0


def test_levels_are_scaled_to_the_venue_price(tmp_path):
    m, store, v = make(tmp_path, FakeVenue({"AAAUSDT": 200.0}))         # testnet trades at 2x the paper price
    open_mirror(m, store, paper(store))
    sl = next(c for c in v.calls if c[0] == "SL")
    assert sl[3] == pytest.approx(198.0)                                 # -1 % from the venue entry, like the paper stop


def test_short_uses_sell_entry_and_buy_stop(tmp_path):
    m, store, v = make(tmp_path)
    open_mirror(m, store, paper(store, side="short", sl=101.0, tp=99.0))
    assert next(c for c in v.calls if c[0] == "market_order")[2] == "SELL"
    assert next(c for c in v.calls if c[0] == "SL")[2] == "BUY"


def test_full_close_normal_order_exact_qty_and_cancels_orders(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    store.delete_position(pid)
    m.on_close(pid, "AAAUSDT", "tp")
    asyncio.run(m.drain())
    mo = [c for c in v.calls if c[0] == "market_order"][-1]
    assert mo[1:] == ("AAAUSDT", "SELL", 0.055)                                        # a normal order, exact qty
    assert v.pos == {} and v.conds == [] and store.kv_get("mirror_map") == {}
    assert actions(store)[-2:] == [("close", "ok"), ("cancel_orders", "ok")] or ("close", "ok") in actions(store)


def test_close_when_the_venue_already_flat_is_clean(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    v.pos.clear()                                                        # venue stop filled first
    store.delete_position(pid)
    m.on_close(pid, "AAAUSDT", "sl")
    asyncio.run(m.drain())
    assert v.n_calls("market_order") == 1                               # only the entry: no close order was sent
    assert events(store)[-1]["cls"] in ("ok",) and store.kv_get("mirror_map") == {}


def test_close_after_a_venue_fault_is_retried_until_flat(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    v.fail["market_order"] = [NetworkError("HTTP 503", status=503)]
    store.delete_position(pid)
    m.on_close(pid, "AAAUSDT", "kill", emergency=True)
    asyncio.run(m.drain())
    assert v.pos and any(e["cls"] == "venue_fault" for e in events(store))   # first try failed, symbol in backoff
    m.clock_.t += 60                                                      # backoff over: the deferred close retries
    asyncio.run(m.drain())
    assert v.pos == {} and v.conds == [] and store.kv_get("mirror_map") == {}


def test_stop_move_cancels_old_then_places_new(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    old = store.kv_get("mirror_map")[str(pid)]["sl_id"]
    v.calls.clear()
    m.on_stop_move(pid, "AAAUSDT", 100.2)
    asyncio.run(m.drain())
    names = [c[0] for c in v.calls if c[0] in ("SL", "cancel_order")]
    assert names == ["cancel_order", "SL"]                               # one closePosition stop per side: cancel, then place
    new = store.kv_get("mirror_map")[str(pid)]["sl_id"]
    assert new != old and {o.order_id for o in v.conds if o.kind == "SL"} == {new}
    assert next(c for c in v.calls if c[0] == "SL")[3] == pytest.approx(100.2)


def test_cancel_of_an_untracked_stop_is_an_our_bug_unknown_order(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    v.conds = [o for o in v.conds if o.kind != "SL"]                     # the stop we track vanished from the venue
    m.on_stop_move(pid, "AAAUSDT", 100.2)
    asyncio.run(m.drain())
    bad = [e for e in events(store) if e["cls"] == "our_bug"]
    assert bad and bad[0]["error_code"] == "-2011" and bad[0]["action"] == "cancel_stop"
    flush(m)
    assert any("OUR BUG" in x and "-2011" in x for x in m.sink.msgs)


def test_partial_close_splits_the_mirror_qty_and_resizes_orders(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)                                           # 0.055 = 56 steps
    v.calls.clear()
    m.on_partial(pid, "AAAUSDT", 0.5, 99.9, 101.0)
    asyncio.run(m.drain())
    mo = next(c for c in v.calls if c[0] == "market_order")
    assert mo[1:] == ("AAAUSDT", "SELL", 0.027)                          # floor(0.055 * 0.5) to the lot step
    mp = store.kv_get("mirror_map")[str(pid)]
    assert mp["qty"] == pytest.approx(0.028)
    assert len(v.conds) == 2                                              # the closePosition TP / SL cover the remainder
    assert next(c for c in v.calls if c[0] == "SL")[3] == pytest.approx(99.9)


def test_partial_close_too_small_is_skipped(tmp_path):
    m, store, v = make(tmp_path, FakeVenue({"AAAUSDT": 100000.0}))       # min size = 1 step: cannot be split
    v.min_notional = 100.0
    pid = paper(store, entry=100000.0, qty=0.01, sl=99000.0, tp=101000.0)
    open_mirror(m, store, pid)
    assert store.kv_get("mirror_map")[str(pid)]["qty"] == 0.002
    m.on_partial(pid, "AAAUSDT", 0.9, 99500.0, 101000.0)                 # 0.9 * 0.002 -> 0.001 closes, 0.001 left: fine
    asyncio.run(m.drain())
    m.on_partial(pid, "AAAUSDT", 0.5, 99500.0, 101000.0)                 # 0.5 * 0.001 -> 0 steps: skipped
    asyncio.run(m.drain())
    assert ("partial_close", "skipped") in actions(store)


# ------------------------------------------------------------------ rejects: each class
@pytest.mark.parametrize("exc,cls", [
    (ExchangeError("Precision is over the maximum", code=-1111, status=400), "our_bug"),
    (ExchangeError("Filter failure: LOT_SIZE", code=-1013, status=400), "our_bug"),
    (ExchangeError("param missing", code=-1102, status=400), "our_bug"),
    (NetworkError("HTTP 503", status=503), "venue_fault"),
    (ExchangeError("internal", code=-1001, status=400), "venue_fault"),
])
def test_rejected_entry_is_recorded_classified_and_not_retried_late(tmp_path, exc, cls):
    m, store, v = make(tmp_path)
    v.fail["market_order"] = [exc]
    pid = paper(store)
    open_mirror(m, store, pid)
    e = next(e for e in events(store) if e["action"] == "entry")
    assert e["cls"] == cls and e["ok"] == 0 and e["error_code"] and e["error_msg"]
    assert store.kv_get("mirror_map")[str(pid)]["state"] == "failed"
    assert v.pos == {} and not [c for c in v.calls if c[0] == "SL"]
    flush(m)
    if cls == "our_bug":
        assert any("OUR BUG" in x for x in m.sink.msgs)                  # immediate
    else:
        assert not any("OUR BUG" in x for x in m.sink.msgs)


def test_venue_fault_goes_to_the_digest_not_to_an_immediate_alert(tmp_path):
    m, store, v = make(tmp_path, digest_s=3600)
    v.fail["market_order"] = [NetworkError("HTTP 503", status=503)]
    open_mirror(m, store, paper(store))
    flush(m)
    assert m.sink.msgs == [] and m.alerts._dig_mirror == {("venue_fault", "http_503"): 1}
    m.alerts.flush_digest()
    flush(m)
    assert any("mirror faults" in x for x in m.sink.msgs)


def test_our_bug_alert_is_rate_limited_per_code_and_symbol(tmp_path):
    m, store, v = make(tmp_path)
    for i in range(3):
        v.fail["market_order"] = [ExchangeError("Precision", code=-1111, status=400)]
        m.backoff.reset("AAAUSDT")
        open_mirror(m, store, paper(store))
    flush(m)
    assert sum("OUR BUG" in x for x in m.sink.msgs) == 1
    m.clock_.t += 3601
    v.fail["market_order"] = [ExchangeError("Precision", code=-1111, status=400)]
    m.backoff.reset("AAAUSDT")
    open_mirror(m, store, paper(store))
    flush(m)
    assert sum("OUR BUG" in x for x in m.sink.msgs) == 2


def test_stop_refused_closes_the_mirror_position_like_the_engine(tmp_path):
    m, store, v = make(tmp_path)
    v.fail["SL"] = [ExchangeError("Order would immediately trigger", code=-2021, status=400)]
    pid = paper(store)
    open_mirror(m, store, pid)
    assert v.pos == {}                                                   # never left unprotected
    assert any(e["action"] == "divergence" and e["error_code"] == "stop_failed" for e in events(store))
    assert any(e["action"] == "stop" and e["cls"] == "our_bug" for e in events(store))


def test_set_leverage_failures_fall_back_then_abort(tmp_path):
    m, store, v = make(tmp_path)
    v.fail["set_leverage"] = [ExchangeError("bad lev", code=-4028, status=400)]
    pid = paper(store)
    open_mirror(m, store, pid)
    assert [c[2] for c in v.calls if c[0] == "set_leverage"] == [8, 4]    # halved, accepted
    assert store.kv_get("mirror_map")[str(pid)]["state"] == "open"


def test_unfilled_market_order_is_a_venue_fault(tmp_path):
    m, store, v = make(tmp_path)
    v.fail["market_order"] = [OrderNotFilled("accepted, never filled")]
    open_mirror(m, store, paper(store))
    e = next(e for e in events(store) if e["action"] == "entry")
    assert (e["cls"], e["error_code"]) == ("venue_fault", "not_filled")


# ------------------------------------------------------------------ timeouts / unknown outcomes
def test_timeout_is_a_venue_fault_and_the_reconciler_adopts_a_filled_entry(tmp_path):
    m, store, v = make(tmp_path, call_timeout_s=0.05)
    v.delay["market_order"] = 0.25                                        # slower than the mirror's timeout; it DID fill
    pid = paper(store)
    open_mirror(m, store, pid)
    e = next(e for e in events(store) if e["action"] == "entry")
    assert (e["cls"], e["error_code"], e["ok"]) == ("venue_fault", "timeout", 0)
    assert store.kv_get("mirror_map")[str(pid)]["state"] == "entering"   # persisted BEFORE the order: ownership is known
    time.sleep(0.3)
    v.delay.clear()
    m.clock_.t += 200                                                    # past the entering grace
    m.backoff = M.Backoff()
    asyncio.run(m.reconcile())
    mp = store.kv_get("mirror_map")[str(pid)]
    assert mp["state"] == "open" and mp["qty"] == 0.055 and v.n_calls("SL") == 1   # adopted + protected
    assert not any(e["action"] == "divergence" and e["error_code"] == "entry_failed" for e in events(store))


def test_timed_out_entry_that_never_filled_is_abandoned_with_a_divergence(tmp_path):
    m, store, v = make(tmp_path, call_timeout_s=0.05)
    v.fail["market_order"] = [M.VenueTimeout("timeout")]
    pid = paper(store)
    open_mirror(m, store, pid)
    m.clock_.t += 200
    m.backoff = M.Backoff()
    asyncio.run(m.reconcile())
    assert store.kv_get("mirror_map")[str(pid)]["state"] == "failed"
    assert [e["error_code"] for e in events(store) if e["action"] == "divergence"] == ["entry_failed"]
    asyncio.run(m.reconcile())
    asyncio.run(m.reconcile())
    assert v.n_calls("market_order") == 1                                 # never enters late


def test_failed_entry_logs_one_divergence_and_never_enters_late(tmp_path):
    m, store, v = make(tmp_path)
    v.fail["market_order"] = [ExchangeError("Precision", code=-1111, status=400)]
    pid = paper(store)
    open_mirror(m, store, pid)
    for _ in range(3):
        m.backoff = M.Backoff()
        asyncio.run(m.reconcile())
    assert sum(e["action"] == "divergence" for e in events(store)) == 1
    assert v.n_calls("market_order") == 1 and v.pos == {}
    store.delete_position(pid)                                            # paper closes: mapping dropped quietly
    m.on_close(pid, "AAAUSDT", "sl")
    asyncio.run(m.drain())
    assert store.kv_get("mirror_map") == {}


# ------------------------------------------------------------------ partial fills / maker
def test_partial_market_fill_protects_exactly_the_filled_qty(tmp_path):
    m, store, v = make(tmp_path)
    v.fill_frac = 0.5
    pid = paper(store)
    open_mirror(m, store, pid)
    assert store.kv_get("mirror_map")[str(pid)]["qty"] == pytest.approx(0.0275)
    assert [c[0] for c in v.calls if c[0] in ("SL", "TP")] == ["SL", "TP"]      # closePosition: covers whatever filled


def test_maker_entry_partial_fill_cancels_the_rest_and_protects_the_fill(tmp_path):
    m, store, v = make(tmp_path, settings={"maker_timeout_s": 5, "maker_reprices": 1})
    v.limit_fill = 0.5
    pid = paper(store)
    open_mirror(m, store, pid, via_maker=True)
    lo = next(c for c in v.calls if c[0] == "limit_order")
    assert lo[2] == "BUY" and lo[4] == pytest.approx(99.99)               # post-only at the venue's best bid
    assert v.n_calls("cancel_order") >= 1                                 # remainder cancelled
    assert store.kv_get("mirror_map")[str(pid)]["qty"] == pytest.approx(0.0275)
    assert v.n_calls("SL") == 1


def test_maker_gtx_reject_reprices_and_no_fill_abandons_the_entry(tmp_path):
    m, store, v = make(tmp_path, settings={"maker_timeout_s": 5, "maker_reprices": 1, "maker_on_timeout": "skip"})
    m.poll_s = 100.0                                                      # one poll per attempt
    v.fail["limit_order"] = [ExchangeError("Post Only would take", code=-5022, status=400)]
    pid = paper(store)
    open_mirror(m, store, pid, via_maker=True)
    rej = next(e for e in events(store) if e["error_code"] == "-5022")
    assert rej["cls"] == "ok"                                             # expected answer, not a failure
    assert v.n_calls("limit_order") == 2 and v.pos == {}
    assert store.kv_get("mirror_map")[str(pid)]["state"] == "failed"


def test_maker_no_fill_falls_back_to_market_when_configured(tmp_path):
    m, store, v = make(tmp_path, settings={"maker_timeout_s": 5, "maker_reprices": 0, "maker_on_timeout": "taker"})
    m.poll_s = 100.0
    pid = paper(store)
    open_mirror(m, store, pid, via_maker=True)
    assert v.n_calls("market_order") == 1 and store.kv_get("mirror_map")[str(pid)]["state"] == "open"


# ------------------------------------------------------------------ reconciler
def test_orphan_venue_position_is_closed_stops_first(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    store.delete_position(pid)                                            # paper closed but the close was lost (restart)
    asyncio.run(m.reconcile())
    assert v.pos == {} and v.conds == []
    ev = events(store)
    assert any(e["action"] == "divergence" and e["error_code"] == "orphan_position" for e in ev)
    oc = next(e for e in ev if e["action"] == "orphan_close")
    assert oc["ok"] == 1 and oc["cls"] == "ok"
    names = [c[0] for c in v.calls]
    assert names.index("cancel_order") < len(names) - 1 and store.kv_get("mirror_map") == {}
    last_close = [c for c in v.calls if c[0] == "market_order"][-1]
    assert last_close[3] == 0.055                                         # exact qty


def test_missing_stop_is_replaced_and_logged_as_a_divergence(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    v.conds = [o for o in v.conds if o.kind != "SL"]                      # the stop disappeared
    n = v.n_calls("SL")
    asyncio.run(m.reconcile())
    assert v.n_calls("SL") == n + 1 and any(o.kind == "SL" for o in v.conds)
    ev = events(store)
    assert any(e["action"] == "divergence" and e["error_code"] == "stop_missing" for e in ev)
    assert any(e["action"] == "stop_replace" and e["ok"] == 1 for e in ev)
    assert store.kv_get("mirror_map")[str(pid)]["sl_id"] == next(o.order_id for o in v.conds if o.kind == "SL")


def test_stop_triggered_but_not_filled_closes_the_position(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    v.conds = []                                                          # stop gone ...
    v.marks["AAAUSDT"] = 98.0                                             # ... and the mark is through it: it triggered, not filled
    asyncio.run(m.reconcile())
    d = [e for e in events(store) if e["action"] == "divergence"]
    assert d[0]["error_code"] == "stop_not_filled" and d[0]["cls"] == "venue_fault"
    assert v.pos == {}


def test_venue_closed_early_is_a_divergence_and_not_reentered(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    v.pos.clear()
    asyncio.run(m.reconcile())
    asyncio.run(m.reconcile())
    assert [e["error_code"] for e in events(store) if e["action"] == "divergence"] == ["venue_closed_early"]
    assert v.n_calls("market_order") == 1


def test_stale_reduce_only_orders_are_cancelled_and_stop_resynced(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    v.conds.append(ConditionalOrder("a:999", "AAAUSDT", "SL", "SELL", 90.0, 0.055, True))
    store.update_position(pid, sl_price=99.5)                             # paper moved its stop while the mirror was down
    asyncio.run(m.reconcile())
    assert "a:999" not in {o.order_id for o in v.conds}
    assert sorted(o.trigger_price for o in v.conds if o.kind == "SL") == [99.5]


# ------------------------------------------------------------------ ownership
def test_foreign_position_and_non_trading_symbols_are_skipped_and_untouched(tmp_path):
    m, store, v = make(tmp_path, FakeVenue({"AAAUSDT": 100.0, "BBBUSDT": 50.0}, status={"BBBUSDT": "BREAK"}))
    v.pos["AAAUSDT"] = PositionInfo("AAAUSDT", "short", 3.0, 99.0)       # a ghost the mirror does not own
    a = paper(store, "AAAUSDT")
    b = paper(store, "BBBUSDT", entry=50.0, qty=3.0, sl=49.0, tp=51.0)
    open_mirror(m, store, a)
    open_mirror(m, store, b)
    assert v.n_calls("market_order") == 0
    assert [e["error_code"] for e in events(store) if e["cls"] == "skipped"] == ["foreign_position", "not_trading"]
    asyncio.run(m.reconcile())
    store.delete_position(a)
    m.on_close(a, "AAAUSDT", "sl")
    asyncio.run(m.drain())
    asyncio.run(m.reconcile())
    assert v.n_calls("market_order") == 0 and v.pos["AAAUSDT"].qty == 3.0     # the ghost is never touched
    assert not any(e["action"] == "divergence" for e in events(store))
    assert store.kv_get("mirror_state")["foreign_positions"] == ["AAAUSDT"]


def test_ownership_survives_a_restart(tmp_path):
    m, store, v = make(tmp_path)
    pid = paper(store)
    open_mirror(m, store, pid)
    m2 = M.Mirror(store, m.alerts, lambda: m.cfg_, clock=m.clock_, venue=v)      # "restart": map reloaded from kv
    m2._next_recon = m2._next_poll = 1e18
    store.delete_position(pid)
    asyncio.run(m2.reconcile())
    assert v.pos == {}                                                    # it closed what it had opened


# ------------------------------------------------------------------ account health
def test_unusable_account_sits_quietly_alerts_once_and_resumes(tmp_path):
    v = FakeVenue()
    v.account = {"totalMarginBalance": "-46.83", "availableBalance": "0.01", "totalWalletBalance": "0.01", "canTrade": True}
    m, store, v = make(tmp_path, v)
    pids = [paper(store), paper(store, "BBBUSDT", entry=50.0, qty=3.0, sl=49.0, tp=51.0)]
    for p in pids:
        open_mirror(m, store, p)
    assert m.account["state"] == "unusable"
    assert not [c for c in v.calls if c[0] in ("market_order", "set_isolated", "set_leverage", "SL", "TP", "limit_order")]
    ev = events(store)
    assert {e["cls"] for e in ev} == {"skipped"} and {e["error_code"] for e in ev} == {"account_unusable"}
    flush(m)
    assert sum("ACCOUNT UNUSABLE" in x for x in m.sink.msgs) == 1       # ONE alert
    asyncio.run(m.reconcile())
    assert not any(e["action"] == "divergence" for e in events(store))    # skipped entries are not divergences
    v.account = {"totalMarginBalance": "5000", "availableBalance": "4900", "totalWalletBalance": "5000", "canTrade": True}
    m.clock_.t += 100
    asyncio.run(m.poll_account())
    assert m.account["state"] == "ok"
    flush(m)
    assert any("usable again" in x for x in m.sink.msgs)
    p3 = paper(store, "AAAUSDT")                                          # new paper position after recovery is mirrored
    open_mirror(m, store, p3)
    assert v.n_calls("market_order") == 1
    assert store.kv_get("mirror_map")[str(pids[0])]["state"] == "skipped"   # the old one is NOT entered late


@pytest.mark.parametrize("info,why", [
    ({"totalMarginBalance": "0", "availableBalance": "900", "totalWalletBalance": "5", "canTrade": True}, "totalMarginBalance"),
    ({"totalMarginBalance": "100", "availableBalance": "900", "totalWalletBalance": "100", "canTrade": False}, "canTrade"),
    ({"totalMarginBalance": "100", "availableBalance": "0.5", "totalWalletBalance": "100", "canTrade": True}, "availableBalance"),
])
def test_account_health_conditions(tmp_path, info, why):
    v = FakeVenue()
    v.account = info
    m, store, v = make(tmp_path, v)
    assert asyncio.run(m.poll_account()) == "unusable"
    assert why in " ".join(m.account["reasons"])


def test_reject_streak_on_two_symbols_marks_the_account_unusable_even_with_a_healthy_balance(tmp_path):
    m, store, v = make(tmp_path)
    for s in ("AAAUSDT", "BBBUSDT", "AAAUSDT"):
        v.fail["market_order"] = [ExchangeError("Margin is insufficient", code=-2019, status=400)]
        m.backoff = M.Backoff()
        pid = paper(store, s, entry=100.0 if s == "AAAUSDT" else 50.0)
        open_mirror(m, store, pid)
    assert m.account["state"] == "unusable" and "order rejects" in m.account["reasons"][0]
    flush(m)
    assert sum("ACCOUNT UNUSABLE" in x for x in m.sink.msgs) == 1
    n = v.n_calls("market_order")
    open_mirror(m, store, paper(store))
    assert v.n_calls("market_order") == n                                 # quiet now
    # the balance fields cannot clear it (the account looks healthy) ...
    m.clock_.t += 100
    assert asyncio.run(m.poll_account()) == "unusable"
    # ... a user retry does, and so does a changed (fresh) account
    asyncio.run(m.retry())
    assert m.account["state"] == "ok"


def test_retry_flag_from_the_api_clears_the_verdict(tmp_path):
    m, store, v = make(tmp_path)
    m.account.update(sticky={"why": "x", "wallet": 1000.0, "margin": 1000.0})
    m._set_account("unusable", ["x"], None)
    store.kv_set(M.KEY_RETRY, time.time() + 1)
    asyncio.run(m.cycle())
    asyncio.run(m.drain())
    assert m.account["state"] == "ok"


def test_fresh_account_clears_a_reject_verdict(tmp_path):
    m, store, v = make(tmp_path)
    asyncio.run(m.poll_account())
    for s in ("AAAUSDT", "BBBUSDT", "AAAUSDT"):
        v.fail["market_order"] = [ExchangeError("Margin is insufficient", code=-2019, status=400)]
        m.backoff = M.Backoff()
        open_mirror(m, store, paper(store, s, entry=100.0 if s == "AAAUSDT" else 50.0))
    assert m.account["state"] == "unusable"
    v.account = {"totalMarginBalance": "7000", "availableBalance": "7000", "totalWalletBalance": "7000", "canTrade": True}
    assert asyncio.run(m.poll_account()) == "ok"


def test_no_keys_means_skipped_not_a_crash(tmp_path):
    store = Store(tmp_path / "tb.sqlite")
    cfg = Settings().merge({"mirror_enabled": True})

    def nokeys():
        raise RuntimeError("no keys")

    m = M.Mirror(store, AlertDispatcher([Sent()], min_interval=0), lambda: cfg, venue_factory=nokeys)
    m._next_recon = m._next_poll = 1e18
    open_mirror(m, store, paper(store))
    assert events(store)[0]["cls"] == "skipped" and events(store)[0]["error_code"] == "no_venue"


# ------------------------------------------------------------------ settings
def test_settings_validation_refuses_live_and_unknown_values():
    s = Settings()
    assert s.mirror_enabled is False and s.mirror_venue == "binance_demo" and s.mirror_size == "min"
    assert s.merge({"mirror_enabled": True}).mirror_enabled is True
    with pytest.raises(ValueError, match="live"):
        s.merge({"mirror_venue": "live"})
    with pytest.raises(ValueError):
        s.merge({"mirror_venue": "paper"})
    with pytest.raises(ValueError):
        s.merge({"mirror_size": "max"})
    assert {"mirror_enabled", "mirror_venue", "mirror_size"} <= set(s.to_api())


def test_disabled_mirror_ignores_hooks(tmp_path):
    m, store, v = make(tmp_path)
    m.cfg_fn = lambda: Settings()
    open_mirror(m, store, paper(store))
    assert v.calls == [] and events(store) == []


def test_hooks_never_raise(tmp_path):
    m, store, v = make(tmp_path)
    m.on_open(1, None)                      # (garbage in)
    m.on_stop_move("x", None, None)
    m.on_close(None, None, None)
    m.store = None
    m.on_partial(1, "AAAUSDT", 0.5, None, None)


# ------------------------------------------------------------------ API summary
def test_api_mirror_summary_shape(tmp_path):
    from fastapi.testclient import TestClient

    from tb.api import create_app

    m, store, v = make(tmp_path)
    m.clock_.t = time.time()                                              # the API summarises the last hour of REAL time
    v.fail["market_order"] = [ExchangeError("Precision", code=-1111, status=400)]
    open_mirror(m, store, paper(store))
    open_mirror(m, store, paper(store, "BBBUSDT", entry=50.0, qty=3.0, sl=49.0, tp=51.0))
    m._write_state(force=True)
    store.patch_settings({"mirror_enabled": True})
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("x")
    c = TestClient(create_app(tmp_path / "tb.sqlite", ui=ui))
    j = c.get("/api/mirror").json()
    assert j["enabled"] is True and j["venue"] == "binance_demo" and j["account_state"] == "ok"
    assert j["counts"]["by_cls"]["our_bug"] == 1 and j["counts"]["by_error_code"]["-1111"] == 1
    assert 0 < j["success_rate_1h"] < 1 and j["success_rate_24h"] == j["success_rate_1h"]
    assert j["latency_p50_ms"] is not None and j["latency_p95_ms"] >= j["latency_p50_ms"]
    assert [p["symbol"] for p in j["open_positions"]] == ["BBBUSDT"]
    assert len(j["events"]) <= 50 and j["events"][0]["id"] > j["events"][-1]["id"]
    assert {"enabled", "venue", "account_state", "success_rate_1h", "success_rate_24h", "latency_p50_ms", "latency_p95_ms",
            "counts", "divergences", "open_positions", "events"} <= set(j)
    assert c.post("/api/mirror/retry").json() == {"ok": True} and store.kv_get(M.KEY_RETRY)


# ------------------------------------------------------------------ the paper book is byte-identical with the mirror on / off
def snapshot(eng, store):
    return {
        # closed_at is the paper exchange's wall-clock second: two runs may straddle a second boundary
        "trades": [{k: v for k, v in dict(r).items() if k != "closed_at"} for r in store.query("SELECT * FROM trades ORDER BY id")],
        "positions": [dict(r) for r in store.query("SELECT * FROM positions ORDER BY id")],
        "paper_state": store.kv_get("paper_state"), "protect": store.kv_get("protect"),
        "equity": (eng.equity, eng.cash, eng.peak_equity), "kill": eng.kill, "safety": eng.safety_view["state"],
        "backoff": sorted(map(str, eng.backoff.keys())),
        "wallet": eng.ex.balance().wallet,
    }


def scenario(store, tmp_path, venue):
    eng, data = mk_engine(store, tmp_path, settings={"mirror_enabled": venue is not None})
    if venue is not None:
        eng.mirror.venue = venue
        eng.mirror._next_recon = eng.mirror._next_poll = 1e18
    def drain():                                                          # let the mirror act after every paper step
        if venue is not None:
            run(eng.mirror.drain())

    open_long(eng, "AAAUSDT")
    drain()
    p = store.get_positions()[0]
    run(eng.prices_cycle())
    data.marks["AAAUSDT"] = p["entry_price"] + 0.6 * (p["entry_price"] - p["sl_price"])
    run(eng.prices_cycle())                                               # may move the stop
    drain()
    open_long(eng, "BBBUSDT")
    drain()
    data.marks["AAAUSDT"] = p["tp_price"] * 1.001
    run(eng.prices_cycle())                                               # TP hit
    drain()
    run(eng.close_all("manual"))
    drain()
    return snapshot(eng, store)


@pytest.mark.parametrize("flavour", ["healthy", "failing", "slow_and_broken"])
def test_paper_book_is_identical_with_the_mirror_on_and_off(tmp_path, flavour):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    off_store = Store(tmp_path / "a" / "tb.sqlite")
    off = scenario(off_store, tmp_path / "a", None)
    v = FakeVenue({"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0})
    if flavour == "failing":
        for name in ("market_order", "set_leverage", "positions", "account_info"):
            v.fail[name] = [ExchangeError("boom", code=-1111, status=400)] * 50
    if flavour == "slow_and_broken":
        v.fail["market_order"] = [NetworkError("HTTP 503", status=503)] * 50
        v.fail["SL"] = [NetworkError("HTTP 503", status=503)] * 50
    on_store = Store(tmp_path / "b" / "tb.sqlite")
    on = scenario(on_store, tmp_path / "b", v)
    assert on == off
    assert v.calls                                                         # the mirror really ran
    assert off["trades"] and off["trades"][0]["exit_reason"]


def test_mirror_error_cannot_reach_the_engine_safety_state(tmp_path):
    store = Store(tmp_path / "tb.sqlite")
    eng, data = mk_engine(store, tmp_path, settings={"mirror_enabled": True})
    eng.mirror.venue = FakeVenue({"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0})
    eng.mirror.venue.fail["market_order"] = [ExchangeError("x", code=-1111, status=400)] * 5
    eng.mirror._next_recon = eng.mirror._next_poll = 1e18
    open_long(eng)
    run(eng.mirror.drain())
    assert not eng.kill.get("tripped") and eng.safety_view["state"] == "normal".upper() or eng.safety_view["state"] == "NORMAL"
    assert eng.backoff.keys() == [] and eng.stats["errors"] == 0
    assert len(store.get_positions()) == 1
