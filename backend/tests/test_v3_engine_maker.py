"""Maker (post-only GTX) entries: Binance request shape, paper simulation, and the engine's wait / re-price /
partial-fill / abort flow (stops only after the fill; the trade lock is free while the order rests)."""
import pytest
from urllib.parse import parse_qs, urlsplit

from conftest import make_klines
from test_engine_core import FAST, mk_engine, closed_of, load_bars, run
from test_exchange_binance import Resp, mk, qs

from tb.exchange.base import ExchangeError
from tb.exchange.paper import PaperExchange

MAKER = {"entry_mode": "maker", "maker_timeout_s": 5, "maker_reprices": 1, "maker_on_timeout": "skip"}


# ------------------------------------------------------------------ Binance request shape / error mapping
def test_binance_limit_order_is_post_only_gtx():
    ex, s = mk(Resp(200, {"orderId": 7, "status": "NEW", "executedQty": "0", "origQty": "0.5", "avgPrice": "0",
                          "price": "2000.5", "side": "BUY"}))
    r = ex.limit_order("ETHUSDT", "BUY", 0.5, 2000.5)
    q = qs(s.calls[0])
    assert s.calls[0]["method"] == "POST" and urlsplit(s.calls[0]["url"]).path == "/fapi/v1/order"
    assert q["type"] == "LIMIT" and q["timeInForce"] == "GTX" and q["price"] == "2000.5" and q["quantity"] == "0.5"
    assert q["newClientOrderId"].startswith("tbm") and "reduceOnly" not in q
    assert r.order_id == "o:7" and r.status == "NEW" and r.filled == 0 and r.price == 2000.5


def test_binance_gtx_rejection_5022_is_an_exchange_error_with_code():
    ex, _ = mk(Resp(400, {"code": -5022, "msg": "Due to the order could not be executed as maker, the Post Only order "
                                                 "will be rejected."}))
    with pytest.raises(ExchangeError) as e:
        ex.limit_order("ETHUSDT", "BUY", 0.5, 2000.5)
    assert e.value.code == -5022


def test_binance_gtx_expired_unfilled_is_also_a_rejection():
    ex, _ = mk(Resp(200, {"orderId": 8, "status": "EXPIRED", "executedQty": "0", "origQty": "0.5", "avgPrice": "0"}))
    with pytest.raises(ExchangeError) as e:
        ex.limit_order("ETHUSDT", "SELL", 0.5, 2100)
    assert e.value.code == -5022


def test_binance_get_order_reports_partial_fill_and_book_ticker():
    ex, s = mk(Resp(200, {"orderId": 7, "status": "PARTIALLY_FILLED", "executedQty": "0.2", "origQty": "0.5",
                          "avgPrice": "2000.5", "price": "2000.5", "side": "BUY"}),
               Resp(200, {"symbol": "ETHUSDT", "bidPrice": "2000.4", "askPrice": "2000.5"}))
    o = ex.get_order("ETHUSDT", "o:7")
    assert o.status == "PARTIALLY_FILLED" and o.filled == pytest.approx(0.2) and o.avg_price == 2000.5
    assert qs(s.calls[0])["orderId"] == "7"
    assert ex.best_bid_ask("ETHUSDT") == (2000.4, 2000.5)
    assert urlsplit(s.calls[1]["url"]).path == "/fapi/v1/ticker/bookTicker" if "url" in s.calls[1] else True


def test_binance_lists_only_our_stale_maker_entries():
    ex, _ = mk(Resp(200, [
        {"symbol": "ETHUSDT", "orderId": 1, "type": "LIMIT", "clientOrderId": "tbmabc", "reduceOnly": False},
        {"symbol": "ETHUSDT", "orderId": 2, "type": "LIMIT", "clientOrderId": "someone", "reduceOnly": False},
        {"symbol": "ETHUSDT", "orderId": 3, "type": "STOP_MARKET", "clientOrderId": "tbx", "reduceOnly": True}]))
    assert ex.open_entry_orders() == [("ETHUSDT", "o:1")]


# ------------------------------------------------------------------ paper venue
def paper_with_book(bid, ask):
    from conftest import FakeData
    from tb.config import Settings

    data = FakeData()
    ex = PaperExchange(data, Settings().merge({"slippage_pct": 0.0}, allow_internal=True))
    ex.marks["AAAUSDT"] = (bid + ask) / 2
    ex.best_bid_ask = lambda s: (bid, ask)
    return ex


def test_paper_post_only_rejects_a_crossing_limit_with_5022():
    ex = paper_with_book(99.9, 100.1)
    with pytest.raises(ExchangeError) as e:
        ex.limit_order("AAAUSDT", "BUY", 1.0, 100.1)              # at the ask: would take
    assert e.value.code == -5022
    with pytest.raises(ExchangeError):
        ex.limit_order("AAAUSDT", "SELL", 1.0, 99.9)


def test_paper_limit_fills_only_when_price_trades_strictly_through_at_the_limit_with_maker_fee():
    ex = paper_with_book(99.9, 100.1)
    o = ex.limit_order("AAAUSDT", "BUY", 1.0, 99.9)
    assert o.status == "NEW" and ex.positions() == []
    ex.tick({"AAAUSDT": 99.9})                                    # touching is not enough (queue position)
    assert ex.get_order("AAAUSDT", o.order_id).filled == 0
    w0 = ex.wallet
    ex.tick({"AAAUSDT": 99.89})
    g = ex.get_order("AAAUSDT", o.order_id)
    assert g.status == "FILLED" and g.filled == 1.0 and g.avg_price == 99.9
    p = ex.positions()[0]
    assert p.entry_price == 99.9 and ex.wallet == pytest.approx(w0 - 99.9 * ex.cfg.fee_maker)    # maker fee, no slippage


def test_paper_partial_fill_then_cancel_keeps_the_filled_part():
    ex = paper_with_book(99.9, 100.1)
    ex.maker_fill_ratio = 0.4
    o = ex.limit_order("AAAUSDT", "BUY", 1.0, 99.9)
    ex.tick({"AAAUSDT": 99.0})
    g = ex.get_order("AAAUSDT", o.order_id)
    assert g.status == "PARTIALLY_FILLED" and g.filled == pytest.approx(0.4)
    ex.cancel_order("AAAUSDT", o.order_id)
    g = ex.get_order("AAAUSDT", o.order_id)
    assert g.filled == pytest.approx(0.4) and ex.limits == {} and ex.positions()[0].qty == pytest.approx(0.4)


def test_paper_later_bar_fills_only_bars_after_placement():
    ex = paper_with_book(99.9, 100.1)
    o = ex.limit_order("AAAUSDT", "BUY", 1.0, 99.9)
    placed = ex.limits[o.order_id].placed_ms
    ex.check_range("AAAUSDT", 99.0, 101.0, placed - 1)            # a bar that started before the order: ignored
    assert ex.positions() == []
    ex.check_range("AAAUSDT", 99.0, 101.0, placed + 1)
    assert ex.positions()[0].qty == 1.0


# ------------------------------------------------------------------ engine flow
def setup(store, tmp_path, extra=None, **kw):
    eng, data = mk_engine(store, tmp_path, settings={**MAKER, **(extra or {})}, **kw)
    sym = "AAAUSDT"
    load_bars(eng, [sym, "BBBUSDT", "BTCUSDT"])
    mark = eng.marks.get(sym) or data.marks[sym]
    eng.marks[sym] = mark
    eng.ex.marks[sym] = mark
    book = {"bid": round(mark * 0.9995, 2), "ask": round(mark * 1.0005, 2)}
    eng.ex.best_bid_ask = lambda s: (book["bid"], book["ask"])
    eng.log = []
    for name in ("limit_order", "place_stop", "place_take_profit", "market_order", "cancel_order"):
        orig = getattr(eng.ex, name)

        def wrap(*a, _o=orig, _n=name, **k):
            eng.log.append(_n)
            return _o(*a, **k)

        setattr(eng.ex, name, wrap)
    return eng, data, sym, book


def trigger(eng, sym, book, factor=0.9990):
    eng.ex.tick({sym: book["bid"] * factor})


def open_it(eng, sym):
    return run(eng.try_open(sym, "long", 0.9, [], closed_of(eng, sym)))


def test_maker_entry_fills_then_places_stop_and_leaves_the_trade_lock_free(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path)
    seen = {}
    calls = {"n": 0}

    async def sleep(s):
        calls["n"] += 1
        seen.setdefault("locked", eng.tlock.locked())
        seen.setdefault("pending", dict(eng.pending))
        if calls["n"] == 2:
            trigger(eng, sym, book)

    eng.maker_sleep = sleep
    assert open_it(eng, sym) is True
    assert seen["locked"] is False and sym in seen["pending"]           # kill / close_all / stop checks keep running
    p = store.get_positions()[0]
    assert p["entry_price"] == book["bid"] and p["sl_order_id"]                  # filled AT the best bid, not above it
    assert eng.log.index("limit_order") < eng.log.index("place_stop")            # the stop exists only after the fill
    assert "market_order" not in eng.log
    assert eng.maker_stats["filled"] == 1 and eng.maker_stats["partial"] == 0 and eng.pending == {}
    assert p["entry_fee"] == pytest.approx(p["qty"] * p["entry_price"] * eng.cfg.fee_maker)
    assert {o.kind for o in eng.ex.open_conditional_orders(sym)} == {"SL", "TP"}


def test_gtx_rejection_5022_reprices_and_then_fills(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path)
    good = dict(book)
    state = {"n": 0}

    def bba(s):
        state["n"] += 1
        if state["n"] == 1:                                              # the book moved: our bid is now at the ask
            return good["ask"], good["ask"]
        return good["bid"], good["ask"]

    eng.ex.best_bid_ask = bba
    polls = {"n": 0}

    async def sleep(s):
        polls["n"] += 1
        if eng.ex.limits:
            trigger(eng, sym, good)

    eng.maker_sleep = sleep
    assert open_it(eng, sym) is True
    assert eng.maker_stats["gtx_rejects"] == 1 and eng.log.count("limit_order") == 2        # rejected once, reposted
    assert store.get_positions()[0]["entry_price"] == good["bid"]


def test_partial_fill_cancels_the_rest_and_protects_exactly_the_filled_quantity(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path)
    eng.ex.maker_fill_ratio = 0.4

    async def sleep(s):
        if eng.ex.limits:
            trigger(eng, sym, book)

    eng.maker_sleep = sleep
    assert open_it(eng, sym) is True
    p = store.get_positions()[0]
    full = eng.cfg.notional_usd / book["bid"]
    assert p["qty"] == pytest.approx(full * 0.4, rel=0.02) and p["qty"] < full
    assert eng.ex.limits == {}                                           # the remainder was cancelled, never re-posted
    sl = [o for o in eng.ex.open_conditional_orders(sym) if o.kind == "SL"][0]
    assert sl.close_position and sl.qty == 0.0                          # closePosition: no quantity at all
    assert eng.maker_stats["partial"] == 1 and eng.maker_stats["filled"] == 1


def test_no_fill_times_out_and_skips_without_a_position_or_a_resting_order(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path)

    async def sleep(s):
        pass

    eng.maker_sleep = sleep
    assert open_it(eng, sym) is False
    assert store.get_positions() == [] and eng.ex.limits == {} and eng.ex.positions() == []
    assert eng.log.count("limit_order") == 2 and eng.log.count("cancel_order") == 2          # 1 re-price = 2 placements
    assert eng.maker_stats["timeouts"] == 1 and eng.maker_stats["skipped"] == 1 and eng.maker_stats["reprices"] == 1
    assert "place_stop" not in eng.log and eng.pending == {}


def test_no_fill_taker_fallback_per_setting(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path, extra={"maker_on_timeout": "taker"})

    async def sleep(s):
        pass

    eng.maker_sleep = sleep
    assert open_it(eng, sym) is True
    assert eng.maker_stats["taker_fallback"] == 1 and "market_order" in eng.log
    assert store.get_positions()[0]["sl_order_id"]


def test_entries_blocked_while_waiting_cancel_the_order_and_open_nothing(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path)

    async def sleep(s):
        eng.kill = {"tripped": True, "reason": "test", "t": 0}

    eng.maker_sleep = sleep
    assert open_it(eng, sym) is False
    assert eng.ex.limits == {} and store.get_positions() == [] and eng.maker_stats["aborted"] == 1


def test_a_fill_that_lands_after_entries_were_blocked_is_closed_not_kept_unprotected(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path)

    async def sleep(s):
        if eng.ex.limits:
            trigger(eng, sym, book)
            eng.running = False                                          # operator pressed stop / kill during the wait

    eng.maker_sleep = sleep
    assert open_it(eng, sym) is False
    assert store.get_positions() == [] and eng.ex.positions() == []      # flattened immediately
    assert len(store.get_trades(5)) == 1                                 # the flat round trip (fees) is recorded


def test_stop_placement_failure_after_a_maker_fill_still_closes_the_position(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path)

    def boom(*a):
        raise ExchangeError("stop rejected", code=-2021)

    eng.ex.place_stop = boom

    async def sleep(s):
        if eng.ex.limits:
            trigger(eng, sym, book)

    eng.maker_sleep = sleep
    assert open_it(eng, sym) is False
    assert store.get_positions() == [] and eng.ex.positions() == []


def test_in_flight_maker_entry_counts_against_the_caps_and_blocks_a_duplicate(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path)
    during = {}

    async def sleep(s):
        if "bk" not in during:
            during["bk"] = eng.book()
            during["dup"] = await eng.try_open(sym, "long", 0.9, [], closed_of(eng, sym))
        if eng.ex.limits:
            trigger(eng, sym, book)

    eng.maker_sleep = sleep
    before = eng.book()
    assert open_it(eng, sym) is True
    assert during["bk"].open_notional > before.open_notional and during["bk"].available < before.available
    assert during["dup"] is False and len(store.get_positions()) == 1


def test_sync_does_not_close_the_unprotected_fill_of_a_pending_maker_entry(store, tmp_path):
    """Between the fill and the stop the position has no record: reconcile must leave it to the entry flow."""
    eng, data, sym, book = setup(store, tmp_path)
    seen = {}

    async def sleep(s):
        if eng.ex.limits:
            trigger(eng, sym, book)
            eng.paper = False                                            # exercise the real-venue reconcile branch
            try:
                async with eng.tlock:
                    await eng.sync_positions({sym: book["bid"]})
            finally:
                eng.paper = True
            seen["still_there"] = bool(eng.ex.positions())

    eng.maker_sleep = sleep
    assert open_it(eng, sym) is True and seen["still_there"] is True
    assert len(store.get_positions()) == 1


def test_taker_mode_is_unchanged_by_the_maker_machinery(store, tmp_path):
    eng, data, sym, book = setup(store, tmp_path, extra={"entry_mode": "taker"})
    assert open_it(eng, sym) is True
    assert "limit_order" not in eng.log and eng.maker_stats["entries"] == 0


def test_stale_maker_orders_are_cancelled_at_startup(store, tmp_path):
    class Ex(PaperExchange):
        pass

    eng, data = mk_engine(store, tmp_path)
    cancelled = []
    eng.paper = False
    eng.ex.open_entry_orders = lambda symbol=None: [("AAAUSDT", "o:9")]
    eng.ex.cancel_order = lambda s, o: cancelled.append((s, o))
    eng.cancel_stale_entries()
    assert cancelled == [("AAAUSDT", "o:9")]
