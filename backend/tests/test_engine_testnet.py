"""binance_demo / testnet semantics: prod data for signals, the ORDER VENUE for everything that touches orders."""
import logging
import time

import pytest

from conftest import FakeData, make_klines
from test_engine_core import FAST, Sent, closed_of, run
from tb import engine as E
from tb import universe as uni
from tb.alerts import AlertDispatcher
from tb.config import Settings
from tb.exchange.base import (Balance, ConditionalOrder, ExchangeError, OrderResult, PositionInfo, Premium, SymbolFilters,
                              SymbolInfo)

BAR = 300_000


class FakeVenue(FakeData):
    """A testnet-like order venue: its own symbol list, filters, mark prices and fills."""
    name = "fakevenue"

    def __init__(self, symbols, marks, min_notional=5.0, tick=0.01):
        super().__init__(tuple(symbols))
        self.marks = dict(marks)
        self.min_notional = min_notional
        self.tick = tick
        self.orders: list[ConditionalOrder] = []
        self.pos: dict[str, PositionInfo] = {}
        self.log: list[tuple] = []
        self.fill_override: float | None = None
        self.next_mark: float | None = None            # what the *next* premium(symbol) call returns (one shot)
        self.wallet = 100.0
        self.info_fail = False

    def exchange_info(self):
        if self.info_fail:
            raise ExchangeError("venue down")
        return {s: SymbolInfo(s, filters=SymbolFilters(self.tick, 0.001, 0.001, self.min_notional)) for s in self.marks}

    def premium(self, symbol=None):
        if symbol and self.next_mark is not None:
            m, self.next_mark = self.next_mark, None
            return [Premium(symbol, m)]
        return [Premium(s, m, m, 0.0001, 0) for s, m in self.marks.items() if symbol in (None, s)]

    def set_leverage(self, s, lev): self.log.append(("lev", s, lev))
    def set_isolated(self, s): self.log.append(("iso", s))

    def market_order(self, symbol, side, qty, reduce_only=False):
        if symbol not in self.marks:
            raise ExchangeError("Invalid symbol", code=-1121)
        px = self.fill_override or self.marks[symbol]
        self.log.append(("market", symbol, side, qty, reduce_only, px))
        if reduce_only:
            self.pos.pop(symbol, None)
            self.orders = [o for o in self.orders if o.symbol != symbol]
        else:
            self.pos[symbol] = PositionInfo(symbol, "long" if side == "BUY" else "short", qty, px, px, 0.0, 8, 0.0, qty * px / 8)
        return OrderResult(f"o:{len(self.log)}", symbol, side, qty, px, "FILLED", qty * px * 0.0005)

    def _cond(self, kind, symbol, side, price, qty):
        mark = self.marks[symbol]
        if (kind == "SL") == (side == "SELL"):
            if price >= mark:
                raise ExchangeError("Order would immediately trigger.", code=-2021)
        elif price <= mark:
            raise ExchangeError("Order would immediately trigger.", code=-2021)
        self.orders.append(ConditionalOrder(f"o:{len(self.orders) + 100}", symbol, kind, side, price, qty, True))
        self.log.append((kind, symbol, side, price, qty))
        return self.orders[-1].order_id

    def place_stop(self, symbol, side, price, qty): return self._cond("SL", symbol, side, price, qty)
    def place_take_profit(self, symbol, side, price, qty): return self._cond("TP", symbol, side, price, qty)
    def cancel_order(self, symbol, oid): self.orders = [o for o in self.orders if o.order_id != oid]
    def cancel_all(self, symbol): self.orders = [o for o in self.orders if o.symbol != symbol]
    def open_conditional_orders(self, symbol=None): return [o for o in self.orders if symbol in (None, o.symbol)]
    def positions(self): return list(self.pos.values())
    def balance(self): return Balance(self.wallet, self.wallet - sum(p.margin for p in self.pos.values()), 0.0)


def mk_demo(store, tmp_path, *, prod_marks, venue_marks, venue_symbols=None, settings=None, min_notional=5.0):
    now = int(time.time() // 300 * 300) + 10.0
    syms = tuple(prod_marks)
    data = FakeData(syms)
    for i, s in enumerate(syms):
        data.kl[s] = make_klines(base=prod_marks[s], seed=i + 1, end_ms=int(now * 1000))
        data.marks[s] = float(data.kl[s]["close"].iloc[-1])
    data.server_time_ms = lambda: int(now * 1000)
    if settings:
        store.patch_settings(settings, allow_internal=True)
    store.save_settings(store.get_settings().merge(FAST, allow_internal=True))
    venue = FakeVenue(venue_symbols or syms, venue_marks, min_notional)
    eng = E.Engine(store, data, venue, "binance_demo", AlertDispatcher([Sent()], min_interval=0),
                   models_path=tmp_path / "models", clock=lambda: now)
    eng.startup()
    eng.universe = list(syms[:2])
    eng.running = True
    eng.safety_view = {"state": "NORMAL", "reasons": []}       # (the guard process is not part of this test)
    run(eng.prices_cycle())
    return eng, data, venue


async def _open(eng, sym, side="long"):
    await eng.update_symbol(sym)
    await eng.try_open(sym, side, 0.9, [], closed_of(eng, sym))


def open_(eng, sym, side="long"):
    run(_open(eng, sym, side))


# ------------------------------------------------------------------ (1) universe = prod ranking  ∩  venue
def test_universe_intersects_venue_and_logs_dropped():
    prod = FakeData(("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "BTCUSDT"))
    # make BBB the most volatile prod coin so it ranks first, then check it is dropped
    prod.kl["BBBUSDT"] = make_klines(base=100.0, seed=7, vol=0.01)
    prod.exchange_info = lambda: {s: SymbolInfo(s, onboard_date_ms=0) for s in prod.marks}
    cfg = Settings().merge({"universe_size": 2})
    base = uni.refresh(prod, cfg, {})
    assert base[0]["symbol"] == "BBBUSDT"
    venue = {s: SymbolInfo(s) for s in ("AAAUSDT", "CCCUSDT", "DDDUSDT", "BTCUSDT")}          # BBBUSDT missing
    venue["CCCUSDT"].status = "BREAK"                                                          # present but not TRADING
    dropped = []
    rows = uni.refresh(prod, cfg, {}, venue_info=venue, dropped=dropped)
    syms = [r["symbol"] for r in rows]
    assert "BBBUSDT" not in syms and "CCCUSDT" not in syms and len(syms) == 2
    assert "BBBUSDT" in dropped
    assert [r["rank"] for r in rows] == [1, 2]


def test_universe_cycle_in_demo_drops_missing_symbols_without_crashing(store, tmp_path, caplog):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0})
    eng.cfg = eng.cfg.merge({"universe_size": 2})
    eng.universe_at = 0
    eng.universe = []
    data.exchange_info = lambda: {s: SymbolInfo(s, onboard_date_ms=0) for s in data.marks}
    with caplog.at_level(logging.WARNING, logger="tb.engine"):
        run(eng.universe_cycle())
    assert "BBBUSDT" not in eng.universe and eng.universe
    venue_syms = set(venue.marks)
    assert set(eng.universe) <= venue_syms
    if eng.dropped_venue:
        assert any("dropped" in r.message and r.message.count("USDT") for r in caplog.records)


def test_venue_outage_keeps_current_universe(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0})
    before = list(eng.universe)
    eng.universe_at = 0
    venue.info_fail = True
    run(eng.universe_cycle())
    assert eng.universe == before


def test_symbol_missing_on_venue_never_blocks_others(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0})
    assert not eng.venue_tradable("BBBUSDT") and eng.venue_tradable("AAAUSDT")
    open_(eng, "BBBUSDT")                                          # entry skipped quietly: no order, no exception
    assert store.get_positions() == [] and not [x for x in venue.log if x[0] == "market"]
    open_(eng, "AAAUSDT")
    assert [p["symbol"] for p in store.get_positions()] == ["AAAUSDT"]
    run(eng.bars_cycle())                                           # whole bar loop survives the missing symbol
    run(eng.prices_cycle())


def test_venue_symbol_not_trading_is_rejected(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0})
    base = venue.exchange_info
    venue.exchange_info = lambda: {s: SymbolInfo(s, status="BREAK" if s == "AAAUSDT" else "TRADING") for s in venue.marks}
    eng.ex = venue
    assert not eng.venue_tradable("AAAUSDT")


# ------------------------------------------------------------------ (2) order-venue prices and filters
def test_sizing_and_levels_use_venue_price_not_prod(store, tmp_path):
    # prod says 100, the testnet book says 200 (synthetic): the order must be sized at 200
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 200.0, "BTCUSDT": 100.0})
    open_(eng, "AAAUSDT")
    p = store.get_positions()[0]
    assert p["entry_price"] == pytest.approx(200.0)
    assert p["qty"] * p["entry_price"] == pytest.approx(160.0, rel=0.01)               # 0.8 qty, not 1.6
    assert (p["entry_price"] - p["sl_price"]) * p["qty"] == pytest.approx(10.0, abs=0.1)
    sl = [x for x in venue.log if x[0] == "SL"][0]
    tp = [x for x in venue.log if x[0] == "TP"][0]
    assert sl[3] < venue.marks["AAAUSDT"] < tp[3]                                       # right side of the VENUE mark
    assert p["mark_price"] == pytest.approx(200.0)


def test_atr_distance_scaled_to_venue_price_level(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 200.0, "BTCUSDT": 100.0},
                               settings={"exit_mode": "atr", "sl_atr": 2.0, "tp_atr": 3.0})
    open_(eng, "AAAUSDT")
    p = store.get_positions()[0]
    atr_pct_prod = closed_of(eng, "AAAUSDT")
    from tb.risk import atr14
    a = atr14(atr_pct_prod["high"].to_numpy(), atr_pct_prod["low"].to_numpy(), atr_pct_prod["close"].to_numpy())
    ref_prod = eng.marks["AAAUSDT"]
    expect_sl_dist = 2.0 * a * (200.0 / ref_prod)                                         # ATR in venue price units
    assert p["entry_price"] - p["sl_price"] == pytest.approx(expect_sl_dist, rel=0.01)


def test_venue_min_notional_filter_applies(store, tmp_path):
    # prod would accept (min 5) but the venue demands 500 notional: skipped, no order
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0}, min_notional=500.0)
    open_(eng, "AAAUSDT")
    assert store.get_positions() == [] and not [x for x in venue.log if x[0] == "market"]


def test_stop_recomputed_from_actual_fill_not_the_reference(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0})
    venue.fill_override = 101.0                                  # slippage on the testnet book
    open_(eng, "AAAUSDT")
    p = store.get_positions()[0]
    assert p["entry_price"] == 101.0
    assert 101.0 - p["sl_price"] == pytest.approx(10.0 / p["qty"], abs=0.02)       # stop measured from the FILL


def test_stop_on_wrong_side_of_venue_mark_is_never_sent(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0})
    venue.fill_override = 100.0
    venue.next_mark = 90.0                  # by the time we place the stop the venue mark is already below SL (93.75)
    open_(eng, "AAAUSDT")
    assert store.get_positions() == []                                              # never left unprotected
    assert not [x for x in venue.log if x[0] == "SL"]                                # the bad stop was not even sent
    assert venue.pos == {}                                                           # closed at market
    assert [x for x in venue.log if x[0] == "market"][-1][4] is True                 # reduce-only


def test_positions_marked_and_exits_use_venue_mark(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 200.0, "BTCUSDT": 100.0})
    open_(eng, "AAAUSDT")
    venue.marks["AAAUSDT"] = 210.0                                                   # venue moves, prod does not
    run(eng.prices_cycle())
    p = store.get_positions()[0]
    assert p["mark_price"] == pytest.approx(210.0) and p["upnl"] == pytest.approx((210.0 - 200.0) * p["qty"])
    # kill/PnL logic is driven by the venue account, not prod marks
    assert eng.equity == pytest.approx(venue.balance().equity)


def test_signals_keep_using_prod_data(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={"AAAUSDT": 100.0, "BTCUSDT": 100.0},
                               venue_marks={"AAAUSDT": 200.0, "BTCUSDT": 100.0})
    run(eng.update_symbol("AAAUSDT"))
    assert eng.kl["AAAUSDT"]["close"].iloc[-1] == pytest.approx(data.kl["AAAUSDT"]["close"].iloc[-1])   # prod candles
    assert eng.marks["AAAUSDT"] == pytest.approx(data.marks["AAAUSDT"])                                  # prod mark kept
    assert eng.vmarks["AAAUSDT"] == pytest.approx(200.0)
