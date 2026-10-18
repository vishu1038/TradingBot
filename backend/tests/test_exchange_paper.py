import time

import pytest

from conftest import FakeData
from tb.config import Settings
from tb.exchange.base import ExchangeError
from tb.exchange.paper import PaperExchange

CFG = Settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)   # clean maths


def mk(cfg=CFG, price=100.0):
    d = FakeData(("AAAUSDT",))
    d.marks["AAAUSDT"] = price
    p = PaperExchange(d, cfg, 100.0)
    p.marks["AAAUSDT"] = price
    return p


def test_market_fill_fee_and_margin():
    p = mk()
    r = p.market_order("AAAUSDT", "BUY", 1.6)
    assert r.avg_price == 100.0 and r.qty == pytest.approx(1.6)
    fee = 1.6 * 100 * 0.0005
    assert r.fee == pytest.approx(fee)
    b = p.balance()
    assert b.wallet == pytest.approx(100 - fee)
    assert b.available == pytest.approx(100 - fee - 20.0)        # 160 notional / 8x
    pos = p.positions()[0]
    assert pos.side == "long" and pos.qty == pytest.approx(1.6) and pos.liq_price == pytest.approx(88.0)


def test_slippage_applied_to_market_fills():
    cfg = Settings().merge({"slippage_pct": 0.001}, allow_internal=True)
    p = mk(cfg)
    assert p.market_order("AAAUSDT", "BUY", 1.0).avg_price == pytest.approx(100.1)
    assert p.market_order("AAAUSDT", "SELL", 1.0, reduce_only=True).avg_price == pytest.approx(99.9)


def test_insufficient_margin_rejected():
    p = mk()
    with pytest.raises(ExchangeError) as e:
        p.market_order("AAAUSDT", "BUY", 10.0)      # 1000 notional / 8 = 125 margin > 100
    assert e.value.code == -2019


def test_round_trip_pnl_and_fees():
    p = mk()
    p.market_order("AAAUSDT", "BUY", 1.6)
    p.tick({"AAAUSDT": 103.125})
    p.marks["AAAUSDT"] = 103.125
    r = p.market_order("AAAUSDT", "SELL", 1.6, reduce_only=True)
    info = p.last_close("AAAUSDT", "long", 0)
    assert info.pnl == pytest.approx(5.0)                         # +3.125% on 160 = +5
    assert info.fees == pytest.approx(1.6 * 100 * 0.0005 + 1.6 * 103.125 * 0.0005)
    assert p.positions() == []
    assert p.balance().wallet == pytest.approx(100 + 5.0 - info.fees)


def test_stop_triggers_on_tick_and_cancels_tp():
    p = mk()
    p.market_order("AAAUSDT", "BUY", 1.6)
    sl = p.place_stop("AAAUSDT", "SELL", 93.75, 1.6)
    p.place_take_profit("AAAUSDT", "SELL", 103.125, 1.6)
    assert len(p.open_conditional_orders()) == 2
    assert p.tick({"AAAUSDT": 94.0}) == []
    ev = p.tick({"AAAUSDT": 93.7})
    assert len(ev) == 1 and ev[0][1].reason == "sl"
    assert ev[0][1].exit_price == pytest.approx(93.7)             # filled at min(stop, price) = 93.7 (gapped through)
    assert ev[0][1].pnl == pytest.approx((93.7 - 100) * 1.6)
    assert p.open_conditional_orders() == [] and p.positions() == []
    assert sl not in p.conds


def test_tp_fill_at_trigger():
    p = mk()
    p.market_order("AAAUSDT", "BUY", 1.6)
    p.place_take_profit("AAAUSDT", "SELL", 103.125, 1.6)      # rounded to the 0.01 tick -> 103.13
    ev = p.tick({"AAAUSDT": 103.5})
    assert ev[0][1].reason == "tp" and ev[0][1].exit_price == pytest.approx(103.13)
    assert ev[0][1].pnl == pytest.approx(5.008)


def test_stop_slippage_on_stop_fills():
    cfg = Settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.001}, allow_internal=True)
    p = mk(cfg)
    p.market_order("AAAUSDT", "BUY", 1.6)
    p.place_stop("AAAUSDT", "SELL", 93.75, 1.6)
    ev = p.tick({"AAAUSDT": 93.75})
    assert ev[0][1].exit_price == pytest.approx(93.75 * 0.999)


def test_sl_wins_when_both_inside_one_range_replay():
    p = mk()
    p.market_order("AAAUSDT", "BUY", 1.6)
    p.place_stop("AAAUSDT", "SELL", 93.75, 1.6)
    p.place_take_profit("AAAUSDT", "SELL", 103.125, 1.6)
    t_after = p.pos["AAAUSDT"].opened_ms + 1000
    ev = p.check_range("AAAUSDT", low=93.0, high=104.0, bar_open_ms=t_after)   # both touched in the bar
    assert ev.reason == "sl"


def test_range_replay_ignores_bars_that_began_before_entry():
    p = mk()
    p.market_order("AAAUSDT", "BUY", 1.6)
    p.place_stop("AAAUSDT", "SELL", 93.75, 1.6)
    assert p.check_range("AAAUSDT", 90.0, 101.0, p.pos["AAAUSDT"].opened_ms - 1) is None
    assert p.positions()


def test_immediate_trigger_rejected():
    p = mk()
    p.market_order("AAAUSDT", "BUY", 1.6)
    with pytest.raises(ExchangeError) as e:
        p.place_stop("AAAUSDT", "SELL", 101.0, 1.6)             # SELL stop above the market
    assert e.value.code == -2021


def test_liquidation_loses_margin():
    p = mk()
    p.market_order("AAAUSDT", "BUY", 1.6)
    ev = p.tick({"AAAUSDT": 80.0})
    info = ev[0][1]
    assert info.reason == "liquidation" and info.exit_price == pytest.approx(88.0)
    # equity lost ~ the isolated margin (20), never more
    assert p.balance().wallet == pytest.approx(100 - 20.0 - 1.6 * 100 * 0.0005, abs=0.2)
    assert p.balance().wallet > 70


def test_short_liquidation_price_and_pnl():
    p = mk()
    p.market_order("AAAUSDT", "SELL", 1.6)
    pos = p.positions()[0]
    assert pos.side == "short" and pos.liq_price == pytest.approx(112.0)
    p.marks["AAAUSDT"] = 96.875
    assert p.positions()[0].upnl == pytest.approx(5.0)


def test_funding_long_pays_positive_rate():
    p = mk()
    p.market_order("AAAUSDT", "BUY", 1.6)
    w0 = p.balance().wallet
    now = time.time() + 9 * 3600
    assert p.funding_due(now)
    paid = p.apply_funding({"AAAUSDT": 0.001}, now)
    assert paid["AAAUSDT"] == pytest.approx(-1.6 * 100 * 0.001)
    assert p.balance().wallet == pytest.approx(w0 - 0.16)
    assert p.apply_funding({"AAAUSDT": 0.001}, now) == {}       # already applied for this slot
    info = (p.market_order("AAAUSDT", "SELL", 1.6, True), p.last_close("AAAUSDT", "long", 0))[1]
    assert info.funding == pytest.approx(-0.16)


def test_state_roundtrip():
    p = mk()
    p.market_order("AAAUSDT", "BUY", 1.6)
    p.place_stop("AAAUSDT", "SELL", 93.75, 1.6)
    st = p.export_state()
    import json
    q = mk()
    q.import_state(json.loads(json.dumps(st)))
    assert q.balance().wallet == pytest.approx(p.balance().wallet)
    assert q.positions()[0].qty == pytest.approx(1.6)
    assert len(q.open_conditional_orders()) == 1


def test_reduce_only_without_position_rejected():
    p = mk()
    with pytest.raises(ExchangeError):
        p.market_order("AAAUSDT", "SELL", 1.0, reduce_only=True)
