"""Binance client: signing and order construction against a mocked HTTP session (no network)."""
import hashlib
import hmac
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from tb.exchange.base import ExchangeError, NetworkError, RateLimitError, SymbolFilters, SymbolInfo
from tb.exchange.binance import BinanceExchange, klines_to_frame, parse_exchange_info, scrub


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._b = body
        self.headers = headers or {}
        self.text = json.dumps(body)

    def json(self):
        if self._b is None:
            raise ValueError
        return self._b


class Sess:
    def __init__(self, *responses):
        self.calls = []
        self.queue = list(responses)

    def request(self, method, url, headers=None, params=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "params": params})
        r = self.queue.pop(0) if self.queue else Resp(200, {})
        if isinstance(r, Exception):
            raise r
        return r


def mk(*responses, **kw):
    s = Sess(*responses)
    ex = BinanceExchange("https://demo-fapi.binance.com", "KEY123", "SECRET456", mode="binance_demo", session=s, **kw)
    ex._offset_at = 10 ** 12          # skip the time sync call
    ex._info = {"ETHUSDT": SymbolInfo("ETHUSDT", filters=SymbolFilters(0.01, 0.001, 0.001, 20.0))}
    ex._info_at = 10 ** 12
    return ex, s


def qs(call):
    return {k: v[0] for k, v in parse_qs(urlsplit(call["url"]).query).items()}


def test_signature_is_hmac_sha256_of_query():
    ex, _ = mk()
    ex._offset_ms = 0
    q = ex.signed_query({"symbol": "ETHUSDT", "side": "BUY"})
    body, sig = q.rsplit("&signature=", 1)
    assert sig == hmac.new(b"SECRET456", body.encode(), hashlib.sha256).hexdigest()
    parsed = parse_qs(body)
    assert parsed["recvWindow"] == ["5000"] and "timestamp" in parsed


def test_known_binance_doc_vector():
    """HMAC vector from the Binance API docs (apiKey/secret are the documented examples)."""
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    payload = ("symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000"
               "&timestamp=1499827319559")
    assert hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest() == \
        "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"


def test_market_order_request_construction():
    ex, s = mk(Resp(200, {"orderId": 11, "status": "FILLED", "executedQty": "0.067", "avgPrice": "2364.5"}))
    res = ex.market_order("ETHUSDT", "buy", 0.06789, reduce_only=False)
    c = s.calls[0]
    assert c["method"] == "POST" and urlsplit(c["url"]).path == "/fapi/v1/order"
    assert c["headers"]["X-MBX-APIKEY"] == "KEY123"
    q = qs(c)
    assert q["symbol"] == "ETHUSDT" and q["side"] == "BUY" and q["type"] == "MARKET"
    assert q["quantity"] == "0.067"                         # floored to the lot step
    assert "reduceOnly" not in q and q["newOrderRespType"] == "RESULT" and q["newClientOrderId"].startswith("tb")
    assert "signature" in q and "SECRET456" not in c["url"]
    assert res.order_id == "o:11" and res.avg_price == 2364.5 and res.qty == pytest.approx(0.067)


def test_reduce_only_flag_and_min_qty():
    ex, s = mk(Resp(200, {"orderId": 1, "status": "FILLED", "executedQty": "1", "avgPrice": "1"}))
    ex.market_order("ETHUSDT", "SELL", 1.0, reduce_only=True)
    assert qs(s.calls[0])["reduceOnly"] == "true"
    with pytest.raises(ExchangeError):
        ex.market_order("ETHUSDT", "SELL", 0.0004)           # below min_qty after rounding


def test_stop_order_legacy_endpoint_params():
    ex, s = mk(Resp(200, {"orderId": 77}))
    oid = ex.place_stop("ETHUSDT", "SELL", 2215.555, 0.067)
    q = qs(s.calls[0])
    assert urlsplit(s.calls[0]["url"]).path == "/fapi/v1/order" and oid == "o:77"
    assert (q["type"], q["side"], q["stopPrice"], q["quantity"]) == ("STOP_MARKET", "SELL", "2215.56", "0.067")
    assert q["reduceOnly"] == "true" and q["workingType"] == "MARK_PRICE"


def test_algo_switch_fallback_on_minus_4120():
    ex, s = mk(Resp(400, {"code": -4120, "msg": "Order type not supported for this endpoint. Please use the Algo Order API endpoints instead."}),
               Resp(200, {"algoId": 555}), Resp(200, {"algoId": 556}))
    oid = ex.place_stop("ETHUSDT", "SELL", 2215.55, 0.067)
    assert oid == "a:555"
    assert [urlsplit(c["url"]).path for c in s.calls] == ["/fapi/v1/order", "/fapi/v1/algoOrder"]
    q = qs(s.calls[1])
    assert q["algoType"] == "CONDITIONAL" and q["type"] == "STOP_MARKET" and q["triggerPrice"] == "2215.55"
    assert "stopPrice" not in q and q["reduceOnly"] == "true" and q["workingType"] == "MARK_PRICE"
    assert q["clientAlgoId"].startswith("tb")
    # once switched, the legacy endpoint is not tried again
    ex.place_take_profit("ETHUSDT", "SELL", 2439.43, 0.067)
    assert urlsplit(s.calls[2]["url"]).path == "/fapi/v1/algoOrder" and qs(s.calls[2])["type"] == "TAKE_PROFIT_MARKET"


def test_other_errors_do_not_trigger_algo_switch():
    ex, s = mk(Resp(400, {"code": -2021, "msg": "Order would immediately trigger."}))
    with pytest.raises(ExchangeError) as e:
        ex.place_stop("ETHUSDT", "SELL", 2215.55, 0.067)
    assert e.value.code == -2021 and not ex._algo_active


def test_cancel_routes_by_id_namespace_and_ignores_unknown():
    ex, s = mk(Resp(200, {}), Resp(400, {"code": -2011, "msg": "Unknown order sent."}))
    ex.cancel_order("ETHUSDT", "a:555")
    ex.cancel_order("ETHUSDT", "o:12")           # already gone: swallowed
    assert s.calls[0]["method"] == "DELETE" and urlsplit(s.calls[0]["url"]).path == "/fapi/v1/algoOrder"
    assert qs(s.calls[0])["algoId"] == "555"
    assert urlsplit(s.calls[1]["url"]).path == "/fapi/v1/order" and qs(s.calls[1])["orderId"] == "12"


def test_timestamp_error_resyncs_and_retries():
    ex, s = mk(Resp(400, {"code": -1021, "msg": "Timestamp for this request is outside of the recvWindow."}),
               Resp(200, {"serverTime": 1_700_000_000_000}), Resp(200, [{"symbol": "X", "positionAmt": "0"}]))
    assert ex.positions() == []
    assert [urlsplit(c["url"]).path for c in s.calls] == ["/fapi/v1/time" if False else "/fapi/v2/positionRisk",
                                                           "/fapi/v1/time", "/fapi/v2/positionRisk"]


def test_429_raises_ratelimit_and_blocks_subsequent_calls():
    ex, s = mk(Resp(429, {"code": -1003, "msg": "Too many requests"}, {"Retry-After": "7"}))
    with pytest.raises(RateLimitError) as e:
        ex.klines("ETHUSDT")
    assert e.value.retry_after == 7
    with pytest.raises(RateLimitError):
        ex.klines("ETHUSDT")
    assert len(s.calls) == 1                      # second call short-circuited by the ban window


def test_get_retries_network_errors_but_post_does_not(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ex, s = mk(ConnectionError("dns"), ConnectionError("dns"), Resp(200, {"serverTime": 5}))
    assert ex.server_time_ms() == 5 and len(s.calls) == 3
    ex2, s2 = mk(ConnectionError("x"))
    with pytest.raises(NetworkError):
        ex2.set_leverage("ETHUSDT", 8)
    assert len(s2.calls) == 1


def test_market_order_network_error_looks_up_by_client_id(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ex, s = mk(ConnectionError("timeout"), Resp(200, {"orderId": 9, "status": "FILLED", "executedQty": "0.067", "avgPrice": "10"}))
    r = ex.market_order("ETHUSDT", "BUY", 0.067)
    assert r.order_id == "o:9"
    assert s.calls[1]["method"] == "GET" and "origClientOrderId" in qs(s.calls[1])
    assert qs(s.calls[1])["origClientOrderId"] == qs(s.calls[0])["newClientOrderId"]


def test_set_isolated_ignores_no_need_to_change():
    ex, _ = mk(Resp(400, {"code": -4046, "msg": "No need to change margin type."}))
    ex.set_isolated("ETHUSDT")


def test_weight_header_tracked():
    ex, _ = mk(Resp(200, [], {"X-MBX-USED-WEIGHT-1M": "123"}))
    ex.klines("ETHUSDT")
    assert ex._used_weight == 123


def test_requires_key_for_signed():
    ex = BinanceExchange("https://fapi.binance.com", session=Sess())
    with pytest.raises(ExchangeError):
        ex.balance()


def test_open_conditional_orders_merges_legacy_and_algo():
    ex, _ = mk(Resp(200, [{"orderId": 1, "symbol": "ETHUSDT", "type": "STOP_MARKET", "side": "SELL", "stopPrice": "2200",
                           "origQty": "0.067", "reduceOnly": True},
                          {"orderId": 2, "symbol": "ETHUSDT", "type": "LIMIT", "side": "SELL", "stopPrice": "0", "origQty": "1"}]),
               Resp(200, [{"algoId": 3, "symbol": "ETHUSDT", "orderType": "TAKE_PROFIT_MARKET", "side": "SELL",
                           "triggerPrice": "2400", "quantity": "0.067", "reduceOnly": True}]))
    o = ex.open_conditional_orders("ETHUSDT")
    assert [(x.order_id, x.kind, x.trigger_price) for x in o] == [("o:1", "SL", 2200.0), ("a:3", "TP", 2400.0)]


def test_last_close_aggregates_fills_and_reason():
    fills = [{"side": "BUY", "price": "100", "qty": "1", "realizedPnl": "0", "commission": "0.05", "commissionAsset": "USDT", "time": 1000},
             {"side": "SELL", "price": "103", "qty": "1", "realizedPnl": "3", "commission": "0.05", "commissionAsset": "USDT", "time": 5000}]
    ex, _ = mk(Resp(200, fills), Resp(200, [{"income": "-0.01"}]), Resp(200, {"algoStatus": "TRIGGERED"}))
    info = ex.last_close("ETHUSDT", "long", 500, sl_order_id=None, tp_order_id="a:3")
    assert info.exit_price == 103 and info.pnl == 3 and info.fees == pytest.approx(0.1)
    assert info.funding == pytest.approx(-0.01) and info.reason == "tp"


def test_parse_exchange_info_and_rounding():
    raw = {"symbols": [{"symbol": "ETHUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                        "onboardDate": 1569398400000,
                        "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "10000"},
                                    {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "8000"},
                                    {"filterType": "MIN_NOTIONAL", "notional": "20"}]}]}
    info = parse_exchange_info(raw)["ETHUSDT"]
    f = info.filters
    assert (f.tick_size, f.step_size, f.min_qty, f.min_notional, f.max_qty) == (0.01, 0.001, 0.001, 20.0, 8000.0)
    assert f.round_qty(0.0679) == 0.067 and f.qty_str(1.0) == "1" and f.price_str(2215.555) == "2215.56"
    tiny = SymbolFilters(tick_size=0.00001, step_size=1.0)
    assert tiny.price_str(0.0000123) == "0.00001" and tiny.qty_str(1234.9) == "1234"     # no scientific notation


def test_klines_frame_schema():
    raw = [[1700000000000, "1", "2", "0.5", "1.5", "10", 1700000299999, "15", 42, "4", "6", "0"]]
    df = klines_to_frame(raw)
    assert list(df.columns)[:6] == ["open_time", "open", "high", "low", "close", "volume"]
    assert df["open_time"].dtype == "int64" and df["close"].iloc[0] == 1.5 and df["trades"].iloc[0] == 42
    assert klines_to_frame([]).empty


def test_scrub_hides_signature():
    assert "abc123" not in scrub("GET /x?a=1&signature=abc123 failed")


def test_repr_never_shows_secret():
    ex, _ = mk()
    assert "SECRET456" not in repr(ex) and "KEY123" not in repr(ex)
