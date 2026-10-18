"""scripts/demo_smoke.py against a stateful mocked Binance (no network, no keys)."""
import importlib.util
import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tb.exchange.binance import BinanceExchange

SPEC = importlib.util.spec_from_file_location("demo_smoke", Path(__file__).resolve().parents[2] / "scripts" / "demo_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._b, self.headers, self.text = status, body, {}, json.dumps(body)

    def json(self):
        return self._b


class MockBinance:
    """Just enough of the futures API; STOP_MARKET on /order answers -4120 when algo_only is set."""

    def __init__(self, hedged=False, algo_only=True, fail_algo=False):
        self.hedged, self.algo_only, self.fail_algo = hedged, algo_only, fail_algo
        self.calls = []
        self.pos = 0.0
        self.entry = 0.0
        self.algos = {}
        self.regular = {}
        self._id = 1000

    def nid(self):
        self._id += 1
        return self._id

    def request(self, method, url, headers=None, params=None, timeout=None):
        u = urlsplit(url)
        path = u.path
        q = {k: v[0] for k, v in parse_qs(u.query).items()} if u.query else dict(params or {})
        self.calls.append((method, path, q))
        if headers and headers.get("X-MBX-APIKEY"):
            assert "signature" in q
        ok = lambda b: Resp(200, b)
        if path == "/fapi/v1/time":
            return ok({"serverTime": int(time.time() * 1000)})
        if path == "/fapi/v2/account":
            return ok({"totalWalletBalance": "5000", "availableBalance": "4900" if self.pos == 0 else "4890",
                       "totalUnrealizedProfit": "0"})
        if path == "/fapi/v1/positionSide/dual":
            return ok({"dualSidePosition": self.hedged})
        if path == "/fapi/v1/exchangeInfo":
            return ok({"symbols": [{"symbol": "ETHUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                                    "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                                                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "9000"},
                                                {"filterType": "MIN_NOTIONAL", "notional": "20"}]}]})
        if path == "/fapi/v1/premiumIndex":
            return ok({"symbol": "ETHUSDT", "markPrice": "2400.00", "indexPrice": "2400", "lastFundingRate": "0.0001",
                       "nextFundingTime": 0})
        if path == "/fapi/v2/positionRisk":
            if not self.pos:
                return ok([])
            return ok([{"symbol": "ETHUSDT", "positionAmt": str(self.pos), "entryPrice": str(self.entry), "markPrice": "2400",
                        "unRealizedProfit": "0", "leverage": "8", "liquidationPrice": "2100", "isolatedMargin": "10",
                        "marginType": "isolated"}])
        if path == "/fapi/v1/openOrders":
            return ok(list(self.regular.values()))
        if path == "/fapi/v1/openAlgoOrders":
            return ok(list(self.algos.values()))
        if path in ("/fapi/v1/marginType", "/fapi/v1/leverage"):
            return ok({})
        if path == "/fapi/v1/order" and method == "POST":
            if q["type"] == "MARKET":
                qty = float(q["quantity"])
                if q.get("reduceOnly") == "true":
                    self.pos = 0.0
                else:
                    self.pos, self.entry = qty, 2400.0
                return ok({"orderId": self.nid(), "status": "FILLED", "executedQty": q["quantity"], "avgPrice": "2400.00"})
            if self.algo_only:
                return Resp(400, {"code": -4120, "msg": "Order type not supported for this endpoint. Please use the Algo Order API endpoints instead."})
            oid = self.nid()
            self.regular[oid] = {"orderId": oid, "symbol": "ETHUSDT", "type": q["type"], "side": q["side"], "stopPrice": q["stopPrice"],
                                 "origQty": q["quantity"], "reduceOnly": True}
            return ok({"orderId": oid})
        if path == "/fapi/v1/algoOrder" and method == "POST":
            if self.fail_algo:
                return Resp(400, {"code": -1102, "msg": "Mandatory parameter missing"})
            aid = self.nid()
            self.algos[aid] = {"algoId": aid, "symbol": "ETHUSDT", "orderType": q["type"], "side": q["side"],
                               "triggerPrice": q["triggerPrice"], "quantity": q["quantity"], "reduceOnly": True}
            return ok({"algoId": aid})
        if method == "DELETE":
            if path == "/fapi/v1/algoOrder":
                self.algos.pop(int(q["algoId"]), None)
            elif path == "/fapi/v1/order":
                self.regular.pop(int(q["orderId"]), None)
            elif path == "/fapi/v1/allOpenOrders":
                self.regular.clear()
            elif path == "/fapi/v1/algoOpenOrders":
                self.algos.clear()
            return ok({})
        raise AssertionError(f"unexpected request {method} {path}")

    def mutating(self):
        return [c for c in self.calls if c[0] in ("POST", "DELETE", "PUT")]


def mk(**kw):
    m = MockBinance(**kw)
    ex = BinanceExchange("https://testnet.binancefuture.com", "THEKEY123", "THESECRET456", mode="binance_demo", session=m)
    return ex, m


def test_dry_run_is_read_only_and_prints_no_keys(capsys):
    ex, m = mk()
    assert smoke.main(["--dry-run"], exchange=ex) == 0
    out = capsys.readouterr().out
    assert m.mutating() == []                                     # not a single state-changing call
    assert "one-way" in out and "wallet 5000.00" in out and "ETHUSDT mark 2400" in out and "min_notional 20" in out
    assert "THEKEY123" not in out and "THESECRET456" not in out and "signature" not in out
    assert "testnet.binancefuture.com" in out


def test_full_round_trip_with_algo_fallback(capsys):
    ex, m = mk(algo_only=True)
    assert smoke.main([], exchange=ex) == 0
    out = capsys.readouterr().out
    posts = [(c[1], c[2]) for c in m.calls if c[0] == "POST"]
    paths = [p for p, _ in posts]
    assert paths[:2] == ["/fapi/v1/marginType", "/fapi/v1/leverage"]
    assert dict(posts)["/fapi/v1/marginType"]["marginType"] == "ISOLATED" and dict(posts)["/fapi/v1/leverage"]["leverage"] == "8"
    entry = posts[2][1]
    # smallest valid entry: min_notional 20 * 1.1 / 2400 = 0.00917 -> 0.010 (step 0.001)
    assert entry["type"] == "MARKET" and entry["side"] == "BUY" and entry["quantity"] == "0.01" and "reduceOnly" not in entry
    algo = [q for p, q in posts if p == "/fapi/v1/algoOrder"]
    assert [a["type"] for a in algo] == ["STOP_MARKET", "TAKE_PROFIT_MARKET"]
    assert algo[0]["triggerPrice"] == "2280" and algo[1]["triggerPrice"] == "2520"
    assert all(a["reduceOnly"] == "true" and a["workingType"] == "MARK_PRICE" and a["algoType"] == "CONDITIONAL" for a in algo)
    assert "algo endpoint" in out and "found ['SL', 'TP']" in out
    # cleaned up: orders cancelled, position closed reduce-only, nothing left
    assert m.pos == 0 and not m.algos and not m.regular
    closing = [q for p, q in posts if p == "/fapi/v1/order" and q["type"] == "MARKET"][-1]
    assert closing["side"] == "SELL" and closing["reduceOnly"] == "true"
    assert "all steps passed" in out and "THEKEY123" not in out and "THESECRET456" not in out


def test_regular_endpoint_when_no_algo_switch(capsys):
    ex, m = mk(algo_only=False)
    assert smoke.main([], exchange=ex) == 0
    assert "regular order endpoint" in capsys.readouterr().out and m.pos == 0 and not m.regular


def test_failure_after_entry_still_closes_position(capsys):
    ex, m = mk(algo_only=True, fail_algo=True)
    assert smoke.main([], exchange=ex) == 1
    out = capsys.readouterr().out
    assert "[FAIL] order flow" in out and m.pos == 0                 # the entry was unwound
    assert "RESULT: 1 step(s) FAILED" in out or "FAILED" in out


def test_hedge_mode_aborts_before_any_order(capsys):
    ex, m = mk(hedged=True)
    assert smoke.main([], exchange=ex) == 1
    out = capsys.readouterr().out
    assert "HEDGE" in out and m.mutating() == []


def test_refuses_real_money_host():
    ex = BinanceExchange("https://fapi.binance.com", "k", "s", session=MockBinance())
    assert smoke.main(["--dry-run"], exchange=ex) == 2


def test_missing_keys_exit_code(monkeypatch, capsys):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    monkeypatch.setattr(smoke, "load_secrets", lambda: type("S", (), {"has_binance": False})())
    assert smoke.main(["--dry-run"]) == 2


def test_smallest_valid_qty():
    from tb.exchange.base import SymbolFilters
    f = SymbolFilters(0.01, 0.001, 0.001, 20.0)
    assert smoke.smallest_valid_qty(2400.0, f) == 0.01
    assert smoke.smallest_valid_qty(10.0, f) == pytest.approx(2.2)
    assert smoke.smallest_valid_qty(100000.0, SymbolFilters(0.1, 0.001, 0.001, 5.0)) == 0.001
