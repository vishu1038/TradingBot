"""Binance USDⓈ-M futures REST client (public + signed).

* ``requests`` only (never urllib: certifi works, the macOS python.org store does not here).
* HMAC-SHA256 signing, recvWindow, server-time offset sync (re-sync on -1021), weight-aware
  throttling (X-MBX-USED-WEIGHT-1M), Retry-After on 429/418.
* Quantity / price rounding from exchangeInfo (tick, lot step, min qty, min notional).
* Conditional orders (STOP_MARKET / TAKE_PROFIT_MARKET): Binance migrated these to the Algo Order
  service. ``conditional_mode`` = "auto" (default) tries ``/fapi/v1/order`` first and switches to
  ``/fapi/v1/algoOrder`` for good when the legacy endpoint answers -4120 (STOP_ORDER_SWITCH_ALGO),
  "algo" always uses the algo endpoints, "legacy" never does. Order ids are namespaced strings:
  ``o:<orderId>`` for plain orders and ``a:<algoId>`` for algo orders.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from .base import (KLINE_COLUMNS, Balance, CloseInfo, ConditionalOrder, Exchange, ExchangeError,
                   NetworkError, OrderResult, Premium, PositionInfo, RateLimitError, SymbolFilters,
                   SymbolInfo, Ticker24h, opposite)

log = logging.getLogger("tb.binance")

ALGO_SWITCH_CODE = -4120
POST_ONLY_REJECT = -5022           # GTX order that would have executed as taker
RECV_WINDOW = 5000
WEIGHT_SOFT_LIMIT = 2000          # of 2400 / minute
_SIG_RE = re.compile(r"signature=[0-9a-fA-F]+")


def scrub(text: str) -> str:
    return _SIG_RE.sub("signature=***", str(text))


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


class BinanceExchange(Exchange):
    name = "binance"

    def __init__(self, base_url: str, api_key: str = "", api_secret: str = "", *, mode: str = "paper_local",
                 session: Any = None, timeout: float = 10.0, conditional_mode: str = "auto"):
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self._secret = api_secret
        self.mode = mode
        self.timeout = timeout
        self.conditional_mode = conditional_mode
        self._algo_active = conditional_mode == "algo"
        if session is None:
            import requests

            session = requests.Session()
        self.s = session
        self._offset_ms = 0
        self._offset_at = 0.0
        self._lock = threading.Lock()
        self._used_weight = 0
        self._weight_minute = 0
        self._info: dict[str, SymbolInfo] = {}
        self._info_at = 0.0
        self._ban_until = 0.0

    def __repr__(self) -> str:
        return f"BinanceExchange(base={self.base_url}, mode={self.mode}, keys={'set' if self._key else 'unset'})"

    # ------------------------------------------------------------------ transport
    def _now_ms(self) -> int:
        return int(time.time() * 1000) + self._offset_ms

    def sync_time(self) -> int:
        t0 = time.time()
        srv = int(self._request("GET", "/fapi/v1/time", {}, signed=False)["serverTime"])
        t1 = time.time()
        self._offset_ms = int(srv - (t0 + t1) / 2 * 1000)
        self._offset_at = t1
        return self._offset_ms

    def server_time_ms(self) -> int:
        return int(self._request("GET", "/fapi/v1/time", {}, signed=False)["serverTime"])

    def signed_query(self, params: dict[str, Any]) -> str:
        """Query string with timestamp, recvWindow and the HMAC signature appended last."""
        p = {k: _fmt(v) for k, v in params.items() if v is not None}
        p["recvWindow"] = str(RECV_WINDOW)
        p["timestamp"] = str(self._now_ms())
        qs = urlencode(p)
        sig = hmac.new(self._secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        return f"{qs}&signature={sig}"

    def _throttle(self) -> None:
        now = time.time()
        if now < self._ban_until:
            raise RateLimitError(f"banned/limited for {self._ban_until - now:.0f}s", retry_after=self._ban_until - now)
        minute = int(now // 60)
        if minute != self._weight_minute:
            self._weight_minute = minute
            self._used_weight = 0
        if self._used_weight >= WEIGHT_SOFT_LIMIT:
            wait = 60 - (now % 60) + 0.2
            log.warning("request weight %d near limit; sleeping %.1fs", self._used_weight, wait)
            time.sleep(wait)

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, *, signed: bool = False,
                 retries: int | None = None, _resynced: bool = False) -> Any:
        params = params or {}
        if signed and not self._key:
            raise ExchangeError("API key required for signed request")
        if signed and (time.time() - self._offset_at > 600):
            try:
                self.sync_time()
            except ExchangeError as e:
                log.warning("time sync failed: %s", e)
        attempts = (3 if method == "GET" else 1) if retries is None else retries
        last: Exception | None = None
        for i in range(attempts):
            self._throttle()
            try:
                if signed:
                    url = f"{self.base_url}{path}?{self.signed_query(params)}"
                    headers = {"X-MBX-APIKEY": self._key}
                    r = self.s.request(method, url, headers=headers, timeout=self.timeout)
                else:
                    clean = {k: _fmt(v) for k, v in params.items() if v is not None}
                    r = self.s.request(method, f"{self.base_url}{path}", params=clean, timeout=self.timeout)
            except Exception as e:  # requests.RequestException and friends
                last = NetworkError(f"{type(e).__name__} on {method} {path}: {scrub(e)}")
                log.warning("%s", last)
                time.sleep(min(0.5 * 2 ** i, 4))
                continue
            try:
                w = r.headers.get("X-MBX-USED-WEIGHT-1M")
                if w:
                    self._used_weight = max(self._used_weight, int(w))
            except Exception:
                pass
            status = r.status_code
            if status == 200:
                try:
                    return r.json()
                except ValueError:
                    raise NetworkError(f"non-JSON 200 from {path}") from None
            body: dict = {}
            try:
                body = r.json()
            except ValueError:
                pass
            code = body.get("code") if isinstance(body, dict) else None
            msg = body.get("msg") if isinstance(body, dict) else (r.text[:200] if hasattr(r, "text") else "")
            if status in (429, 418):
                ra = float(r.headers.get("Retry-After", 5 if status == 429 else 60))
                self._ban_until = time.time() + ra
                raise RateLimitError(f"HTTP {status} on {path}: {msg}", retry_after=ra, code=code, status=status)
            if code == -1021 and signed and not _resynced:  # timestamp outside recvWindow
                self.sync_time()
                return self._request(method, path, params, signed=signed, retries=retries, _resynced=True)
            if status >= 500:
                last = NetworkError(f"HTTP {status} on {method} {path}: {msg}", code=code, status=status)
                if method != "GET":
                    raise last   # order state unknown; caller must reconcile, never blind-retry
                time.sleep(min(0.5 * 2 ** i, 4))
                continue
            raise ExchangeError(f"HTTP {status} on {method} {path}: code={code} {msg}", code=code, status=status)
        assert last is not None
        raise last

    # ------------------------------------------------------------------ market data
    def klines(self, symbol: str, interval: str = "5m", limit: int = 500, start_ms: int | None = None,
               end_ms: int | None = None) -> pd.DataFrame:
        raw = self._request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit,
                                                       "startTime": start_ms, "endTime": end_ms})
        return klines_to_frame(raw)

    def tickers_24h(self) -> list[Ticker24h]:
        raw = self._request("GET", "/fapi/v1/ticker/24hr")
        return [Ticker24h(r["symbol"], float(r["lastPrice"]), float(r["quoteVolume"]),
                          float(r["priceChangePercent"]), float(r.get("highPrice", 0)), float(r.get("lowPrice", 0)))
                for r in raw]

    def exchange_info(self) -> dict[str, SymbolInfo]:
        if self._info and time.time() - self._info_at < 3600:
            return self._info
        raw = self._request("GET", "/fapi/v1/exchangeInfo")
        self._info = parse_exchange_info(raw)
        self._info_at = time.time()
        return self._info

    def premium(self, symbol: str | None = None) -> list[Premium]:
        raw = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol} if symbol else {})
        rows = raw if isinstance(raw, list) else [raw]
        out = []
        for r in rows:
            out.append(Premium(r["symbol"], float(r["markPrice"]), float(r.get("indexPrice") or 0),
                               float(r.get("lastFundingRate") or 0), int(r.get("nextFundingTime") or 0)))
        return out

    def open_interest(self, symbol: str) -> float:
        return float(self._request("GET", "/fapi/v1/openInterest", {"symbol": symbol})["openInterest"])

    def open_interest_hist(self, symbol: str, period: str = "5m", limit: int = 13) -> list[tuple[int, float]]:
        raw = self._request("GET", "/futures/data/openInterestHist",
                            {"symbol": symbol, "period": period, "limit": limit})
        return [(int(r["timestamp"]), float(r["sumOpenInterestValue"])) for r in raw]

    def long_short_ratio(self, symbol: str, period: str = "5m") -> float | None:
        raw = self._request("GET", "/futures/data/globalLongShortAccountRatio",
                            {"symbol": symbol, "period": period, "limit": 1})
        return float(raw[-1]["longShortRatio"]) if raw else None

    def depth(self, symbol: str, limit: int = 1000):
        raw = self._request("GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit})   # weight 20 at limit=1000
        return ([(float(p), float(q)) for p, q in raw["bids"]], [(float(p), float(q)) for p, q in raw["asks"]])

    @property
    def used_weight(self) -> int:
        """Request weight used in the current minute (as last reported by the exchange)."""
        return self._used_weight if int(time.time() // 60) == self._weight_minute else 0

    # ------------------------------------------------------------------ trading
    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": int(leverage)}, signed=True)

    def set_isolated(self, symbol: str) -> None:
        try:
            self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"}, signed=True)
        except ExchangeError as e:
            if e.code != -4046:        # "No need to change margin type"
                raise

    def market_order_params(self, symbol: str, side: str, qty: float, reduce_only: bool,
                            client_id: str | None = None) -> dict[str, Any]:
        f = self.filters(symbol)
        q = f.round_qty(qty)
        if q < f.min_qty or q <= 0:
            raise ExchangeError(f"qty {qty} below min_qty {f.min_qty} for {symbol}")
        p: dict[str, Any] = {"symbol": symbol, "side": side.upper(), "type": "MARKET", "quantity": f.qty_str(qty),
                             "newOrderRespType": "RESULT",
                             "newClientOrderId": client_id or f"tb{uuid.uuid4().hex[:24]}"}
        if reduce_only:
            p["reduceOnly"] = True
        return p

    def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> OrderResult:
        params = self.market_order_params(symbol, side, qty, reduce_only)
        try:
            r = self._request("POST", "/fapi/v1/order", params, signed=True)
        except NetworkError:
            # outcome unknown: look the order up by our idempotency id before giving up
            r = self._request("GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": params["newClientOrderId"]},
                              signed=True)
        if r.get("status") not in ("FILLED", "PARTIALLY_FILLED"):
            time.sleep(0.3)
            r = self._request("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": r["orderId"]}, signed=True)
        return OrderResult(f"o:{r['orderId']}", symbol, side.upper(), float(r.get("executedQty") or qty),
                           float(r.get("avgPrice") or 0), r.get("status", ""))

    # ---- maker (post-only) entries
    def best_bid_ask(self, symbol: str) -> tuple[float, float]:
        r = self._request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})        # weight 2
        if isinstance(r, list):
            r = next((x for x in r if x.get("symbol") == symbol), {})
        bid, ask = float(r.get("bidPrice") or 0), float(r.get("askPrice") or 0)
        if bid <= 0 or ask <= 0:
            raise ExchangeError(f"no book for {symbol}")
        return bid, ask

    @staticmethod
    def _order_result(r: dict, symbol: str, side: str, price: float = 0.0) -> OrderResult:
        ex = float(r.get("executedQty") or 0)
        return OrderResult(f"o:{r['orderId']}", symbol, str(r.get("side") or side).upper(), float(r.get("origQty") or ex),
                           float(r.get("avgPrice") or 0), str(r.get("status", "")), 0.0, ex,
                           float(r.get("price") or price or 0))

    def limit_order(self, symbol: str, side: str, qty: float, price: float, post_only: bool = True,
                    reduce_only: bool = False) -> OrderResult:
        f = self.filters(symbol)
        q = f.round_qty(qty)
        if q < f.min_qty or q <= 0:
            raise ExchangeError(f"qty {qty} below min_qty {f.min_qty} for {symbol}")
        cid = f"tbm{uuid.uuid4().hex[:23]}"           # tbm... = maker entry (stale ones are cancelled at engine start)
        p: dict[str, Any] = {"symbol": symbol, "side": side.upper(), "type": "LIMIT", "quantity": f.qty_str(qty),
                             "price": f.price_str(price), "timeInForce": "GTX" if post_only else "GTC",
                             "newOrderRespType": "RESULT", "newClientOrderId": cid}
        if reduce_only:
            p["reduceOnly"] = True
        try:
            r = self._request("POST", "/fapi/v1/order", p, signed=True)
        except NetworkError:
            r = self._request("GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": cid}, signed=True)
        res = self._order_result(r, symbol, side, price)
        if post_only and res.status == "EXPIRED" and res.filled <= 0:      # a GTX that would have taken: expired unfilled
            raise ExchangeError("Post Only order rejected (would take liquidity)", code=POST_ONLY_REJECT)
        return res

    def get_order(self, symbol: str, order_id: str) -> OrderResult:
        _, _, oid = order_id.partition(":")
        r = self._request("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": oid or order_id}, signed=True)
        return self._order_result(r, symbol, "")

    def open_entry_orders(self, symbol: str | None = None) -> list[tuple[str, str]]:
        out = []
        for r in self._request("GET", "/fapi/v1/openOrders", {"symbol": symbol} if symbol else {}, signed=True):
            if r.get("type") == "LIMIT" and not r.get("reduceOnly") \
                    and str(r.get("clientOrderId", "")).startswith("tbm"):
                out.append((r["symbol"], f"o:{r['orderId']}"))
        return out

    def conditional_params(self, kind: str, symbol: str, side: str, price: float, qty: float,
                           algo: bool) -> dict[str, Any]:
        f = self.filters(symbol)
        typ = "STOP_MARKET" if kind == "SL" else "TAKE_PROFIT_MARKET"
        p: dict[str, Any] = {"symbol": symbol, "side": side.upper(), "type": typ, "quantity": f.qty_str(qty),
                             "reduceOnly": True, "workingType": "MARK_PRICE"}
        if algo:
            p["algoType"] = "CONDITIONAL"
            p["triggerPrice"] = f.price_str(price)
            p["clientAlgoId"] = f"tb{uuid.uuid4().hex[:24]}"
        else:
            p["stopPrice"] = f.price_str(price)
            p["newClientOrderId"] = f"tb{uuid.uuid4().hex[:24]}"
        return p

    def _place_conditional(self, kind: str, symbol: str, side: str, price: float, qty: float) -> str:
        if not self._algo_active and self.conditional_mode != "algo":
            try:
                r = self._request("POST", "/fapi/v1/order", self.conditional_params(kind, symbol, side, price, qty, False),
                                  signed=True)
                return f"o:{r['orderId']}"
            except ExchangeError as e:
                if e.code != ALGO_SWITCH_CODE or self.conditional_mode == "legacy":
                    raise
                log.warning("legacy conditional order rejected with %s; switching to the algo order endpoint", e.code)
                self._algo_active = True
        try:
            r = self._request("POST", "/fapi/v1/algoOrder", self.conditional_params(kind, symbol, side, price, qty, True),
                              signed=True)
        except ExchangeError as e:
            if e.status == 404 and self.conditional_mode == "auto":   # host without the algo service
                self._algo_active = False
                self.conditional_mode = "legacy"
                log.warning("algo endpoint 404; falling back to legacy conditional orders")
                return self._place_conditional(kind, symbol, side, price, qty)
            raise
        return f"a:{r['algoId']}"

    def place_stop(self, symbol: str, side: str, stop_price: float, qty: float) -> str:
        return self._place_conditional("SL", symbol, side, stop_price, qty)

    def place_take_profit(self, symbol: str, side: str, tp_price: float, qty: float) -> str:
        return self._place_conditional("TP", symbol, side, tp_price, qty)

    @staticmethod
    def _gone(e: ExchangeError) -> bool:
        return e.code in (-2011, -2013) or "unknown order" in str(e).lower() or "does not exist" in str(e).lower()

    def cancel_order(self, symbol: str, order_id: str) -> None:
        kind, _, oid = order_id.partition(":")
        try:
            if kind == "a":
                self._request("DELETE", "/fapi/v1/algoOrder", {"algoId": oid}, signed=True)
            else:
                self._request("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": oid}, signed=True)
        except ExchangeError as e:
            if not self._gone(e):
                raise

    def cancel_all(self, symbol: str) -> None:
        try:
            self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
        except ExchangeError as e:
            if not self._gone(e):
                raise
        if self.conditional_mode != "legacy":
            try:
                self._request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol}, signed=True)
            except ExchangeError as e:
                if self._algo_active and not self._gone(e):
                    raise
                log.debug("algoOpenOrders cancel ignored: %s", e)

    def open_conditional_orders(self, symbol: str | None = None) -> list[ConditionalOrder]:
        out: list[ConditionalOrder] = []
        p = {"symbol": symbol} if symbol else {}
        for r in self._request("GET", "/fapi/v1/openOrders", p, signed=True):
            if r.get("type") in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                out.append(ConditionalOrder(f"o:{r['orderId']}", r["symbol"], "SL" if r["type"] == "STOP_MARKET" else "TP",
                                            r["side"], float(r["stopPrice"]), float(r["origQty"]),
                                            bool(r.get("reduceOnly") or r.get("closePosition"))))
        if self.conditional_mode != "legacy":
            try:
                for r in self._request("GET", "/fapi/v1/openAlgoOrders", p, signed=True) or []:
                    ot = r.get("orderType") or r.get("type")
                    if ot in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                        out.append(ConditionalOrder(f"a:{r['algoId']}", r["symbol"], "SL" if ot == "STOP_MARKET" else "TP",
                                                    r["side"], float(r["triggerPrice"]), float(r.get("quantity") or 0),
                                                    bool(r.get("reduceOnly") or r.get("closePosition"))))
            except ExchangeError as e:
                if self._algo_active:
                    raise
                log.debug("openAlgoOrders unavailable: %s", e)
        return out

    def positions(self) -> list[PositionInfo]:
        out = []
        for r in self._request("GET", "/fapi/v2/positionRisk", {}, signed=True):
            amt = float(r["positionAmt"])
            if amt == 0:
                continue
            out.append(PositionInfo(r["symbol"], "long" if amt > 0 else "short", abs(amt), float(r["entryPrice"]),
                                    float(r.get("markPrice") or 0), float(r.get("unRealizedProfit") or 0),
                                    int(float(r.get("leverage") or 1)), float(r.get("liquidationPrice") or 0),
                                    float(r.get("isolatedMargin") or 0), r.get("marginType", "isolated").lower() == "isolated"))
        return out

    def leverage_brackets(self, symbol: str):
        """Signed ``GET /fapi/v1/leverageBracket``; tiers for ``symbol`` (None if the venue lists none)."""
        from ..risk import parse_brackets

        raw = self._request("GET", "/fapi/v1/leverageBracket", {"symbol": symbol}, signed=True)
        return parse_brackets(raw).get(symbol)

    def position_mode_hedged(self) -> bool:
        return bool(self._request("GET", "/fapi/v1/positionSide/dual", {}, signed=True).get("dualSidePosition"))

    def balance(self) -> Balance:
        a = self._request("GET", "/fapi/v2/account", {}, signed=True)
        return Balance(float(a["totalWalletBalance"]), float(a["availableBalance"]), float(a["totalUnrealizedProfit"]))

    def user_trades(self, symbol: str, since_ms: int) -> list[dict]:
        return self._request("GET", "/fapi/v1/userTrades", {"symbol": symbol, "startTime": since_ms, "limit": 100},
                             signed=True)

    def funding_income(self, symbol: str, since_ms: int) -> float:
        rows = self._request("GET", "/fapi/v1/income", {"symbol": symbol, "incomeType": "FUNDING_FEE",
                                                        "startTime": since_ms, "limit": 100}, signed=True)
        return sum(float(r["income"]) for r in rows)

    def _order_triggered(self, symbol: str, order_id: str | None) -> bool:
        if not order_id:
            return False
        kind, _, oid = order_id.partition(":")
        try:
            if kind == "a":
                r = self._request("GET", "/fapi/v1/algoOrder", {"algoId": oid}, signed=True)
                return str(r.get("algoStatus", "")).upper() in ("TRIGGERED", "FINISHED", "FILLED")
            r = self._request("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": oid}, signed=True)
            return r.get("status") == "FILLED"
        except ExchangeError as e:
            log.warning("order state lookup failed for %s: %s", order_id, e)
            return False

    def last_close(self, symbol: str, side: str, since_ms: int, sl_order_id: str | None = None,
                   tp_order_id: str | None = None, entry_price: float = 0.0) -> CloseInfo | None:
        fills = self.user_trades(symbol, since_ms)
        close_side = "SELL" if side == "long" else "BUY"
        closing = [f for f in fills if f["side"] == close_side and int(f["time"]) >= since_ms]
        if not closing:
            return None
        qty = sum(float(f["qty"]) for f in closing)
        px = sum(float(f["price"]) * float(f["qty"]) for f in closing) / qty
        pnl = sum(float(f["realizedPnl"]) for f in fills if int(f["time"]) >= since_ms)
        fees = sum(float(f["commission"]) for f in fills if int(f["time"]) >= since_ms
                   and f.get("commissionAsset", "USDT") in ("USDT", "USDC", "BUSD"))
        try:
            funding = self.funding_income(symbol, since_ms)
        except ExchangeError:
            funding = 0.0
        if self._order_triggered(symbol, sl_order_id):
            reason = "sl"
        elif self._order_triggered(symbol, tp_order_id):
            reason = "tp"
        else:
            reason = "manual"
        return CloseInfo(px, pnl, fees, funding, reason, int(max(int(f["time"]) for f in closing) / 1000))


def klines_to_frame(raw: list[list[Any]]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in KLINE_COLUMNS}).astype({"open_time": "int64"})
    df = pd.DataFrame(raw).iloc[:, :11]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
                  "taker_buy_volume", "taker_buy_quote_volume"]
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"):
        df[c] = df[c].astype("float64")
    df["open_time"] = df["open_time"].astype("int64")
    df["trades"] = df["trades"].astype("int64")
    return df[KLINE_COLUMNS].reset_index(drop=True)


def parse_exchange_info(raw: dict) -> dict[str, SymbolInfo]:
    out: dict[str, SymbolInfo] = {}
    for s in raw.get("symbols", []):
        f = SymbolFilters()
        for flt in s.get("filters", []):
            t = flt.get("filterType")
            if t == "PRICE_FILTER":
                f.tick_size = float(flt["tickSize"])
            elif t == "LOT_SIZE":
                f.step_size = float(flt["stepSize"])
                f.min_qty = float(flt["minQty"])
                f.max_qty = float(flt["maxQty"])
            elif t == "MARKET_LOT_SIZE":
                f.max_qty = min(f.max_qty, float(flt["maxQty"])) if f.max_qty else float(flt["maxQty"])
            elif t == "MIN_NOTIONAL":
                f.min_notional = float(flt.get("notional") or flt.get("minNotional") or 5)
        out[s["symbol"]] = SymbolInfo(s["symbol"], s.get("status", ""), s.get("contractType", ""),
                                      s.get("quoteAsset", ""), int(s.get("onboardDate") or 0), f)
    return out
