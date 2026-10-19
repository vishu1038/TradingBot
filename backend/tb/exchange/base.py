"""Exchange interface (Binance USDⓈ-M in v1; BitMEX can implement the same surface later)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any

import pandas as pd

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                 "taker_buy_volume", "taker_buy_quote_volume"]


class ExchangeError(Exception):
    def __init__(self, msg: str, code: int | None = None, status: int | None = None):
        super().__init__(msg)
        self.code = code
        self.status = status


class OrderNotFilled(ExchangeError):
    """A market order was accepted but nothing executed (thin / synthetic testnet books). The order has been cancelled:
    the position did NOT change. Callers must treat it as a failed order, never as a fill."""


class NetworkError(ExchangeError):
    """Transport-level failure (timeout, DNS, 5xx). Safe to retry reads."""


class RateLimitError(ExchangeError):
    def __init__(self, msg: str, retry_after: float = 1.0, **kw: Any):
        super().__init__(msg, **kw)
        self.retry_after = retry_after


@dataclass
class SymbolFilters:
    tick_size: float = 0.01
    step_size: float = 0.001
    min_qty: float = 0.001
    min_notional: float = 5.0
    max_qty: float = 1e12

    def round_price(self, price: float) -> float:
        return float(_quant(price, self.tick_size, ROUND_HALF_UP))

    def round_qty(self, qty: float) -> float:
        """Floor to the lot step."""
        return float(_quant(qty, self.step_size, ROUND_DOWN))

    def price_str(self, price: float) -> str:
        return fmt_decimal(_quant(price, self.tick_size, ROUND_HALF_UP))

    def qty_str(self, qty: float) -> str:
        return fmt_decimal(_quant(qty, self.step_size, ROUND_DOWN))


def _quant(x: float, step: float, rounding: str) -> Decimal:
    d = Decimal(repr(float(x)))
    s = Decimal(repr(float(step)))
    if s <= 0:
        return d
    return (d / s).to_integral_value(rounding=rounding) * s


def fmt_decimal(d: Decimal) -> str:
    s = format(d.normalize(), "f")
    return s if "." not in s else s.rstrip("0").rstrip(".")


@dataclass
class SymbolInfo:
    symbol: str
    status: str = "TRADING"
    contract_type: str = "PERPETUAL"
    quote_asset: str = "USDT"
    onboard_date_ms: int = 0
    filters: SymbolFilters = field(default_factory=SymbolFilters)


@dataclass
class Ticker24h:
    symbol: str
    last_price: float
    quote_volume: float
    price_change_pct: float
    high: float = 0.0
    low: float = 0.0


@dataclass
class Premium:
    symbol: str
    mark_price: float
    index_price: float = 0.0
    funding_rate: float = 0.0       # last settled funding rate
    next_funding_ms: int = 0


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str                       # BUY | SELL
    qty: float
    avg_price: float = 0.0
    status: str = "NEW"
    fee: float = 0.0                # quote-currency fee for fills known at placement time (paper)
    executed_qty: float | None = None   # limit orders: filled so far (None = ``qty`` is the executed quantity)
    price: float = 0.0              # limit orders: the limit price

    @property
    def filled(self) -> float:
        """Executed quantity (market orders report it in ``qty``; limit orders in ``executed_qty``)."""
        return self.qty if self.executed_qty is None else self.executed_qty


@dataclass
class PositionInfo:
    symbol: str
    side: str                       # long | short
    qty: float
    entry_price: float
    mark_price: float = 0.0
    upnl: float = 0.0
    leverage: int = 1
    liq_price: float = 0.0
    margin: float = 0.0
    isolated: bool = True


@dataclass
class Balance:
    wallet: float                   # wallet balance (realised, excl. unrealised pnl)
    available: float
    upnl: float = 0.0

    @property
    def equity(self) -> float:
        return self.wallet + self.upnl


@dataclass
class ConditionalOrder:
    order_id: str
    symbol: str
    kind: str                       # SL | TP
    side: str                       # BUY | SELL
    trigger_price: float
    qty: float
    reduce_only: bool = True


@dataclass
class CloseInfo:
    exit_price: float
    pnl: float                      # gross price pnl
    fees: float                     # entry + exit fees (positive cost)
    funding: float                  # signed cashflow (+ received)
    reason: str                     # tp | sl | trail | kill | manual | liquidation
    closed_at: int = 0

    # NOTE: ``pnl`` here is GROSS price pnl and ``funding`` a signed cashflow (+ received); the engine
    # converts to the trades-table convention (pnl NET, funding as a COST) through these helpers.
    @property
    def net_pnl(self) -> float:
        return self.pnl - self.fees + self.funding

    @property
    def funding_cost(self) -> float:
        return -self.funding


def opposite(side: str) -> str:
    return "SELL" if side.upper() == "BUY" else "BUY"


def order_side(pos_side: str) -> str:
    """Entry order side for a position side ('long'/'short')."""
    return "BUY" if pos_side == "long" else "SELL"


class Exchange(ABC):
    """Everything the engine needs. Market-data calls work without keys."""

    name = "base"
    mode = "paper_local"

    # ---- market data
    @abstractmethod
    def server_time_ms(self) -> int: ...

    @abstractmethod
    def klines(self, symbol: str, interval: str = "5m", limit: int = 500, start_ms: int | None = None,
               end_ms: int | None = None) -> pd.DataFrame:
        """Archive-format frame (KLINE_COLUMNS). The last row may be the still-forming candle."""

    @abstractmethod
    def tickers_24h(self) -> list[Ticker24h]: ...

    @abstractmethod
    def exchange_info(self) -> dict[str, SymbolInfo]: ...

    @abstractmethod
    def premium(self, symbol: str | None = None) -> list[Premium]:
        """Mark price + funding for one symbol or (None) all."""

    @abstractmethod
    def open_interest(self, symbol: str) -> float:
        """Current open interest in contracts (base units)."""

    @abstractmethod
    def open_interest_hist(self, symbol: str, period: str = "5m", limit: int = 13) -> list[tuple[int, float]]:
        """[(ts_ms, open_interest_value_usd)] oldest first."""

    @abstractmethod
    def long_short_ratio(self, symbol: str, period: str = "5m") -> float | None: ...

    def depth(self, symbol: str, limit: int = 1000) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Order book ``(bids high->low, asks low->high)`` as ``(price, qty)``. Optional capability."""
        raise ExchangeError(f"{self.name} exchange has no order-book support")

    # ---- trading
    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> None: ...

    @abstractmethod
    def set_isolated(self, symbol: str) -> None: ...

    @abstractmethod
    def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> OrderResult: ...

    @abstractmethod
    def place_stop(self, symbol: str, side: str, stop_price: float, qty: float) -> str:
        """Reduce-only STOP_MARKET (workingType MARK_PRICE). ``side`` is the order side (closing)."""

    @abstractmethod
    def place_take_profit(self, symbol: str, side: str, tp_price: float, qty: float) -> str:
        """Reduce-only TAKE_PROFIT_MARKET."""

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> None: ...

    @abstractmethod
    def cancel_all(self, symbol: str) -> None: ...

    @abstractmethod
    def open_conditional_orders(self, symbol: str | None = None) -> list[ConditionalOrder]: ...

    @abstractmethod
    def positions(self) -> list[PositionInfo]: ...

    @abstractmethod
    def balance(self) -> Balance: ...

    @abstractmethod
    def last_close(self, symbol: str, side: str, since_ms: int, sl_order_id: str | None = None,
                   tp_order_id: str | None = None, entry_price: float = 0.0,
                   booked: dict | None = None) -> CloseInfo | None:
        """Describe how a position that vanished from the exchange was closed. ``booked`` (venues that rebuild the
        close from the fill history): a partial take-profit already booked as its own trade -
        ``{"order_ids": [...], "entry_fee": usd, "funding": signed usd}`` - which must not be counted twice."""

    def leverage_brackets(self, symbol: str):
        """Maintenance-margin tiers (``tb.risk.Bracket`` list) for ``symbol``, or None when unavailable."""
        return None

    # ---- maker (post-only) entries; optional capability, implemented by the Binance and paper venues
    def best_bid_ask(self, symbol: str) -> tuple[float, float]:
        """(best bid, best ask) of the venue's own order book."""
        raise ExchangeError(f"{self.name} exchange has no best bid/ask support")

    def limit_order(self, symbol: str, side: str, qty: float, price: float, post_only: bool = True,
                    reduce_only: bool = False) -> OrderResult:
        """Resting LIMIT order; ``post_only`` = time-in-force GTX. A post-only order that would take liquidity is
        rejected with ``ExchangeError(code=-5022)``. Returns immediately (status NEW / PARTIALLY_FILLED / FILLED)."""
        raise ExchangeError(f"{self.name} exchange has no limit-order support")

    def get_order(self, symbol: str, order_id: str) -> OrderResult:
        """Current state of an order placed by ``limit_order`` (status, executed qty, average fill price)."""
        raise ExchangeError(f"{self.name} exchange has no order lookup support")

    def open_entry_orders(self, symbol: str | None = None) -> list[tuple[str, str]]:
        """[(symbol, order_id)] of resting NON-conditional maker entries (stale ones left by a crashed engine)."""
        return []

    def replace_conditional(self, symbol: str, kind: str, side: str, price: float, qty: float,
                            old_id: str | None = None, retries: int = 3,
                            pause_s: float = 0.4) -> tuple[str, str | None]:
        """Move a protective order: place the NEW reduce-only stop (``kind`` "SL") / take-profit ("TP") first and cancel the
        old one only afterwards, so the position is never without protection. Returns ``(new_id, stale_id)``;
        ``stale_id`` is the old order when it could not be cancelled (harmless: reduce-only; the caller retries the
        cancel). If every attempt to place the new order fails the exception propagates and the OLD order is untouched.
        -2021 ("would immediately trigger": the price is already through the level) is never retried."""
        import logging
        import time

        log = logging.getLogger("tb.exchange")
        place = self.place_stop if kind == "SL" else self.place_take_profit
        last: ExchangeError | None = None
        new_id = None
        for i in range(max(1, retries)):
            try:
                new_id = place(symbol, side, price, qty)
                break
            except ExchangeError as e:
                last = e
                log.warning("replace %s %s: placing the new order @ %s failed (%s) [attempt %d/%d]", kind, symbol, price,
                            e, i + 1, retries)
                if e.code == -2021:
                    break
                if i + 1 < retries:
                    time.sleep(pause_s * (i + 1))
        if new_id is None:
            assert last is not None
            raise last
        if old_id:
            try:
                self.cancel_order(symbol, old_id)
            except ExchangeError as e:
                log.warning("replace %s %s: new order %s is live but cancelling the old %s failed: %s", kind, symbol,
                            new_id, old_id, e)
                return new_id, old_id
        return new_id, None

    # ---- helpers
    def filters(self, symbol: str) -> SymbolFilters:
        info = self.exchange_info().get(symbol)
        return info.filters if info else SymbolFilters()
