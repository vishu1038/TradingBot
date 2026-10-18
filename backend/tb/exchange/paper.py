"""Paper exchange: real prod market data, simulated fills. Same interface as the real client.

* market fills at the cached mark/last price +- slippage, taker fee both sides
* exchange-side stops/take-profits are simulated on every ``tick`` (SL wins ties: pessimistic)
* isolated margin liquidation price; funding applied at 00/08/16 UTC
* state is JSON-serialisable (``export_state`` / ``import_state``) so the engine survives restarts
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import pandas as pd

from ..config import Settings
from ..risk import liq_price
from .base import (Balance, CloseInfo, ConditionalOrder, Exchange, ExchangeError, OrderResult, Premium,
                   PositionInfo, SymbolFilters, SymbolInfo, Ticker24h)

FUNDING_PERIOD_S = 8 * 3600


@dataclass
class _Pos:
    symbol: str
    side: str
    qty: float
    entry: float
    leverage: int
    margin: float
    liq: float
    opened_ms: int
    fees: float = 0.0
    funding: float = 0.0


@dataclass
class _Cond:
    id: str
    symbol: str
    kind: str          # SL | TP
    side: str          # closing order side
    trigger: float
    qty: float


class PaperExchange(Exchange):
    name = "paper"
    mode = "paper_local"

    def __init__(self, data: Exchange, cfg: Settings, start_equity: float | None = None):
        self.data = data
        self.cfg = cfg
        self.wallet = float(start_equity if start_equity is not None else cfg.paper_start_equity)
        self.pos: dict[str, _Pos] = {}
        self.conds: dict[str, _Cond] = {}
        self.closes: dict[str, CloseInfo] = {}
        self.marks: dict[str, float] = {}
        self.leverage: dict[str, int] = {}
        self._seq = 0
        self._slot = int(time.time() // FUNDING_PERIOD_S)
        self._lock = threading.RLock()

    def set_cfg(self, cfg: Settings) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ market data passthrough
    def server_time_ms(self) -> int:
        return self.data.server_time_ms()

    def klines(self, symbol, interval="5m", limit=500, start_ms=None, end_ms=None) -> pd.DataFrame:
        return self.data.klines(symbol, interval, limit, start_ms, end_ms)

    def tickers_24h(self) -> list[Ticker24h]:
        return self.data.tickers_24h()

    def exchange_info(self) -> dict[str, SymbolInfo]:
        return self.data.exchange_info()

    def premium(self, symbol: str | None = None) -> list[Premium]:
        return self.data.premium(symbol)

    def open_interest(self, symbol: str) -> float:
        return self.data.open_interest(symbol)

    def open_interest_hist(self, symbol, period="5m", limit=13):
        return self.data.open_interest_hist(symbol, period, limit)

    def long_short_ratio(self, symbol, period="5m"):
        return self.data.long_short_ratio(symbol, period)

    def depth(self, symbol, limit=1000):
        return self.data.depth(symbol, limit)

    # ------------------------------------------------------------------ pricing
    def _price(self, symbol: str) -> float:
        p = self.marks.get(symbol)
        if p:
            return p
        prem = self.data.premium(symbol)
        if not prem:
            raise ExchangeError(f"no price for {symbol}")
        self.marks[symbol] = prem[0].mark_price
        return prem[0].mark_price

    def _id(self) -> str:
        self._seq += 1
        return f"p:{self._seq}"

    def _slip(self, side: str, price: float, pct: float) -> float:
        return price * (1 + pct) if side == "BUY" else price * (1 - pct)

    # ------------------------------------------------------------------ trading
    def set_leverage(self, symbol: str, leverage: int) -> None:
        self.leverage[symbol] = int(leverage)

    def set_isolated(self, symbol: str) -> None:
        return None

    def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> OrderResult:
        side = side.upper()
        with self._lock:
            f = self.filters(symbol)
            q = f.round_qty(qty)
            if q < f.min_qty or q <= 0:
                raise ExchangeError(f"qty {qty} below min_qty {f.min_qty}", code=-4003)
            px = self._slip(side, self._price(symbol), self.cfg.slippage_pct)
            p = self.pos.get(symbol)
            if reduce_only:
                if not p or (p.side == "long") != (side == "SELL"):
                    raise ExchangeError("ReduceOnly Order is rejected", code=-2022)
                q = min(q, p.qty)
                info = self._close_qty(p, q, px, "manual")
                return OrderResult(self._id(), symbol, side, q, px, "FILLED", info.fees)
            if p and (p.side == "long") != (side == "BUY"):
                raise ExchangeError("opposite position open (one-way mode)", code=-4061)
            lev = self.leverage.get(symbol, self.cfg.leverage)
            margin = q * px / lev
            fee = q * px * self.cfg.fee_taker
            used = sum(x.margin for x in self.pos.values())
            if margin + fee > self.wallet - used:
                raise ExchangeError("Margin is insufficient", code=-2019)
            self.wallet -= fee
            ps = "long" if side == "BUY" else "short"
            if p:   # add to position: weighted entry
                tot = p.qty + q
                p.entry = (p.entry * p.qty + px * q) / tot
                p.qty = tot
                p.margin += margin
                p.fees += fee
                p.liq = liq_price(p.side, p.entry, p.leverage, self.cfg.maint_margin_rate)
            else:
                self.pos[symbol] = _Pos(symbol, ps, q, px, lev, margin,
                                        liq_price(ps, px, lev, self.cfg.maint_margin_rate),
                                        int(time.time() * 1000), fees=fee)
            return OrderResult(self._id(), symbol, side, q, px, "FILLED", fee)

    def _place_cond(self, kind: str, symbol: str, side: str, price: float, qty: float) -> str:
        with self._lock:
            f = self.filters(symbol)
            price = f.round_price(price)
            mark = self._price(symbol)
            side = side.upper()
            # a SELL stop below the market / SELL take-profit above it (mirror for BUY)
            if (kind == "SL") == (side == "SELL"):
                bad = price >= mark
            else:
                bad = price <= mark
            if bad:
                raise ExchangeError("Order would immediately trigger", code=-2021)
            c = _Cond(self._id(), symbol, kind, side, price, f.round_qty(qty))
            self.conds[c.id] = c
            return c.id

    def place_stop(self, symbol: str, side: str, stop_price: float, qty: float) -> str:
        return self._place_cond("SL", symbol, side, stop_price, qty)

    def place_take_profit(self, symbol: str, side: str, tp_price: float, qty: float) -> str:
        return self._place_cond("TP", symbol, side, tp_price, qty)

    def cancel_order(self, symbol: str, order_id: str) -> None:
        with self._lock:
            self.conds.pop(order_id, None)

    def cancel_all(self, symbol: str) -> None:
        with self._lock:
            for k in [k for k, c in self.conds.items() if c.symbol == symbol]:
                del self.conds[k]

    def open_conditional_orders(self, symbol: str | None = None) -> list[ConditionalOrder]:
        with self._lock:
            return [ConditionalOrder(c.id, c.symbol, c.kind, c.side, c.trigger, c.qty, True)
                    for c in self.conds.values() if symbol is None or c.symbol == symbol]

    # ------------------------------------------------------------------ accounting
    def _upnl(self, p: _Pos, price: float) -> float:
        return (price - p.entry) * p.qty * (1 if p.side == "long" else -1)

    def positions(self) -> list[PositionInfo]:
        with self._lock:
            out = []
            for p in self.pos.values():
                m = self.marks.get(p.symbol, p.entry)
                out.append(PositionInfo(p.symbol, p.side, p.qty, p.entry, m, self._upnl(p, m), p.leverage, p.liq,
                                        p.margin, True))
            return out

    def balance(self) -> Balance:
        with self._lock:
            upnl = sum(self._upnl(p, self.marks.get(p.symbol, p.entry)) for p in self.pos.values())
            used = sum(p.margin for p in self.pos.values())
            return Balance(self.wallet, max(0.0, self.wallet - used), upnl)

    def _close_qty(self, p: _Pos, qty: float, price: float, reason: str, *, extra_fee: float | None = None) -> CloseInfo:
        qty = min(qty, p.qty)
        gross = (price - p.entry) * qty * (1 if p.side == "long" else -1)
        fee = qty * price * self.cfg.fee_taker if extra_fee is None else extra_fee
        self.wallet += gross - fee
        frac = qty / p.qty
        entry_fee = p.fees * frac
        funding = p.funding * frac
        info = CloseInfo(price, gross, entry_fee + fee, funding, reason, int(time.time()))
        p.qty -= qty
        p.margin *= (1 - frac)
        p.fees -= entry_fee
        p.funding -= funding
        if p.qty <= 1e-12:
            del self.pos[p.symbol]
            self.cancel_all(p.symbol)
            self.closes[p.symbol] = info
        return info

    # ------------------------------------------------------------------ simulation
    def tick(self, marks: Mapping[str, float]) -> list[tuple[str, CloseInfo]]:
        """Advance with new mark prices; returns [(symbol, CloseInfo)] for exchange-side closes."""
        events: list[tuple[str, CloseInfo]] = []
        with self._lock:
            for s, m in marks.items():
                if m and m > 0:
                    self.marks[s] = float(m)
            for s in list(self.pos):
                if s in marks and marks[s] > 0:
                    ev = self._eval(s, float(marks[s]))
                    if ev:
                        events.append((s, ev))
        return events

    def check_range(self, symbol: str, low: float, high: float, bar_open_ms: int) -> CloseInfo | None:
        """Replay a completed candle's wicks (adverse extreme first = SL first). Only bars that
        started after the entry are considered, so pre-entry prices can never trigger a stop."""
        with self._lock:
            p = self.pos.get(symbol)
            if not p or bar_open_ms <= p.opened_ms:
                return None
            adverse, fav = (low, high) if p.side == "long" else (high, low)
            return self._eval(symbol, adverse) or (self._eval(symbol, fav) if symbol in self.pos else None)

    def _eval(self, symbol: str, m: float) -> CloseInfo | None:
        p = self.pos.get(symbol)
        if not p:
            return None
        long = p.side == "long"
        # 1) liquidation
        if (long and m <= p.liq) or (not long and m >= p.liq):
            clearing = p.qty * p.liq * self.cfg.maint_margin_rate
            return self._close_qty(p, p.qty, p.liq, "liquidation", extra_fee=clearing)
        # 2) conditional orders, SL before TP
        conds = sorted((c for c in self.conds.values() if c.symbol == symbol), key=lambda c: c.kind != "SL")
        for c in conds:
            if c.kind == "SL":
                hit = m <= c.trigger if long else m >= c.trigger
                if hit:
                    base = min(c.trigger, m) if long else max(c.trigger, m)   # gap through the stop fills worse
                    # sim convention (tb.sim.template): slippage + stop_slippage_extra on stop fills
                    s = self.cfg.slippage_pct + self.cfg.stop_slippage_pct
                    px = base * (1 - s) if long else base * (1 + s)
                    return self._close_qty(p, min(c.qty, p.qty), px, "sl")
            else:
                hit = m >= c.trigger if long else m <= c.trigger
                if hit:
                    px = c.trigger                  # resting-limit style: exactly the TP price, no slippage (sim)
                    return self._close_qty(p, min(c.qty, p.qty), px, "tp")
        return None

    def funding_due(self, now: float | None = None) -> bool:
        return int((now or time.time()) // FUNDING_PERIOD_S) > self._slot

    def apply_funding(self, rates: Mapping[str, float], now: float | None = None) -> dict[str, float]:
        """Apply the funding settlement(s) crossed since the last call. Long pays when rate > 0."""
        paid: dict[str, float] = {}
        with self._lock:
            slot = int((now or time.time()) // FUNDING_PERIOD_S)
            if slot <= self._slot:
                return paid
            self._slot = slot
            for s, p in self.pos.items():
                r = rates.get(s)
                if r is None:
                    continue
                mark = self.marks.get(s, p.entry)
                cash = -(p.qty * mark * r) * (1 if p.side == "long" else -1)
                self.wallet += cash
                p.funding += cash
                paid[s] = cash
        return paid

    def last_close(self, symbol, side, since_ms, sl_order_id=None, tp_order_id=None, entry_price=0.0):
        with self._lock:
            return self.closes.pop(symbol, None)

    # ------------------------------------------------------------------ persistence
    def adopt(self, symbol: str, side: str, qty: float, entry: float, leverage: int, opened_ms: int,
              fees: float = 0.0, funding: float = 0.0) -> None:
        with self._lock:
            self.pos[symbol] = _Pos(symbol, side, qty, entry, leverage, qty * entry / leverage,
                                    liq_price(side, entry, leverage, self.cfg.maint_margin_rate), opened_ms,
                                    fees, funding)

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return {"wallet": self.wallet, "slot": self._slot, "seq": self._seq,
                    "pos": {k: asdict(v) for k, v in self.pos.items()},
                    "conds": {k: asdict(v) for k, v in self.conds.items()},
                    "closes": {k: asdict(v) for k, v in self.closes.items()},
                    "leverage": dict(self.leverage)}

    def import_state(self, st: Mapping[str, Any]) -> None:
        with self._lock:
            self.wallet = float(st["wallet"])
            self._slot = int(st.get("slot", self._slot))
            self._seq = int(st.get("seq", 0))
            self.pos = {k: _Pos(**v) for k, v in st.get("pos", {}).items()}
            self.conds = {k: _Cond(**v) for k, v in st.get("conds", {}).items()}
            self.closes = {k: CloseInfo(**v) for k, v in st.get("closes", {}).items()}
            self.leverage = {k: int(v) for k, v in st.get("leverage", {}).items()}
