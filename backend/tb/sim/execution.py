"""Execution models shared by labels, backtests and the cross-sectional portfolio simulator (V3 experiment D).

Two ways to get into (or out of) a position at a decision made on the CLOSE of 5m bar ``i``:

taker   market order at the OPEN of bar ``i+1`` with adverse slippage ``slippage``; fee ``fee_taker`` (0.05 %).
        This is exactly what ``tb.sim.template`` / ``tb.labels`` do today.

maker   post-only limit at the best bid (buy) / best ask (sell) at decision time, fee ``fee_maker`` (0.02 %).
        Without an order book the best bid/ask is proxied by the last traded price = ``close[i]`` (the
        spread is ignored, i.e. the limit is as aggressive as a passive order can be).  CONSERVATIVE
        fill rule: the order rests from bar ``i+1`` on and is filled only when a bar TRADES THROUGH the
        limit: ``low < L*(1 - through_pct)`` for a buy, ``high > L*(1 + through_pct)`` for a sell.
        Merely touching the price does not fill (queue position).  A fill is booked at exactly ``L``
        (never at a better price, even when the bar gaps through the limit).  After ``timeout_bars``
        bars without a fill the order is cancelled and re-posted at the then-best price
        (``close`` of the last bar), up to ``reprices`` times.  When the last placement times out the
        order is dropped (``on_timeout="skip"``) or converted to a taker order at the open of the
        next bar (``"taker"``).  Because a maker order only fills if price moves against it first, the
        simulation contains the adverse-selection cost of resting orders; the saved fee/slippage is
        paid for with missed trades.

Everything is vectorised over many decisions (``fill_orders``); ``fill_order`` is the one-order wrapper.
numpy only (imports under both 3.14 and 3.12).

``low_vol_gate``: a no-trade filter for the regime where fees dominate: trade only when the typical
move over the holding period is at least ``min_ratio`` times the round-trip cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

KIND_NONE, KIND_MAKER, KIND_TAKER = 0, 1, 2


@dataclass
class ExecConfig:
    mode: str = "taker"                # taker | maker
    fee_taker: float = 0.0005          # per side, fraction of notional
    fee_maker: float = 0.0002
    slippage: float = 0.0002           # adverse slippage of a market order, fraction
    timeout_bars: int = 3              # 5m bars a resting limit lives before it is cancelled
    reprices: int = 2                  # re-posts after a timeout (placements = reprices + 1)
    on_timeout: str = "skip"           # skip | taker
    through_pct: float = 0.0           # extra penetration (fraction of price) beyond the limit required to fill

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "ExecConfig":
        if self.mode not in ("taker", "maker"):
            raise ValueError(f"mode must be taker|maker, got {self.mode!r}")
        if self.on_timeout not in ("skip", "taker"):
            raise ValueError(f"on_timeout must be skip|taker, got {self.on_timeout!r}")
        if self.timeout_bars < 1 or self.reprices < 0:
            raise ValueError("timeout_bars >= 1 and reprices >= 0 required")
        return self


@dataclass
class Fill:
    filled: bool
    kind: int            # KIND_NONE | KIND_MAKER | KIND_TAKER (taker = plain taker order or the timeout fallback)
    bar: int             # index of the bar the fill happened in (taker: the bar whose open was used); -1 if none
    price: float         # fill price (NaN if none)
    fee_rate: float      # fraction of notional charged on this fill
    placements: int      # limit orders posted (0 for plain taker)


def fill_orders(o, h, l, c, sig_idx, side, cfg: ExecConfig, on_timeout: str | None = None) -> dict:
    """Execute many orders decided on the close of bar ``sig_idx[n]`` (arrays indexed like ``o/h/l/c``).

    ``side``: +1 buy / -1 sell (scalar or array).  ``on_timeout`` overrides ``cfg.on_timeout`` (positions
    that must be closed use "taker").  Returns arrays: filled(bool), kind(int8), bar(int64, -1 none),
    price(float64, NaN none), fee_rate(float64), placements(int32).  An order whose placement window runs
    past the end of the data is unresolved (filled False)."""
    cfg.validate()
    o = np.asarray(o, "float64"); h = np.asarray(h, "float64")
    l = np.asarray(l, "float64"); c = np.asarray(c, "float64")
    n = len(o)
    i = np.atleast_1d(np.asarray(sig_idx, dtype="int64"))
    m = len(i)
    sd = np.broadcast_to(np.atleast_1d(np.asarray(side, dtype="float64")), (m,)).astype("float64")
    ot = cfg.on_timeout if on_timeout is None else on_timeout
    filled = np.zeros(m, bool)
    kind = np.zeros(m, "int8")
    bar = np.full(m, -1, "int64")
    price = np.full(m, np.nan)
    fee = np.zeros(m)
    plc = np.zeros(m, "int32")
    if m == 0:
        return dict(filled=filled, kind=kind, bar=bar, price=price, fee_rate=fee, placements=plc)
    in_rng = (i >= 0) & (i < n)

    if cfg.mode == "taker":
        b = i + 1
        ok = in_rng & (b < n)
        ob = np.where(ok, o[np.clip(b, 0, n - 1)], np.nan)
        ok &= np.isfinite(ob) & (ob > 0)
        filled[ok] = True
        kind[ok] = KIND_TAKER
        bar[ok] = b[ok]
        price[ok] = ob[ok] * (1.0 + sd[ok] * cfg.slippage)
        fee[ok] = cfg.fee_taker
        return dict(filled=filled, kind=kind, bar=bar, price=price, fee_rate=fee, placements=plc)

    # ---- maker
    L = np.where(in_rng, c[np.clip(i, 0, n - 1)], np.nan)          # limit price = last traded price
    active = in_rng & np.isfinite(L) & (L > 0)
    unresolved = np.zeros(m, bool)                                  # ran out of data while resting
    start = i + 1
    tp = cfg.through_pct
    for _attempt in range(cfg.reprices + 1):
        plc[active] += 1
        for w in range(cfg.timeout_bars):
            b = start + w
            inb = b < n
            unresolved |= active & ~inb
            active &= inb
            if not active.any():
                break
            bb = np.clip(b, 0, n - 1)
            hit = np.where(sd > 0, l[bb] < L * (1.0 - tp), h[bb] > L * (1.0 + tp)) & active
            if hit.any():
                filled[hit] = True
                kind[hit] = KIND_MAKER
                bar[hit] = b[hit]
                price[hit] = L[hit]
                fee[hit] = cfg.fee_maker
                active &= ~hit
        if not active.any():
            break
        nxt = start + cfg.timeout_bars
        last = nxt - 1
        inb = last < n
        unresolved |= active & ~inb
        active &= inb
        L = np.where(active, c[np.clip(last, 0, n - 1)], L)       # re-post at the new best price
        active &= np.isfinite(L) & (L > 0)
        start = nxt
    if ot == "taker":
        b = start
        ok = active & (b < n)
        unresolved |= active & ~(b < n)
        ob = np.where(ok, o[np.clip(b, 0, n - 1)], np.nan)
        ok &= np.isfinite(ob) & (ob > 0)
        filled[ok] = True
        kind[ok] = KIND_TAKER
        bar[ok] = b[ok]
        price[ok] = ob[ok] * (1.0 + sd[ok] * cfg.slippage)
        fee[ok] = cfg.fee_taker
    return dict(filled=filled, kind=kind, bar=bar, price=price, fee_rate=fee, placements=plc)


def fill_order(o, h, l, c, i: int, side: int, cfg: ExecConfig, on_timeout: str | None = None) -> Fill:
    """One order; see ``fill_orders``."""
    r = fill_orders(o, h, l, c, [i], side, cfg, on_timeout)
    ok = bool(r["filled"][0])
    return Fill(ok, int(r["kind"][0]), int(r["bar"][0]), float(r["price"][0]), float(r["fee_rate"][0]),
                int(r["placements"][0]))


# --------------------------------------------------------------------------------------------
# costs and the low-volatility no-trade gate
# --------------------------------------------------------------------------------------------
def round_trip_cost(cfg: ExecConfig, entry_maker: float | None = None, exit_maker: float | None = None,
                    stop_slippage_extra: float = 0.0) -> float:
    """Expected round-trip cost as a fraction of notional (fees + slippage of both fills).

    ``entry_maker`` / ``exit_maker``: probability that the leg is executed as maker (default: 1.0 in maker
    mode, 0.0 in taker mode).  Taker legs pay ``fee_taker + slippage``, maker legs ``fee_maker``;
    ``stop_slippage_extra`` is added once (a stop exit)."""
    d = 1.0 if cfg.mode == "maker" else 0.0
    pe = d if entry_maker is None else entry_maker
    px = d if exit_maker is None else exit_maker
    leg = lambda p: p * cfg.fee_maker + (1 - p) * (cfg.fee_taker + cfg.slippage)  # noqa: E731
    return float(leg(pe) + leg(px) + stop_slippage_extra)


def low_vol_gate(atr_frac, cfg: ExecConfig, horizon_bars: float, min_ratio: float = 3.0,
                 round_trip: float | None = None):
    """True where trading is worthwhile: the typical move over the holding period must be at least
    ``min_ratio`` x the round-trip cost.  ``atr_frac`` = per-bar ATR / price (scalar or array);
    the typical move is ``atr_frac * sqrt(horizon_bars)`` (random-walk scaling).  False means the fees
    dominate and the trade should be skipped."""
    rt = round_trip_cost(cfg) if round_trip is None else round_trip
    a = np.asarray(atr_frac, "float64")
    move = a * np.sqrt(max(float(horizon_bars), 1.0))
    ok = np.isfinite(move) & (move >= min_ratio * rt)
    return bool(ok) if ok.ndim == 0 else ok


def fees_dominate(atr_frac, cfg: ExecConfig, horizon_bars: float, min_ratio: float = 3.0):
    """Complement of ``low_vol_gate`` (NaN input -> True)."""
    return ~np.asarray(low_vol_gate(atr_frac, cfg, horizon_bars, min_ratio))
