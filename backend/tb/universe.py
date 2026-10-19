"""Hot-list ranking: volatile + liquid USDT perpetuals, with a retention buffer."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import Settings
from .exchange.base import Exchange, NetworkError, Premium, SymbolInfo, Ticker24h
from .risk import atr_pct

log = logging.getLogger("tb.universe")

ATR_BARS = 72          # last 6h of 5m bars
ATR_FETCH = 90         # 72 + warm-up; limit < 100 keeps the klines request at weight 1


@dataclass
class Candidate:
    symbol: str
    quote_volume: float
    change_pct: float
    last_price: float
    funding_rate: float
    atr_pct: float | None = None
    score: float = 0.0


def _atr_bars(cfg: Settings) -> int:
    """Volatility window for ranking (5m bars): 72 = 6 h (default), 12 = 1 h reacts to what moves now."""
    return int(getattr(cfg, "universe_atr_bars", ATR_BARS) or ATR_BARS)


def _atr_fetch(cfg: Settings) -> int:
    return _atr_bars(cfg) + 18          # ATR warm-up; stays < 100 bars (request weight 1)


def _retention(cfg: Settings) -> float:
    return float(getattr(cfg, "universe_retention", 1.5) or 1.5)


def eligible(info: SymbolInfo, t: Ticker24h, prem: Premium | None, cfg: Settings, now_ms: int) -> bool:
    if info.status != "TRADING" or info.contract_type != "PERPETUAL" or info.quote_asset != "USDT":
        return False
    if t.quote_volume < cfg.min_universe_volume_usd:
        return False
    if info.onboard_date_ms and now_ms - info.onboard_date_ms < cfg.min_listing_days * 86_400_000:
        return False
    if prem is not None and abs(prem.funding_rate) > cfg.max_abs_funding:
        return False
    return True


def score(atr_pct_: float, quote_volume: float) -> float:
    return atr_pct_ * math.sqrt(quote_volume)


def select(ranked: Sequence[str], current: Sequence[str], size: int, retention: float = 1.5) -> list[str]:
    """Retention buffer: a member stays while it ranks within top ceil(retention*size); free slots are
    filled from the top of the ranking. retention=1.0 = no buffer (pure top-N). Returned in rank order."""
    keep_rank = math.ceil(max(1.0, retention) * size)
    top_keep = set(ranked[:keep_rank])
    kept = [s for s in ranked if s in current and s in top_keep][:size]
    chosen = set(kept)
    for s in ranked:
        if len(chosen) >= size:
            break
        chosen.add(s)
    return [s for s in ranked if s in chosen]


def rank_candidates(cands: Sequence[Candidate]) -> list[Candidate]:
    ok = [c for c in cands if c.atr_pct is not None]
    for c in ok:
        c.score = score(c.atr_pct, c.quote_volume)
    return sorted(ok, key=lambda c: c.score, reverse=True)


def venue_filter(ranked: Sequence[str], venue_info: Mapping[str, SymbolInfo]) -> list[str]:
    """Keep only symbols that are TRADING on the order venue (rank order preserved)."""
    return [s for s in ranked if s in venue_info and venue_info[s].status == "TRADING"]


def whitelist_symbols(whitelist: Sequence[str], prod_info: Mapping[str, SymbolInfo],
                      venue_info: Mapping[str, SymbolInfo] | None = None, dropped: list[str] | None = None) -> list[str]:
    """The fixed universe: the whitelist (its order) restricted to symbols TRADING on prod (the signal data) and, when
    given, on the order venue. No volume / funding / listing-age / ranking filters and no ``universe_size`` cap: the
    operator named these coins. Coins the venue lacks are appended to ``dropped``."""
    out = []
    for s in dict.fromkeys(whitelist):
        i = prod_info.get(s)
        if i is None or i.status != "TRADING" or i.contract_type != "PERPETUAL":
            log.warning("universe whitelist: %s is not a TRADING perpetual on prod; skipped", s)
            continue
        if venue_info is not None and not (s in venue_info and venue_info[s].status == "TRADING"):
            if dropped is not None:
                dropped.append(s)
            continue
        out.append(s)
    return out


def _whitelist_rows(ex: Exchange, cfg: Settings, info: Mapping[str, SymbolInfo], tickers: Mapping[str, Ticker24h],
                    prem: Mapping[str, Premium], current: Mapping[str, int], now: float,
                    venue_info: Mapping[str, SymbolInfo] | None, dropped: list[str] | None) -> list[dict]:
    chosen = whitelist_symbols(cfg.universe_whitelist or [], info, venue_info, dropped)
    rows = []
    for rank, s in enumerate(chosen, 1):
        t = tickers.get(s)
        atr = None
        try:
            kl = ex.klines(s, "5m", _atr_fetch(cfg))
            atr = atr_pct(kl["high"].to_numpy(), kl["low"].to_numpy(), kl["close"].to_numpy(), _atr_bars(cfg))
        except NetworkError as e:
            log.warning("universe: klines failed for %s: %s", s, e)
        vol = t.quote_volume if t else 0.0
        last = (t.last_price if t else 0.0) or (prem[s].mark_price if s in prem else 0.0)
        oi_usd = oi_chg = None
        try:
            oi = ex.open_interest(s)
            oi_usd = oi * (prem[s].mark_price if s in prem else last)
            hist = ex.open_interest_hist(s, "5m", 13)
            if len(hist) >= 2 and hist[0][1] > 0:
                oi_chg = (hist[-1][1] / hist[0][1] - 1) * 100
        except Exception as e:  # OI is decoration; never let it break the refresh
            log.warning("universe: OI failed for %s: %s", s, e)
        rows.append({"symbol": s, "rank": rank, "score": score(atr, vol) if atr is not None else 0.0, "atr_pct": atr,
                     "quote_volume_24h": vol, "change_24h_pct": t.price_change_pct if t else 0.0,
                     "funding_rate": prem[s].funding_rate if s in prem else 0.0, "open_interest_usd": oi_usd,
                     "oi_change_1h_pct": oi_chg, "last_price": last, "in_universe_since": current.get(s, int(now)),
                     "updated_at": int(now)})
    return rows


def refresh(ex: Exchange, cfg: Settings, current: Mapping[str, int], now: float | None = None,
            venue_info: Mapping[str, SymbolInfo] | None = None, dropped: list[str] | None = None) -> list[dict]:
    """Build the new universe rows. ``current`` = {symbol: in_universe_since}.

    ``venue_info`` (the ORDER venue's exchangeInfo, binance_demo/live) restricts the result to symbols the
    venue can trade: the ranking is computed on prod data, then intersected with the venue; coins of the prod
    top list that the venue lacks are appended to ``dropped`` and replaced by the next-ranked tradable ones."""
    now = now or time.time()
    info = ex.exchange_info()
    tickers = {t.symbol: t for t in ex.tickers_24h()}
    prem = {p.symbol: p for p in ex.premium()}
    if cfg.universe_whitelist:
        return _whitelist_rows(ex, cfg, info, tickers, prem, current, now, venue_info, dropped)
    pre = []
    for s, t in tickers.items():
        i = info.get(s)
        if i and eligible(i, t, prem.get(s), cfg, int(now * 1000)):
            pre.append(Candidate(s, t.quote_volume, t.price_change_pct, t.last_price,
                                 prem[s].funding_rate if s in prem else 0.0))
    # ATR needs one klines call per candidate; rank by volume first and cap the fan-out
    pre.sort(key=lambda c: c.quote_volume, reverse=True)
    pre = pre[:150]
    for c in pre:
        try:
            kl = ex.klines(c.symbol, "5m", _atr_fetch(cfg))
            c.atr_pct = atr_pct(kl["high"].to_numpy(), kl["low"].to_numpy(), kl["close"].to_numpy(), _atr_bars(cfg))
        except NetworkError as e:
            log.warning("universe: klines failed for %s: %s", c.symbol, e)
    ranked = rank_candidates(pre)
    by = {c.symbol: c for c in ranked}
    order = [c.symbol for c in ranked]
    chosen = select(order, list(current), cfg.universe_size, _retention(cfg))
    if venue_info is not None:
        tradable = venue_filter(order, venue_info)
        if dropped is not None:
            dropped.extend(s for s in chosen if s not in set(tradable))
        chosen = select(tradable, [s for s in current if s in set(tradable)], cfg.universe_size, _retention(cfg))
    rows = []
    for rank, s in enumerate(chosen, 1):
        c = by[s]
        oi_usd = oi_chg = None
        try:
            oi = ex.open_interest(s)
            oi_usd = oi * (prem[s].mark_price if s in prem else c.last_price)
            hist = ex.open_interest_hist(s, "5m", 13)
            if len(hist) >= 2 and hist[0][1] > 0:
                oi_chg = (hist[-1][1] / hist[0][1] - 1) * 100
        except Exception as e:  # OI is decoration; never let it break the refresh
            log.warning("universe: OI failed for %s: %s", s, e)
        rows.append({"symbol": s, "rank": rank, "score": c.score, "atr_pct": c.atr_pct,
                     "quote_volume_24h": c.quote_volume, "change_24h_pct": c.change_pct,
                     "funding_rate": c.funding_rate, "open_interest_usd": oi_usd, "oi_change_1h_pct": oi_chg,
                     "last_price": c.last_price, "in_universe_since": current.get(s, int(now)),
                     "updated_at": int(now)})
    return rows
