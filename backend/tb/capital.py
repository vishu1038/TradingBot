"""The bot's own capital on a (possibly much larger) exchange account.

On the testnet / a funded account the wallet can be far bigger than the money the user wants at risk.
``capital_usd`` (setting; null = whole wallet) defines the bot's stake. In binance_demo/live::

    bot equity = capital_usd + net realised PnL of the bot's own trades since the stake was set
                 + open unrealised PnL, capped by the real wallet equity

Slots, derating, the kill switch and drawdown all use the bot equity. The starting point is the id of
the newest trade in the store at the moment the stake was first set (kv ``capital_floor_trade_id``), so
restarts and ``reset_kill`` never reset it. Shared by the engine and the (independent) guard.
"""
from __future__ import annotations

from typing import Any

FLOOR_KEY = "capital_floor_trade_id"
BASIS_KEY = "capital_basis"           # the capital_usd value the peak/start equity refer to


def is_active(cfg: Any, mode: str) -> bool:
    return mode != "paper_local" and getattr(cfg, "capital_usd", None) is not None


def floor_id(store, create: bool = False) -> int | None:
    v = store.kv_get(FLOOR_KEY)
    if v is None and create:
        r = store.one("SELECT COALESCE(MAX(id),0) m FROM trades")
        v = int(r["m"])
        store.kv_set(FLOOR_KEY, v)
    return None if v is None else int(v)


def realised_pnl(store, floor: int) -> float:
    """Net realised PnL (trades.pnl is net of fees and funding) of trades recorded after the floor."""
    return float(store.one("SELECT COALESCE(SUM(pnl),0) s FROM trades WHERE id>?", (floor,))["s"])


def bot_equity(store, cfg: Any, mode: str, wallet_equity: float, upnl: float, *, create: bool = False) -> float | None:
    """Bot equity, or ``wallet_equity`` when no stake applies; None if the stake is not initialised yet."""
    if not is_active(cfg, mode):
        return wallet_equity
    f = floor_id(store, create)
    if f is None:
        return None
    return min(float(cfg.capital_usd) + realised_pnl(store, f) + upnl, wallet_equity)
