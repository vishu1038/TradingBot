"""Shadow mode: challengers (and the champion) score every bar; their would-be trades are
simulated with the live trade template and recorded, so a challenger can be compared with the
champion on the same period before a manual promotion.

Pure logic only (no I/O): the engine feeds it signals / prices / bars and persists the output.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .config import Settings
from .exchange.base import SymbolFilters
from .risk import levels


@dataclass
class ShadowPos:
    model: str
    symbol: str
    side: str
    entry: float
    qty: float
    sl: float
    tp: float
    opened_at: int          # seconds
    prob: float
    entry_fee: float


class ShadowBook:
    """Open shadow positions, one per (model, symbol)."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.pos: dict[str, ShadowPos] = {}

    @staticmethod
    def _key(model: str, symbol: str) -> str:
        return f"{model}|{symbol}"

    def has(self, model: str, symbol: str) -> bool:
        return self._key(model, symbol) in self.pos

    def open(self, model: str, symbol: str, side: str, prob: float, price: float, t: float,
             atr: float | None, f: SymbolFilters) -> bool:
        if self.has(model, symbol) or price <= 0:
            return False
        cfg = self.cfg
        fill = price * (1 + cfg.slippage_pct) if side == "long" else price * (1 - cfg.slippage_pct)
        qty = f.round_qty(cfg.notional_usd / fill)
        if qty <= 0:
            return False
        lv = levels(cfg, side, fill, qty, atr, f)
        if lv is None:
            return False
        self.pos[self._key(model, symbol)] = ShadowPos(model, symbol, side, fill, qty, lv[0], lv[1], int(t), float(prob),
                                                       qty * fill * cfg.fee_taker)
        return True

    def _close(self, p: ShadowPos, px: float, reason: str, t: float) -> dict:
        d = 1.0 if p.side == "long" else -1.0
        gross = (px - p.entry) * p.qty * d
        fees = p.entry_fee + p.qty * px * self.cfg.fee_taker
        del self.pos[self._key(p.model, p.symbol)]
        return {"model": p.model, "symbol": p.symbol, "side": p.side, "entry_time": p.opened_at, "exit_time": int(t),
                "entry_price": p.entry, "exit_price": px, "qty": p.qty, "pnl": gross - fees, "fees": fees,
                "exit_reason": reason, "prob": p.prob}

    def _eval(self, p: ShadowPos, m: float, t: float) -> dict | None:
        long = p.side == "long"
        cfg = self.cfg
        if (long and m <= p.sl) or (not long and m >= p.sl):      # SL first (pessimistic)
            base = min(p.sl, m) if long else max(p.sl, m)
            s = cfg.slippage_pct + cfg.stop_slippage_pct          # tb.sim.template convention
            px = base * (1 - s) if long else base * (1 + s)
            return self._close(p, px, "sl", t)
        if (long and m >= p.tp) or (not long and m <= p.tp):
            return self._close(p, p.tp, "tp", t)                  # TP: no slippage
        return None

    def on_price(self, symbol: str, price: float, t: float | None = None) -> list[dict]:
        t = t or time.time()
        out = []
        for p in [p for p in self.pos.values() if p.symbol == symbol]:
            r = self._eval(p, price, t)
            if r:
                out.append(r)
        return out

    def on_bar(self, symbol: str, bar_open_ms: int, low: float, high: float, t: float | None = None) -> list[dict]:
        """Replay a completed bar's wicks (adverse extreme first); only bars that opened after entry."""
        t = t or time.time()
        out = []
        for p in [p for p in self.pos.values() if p.symbol == symbol]:
            if bar_open_ms <= p.opened_at * 1000:
                continue
            adverse, fav = (low, high) if p.side == "long" else (high, low)
            r = self._eval(p, adverse, t) or self._eval(p, fav, t)
            if r:
                out.append(r)
        return out

    def export_state(self) -> dict[str, Any]:
        return {k: asdict(v) for k, v in self.pos.items()}

    def import_state(self, st: Mapping[str, Any]) -> None:
        self.pos = {k: ShadowPos(**v) for k, v in st.items()}


# ----------------------------------------------------------------------------- statistics / promotion
def shadow_stats(trades: list[Mapping[str, Any]]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": 0.0, "expectancy": 0.0, "pnl": 0.0}
    pnl = float(sum(t["pnl"] for t in trades))
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return {"n_trades": n, "win_rate": wins / n, "expectancy": pnl / n, "pnl": pnl}


def promotion_check(chal: Mapping[str, Any], champ: Mapping[str, Any], drift_level: int,
                    min_trades: int = 50) -> tuple[bool, str]:
    """>= min_trades shadow trades AND higher expectancy than the champion over the same period
    AND no drift alarm (any drift level above 0 blocks promotion)."""
    if chal["n_trades"] < min_trades:
        return False, f"needs >= {min_trades} shadow trades (has {chal['n_trades']})"
    if not chal["expectancy"] > champ["expectancy"]:
        return False, (f"expectancy {chal['expectancy']:.3f} does not beat champion "
                       f"{champ['expectancy']:.3f} over the same period")
    if drift_level > 0:
        return False, "drift alarm active"
    return True, "ok"


@dataclass
class Registry:
    """Runtime status overrides (persisted in the store) on top of ``models/*.json`` statuses."""
    overrides: dict[str, str]
    previous_champion: str | None = None

    def to_dict(self) -> dict:
        return {"overrides": dict(self.overrides), "previous_champion": self.previous_champion}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "Registry":
        d = d or {}
        return cls(dict(d.get("overrides", {})), d.get("previous_champion"))

    def promote(self, model: str, current_champion: str | None) -> "Registry":
        ov = dict(self.overrides)
        if current_champion and current_champion != model:
            ov[current_champion] = "retired"
        ov[model] = "champion"
        return Registry(ov, current_champion if current_champion != model else self.previous_champion)

    def rollback(self, current_champion: str | None) -> "Registry":
        if not self.previous_champion:
            raise ValueError("no previous champion to roll back to")
        ov = dict(self.overrides)
        if current_champion:
            ov[current_champion] = "retired"
        ov[self.previous_champion] = "champion"
        return Registry(ov, current_champion)


def vs_champion(chal: Mapping[str, Any], champ_same_period: Mapping[str, Any]) -> float | None:
    if chal["n_trades"] == 0:
        return None
    return float(chal["expectancy"] - champ_same_period["expectancy"])
