"""Shadow mode: challengers (and the champion) score every decision point; their would-be trades are
simulated with the live trade template and recorded, so a challenger can be compared with the
champion on the same period before a manual promotion.

V3: every model (any ``kind``) brings its own ``exit_cfg`` (``ExitSpec``) and ``entry_cfg`` (``EntryCfg``);
maker entries are simulated with the shared conservative model ``tb.sim.execution`` (a later bar must trade
strictly THROUGH the limit); cross-sectional models open / close legs on their rebalance.

Pure logic only (no I/O): the engine feeds it signals / prices / bars and persists the output.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import numpy as np

from .config import Settings
from .entry import EntryCfg
from .exchange.base import SymbolFilters
from .exits import ExitSpec
from .protect import Protect, breakeven_price, next_stop, partial_level, partial_split, stop_loss_usd, tighter
from .risk import levels, size_trade

log = logging.getLogger("tb.shadow")

BAR_MS = 300_000


@dataclass
class ShadowPos:
    model: str
    symbol: str
    side: str
    entry: float
    qty: float
    sl: float
    tp: float | None        # None: stop only (cross-sectional legs close on the rebalance)
    opened_at: int          # seconds
    prob: float
    entry_fee: float
    deadline: int | None = None   # time stop (seconds): closed at the first bar close / tick at or after it
    # ---- profit protection (same maths as the live engine / tb.sim): break-even, trailing stop, partial take-profit.
    # The partial's economics are FOLDED into the position's single trade row (one row per position, win = net of both parts).
    sl0: float = 0.0              # initial stop (initial risk = |entry - sl0|)
    best: float = 0.0             # best price seen
    atr: float | None = None      # ATR the trailing stop uses (None = no trailing: pct exits)
    trail_active: bool = False
    partial_done: bool = False
    init_qty: float = 0.0
    part_gross: float = 0.0
    part_fees: float = 0.0
    step_size: float = 1e-9       # lot filters, for the partial take-profit split
    min_qty: float = 0.0
    min_notional: float = 0.0


@dataclass
class ShadowPending:
    """A simulated post-only entry waiting for a later bar to trade through its limit (never persisted)."""
    model: str
    symbol: str
    side: str
    decision_open_ms: int    # open_time of the bar whose close was the decision (the limit rests from the next bar)
    prob: float
    atr: float | None
    spec: ExitSpec
    entry: EntryCfg
    f: SymbolFilters
    created: float


def exec_config(cfg: Settings, entry: EntryCfg):
    """``tb.sim.execution.ExecConfig`` for a maker entry (the SAME conservative model the labels / backtests use)."""
    from .sim import execution as X

    return X.ExecConfig(mode="maker", fee_taker=cfg.fee_taker, fee_maker=cfg.fee_maker, slippage=cfg.slippage_pct,
                        timeout_bars=max(1, math.ceil(entry.maker_timeout_s / 300.0)), reprices=entry.maker_reprices,
                        on_timeout=entry.maker_on_timeout)


def simulate_entry(o, h, l, c, idx: int, side: str, cfg: Settings, entry: EntryCfg):
    """Resolve one maker entry decided on the close of bar ``idx``.

    Returns ``("filled", bar, price, fee_rate, is_maker)``, ``("dropped",)`` (timed out, skipped) or ``("pending",)``
    (not enough bars after the decision yet). Uses ``tb.sim.execution.fill_order``: the limit rests at the decision
    close from the next bar on and fills only when a bar trades strictly THROUGH it, at the limit price, fee_maker;
    after ``ceil(maker_timeout_s / 300)`` bars it is re-posted at the then-last close, ``maker_reprices`` times; then
    skipped or (maker_on_timeout = taker) taken at the next bar's open with slippage + fee_taker."""
    from .sim import execution as X

    ec = exec_config(cfg, entry)
    fl = X.fill_order(o, h, l, c, idx, 1 if side == "long" else -1, ec)
    if fl.filled:
        return ("filled", int(fl.bar), float(fl.price), float(fl.fee_rate), fl.kind == X.KIND_MAKER)
    need = (ec.reprices + 1) * ec.timeout_bars + (1 if ec.on_timeout == "taker" else 0)
    if len(o) - 1 - idx >= need:
        return ("dropped",)
    return ("pending",)


class ShadowBook:
    """Open shadow positions (and pending maker entries), one per (model, symbol)."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.pos: dict[str, ShadowPos] = {}
        self.pending: dict[str, ShadowPending] = {}
        self.stats: dict[str, dict[str, int]] = {}          # model -> maker entry outcomes (placed / filled / dropped)

    @staticmethod
    def _key(model: str, symbol: str) -> str:
        return f"{model}|{symbol}"

    def has(self, model: str, symbol: str) -> bool:
        k = self._key(model, symbol)
        return k in self.pos or k in self.pending

    def _stat(self, model: str, what: str) -> None:
        d = self.stats.setdefault(model, {"placed": 0, "filled": 0, "dropped": 0})
        d[what] = d.get(what, 0) + 1

    def _make_pos(self, model: str, symbol: str, side: str, prob: float, fill: float, t: float, atr: float | None,
                  f: SymbolFilters, spec: ExitSpec | None, fee_rate: float) -> ShadowPos | None:
        cfg = self.cfg
        if spec is None or spec.mode == "settings":
            qty = f.round_qty(cfg.notional_usd / fill)
            if qty <= 0:
                return None
            lv = levels(cfg, side, fill, qty, atr, f)
            if lv is None:
                return None
            sl, tp = lv[0], lv[1]
        else:
            sz = size_trade(spec.effective(cfg), side, fill, spec.atr_arg(atr, fill), f, None, 1.0, None)
            if not sz.ok:
                return None
            qty, sl, tp = sz.qty, sz.sl_price, (sz.tp_price if spec.tp_enabled else None)
        dl = int(t) + int(spec.horizon_bars) * 300 if spec is not None and spec.horizon_bars else None
        if not self._side_ok(model, side, qty, fill, sl):
            return None
        trail_atr = None if (spec is not None and spec.mode == "pct") else atr
        return ShadowPos(model, symbol, side, fill, qty, sl, tp, int(t), float(prob), qty * fill * fee_rate, dl,
                         sl0=sl, best=fill, atr=trail_atr, init_qty=qty, step_size=f.step_size, min_qty=f.min_qty,
                         min_notional=f.min_notional)

    def _pc(self) -> Protect:
        return Protect.from_settings(self.cfg)

    def _side_ok(self, model: str, side: str, qty: float, entry: float, sl: float) -> bool:
        """max_side_risk_pct against the open-risk budget (max_open_risk_pct x a static equity proxy: ``capital_usd`` or
        ``paper_start_equity``): a model's open stop-out risk on ONE side, incl. the new entry, stays under the cap.
        100 % = off (the shadow book has no total open-risk cap that would bound a single side)."""
        cfg = self.cfg
        if cfg.max_side_risk_pct >= 100:
            return True
        c = (cfg.fee_taker, cfg.slippage_pct, cfg.stop_slippage_pct)
        cap = cfg.max_side_risk_pct / 100.0 * cfg.max_open_risk_pct / 100.0 * float(cfg.capital_usd or cfg.paper_start_equity)
        used = sum(stop_loss_usd(q.side, q.qty, q.entry, q.sl, *c) for q in self.pos.values()
                   if q.model == model and q.side == side)
        return used + stop_loss_usd(side, qty, entry, sl, *c, entry_fee=True) <= cap + 1e-9

    def open(self, model: str, symbol: str, side: str, prob: float, price: float, t: float,
             atr: float | None, f: SymbolFilters, spec: ExitSpec | None = None, entry: EntryCfg | None = None,
             decision_open_ms: int | None = None) -> bool:
        """Open (taker: immediately at ``price`` +slippage) or, with ``entry.entry_mode == "maker"`` and a
        ``decision_open_ms``, queue a simulated post-only entry resolved by later bars (``on_bar(..., frame=)``)."""
        if self.has(model, symbol) or price <= 0:
            return False
        cfg = self.cfg
        if entry is not None and entry.entry_mode == "maker" and decision_open_ms is not None:
            self.pending[self._key(model, symbol)] = ShadowPending(model, symbol, side, int(decision_open_ms), float(prob),
                                                                  atr, spec or ExitSpec(), entry, f, float(t))
            self._stat(model, "placed")
            return True
        fill = price * (1 + cfg.slippage_pct) if side == "long" else price * (1 - cfg.slippage_pct)
        p = self._make_pos(model, symbol, side, prob, fill, t, atr, f, spec, cfg.fee_taker)
        if p is None:
            return False
        self.pos[self._key(model, symbol)] = p
        return True

    def close_leg(self, model: str, symbol: str, price: float, t: float, reason: str = "rebalance") -> dict | None:
        """Close one position at ``price`` (market exit: adverse slippage + taker fee), e.g. a rebalance."""
        p = self.pos.get(self._key(model, symbol))
        if p is None:
            return None
        s = self.cfg.slippage_pct
        px = price * (1 - s) if p.side == "long" else price * (1 + s)
        return self._close(p, px, reason, t)

    def _close(self, p: ShadowPos, px: float, reason: str, t: float) -> dict:
        d = 1.0 if p.side == "long" else -1.0
        gross = (px - p.entry) * p.qty * d + p.part_gross
        fees = p.entry_fee + p.qty * px * self.cfg.fee_taker + p.part_fees
        del self.pos[self._key(p.model, p.symbol)]
        if reason == "sl" and p.trail_active and (p.sl - (p.sl0 or p.sl)) * d > 0:
            reason = "trail"                                      # a stop that was ratcheted (trail) is reported as such
        return {"model": p.model, "symbol": p.symbol, "side": p.side, "entry_time": p.opened_at, "exit_time": int(t),
                "entry_price": p.entry, "exit_price": px, "qty": p.init_qty or p.qty, "pnl": gross - fees, "fees": fees,
                "exit_reason": reason, "prob": p.prob}

    def _eval(self, p: ShadowPos, m: float, t: float) -> dict | None:
        long = p.side == "long"
        cfg = self.cfg
        if (long and m <= p.sl) or (not long and m >= p.sl):      # SL first (pessimistic)
            base = min(p.sl, m) if long else max(p.sl, m)
            s = cfg.slippage_pct + cfg.stop_slippage_pct          # tb.sim.template convention
            px = base * (1 - s) if long else base * (1 + s)
            return self._close(p, px, "sl", t)
        if p.tp is not None and ((long and m >= p.tp) or (not long and m <= p.tp)):
            return self._close(p, p.tp, "tp", t)                  # TP: no slippage
        return None

    def on_price(self, symbol: str, price: float, t: float | None = None) -> list[dict]:
        t = t or time.time()
        out = []
        for p in [p for p in self.pos.values() if p.symbol == symbol]:
            r = self._eval(p, price, t)
            if r is None and p.deadline and t >= p.deadline:      # time stop (labels: a timeout is closed at market)
                r = self.close_leg(p.model, p.symbol, price, t, "timeout")
            if r is None:
                self._partial(p, price)
                self._ratchet(p, price)
            if r:
                out.append(r)
        return out

    def _partial(self, p: ShadowPos, ref: float) -> None:
        """Partial take-profit when ``ref`` reached its level: ``partial_tp_frac`` is closed at the level (taker fee,
        adverse slippage), the remainder's stop moves to at least break-even. Booked into the position (folded row)."""
        if p.partial_done:
            return
        pc = self._pc()
        lvl = partial_level(p.side, p.entry, p.sl0 or p.sl, p.tp, pc)
        d = 1.0 if p.side == "long" else -1.0
        if lvl is None or (ref - lvl) * d < 0:
            return
        from .exchange.base import SymbolFilters

        split = partial_split(p.qty, lvl, pc.partial_tp_frac, SymbolFilters(0.0, p.step_size, p.min_qty, p.min_notional))
        p.partial_done = True
        if split is None:
            return
        q, rem = split
        px = lvl * (1.0 - d * self.cfg.slippage_pct)
        frac = q / p.qty
        p.part_gross += (px - p.entry) * q * d
        p.part_fees += p.entry_fee * frac + q * px * self.cfg.fee_taker
        p.entry_fee *= (1.0 - frac)
        p.qty = rem
        p.sl = tighter(p.side, p.sl, breakeven_price(p.side, p.entry, self.cfg.fee_taker))

    def _ratchet(self, p: ShadowPos, fav: float) -> None:
        """Break-even / trailing stop update from the favourable extreme just seen (effective from the next evaluation)."""
        p.best = max(p.best or fav, fav) if p.side == "long" else min(p.best or fav, fav)
        p.trail_active, p.sl, _ = next_stop(p.side, p.entry, p.sl0 or p.sl, p.sl, p.best, p.atr, p.trail_active, self._pc())

    def _bar_eval(self, p: ShadowPos, bar_open_ms: int, low: float, high: float, t: float,
                  close: float | None = None) -> dict | None:
        if bar_open_ms <= p.opened_at * 1000:
            return None
        adverse, fav = (low, high) if p.side == "long" else (high, low)
        r = self._eval(p, adverse, t)                             # stop first (pessimistic) ...
        if r is None:
            self._partial(p, fav)                                 # ... then the partial take-profit ...
            r = self._eval(p, fav, t)                             # ... then the TP for what is left
        if r is None:
            self._ratchet(p, fav)
        if r is None and p.deadline and close is not None and (bar_open_ms + BAR_MS) / 1000.0 >= p.deadline:
            r = self.close_leg(p.model, p.symbol, close, (bar_open_ms + BAR_MS) / 1000.0, "timeout")   # at that bar's close
        return r

    def _resolve_pending(self, symbol: str, frame, t: float) -> tuple[list[dict], set[str]]:
        """Fill / drop pending maker entries of ``symbol`` against the closed-bar ``frame``; positions created by a
        fill are replayed over the bars that followed the fill bar. Returns (closed trades, keys handled)."""
        out: list[dict] = []
        handled: set[str] = set()
        ot = frame["open_time"].to_numpy()
        for k, pd_ in [(k, v) for k, v in self.pending.items() if v.symbol == symbol]:
            i = int(np.searchsorted(ot, pd_.decision_open_ms))
            if i >= len(ot) or int(ot[i]) != pd_.decision_open_ms:
                if time.time() - pd_.created > 86_400 or (len(ot) and int(ot[0]) > pd_.decision_open_ms):
                    del self.pending[k]                       # decision bar left the cache
                continue
            res = simulate_entry(frame["open"].to_numpy(), frame["high"].to_numpy(), frame["low"].to_numpy(),
                                 frame["close"].to_numpy(), i, pd_.side, self.cfg, pd_.entry)
            if res[0] == "pending":
                continue
            del self.pending[k]
            if res[0] == "dropped":
                self._stat(pd_.model, "dropped")
                log.debug("shadow maker entry %s %s %s timed out", pd_.model, pd_.side, symbol)
                continue
            _, fb, px, fee_rate, _is_maker = res
            opened = (int(ot[fb]) + BAR_MS) // 1000                # booked at the END of the fill bar (SL/TP checks start after it)
            p = self._make_pos(pd_.model, symbol, pd_.side, pd_.prob, px, opened, pd_.atr, pd_.f, pd_.spec, fee_rate)
            if p is None:
                continue
            self._stat(pd_.model, "filled")
            self.pos[k] = p
            handled.add(k)
            for j in range(fb + 1, len(ot)):
                r = self._bar_eval(p, int(ot[j]), float(frame["low"].iloc[j]), float(frame["high"].iloc[j]),
                                   (int(ot[j]) + BAR_MS) / 1000.0, float(frame["close"].iloc[j]))
                if r:
                    out.append(r)
                    break
        return out, handled

    def on_bar(self, symbol: str, bar_open_ms: int, low: float, high: float, t: float | None = None,
               frame=None, close: float | None = None) -> list[dict]:
        """Replay a completed bar's wicks (adverse extreme first); only bars that opened after entry.
        With ``frame`` (the closed 5m bars incl. this one) pending maker entries are resolved first."""
        t = t or time.time()
        out: list[dict] = []
        handled: set[str] = set()
        if frame is not None and self.pending:
            out, handled = self._resolve_pending(symbol, frame, t)
        for k, p in [(k, p) for k, p in self.pos.items() if p.symbol == symbol and k not in handled]:
            r = self._bar_eval(p, bar_open_ms, low, high, t, close)
            if r:
                out.append(r)
        return out

    def export_state(self) -> dict[str, Any]:
        return {k: asdict(v) for k, v in self.pos.items()}

    def import_state(self, st: Mapping[str, Any]) -> None:
        names = {f.name for f in fields(ShadowPos)}
        self.pos = {k: ShadowPos(**{a: b for a, b in v.items() if a in names}) for k, v in st.items()}
        for p in self.pos.values():                               # positions saved before profit protection existed
            if not p.sl0:
                p.sl0, p.best = p.sl, p.entry
            if not p.init_qty:
                p.init_qty = p.qty


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
