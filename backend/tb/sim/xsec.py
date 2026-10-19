"""Market-neutral cross-sectional portfolio simulator (V3 experiment A, with the D execution models).

Every ``rebalance_every_h`` hours, on the CLOSE of hour k, the tradable universe is ranked by ``score[k]``; the top
``top_k`` coins are held long, the bottom ``bottom_k`` short, each leg with an exchange-style stop.

Template (docs/ARCHITECTURE.md, adapted to a portfolio):

* **Timing.**  Decision on the close of hour k (data of hours <= k only).  Orders are executed on the 5m grid by
  ``tb.sim.execution`` starting with the 5m bar after the signal bar: taker = open of the next 5m bar + slippage,
  maker = post-only limit at the last price that fills only when a later bar trades THROUGH it (timeout -> skip the
  entry; an exit falls back to taker).  Fees: taker 0.05 %/side (+ slippage), maker 0.02 %/side.
* **Legs.**  A coin that stays in the same target set at the next rebalance keeps its position (no trade, no fee,
  its original stop stays); a coin that leaves is closed, a new one is opened.  A leg lives at most ``max_hold_h``.
* **Stop per leg.**  Distance = ``stop_vol_mult`` x the coin's trailing 7 d daily volatility (clipped to
  [stop_min, stop_max]), placed from the actual fill.  Checked on every 5m bar from the fill bar on (the fill bar
  included: pessimistic), filled at the stop or at the bar open when it gaps through, with ``slippage +
  stop_slippage_extra`` adverse and a taker fee.  There is no take-profit; legs are closed by the next rebalance.
* **Equal risk.**  Notional per leg = risk / (stop distance + round-trip taker cost); risk per leg =
  ``open_risk_pct`` / (number of target legs) of equity x the drawdown derate multiplier (``tb.risk``); the sum of
  loss-if-every-stop-fills never exceeds ``open_risk_pct`` of equity (the engine's open-risk cap); total notional
  <= ``max_total_leverage`` x equity; per-leg leverage from ``tb.risk.choose_leverage`` (liquidation >= 1.5 x stop
  distance); free margin must cover each new leg.
* **Funding.**  Settlements (00/08/16 UTC) are charged to legs opened strictly before the settlement hour.
* **Kill switch.**  Drawdown from the peak >= ``kill_dd_pct`` closes everything at the hour close and stays off.
* **PnL convention.**  trade ``pnl`` = net (gross - fees - funding); ``fees`` >= 0 cost; ``funding`` positive = paid.
* Positions still open on the last hour are closed at its close (taker), so a run is fully realised.

Known simplifications: hourly equity marks; no market impact beyond ``slippage`` (the universe is liquidity-filtered,
leg notionals are small); stops cannot be gapped through a liquidation price (stops sit well inside it).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from . import safety
from .execution import KIND_MAKER, ExecConfig, fill_orders, low_vol_gate

HOUR_MS = 3_600_000
SUB = 12
DAY_MS = 86_400_000
WINDOW_5M = 48                       # 5m bars loaded after a decision for execution (>= (reprices+1)*timeout + 1)


@dataclass
class XsecConfig:
    start_equity: float = 100.0
    rebalance_every_h: int = 24
    top_k: int = 5
    bottom_k: int = 5
    entry_mode: str = "taker"                    # taker | maker (entries)
    exit_mode: str | None = None                 # None = same as entry_mode (exits fall back to taker on timeout)
    entry_on_timeout: str = "skip"               # skip | taker (a maker entry that never fills)
    stop_vol_mult: float = 2.0
    stop_min: float = 0.02
    stop_max: float = 0.20
    stop_slippage_extra: float = 0.0005
    open_risk_pct: float = 20.0
    max_total_leverage: float = 20.0
    max_leverage: int = 20
    min_notional_usd: float = 5.0
    kill_dd_pct: float = 50.0
    derate_enabled: bool = True
    apply_funding: bool = True
    min_universe: int = 0                        # 0 = max(10, 2 * (top_k + bottom_k))
    max_hold_h: int = 168
    keep_existing: bool = True
    gate_min_ratio: float = 0.0                  # >0: skip a leg when the coin's typical move over one rebalance period < ratio x round-trip cost
    dollar_neutral: bool = True                  # shrink the heavier side's new legs so long = short notional
    close_at_end: bool = True
    exec: ExecConfig = field(default_factory=ExecConfig)

    def exec_for(self, mode: str, on_timeout: str) -> ExecConfig:
        return replace(self.exec, mode=mode, on_timeout=on_timeout)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class _Leg:
    __slots__ = ("s", "side", "qty", "entry", "entry_bar", "entry_fee", "stop", "stop_bar", "stop_px", "expire_bar",
                 "notional", "margin", "funding", "last", "k_open", "maker")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


@dataclass
class XsecResult:
    trades: list[dict]
    ts: np.ndarray               # close time (ms) of every simulated hour
    equity: np.ndarray           # equity at each hour close
    metrics: dict
    stats: dict
    cfg: XsecConfig
    daily_ret: np.ndarray        # per UTC-day returns of equity
    kill_time: int | None = None


def sigma_day(panel) -> np.ndarray:
    """[T, S] trailing 168 h volatility of hourly log returns scaled to a day (fraction, not percent); row k uses
    hours <= k."""
    cached = getattr(panel, "_sigma_day", None)
    if cached is not None:
        return cached
    c = pd.DataFrame(panel.close).ffill(limit=3)
    lr = np.log(c.where(c > 0)).diff()
    sd = lr.rolling(168, min_periods=120).std().to_numpy() * math.sqrt(24.0)
    try:
        panel._sigma_day = sd
    except Exception:  # pragma: no cover
        pass
    return sd


def _window(panel, s: int, lo: int, n: int):
    hi = min(panel.o5.shape[1], lo + n)
    return tuple(np.asarray(a[s, lo:hi], dtype="float64") for a in (panel.o5, panel.h5, panel.l5, panel.c5))


def _metrics(trades: list[dict], ts: np.ndarray, eq: np.ndarray, cfg: XsecConfig, stats: dict,
             expo: np.ndarray, gross_lev: np.ndarray) -> tuple[dict, np.ndarray]:
    start = float(cfg.start_equity)
    end = float(eq[-1]) if len(eq) else start
    n_h = len(ts)
    days = n_h / 24.0
    m: dict = {"start_equity": start, "end_equity": end, "net_return_pct": (end / start - 1.0) * 100.0, "days": days}
    if n_h:
        run_peak = np.maximum.accumulate(np.maximum(eq, start))
        m["max_drawdown_pct"] = float(((run_peak - eq) / run_peak).max() * 100.0)
    else:
        m["max_drawdown_pct"] = 0.0
    # per UTC-day equity (last hour of each day), prefixed by the start equity
    if n_h:
        day = ts // DAY_MS
        last = np.nonzero(np.diff(day, append=day[-1] + 1))[0]
        de = np.concatenate([[start], eq[last]])
        r = np.diff(de) / np.where(de[:-1] > 0, de[:-1], np.nan)
        r = r[np.isfinite(r)]
    else:
        r = np.array([])
    m["sharpe"] = float(r.mean() / r.std(ddof=1) * math.sqrt(365)) if len(r) > 2 and r.std(ddof=1) > 0 else float("nan")
    dn = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2))) if len(r) else 0.0
    m["sortino"] = float(r.mean() / dn * math.sqrt(365)) if dn > 0 else float("nan")
    m["cagr_pct"] = float(((end / start) ** (365.0 / days) - 1.0) * 100.0) if days > 0 and end > 0 else float("nan")
    n = len(trades)
    m["n_trades"] = n
    m["trades_per_day"] = n / days if days > 0 else 0.0
    traded = stats["traded_notional"]
    mean_eq = float(np.mean(eq)) if n_h else start
    m["turnover_x_per_day"] = traded / days / mean_eq if days > 0 and mean_eq > 0 else 0.0
    m["avg_gross_leverage"] = float(gross_lev.mean()) if len(gross_lev) else 0.0
    m["avg_net_exposure_frac"] = float(np.mean(np.abs(expo))) if len(expo) else 0.0       # |net| / gross while invested
    if n:
        pnl = np.array([t["pnl"] for t in trades])
        m["win_rate"] = float((pnl > 0).mean())
        m["expectancy"] = float(pnl.mean())
        m["total_pnl"] = float(pnl.sum())
        m["fees_paid"] = float(sum(t["fees"] for t in trades))
        m["funding_paid"] = float(sum(t["funding"] for t in trades))
        m["gross_pnl"] = float(sum(t["gross_pnl"] for t in trades))
        m["avg_notional"] = float(np.mean([t["notional"] for t in trades]))
        m["avg_hold_h"] = float(np.mean([t["hold_h"] for t in trades]))
        reasons: dict[str, int] = {}
        for t in trades:
            reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
        m["exit_reasons"] = reasons
        m["stop_rate"] = reasons.get("sl", 0) / n
        m["long_trades"] = sum(1 for t in trades if t["side"] == "long")
        m["short_trades"] = n - m["long_trades"]
    else:
        for k in ("win_rate", "expectancy", "total_pnl", "fees_paid", "funding_paid", "gross_pnl", "avg_notional",
                  "avg_hold_h", "stop_rate"):
            m[k] = 0.0
        m["exit_reasons"] = {}
        m["long_trades"] = m["short_trades"] = 0
    att = stats["entries"] + stats["entry_no_fill"]
    m["entry_fill_rate"] = stats["entries"] / att if att else float("nan")
    return m, r


def run_xsec(panel, score: np.ndarray, cfg: XsecConfig | None = None, k_start: int = 0, k_end: int | None = None
             ) -> XsecResult:
    """Simulate hours ``k_start .. k_end-1``; ``score`` [T, S] (NaN = not tradable / unscored).  The first
    decision is on the close of hour ``k_start``; legs live from the next hour; the run is flat at the end of
    hour ``k_end - 1``."""
    from tb import risk as _risk

    cfg = cfg or XsecConfig()
    T, S = panel.T, panel.S
    k_end = T if k_end is None else min(int(k_end), T)
    k_start = max(0, int(k_start))
    if k_end - k_start < 2:
        raise ValueError("need at least 2 hours")
    if score.shape != (T, S):
        raise ValueError(f"score must be [T={T}, S={S}], got {score.shape}")
    R = max(1, int(cfg.rebalance_every_h))
    min_univ = cfg.min_universe or max(10, 2 * (cfg.top_k + cfg.bottom_k))
    xm_in = cfg.entry_mode
    xm_out = cfg.exit_mode or cfg.entry_mode
    e_in = cfg.exec_for(xm_in, cfg.entry_on_timeout).validate()
    e_out = cfg.exec_for(xm_out, "taker").validate()
    fee_t, slip = cfg.exec.fee_taker, cfg.exec.slippage
    sslip = slip + cfg.stop_slippage_extra
    cost_frac = 2.0 * fee_t + slip + sslip              # conservative round trip (taker both sides + stop slippage)
    sig_d = sigma_day(panel)
    N5 = panel.o5.shape[1]
    stats = {k: 0 for k in ("rebalances", "rebalances_skipped_small_universe", "entry_attempts", "entries",
                            "entry_no_fill", "skip_risk", "skip_min_notional", "skip_leverage", "skip_margin",
                            "skip_no_vol", "exit_fallback", "kill_closes", "kept_legs", "maker_entries",
                            "maker_exits", "taker_exits", "derated_rebalances", "stops", "skip_gate")}
    stats["traded_notional"] = 0.0
    stats["min_derate_mult"] = 1.0

    wallet = float(cfg.start_equity)
    peak = wallet
    legs: list[_Leg] = []
    trades: list[dict] = []
    killed = False
    kill_time = None
    n_h = k_end - k_start
    eq = np.empty(n_h)
    ts = (panel.t0 + (np.arange(k_start, k_end, dtype="int64") + 1) * HOUR_MS)
    expo = []
    glev = np.zeros(n_h)

    def mark(leg: _Leg, k: int) -> float:
        c = panel.close[k, leg.s]
        if np.isfinite(c):
            leg.last = float(c)
        return leg.last

    def close_leg(leg: _Leg, px: float, bar: int, reason: str, fee_rate: float):
        nonlocal wallet
        gross = leg.side * leg.qty * (px - leg.entry)
        exit_fee = fee_rate * leg.qty * px
        wallet += gross - exit_fee
        fees = leg.entry_fee + exit_fee
        net = gross - fees - leg.funding
        stats["traded_notional"] += leg.qty * px
        trades.append({
            "id": len(trades) + 1, "symbol": panel.syms[leg.s], "side": "long" if leg.side > 0 else "short",
            "entry_price": leg.entry, "exit_price": px, "qty": leg.qty,
            "opened_at": int(panel.t0 + leg.entry_bar * 300_000) // 1000,
            "closed_at": int(panel.t0 + bar * 300_000) // 1000,
            "pnl": net, "fees": fees, "funding": leg.funding, "exit_reason": reason, "gross_pnl": gross,
            "notional": leg.notional, "hold_h": (bar - leg.entry_bar) / 12.0, "k_open": leg.k_open,
            "maker_entry": bool(leg.maker),
        })

    def stop_scan(s: int, side: int, fb: int, stop: float):
        """(stop_bar, fill_px) of the first 5m bar from fb on that touches the stop within max_hold_h, else (big, nan)."""
        w = cfg.max_hold_h * SUB
        hi = min(N5, fb + w)
        if side > 0:
            arr = np.asarray(panel.l5[s, fb:hi])
            hit = arr <= stop
        else:
            arr = np.asarray(panel.h5[s, fb:hi])
            hit = arr >= stop
        if not hit.any():
            return 1 << 60, float("nan")
        sb = fb + int(np.argmax(hit))
        ob = float(panel.o5[s, sb]) if sb > fb else float("nan")
        if side > 0:
            base = min(stop, ob) if np.isfinite(ob) else stop
            px = base * (1.0 - sslip)
        else:
            base = max(stop, ob) if np.isfinite(ob) else stop
            px = base * (1.0 + sslip)
        return sb, px

    def unreal(leg: _Leg) -> float:
        return leg.side * leg.qty * (leg.last - leg.entry)

    for k in range(k_start, k_end):
        hr0, hr1 = k * SUB, (k + 1) * SUB
        # ---- 1. funding settled at the start of this hour (legs opened before this hour)
        if cfg.apply_funding and legs:
            for leg in legs:
                rate = panel.fund_ev[k, leg.s]
                if rate != 0.0 and leg.entry_bar // SUB < k and np.isfinite(rate):
                    px = panel.open[k, leg.s]
                    px = float(px) if np.isfinite(px) else leg.last
                    pay = leg.side * float(rate) * leg.qty * px
                    leg.funding += pay
                    wallet -= pay
        # ---- 2. stops inside this hour
        if legs:
            keep = []
            for leg in legs:
                if leg.stop_bar < hr1:
                    stats["stops"] += 1
                    close_leg(leg, leg.stop_px, leg.stop_bar, "sl", fee_t)
                else:
                    keep.append(leg)
            legs = keep
        # ---- 3. mark to market, peak, kill switch
        for leg in legs:
            mark(leg, k)
        equity_now = wallet + sum(unreal(leg) for leg in legs)
        if equity_now > peak:
            peak = equity_now
        if legs:
            gl = sum(leg.qty * leg.last for leg in legs)
            nl = sum(leg.side * leg.qty * leg.last for leg in legs)
            expo.append(nl / gl if gl > 0 else 0.0)
            glev[k - k_start] = gl / max(equity_now, 1e-9)
        if (not killed and cfg.kill_dd_pct < 100 and peak > 0
                and (peak - equity_now) / peak * 100.0 >= cfg.kill_dd_pct):
            killed = True
            kill_time = int(ts[k - k_start])
            for leg in legs:
                stats["kill_closes"] += 1
                close_leg(leg, leg.last * (1.0 - leg.side * slip), hr1 - 1, "kill", fee_t)
            legs = []
            equity_now = wallet
        last_hour = k == k_end - 1
        # ---- 4. rebalance on the close of this hour
        if (not killed and not last_hour and (k - k_start) % R == 0):
            sc = score[k]
            elig = np.isfinite(sc) & np.isfinite(panel.close[k])
            n_el = int(elig.sum())
            if n_el < max(min_univ, cfg.top_k + cfg.bottom_k):
                stats["rebalances_skipped_small_universe"] += 1
            else:
                stats["rebalances"] += 1
                i5 = hr1 - 1
                idx = np.nonzero(elig)[0]
                order = idx[np.lexsort((idx, -sc[idx]))]            # best first, ties by index
                longs = [int(x) for x in order[:cfg.top_k]]
                shorts = [int(x) for x in order[::-1][:cfg.bottom_k] if int(x) not in longs]
                target = {s: 1 for s in longs}
                target.update({s: -1 for s in shorts})
                # exits
                remaining = []
                for leg in legs:
                    keep_it = (cfg.keep_existing and target.get(leg.s) == leg.side and hr1 < leg.expire_bar)
                    if keep_it:
                        stats["kept_legs"] += 1
                        remaining.append(leg)
                        continue
                    o, h, l, c = _window(panel, leg.s, i5, WINDOW_5M)
                    r = fill_orders(o, h, l, c, [0], -leg.side, e_out, "taker")
                    if bool(r["filled"][0]):
                        fb = i5 + int(r["bar"][0])
                        if leg.stop_bar <= fb:
                            close_leg(leg, leg.stop_px, leg.stop_bar, "sl", fee_t)
                            stats["stops"] += 1
                        else:
                            close_leg(leg, float(r["price"][0]), fb, "rebalance", float(r["fee_rate"][0]))
                            stats["maker_exits" if int(r["kind"][0]) == KIND_MAKER else "taker_exits"] += 1
                    else:                                              # data hole: close at the last known price
                        stats["exit_fallback"] += 1
                        close_leg(leg, leg.last * (1.0 - leg.side * slip), i5, "rebalance", fee_t)
                legs = remaining
                held = {leg.s for leg in legs}
                new = [(s, d) for s, d in target.items() if s not in held]
                if new:
                    for leg in legs:
                        mark(leg, k)
                    equity_now = wallet + sum(unreal(leg) for leg in legs)
                    dd = max(0.0, (peak - equity_now) / peak * 100.0) if peak > 0 else 0.0
                    scale = safety.derate_multiplier(dd, cfg.kill_dd_pct) if cfg.derate_enabled else 1.0
                    if scale < 1.0:
                        stats["derated_rebalances"] += 1
                        stats["min_derate_mult"] = min(stats["min_derate_mult"], scale)
                    n_t = len(target)
                    risk_total = cfg.open_risk_pct / 100.0 * equity_now * scale
                    per_leg = risk_total / n_t
                    head = risk_total - sum(max(0.0, leg.side * leg.qty * (leg.last - leg.stop))
                                            + cost_frac * leg.qty * leg.last for leg in legs)
                    cap_n = cfg.max_total_leverage * equity_now / n_t
                    margin_used = sum(leg.margin for leg in legs)
                    plan = []                                           # [s, side, stop_dist, notional, ref]
                    for s, d in sorted(new, key=lambda x: (-x[1], x[0])):
                        stats["entry_attempts"] += 1
                        sg = sig_d[k, s]
                        ref = float(panel.c5[s, i5])
                        if not (np.isfinite(sg) and sg > 0 and np.isfinite(ref) and ref > 0):
                            stats["skip_no_vol"] += 1
                            continue
                        if cfg.gate_min_ratio > 0 and not low_vol_gate(float(sg) / math.sqrt(24.0 * SUB), e_in, R * SUB,
                                                                       cfg.gate_min_ratio):
                            stats["skip_gate"] += 1
                            continue
                        sd = min(max(cfg.stop_vol_mult * float(sg), cfg.stop_min), cfg.stop_max)
                        risk = min(per_leg, head)
                        if risk <= 0:
                            stats["skip_risk"] += 1
                            continue
                        notional = min(risk / (sd + cost_frac), cap_n)
                        head -= notional * (sd + cost_frac)
                        plan.append([s, d, sd, notional, ref])
                    if cfg.dollar_neutral and plan:
                        # shrink the heavier side's NEW legs so that long and short notional match after entry
                        lx = sum(leg.qty * leg.last for leg in legs if leg.side > 0)
                        sx = sum(leg.qty * leg.last for leg in legs if leg.side < 0)
                        ln = sum(p_[3] for p_ in plan if p_[1] > 0)
                        sn = sum(p_[3] for p_ in plan if p_[1] < 0)
                        if lx + ln > sx + sn and ln > 0:
                            f = min(1.0, max(0.0, sx + sn - lx) / ln)
                            for p_ in plan:
                                if p_[1] > 0:
                                    p_[3] *= f
                        elif sx + sn > lx + ln and sn > 0:
                            f = min(1.0, max(0.0, lx + ln - sx) / sn)
                            for p_ in plan:
                                if p_[1] < 0:
                                    p_[3] *= f
                    for s, d, sd, notional, ref in plan:
                        if notional < cfg.min_notional_usd:
                            stats["skip_min_notional"] += 1
                            continue
                        side_s = "long" if d > 0 else "short"
                        lev = _risk.choose_leverage(side_s, ref, ref * (1.0 - d * sd), notional, cfg.max_leverage)
                        if lev is None:
                            stats["skip_leverage"] += 1
                            continue
                        o, h, l, c = _window(panel, s, i5, WINDOW_5M)
                        r = fill_orders(o, h, l, c, [0], d, e_in, cfg.entry_on_timeout)
                        if not bool(r["filled"][0]):
                            stats["entry_no_fill"] += 1
                            continue
                        fb = i5 + int(r["bar"][0])
                        px = float(r["price"][0])
                        fee_rate = float(r["fee_rate"][0])
                        qty = notional / px
                        margin = notional / lev
                        efee = fee_rate * notional
                        if wallet - margin_used < margin + efee:
                            stats["skip_margin"] += 1
                            continue
                        stop = px * (1.0 - d * sd)
                        sb, spx = stop_scan(s, d, fb, stop)
                        wallet -= efee
                        stats["traded_notional"] += notional
                        stats["entries"] += 1
                        stats["maker_entries"] += int(int(r["kind"][0]) == KIND_MAKER)
                        margin_used += margin
                        legs.append(_Leg(s=s, side=d, qty=qty, entry=px, entry_bar=fb, entry_fee=efee, stop=stop,
                                         stop_bar=sb, stop_px=spx, expire_bar=fb + cfg.max_hold_h * SUB,
                                         notional=notional, margin=margin, funding=0.0, last=px, k_open=k,
                                         maker=int(r["kind"][0]) == KIND_MAKER))
        # ---- 5. final flat
        if last_hour and legs and cfg.close_at_end:
            for leg in legs:
                px = mark(leg, k)
                close_leg(leg, px * (1.0 - leg.side * slip), hr1 - 1, "end", fee_t)
            legs = []
            equity_now = wallet
        elif last_hour:
            for leg in legs:
                mark(leg, k)
            equity_now = wallet + sum(unreal(leg) for leg in legs)
        else:
            equity_now = wallet + sum(unreal(leg) for leg in legs)
        eq[k - k_start] = equity_now
        if equity_now > peak:
            peak = equity_now
    stats["open_at_end"] = len(legs)
    expo_a = np.array(expo)
    m, daily = _metrics(trades, ts, eq, cfg, stats, expo_a, glev[glev > 0])
    return XsecResult(trades=trades, ts=ts, equity=eq, metrics=m, stats=stats, cfg=cfg, daily_ret=daily,
                      kill_time=kill_time)


# --------------------------------------------------------------------------------------------
# helpers for comparing a model with its baselines
# --------------------------------------------------------------------------------------------
def pooled_metrics(results: list[XsecResult]) -> dict:
    """Chain independent runs (each started from its own equity) into one compounded return / daily series."""
    if not results:
        return {}
    rets = np.concatenate([r.daily_ret for r in results]) if results else np.array([])
    eq = np.cumprod(1.0 + rets) if len(rets) else np.array([1.0])
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = float(((peak - eq) / peak).max() * 100.0) if len(eq) else 0.0
    sh = float(rets.mean() / rets.std(ddof=1) * math.sqrt(365)) if len(rets) > 2 and rets.std(ddof=1) > 0 else float("nan")
    days = sum(r.metrics["days"] for r in results)
    tr = [t for r in results for t in r.trades]
    out = {
        "net_return_pct": float((eq[-1] - 1.0) * 100.0) if len(eq) else 0.0,
        "sharpe": sh, "max_drawdown_pct": dd, "days": days, "n_trades": len(tr),
        "trades_per_day": len(tr) / days if days else 0.0,
        "fees_paid": float(sum(r.metrics["fees_paid"] for r in results)),
        "funding_paid": float(sum(r.metrics["funding_paid"] for r in results)),
        "gross_pnl": float(sum(r.metrics["gross_pnl"] for r in results)),
        "turnover_x_per_day": float(np.mean([r.metrics["turnover_x_per_day"] for r in results])),
        "win_rate": float(np.mean([t["pnl"] > 0 for t in tr])) if tr else 0.0,
        "expectancy": float(np.mean([t["pnl"] for t in tr])) if tr else 0.0,
        "kills": int(sum(r.kill_time is not None for r in results)),
    }
    return out


def z_vs_random(model_return_pct: float, random_returns_pct) -> float:
    r = np.asarray(random_returns_pct, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) <= 0:
        return float("nan")
    return float((model_return_pct - r.mean()) / r.std(ddof=1))
