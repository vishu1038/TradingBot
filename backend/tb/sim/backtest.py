"""Event-driven multi-symbol 5m backtester implementing the shared trade template.

Per global 5m step (union of all symbols' bars) the order of events is:
  1. funding settlements that fall inside the bar, charged to positions opened BEFORE the settlement
  2. entries: signals from the previous bar's close fill at THIS bar's open (no look-ahead),
     highest probability first, subject to: kill switch, one-position-per-symbol, plausibility
     exclusion, slot cap floor(equity/slice), free margin, correlation cap, derating/drift sizing
  3. exits for every open position on this bar's OHLC (liquidation, then stop, then TP => SL first)
  4. mark-to-market, peak/drawdown, kill switch (closes everything at the bar close)
Trailing-stop updates use the bar's high/low only from the NEXT bar on (intrabar order unknown).
See tb.sim.template for the fill model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import safety, signals as sig
from .template import SimConfig, atr_wilder, liq_price, size_and_levels

BAR_MS = 300_000


@dataclass
class Position:
    id: int
    sym: str
    sidx: int
    side: int
    entry_gi: int
    entry_time: int
    entry_price: float       # slipped fill
    qty: float
    notional: float
    margin: float
    tp: float
    sl: float                # current stop (ratchets)
    sl0: float
    liq: float
    atr: float
    prob: float
    scale: float
    entry_fee: float
    shadow: bool = False
    best: float = 0.0
    trail_active: bool = False
    funding: float = 0.0     # positive = paid
    last: float = 0.0
    bars: int = 0


@dataclass
class BacktestResult:
    trades: list[dict]
    ts: np.ndarray           # global 5m timeline (ms)
    equity: np.ndarray       # equity at each step (MTM at bar close)
    cfg: SimConfig
    metrics: dict
    stats: dict              # safety-layer / skip counters
    symbols: list[str]
    kill_time: int | None = None


def _corr(a: np.ndarray, b: np.ndarray, lo: int, hi: int, min_obs: int):
    x, y = a[lo:hi], b[lo:hi]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_obs:
        return None
    x, y = x[ok], y[ok]
    sx, sy = x.std(), y.std()
    if sx <= 0 or sy <= 0:
        return None
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def run_backtest(klines: dict[str, pd.DataFrame], signals, cfg: SimConfig | None = None,
                 funding: dict[str, pd.DataFrame] | None = None) -> BacktestResult:
    cfg = cfg or SimConfig()
    syms = sorted(klines)
    S = len(syms)
    if S == 0:
        raise ValueError("no symbols")
    use_probs = sig.carries_probabilities(signals)
    stats = {k: 0 for k in (
        "signals", "entries", "skip_kill", "skip_has_position", "skip_fault", "skip_no_slot",
        "skip_no_margin", "skip_corr", "skip_no_atr", "skip_min_notional", "skip_drift_block",
        "derated_trades", "drift_half_trades", "shadow_trades", "shadow_wins", "kill_closes",
        "faults_bars", "fault_episodes", "fault_excluded_bars", "drift_fired_half",
        "drift_fired_block")}
    stats["derate_source"] = safety.DERATE_SOURCE
    stats["min_derate_mult"] = 1.0
    stats["fault_by_symbol"] = {}

    # ---- per-symbol arrays
    T, O, H, L, C, ATR, PL, PS, EXC = [], [], [], [], [], [], [], [], []
    for s in syms:
        kl = klines[s]
        t = kl["open_time"].to_numpy("int64")
        o, h, l, c = (kl[k].to_numpy("float64") for k in ("open", "high", "low", "close"))
        atr = atr_wilder(h, l, c, cfg.atr_period)
        pl, ps = sig.resolve(signals, s, kl)
        if cfg.plausibility_enabled:
            exc, fc = safety.symbol_fault_state(kl, atr, cfg.fault_mature_n, cfg.fault_recover_m)
            stats["faults_bars"] += fc["fault_bars"]
            stats["fault_episodes"] += fc["episodes"]
            stats["fault_excluded_bars"] += fc["excluded_bars"]
            if fc["fault_bars"]:
                stats["fault_by_symbol"][s] = fc
        else:
            exc = np.zeros(len(t), bool)
        T.append(t); O.append(o); H.append(h); L.append(l); C.append(c)
        ATR.append(atr); PL.append(pl); PS.append(ps); EXC.append(exc)
    ts = np.unique(np.concatenate(T))
    nT = len(ts)
    gi = [np.searchsorted(ts, t) for t in T]
    row_at = np.full((S, nT), -1, dtype=np.int32)
    for k in range(S):
        row_at[k, gi[k]] = np.arange(len(T[k]), dtype=np.int32)

    # ---- return matrix for the correlation cap (NaN across gaps)
    rets = np.full((S, nT), np.nan)
    for k in range(S):
        c, t = C[k], T[k]
        if len(c) > 1:
            r = c[1:] / c[:-1] - 1.0
            r[np.diff(t) != BAR_MS] = np.nan
            rets[k, gi[k][1:]] = r

    # ---- funding events mapped to global step index: {sidx: {gi: rate}}
    fund_at: list[dict[int, float]] = [dict() for _ in range(S)]
    if cfg.apply_funding and funding:
        for k, s in enumerate(syms):
            f = funding.get(s)
            if f is None or len(f) == 0:
                continue
            ft = f["calc_time"].to_numpy("int64")
            fr = f["funding_rate"].to_numpy("float64")
            g = np.searchsorted(ts, ft, side="right") - 1
            for gg, tt, rr in zip(g, ft, fr):
                if gg >= 0 and tt < ts[gg] + BAR_MS and math.isfinite(rr):
                    fund_at[k][int(gg)] = fund_at[k].get(int(gg), 0.0) + float(rr)

    # ---- entry events (signal row r -> fill at row r+1)
    ev_gi, ev_prob, ev_s, ev_row, ev_side = [], [], [], [], []
    for k in range(S):
        pl, ps = PL[k], PS[k]
        long_ok = pl >= cfg.thr_long
        short_ok = (ps >= cfg.thr_short) if cfg.short_enabled else np.zeros(len(pl), bool)
        both = long_ok & short_ok
        side = np.where(both, np.where(pl - cfg.thr_long >= ps - cfg.thr_short, 1, -1),
                        np.where(long_ok, 1, -1))
        rows = np.nonzero(long_ok | short_ok)[0]
        rows = rows[rows + 1 < len(T[k])]
        rows = rows[T[k][rows + 1] - T[k][rows] == BAR_MS]
        if len(rows) == 0:
            continue
        sd = side[rows]
        pr = np.where(sd > 0, pl[rows], ps[rows])
        ev_gi.append(gi[k][rows + 1]); ev_prob.append(pr); ev_s.append(np.full(len(rows), k))
        ev_row.append(rows + 1); ev_side.append(sd)
    if ev_gi:
        ev_gi = np.concatenate(ev_gi); ev_prob = np.concatenate(ev_prob); ev_s = np.concatenate(ev_s)
        ev_row = np.concatenate(ev_row); ev_side = np.concatenate(ev_side)
        order = np.lexsort((-ev_prob, ev_gi))
        ev_gi, ev_prob, ev_s, ev_row, ev_side = (a[order] for a in (ev_gi, ev_prob, ev_s, ev_row, ev_side))
    else:
        ev_gi = np.array([], dtype=np.int64); ev_prob = ev_s = ev_row = ev_side = ev_gi
    stats["signals"] = int(len(ev_gi))
    n_ev = len(ev_gi)

    # ---- state
    wallet = float(cfg.start_equity)
    peak = wallet
    eq = np.empty(nT)
    open_pos: list[Position] = []
    trades: list[dict] = []
    drift = safety.DriftMonitor(cfg.drift_n) if (cfg.drift_enabled and use_probs) else None
    killed = False
    kill_time = None
    open_steps = 0
    next_id = 1
    ep = 0                          # event pointer
    fee = cfg.fee_taker
    slip = cfg.slippage
    sslip = cfg.slippage + cfg.stop_slippage_extra

    def unreal(p: Position) -> float:
        return p.side * p.qty * (p.last - p.entry_price)

    def close(p: Position, price: float, reason: str, t_exit: int, liq: bool = False):
        nonlocal wallet, next_id
        gross = p.side * p.qty * (price - p.entry_price)
        exit_fee = fee * p.qty * price
        liq_fee = 0.0
        if liq:   # isolated: the loss is capped at the margin and the whole margin is lost
            gross = max(gross, -p.margin)
            exit_fee = min(exit_fee, max(0.0, p.margin + gross))
            liq_fee = p.margin + gross - exit_fee
        fees = p.entry_fee + exit_fee + liq_fee
        net = gross - fees - p.funding
        win = net > 0
        if p.shadow:
            stats["shadow_trades"] += 1
            stats["shadow_wins"] += int(win)
            if drift is not None:
                drift.add(p.prob, win)
            return
        wallet += gross - exit_fee - liq_fee
        trades.append({
            "id": len(trades) + 1, "symbol": p.sym, "side": "long" if p.side > 0 else "short",
            "entry_price": p.entry_price, "exit_price": price, "qty": p.qty,
            "opened_at": p.entry_time // 1000, "closed_at": t_exit // 1000,
            "pnl": net, "fees": fees, "funding": p.funding, "exit_reason": reason, "prob": p.prob,
            "gross_pnl": gross, "notional": p.notional, "bars_held": p.bars, "scale": p.scale,
        })
        if drift is not None:
            drift.add(p.prob, win)

    ti = 0
    while ti < nT:
        # ---- fast-forward flat periods
        if not open_pos:
            nxt = int(ev_gi[ep]) if ep < n_ev else nT
            if killed:
                stats["skip_kill"] += n_ev - ep
                ep = n_ev
                nxt = nT
            if nxt > ti:
                eq[ti:nxt] = wallet
                ti = nxt
                continue
        T0 = int(ts[ti])
        # 1. funding
        if cfg.apply_funding:
            for p in open_pos:
                if p.shadow:
                    continue
                rate = fund_at[p.sidx].get(ti)
                if rate is not None and p.entry_gi < ti:
                    r = row_at[p.sidx, ti]
                    if r >= 0:
                        pay = p.side * rate * p.qty * O[p.sidx][r]
                        p.funding += pay
                        wallet -= pay
        # 2. entries
        while ep < n_ev and ev_gi[ep] == ti:
            k = int(ev_s[ep]); r = int(ev_row[ep]); side = int(ev_side[ep]); prob = float(ev_prob[ep])
            ep += 1
            if killed:
                stats["skip_kill"] += 1
                continue
            if cfg.one_per_symbol and any(p.sidx == k for p in open_pos):
                stats["skip_has_position"] += 1
                continue
            if EXC[k][r - 1]:
                stats["skip_fault"] += 1
                continue
            o = O[k][r]
            real = [p for p in open_pos if not p.shadow]
            atr_sig = ATR[k][r - 1]
            fill = o * (1.0 + side * slip)
            if drift is not None and drift.blocked:
                # no real entries; run the trade as a shadow so the monitor can recover
                lv = size_and_levels(o, fill, side, atr_sig, cfg, 1.0)
                stats["skip_drift_block"] += 1
                if lv is not None:
                    qty, tp, sl = lv
                    open_pos.append(Position(0, syms[k], k, side, ti, T0, fill, qty, qty * fill, 0.0, tp, sl,
                                             sl, liq_price(fill, side, cfg.leverage, cfg.mmr),
                                             float(atr_sig), prob, 1.0, 0.0, shadow=True,
                                             best=fill, last=fill))
                continue
            equity_now = wallet + sum(unreal(p) for p in real)
            n_slots = min(cfg.max_positions, int(math.floor(equity_now / cfg.slice_margin_usd + 1e-9)))
            if len(real) >= n_slots:
                stats["skip_no_slot"] += 1
                continue
            scale = 1.0
            if cfg.derate_enabled and cfg.kill_dd_pct > 0:
                ref = float(cfg.start_equity) if cfg.dd_basis == "start" else peak
                dd = max(0.0, (ref - equity_now) / ref * 100.0) if ref > 0 else 0.0
                scale = safety.derate_multiplier(dd, cfg.kill_dd_pct)
            dscale = drift.multiplier if drift is not None else 1.0
            scale_eff = scale * dscale
            lv = size_and_levels(o, fill, side, atr_sig, cfg, scale_eff)
            if lv is None:
                stats["skip_no_atr"] += 1
                continue
            qty, tp, sl = lv
            notional = qty * fill
            if notional < cfg.min_notional_usd:
                stats["skip_min_notional"] += 1
                continue
            margin = notional / cfg.leverage
            efee = fee * notional
            free = wallet - sum(p.margin for p in real)
            if free < margin + efee:
                stats["skip_no_margin"] += 1
                continue
            # correlation cap: 24h of 5m returns up to the signal bar's close
            same = [p for p in real if p.side == side]
            if len(same) >= cfg.corr_max_correlated:
                lo = max(0, ti - cfg.corr_window)
                hi = ti
                nc = 0
                for p in same:
                    c = _corr(rets[k], rets[p.sidx], lo, hi, cfg.corr_min_obs)
                    if c is not None and c > cfg.corr_max:
                        nc += 1
                if nc >= cfg.corr_max_correlated:
                    stats["skip_corr"] += 1
                    continue
            wallet -= efee
            if scale_eff < 1.0:
                stats["derated_trades"] += int(scale < 1.0)
                stats["drift_half_trades"] += int(dscale < 1.0)
                stats["min_derate_mult"] = min(stats["min_derate_mult"], scale_eff)
            p = Position(next_id, syms[k], k, side, ti, T0, fill, qty, notional, margin, tp, sl, sl,
                         liq_price(fill, side, cfg.leverage, cfg.mmr), float(atr_sig), prob, scale_eff,
                         efee, best=fill, last=fill)
            next_id += 1
            stats["entries"] += 1
            open_pos.append(p)
        # 3. exits (per bar)
        still: list[Position] = []
        for p in open_pos:
            r = row_at[p.sidx, ti]
            if r < 0:
                still.append(p)
                continue
            o, h, l, c = O[p.sidx][r], H[p.sidx][r], L[p.sidx][r], C[p.sidx][r]
            p.bars += 1
            d = p.side
            res = None
            stop = p.sl
            if d > 0:
                stop_hit, liq_hit, tp_hit = l <= stop, l <= p.liq, h >= p.tp
                liq_protected = stop_hit and stop > p.liq and o > p.liq
            else:
                stop_hit, liq_hit, tp_hit = h >= stop, h >= p.liq, l <= p.tp
                liq_protected = stop_hit and stop < p.liq and o < p.liq
            if liq_hit and not liq_protected:
                gapped = (o <= p.liq) if d > 0 else (o >= p.liq)
                res = ("liquidation", o if gapped else p.liq, True)
            elif stop_hit:
                base = min(stop, o) if d > 0 else max(stop, o)
                px = base * (1.0 - d * sslip)
                trailed = p.trail_active and ((stop > p.sl0) if d > 0 else (stop < p.sl0))
                res = ("trail" if trailed else "sl", px, False)
            elif tp_hit:
                res = ("tp", p.tp, False)
            if res is not None:
                close(p, res[1], res[0], T0 + BAR_MS - 1, res[2])
                continue
            # trailing update (effective from the next bar)
            if cfg.trail_enabled and p.atr > 0:
                risk = abs(p.entry_price - p.sl0)
                if d > 0:
                    p.best = max(p.best, h)
                else:
                    p.best = min(p.best, l)
                fav = (p.best - p.entry_price) * d
                if p.trail_active or fav >= cfg.trail_activate_r * risk:
                    p.trail_active = True
                    cand = p.best - d * cfg.trail_distance_atr * p.atr
                    p.sl = max(p.sl, cand) if d > 0 else min(p.sl, cand)
            p.last = c
            still.append(p)
        open_pos = still
        # 4. mark to market / kill switch
        real = [p for p in open_pos if not p.shadow]
        if real:
            open_steps += 1
        equity_now = wallet + sum(unreal(p) for p in real)
        if equity_now > peak:
            peak = equity_now
        ref = float(cfg.start_equity) if cfg.dd_basis == "start" else peak
        if (not killed and cfg.kill_dd_pct < 100 and ref > 0
                and (ref - equity_now) / ref * 100.0 >= cfg.kill_dd_pct):
            killed = True
            kill_time = T0
            for p in real:
                px = p.last * (1.0 - p.side * slip)
                stats["kill_closes"] += 1
                close(p, px, "kill", T0 + BAR_MS - 1)
            open_pos = []
            equity_now = wallet
        eq[ti] = equity_now
        ti += 1

    if drift is not None:
        stats["drift_fired_half"] = drift.fired_half
        stats["drift_fired_block"] = drift.fired_block
        stats["drift_final_z"] = drift.z
    # trades still open at the end of data are left open (reported)
    stats["open_at_end"] = len([p for p in open_pos if not p.shadow])
    from .metrics import compute_metrics
    m = compute_metrics(trades, ts, eq, cfg, open_steps)
    return BacktestResult(trades=trades, ts=ts, equity=eq, cfg=cfg, metrics=m, stats=stats,
                          symbols=syms, kill_time=kill_time)
