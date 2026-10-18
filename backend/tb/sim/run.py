"""CLI: run the backtester on the downloaded archive and write a lab report.

  python -m tb.sim.run --signals random --exit fixed
  python -m tb.sim.run --symbols BTCUSDT,SOLUSDT --signals random --p 0.003 --exit atr --trail
  python -m tb.sim.run --signals every_n --n 48 --exit fixed --start 2026-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from tb.data import loader
from tb.data.paths import BAR_MS

from . import signals as sig
from .backtest import run_backtest
from .report import write_report
from .template import SimConfig, atr_wilder, breakeven_winrate


def load_inputs(symbols: list[str] | None, start: str | None = None, end: str | None = None,
                root=None, months: int | None = None):
    syms = symbols or loader.symbols_available(root)
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) if start else None
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None
    kls, fund = {}, {}
    for s in syms:
        kl, _me, fu = loader.load(s, root)
        if months and t0 is None and len(kl):
            t0s = int(kl["open_time"].iloc[-1]) - int(months * 30.4375 * 86_400_000)
        else:
            t0s = t0
        if t0s is not None:
            kl = kl[kl["open_time"] >= t0s]
            fu = fu[fu["calc_time"] >= t0s]
        if t1 is not None:
            kl = kl[kl["open_time"] < t1]
            fu = fu[fu["calc_time"] < t1]
        if len(kl) > 300:
            kls[s] = kl.reset_index(drop=True)
            fund[s] = fu.reset_index(drop=True)
    return kls, fund


def median_atr_frac(kls: dict[str, pd.DataFrame]) -> float:
    vals = []
    for kl in kls.values():
        a = atr_wilder(kl["high"].to_numpy(), kl["low"].to_numpy(), kl["close"].to_numpy())
        vals.append(np.nanmedian(a / kl["close"].to_numpy()))
    return float(np.median(vals)) if vals else 0.003


def make_signals(kind: str, p: float, seed: int, side: str, n: int):
    if kind == "random":
        return sig.random_entries(p, seed, side)
    if kind == "every_n":
        return sig.always_long_every_n(n)
    raise ValueError(f"unknown --signals {kind!r}")


def build_cfg(a) -> SimConfig:
    return SimConfig(
        start_equity=a.start_equity, exit_mode=a.exit, tp_usd=a.tp_usd, sl_usd=a.sl_usd,
        tp_atr=a.tp_atr, sl_atr=a.sl_atr, trail_enabled=a.trail, trail_activate_r=a.activate_r,
        trail_distance_atr=a.trail_atr, short_enabled=a.shorts or a.side in ("short", "both"),
        thr_long=a.thr_long, thr_short=a.thr_short, max_positions=a.max_positions,
        kill_dd_pct=a.kill_dd, fee_taker=a.fee, slippage=a.slippage,
        stop_slippage_extra=a.stop_slip_extra, derate_enabled=not a.no_derate,
        plausibility_enabled=not a.no_plausibility, drift_enabled=not a.no_drift)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=None, help="comma list (default: every symbol in the archive)")
    ap.add_argument("--signals", default="random", choices=["random", "every_n"])
    ap.add_argument("--p", type=float, default=0.004, help="random: per-bar entry probability")
    ap.add_argument("--side", default="long", choices=["long", "short", "both"])
    ap.add_argument("--n", type=int, default=48, help="every_n: bars between entries")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exit", default="fixed", choices=["fixed", "atr"])
    ap.add_argument("--tp-usd", type=float, default=5.0)
    ap.add_argument("--sl-usd", type=float, default=10.0)
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--trail", action="store_true")
    ap.add_argument("--activate-r", type=float, default=1.0)
    ap.add_argument("--trail-atr", type=float, default=1.0)
    ap.add_argument("--shorts", action="store_true")
    ap.add_argument("--thr-long", type=float, default=0.70)
    ap.add_argument("--thr-short", type=float, default=0.80)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--start-equity", type=float, default=100.0)
    ap.add_argument("--kill-dd", type=float, default=50.0)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--stop-slip-extra", type=float, default=0.0005)
    ap.add_argument("--no-derate", action="store_true")
    ap.add_argument("--no-plausibility", action="store_true")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--start", default=None, help="UTC date, inclusive")
    ap.add_argument("--end", default=None, help="UTC date, exclusive")
    ap.add_argument("--months", type=int, default=None, help="only the last N months of data")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None, help="report dir (default var/lab)")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else None
    t0 = time.time()
    kls, fund = load_inputs(symbols, a.start, a.end, months=a.months)
    if not kls:
        print("no kline data found - run scripts/download_archive.py first", file=sys.stderr)
        return 2
    cfg = build_cfg(a)
    spec = make_signals(a.signals, a.p, a.seed, a.side, a.n)
    res = run_backtest(kls, spec, cfg, fund)
    atr_frac = median_atr_frac(kls)
    title = a.title or (f"{a.signals} entries, {a.exit} exit{' + trail' if a.trail else ''} "
                        f"({len(kls)} symbols)")
    m = res.metrics
    be = breakeven_winrate(cfg, atr_frac if cfg.exit_mode == "atr" else None)
    print(f"{title}: {len(kls)} symbols, {m['days']:.0f} days, {time.time() - t0:.1f}s")
    print(f"  trades {m['n_trades']}  win {m['win_rate'] * 100:.1f}% (break-even {be['breakeven_winrate'] * 100:.1f}%, "
          f"zero-cost {be['zero_cost_winrate'] * 100:.1f}%)  expectancy {m['expectancy']:.3f}  "
          f"PF {m['profit_factor']:.2f}")
    print(f"  return {m['total_return_pct']:.1f}%  maxDD {m['max_drawdown_pct']:.1f}%  sharpe {m['sharpe']:.2f}  "
          f"fees {m['fees_paid']:.1f}  funding {m['funding_paid']:.2f}  exposure {m['exposure_pct']:.1f}%")
    print("  exits", m["exit_reasons"])
    print("  safety", json.dumps({k: v for k, v in res.stats.items()
                                  if k != "fault_by_symbol" and v not in (0, None)}, default=str))
    if not a.no_report:
        path = write_report(res, title, extra={"signals": a.signals, "p": a.p, "seed": a.seed},
                            atr_frac=atr_frac, out_dir=a.out)
        print("  report:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
