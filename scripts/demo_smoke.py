#!/usr/bin/env python
"""Binance Futures demo/testnet smoke test (run once keys exist; never touches live).

    .venv/bin/python scripts/demo_smoke.py --dry-run        # read-only checks only
    .venv/bin/python scripts/demo_smoke.py                  # full order round trip on ETHUSDT

Keys come from .env / the environment (BINANCE_API_KEY, BINANCE_API_SECRET) and are never printed.
Target: BINANCE_ENV=testnet -> https://testnet.binancefuture.com, otherwise https://demo-fapi.binance.com
(override with --env). The real-money host is refused.

Steps (each prints OK/FAIL and the key facts of the response):
  1 server time/clock skew   2 account balance   3 position mode (must be ONE-WAY)   4 symbol filters/mark
  --- everything below is skipped with --dry-run ---
  5 isolated margin + 8x     6 smallest valid MARKET entry   7 closePosition SL (algo endpoint on -4120) and TP
  8 list open orders + algo orders   9 cancel them   10 close the position   11 final state check
If anything fails after the entry, the script still cancels the orders and closes the position.
Exit code 0 = all steps passed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from tb.config import DEMO_BASE, PROD_BASE, TESTNET_BASE, load_secrets  # noqa: E402
from tb.exchange.base import ExchangeError, opposite  # noqa: E402
from tb.exchange.binance import BinanceExchange  # noqa: E402


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def ok(self, step: str, msg: str = "") -> None:
        print(f"[ OK ] {step}{': ' + msg if msg else ''}", flush=True)

    def fail(self, step: str, msg: str) -> None:
        self.failed += 1
        print(f"[FAIL] {step}: {msg}", flush=True)

    def info(self, msg: str) -> None:
        print(f"       {msg}", flush=True)


def smallest_valid_qty(mark: float, f) -> float:
    """Smallest lot-step multiple that satisfies min_qty and min_notional (10% headroom for price moves)."""
    need = max(f.min_qty, f.min_notional * 1.1 / mark)
    steps = math.ceil(need / f.step_size - 1e-9)
    return round(steps * f.step_size, 12)


def run(ex: BinanceExchange, symbol: str, leverage: int, dry_run: bool, rep: Report) -> None:
    # 1 -------------------------------------------------------------------------------- clock
    try:
        skew = ex.sync_time()
        rep.ok("server time", f"clock offset {skew} ms")
    except ExchangeError as e:
        rep.fail("server time", str(e))
        return
    # 2 -------------------------------------------------------------------------------- balance
    try:
        b = ex.balance()
        rep.ok("account balance", f"wallet {b.wallet:.2f} USDT, available {b.available:.2f}, unrealised {b.upnl:+.2f}")
    except ExchangeError as e:
        rep.fail("account balance", f"{e} (wrong keys, or keys not enabled for futures?)")
        return
    # 3 -------------------------------------------------------------------------------- position mode
    try:
        hedged = ex.position_mode_hedged()
    except ExchangeError as e:
        rep.fail("position mode", str(e))
        return
    if hedged:
        rep.fail("position mode", "account is in HEDGE (dual-side) mode; the bot needs ONE-WAY mode. "
                                  "Switch it in the futures preferences (no open positions allowed), then re-run.")
        return
    rep.ok("position mode", "one-way")
    # 4 -------------------------------------------------------------------------------- symbol
    try:
        info = ex.exchange_info().get(symbol)
        if not info or info.status != "TRADING":
            rep.fail("symbol", f"{symbol} is not TRADING on this venue")
            return
        f = info.filters
        mark = ex.premium(symbol)[0].mark_price
        rep.ok("symbol", f"{symbol} mark {mark:g}, tick {f.tick_size:g}, step {f.step_size:g}, min_qty {f.min_qty:g}, "
                         f"min_notional {f.min_notional:g}")
        open_pos = [p for p in ex.positions() if p.symbol == symbol]
        orders = ex.open_conditional_orders(symbol)
        rep.ok("existing state", f"{len(open_pos)} open position(s), {len(orders)} conditional order(s) on {symbol}")
    except ExchangeError as e:
        rep.fail("symbol", str(e))
        return
    if dry_run:
        print("--dry-run: stopping before any state-changing call.")
        return
    if open_pos or orders:
        rep.fail("precondition", f"{symbol} already has a position or conditional orders; clean it up first")
        return
    qty = smallest_valid_qty(mark, f)
    notional = qty * mark
    if notional / leverage > b.available:
        rep.fail("margin", f"need {notional / leverage:.2f} USDT margin, only {b.available:.2f} available")
        return
    entered = False
    try:
        # 5 ---------------------------------------------------------------------------- isolated + leverage
        ex.set_isolated(symbol)
        rep.ok("margin type", "ISOLATED")
        ex.set_leverage(symbol, leverage)
        rep.ok("leverage", f"{leverage}x")
        # 6 ---------------------------------------------------------------------------- entry
        res = ex.market_order(symbol, "BUY", qty)
        entered = True
        fill = res.avg_price or mark
        rep.ok("market entry", f"BUY {res.qty:g} {symbol} @ {fill:g} (notional {res.qty * fill:.2f}), id {res.order_id}, "
                               f"status {res.status}")
        # 7 ---------------------------------------------------------------------------- protective orders
        mark_now = ex.premium(symbol)[0].mark_price
        sl_px, tp_px = f.round_price(fill * 0.95), f.round_price(fill * 1.05)
        if not (sl_px < mark_now < tp_px):
            rep.fail("levels", f"mark {mark_now:g} is outside SL {sl_px:g} / TP {tp_px:g}; the market moved >5%?")
        else:
            sl_id = ex.place_stop(symbol, opposite("BUY"), sl_px)
            rep.ok("stop-loss", f"closePosition STOP_MARKET SELL @ {sl_px:g}, id {sl_id} "
                                f"({'algo endpoint' if sl_id.startswith('a:') else 'regular order endpoint'})")
            tp_id = ex.place_take_profit(symbol, opposite("BUY"), tp_px)
            rep.ok("take-profit", f"closePosition TAKE_PROFIT_MARKET SELL @ {tp_px:g}, id {tp_id}")
        # 8 ---------------------------------------------------------------------------- list
        orders = ex.open_conditional_orders(symbol)
        for o in orders:
            rep.info(f"open: {o.order_id} {o.kind} {o.side} trigger {o.trigger_price:g} closePosition={o.close_position}")
        kinds = sorted(o.kind for o in orders)
        (rep.ok if kinds == ["SL", "TP"] else rep.fail)("list orders", f"found {kinds} (expected ['SL', 'TP'])")
        stops_ok = all(o.close_position for o in orders)
        (rep.ok if stops_ok else rep.fail)("closePosition flag", "all conditional orders are closePosition" if stops_ok
                                           else "a conditional order is not closePosition")
    except ExchangeError as e:
        rep.fail("order flow", f"{e} (code {e.code}, http {e.status})")
    finally:
        # 9/10 -------------------------------------------------------------------------- always clean up
        try:
            for o in ex.open_conditional_orders(symbol):
                ex.cancel_order(symbol, o.order_id)
            ex.cancel_all(symbol)
            left = ex.open_conditional_orders(symbol)
            (rep.ok if not left else rep.fail)("cancel orders", "none left" if not left else f"{len(left)} still open")
        except ExchangeError as e:
            rep.fail("cancel orders", str(e))
        try:
            pos = [p for p in ex.positions() if p.symbol == symbol]
            if pos or entered:
                for p in pos:
                    r = ex.market_order(symbol, opposite("BUY"), p.qty)
                    rep.ok("close position", f"SELL {r.qty:g} (normal market order) @ {r.avg_price:g}")
        except ExchangeError as e:
            rep.fail("close position", f"{e} - CLOSE {symbol} MANUALLY on the exchange")
        # 11 ---------------------------------------------------------------------------- final state
        try:
            time.sleep(0.5)
            pos = [p for p in ex.positions() if p.symbol == symbol]
            left = ex.open_conditional_orders(symbol)
            b2 = ex.balance()
            (rep.ok if not pos and not left else rep.fail)(
                "final state", f"{len(pos)} position(s), {len(left)} order(s); wallet {b2.wallet:.4f} USDT "
                               f"(round trip cost {b.wallet - b2.wallet:+.4f})")
        except ExchangeError as e:
            rep.fail("final state", str(e))


def base_url(env: str) -> str:
    url = {"testnet": TESTNET_BASE, "demo": DEMO_BASE}[env]
    assert url != PROD_BASE
    return url


def main(argv: list[str] | None = None, *, exchange: BinanceExchange | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="read-only calls only")
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--leverage", type=int, default=8)
    ap.add_argument("--env", choices=["testnet", "demo"], default=None,
                    help="default: testnet if BINANCE_ENV=testnet else demo (demo-fapi.binance.com)")
    args = ap.parse_args(argv)
    env = args.env or ("testnet" if os.environ.get("BINANCE_ENV", "").strip().lower() == "testnet" else "demo")
    rep = Report()
    if exchange is None:
        sec = load_secrets()
        if not sec.has_binance:
            print("BINANCE_API_KEY / BINANCE_API_SECRET not found in .env or the environment.", file=sys.stderr)
            return 2
        exchange = BinanceExchange(base_url(env), sec.api_key, sec.api_secret, mode="binance_demo")
    if exchange.base_url.rstrip("/") == PROD_BASE:
        print("refusing to run against the real-money host", file=sys.stderr)
        return 2
    print(f"venue: {exchange.base_url}  symbol: {args.symbol}  {'DRY RUN' if args.dry_run else 'LIVE ORDERS (fake money)'}")
    run(exchange, args.symbol, args.leverage, args.dry_run, rep)
    print("RESULT:", "all steps passed" if rep.failed == 0 else f"{rep.failed} step(s) FAILED")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
