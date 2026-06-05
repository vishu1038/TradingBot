"""Launch the live web dashboard with a paper-trading engine you can watch in real time.

This is the "show me everything happening live" entry point. It wires together:

    ReplayClient  -> feeds candles bar-by-bar (no exchange needed)
        |
    TradingEngine -> paper-trades a strategy, publishing fills/equity/state
        |
    EventBus      -> fans those events out
        |
    DashboardServer (http://localhost:8765) -> streams them to your browser via SSE

Everything runs offline and deterministically, so it works on the corporate network and on
a BeagleBone alike. The "Train RL" button trains a policy on the same feed and streams
progress to the page.

Usage
-----
    python run_live_demo.py                      # synthetic replay feed, EMA strategy
    python run_live_demo.py --strategy rl        # use the trained RL policy (train first)
    python run_live_demo.py --speed 0.2          # seconds per bar (default 0.3; lower = faster)
    python run_live_demo.py --host 0.0.0.0       # expose on the LAN (phone / other devices)
    python run_live_demo.py --port 9000          # custom port

Going LIVE on testnet (when you have network + keys):
    Put Binance *testnet* keys in .env, then run main.py / a live runner that swaps
    ReplayClient for BinanceFuturesClient. This demo is paper-only by construction.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser

from core.event_bus import GLOBAL_BUS, BusLogHandler
from connectors.replay import ReplayClient, make_synthetic_ohlcv
from risk.risk_manager import RiskManager, RiskLimits
from engine.trading_engine import TradingEngine
from alerts.notifier import ConsoleNotifier
from promotion.evaluator import PromotionEvaluator
from web.server import AppContext, DashboardServer


def build_strategy(name: str):
    name = name.lower()
    if name == "ema":
        from strategies.ema_cross import EmaCrossStrategy
        return EmaCrossStrategy()
    if name == "ml":
        from strategies.ml_strategy import MLStrategy
        # ML needs a trained model; fall back to EMA with a warning if absent.
        return MLStrategy()
    if name == "rl":
        from strategies.rl_strategy import RLStrategy
        return RLStrategy()
    raise SystemExit(f"unknown strategy {name!r}; choose ema|ml|rl")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live trading-bot dashboard (paper, replay feed).")
    ap.add_argument("--strategy", default="ema", help="ema | ml | rl")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--speed", type=float, default=0.3, help="seconds per replayed bar")
    ap.add_argument("--bars", type=int, default=3000, help="length of the synthetic feed")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on the LAN")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    # Logging -> console AND the dashboard's live log pane.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    root = logging.getLogger()
    bus_handler = BusLogHandler(GLOBAL_BUS, level=logging.INFO)
    bus_handler.setFormatter(logging.Formatter("%(asctime)s :: %(message)s", "%H:%M:%S"))
    root.addHandler(bus_handler)
    log = logging.getLogger("run_live_demo")

    # Shared synthetic feed (same data drives trading AND the Train button).
    df = make_synthetic_ohlcv(n=args.bars)
    client = ReplayClient(df=df, symbol=args.symbol, warmup=200)

    strategy = build_strategy(args.strategy)
    risk = RiskManager(RiskLimits(), starting_equity=10_000)
    evaluator = PromotionEvaluator()

    engine = TradingEngine(
        client=client,
        strategy=strategy,
        risk_manager=risk,
        database=None,                 # connector path -> uses ReplayClient feed directly
        symbol=args.symbol,
        timeframe=args.timeframe,
        mode="paper",
        initial_capital=10_000,
        notifier=ConsoleNotifier(),
        evaluator=evaluator,
        poll_seconds=args.speed,       # one replayed bar per poll
        event_bus=GLOBAL_BUS,
    )

    def start_training():
        from learning.train import train_to_bus
        return train_to_bus(GLOBAL_BUS, df=df, iterations=20)

    ctx = AppContext(event_bus=GLOBAL_BUS, engine=engine, start_training=start_training)
    server = DashboardServer(ctx, host=args.host, port=args.port)
    server.start()

    log.info("=" * 64)
    log.info("Dashboard ready -> %s", server.url)
    if args.host == "0.0.0.0":
        log.info("On the LAN/phone, open http://<this-machine-ip>:%d/", args.port)
    log.info("Click ▶ Start to begin paper trading the '%s' strategy.", args.strategy)
    log.info("=" * 64)

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.0), webbrowser.open(server.url)),
                         daemon=True).start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Shutting down…")
        engine.stop()
        server.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
