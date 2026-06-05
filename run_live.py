"""Live paper trading across MULTIPLE symbols, shown in one web dashboard.

Runs a separate paper-trading engine per symbol, all sharing one event bus and one
dashboard. Each fill is symbol-tagged, so the dashboard's trades table shows every
symbol interleaved; the header shows the aggregated portfolio (summed equity / PnL /
trades across all symbols).

Data source — automatic
-----------------------
On startup it probes whether Binance market data is reachable:
  * REACHABLE  -> uses the real BinanceFuturesClient (real live market data, paper fills).
                  No API keys needed: only public market data is used; paper mode never
                  places real orders.
  * BLOCKED    -> falls back to a per-symbol ReplayClient (synthetic feed) so you can still
                  watch multi-symbol paper trading immediately. The SAME command switches to
                  live data the moment the network allows (e.g. back on a hotspot).

Usage
-----
    python run_live.py
    python run_live.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT
    python run_live.py --timeframe 1m --host 0.0.0.0      # expose on LAN/phone
    python run_live.py --force-replay                      # don't even try the exchange
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import typing
import urllib.request
import webbrowser

from core.event_bus import GLOBAL_BUS, BusLogHandler
from connectors.replay import ReplayClient, make_synthetic_ohlcv
from risk.risk_manager import RiskManager, RiskLimits
from engine.trading_engine import TradingEngine
from alerts.notifier import ConsoleNotifier
from promotion.evaluator import PromotionEvaluator
from strategies.ema_cross import EmaCrossStrategy
from web.server import AppContext, DashboardServer

logger = logging.getLogger("run_live")

# Rough starting price per symbol for the synthetic fallback so each feed looks distinct.
_REPLAY_START_PRICE = {
    "BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0, "SOLUSDT": 150.0,
    "BNBUSDT": 600.0, "XRPUSDT": 0.6, "ADAUSDT": 0.45, "DOGEUSDT": 0.15,
}


def exchange_reachable(timeout: float = 6.0) -> bool:
    """True if Binance futures market data answers. Tries prod, then testnet."""
    for url in ("https://fapi.binance.com/fapi/v1/ping",
                "https://testnet.binancefuture.com/fapi/v1/ping"):
        try:
            urllib.request.urlopen(url, timeout=timeout)
            return True
        except Exception:
            continue
    return False


def build_real_client():
    """Construct a BinanceFuturesClient using whatever keys CONFIG has (may be empty).

    Only public market data is used downstream, so empty keys are fine; balance/order
    calls that need auth are never invoked in paper mode.
    """
    from config import CONFIG
    from connectors.binance_futures import BinanceFuturesClient

    # Market-data-only client: paper trading consumes ONLY public market data, so we skip
    # the two parts of the stock connector that need API keys / aren't needed for REST
    # polling and that otherwise crash with empty keys:
    #   * get_balances() -> hits the auth-only /fapi/v1/account (non-JSON body -> crash)
    #   * _start_ws()    -> a live order-book websocket we don't need (REST candles drive us)
    # PROD (testnet=False) gives the real, current market — the point of "live" paper trading.
    class BinanceMarketDataClient(BinanceFuturesClient):
        # Name keeps "Binance" so the engine's exchange inference tags fills correctly.
        def get_balances(self):
            return {}

        def _start_ws(self):
            return  # no websocket; the engine marks price off REST candles

    return BinanceMarketDataClient(CONFIG.binance_public_key, CONFIG.binance_secret_key,
                                   testnet=False)


def build_training_df(use_live: bool, real_client, symbol: str, timeframe: str):
    """OHLCV frame to train the RL policy on: live history if reachable, else synthetic.

    Returns a pandas DataFrame with timestamp/open/high/low/close/volume columns — the
    shape train_to_bus / TradingEnv expect.
    """
    import pandas as pd

    if use_live and real_client is not None:
        try:
            contract = real_client.contracts.get(symbol)
            candles = real_client.get_historical_candles(contract, timeframe)
            if candles and len(candles) >= 500:
                logger.info("Training data: %d live %s %s candles.",
                            len(candles), symbol, timeframe)
                return pd.DataFrame({
                    "timestamp": [c.timestamp for c in candles],
                    "open": [c.open for c in candles],
                    "high": [c.high for c in candles],
                    "low": [c.low for c in candles],
                    "close": [c.close for c in candles],
                    "volume": [c.volume for c in candles],
                })
            logger.warning("Live history too short (%d candles); training on synthetic.",
                           0 if not candles else len(candles))
        except Exception as e:
            logger.warning("Could not fetch live history (%s); training on synthetic.", e)
    return make_synthetic_ohlcv(n=3000)


class MultiEngineController:
    """Fans start/stop/readiness across N per-symbol engines and publishes aggregate state.

    Exposes the same interface the dashboard expects of a single engine
    (start / stop / get_state / assess_readiness), so the web layer is unchanged.
    """

    def __init__(self, engines: typing.Dict[str, TradingEngine], event_bus,
                 initial_capital: float):
        self.engines = engines
        self.event_bus = event_bus
        self.initial_capital_total = initial_capital * len(engines)
        self.symbol = f"{len(engines)} symbols: " + ", ".join(engines.keys())
        self.mode = next(iter(engines.values())).mode if engines else "paper"
        self.running = False
        self._stop = threading.Event()
        self._thread: typing.Optional[threading.Thread] = None

    # --------------------------------------------------------------- controls
    def start(self) -> None:
        for eng in self.engines.values():
            eng.start()
        self.running = True
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._aggregate_loop, daemon=True)
            self._thread.start()
        logger.info("MultiEngineController started %d engines.", len(self.engines))

    def stop(self) -> None:
        for eng in self.engines.values():
            eng.stop()
        self.running = False
        self._stop.set()
        logger.info("MultiEngineController stopped all engines.")

    # ------------------------------------------------------------- aggregation
    def _aggregate_loop(self) -> None:
        """Publish one combined state + equity event per second for the dashboard."""
        while not self._stop.is_set():
            state = self.get_state()
            try:
                self.event_bus.publish("state", state)
                self.event_bus.publish("equity", {"ts": int(time.time() * 1000),
                                                   "equity": state["equity"]})
            except Exception:
                pass
            self._stop.wait(1.0)

    def get_state(self) -> dict:
        eq = cash = rpnl = upnl = 0.0
        trades = 0
        halted = False
        per_symbol = []
        for sym, eng in self.engines.items():
            s = eng.get_state()
            eq += s["equity"]; cash += s["cash"]
            rpnl += s["realized_pnl"]; upnl += s["unrealized_pnl"]
            trades += s["n_trades"]
            halted = halted or s["halted"]
            per_symbol.append({"symbol": sym, "equity": round(s["equity"], 2),
                               "position": round(s["position"], 6),
                               "n_trades": s["n_trades"],
                               "realized_pnl": round(s["realized_pnl"], 2)})
        return {
            "mode": self.mode,
            "symbol": self.symbol,
            "running": any(e.running for e in self.engines.values()),
            "equity": round(eq, 2),
            "cash": round(cash, 2),
            "position": 0.0,
            "realized_pnl": round(rpnl, 2),
            "unrealized_pnl": round(upnl, 2),
            "n_trades": trades,
            "halted": halted,
            "last_signal": 0.0,
            "per_symbol": per_symbol,
        }

    def assess_readiness(self) -> dict:
        """Aggregate readiness: report the per-symbol verdicts plus a portfolio rollup."""
        results = {sym: eng.assess_readiness() for sym, eng in self.engines.items()}
        ready_syms = [s for s, v in results.items() if v.get("ready")]
        reasons = [f"{s}: {'READY' if v.get('ready') else 'not ready'}"
                   for s, v in results.items()]
        return {
            "ready": len(ready_syms) == len(self.engines) and bool(self.engines),
            "score": round(sum(v.get("score", 0.0) for v in results.values())
                           / max(1, len(self.engines)), 4),
            "reasons": reasons,
            "recommendation": (f"{len(ready_syms)}/{len(self.engines)} symbols ready. "
                               "Graduating is necessary but not sufficient — start any "
                               "live trial at reduced size."),
            "per_symbol": results,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-symbol live/paper trading dashboard.")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--capital", type=float, default=10_000, help="capital PER symbol")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--poll", type=float, default=None, help="seconds between polls")
    ap.add_argument("--force-replay", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    root = logging.getLogger()
    bh = BusLogHandler(GLOBAL_BUS, level=logging.INFO)
    bh.setFormatter(logging.Formatter("%(asctime)s :: %(message)s", "%H:%M:%S"))
    root.addHandler(bh)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # --- choose data source ------------------------------------------------
    use_live = False
    real_client = None
    if not args.force_replay and exchange_reachable():
        try:
            real_client = build_real_client()
            use_live = True
            logger.info("Exchange REACHABLE — using LIVE market data (paper fills).")
        except Exception as e:
            logger.warning("Real connector failed (%s); falling back to replay.", e)
    if not use_live:
        logger.warning("Exchange not reachable — using SYNTHETIC replay feed per symbol. "
                       "Re-run on an exchange-reachable network for live data.")

    poll = args.poll if args.poll is not None else (10.0 if use_live else 0.4)

    # --- build one engine per symbol --------------------------------------
    engines: typing.Dict[str, TradingEngine] = {}
    for i, sym in enumerate(symbols):
        if use_live:
            client = real_client
            if sym not in getattr(client, "contracts", {}):
                logger.warning("Symbol %s not on exchange; skipping.", sym)
                continue
        else:
            start_px = _REPLAY_START_PRICE.get(sym, 100.0 * (i + 1))
            df = make_synthetic_ohlcv(n=3000, seed=7 + i, start_price=start_px)
            client = ReplayClient(df=df, symbol=sym, warmup=200)

        engines[sym] = TradingEngine(
            client=client,
            strategy=EmaCrossStrategy(),
            risk_manager=RiskManager(RiskLimits(), starting_equity=args.capital),
            database=None,
            symbol=sym,
            timeframe=args.timeframe,
            mode="paper",
            initial_capital=args.capital,
            notifier=ConsoleNotifier(),
            evaluator=PromotionEvaluator(),
            poll_seconds=poll,
            event_bus=GLOBAL_BUS,
            publish_step_updates=False,    # the controller publishes the aggregate
        )

    if not engines:
        raise SystemExit("No tradable symbols; aborting.")

    controller = MultiEngineController(engines, GLOBAL_BUS, args.capital)

    # Wire the "Train RL" button: train a CEM policy on the first symbol's data (live
    # history when the exchange is reachable, else synthetic), streaming progress to the
    # dashboard. Without this the page reports "Training is not configured".
    train_symbol = next(iter(engines.keys()))

    def start_training():
        from learning.train import train_to_bus
        df = build_training_df(use_live, real_client, train_symbol, args.timeframe)
        logger.info("Training RL policy on %s (%s)...", train_symbol,
                    "live data" if use_live else "synthetic feed")
        return train_to_bus(GLOBAL_BUS, df=df, iterations=20)

    ctx = AppContext(event_bus=GLOBAL_BUS, engine=controller, start_training=start_training)
    server = DashboardServer(ctx, host=args.host, port=args.port)
    server.start()

    src = "LIVE Binance data" if use_live else "synthetic replay feed"
    logger.info("=" * 66)
    logger.info("Multi-symbol paper trading | %s | %d symbols: %s",
                src, len(engines), ", ".join(engines.keys()))
    logger.info("Dashboard -> %s", server.url)
    if args.host == "0.0.0.0":
        logger.info("From phone/LAN: http://<this-machine-ip>:%d/", args.port)
    logger.info("Click ▶ Start. Trades for all symbols stream into one table.")
    logger.info("=" * 66)

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.0), webbrowser.open(server.url)),
                         daemon=True).start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        controller.stop()
        server.stop()


if __name__ == "__main__":
    main()
