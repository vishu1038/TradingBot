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
from learning.train import train_to_bus
from alerts.notifier import ConsoleNotifier
from promotion.evaluator import PromotionEvaluator
from strategies.ema_cross import EmaCrossStrategy
from web.server import AppContext, DashboardServer

logger = logging.getLogger("run_live")

# Default symbol set leans toward higher-volatility alts (more trading signal for the RL
# policy to learn from) while keeping BTC/ETH as lower-vol anchors. Override with --symbols.
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,XRPUSDT,SUIUSDT"

# Rough starting price per symbol for the synthetic fallback so each feed looks distinct.
_REPLAY_START_PRICE = {
    "BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0, "SOLUSDT": 150.0,
    "BNBUSDT": 600.0, "XRPUSDT": 0.6, "ADAUSDT": 0.45, "DOGEUSDT": 0.15,
    "AVAXUSDT": 35.0, "LINKUSDT": 18.0, "SUIUSDT": 3.5, "NEARUSDT": 6.0,
    "DOTUSDT": 7.0, "APTUSDT": 9.0, "INJUSDT": 25.0, "TIAUSDT": 8.0,
}

# Timeframe -> milliseconds, used to page backwards through klines when fetching history.
_TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000,
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


def fetch_history(real_client, symbol: str, timeframe: str, n_bars: int):
    """Assemble up to `n_bars` recent candles for `symbol` as a DataFrame.

    The stock connector's get_historical_candles caps at Binance's 1000/request, which on a
    1m timeframe is only ~16h of data — far too short to train on. This pages BACKWARDS
    through /fapi/v1/klines via endTime (Binance allows up to 1500/request) until it has
    n_bars or runs out of history, giving the RL policy a much longer training period.
    """
    import pandas as pd

    contract = real_client.contracts.get(symbol)
    if contract is None:
        return None

    rows: list = []
    end_time = None
    remaining = int(n_bars)
    safety = 0
    while remaining > 0 and safety < 60:
        safety += 1
        data = {"symbol": symbol, "interval": timeframe, "limit": min(1500, remaining)}
        if end_time is not None:
            data["endTime"] = end_time
        raw = real_client._make_request("GET", "/fapi/v1/klines", data)
        if not raw:
            break
        chunk = [(int(c[0]), float(c[1]), float(c[2]), float(c[3]),
                  float(c[4]), float(c[5])) for c in raw]
        rows = chunk + rows
        remaining -= len(chunk)
        end_time = chunk[0][0] - 1          # page strictly older than the earliest bar
        if len(chunk) < data["limit"]:
            break                            # exchange returned all it has
    if not rows:
        return None

    rows = sorted(set(rows), key=lambda r: r[0])   # de-dup overlaps, chronological
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low",
                                       "close", "volume"])


def build_training_df(use_live: bool, real_client, symbol: str, timeframe: str,
                      n_bars: int = 5000):
    """OHLCV frame to train the RL policy on: live history if reachable, else synthetic."""
    if use_live and real_client is not None:
        try:
            df = fetch_history(real_client, symbol, timeframe, n_bars)
            if df is not None and len(df) >= 500:
                logger.info("Training data: %d live %s %s candles.",
                            len(df), symbol, timeframe)
                return df
            logger.warning("Live history too short (%s candles); training on synthetic.",
                           0 if df is None else len(df))
        except Exception as e:
            logger.warning("Could not fetch live history (%s); training on synthetic.", e)
    return make_synthetic_ohlcv(n=max(3000, n_bars))


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


class ContinuousTrainer:
    """Continuously retrains a per-symbol RL policy on the freshest market data and
    hot-swaps it into the live engine, so each symbol's strategy keeps adapting.

    How this answers "keep training itself based on the reward/penalty on each trade":
    the CEM objective is the episode return of the TradingEnv, whose per-bar reward is
    ``position * next_bar_return - cost*|turnover|`` (learning/env.py). That is exactly the
    realized PnL of trading, net of fees+slippage — profitable positions are rewarded and
    losing / churny ones are penalized. Each cycle WARM-STARTS CEM from the previous policy
    (so it refines rather than restarts), retrains on the latest bars, saves
    ``models/rl_<symbol>.npz``, and assigns a fresh ``RLStrategy`` onto the engine. The
    engine reads ``self.strategy`` at the top of every step, so the swap takes effect on the
    next iteration with no restart.

    Note: CEM on rolling windows is incremental *batch* retraining, not true per-tick online
    RL — but it is continuous (loops forever) and each round folds in the newest price action
    and the realized trading reward. It is honest to call this "continuously learning"; it is
    NOT evidence of edge (validate out-of-sample before trusting any policy).
    """

    def __init__(self, engines, event_bus, fetch_df, *, iterations: int,
                 population: int, train_bars: int, interval: float,
                 workers: int = 0, swap_into_engine: bool = True, advisor=None):
        self.engines = engines
        self.event_bus = event_bus
        self.fetch_df = fetch_df            # callable(symbol) -> DataFrame | None
        self.iterations = iterations
        self.population = population
        self.train_bars = train_bars
        self.interval = interval
        # Workers <= 0 means "decide from CPU count, capped at #symbols". This is the number of
        # SEPARATE PROCESSES training runs across — the whole point of the offload: it pulls
        # CEM off the main process so the HTTP/SSE server isn't GIL-starved (no more dashboard
        # connection errors) and symbols train truly in parallel across cores.
        import os as _os
        n_sym = max(1, len(engines))
        self.workers = int(workers) if workers and workers > 0 else \
            max(1, min(n_sym, (_os.cpu_count() or 2) - 1))
        self.swap_into_engine = swap_into_engine
        self.advisor = advisor              # optional ClaudeAdvisor: wraps the swapped strategy
        self._theta: typing.Dict[str, typing.Any] = {}   # symbol -> last flat policy
        self._stop = threading.Event()
        self._thread: typing.Optional[threading.Thread] = None
        self.cycle = 0
        self.running = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("ContinuousTrainer started: %d symbols across %d worker process(es), "
                    "%d iters/cycle, pop=%d, %d bars, every %.0fs between sweeps.",
                    len(self.engines), self.workers, self.iterations, self.population,
                    self.train_bars, self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.running = False

    def _publish(self, phase: str, sym: str, **kw) -> None:
        """Coarse, symbol-tagged train event. Per-iteration progress is NOT streamed in the
        multi-core path (it can't cross a process boundary) — instead each symbol emits a
        'start' when submitted and a 'done'/'error' when its worker returns. Fewer events also
        means a lighter SSE stream, which helps the dashboard stay connected."""
        if self.event_bus is None:
            return
        try:
            payload = {"phase": phase, "symbol": sym}
            payload.update(kw)
            self.event_bus.publish("train", payload)
        except Exception:
            pass

    def _prepare_job(self, sym):
        """Fetch + trim the freshest bars for one symbol on the MAIN process (the worker only
        does CPU-bound CEM). Returns (df, model_path) or None to skip."""
        import os
        df = self.fetch_df(sym)
        if df is None or len(df) < 300:
            logger.warning("[trainer] %s: insufficient data (%s bars); skipping.",
                           sym, 0 if df is None else len(df))
            return None
        if self.train_bars and len(df) > self.train_bars:
            df = df.iloc[-self.train_bars:].reset_index(drop=True)
        return df, os.path.join("models", f"rl_{sym}.npz")

    def _apply_result(self, sym, res, source: str = "worker") -> None:
        """Hot-swap a finished policy into the live engine + publish the outcome. `source` is
        just for the log line ("worker" = process pool, "in-thread" = sequential fallback)."""
        from strategies.rl_strategy import RLStrategy
        if not res or not res.get("ok"):
            err = (res or {}).get("error", "unknown")
            logger.error("[trainer] %s training failed: %s", sym, err)
            self._publish("error", sym, error=str(err))
            return
        self._theta[sym] = res.get("theta")
        if self.swap_into_engine and res.get("policy") and sym in self.engines:
            strat = RLStrategy(policy=res["policy"])
            if self.advisor is not None:    # optional Claude advisory overlay (gates trades)
                from advisors.claude_cli_advisor import AdvisedStrategy
                strat = AdvisedStrategy(strat, self.advisor, sym)
            self.engines[sym].strategy = strat
        best = res.get("best_reward", 0.0)
        logger.info("[trainer] %s cycle %d: best=%+.5f — RL policy hot-swapped (%s).",
                    sym, self.cycle, best, source)
        self._publish("done", sym, model_path=res.get("model_path"), best_reward=best)

    def _loop(self) -> None:
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        from learning.train import _pool_train_worker, train_policy

        os.makedirs("models", exist_ok=True)

        # Try to bring up a process pool. If the platform refuses (rare), degrade gracefully to
        # in-thread sequential training so the loop still runs — just without the GIL relief.
        executor = None
        if self.workers > 1:
            try:
                executor = ProcessPoolExecutor(max_workers=self.workers)
            except Exception as e:
                logger.warning("[trainer] process pool unavailable (%s); running sequentially "
                               "in-thread.", e)
                executor = None

        try:
            while not self._stop.is_set():
                self.cycle += 1
                jobs = {}  # symbol -> (df, model_path)
                for sym in list(self.engines.keys()):
                    if self._stop.is_set():
                        break
                    try:
                        prepared = self._prepare_job(sym)
                    except Exception as e:
                        logger.error("[trainer] %s data fetch error: %s", sym, e)
                        continue
                    if prepared is not None:
                        jobs[sym] = prepared

                if executor is not None:
                    # ---- multi-core path: one process per symbol, collect as they finish ----
                    # Bounded so a stuck worker can NEVER hang the loop: if a whole sweep
                    # overruns the deadline we cancel, fall back to in-thread for the rest of
                    # this run, and shut the pool down. Workers spawn lazily (Windows re-imports
                    # the app per process), so the first sweep is the slow one.
                    futures = {}
                    for sym, (df, model_path) in jobs.items():
                        if self._stop.is_set():
                            break
                        kw = dict(iterations=self.iterations, population=self.population,
                                  init_theta=self._theta.get(sym), model_path=model_path,
                                  seed=self.cycle)
                        self._publish("start", sym, iterations=self.iterations, bars=len(df))
                        try:
                            futures[executor.submit(_pool_train_worker, (sym, df, kw))] = sym
                        except Exception as e:
                            logger.error("[trainer] %s submit failed: %s", sym, e)
                    # Generous: worker spawn + heavy import + train, scaled to the batch.
                    deadline_s = 120.0 + 45.0 * len(futures)
                    try:
                        for fut in as_completed(futures, timeout=deadline_s):
                            if self._stop.is_set():
                                break
                            sym = futures[fut]
                            try:
                                res = fut.result()
                            except BrokenProcessPool:
                                raise               # handled below — pool is dead
                            except Exception as e:
                                res = {"ok": False, "symbol": sym, "error": str(e)}
                            self._apply_result(sym, res, source="worker")
                    except TimeoutError:
                        # A slow/asleep sweep (e.g. laptop suspended) — NOT a broken pool.
                        # Drop this sweep's stragglers and keep the pool for the next cycle;
                        # the next sweep retrains the same symbols anyway.
                        pending = [s for f, s in futures.items() if not f.done()]
                        logger.warning("[trainer] sweep exceeded %ds; skipping stragglers %s "
                                       "this cycle (pool kept).", int(deadline_s), pending)
                        for f in futures:
                            f.cancel()
                    except BrokenProcessPool as e:
                        # The pool is genuinely dead — fall back to in-thread for the rest of
                        # the run so training continues (without GIL relief).
                        logger.error("[trainer] worker pool broke (%s); switching to in-thread "
                                     "sequential training.", e)
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except Exception:
                            pass
                        executor = None
                else:
                    # ---- fallback: sequential, in this thread (no GIL relief) --------------
                    for sym, (df, model_path) in jobs.items():
                        if self._stop.is_set():
                            break
                        self._publish("start", sym, iterations=self.iterations, bars=len(df))
                        try:
                            res = train_policy(
                                df, iterations=self.iterations, population=self.population,
                                init_theta=self._theta.get(sym), model_path=model_path,
                                seed=self.cycle)
                            res["symbol"] = sym
                        except Exception as e:
                            res = {"ok": False, "symbol": sym, "error": str(e)}
                        self._apply_result(sym, res, source="in-thread")

                # Pause between full sweeps; responsive to stop().
                self._stop.wait(self.interval)
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-symbol live/paper trading dashboard.")
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--capital", type=float, default=10_000, help="capital PER symbol")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--poll", type=float, default=None, help="seconds between polls")
    ap.add_argument("--force-replay", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    # --- training (RL) -----------------------------------------------------
    ap.add_argument("--history", type=int, default=8000,
                    help="bars of history to fetch for training (paged from the exchange)")
    ap.add_argument("--no-continuous", action="store_true",
                    help="disable the always-on background retraining loop")
    ap.add_argument("--train-iterations", type=int, default=30,
                    help="CEM iterations per symbol per continuous cycle (training LENGTH)")
    ap.add_argument("--train-population", type=int, default=50,
                    help="CEM population per iteration (more = more thorough search)")
    ap.add_argument("--train-bars", type=int, default=4000,
                    help="most-recent bars each retraining cycle trains on (training LENGTH)")
    ap.add_argument("--train-interval", type=float, default=45.0,
                    help="seconds to pause between full retraining sweeps")
    ap.add_argument("--train-workers", type=int, default=0,
                    help="parallel training PROCESSES (0 = auto: min(#symbols, cpu-1)). "
                         "Offloads CEM off the main process so the dashboard stays responsive.")
    # --- optional Claude CLI advisor (no API key — uses the local `claude` subscription) ---
    ap.add_argument("--claude-advisor", action="store_true",
                    help="enable a SLOW Claude advisory overlay via the `claude` CLI that can "
                         "veto/bias trades (advisory only; needs the claude CLI on PATH)")
    ap.add_argument("--advisor-interval", type=float, default=300.0,
                    help="seconds between Claude advisory refreshes per symbol")
    ap.add_argument("--advisor-model", default="opus",
                    help="model for the Claude advisor (CLI alias, e.g. opus/sonnet/haiku)")
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

    # Per-symbol training-data provider: a long live history (paged from the exchange) when
    # reachable, else a stable synthetic frame for that symbol. Shared by the manual "Train
    # RL" button and the continuous retraining loop.
    def fetch_training_df(symbol: str):
        if use_live and real_client is not None:
            df = fetch_history(real_client, symbol, args.timeframe, args.history)
            if df is not None and len(df) >= 300:
                return df
            logger.warning("[trainer] %s: live history unavailable; using synthetic.", symbol)
        i = symbols.index(symbol) if symbol in symbols else 0
        start_px = _REPLAY_START_PRICE.get(symbol, 100.0 * (i + 1))
        return make_synthetic_ohlcv(n=max(3000, args.history), seed=7 + i,
                                    start_price=start_px)

    # Optional Claude advisory overlay (off unless --claude-advisor). Uses the local `claude`
    # CLI (no API key). Advisory only: it can veto/bias the engine's trades, never place them.
    advisor = None
    if args.claude_advisor:
        from advisors.claude_cli_advisor import ClaudeAdvisor, available as _claude_ok
        if _claude_ok():
            advisor = ClaudeAdvisor(list(engines.keys()), fetch_training_df, GLOBAL_BUS,
                                    interval=args.advisor_interval, timeframe=args.timeframe,
                                    model=args.advisor_model)
        else:
            logger.warning("--claude-advisor requested but `claude` CLI not on PATH; skipping.")

    # Always-on background learner: keeps each symbol's RL policy adapting to fresh data and
    # hot-swaps it into the live engine. Disable with --no-continuous.
    trainer = None
    if not args.no_continuous:
        trainer = ContinuousTrainer(
            engines, GLOBAL_BUS, fetch_training_df,
            iterations=args.train_iterations, population=args.train_population,
            train_bars=args.train_bars, interval=args.train_interval,
            workers=args.train_workers, advisor=advisor,
        )

    # Wire the "Train RL" button: train one symbol now on a long history, warm-starting from
    # the continuous trainer's latest policy for that symbol if it has one. Streams progress
    # to the dashboard. Without this the page reports "Training is not configured".
    train_symbol = next(iter(engines.keys()))

    def start_training():
        df = fetch_training_df(train_symbol)
        logger.info("Manual training: RL policy on %s (%d bars, %s)...", train_symbol,
                    len(df), "live data" if use_live else "synthetic feed")
        init_theta = trainer._theta.get(train_symbol) if trainer is not None else None
        import os
        return train_to_bus(GLOBAL_BUS, df=df, iterations=max(20, args.train_iterations),
                            population=args.train_population, init_theta=init_theta,
                            model_path=os.path.join("models", f"rl_{train_symbol}.npz"),
                            symbol=train_symbol)

    ctx = AppContext(event_bus=GLOBAL_BUS, engine=controller, start_training=start_training)
    server = DashboardServer(ctx, host=args.host, port=args.port)
    server.start()
    if advisor is not None:
        advisor.start()
    if trainer is not None:
        trainer.start()

    src = "LIVE Binance data" if use_live else "synthetic replay feed"
    logger.info("=" * 66)
    logger.info("Multi-symbol paper trading | %s | %d symbols: %s",
                src, len(engines), ", ".join(engines.keys()))
    logger.info("Dashboard -> %s", server.url)
    if args.host == "0.0.0.0":
        logger.info("From phone/LAN: http://<this-machine-ip>:%d/", args.port)
    if trainer is not None:
        logger.info("Continuous RL training: ON — policies retrain on fresh data and "
                    "hot-swap into the live engines (toggle with --no-continuous).")
    else:
        logger.info("Continuous RL training: OFF (engines trade EMA cross).")
    logger.info("Click ▶ Start. Trades for all symbols stream into one table.")
    logger.info("=" * 66)

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.0), webbrowser.open(server.url)),
                         daemon=True).start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        if advisor is not None:
            advisor.stop()
        if trainer is not None:
            trainer.stop()
        controller.stop()
        server.stop()


if __name__ == "__main__":
    main()
