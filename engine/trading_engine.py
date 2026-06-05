"""Live / paper trading loop.

`TradingEngine` ties a `Strategy`, a `RiskManager`, and an exchange connector together
into a background thread that, each iteration:

    1. pulls recent candles,
    2. asks the strategy for a target position (in {-1, 0, +1}),
    3. scales it through the risk manager into an allowed signed exposure,
    4. executes the delta versus the current position
       (paper: simulated against current bid/ask + fee/slippage; live: real order),
    5. marks the account to market, feeds equity back to the risk manager, and
    6. persists the fill and an equity point if a database is given.

Safety
------
Live trading is refused unless ALL of these hold:
    * env var ``ALLOW_LIVE_TRADING == "true"`` (read at construction time), and
    * the caller passes ``confirm_live=True``.
If either is missing the engine logs *why* and forces ``mode`` back to ``"paper"``.
Real ``client.place_order`` calls happen ONLY in live mode; paper mode never touches
the exchange for order placement.

The exchange-specific ``place_order`` argument order differs between Binance and BitMEX,
so order placement is abstracted behind :meth:`_place_real_order`.
"""

import logging
import os
import threading
import time
import typing
import uuid
from datetime import datetime

import pandas as pd

logger = logging.getLogger()

# Paper-trading cost model (fraction of notional).
PAPER_FEE = 0.0004
PAPER_SLIPPAGE = 0.0002

# Map a timeframe string to its length in seconds, used to pace the live loop.
_TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1_800,
    "1h": 3_600, "4h": 14_400, "1d": 86_400,
}


class TradingEngine:
    """Threaded paper/live trading loop. See module docstring for behaviour."""

    def __init__(
        self,
        client,
        strategy,
        risk_manager,
        database=None,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        mode: str = "paper",
        initial_capital: float = 10_000,
        notifier=None,
        evaluator=None,
        run_id: typing.Optional[str] = None,
        poll_seconds: typing.Optional[float] = None,
        confirm_live: bool = False,
        event_bus=None,
        publish_step_updates: bool = True,
    ):
        self.client = client
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.database = database
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = float(initial_capital)
        self.notifier = notifier
        self.evaluator = evaluator
        self.event_bus = event_bus       # optional pub/sub for the live dashboard
        # When several engines share one bus/dashboard, a controller aggregates and
        # publishes the combined state/equity; individual engines then suppress their
        # own per-step state/equity events (fills are always published, symbol-tagged).
        self.publish_step_updates = publish_step_updates
        self.run_id = run_id or uuid.uuid4().hex[:12]

        # Resolve and guard the requested mode (may downgrade live -> paper).
        self.mode = self._resolve_mode(mode, confirm_live)

        # Loop pacing: explicit override, else fast for paper, timeframe-based for live.
        if poll_seconds is not None:
            self.poll_seconds = float(poll_seconds)
        elif self.mode in ("live", "testnet"):
            self.poll_seconds = float(_TF_SECONDS.get(timeframe, 3_600))
        else:
            self.poll_seconds = 5.0

        # Internal paper/live account state.
        self.cash = self.initial_capital
        self.position = 0.0            # signed quantity of the base asset held
        self.entry_price = 0.0         # avg entry of the current open position
        self.realized_pnl = 0.0
        self.equity = self.initial_capital
        self.last_price = 0.0
        self.last_signal = 0.0
        self.n_trades = 0
        self._equity_history: typing.List[typing.Tuple[int, float]] = []

        # Threading.
        self.running = False
        self._thread: typing.Optional[threading.Thread] = None

        # Optional hook fed each fill, shaped for TradesWatch.add_trade.
        self.trade_callback: typing.Optional[typing.Callable[[dict], None]] = None

        self._exchange_name = self._infer_exchange(client)

    # ------------------------------------------------------------------ setup
    def _resolve_mode(self, mode: str, confirm_live: bool) -> str:
        """Return a safe mode, downgrading live->paper unless all gates pass."""
        mode = (mode or "paper").lower()
        if mode == "paper":
            return "paper"
        # TESTNET: real orders on Binance's testnet matching engine, but FAKE money. It places
        # real orders (so fills/slippage/min-notional/leverage/liquidation are real) yet needs
        # NO real-money gate — the whole point is risk-free realism. The client must itself be
        # pointed at testnet (testnet=True) with testnet keys; this engine just routes orders.
        if mode == "testnet":
            return "testnet"
        if mode != "live":
            return "paper"

        allow_env = os.getenv("ALLOW_LIVE_TRADING", "").strip().lower() == "true"
        if not allow_env:
            logger.warning(
                "LIVE trading requested but env ALLOW_LIVE_TRADING != 'true'; "
                "forcing PAPER mode."
            )
            return "paper"
        if not confirm_live:
            logger.warning(
                "LIVE trading requested but confirm_live=False; forcing PAPER mode."
            )
            return "paper"

        logger.warning(
            "!!! LIVE TRADING ENABLED on %s %s — real orders will be placed. !!!",
            self._infer_exchange(self.client), self.symbol,
        )
        return "live"

    @staticmethod
    def _infer_exchange(client) -> str:
        name = type(client).__name__.lower()
        if "binance" in name:
            return "binance"
        if "bitmex" in name:
            return "bitmex"
        if "replay" in name:
            return "replay"
        return "unknown"

    # ----------------------------------------------------------- thread mgmt
    def start(self) -> None:
        """Spin up the background trading loop (idempotent)."""
        if self.running:
            logger.info("TradingEngine already running.")
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("TradingEngine started (mode=%s, symbol=%s, run_id=%s).",
                    self.mode, self.symbol, self.run_id)
        self._notify(f"Engine started: {self.mode} {self.symbol} ({self.run_id})")
        self._publish("engine", {"event": "started", "state": self.get_state()})

    def stop(self) -> None:
        """Signal the loop to stop. Does not join (loop is daemon)."""
        if not self.running:
            return
        self.running = False
        logger.info("TradingEngine stop requested (run_id=%s).", self.run_id)
        self._notify(f"Engine stopped: {self.mode} {self.symbol} ({self.run_id})")
        self._publish("engine", {"event": "stopped", "state": self.get_state()})

    # ------------------------------------------------------------- main loop
    def _run_loop(self) -> None:
        while self.running:
            try:
                self._step()
            except Exception as e:  # never let one bad iteration kill the thread
                logger.error("TradingEngine iteration error: %s", e, exc_info=True)
            # Sleep in small slices so stop() is responsive.
            slept = 0.0
            while self.running and slept < self.poll_seconds:
                time.sleep(min(0.5, self.poll_seconds - slept))
                slept += 0.5

    def _step(self) -> None:
        """One trading iteration."""
        df = self._get_candles()
        if df is None or len(df) == 0:
            logger.debug("No candles available for %s; skipping iteration.", self.symbol)
            return

        signals = self.strategy.generate_signals(df)
        if signals is None or len(signals) == 0:
            return
        raw_signal = float(signals.iloc[-1])
        self.last_signal = raw_signal

        price = self._mark_price(df)
        if price <= 0:
            return
        self.last_price = price

        # Risk manager scales the raw signal into an allowed signed exposure fraction.
        target_fraction = self.risk_manager.target_position(raw_signal)

        # Desired signed quantity for that exposure at the current price/equity.
        equity_now = self._compute_equity(price)
        desired_qty = (target_fraction * equity_now) / price

        # Execute the delta vs. the current position.
        delta = desired_qty - self.position
        # Ignore dust deltas to avoid churn.
        if abs(delta) > 1e-9 and abs(delta * price) >= 0.01:
            self._execute(delta, price)

        # Mark to market and feed equity back into the risk manager.
        equity_now = self._compute_equity(price)
        self.equity = equity_now
        self.risk_manager.update_equity(equity_now)
        self._record_equity(equity_now)

        # Stream a live equity point + full state snapshot to any dashboard subscribers.
        # Suppressed when a controller publishes aggregated updates (multi-symbol mode).
        if self.publish_step_updates:
            self._publish("equity", {"ts": self._now_ms(), "equity": round(equity_now, 2),
                                     "price": round(price, 2), "signal": self.last_signal})
            self._publish("state", self.get_state())

        if self.database is not None:
            try:
                self.database.save_equity_point(self.run_id, self._now_ms(), equity_now)
            except Exception as e:
                logger.error("Failed to persist equity point: %s", e)

        # Kill switch: flatten and pause.
        if getattr(self.risk_manager, "halted", False):
            if self.position != 0.0:
                logger.warning("Risk halt active — flattening position.")
                self._execute(-self.position, price, reason="HALT_FLATTEN")
                self.equity = self._compute_equity(price)
            logger.warning("Risk manager HALTED — pausing trading.")
            self._notify("RISK HALT — trading paused, position flattened.")
            self.running = False

    # --------------------------------------------------------------- candles
    def _get_candles(self) -> typing.Optional[pd.DataFrame]:
        """Pull recent candles via DataManager (if DB given) else the connector."""
        if self.database is not None:
            try:
                from data.data_manager import DataManager
                dm = DataManager(self.database)
                return dm.get_candles(self.client, self.symbol, self.timeframe)
            except Exception as e:
                logger.error("DataManager.get_candles failed (%s); falling back.", e)

        contract = self.client.contracts.get(self.symbol)
        if contract is None:
            logger.error("Unknown symbol %r on %s.", self.symbol, self._exchange_name)
            return None
        candles = self.client.get_historical_candles(contract, self.timeframe)
        if not candles:
            return None
        return pd.DataFrame(
            [(c.timestamp, c.open, c.high, c.low, c.close, c.volume) for c in candles],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

    def _mark_price(self, df: pd.DataFrame) -> float:
        """Best available mark price: live bid/ask mid if present, else last close."""
        prices = getattr(self.client, "prices", {}).get(self.symbol)
        if prices and prices.get("bid") and prices.get("ask"):
            return (float(prices["bid"]) + float(prices["ask"])) / 2.0
        try:
            return float(df["close"].iloc[-1])
        except Exception:
            return 0.0

    def _bid_ask(self, fallback: float) -> typing.Tuple[float, float]:
        """Current (bid, ask), falling back to a flat price if unavailable."""
        prices = getattr(self.client, "prices", {}).get(self.symbol)
        if prices and prices.get("bid") and prices.get("ask"):
            return float(prices["bid"]), float(prices["ask"])
        return fallback, fallback

    # ------------------------------------------------------------- execution
    def _execute(self, delta_qty: float, mark_price: float, reason: str = "") -> None:
        """Execute a signed quantity change. Paper simulates; live places real orders."""
        side = "BUY" if delta_qty > 0 else "SELL"
        qty = abs(delta_qty)

        if self.mode in ("live", "testnet"):
            fill_price = self._place_real_order(side, qty, mark_price)
            if fill_price is None:
                logger.error("%s order returned no fill; state unchanged.", self.mode)
                return
        else:
            fill_price = self._simulate_fill(side, mark_price)

        self._apply_fill(side, qty, fill_price, reason)

    def _simulate_fill(self, side: str, mark_price: float) -> float:
        """Paper fill price: cross the spread and add slippage in the adverse direction."""
        bid, ask = self._bid_ask(mark_price)
        base = ask if side == "BUY" else bid
        slip = base * PAPER_SLIPPAGE
        return base + slip if side == "BUY" else base - slip

    def _place_real_order(self, side: str, qty: float,
                          mark_price: float) -> typing.Optional[float]:
        """Abstract over the differing Binance/BitMEX place_order signatures."""
        contract = self.client.contracts.get(self.symbol)
        if contract is None:
            logger.error("Cannot place live order: unknown symbol %r.", self.symbol)
            return None
        try:
            if self._exchange_name == "bitmex":
                # BitmexClient.place_order(contract, order_type, quantity, side, ...)
                status = self.client.place_order(contract, "Market", qty, side)
            else:
                # BinanceFuturesClient.place_order(contract, side, quantity, order_type, ...)
                status = self.client.place_order(contract, side, qty, "MARKET")
        except Exception as e:
            logger.error("Live place_order raised: %s", e)
            return None

        if status is None:
            return None
        avg = getattr(status, "avg_price", 0.0) or 0.0
        return float(avg) if avg else mark_price

    def _apply_fill(self, side: str, qty: float, fill_price: float, reason: str) -> None:
        """Update cash/position/realized-pnl for a fill and emit the trade record."""
        signed = qty if side == "BUY" else -qty
        fee = abs(qty * fill_price) * PAPER_FEE

        prev_pos = self.position
        new_pos = prev_pos + signed

        realized_delta = 0.0
        # Realize PnL on the portion of the trade that reduces/closes the position.
        if prev_pos != 0 and (prev_pos > 0) != (signed > 0):
            closing = min(abs(signed), abs(prev_pos))
            direction = 1 if prev_pos > 0 else -1
            realized_delta = (fill_price - self.entry_price) * closing * direction
            self.realized_pnl += realized_delta

        # Cash flow: pay for buys, receive for sells, always pay fee.
        self.cash -= signed * fill_price
        self.cash -= fee

        # Update entry price / position.
        if (prev_pos >= 0 and signed > 0) or (prev_pos <= 0 and signed < 0):
            # Adding to (or opening) a position in the same direction: weighted avg.
            total = abs(prev_pos) + abs(signed)
            if total > 0:
                self.entry_price = (
                    abs(prev_pos) * self.entry_price + abs(signed) * fill_price
                ) / total
        elif (prev_pos > 0) != (new_pos > 0) and new_pos != 0:
            # Flipped through zero: remaining position opens at the fill price.
            self.entry_price = fill_price
        elif new_pos == 0:
            self.entry_price = 0.0

        self.position = new_pos
        self.n_trades += 1

        logger.info("FILL [%s] %s %.6f %s @ %.2f (fee %.4f, rpnl %.4f)%s",
                    self.mode, side, qty, self.symbol, fill_price, fee, realized_delta,
                    f" [{reason}]" if reason else "")

        trade = {
            "run_id": self.run_id,
            "timestamp": self._now_ms(),
            "exchange": self._exchange_name,
            "symbol": self.symbol,
            "strategy": getattr(self.strategy, "name", "strategy"),
            "side": side,
            "quantity": round(qty, 8),
            "price": round(fill_price, 8),
            "pnl": round(realized_delta, 8),
            "status": "filled",
        }

        if self.database is not None:
            try:
                self.database.save_trade(trade)
            except Exception as e:
                logger.error("Failed to persist trade: %s", e)

        self._emit_trade_callback(trade)
        self._notify(
            f"Fill: {side} {qty:.6f} {self.symbol} @ {fill_price:.2f} "
            f"(rpnl {realized_delta:.2f})"
        )

    def _emit_trade_callback(self, trade: dict) -> None:
        ui = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "symbol": trade["symbol"],
            "exchange": trade["exchange"],
            "strategy": trade["strategy"],
            "side": trade["side"],
            "quantity": trade["quantity"],
            "price": trade.get("price"),
            "status": trade["status"],
            "pnl": trade["pnl"],
        }
        # Always stream the fill to the event bus (the dashboard listens here);
        # the Tk trade_callback is optional and only present in the desktop GUI.
        self._publish("fill", ui)
        if self.trade_callback is None:
            return
        try:
            self.trade_callback(ui)
        except Exception as e:
            logger.error("trade_callback raised: %s", e)

    # --------------------------------------------------------------- account
    def _compute_equity(self, price: float) -> float:
        """Equity = cash + position marked to market."""
        return self.cash + self.position * price

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        return (price - self.entry_price) * self.position

    def _record_equity(self, equity: float) -> None:
        self._equity_history.append((self._now_ms(), equity))

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # ----------------------------------------------------------- event bus
    def _publish(self, type: str, data: typing.Any = None) -> None:
        if self.event_bus is None:
            return
        try:
            self.event_bus.publish(type, data)
        except Exception as e:
            logger.error("event_bus.publish(%s) raised: %s", type, e)

    # ----------------------------------------------------------- notifier
    def _notify(self, message: str) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.send(message)
        except Exception as e:
            logger.error("Notifier.send raised: %s", e)

    # ------------------------------------------------------------- evaluator
    def assess_readiness(self) -> dict:
        """Ask the optional evaluator whether this run is fit to promote.

        Duck-typed: any object exposing ``assess(equity_curve, returns)`` works. Returns
        the evaluator's verdict dict, or a default ``not ready`` verdict if absent.
        """
        if self.evaluator is None:
            return {"ready": False, "reasons": ["no evaluator"]}

        if self._equity_history:
            idx = [ts for ts, _ in self._equity_history]
            vals = [eq for _, eq in self._equity_history]
            equity_curve = pd.Series(vals, index=idx, name="equity")
            days_running = max(0.0, (idx[-1] - idx[0]) / 86_400_000.0)
        else:
            equity_curve = pd.Series(dtype="float64", name="equity")
            days_running = 0.0
        returns = equity_curve.pct_change().dropna()

        try:
            verdict = self.evaluator.assess(
                equity_curve, returns,
                n_trades=self.n_trades, days_running=days_running,
                timeframe=self.timeframe,
            )
        except Exception as e:
            logger.error("Evaluator.assess failed: %s", e)
            return {"ready": False, "reasons": [f"evaluator error: {e}"]}

        # Normalize to a plain dict for GUI/alerts regardless of verdict type.
        return verdict.as_dict() if hasattr(verdict, "as_dict") else verdict

    # ----------------------------------------------------------------- state
    def get_state(self) -> dict:
        """Snapshot of engine/account state for the GUI and callers."""
        price = self.last_price
        return {
            "mode": self.mode,
            "symbol": self.symbol,
            "running": self.running,
            "position": self.position,
            "equity": self.equity,
            "cash": self.cash,
            "unrealized_pnl": self._unrealized_pnl(price),
            "realized_pnl": self.realized_pnl,
            "n_trades": self.n_trades,
            "halted": getattr(self.risk_manager, "halted", False),
            "last_signal": self.last_signal,
        }
