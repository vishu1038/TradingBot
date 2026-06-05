import logging
import tkinter as tk
import logging

from connectors.bitmex import BitmexClient
from connectors.binance_futures import BinanceFuturesClient

from interface.styling import *
from interface.logging_component import Logging
from interface.watchlist_component import Watchlist
from interface.trades_component import TradesWatch
from interface.control_component import ControlPanel


logger = logging.getLogger()



class Root(tk.Tk):
    def __init__(self, binance: BinanceFuturesClient, bitmex: BitmexClient, engine=None):
        super().__init__()

        self.binance = binance
        self.bitmex = bitmex
        self.engine = engine
        self.title("Trading Bot")

        self.configure(bg=BG_COLOR)

        self._left_frame = tk.Frame(self, bg=BG_COLOR)
        self._left_frame.pack(side=tk.LEFT)

        self._right_frame = tk.Frame(self, bg=BG_COLOR)
        self._right_frame.pack(side=tk.LEFT)

        self._watchlist_frame = Watchlist(self.binance.contracts, self.bitmex.contracts,
                                          self._left_frame, bg=BG_COLOR)
        self._watchlist_frame.pack(side=tk.TOP)

        self._logging_frame = Logging(self._left_frame, bg=BG_COLOR)
        self._logging_frame.pack(side=tk.TOP)

        self._trades_frame = TradesWatch(self._right_frame, bg=BG_COLOR)
        self._trades_frame.pack(side=tk.TOP)

        # Control panel wired to the engine (if one was supplied). The panel stays
        # decoupled: it only receives callables, never the engine object itself.
        if self.engine is not None:
            # Feed each engine fill into the trades table. add_trade must run on the
            # Tk thread, so marshal the call via .after(0, ...).
            self.engine.trade_callback = self._on_engine_trade

            self._control_frame = ControlPanel(
                self._right_frame,
                on_start=self.engine.start,
                on_stop=self.engine.stop,
                on_check_readiness=self.engine.assess_readiness,
                get_state=self.engine.get_state,
                symbol=getattr(self.engine, "symbol", "BTCUSDT"),
                mode=getattr(self.engine, "mode", "paper"),
                bg=BG_COLOR,
            )
            self._control_frame.pack(side=tk.TOP, pady=6)

        self._update_ui()

    def _on_engine_trade(self, data: dict):
        """Engine trade_callback hook: marshal the UI update onto the Tk thread."""
        try:
            self.after(0, lambda: self._trades_frame.add_trade(data))
        except RuntimeError:
            # Tk may be shutting down; ignore.
            pass

    def _update_ui(self):

        # Logs

        for log in self.bitmex.logs:
            if not log['displayed']:
                self._logging_frame.add_log(log['log'])
                log['displayed'] = True

        for log in self.binance.logs:
            if not log['displayed']:
                self._logging_frame.add_log(log['log'])
                log['displayed'] = True

        # WatchList Prices

        try:
            for key, value in self._watchlist_frame.body_widgets['symbol'].items():

                symbol = self._watchlist_frame.body_widgets['symbol'][key].cget("text")
                exchange = self._watchlist_frame.body_widgets['exchange'][key].cget("text")

                if exchange == "Binance":
                    if symbol not in self.binance.contracts:
                        continue

                    if symbol not in self.binance.prices:
                        self.binance.get_bid_ask(self.binance.contracts[symbol])
                        continue

                    precision = self.binance.contracts[symbol].price_decimals

                    prices = self.binance.prices[symbol]

                elif exchange == "Bitmex":
                    if symbol not in self.bitmex.contracts:
                        continue

                    if symbol not in self.bitmex.prices:
                        continue

                    precision = self.bitmex.contracts[symbol].price_decimals

                    prices = self.bitmex.prices[symbol]

                else:
                    continue

                if prices['bid'] is not None:
                    price_str = "{0:.{prec}f}".format(prices['bid'], prec=precision)
                    self._watchlist_frame.body_widgets['bid_var'][key].set(price_str)

                if prices['ask'] is not None:
                    price_str = "{0:.{prec}f}".format(prices['ask'], prec=precision)
                    self._watchlist_frame.body_widgets['ask_var'][key].set(price_str)

        except RuntimeError as e:
            logger.error("Error while looping through the watchlist dictionary: %s", e)
        self.after(1500, self._update_ui)
