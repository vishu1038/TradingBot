"""Tkinter control panel for the trading engine.

Decoupled from any concrete engine: it is wired with plain callables
(``on_start``, ``on_stop``, ``on_check_readiness``, ``get_state``) so the GUI never
imports the engine directly. It polls ``get_state()`` on a ``self.after`` loop to refresh
its live readouts (equity, position, PnL, trade count, halt status) and shows the
strategy selector, symbol entry, mode label, and promotion status.
"""

import logging
import tkinter as tk
import typing

from interface.styling import *

logger = logging.getLogger()


class ControlPanel(tk.Frame):
    def __init__(
        self,
        *args,
        on_start: typing.Optional[typing.Callable[[], None]] = None,
        on_stop: typing.Optional[typing.Callable[[], None]] = None,
        on_check_readiness: typing.Optional[typing.Callable[[], dict]] = None,
        get_state: typing.Optional[typing.Callable[[], dict]] = None,
        get_strategy: typing.Optional[typing.Callable[[], str]] = None,
        strategies: typing.Optional[typing.List[str]] = None,
        symbol: str = "BTCUSDT",
        mode: str = "PAPER",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._on_start = on_start
        self._on_stop = on_stop
        self._on_check_readiness = on_check_readiness
        self._get_state = get_state

        self.configure(bg=BG_COLOR)

        # --- Inputs row -------------------------------------------------
        self._inputs = tk.Frame(self, bg=BG_COLOR)
        self._inputs.pack(side=tk.TOP, fill=tk.X, pady=4)

        tk.Label(self._inputs, text="Strategy:", bg=BG_COLOR, fg=FG_COLOR,
                 font=BOLD_FONT).grid(row=0, column=0, padx=4, sticky="w")
        self.strategy_var = tk.StringVar(value=(strategies or ["EMA", "ML"])[0])
        self._strategy_menu = tk.OptionMenu(self._inputs, self.strategy_var,
                                            *(strategies or ["EMA", "ML"]))
        self._strategy_menu.configure(bg=BG_COLOR_2, fg=FG_COLOR, font=GLOBAL_FONT,
                                      highlightthickness=0)
        self._strategy_menu.grid(row=0, column=1, padx=4)

        tk.Label(self._inputs, text="Symbol:", bg=BG_COLOR, fg=FG_COLOR,
                 font=BOLD_FONT).grid(row=0, column=2, padx=4, sticky="w")
        self.symbol_var = tk.StringVar(value=symbol)
        self._symbol_entry = tk.Entry(self._inputs, textvariable=self.symbol_var,
                                     bg=BG_COLOR_2, fg=FG_COLOR, font=GLOBAL_FONT,
                                     justify=tk.CENTER, width=10)
        self._symbol_entry.grid(row=0, column=3, padx=4)

        tk.Label(self._inputs, text="Mode:", bg=BG_COLOR, fg=FG_COLOR,
                 font=BOLD_FONT).grid(row=0, column=4, padx=4, sticky="w")
        self.mode_var = tk.StringVar(value=mode.upper())
        self._mode_label = tk.Label(self._inputs, textvariable=self.mode_var,
                                    bg=BG_COLOR, fg=FG_COLOR_2, font=BOLD_FONT)
        self._mode_label.grid(row=0, column=5, padx=4)

        # --- Buttons row ------------------------------------------------
        self._buttons = tk.Frame(self, bg=BG_COLOR)
        self._buttons.pack(side=tk.TOP, fill=tk.X, pady=4)

        self._start_btn = tk.Button(self._buttons, text="Start", command=self._handle_start,
                                    bg=BG_COLOR_2, fg=FG_COLOR, font=GLOBAL_FONT, width=10)
        self._start_btn.grid(row=0, column=0, padx=4)

        self._stop_btn = tk.Button(self._buttons, text="Stop", command=self._handle_stop,
                                   bg=BG_COLOR_2, fg=FG_COLOR, font=GLOBAL_FONT, width=10)
        self._stop_btn.grid(row=0, column=1, padx=4)

        self._readiness_btn = tk.Button(self._buttons, text="Check Readiness",
                                        command=self._handle_check_readiness,
                                        bg=BG_COLOR_2, fg=FG_COLOR, font=GLOBAL_FONT,
                                        width=14)
        self._readiness_btn.grid(row=0, column=2, padx=4)

        # --- Readout labels --------------------------------------------
        self._readout = tk.Frame(self, bg=BG_COLOR)
        self._readout.pack(side=tk.TOP, fill=tk.X, pady=4)

        self._vars: typing.Dict[str, tk.StringVar] = {}
        readouts = [
            ("Equity", "equity"),
            ("Position", "position"),
            ("Realized PnL", "realized_pnl"),
            ("Unrealized PnL", "unrealized_pnl"),
            ("# Trades", "n_trades"),
            ("HALT", "halted"),
        ]
        for i, (label, key) in enumerate(readouts):
            tk.Label(self._readout, text=label + ":", bg=BG_COLOR, fg=FG_COLOR,
                     font=BOLD_FONT).grid(row=i, column=0, sticky="w", padx=4)
            var = tk.StringVar(value="-")
            self._vars[key] = var
            tk.Label(self._readout, textvariable=var, bg=BG_COLOR, fg=FG_COLOR_2,
                     font=GLOBAL_FONT).grid(row=i, column=1, sticky="w", padx=4)

        # Promotion status line.
        tk.Label(self._readout, text="Promotion:", bg=BG_COLOR, fg=FG_COLOR,
                 font=BOLD_FONT).grid(row=len(readouts), column=0, sticky="w", padx=4)
        self.promotion_var = tk.StringVar(value="unknown")
        tk.Label(self._readout, textvariable=self.promotion_var, bg=BG_COLOR,
                 fg=FG_COLOR_2, font=GLOBAL_FONT).grid(
            row=len(readouts), column=1, sticky="w", padx=4)

        self._refresh_loop()

    # ------------------------------------------------------------ handlers
    def _handle_start(self) -> None:
        if self._on_start is not None:
            try:
                self._on_start()
            except Exception as e:
                logger.error("Start callback failed: %s", e)

    def _handle_stop(self) -> None:
        if self._on_stop is not None:
            try:
                self._on_stop()
            except Exception as e:
                logger.error("Stop callback failed: %s", e)

    def _handle_check_readiness(self) -> None:
        if self._on_check_readiness is None:
            self.promotion_var.set("no evaluator")
            return
        try:
            verdict = self._on_check_readiness() or {}
        except Exception as e:
            logger.error("Readiness callback failed: %s", e)
            self.promotion_var.set("error")
            return
        ready = verdict.get("ready", False)
        reasons = verdict.get("reasons", [])
        text = "READY" if ready else "NOT READY"
        if reasons:
            text += " (" + "; ".join(str(r) for r in reasons) + ")"
        self.promotion_var.set(text)

    # ------------------------------------------------------------ refresh
    def _refresh_loop(self) -> None:
        if self._get_state is not None:
            try:
                state = self._get_state() or {}
                self._apply_state(state)
            except Exception as e:
                logger.error("get_state failed: %s", e)
        self.after(1000, self._refresh_loop)

    def _apply_state(self, state: dict) -> None:
        if "mode" in state:
            self.mode_var.set(str(state["mode"]).upper())
        if "equity" in state:
            self._vars["equity"].set(f"{state['equity']:.2f}")
        if "position" in state:
            self._vars["position"].set(f"{state['position']:.6f}")
        if "realized_pnl" in state:
            self._vars["realized_pnl"].set(f"{state['realized_pnl']:.2f}")
        if "unrealized_pnl" in state:
            self._vars["unrealized_pnl"].set(f"{state['unrealized_pnl']:.2f}")
        if "n_trades" in state:
            self._vars["n_trades"].set(str(state["n_trades"]))
        if "halted" in state:
            self._vars["halted"].set("YES" if state["halted"] else "no")
