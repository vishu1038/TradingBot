"""Risk management — the hard safety floor that sits above every strategy.

No matter how clever a rule/ML/RL strategy thinks it is, it goes through here. The risk
manager decides *how much* to act on a target signal and can force a flat position (kill
switch) when account-level limits are breached. In live/paper trading the executor consults
this before sending any order; the backtester uses `risk_fraction` for the simplified case.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger()


@dataclass
class RiskLimits:
    risk_fraction: float = 0.5        # fraction of equity per full-strength signal
    max_position: float = 1.0         # hard cap on signed exposure fraction
    max_drawdown: float = 0.20        # kill switch if equity falls this far from peak
    stop_loss_pct: float = 0.05       # per-position stop (executor-enforced)
    take_profit_pct: float = 0.10     # per-position target (executor-enforced)
    max_open_positions: int = 1


class RiskManager:
    def __init__(self, limits: RiskLimits = None, starting_equity: float = 10_000.0):
        self.limits = limits or RiskLimits()
        self.peak_equity = starting_equity
        self.equity = starting_equity
        self.halted = False

    def update_equity(self, equity: float) -> None:
        """Feed the latest account equity; trips the kill switch on drawdown breach."""
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = equity / self.peak_equity - 1.0
        if drawdown <= -self.limits.max_drawdown and not self.halted:
            self.halted = True
            logger.warning("RISK HALT: drawdown %.1f%% breached limit %.1f%%",
                           drawdown * 100, -self.limits.max_drawdown * 100)

    def target_position(self, signal: float) -> float:
        """Convert a raw signal in [-1, 1] into an allowed signed exposure fraction.

        Returns 0 (flat) while halted. Otherwise scales by risk_fraction and clamps to
        +/- max_position.
        """
        if self.halted:
            return 0.0
        scaled = signal * self.limits.risk_fraction
        cap = self.limits.max_position
        return max(-cap, min(cap, scaled))

    def position_size(self, signal: float, price: float) -> float:
        """Quantity (in contracts/coins) for `signal` at `price`, given current equity."""
        if price <= 0:
            return 0.0
        notional = abs(self.target_position(signal)) * self.equity
        qty = notional / price
        return qty if signal >= 0 else -qty

    def reset(self, starting_equity: float = None) -> None:
        if starting_equity is not None:
            self.equity = self.peak_equity = starting_equity
        self.halted = False
