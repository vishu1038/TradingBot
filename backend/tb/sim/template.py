"""The shared trade template as pure functions (sizing, TP/SL levels, liquidation, ATR, break-even).

Fill model (documented once, used by the event simulator and mirrored by the labels module):
  * entry: market at the OPEN of the bar after the signal bar, adverse slippage `slippage`
  * TP: resting limit -> fills at exactly the TP price (no slippage), taker fee charged
  * SL / trailing stop: stop-market -> fills at the stop price (or the bar open if the bar gaps
    through it) with adverse slippage `slippage + stop_slippage_extra`
  * kill-switch / liquidation / forced close: market at the bar close with `slippage`
  * fee = fee_taker * qty * fill_price on each side
  * TP/SL levels are measured from the actual (slipped) entry fill (as in tb.labels) and are GROSS price levels: a
    "+5 USD TP" is +5 USD of price PnL on the position's notional, fees/slippage come on top.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    start_equity: float = 100.0
    slice_margin_usd: float = 20.0
    leverage: float = 8.0
    max_positions: int = 5                 # cap; effective = min(cap, floor(equity / slice_margin))
    exit_mode: str = "fixed"               # fixed | atr
    tp_usd: float = 5.0
    sl_usd: float = 10.0
    tp_atr: float = 1.5
    sl_atr: float = 1.0
    trail_enabled: bool = True             # profit protection (2026-10-19): same defaults as tb.config.Settings
    trail_activate_r: float = 1.5          # activate once price moved this many initial risks in favour
    trail_distance_atr: float = 1.0        # stop trails this many ATRs behind the best price
    breakeven_r: float = 1.0               # stop -> entry +- round-trip fees after this many initial risks in favour (0 = off)
    sl_cooldown_bars: int = 24             # no new entry on a coin for N bars after a LOSING sl / trail / kill / liquidation exit
    max_open_risk_pct: float = 20.0        # open-risk budget (% of equity) the side cap below is a share of
    max_side_risk_pct: float = 60.0        # open stop-out risk of ONE side (incl. the new entry) <= this % of that budget; >= 100 = off
    partial_tp_r: float = 1.5              # close partial_tp_frac of the position at this many initial risks in favour (0 = off)
    partial_tp_frac: float = 0.5
    short_enabled: bool = True             # symmetric long/short (2026-10-19): same defaults as tb.config.Settings
    thr_long: float = 0.70
    thr_short: float = 0.70
    min_tp_usd: float = 2.0                # dollar floor (user 2026-10-19): skip a trade whose TP / SL span fewer USD (0 = off); also the
    min_sl_usd: float = 2.0                # floor of the partial take-profit's gain (otherwise: no partial, run to the full TP)
    max_net_exposure_pct: float = 150.0    # reversal protection (tb.reversal): |sum(side x notional x beta)| <= this % of equity (0 = off)
    reversal_z: float = 2.5                # basket z-score against the book's net side that locks winners + blocks that side (0 = off)
    reversal_lock_r: float = 0.5
    reversal_block_bars: int = 6
    fee_taker: float = 0.0005
    slippage: float = 0.0002
    stop_slippage_extra: float = 0.0005   # same as labels' stop_slippage_pct
    mmr: float = 0.005                     # maintenance margin rate used for the liquidation price
    kill_dd_pct: float = 50.0              # drawdown that closes everything (>=100: off)
    dd_basis: str = "start"                # drawdown vs the starting equity ("start", as live) or the peak ("peak")
    corr_max: float = 0.8
    corr_max_correlated: int = 2
    corr_window: int = 288                 # 24h of 5m bars
    corr_min_obs: int = 144
    # safety & robustness layer (docs/ARCHITECTURE.md), mirrored from the live engine
    derate_enabled: bool = True            # risk x tb.risk.derate_multiplier(dd_pct, kill_dd_pct)
    min_notional_usd: float = 5.0          # exchange min notional: skip trades derated below it
    plausibility_enabled: bool = True
    fault_mature_n: int = 3
    fault_recover_m: int = 5
    drift_enabled: bool = True             # auto-off for signal sources without real probabilities
    drift_n: int = 30
    atr_period: int = 14
    one_per_symbol: bool = True
    apply_funding: bool = True

    @property
    def notional(self) -> float:
        return self.slice_margin_usd * self.leverage

    def to_dict(self) -> dict:
        return asdict(self)


def atr_wilder(high, low, close, n: int = 14) -> np.ndarray:
    """ATR(n), Wilder smoothing (EWM alpha=1/n seeded at the first TR); NaN for the first n-1 bars.
    Value at index i only uses bars <= i."""
    h = np.asarray(high, dtype="float64")
    l = np.asarray(low, dtype="float64")
    c = np.asarray(close, dtype="float64")
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()
    return atr


def liq_price(entry: float, side: int, leverage: float, mmr: float) -> float:
    """Isolated-margin liquidation price (Binance form, margin = notional / leverage, no fees):
    long : entry * (1 - 1/L) / (1 - mmr)         short: entry * (1 + 1/L) / (1 + mmr)"""
    if side > 0:
        return entry * (1.0 - 1.0 / leverage) / (1.0 - mmr)
    return entry * (1.0 + 1.0 / leverage) / (1.0 + mmr)


def size_and_levels(open_px: float, fill: float, side: int, atr: float, cfg: SimConfig,
                    scale: float = 1.0):
    """-> (qty, tp_price, sl_price) or None when the template cannot size the trade (NaN ATR ...).

    Barrier levels are measured from the actual (slipped) entry `fill`, exactly like tb.labels, so
    entry slippage does not change the payoffs; `open_px` is kept for the caller's reference.  `scale` (<= 1) is the
    risk multiplier from drawdown derating / drift: it scales the risk budget (sl_usd) and the slice
    notional cap, never the price levels."""
    cap_notional = cfg.notional * scale
    if cfg.exit_mode == "fixed":
        # tp_usd / sl_usd are defined on the full slice notional -> constant price distances
        tp_d = cfg.tp_usd / cfg.notional * fill
        sl_d = cfg.sl_usd / cfg.notional * fill
        qty = cap_notional / fill
    elif cfg.exit_mode == "atr":
        if not (atr > 0 and math.isfinite(atr)):
            return None
        tp_d, sl_d = cfg.tp_atr * atr, cfg.sl_atr * atr
        qty = min(cfg.sl_usd * scale / sl_d, cap_notional / fill)
    else:
        raise ValueError(f"unknown exit_mode {cfg.exit_mode!r}")
    sl = fill - side * sl_d
    if sl <= 0:
        return None
    return qty, fill + side * tp_d, sl


def breakeven_winrate(cfg: SimConfig, atr_frac: float | None = None, funding_per_trade_usd: float = 0.0
                      ) -> dict:
    """Win probability at which a trade's expected PnL is zero for the given exit config, fees and
    slippage, assuming wins exit at TP and losses at SL (no timeouts, no trailing).  Same algebra as
    tb.labels.breakeven_from_cfg.

    atr_frac = ATR/price, needed for exit_mode='atr' (sizes the notional, which is capped by the
    slice).  Returns net win/loss USD, the break-even rate and the zero-cost rate sl/(tp+sl)
    (= the win rate of a random entry in a driftless market).
    """
    fee, slip, extra = cfg.fee_taker, cfg.slippage, cfg.stop_slippage_extra
    cap = cfg.notional
    if cfg.exit_mode == "fixed":
        notional = cap
        gross_w, gross_l = cfg.tp_usd, cfg.sl_usd
    else:
        if not atr_frac:
            raise ValueError("atr mode needs atr_frac (ATR / price)")
        notional = min(cfg.sl_usd / (cfg.sl_atr * atr_frac), cap)
        gross_w, gross_l = notional * cfg.tp_atr * atr_frac, notional * cfg.sl_atr * atr_frac
    # levels are measured from the slipped fill, so entry slippage does not change the payoffs
    win = gross_w - 2 * fee * notional - funding_per_trade_usd
    loss = gross_l + 2 * fee * notional + (slip + extra) * notional + funding_per_trade_usd
    return {"win_usd": win, "loss_usd": loss, "breakeven_winrate": loss / (win + loss),
            "zero_cost_winrate": gross_l / (gross_w + gross_l), "notional": notional,
            "round_trip_fees_usd": 2 * fee * notional}
