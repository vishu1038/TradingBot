"""Reinforcement-learning trading environment (Phase 3).

A single-asset, single-position trading MDP built on the same feature engineering and cost
model as the rest of the bot, so an RL policy is directly comparable to the rule-based and
supervised-ML strategies.

Gymnasium is used if installed (then `TradingEnv` is a true `gym.Env`); otherwise a
dependency-light standalone class with the same `reset`/`step` surface and minimal
`observation_space`/`action_space` shims is provided. This matters because gymnasium and
stable-baselines3 are not generally installable on Python 3.14 — the default training path
must work with numpy alone.

MDP definition
--------------
The agent decides, at each bar `t`, the target position to *carry into* bar `t+1`. This
mirrors the look-ahead-free convention of the other strategies (decide at `t`, the PnL of
that decision is realized over the next bar).

State / observation (a flat float32 vector), in this exact order:
    1. `window` rows x `len(FEATURE_COLUMNS)` of z-score-normalized indicator features,
       flattened ROW-MAJOR oldest-first: [row_{t-window+1}_feat0, ..., row_t_featN].
    2. current position, scalar in {-1.0, 0.0, +1.0}.
    3. current unrealized PnL fraction of the open position (0.0 when flat).
    Total length = window * len(FEATURE_COLUMNS) + 2.

    Normalization: each feature column is z-scored using the mean/std of the WHOLE provided
    DataFrame (after add_indicators().dropna()). Std is floored to 1e-8 to avoid divide-by-
    zero, and any residual NaN/inf is mapped to 0.0. The same normalization stats are reused
    at inference time by `RLStrategy` (it calls `build_observation` from this module), which
    guarantees train/inference parity.

Action: Discrete(3), interpreted as a target position:
    0 -> short (-1)
    1 -> flat  ( 0)
    2 -> long  (+1)

Reward (selectable via `reward_mode`):
    "pnl" (default):
        mark-to-market PnL of the position carried over the next bar, minus transaction
        cost (fee+slippage) charged on the change in notional when the position changes:
            reward = pos_new * next_bar_return - cost_per_turn * |pos_new - pos_old|
        Equivalent to the per-bar strategy return the Backtester computes.
    "sharpe":
        differential / risk-adjusted increment. Maintains EWMA estimates of the mean and
        variance of the per-step pnl and returns the marginal contribution to the Sharpe
        ratio (Moody & Saffell differential Sharpe). Rewards smooth equity growth and
        penalizes volatility.
    "pnl_minus_drawdown":
        the "pnl" reward minus a penalty proportional to any NEW drawdown opened on this
        step (how much further below the running equity peak we fell). Discourages deep
        underwater excursions.

Episode termination:
    * `terminated=True` at the end of the data.
    * `truncated=True` if equity falls below `(1 - max_drawdown_blowout)` of its peak (a
      risk blowout), if `max_drawdown_blowout` is set.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from strategies.features import add_indicators, FEATURE_COLUMNS

logger = logging.getLogger()

try:  # Real gymnasium if available, else a dependency-light fallback.
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:  # pragma: no cover - Python 3.14 fallback path
    _HAS_GYM = False

# Action index -> target position.
ACTION_TO_POSITION = {0: -1, 1: 0, 2: 1}
POSITION_TO_ACTION = {-1: 0, 0: 1, 1: 2}
N_ACTIONS = 3


# --------------------------------------------------------------------------- #
# Minimal space shims (only used when gymnasium is unavailable).
# --------------------------------------------------------------------------- #
class _DiscreteShim:
    """Stand-in for gymnasium.spaces.Discrete with the same `.n` and `.sample()`."""

    def __init__(self, n: int):
        self.n = int(n)
        self._rng = np.random.default_rng()

    def sample(self) -> int:
        return int(self._rng.integers(self.n))

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class _BoxShim:
    """Stand-in for gymnasium.spaces.Box exposing `.shape` and `.dtype`."""

    def __init__(self, shape: Tuple[int, ...], dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.low = -np.inf
        self.high = np.inf

    def __repr__(self) -> str:
        return f"Box(shape={self.shape})"


# --------------------------------------------------------------------------- #
# Shared feature/observation utilities — imported by BOTH the env and the
# RLStrategy so training and inference build identical inputs.
# --------------------------------------------------------------------------- #
def compute_normalization(features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) per FEATURE_COLUMNS over `features` (already indicator-augmented,
    NaNs dropped). Std is floored to 1e-8 for numerical safety."""
    arr = features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean.astype(np.float32), std.astype(np.float32)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Run add_indicators and drop look-back NaNs, returning a clean, index-reset frame."""
    feats = add_indicators(df).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return feats


def normalized_feature_matrix(features: pd.DataFrame, mean: np.ndarray,
                              std: np.ndarray) -> np.ndarray:
    """Z-score the FEATURE_COLUMNS matrix and scrub NaN/inf -> 0.0 (float32)."""
    arr = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    norm = (arr - mean) / std
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_observation(norm_matrix: np.ndarray, t: int, window: int,
                      position: float, unrealized_pnl: float) -> np.ndarray:
    """Assemble the flat observation vector for the bar at integer index `t`.

    `norm_matrix` is the full (n_bars, n_features) normalized matrix from
    `normalized_feature_matrix`. The window is rows (t-window+1 .. t) inclusive, oldest
    first. If `t < window-1` the window is left-padded with zeros so the layout/length is
    constant. Appends [position, unrealized_pnl]. See module docstring for the exact layout.
    """
    n_features = norm_matrix.shape[1]
    start = t - window + 1
    if start < 0:
        pad = np.zeros((-start, n_features), dtype=np.float32)
        win = np.vstack([pad, norm_matrix[0:t + 1]])
    else:
        win = norm_matrix[start:t + 1]
    flat = win.reshape(-1)
    tail = np.array([position, unrealized_pnl], dtype=np.float32)
    return np.concatenate([flat, tail]).astype(np.float32)


def observation_dim(window: int) -> int:
    """Length of the observation vector for a given window."""
    return window * len(FEATURE_COLUMNS) + 2


_EnvBase = gym.Env if _HAS_GYM else object


class TradingEnv(_EnvBase):
    """Single-asset trading environment. Gymnasium-compatible when available."""

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, window: int = 32, fee: float = 0.0004,
                 slippage: float = 0.0002, initial_capital: float = 10_000.0,
                 reward_mode: str = "pnl", max_drawdown_blowout: Optional[float] = None,
                 churn_penalty: float = 0.0):
        if _HAS_GYM:
            super().__init__()
        if reward_mode not in ("pnl", "sharpe", "pnl_minus_drawdown"):
            raise ValueError(f"unknown reward_mode: {reward_mode!r}")

        self.window = int(window)
        self.cost_per_turn = float(fee) + float(slippage)
        self.initial_capital = float(initial_capital)
        self.reward_mode = reward_mode
        self.max_drawdown_blowout = max_drawdown_blowout
        # Extra per-unit-turnover penalty applied ONLY to the training reward (not to equity),
        # to bias the learned policy away from churn beyond what the literal fee discourages.
        # The equity/PnL still uses the true exchange cost, so backtest equity stays honest.
        self.churn_penalty = float(churn_penalty)

        # Precompute features, normalization stats, and the per-bar close returns.
        self.features = prepare_frame(df)
        if len(self.features) < self.window + 2:
            raise ValueError(
                f"need at least window+2={self.window + 2} clean bars, got {len(self.features)}")
        self.mean, self.std = compute_normalization(self.features)
        self.norm_matrix = normalized_feature_matrix(self.features, self.mean, self.std)
        self.close = self.features["close"].to_numpy(dtype=np.float64)
        # Forward one-bar simple return at index t = (close[t+1]/close[t] - 1).
        self.fwd_ret = np.zeros(len(self.close), dtype=np.float64)
        self.fwd_ret[:-1] = self.close[1:] / self.close[:-1] - 1.0

        self.n_bars = len(self.features)
        # Episode runs decisions on bars [window-1 .. n_bars-2] (need a next bar for PnL).
        self.start_t = self.window - 1
        self.end_t = self.n_bars - 2

        obs_dim = observation_dim(self.window)
        if _HAS_GYM:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                 shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(N_ACTIONS)
        else:
            self.observation_space = _BoxShim((obs_dim,), np.float32)
            self.action_space = _DiscreteShim(N_ACTIONS)

        self._reset_state()

    # --- internal state ---------------------------------------------------- #
    def _reset_state(self) -> None:
        self.t = self.start_t
        self.position = 0.0          # current signed position carried into bar t
        self.entry_price = np.nan    # price at which the open position was entered
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        # Differential-Sharpe running moments (Moody & Saffell).
        self._a = 0.0                # EWMA of return
        self._b = 0.0                # EWMA of return^2
        self._eta = 0.01             # adaptation rate

    def _unrealized_pnl(self) -> float:
        if self.position == 0.0 or np.isnan(self.entry_price) or self.entry_price == 0.0:
            return 0.0
        price = self.close[self.t]
        return float(self.position * (price / self.entry_price - 1.0))

    def _obs(self) -> np.ndarray:
        return build_observation(self.norm_matrix, self.t, self.window,
                                 self.position, self._unrealized_pnl())

    # --- Gym API ----------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if _HAS_GYM:
            super().reset(seed=seed)
        elif seed is not None:
            np.random.seed(seed)
        self._reset_state()
        info = {"equity": self.equity, "position": self.position, "step": self.t}
        return self._obs(), info

    def step(self, action: int):
        """Apply `action` (target position) at bar t; PnL accrues over bar t->t+1."""
        target = float(ACTION_TO_POSITION[int(action)])
        prev_position = self.position
        turnover = abs(target - prev_position)
        cost = self.cost_per_turn * turnover

        # Update the open-position bookkeeping on a change.
        if target != prev_position:
            if target == 0.0:
                self.entry_price = np.nan
            else:
                self.entry_price = self.close[self.t]
        self.position = target

        next_ret = self.fwd_ret[self.t]
        pnl = target * next_ret - cost          # per-bar strategy return (matches Backtester)

        prev_equity = self.equity
        self.equity = self.equity * (1.0 + pnl)
        new_peak = max(self.peak_equity, self.equity)
        # New drawdown opened this step (fraction of peak), >= 0.
        new_dd = max(0.0, (self.peak_equity - self.equity) / self.peak_equity
                     - max(0.0, (self.peak_equity - prev_equity) / self.peak_equity))
        self.peak_equity = new_peak

        reward = self._shape_reward(pnl, new_dd)
        # Training-only churn regularizer: discourage turnover beyond the literal fee. Does
        # NOT touch equity (above), so the simulated PnL/equity curve stays realistic.
        if self.churn_penalty:
            reward -= self.churn_penalty * turnover

        self.t += 1
        terminated = self.t > self.end_t
        truncated = False
        if (self.max_drawdown_blowout is not None
                and self.equity < self.peak_equity * (1.0 - self.max_drawdown_blowout)):
            truncated = True

        info = {"equity": self.equity, "position": self.position,
                "step": self.t, "pnl": pnl, "cost": cost}
        # When the episode is over there is no further bar; clamp obs to last valid index.
        obs = self._obs() if not (terminated or truncated) else self._obs_clamped()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _obs_clamped(self) -> np.ndarray:
        t = min(self.t, self.n_bars - 1)
        return build_observation(self.norm_matrix, t, self.window,
                                 self.position, self._unrealized_pnl())

    def _shape_reward(self, pnl: float, new_dd: float) -> float:
        if self.reward_mode == "pnl":
            return pnl
        if self.reward_mode == "pnl_minus_drawdown":
            return pnl - new_dd
        # "sharpe": differential Sharpe increment D_t.
        a_prev, b_prev = self._a, self._b
        delta_a = pnl - a_prev
        delta_b = pnl * pnl - b_prev
        denom = (b_prev - a_prev * a_prev) ** 1.5
        if denom <= 1e-12:
            d_t = 0.0
        else:
            d_t = (b_prev * delta_a - 0.5 * a_prev * delta_b) / denom
        self._a = a_prev + self._eta * delta_a
        self._b = b_prev + self._eta * delta_b
        return float(d_t)
