"""RL-policy strategy — wraps a trained reinforcement-learning policy as a Strategy.

Loads the params saved by `learning/train.py` (a numpy linear-softmax policy plus the
normalization stats and window used during training) and replays the policy bar-by-bar to
emit target positions in {-1, 0, +1}.

Train/inference parity is guaranteed by reusing the SAME observation builder the environment
uses (`learning.env.build_observation` and friends): features are computed with
`add_indicators`, z-scored with the stored mean/std, and assembled with the stored window.

Look-ahead discipline: the policy decides at bar `t` using only data up to `t`, and we SHIFT
the resulting target by one bar so it executes at `t+1` — matching EmaCrossStrategy and
MLStrategy. The Backtester therefore applies no further shift double-counting.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import Strategy
from strategies.features import FEATURE_COLUMNS
from learning.env import (ACTION_TO_POSITION, prepare_frame,
                          normalized_feature_matrix, build_observation)

logger = logging.getLogger()


class RLStrategy(Strategy):
    name = "rl_policy"

    def __init__(self, policy: Optional[dict] = None, path: Optional[str] = None):
        """Provide either `policy` (a dict with keys W, b, mean, std, window) or a `path`
        to an .npz saved by learning/train.py. With neither, generate_signals raises."""
        self.W = None
        self.b = None
        self.mean = None
        self.std = None
        self.window = None
        if policy is not None:
            self._set_policy(policy)
        elif path is not None:
            self.load(path)

    def _set_policy(self, policy: dict) -> None:
        self.W = np.asarray(policy["W"], dtype=np.float32)
        self.b = np.asarray(policy["b"], dtype=np.float32)
        self.mean = np.asarray(policy["mean"], dtype=np.float32)
        self.std = np.asarray(policy["std"], dtype=np.float32)
        self.window = int(np.asarray(policy["window"]).item())

    def load(self, path: str) -> "RLStrategy":
        """Load policy params + normalization stats from an .npz file."""
        with np.load(path, allow_pickle=False) as blob:
            self._set_policy({k: blob[k] for k in ("W", "b", "mean", "std", "window")})
        logger.info("RLStrategy loaded from %s (window=%d, obs_dim=%d)",
                    path, self.window, self.W.shape[0])
        return self

    def _act(self, obs: np.ndarray) -> int:
        return int(np.argmax(obs @ self.W + self.b))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self.W is None:
            raise RuntimeError(
                "RLStrategy has no policy loaded. Pass policy=... or path=... (e.g. "
                "models/rl_policy.npz produced by learning/train.py).")

        signal = pd.Series(0, index=df.index, dtype=int)

        # Rebuild observations exactly as TradingEnv does.
        feats = prepare_frame(df)
        if len(feats) < self.window:
            return signal  # not enough clean bars to form a single window
        norm_matrix = normalized_feature_matrix(feats, self.mean, self.std)

        # Map cleaned (NaN-dropped) integer indices back to the original df index so the
        # signal aligns with df. prepare_frame resets index, so we recover the original
        # labels via the same add_indicators().dropna() ordering.
        valid_index = df.index[-len(feats):]

        position = 0.0
        entry_price = np.nan
        close = feats["close"].to_numpy(dtype=np.float64)

        decisions = np.zeros(len(feats), dtype=int)
        for t in range(self.window - 1, len(feats)):
            if position == 0.0 or np.isnan(entry_price) or entry_price == 0.0:
                unreal = 0.0
            else:
                unreal = float(position * (close[t] / entry_price - 1.0))
            obs = build_observation(norm_matrix, t, self.window, position, unreal)
            action = self._act(obs)
            target = float(ACTION_TO_POSITION[action])
            if target != position:
                entry_price = close[t] if target != 0.0 else np.nan
                position = target
            decisions[t] = int(target)

        signal.loc[valid_index] = decisions

        # Decide at t, execute at t+1 (no look-ahead) — same convention as the others.
        return signal.shift(1).fillna(0).astype(int)
