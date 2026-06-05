"""Train an RL trading policy — runs out-of-the-box with numpy only.

Default agent
-------------
A tiny numpy **linear-softmax policy** optimized by the **cross-entropy method (CEM)** over
whole-episode return. The policy is a single weight matrix W of shape
(obs_dim, n_actions) plus a bias; the action is `argmax`/`softmax(W^T obs + b)`. CEM keeps a
Gaussian over the flattened parameter vector, samples a population each iteration, evaluates
the mean episode reward of each sample on the env, and refits the Gaussian to the top-k
"elite" samples. This is gradient-free, dependency-free (numpy only), tiny, and reliably
learns a non-trivial policy on this env. No torch / stable-baselines3 required.

Optional SB3/PPO path
---------------------
If both `gymnasium` and `stable_baselines3` import, `--sb3` trains PPO instead and saves a
`.zip`. It is fully optional and never affects the default path. See learning/README.md.

CLI
---
    python learning/train.py                 # synthetic data, CEM agent
    python learning/train.py BTCUSDT 1h      # real cached data if keys/config available
    python learning/train.py --sb3           # PPO via stable-baselines3 (if installable)

Output: learning progress, models/rl_policy.npz (the learned params + normalization stats),
and a Backtester evaluation summary directly comparable to the other strategies.
"""

import logging
import os
import sys

# Allow `python learning/train.py` from the repo root by putting the root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from backtesting.engine import Backtester
from backtesting.metrics import print_summary
from learning.env import (TradingEnv, ACTION_TO_POSITION, N_ACTIONS,
                          observation_dim, prepare_frame)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
logger = logging.getLogger()

MODEL_PATH = os.path.join("models", "rl_policy.npz")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def synthetic_ohlcv(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Geometric random walk with mild momentum. Mirrors run_backtest's generator idea but
    implemented locally (no import). NOT a market model — used only to exercise the pipeline."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 0.01, n)
    drift = 0.3 * pd.Series(shocks).rolling(5).mean().fillna(0).values
    log_ret = shocks + drift
    close = 30_000 * np.exp(np.cumsum(log_ret))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(10, 100, n)
    ts = (1_600_000_000_000 + np.arange(n) * 3_600_000).astype("int64")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": volume})


def load_real(symbol: str, timeframe: str, limit: int = 3000) -> pd.DataFrame:
    """Best-effort real-data load via the data manager. Caller wraps this in try/except."""
    from config import CONFIG
    from connectors.binance_futures import BinanceFuturesClient
    from data.database import Database
    from data.data_manager import DataManager

    client = BinanceFuturesClient(CONFIG.binance_public_key, CONFIG.binance_secret_key,
                                  CONFIG.use_testnet)
    dm = DataManager(Database(CONFIG.database_path))
    return dm.get_candles(client, symbol, timeframe, limit=limit)


# --------------------------------------------------------------------------- #
# Linear-softmax policy + CEM agent (numpy only)
# --------------------------------------------------------------------------- #
class LinearPolicy:
    """Linear-softmax policy: logits = obs @ W + b, action = argmax(logits) (greedy)."""

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.W = np.zeros((obs_dim, n_actions), dtype=np.float32)
        self.b = np.zeros(n_actions, dtype=np.float32)

    @property
    def n_params(self) -> int:
        return self.obs_dim * self.n_actions + self.n_actions

    def set_flat(self, theta: np.ndarray) -> None:
        w_size = self.obs_dim * self.n_actions
        self.W = theta[:w_size].reshape(self.obs_dim, self.n_actions).astype(np.float32)
        self.b = theta[w_size:].astype(np.float32)

    def get_flat(self) -> np.ndarray:
        return np.concatenate([self.W.reshape(-1), self.b]).astype(np.float32)

    def act(self, obs: np.ndarray) -> int:
        logits = obs @ self.W + self.b
        return int(np.argmax(logits))


def run_episode(env: TradingEnv, policy: LinearPolicy) -> float:
    """Greedy roll-out of `policy` on `env`; returns total reward."""
    obs, _ = env.reset()
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
    return total


def train_cem(env: TradingEnv, obs_dim: int, iterations: int = 25,
              population: int = 40, elite_frac: float = 0.25, init_std: float = 0.5,
              seed: int = 0) -> LinearPolicy:
    """Cross-entropy method over the flat policy parameters. Returns the best policy."""
    rng = np.random.default_rng(seed)
    policy = LinearPolicy(obs_dim)
    n_params = policy.n_params
    mean = np.zeros(n_params, dtype=np.float64)
    std = np.full(n_params, init_std, dtype=np.float64)
    n_elite = max(2, int(population * elite_frac))

    best_theta = mean.copy()
    best_reward = -np.inf

    logger.info("CEM: %d params, %d iters, pop=%d, elite=%d",
                n_params, iterations, population, n_elite)
    for it in range(iterations):
        samples = rng.normal(mean, std, size=(population, n_params))
        rewards = np.empty(population, dtype=np.float64)
        for i in range(population):
            policy.set_flat(samples[i])
            rewards[i] = run_episode(env, policy)

        elite_idx = np.argsort(rewards)[-n_elite:]
        elite = samples[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3       # noise floor keeps exploration alive

        if rewards[elite_idx[-1]] > best_reward:
            best_reward = float(rewards[elite_idx[-1]])
            best_theta = samples[elite_idx[-1]].copy()

        logger.info("iter %2d/%d | avg=%+.5f | elite_avg=%+.5f | best=%+.5f",
                    it + 1, iterations, rewards.mean(), rewards[elite_idx].mean(), best_reward)

    policy.set_flat(best_theta)
    return policy


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_policy(path: str, policy: LinearPolicy, env: TradingEnv) -> None:
    """Save params + the exact normalization stats / window so inference matches training."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, W=policy.W, b=policy.b,
             mean=env.mean, std=env.std,
             window=np.int64(env.window), obs_dim=np.int64(observation_dim(env.window)))
    logger.info("Saved RL policy -> %s", path)


# --------------------------------------------------------------------------- #
# Evaluation via the shared Backtester (comparable to other strategies)
# --------------------------------------------------------------------------- #
def evaluate(df: pd.DataFrame, timeframe: str = "1h") -> None:
    from strategies.rl_strategy import RLStrategy
    strat = RLStrategy(path=MODEL_PATH)
    bt = Backtester(fee=0.0004, slippage=0.0002, initial_capital=10_000, risk_fraction=0.5)
    result = bt.run(df, strat.generate_signals(df))
    print_summary(result, timeframe, title="RL policy (CEM linear, full series)")


# --------------------------------------------------------------------------- #
# Optional SB3/PPO path
# --------------------------------------------------------------------------- #
def train_sb3(df: pd.DataFrame, timesteps: int = 20_000):  # pragma: no cover - optional
    try:
        import gymnasium  # noqa: F401
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("--sb3 requested but gymnasium/stable_baselines3 not importable. "
                     "On Python 3.14 these are usually unavailable; use the default path.")
        return
    logger.info("Training PPO via stable-baselines3 for %d timesteps", timesteps)
    env = TradingEnv(df, reward_mode="pnl")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    os.makedirs("models", exist_ok=True)
    model.save(os.path.join("models", "rl_ppo"))
    logger.info("Saved PPO model -> models/rl_ppo.zip (load via PPO.load in your own runner)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_sb3 = "--sb3" in sys.argv

    symbol = args[0] if len(args) > 0 else None
    timeframe = args[1] if len(args) > 1 else "1h"

    df = None
    if symbol:
        try:
            logger.info("Attempting to load real data for %s %s", symbol, timeframe)
            df = load_real(symbol, timeframe)
            if df is None or len(df) < 500:
                raise ValueError(f"insufficient real candles ({0 if df is None else len(df)})")
        except Exception as exc:  # noqa: BLE001 - graceful fallback by design
            logger.warning("Real-data load failed (%s); falling back to synthetic.", exc)
            df = None
    if df is None:
        logger.info("Using synthetic random-walk data (pipeline demo only).")
        df, timeframe = synthetic_ohlcv(), "1h"

    if use_sb3:
        train_sb3(df)
        # Note: SB3 saves its own .zip; the default npz evaluation below is skipped.
        return

    env = TradingEnv(df, window=32, reward_mode="pnl")
    obs_dim = observation_dim(env.window)
    logger.info("Env ready: %d clean bars, obs_dim=%d, actions=%d",
                env.n_bars, obs_dim, N_ACTIONS)

    policy = train_cem(env, obs_dim, iterations=20, population=40)
    save_policy(MODEL_PATH, policy, env)

    evaluate(df, timeframe)

    print("\nNote: on synthetic random-walk data, near-zero/negative net return after costs "
          "is EXPECTED and correct — there is no real edge to learn. A positive in-sample "
          "result here is NOT evidence of edge; validate any RL policy on REAL data with a "
          "chronological train/test split (see backtesting walk-forward).")


if __name__ == "__main__":
    main()
