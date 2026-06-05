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
class MLPPolicy:
    """Feed-forward policy: tanh-activated hidden layers feeding 3 action logits, greedy
    argmax. `hidden=()` collapses to a single linear layer — i.e. exactly the original
    linear-softmax policy — so the old behaviour is a strict special case. A hidden layer lets
    the policy learn NON-LINEAR interactions between indicators (e.g. "go long only when RSI is
    low AND the MACD histogram is turning up AND volatility is rising"), which a single linear
    layer literally cannot represent. Still numpy-only and CEM-trainable (no autodiff)."""

    def __init__(self, obs_dim: int, hidden=(), n_actions: int = N_ACTIONS):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = tuple(int(h) for h in hidden)
        sizes = (self.obs_dim,) + self.hidden + (self.n_actions,)
        # Each layer is a mutable [W, b] pair so set_flat can rebind in place.
        self.layers = [[np.zeros((nin, nout), dtype=np.float32),
                        np.zeros(nout, dtype=np.float32)]
                       for nin, nout in zip(sizes[:-1], sizes[1:])]

    @property
    def n_params(self) -> int:
        return int(sum(W.size + b.size for W, b in self.layers))

    def set_flat(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float32).reshape(-1)
        off = 0
        for layer in self.layers:
            W, b = layer
            wsz = W.size
            layer[0] = theta[off:off + wsz].reshape(W.shape); off += wsz
            bsz = b.size
            layer[1] = theta[off:off + bsz].copy(); off += bsz

    def get_flat(self) -> np.ndarray:
        parts = []
        for W, b in self.layers:
            parts.append(W.reshape(-1)); parts.append(b)
        return np.concatenate(parts).astype(np.float32)

    def _logits(self, obs: np.ndarray) -> np.ndarray:
        h = np.asarray(obs, dtype=np.float32)
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            z = h @ W + b
            h = z if i == last else np.tanh(z)   # tanh on hidden, raw logits on output
        return h

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmax(self._logits(obs)))

    def as_dict(self) -> dict:
        """In-memory representation used to hot-swap an RLStrategy without touching disk."""
        return {"layers": [(W, b) for W, b in self.layers], "hidden": self.hidden}


class LinearPolicy(MLPPolicy):
    """Backward-compatible alias: the original linear-softmax policy (no hidden layer).
    Exposes `.W`/`.b` so existing callers and the legacy save format keep working."""

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS):
        super().__init__(obs_dim, hidden=(), n_actions=n_actions)

    @property
    def W(self) -> np.ndarray:
        return self.layers[0][0]

    @property
    def b(self) -> np.ndarray:
        return self.layers[0][1]


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
              seed: int = 0, progress_cb=None, init_theta=None, hidden=()) -> MLPPolicy:
    """Cross-entropy method over the flat policy parameters. Returns the best policy.

    If `progress_cb` is given, it is called once per iteration with a dict of progress
    metrics (iteration, total, avg/elite/best reward). This lets a live dashboard show
    training advancing in real time without coupling the optimizer to any UI.

    `init_theta` (optional flat parameter vector) WARM-STARTS the search: the CEM mean is
    seeded with it instead of zeros, so continuous retraining refines the previous policy
    rather than restarting from scratch each cycle. The best policy returned carries its
    achieved episode reward on `policy.best_reward`.
    """
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(obs_dim, hidden=hidden)
    n_params = policy.n_params
    if init_theta is not None and len(np.asarray(init_theta).reshape(-1)) == n_params:
        mean = np.asarray(init_theta, dtype=np.float64).reshape(-1).copy()
    else:
        mean = np.zeros(n_params, dtype=np.float64)
    std = np.full(n_params, init_std, dtype=np.float64)
    n_elite = max(2, int(population * elite_frac))

    best_theta = mean.copy()
    best_reward = -np.inf

    logger.info("CEM: %d params (hidden=%s), %d iters, pop=%d, elite=%d",
                n_params, hidden or "linear", iterations, population, n_elite)
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

        if progress_cb is not None:
            try:
                progress_cb({
                    "iteration": it + 1,
                    "total": iterations,
                    "avg_reward": float(rewards.mean()),
                    "elite_reward": float(rewards[elite_idx].mean()),
                    "best_reward": best_reward,
                })
            except Exception as e:  # a UI hook must never kill training
                logger.error("train progress_cb raised: %s", e)

    policy.set_flat(best_theta)
    policy.best_reward = best_reward
    return policy


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_policy(path: str, policy: MLPPolicy, env: TradingEnv) -> None:
    """Save params + the exact normalization stats / window so inference matches training.

    Layered format (supports linear AND multi-layer MLP): a `kind`/`n_layers` header plus
    `W{i}`/`b{i}` per layer. For a single-layer (linear) policy we ALSO write legacy `W`/`b`
    keys so older readers keep working."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob = {
        "kind": "mlp",
        "n_layers": np.int64(len(policy.layers)),
        "hidden": np.asarray(policy.hidden, dtype=np.int64),
        "mean": env.mean, "std": env.std,
        "window": np.int64(env.window),
        "obs_dim": np.int64(observation_dim(env.window)),
    }
    for i, (W, b) in enumerate(policy.layers):
        blob[f"W{i}"] = W
        blob[f"b{i}"] = b
    if len(policy.layers) == 1:           # legacy compatibility for linear policies
        blob["W"] = policy.layers[0][0]
        blob["b"] = policy.layers[0][1]
    np.savez(path, **blob)
    logger.info("Saved RL policy -> %s (layers=%d, hidden=%s)",
                path, len(policy.layers), policy.hidden or "linear")


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
# Dashboard entry point — train while streaming progress to an event bus
# --------------------------------------------------------------------------- #
DEFAULT_HIDDEN = (24,)   # one tanh hidden layer of 24 units — the default "less basic" policy


def train_policy(df: pd.DataFrame, *, iterations: int = 20, population: int = 40,
                 window: int = 32, seed: int = 0, init_theta=None,
                 model_path: str = MODEL_PATH, init_std: float = 0.5, hidden=DEFAULT_HIDDEN,
                 churn_penalty: float = 0.0, progress_cb=None) -> dict:
    """Pure CEM training: build the env, train, save the policy, return a plain dict.

    No event bus — every value in/out is picklable, so this runs unchanged either inline on a
    thread (the manual Train button) or inside a separate worker PROCESS (the continuous
    multi-core trainer). `init_theta` warm-starts CEM (incremental learning); `model_path`
    lets each symbol persist to its own file; `hidden` sets the MLP architecture (default one
    24-unit tanh layer; `()` = the old linear policy). `progress_cb` is forwarded to CEM (only
    used in the in-process path — it cannot cross a process boundary).
    """
    env = TradingEnv(df, window=window, reward_mode="pnl", churn_penalty=churn_penalty)
    obs_dim = observation_dim(env.window)
    # A warm-start vector only fits if its length matches THIS architecture; otherwise (e.g.
    # the architecture changed since the last cycle) start fresh rather than crash.
    policy = train_cem(env, obs_dim, iterations=iterations, population=population, seed=seed,
                       init_theta=init_theta, init_std=init_std, hidden=hidden,
                       progress_cb=progress_cb)
    save_policy(model_path, policy, env)
    best = float(getattr(policy, "best_reward", 0.0))
    pdict = policy.as_dict()
    pdict.update({"mean": env.mean, "std": env.std, "window": env.window})
    return {
        "ok": True,
        "model_path": model_path,
        "iterations": iterations,
        "bars": env.n_bars,
        "best_reward": best,
        "theta": policy.get_flat(),
        "policy": pdict,
    }


def _pool_train_worker(payload):
    """Top-level, picklable entry point for ProcessPoolExecutor — trains ONE symbol in a
    worker process. `payload` is (symbol, df, kwargs_dict). Never raises across the pool
    boundary: failures come back as {"ok": False, ...} so one bad symbol can't kill the loop.
    """
    symbol, df, kw = payload
    try:
        res = train_policy(df, **kw)
        res["symbol"] = symbol
        return res
    except Exception as e:  # noqa: BLE001 - must not propagate across the process boundary
        return {"ok": False, "symbol": symbol, "error": str(e)}


def train_to_bus(event_bus, df: pd.DataFrame = None, iterations: int = 20,
                 population: int = 40, window: int = 32, seed: int = 0,
                 init_theta=None, model_path: str = MODEL_PATH,
                 symbol: str = None, init_std: float = 0.5, hidden=DEFAULT_HIDDEN) -> dict:
    """Run CEM training inline, publishing `train` events to `event_bus`, and save the policy.

    Used by the dashboard's /api/train button (the in-process, progress-streaming path).
    Publishes: train {phase: "start"|"iter"|"done"|"error", symbol?, ...}. Delegates the
    actual optimization to `train_policy` so the inline and multi-core paths stay identical.
    """
    def publish(phase, **kw):
        if event_bus is not None:
            try:
                payload = {"phase": phase}
                if symbol is not None:
                    payload["symbol"] = symbol
                payload.update(kw)
                event_bus.publish("train", payload)
            except Exception:
                pass

    try:
        if df is None:
            df = synthetic_ohlcv()
        publish("start", iterations=iterations, bars=len(df), obs_dim=observation_dim(window))
        logger.info("Training%s: %d bars, %d iters%s",
                    f" {symbol}" if symbol else "", len(df), iterations,
                    " (warm start)" if init_theta is not None else "")

        res = train_policy(df, iterations=iterations, population=population, window=window,
                           seed=seed, init_theta=init_theta, model_path=model_path,
                           init_std=init_std, hidden=hidden,
                           progress_cb=lambda p: publish("iter", **p))
        best = res["best_reward"]
        publish("done", model_path=model_path, best_reward=best)
        logger.info("Training complete%s -> %s (best=%+.5f)",
                    f" {symbol}" if symbol else "", model_path, best)
        if symbol is not None:
            res["symbol"] = symbol
        return res
    except Exception as e:
        logger.error("Dashboard training failed: %s", e, exc_info=True)
        publish("error", error=str(e))
        return {"ok": False, "error": str(e)}


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
