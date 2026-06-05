# Reinforcement Learning (Phase 3)

This package adds a reinforcement-learning trading agent that plugs into the same pipeline
(features -> environment -> policy -> backtester -> metrics) as the rule-based and
supervised-ML strategies, so an RL policy is directly comparable to them.

Everything here runs with **numpy only**. `gymnasium` and `stable-baselines3` are used *if
present* but are not required — important because they are generally not installable on
Python 3.14.

## Files
- `learning/env.py` — `TradingEnv`, the trading MDP (Gymnasium `gym.Env` if available, else a
  dependency-light standalone class with the same `reset`/`step` API and space shims). Also
  hosts the shared observation builder used by both training and inference.
- `learning/train.py` — training entry point. Default agent is a numpy linear-softmax policy
  optimized by the cross-entropy method (CEM). Optional `--sb3` PPO path.
- `strategies/rl_strategy.py` — `RLStrategy(Strategy)`: wraps a trained policy and emits
  `{-1, 0, +1}` target positions via the standard `generate_signals(df)` contract.

## The MDP

### State / observation
A flat `float32` vector, length `window * len(FEATURE_COLUMNS) + 2`, laid out as:

1. The last `window` rows of the normalized `FEATURE_COLUMNS` matrix, flattened **row-major,
   oldest-first**: `[row_{t-window+1} feat0..featN, ..., row_t feat0..featN]`.
   - Features come from `strategies.features.add_indicators` (after `.dropna()`).
   - Each feature column is **z-scored** using the mean/std of the whole training frame; std
     is floored to `1e-8`; residual NaN/inf is mapped to `0.0`.
   - Early bars (`t < window-1`) are left-padded with zero rows so the length is constant.
2. **Current position**: scalar in `{-1.0, 0.0, +1.0}`.
3. **Current unrealized PnL fraction** of the open position (`0.0` when flat).

The normalization stats (`mean`, `std`) and `window` are saved with the policy so
`RLStrategy` reconstructs identical observations at inference time (train/inference parity).

### Action
`Discrete(3)`, interpreted as a **target position**:

| action | position |
|-------:|:---------|
| 0      | short (-1) |
| 1      | flat  ( 0) |
| 2      | long  (+1) |

### Reward (`reward_mode`)
The reward is computed on the bar *following* the decision (decide at `t`, PnL realized over
`t -> t+1`):

- **`pnl`** (default): `pos_new * next_bar_return - cost_per_turn * |pos_new - pos_old|`.
  This equals the per-bar net strategy return the `Backtester` computes (`fee + slippage`
  charged on the change in notional).
- **`sharpe`**: differential / risk-adjusted increment (Moody & Saffell differential Sharpe).
  Keeps EWMA estimates of the mean and variance of per-step PnL and returns the marginal
  contribution to the Sharpe ratio. Rewards smooth growth, penalizes volatility.
- **`pnl_minus_drawdown`**: the `pnl` reward minus the *new* drawdown opened this step
  (how much further below the running equity peak we fell). Discourages deep drawdowns.

### Episode termination
- `terminated=True` at the end of the data.
- `truncated=True` if `max_drawdown_blowout` is set and equity falls below
  `(1 - max_drawdown_blowout)` of its peak (risk blowout).

## The default agent: linear-softmax + cross-entropy method (CEM)
The policy is a single weight matrix `W` of shape `(obs_dim, 3)` plus bias `b`; the greedy
action is `argmax(obs @ W + b)`. CEM maintains a Gaussian over the flattened parameter
vector, samples a population each iteration, evaluates each sample's total episode reward on
the env, and refits the Gaussian to the top-`k` elites (with a small noise floor to keep
exploring). Gradient-free, tiny, numpy-only, and reliably learns a non-trivial policy.

## How to train
```bash
# Synthetic random-walk data (offline, default):
python learning/train.py

# Real cached data via data/data_manager (needs keys/config; falls back to synthetic):
python learning/train.py BTCUSDT 1h
```
Training:
1. builds an OHLCV dataset (synthetic geometric random walk, or real cached candles),
2. constructs `TradingEnv`,
3. runs CEM for a small number of iterations (prints improving episode reward),
4. saves the learned params to `models/rl_policy.npz`,
5. evaluates the trained policy through the `Backtester` and prints a metrics summary that is
   directly comparable to the EMA / ML strategies.

**Honesty note:** on synthetic random-walk data, near-zero/negative net return after costs is
*expected and correct* — there is no real edge to learn. A positive in-sample number is not
evidence of edge. Validate any RL policy on **real data** with a **chronological**
train/test split (see `backtesting.engine.Backtester.walk_forward`).

## Using a trained policy in a backtest
```python
from strategies.rl_strategy import RLStrategy
from backtesting.engine import Backtester

strat = RLStrategy(path="models/rl_policy.npz")
bt = Backtester(fee=0.0004, slippage=0.0002, initial_capital=10_000, risk_fraction=0.5)
result = bt.run(df, strat.generate_signals(df))
```

## Saved-policy format (`models/rl_policy.npz`)
A numpy `.npz` archive with:

| key      | shape                  | meaning |
|----------|------------------------|---------|
| `W`      | `(obs_dim, 3)`         | linear policy weights |
| `b`      | `(3,)`                 | bias |
| `mean`   | `(len(FEATURE_COLUMNS),)` | per-feature normalization mean |
| `std`    | `(len(FEATURE_COLUMNS),)` | per-feature normalization std (floored) |
| `window` | scalar int             | observation window |
| `obs_dim`| scalar int             | `window * len(FEATURE_COLUMNS) + 2` |

## Optional upgrade path: Stable-Baselines3 / PPO
When `gymnasium` and `stable_baselines3` are both importable (e.g. on a supported Python),
you can swap in a real deep-RL algorithm without changing the environment:

```bash
python learning/train.py --sb3          # trains PPO, saves models/rl_ppo.zip
```

Because `TradingEnv` already subclasses `gym.Env` (with proper `Box`/`Discrete` spaces) when
gymnasium is present, SB3 consumes it directly:

```python
from stable_baselines3 import PPO
from learning.env import TradingEnv

env = TradingEnv(df, reward_mode="sharpe")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("models/rl_ppo")
```

To trade the PPO model, write a thin `RLStrategy`-style wrapper whose `_act` calls
`model.predict(obs, deterministic=True)` instead of the linear `argmax`, reusing the same
observation builder from `learning/env.py` so parity is preserved. The MDP (state/action/
reward), the env, and the backtester all stay identical — only the policy class changes.
```
