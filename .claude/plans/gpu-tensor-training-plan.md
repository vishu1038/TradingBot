# Plan: dual-mode training backend — auto-shift to GPU tensors by PC specs

## Context / why
RL/policy training is currently **numpy-only CEM** on CPU (chosen because gymnasium/SB3/torch
weren't installable on Python 3.14). On a machine with a capable GPU that leaves a lot on the
table: CEM evaluates a *population* of candidate policies each iteration — embarrassingly
parallel — so a GPU can run a much larger population / longer horizon far faster, and unlocks
gradient-based RL (PPO) later. User wants the bot to **detect the PC's specs at startup and shift
training onto GPU tensors when available**, else fall back to the current CPU path.

## Design: a backend abstraction + auto-selection
1. **`learning/backend.py`** — pick an array module `xp` and a device:
   - Probe in order: CUDA torch (`torch.cuda.is_available()`), Apple MPS, CuPy (`cupy.cuda`),
     else numpy. Also read CPU cores + RAM (already used to size `--train-workers`) and GPU VRAM.
   - Expose `get_backend(prefer="auto"|"cpu"|"gpu") -> Backend` with a uniform tiny API
     (`asarray`, `matmul`, `tanh`, `argmax`, RNG, `to_numpy`). CEM/MLP are written against `xp`
     so the SAME code runs on numpy or GPU tensors.
   - `--device auto|cpu|gpu` flag (default auto); log the chosen backend + why.
2. **Phase A (recommended first): tensor-batched CEM on GPU.** Keep the CEM algorithm but evaluate
   the whole population in one batched tensor pass: stack the N candidate policies' weights as a
   tensor, run all rollouts over the bar matrix at once (vectorized over population AND time where
   possible). Same algorithm/results, big speedup. Lowest risk — `MLPPolicy._logits` is already
   just matmuls + tanh; port it to `xp`.
3. **Phase B (optional, bigger): gradient RL.** When torch+GPU present, offer a PPO/A2C policy as
   an alternative `--algo ppo`. More sample-efficient than CEM but heavier and a different code
   path; gate behind successful torch import. Keep CEM as the always-available default.

## Hardware-tiered defaults (the "based on PC specs" part)
A small policy table chosen at startup:
- **GPU (CUDA/MPS, ≥4GB)** → tensor backend; bump CEM population/iterations and train-bars
  (GPU eats the larger batch); workers=1 (GPU is the parallelism).
- **Strong CPU (≥8 cores)** → numpy + ProcessPool (current), larger `--train-workers`.
- **Weak CPU (≤4 cores)** → numpy, small population, fewer workers, longer interval.
Expose the table but let CLI flags override every value.

## Python 3.14 caveat (important)
torch/CuPy wheels may not exist for 3.14 yet. Options, in order:
1. Run the **GPU training in a separate Python 3.11/3.12 venv** with torch+CUDA, invoked as the
   training worker process (the trainer already shells training into worker PROCESSES — point the
   pool at the GPU venv's interpreter). Main bot stays on 3.14. **Cleanest given the constraint.**
2. Use CuPy if it ships for 3.14 (numpy-drop-in → minimal code change to the `xp` abstraction).
3. If neither is available, the auto-detect simply reports "no GPU backend available" and stays on
   numpy — no regression.
Verify wheel availability before committing to (1) vs (2).

## Touch points
- New `learning/backend.py` (device probe + `xp` selection).
- `learning/train.py` — `MLPPolicy`/`train_cem` parametrized on `xp`; `_pool_train_worker` learns
  to target the GPU interpreter when the GPU backend is selected.
- `run_live.py` — `--device` flag; spec-based default sizing; log the chosen backend.
- `requirements-gpu.txt` (separate, optional) for the torch/CuPy venv.

## Migration phases
1. `backend.py` + probe + `--device` flag; CEM ported to `xp` but still numpy (no behavior change).
2. Wire CuPy/torch tensor path; benchmark CPU vs GPU on identical seeds (assert equal policy, lower
   wall-clock).
3. Spec-tiered defaults.
4. (Optional) PPO backend behind `--algo ppo`.

## Constraints
- numpy CPU path must remain the guaranteed default — GPU is an accelerator, never a hard dep.
- Same seed should give the same (or statistically equivalent) policy across backends; assert in
  the benchmark step. Beware float32 nondeterminism on GPU — document tolerance.
