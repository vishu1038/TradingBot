# ml/ - training, evaluation, export

Trains the entry model on the GPU (`mps`) and exports it as `models/{name}.npz` + `.json`, which the
engine runs in **pure numpy** (`backend/tb/model_np.py`). Features / labels live in
`backend/tb/features.py` and `backend/tb/labels.py` (numpy/pandas only, shared with the engine and sim).

## Environment (read this before running anything)

* `.venv-ml/bin/python` = Python 3.12, torch 2.2.2, numpy 1.26, pandas 3, pyarrow 19 (pyarrow < 20 is
  required because pyarrow >= 21 needs numpy 2), pytest.
* **torch CPU math crashes** on this Ryzen hackintosh (Intel MKL on AMD: exit code 1, no traceback, even
  for a 64x64 matmul). All model math is on `mps`; tensors are built in numpy and moved with
  `torch.from_numpy(x).to("mps")`; modules are `.to("mps")` before any forward. `trainer.require_mps()`
  refuses to run otherwise. If a process dies with rc=1 and no output, this is why.
* `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` (+ `LOW=0.6`, torch requires low < high) is set by
  `trainer.py`/`walkforward.py` before torch loads. Run training with `nice -n 10`.
* Memory: features are stored fp16 (the whole 25-symbol / 13-month dataset is ~0.3 GB), windows are
  gathered per batch in a background thread (queue depth 3) and cast to fp32 on the GPU. If the array
  ever exceeds the RAM budget, `build_dataset(..., mmap=True)` memory-maps the cached `.npy` files.
  `torch.mps.empty_cache()` runs between models (`trainer.free_gpu`).

## Run

```bash
cd <repo>
export PYTHONPATH=backend
# full pipeline: dataset -> walk-forward (per arch x exit mode) -> choose on OOS -> ONE holdout eval -> export
nice -n 10 .venv-ml/bin/python ml/walkforward.py run --archs cnn,gru --modes fixed,atr
# walk-forward only (never touches the holdout)
nice -n 10 .venv-ml/bin/python ml/walkforward.py wf --archs cnn --modes fixed
# smoke test on generated data
nice -n 10 .venv-ml/bin/python ml/walkforward.py run --synthetic --archs cnn --modes fixed --steps 300
```

Useful flags: `--symbols A,B`, `--folds 4 --test-days 30 --val-days 14 --holdout-days 60`,
`--steps 2500 --gru-steps 800 --batch 512 --lr 2e-3 --patience 4 --pos-weight auto|none`.
Env: `TB_ARCHIVE` (default `data/archive`), `TB_VAR` (default `var`), `TB_MODELS` (default `models`).

Outputs

| path | content |
|---|---|
| `var/ml_cache/<key>/` | cached dataset (`.npy`, keyed by symbols + file mtimes + feature list) |
| `var/ml/wf_{arch}_{mode}.json` | full walk-forward result per fold + pooled |
| `var/ml/final_{name}.json` | the single holdout evaluation |
| `models/{name}.npz/.json` | exported model, `status: "challenger"`, `parent` = current champion |
| `var/lab/train-{name}.json` | Lab report `{id, created_at, kind:"train", title, summary, equity[{t,equity}]}` |

## Protocol (what is and is not allowed to see what)

* Data: all symbols in `data/archive`, 5m bars; features after a 576-bar warm-up; labels from
  `tb.labels` for both exit modes and both sides.
* Splits (`splits.py`): `|train|purge|val|purge|test|` expanding-window folds (default 4 x 30-day test
  blocks, 14-day val), purge = horizon (288) + window (64) bars, and a locked holdout (last 60 days).
* Per fold: normalisation stats (median / 1.4826 MAD) from train only; early stopping on val BCE;
  **Platt calibration fitted on val only** (per head, exported in the artifact); thresholds 0.5..0.9 chosen
  on val only (max expectancy, >= 30 de-overlapped trades); test block evaluated with those.
* Model selection uses walk-forward OOS only. The holdout is evaluated once, for the chosen model.
* Trades in evaluation obey the engine's position rules: one open position per symbol, <= 5 open overall.
  The **random-entry baseline** draws the same number of signals uniformly at random with the same rules and
  the same exits and costs. `breakeven_win_rate` is computed from the simulated TP / non-TP net PnL of the
  population (fees + slippage included). Edge counts only if expectancy > 0 *and* well above the baseline.
* Drift: for the trades taken at the chosen threshold the report gives how often the live drift rule
  (30-trade window, z < -2 / z < -3) would fire with calibrated vs uncalibrated probabilities.

## Files

`dataset.py` archive -> features+labels, cache, RAM budget; `splits.py` time splits; `models.py` torch archs
(cnn, gru, transformer); `trainer.py` mps training loop + prefetch; `evaluate.py` AUC / calibration /
trading table / drift; `walkforward.py` the driver + export + lab report; `synth.py` synthetic data.
Tests: `backend/tests/test_ml_*.py` (run from `backend/`; features / labels / model_np / eval under both
venvs, parity + training smoke under `.venv-ml` only).

## Artifact contract

`models/{name}.json`: name, version, created_at, **status** (`champion` | `challenger` | `retired`; new models are
always `challenger`, never auto-promoted), **parent** (the champion at export time, for rollback), arch,
arch_cfg, feature_names, window, norm {mean, std, clip}, **calibration** {long, short} (`platt` a/b or
`isotonic` knots or `none`), thresholds {long, short}, exit_cfg, metrics. `models/{name}.npz`: the torch
state_dict as float32. `tb.model_np.load(path)` returns the scorer; `predict_windows` gives calibrated
probabilities for batches, `score(symbol, feats)` gives `Score` with the top-5 occlusion reasons.
numpy vs torch(mps) parity: max abs logit diff ~1e-6 for all archs (`test_ml_parity.py`, limit 1e-4).
