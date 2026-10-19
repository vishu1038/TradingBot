# V3 experiments: contract (user-approved 2026-10-19: "do all, deploy")

Why: every 5m-decision / 5m-ATR-exit model scored ~coin-flip (docs/ARCHITECTURE.md, var/ml/*). The user
watched 5m charts only to TIME entries; positions were decided on context and held hours–days (TP +3 %,
SL −6 % on $160). V3 changes the question, the timeframe and the costs, and runs many models in parallel.
Everything here extends docs/ARCHITECTURE.md; its safety layer, PnL convention and risk caps still apply.

## Experiments (all evaluated with the same honest walk-forward, locked holdout, and BOTH baselines:
random entries with identical exits, and the ATR%-regime rule; report multiple-testing count)
- **B — multi-timeframe ("user style")**: features from 5m + 1h + 4h (+1d) built by RESAMPLING the 5m
  archive (no new data). Decision: a coin is "set up" from 1h/4h context; entry timing on 5m within the
  setup window. Exits sized on 1h/4h ATR or fixed pct (3 %/6 %). Per-coin cooldown; target a few trades per
  day in total. Labels: TP-before-SL over 1–4 days with those exits.
- **C — gradient-boosted trees**: sklearn `HistGradientBoostingClassifier` (no new deps, CPU) on the same
  feature sets as B and the old 5m set. Sanity baseline for the nets.
- **E — vehicle-style state estimation**: Kalman-filtered trend level/slope and volatility (+ their
  uncertainty) per timeframe as a feature group `kalman` (via `tb.features.register_feature_group`).
  The user may send vehicle algorithms; analogues get added here.
- **A — cross-sectional ranking**: at each decision time (e.g. every 1 h) rank the universe by predicted
  relative forward return (vs the universe mean) over 24 h; long top-k, short bottom-k, equal risk,
  rebalance every N h; market-neutral. Exchange-side stop on every leg.
- **D — cheaper execution**: maker (post-only, GTX) limit entries at best bid/ask with re-price/timeout,
  fee 0.02 % vs taker 0.05 %; sim models fill probability conservatively (fill only if a later bar trades
  THROUGH the limit price). Also a low-volatility no-trade gate where fees dominate.

## Model artifact additions (`models/{name}.json`)
`kind`: `"per_symbol_5m"` (existing, default) | `"mtf"` | `"gbm"` | `"xsec"`;
`timeframes`: e.g. `["5m","1h","4h"]`; `history_bars_5m`: 5m bars of history the live engine must supply;
`exit_cfg.atr_tf`: timeframe of the ATR used for exits (`"5m"` default); `exit_cfg.exit_mode` may be
`"pct"` (`tp_pct`, `sl_pct`); `entry_cfg`: `{decision_every_bars, cooldown_bars, max_trades_per_day,
entry_mode: "taker"|"maker", maker_timeout_s, maker_reprices}`; for `xsec`: `{rebalance_every_h, top_k,
bottom_k, horizon_h}`. `gbm` weights stored as a pickle next to the json (`.pkl`, sklearn); everything
else stays numpy `.npz`. `status` is always `"challenger"` on export; names are unique and descriptive.

## Scorer interface v2 (`backend/tb/scorer.py`)
Existing per-symbol `score(symbol, feats)` stays. New: `Scorer.kind`, `Scorer.history_bars_5m`, and
`Scorer.score_universe(frames: dict[symbol, FrameBundle]) -> dict[symbol, Score]` (xsec ranks need all
coins at once; Score.prob_long/prob_short become "go long / go short" confidences, reasons as before).
`FrameBundle` = the 5m klines/metrics/funding (+ depth when present) for that symbol, long enough for the
model's `history_bars_5m`. Feature building for a model lives in ONE function shared by training and live
(`tb.features_mtf.build_for_model(meta, bundle)`), so train/live parity is testable.

## Engine requirements
- Load and SHADOW many challengers at once (any kind); champion still via the existing promotion rule.
- Fetch enough 5m history per symbol for the largest `history_bars_5m` among loaded models (page REST).
- Respect `entry_cfg` (decision cadence, cooldown, max trades/day) and `entry_mode: maker` (post-only GTX,
  re-price, timeout → skip or taker per setting) — in paper, testnet and the shadow simulator.
- xsec champion executes a target portfolio (open/close legs on rebalance) under all existing caps
  (open-risk, total leverage, margin ratio, kill, correlation not applied within a neutral pair book).
- Never live-mode changes; PAPER TEST ONLY rule unchanged.

## Ownership (parallel agents; don't edit another agent's files)
- ML-1 (B, C, E): `backend/tb/features_mtf.py`, `backend/tb/features_kalman.py`, `backend/tb/labels.py`
  (additive: pct/atr_tf exits, cooldown-aware labels), `backend/tb/model_np.py` (mtf nets), `backend/tb/model_gbm.py`,
  `ml/` (mtf/gbm/kalman paths), tests `backend/tests/test_v3_mtf_*.py`.
- ML-2 (A, D): `backend/tb/features_xsec.py`, `backend/tb/model_xsec.py`, `backend/tb/sim/xsec.py`,
  `backend/tb/sim/execution.py` (maker model, shared with labels via import), `ml/xsec_*.py`, tests
  `backend/tests/test_v3_xsec_*.py`, `backend/tests/test_v3_exec_*.py`.
- Engine: `backend/tb/scorer.py`, `backend/tb/engine.py`, `backend/tb/shadow.py`, `backend/tb/exchange/*`,
  `backend/tb/api.py`, `backend/tb/config.py`, tests `backend/tests/test_v3_engine_*.py`, docs updates.
- GPU/CPU: other jobs may be running (ml/run_all229.sh, ml/run_retrain_challenger.sh). Check with `ps`;
  at most 2 GPU trainings at once; `nice -n 10`; long runs DETACHED via `nohup` scripts that post results
  with `ml/notify_text.py` (no Claude polling). System RAM is tight (~3 GB free): memmap/stream.
