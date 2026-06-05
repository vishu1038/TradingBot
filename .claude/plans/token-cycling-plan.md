# Plan: 5m timeframe, churn penalty, and a live token-cycling universe

## Context
The live bot trades a fixed symbol list on 1m candles. Two problems surfaced from the live
stats and real-data backtests:
1. **1m churns on noise** — fees dominate. On real Binance data the base EMA strategy went from
   **-6% (1m)** to **+16.5% (5m)** purely from a better move-to-fee ratio. SOL's RL policy also
   over-trades (130 trades, -$75 while peers were positive).
2. **The token list is static** — the bot can't follow what's actually moving in the market.

The user wants: (a) trade on 5-minute windows, (b) lean toward volatile tokens for quicker
opportunities, and (c) **a mechanism that keeps cycling the bot into currently-active tokens**.
Decisions confirmed: re-rank **every 30 min**, **fully dynamic** (no BTC/ETH anchors), **8
symbols** concurrent, with anti-thrash guards.

## Already applied this session (pre-plan-mode; keep)
- `strategies/ema_cross.py` — volatility deadband + hysteresis (default `band_mult=1.0`,
  `trend_filter=0`). Committed (`d653185`, pushed).
- `run_live.py` — `--timeframe` default flipped to **5m**; `select_active_universe()` helper;
  `--auto-universe` + `--universe-size` flags wired into `main()` symbol resolution.
- `learning/env.py` + `learning/train.py` — `churn_penalty` param on `TradingEnv` (training-only
  reward regularizer, does NOT touch equity) threaded through `train_policy`. Default 0.0 = no-op
  until validated/wired (below).

## Work to do

### 1. Finish & wire the RL churn penalty (attacks the SOL over-trading)
- Re-run the out-of-sample validation harness on SOL (train on first 6000 bars, evaluate on a
  held-out tail) across `churn_penalty ∈ {0, 0.0005, 0.001, 0.002, 0.004}`; pick the value that
  cuts OOS trade count most while keeping OOS return ≥ the `churn=0` baseline.
- Add `--churn-penalty` CLI flag (default = the chosen value) and thread it:
  `main()` → `ContinuousTrainer.__init__` (new param) → the per-symbol kwargs dict built in
  `ContinuousTrainer._prepare_job(sym)` → `train_policy(..., churn_penalty=...)`.
- Validation harness is throwaway (`_validate_*.py`, gitignored/deleted after).

### 2. `UniverseManager` — live token cycling (the core feature)
New class in `run_live.py`, background daemon thread (mirrors `ContinuousTrainer`'s start/stop
pattern). Every `--cycle-interval` seconds (default **1800**):
- `desired = select_active_universe(real_client, size=universe_size, always_include=())`
  (fully dynamic per user choice).
- **Anti-thrash:** keep a currently-active symbol unless it falls out of the top
  `ceil(size * 1.5)` ranking, AND enforce a **min-dwell** (don't remove a symbol added less than
  one full cycle ago). This prevents flapping on rank noise.
- **Add** symbols via a `make_engine(sym)` factory (see #3): build engine, `controller.add_engine`,
  start it if the controller is running. `ContinuousTrainer` auto-picks it up (it snapshots keys).
- **Remove** symbols via `controller.remove_engine(sym)`: stops the engine, banks its final equity
  into a retired tally (see #4), drops it from the dict under lock, deletes `trainer._theta[sym]`.
- Publish a `"universe"` event + `log()` line so the dashboard/console shows each rotation.
- **Live-only:** if not `use_live`, log a warning and don't start (synthetic ReplayClients are a
  fixed set). Enable via new `--cycle-universe` flag (implies `--auto-universe` for the initial set).

### 3. Extract a `make_engine(sym)` factory in `main()`
The per-symbol construction loop (`run_live.py` ~line 552, `TradingEngine(...)`) becomes a closure
capturing `use_live`, `real_client`, the synthetic builder, `args`, `poll`, `GLOBAL_BUS`. Used for
both the initial set and runtime additions (DRY, single source of truth for engine config).

### 4. Make `MultiEngineController` mutation-safe + equity-continuous
File `run_live.py`, class at ~line 205. **Critical** (confirmed via exploration): `get_state()`
iterates `self.engines` from two threads (its aggregate loop + the HTTP handler) with no lock.
- Add `self._lock = threading.RLock()`; wrap the `get_state()` iteration and the new add/remove.
- `add_engine(sym, engine)`: under lock — `engines[sym]=engine`; `initial_capital_total += capital`;
  refresh `self.symbol` label; if `self.running`, `engine.start()`.
- `remove_engine(sym)`: under lock — `engine.stop()`; `self._retired_equity += engine.equity`,
  `self._retired_realized += engine.realized_pnl`; pop from dict; refresh label.
- `get_state()`: add `self._retired_equity` / `self._retired_realized` into the `eq` / `rpnl` sums
  so the portfolio curve stays **continuous** across removals (no artificial jump). `initial_capital_total`
  already grows on add, so return-on-capital stays honest.

### 5. CLI flags (in `main()` argparse)
- `--cycle-universe` (store_true) — enable live cycling. Off by default.
- `--cycle-interval` (float, default 1800.0) — seconds between re-ranks.
- `--churn-penalty` (float, default = value chosen in #1).
- Start/stop `UniverseManager` alongside the trainer/advisor (after `server.start()`, and in the
  `KeyboardInterrupt` handler), guarded by `use_live`.

## Critical files
- `run_live.py` — `UniverseManager` (new), `MultiEngineController` add/remove/lock/retired-equity,
  `make_engine` factory, flags, `main()` wiring. (`select_active_universe`, 5m, `--auto-universe`
  already added.)
- `learning/env.py`, `learning/train.py` — churn penalty (added; needs default wired).
- No change needed to `ContinuousTrainer` loop, `web/server.py`, or `engine/trading_engine.py`.

## Known limitations (note in code comments, not fixing now)
- All engines share one `real_client`; its REST methods are called concurrently from each engine's
  thread. This is **pre-existing** (8 engines already share it) and tolerated; cycling doesn't make
  it worse. Not adding a client lock in this change.
- Removed symbols' past fills remain in the dashboard trades table (no delete event). Acceptable.
- Real-money min-notional/lot-step/leverage/liquidation still unmodeled (separate future work for
  the ~$100 target); cycling is a paper-learning feature.

## Verification
1. **Churn:** run the OOS harness; confirm chosen penalty reduces OOS trades with return ≥ baseline.
2. **Compile/flags:** `python -m py_compile run_live.py learning/env.py learning/train.py`;
   `python run_live.py --help` shows `--cycle-universe`, `--cycle-interval`, `--churn-penalty`.
3. **Selector:** `select_active_universe(client, size=8, always_include=())` returns 8 live movers.
4. **Live smoke (short):** launch `python run_live.py --cycle-universe --cycle-interval 120 --no-browser`
   on the exchange-reachable network; click Start; over a few minutes confirm: engines spin up for the
   selected set, `/api/state` stays responsive (~150ms), a rotation event fires, portfolio equity does
   NOT jump on a removal, and the trainer hot-swaps RL policies for newly added symbols. Then stop.
5. Clean up throwaway `_validate_*.py` / `_test_universe.py`; commit on
   `feature/automated-trading-scaffold`.
