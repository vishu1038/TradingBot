# TradingBot v2 — architecture & contracts

Source of truth that every component is built against. If you change a contract here, every
consumer must change with it.

## Goal
Grow a small Binance USDⓈ-M futures account (~$100 real, after paper). 5m-chart trading on the
most volatile liquid perps, entries chosen by a neural network running on the GPU, exits by a
fixed-risk template. Mostly long; shorts only on a stricter threshold.

## Process layout (the UI can never hurt trading)
```
 ┌──────────────┐  SQLite (WAL)   ┌──────────────┐  REST + WS   ┌──────────┐
 │  tb-engine   │ ──── state ───▶ │   tb-api     │ ───────────▶ │  ui/     │
 │ (trading)    │ ◀── commands ── │ (FastAPI)    │ ◀─ commands ─│ browser  │
 └──────┬───────┘                 └──────────────┘              └──────────┘
        │ REST/WS                     ml/ (py3.12, GPU) trains offline → models/*.npz
     Binance                          engine loads the .npz and runs inference in numpy
```
- `tb-engine` and `tb-api` are **separate OS processes**. The engine never imports the API and
  never waits on it. Killing the API or closing the browser changes nothing for open trades.
- They share one SQLite DB (`var/tb.sqlite`, WAL mode). The engine writes state/events; the API
  reads them and inserts rows into a `commands` table that the engine polls (~0.5 s).
- Live push: the engine appends to an `events` table (autoincrement id); the API tails it and
  fans out over WebSocket.
- Model inference in the engine is **pure numpy** (weights exported from torch to `.npz`), so the
  backend runs anywhere (Mac, VPS, Pi) without torch.

## Repo layout & ownership
| Path | Owner | Python |
|---|---|---|
| `ui/` | UI | none (static ES modules, no build step) |
| `backend/tb/` exchange, engine, api, store, risk, universe, alerts (Google Chat webhook), scorer | backend | 3.14 `.venv` |
| `backend/tb/data/` (archive downloader, live kline fetch) + `backend/tb/sim/` (backtester) | data | 3.14 `.venv` |
| `backend/tb/features.py`, `backend/tb/labels.py`, `backend/tb/model_np.py`, `ml/` | ML | features/labels/model_np: numpy-only, must import under BOTH 3.14 and 3.12; `ml/` is 3.12 `.venv-ml` |
| `docs/` | lead | — |

Shared runtime dirs (gitignored): `var/` (DB, logs), `data/archive/` (parquet), `models/` (npz).

## Exchange & modes
- Binance USDⓈ-M only in v1 (exchange layer is an interface so BitMEX can be added later).
- Modes: `paper_local` (real prod market data, simulated fills — no keys), `binance_demo`
  (real orders on Binance demo/testnet, fake money), `live` (real money).
- Base URLs: prod `https://fapi.binance.com`; demo `https://demo-fapi.binance.com`
  (legacy testnet `https://testnet.binancefuture.com` also accepted via config).
- `live` requires env `ALLOW_LIVE_TRADING=true` AND an explicit config flag. Keys only from
  `.env` (`BINANCE_API_KEY`, `BINANCE_API_SECRET`); never logged.
- HTTPS: use `requests` (certifi). Never bare `urllib` (macOS python.org cert store fails here),
  and never swallow network errors silently — log the exception.

## Trade template (shared by sim, labels, engine)
- Account split into slices: `slice_margin_usd=20`, `leverage=8`, isolated margin →
  notional = 160 USD. `max_positions = floor(equity / slice_margin)` capped by setting (default 5).
- Exit modes (setting `exit_mode`):
  - `fixed`: TP at +`tp_usd` (5) PnL, SL at −`sl_usd` (10) PnL on the position's notional.
  - `atr`: TP = entry ± `tp_atr`·ATR14(5m) (1.5), SL = entry ∓ `sl_atr`·ATR14 (1.0); qty sized so
    the SL loss ≈ `sl_usd`, capped by slice notional.
  - optional trailing stop: once price has moved `trail.activate_r` × (initial risk) in favour,
    the stop trails at `trail.distance_atr` × ATR behind the best price; never loosens.
- Fees: taker 0.05% per side (config). Slippage model: 0.02% + extra on stop fills (config).
  Funding applied at 00/08/16 UTC from the funding series.
- Same-bar ambiguity (TP and SL both inside one 5m candle): assume **SL first** (pessimistic).
- Entry happens at the **open of the bar after** the signal bar (no look-ahead).

## Hard safety invariants (engine)
1. The SL is placed **on the exchange** (reduce-only STOP_MARKET, workingType MARK_PRICE) right
   after the entry fills. If placing it fails, close the position at market immediately.
   (Binance has been migrating conditional orders to an Algo Order endpoint — if `/fapi/v1/order`
   rejects STOP_MARKET with an algo-switch error, use the algo order endpoint. Verify against
   current docs.)
2. Every engine loop checks: no open position without a live stop. If found → close it.
3. Kill switch: equity drawdown from peak ≥ `kill_dd_pct` (paper 50, live 25) → close all, stop
   opening, require manual `reset_kill`.
4. Correlation cap: before opening, compute 24h 5m-return correlation with each open position;
   if corr > `corr_cap.max_corr` (0.8) with ≥ `corr_cap.max_correlated` (2) same-direction
   positions → skip.
5. A symbol dropping out of the universe never force-closes its trade; it runs to TP/SL.

## Alerts
No Telegram. `tb/alerts.py` has a pluggable `Notifier` interface; v1 channel = Google Chat
incoming webhook (env `GOOGLE_CHAT_WEBHOOK_URL`, POST `{"text": ...}`; URL holds secrets — never
logged). Sent from a background queue, rate-limited (~1 msg/s), never blocks the engine. Later
channel: the native Android app (push + widget for live trades).

## Safety & robustness layer (vehicle-software patterns, user-approved 2026-10-18)
1. **Independent guard process** (`python -m tb.guard`, ISO 26262-style safety monitor). Own
   exchange client; does not import the engine. Every ~5 s it checks: engine heartbeat age,
   every open position has a live exchange-side stop, account drawdown vs kill limit. It writes
   `safety_state` to the store, which the engine MUST obey before any entry:
   `NORMAL` → `DEGRADED` (no new entries, keep stops) → `FLATTEN` (close all) → `HALTED`
   (manual `reset_kill` only). Engine heartbeat stale > 30 s with open positions → guard ensures
   stops exist (places them itself if missing, demo/live) and sets DEGRADED. In `paper_local`
   the guard acts through the commands table (paper fills live in the engine).
2. **Data plausibility + fault debouncing.** Reject a bar/tick before use when: single-bar
   |return| > max(15%, 8×ATR14); high < low or non-positive prices; zero volume; bar timestamp
   stale > 2 bars; |mark − last| > 1%. A fault must occur `fault_mature_n` (3) consecutive
   cycles to mature; it clears after `fault_recover_m` (5) good cycles (hysteresis). Escalation
   per symbol: warn → exclude symbol from entries; exchange-wide (REST/WS errors): warn →
   DEGRADED → FLATTEN only if stops cannot be confirmed.
3. **Drawdown derating curve.** `risk.derate_multiplier(dd_pct, kill_dd_pct) -> float`:
   1.0 until 40% of the kill limit, linear to 0.5 at 80%, linear to 0 at 100% (live 25:
   10% → 1.0, 20% → 0.5, 25% → 0; paper 50: 20/40/50). Risk per trade = sl_usd × multiplier;
   if the result is below the exchange min notional, skip the trade. The backtester uses the
   SAME function (import from tb.risk).
4. **Drift ("slip") detection.** Over the last `drift_n` (30) closed trades: predicted = mean
   model prob of the taken side, realised = win rate; z = (realised − predicted) /
   sqrt(p(1−p)/n). z < −2 → size × 0.5; z < −3 → DEGRADED for entries until recovered;
   clears when z > −1 (hysteresis). Also track fill slippage vs the sim assumption; > 2× → alert
   + size × 0.5. Requires calibrated model probabilities (ML must report calibration).
5. **Shadow mode + staged rollout.** `models/*.json` has `status`: `champion` | `challenger` |
   `retired`. Challengers score every bar alongside the champion; their would-be trades are
   simulated with the same template into `shadow_trades`. Promotion (control action
   `promote_model`, manual) requires ≥ 50 shadow trades AND higher expectancy than the champion
   over the same period AND no drift alarm. `rollback_model` restores the previous champion
   instantly. Stages: paper_local → binance_demo → live; the first 20 live trades run at
   size × 0.5.

API additions: `/api/status` gains `safety:{state, reasons:[str], derate_mult, drift:{z, n,
mult}, faults:[{symbol, kind, count, since}]}`; `GET /api/shadow` →
`[{model, status, n_trades, win_rate, expectancy, pnl, vs_champion}]`; control actions
`promote_model` / `rollback_model` take `model`. WS types add `safety`, `shadow_trade`.

## Universe (hot list)
Refresh every `universe_refresh_s` (3600). Candidates: USDⓈ-M perpetuals quoted in USDT, status
TRADING. Exclude: 24h quote volume < 50M USD, listed < 7 days (`onboardDate`), |funding| > 0.1%.
Score = ATR%(5m, last 6h) × sqrt(24h quote volume). Keep top `universe_size` (8), with a
retention buffer: a current member stays while it ranks within top ceil(1.5·size).

## Scorer interface (engine ↔ model)
`backend/tb/scorer.py`:
```python
class Scorer(Protocol):
    name: str; version: str
    def score(self, symbol: str, feats: np.ndarray) -> Score: ...
# Score = dataclass(prob_long: float, prob_short: float, reasons: list[Reason])
# Reason = dataclass(feature: str, label: str, value: float, contribution: float)
```
`feats` is the `[T, F]` float32 window from `features.build_features` (latest bar last).
Default scorer when no model file exists: `NullScorer` → probs 0, never trades.
Trade if `prob_long ≥ prob_threshold_long` (default 0.70) or, when `short_enabled`,
`prob_short ≥ prob_threshold_short` (default 0.80). Thresholds are configurable and the Lab
reports which value is break-even after fees for the chosen exit mode.

## Features contract (`backend/tb/features.py`, numpy/pandas only)
```python
FEATURE_NAMES: list[str]                     # stable order, F ≈ 40
WINDOW = 64                                   # bars of history the model sees
def build_features(kl: pd.DataFrame, metrics: pd.DataFrame | None,
                   funding: pd.DataFrame | None, btc_kl: pd.DataFrame) -> np.ndarray  # [N, F] float32
```
Inputs are the archive frames below, aligned on 5m `open_time`. Every feature must be computable
live from data available at that bar's close (no future data). Missing metrics → NaN → filled
with 0 plus a `has_metrics` flag. Features are scale-free (returns, ratios, z-scores).

## Data formats (`data/archive/`, parquet, UTC ms timestamps)
- `klines/5m/{SYMBOL}.parquet`: open_time, open, high, low, close, volume, quote_volume,
  trades, taker_buy_volume, taker_buy_quote_volume
- `metrics/{SYMBOL}.parquet` (5m, from data.binance.vision `futures/um/daily/metrics`):
  create_time, sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio,
  sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio
- `funding/{SYMBOL}.parquet`: calc_time, funding_rate
Source: `https://data.binance.vision/data/futures/um/{monthly|daily}/{klines|metrics|fundingRate}/…`
(monthly zips for full months, daily for the current month). Live top-up via REST.

## Labels contract (`backend/tb/labels.py`)
```python
def label_trades(kl: pd.DataFrame, side: Literal["long","short"], exit_cfg: dict,
                 horizon_bars: int = 288) -> np.ndarray  # [N] float32: 1 TP first, 0 SL first/timeout, NaN if not enough future
```
Uses the exact trade template above (entry next bar open, fees, SL-first ambiguity). The
backtester and the labels must agree; the data agent's sim tests check that.

## Model artifact (`models/{name}.npz` + `models/{name}.json`)
JSON: name, version, created_at, arch, feature_names, window, norm (mean/std per feature),
thresholds, exit_cfg, metrics (walk-forward results). NPZ: weights. `model_np.load(path)`
returns a `Scorer`. numpy forward must match the torch MPS forward to ~1e-4.

## GPU / ML constraints (this machine: Ryzen 5 3500 hackintosh, RX 6600 XT 8 GB, 24 GB RAM)
- Training: `.venv-ml` (Python 3.12, torch 2.2.2 — the last x86_64 macOS build), device `mps`.
- **torch CPU math crashes** (Intel MKL on AMD; even a 64×64 CPU matmul exits rc=1 silently).
  All model math on `mps`. Data prep in numpy (OpenBLAS, fine). `torch.from_numpy(x).to("mps")`
  is safe.
- Memory: dataset in RAM / `np.memmap` as fp16, stream batches (~7 ms per 512×64×40 batch),
  one prefetch thread, budget ≤ ~8 GB RAM; `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` (the card also
  drives the display). Measured: 1D-CNN ~50 ms/step at batch 512; GRU learns better but ~10×
  slower; a transformer OOM'd after other models — free memory between runs.
- Validation: walk-forward with a purge gap ≥ label horizon; locked final holdout; compare to
  the random-entry baseline with identical exits; report calibration.

## API contract (tb-api, default `http://127.0.0.1:8787`, CORS open to localhost)
All timestamps unix **seconds** in the API. JSON.
- `GET /api/health` → `{ok, engine_alive, engine_heartbeat_age_s, mode, version}`
- `GET /api/status` → `{mode, running, equity, start_equity, peak_equity, cash, realized_pnl,
  unrealized_pnl, drawdown_pct, open_positions, max_positions, trades_today, win_rate,
  kill:{tripped, reason, dd_limit_pct}, model:{name, version, trained_at, threshold_long,
  threshold_short}, universe_updated_at}`
- `GET /api/market` → `{btc_price, btc_change_24h_pct, btc_trend:"up"|"down"|"flat",
  market_volatility_pct, avg_funding_rate, long_short_ratio, updated_at}`
- `GET /api/universe` → `[{symbol, score, atr_pct, quote_volume_24h, change_24h_pct,
  funding_rate, open_interest_usd, oi_change_1h_pct, last_price, in_universe_since}]`
- `GET /api/candles?symbol=&tf=5m&limit=500` → `[{t, o, h, l, c, v}]`
- `GET /api/signals?symbol=&limit=200` → `[{t, symbol, prob_long, prob_short,
  action:"long"|"short"|"none", reasons:[{feature,label,value,contribution}]}]`
- `GET /api/positions` → `[{id, symbol, side, entry_price, qty, notional, leverage, margin,
  sl_price, tp_price, trail_active, trail_price, mark_price, upnl, opened_at, prob, reasons}]`
- `GET /api/trades?limit=200` → `[{id, symbol, side, entry_price, exit_price, qty, opened_at,
  closed_at, pnl, fees, funding, exit_reason:"tp"|"sl"|"trail"|"kill"|"manual"|"liquidation",
  prob}]`
- `GET /api/equity?limit=2000` → `[{t, equity}]`
- `GET /api/logs?limit=200` → `[{t, level, msg}]`
- `GET /api/settings` / `PUT /api/settings` (partial merge) → `{slice_margin_usd, leverage,
  max_positions, exit_mode, tp_usd, sl_usd, tp_atr, sl_atr, trail:{enabled, activate_r,
  distance_atr}, short_enabled, prob_threshold_long, prob_threshold_short,
  corr_cap:{max_corr, max_correlated}, universe_size, universe_refresh_s, kill_dd_pct, fee_taker,
  sound}`
- `POST /api/control` `{action:"start"|"stop"|"kill"|"reset_kill"|"close_position"|"close_all",
  position_id?}` → `{ok, error?}`
- `GET /api/lab/reports` → `[{id, created_at, kind:"backtest"|"train", title, summary:{...},
  equity:[{t,equity}]}]`
- `WS /ws` → messages `{type, data}`; types: `status`, `tick` `{symbol,t,price}`, `candle`
  `{symbol,tf,candle}`, `signal`, `position_opened`, `position_updated`, `position_closed`,
  `universe`, `equity` `{t,equity}`, `log`, `kill`.
