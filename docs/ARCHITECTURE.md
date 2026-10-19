# TradingBot v2 — architecture & contracts

Source of truth that every component is built against. If you change a contract here, every
consumer must change with it.

## Goal
Grow a small Binance USDⓈ-M futures account (~$100 real, after paper). 5m-chart trading on the
most volatile liquid perps, entries chosen by a neural network running on the GPU, exits by a
fixed-risk template. Long and short are symmetric (user 2026-10-19): the side comes from the model's data, never from a built-in
lean (equal default thresholds, shorts on, no side-specific caps), and the book is protected against a market reversal ("Reversal protection").

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

## UI layout & style (user-approved 2026-10-18)
**Style: hybrid.** Terminal chrome — black background, box-drawn panel frames with `╭─ TITLE ─╮`
headers, `/` command palette, ASCII spinners, typed-out log lines,
rainbow logo — plus modern foreground controls (pill buttons, toggles, sliders, smooth gauges,
glowing cards, charts) sized for touch. **Typography (user, 2026-10-18): all text — letters AND numbers — in a modern sans font, not a
terminal font**: system UI stack (`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI",
Roboto, "Helvetica Neue", Arial, sans-serif`), no web-font download; numbers use
`font-variant-numeric: tabular-nums` so columns align. Monospace only for decorative glyphs
(box-drawing borders, ASCII spinners, logo art). **Lots of colour** (Claude Code multicolour): each panel
its own accent (coral, violet, cyan, lime, amber, pink…), each coin a fixed colour everywhere,
probabilities on a cool→hot gradient; profit green / loss red reserved for money only.
**Effects:** "loud but readable" default (flash + glitch on TP/SL/kill only) + a calm↔degen
slider in Settings; respect prefers-reduced-motion. **Keyboard-first** on desktop (shortcuts for
every action, j/k between coins, `/` palette: /kill /history /lab /coin SOL /pause WIF …).

**Page 1 (main):**
- Top bar (always visible, every page): logo, mode pill, safety lamp, equity + PnL, drawdown +
  derate gauge, START/STOP, **KILL**, `/` palette.
- Symbols bar: each hot coin = price, 24h %, model prob, ● if the bot holds it; ⟳ force refresh.
- Left column: **Hero card** (small, top-left) rotating every 30 s through 4 cards — PnL+health,
  open trades (uPnL + TP/SL progress bars), mini chart, bot brain (per-coin prob vs threshold) —
  with a progress bar; **pauses on hover/touch**; **events jump to their card** (TP/SL → open
  trades, new entry → brain, kill/degraded → health); swipe/drag to flip. Below it the **main
  candlestick chart** (fixed, never rotates): candles, volume, EMAs, entry/TP/SL/trail lines,
  model-prob strip with dashed threshold. Below the chart: **Live trades** = the bot's open
  positions (live uPnL, close [×]) + recently closed trades.
- Right column, stacked: **Depth chart** (bid/ask cumulative depth at ±0.2/1/2/3/4/5 % — the same
  bands the model reads; a small ladder tab inside), **Bot stats & controls** (win rate, open/max,
  model, drift; start/stop, close position, pause coin, force universe refresh), **Market / stats**
  (BTC trend, volatility, funding, L/S ratio).
**Page 2: Bot settings** (full page): risk, exits, universe, model thresholds, safety, alerts,
effects slider, sound — and a **separate LIVE-mode card** with type-to-confirm boxes (e.g. type
`LIVE $100`), the −25 % kill + derating summary, and its own arm/disarm controls.
**Extra tabs:** Markets (universe table, heatmap, data faults), Trades history, Bots/Equity
(equity curve, shadow models with promote/rollback), Lab (reports).
**Handhelds (iPad/phone), touch-first:** hero + chart on top, swipe left/right through Depth →
Bot → Market → Trades panels, bottom tab bar for pages. Desk monitor (1080p+) is the primary
target; iPad and phone must work.

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
- Trade SIZE (notional) = `slice_margin_usd × leverage` = 160 USD (template), risk-sized by the exit mode.
  Per-trade LEVERAGE is chosen automatically (`risk.choose_leverage`, used by engine, paper and sim): the highest
  integer L ≤ `max_leverage` (20, 1–125) and ≤ the symbol bracket's maxLeverage such that the isolated
  liquidation price (Binance form, tier mmr/cum) is ≥ 1.5 × |entry − SL| from entry; margin = notional / L;
  `POST /fapi/v1/leverage` is sent before each entry (isolated margin).
- Concurrency has no fixed cap (`max_positions` optional, default null). A new entry is allowed only if,
  INCLUDING it: Σnotional / bot equity ≤ `max_total_leverage` (20); Σ maintenance margin / equity ≤
  `max_margin_ratio` (0.5; MM = notional × mmr − cum from `GET /fapi/v1/leverageBracket`, default mmr 1% when
  unknown, e.g. paper); Σ loss-if-every-SL-fills (incl. fees + stop slippage) / equity ≤ `max_open_risk_pct`
  (20 %); free margin ≥ slice margin + entry fee. The blocking rule is logged. `/api/status.max_positions` =
  open + how many more template slices currently fit; it also reports `total_leverage`, `margin_ratio`,
  `open_risk_pct` and their maxima.
- Exit modes (setting `exit_mode`):
  - `fixed`: TP at +`tp_usd` (5) PnL, SL at −`sl_usd` (10) PnL on the position's notional.
  - `atr`: TP = entry ± `tp_atr`·ATR14(5m) (1.5), SL = entry ∓ `sl_atr`·ATR14 (1.0); qty sized so
    the SL loss ≈ `sl_usd`, capped by slice notional.
  - trailing stop (default ON since 2026-10-19: `activate_r` 1.5, `distance_atr` 1.0): once price has moved
    `trail.activate_r` × (initial risk) in favour, the stop trails at `trail.distance_atr` × ATR behind the best price;
    never loosens. See "Profit protection" below.
- Fees: taker 0.05% per side (config). Slippage model: 0.02% + extra on stop fills (config).
  Funding applied at 00/08/16 UTC from the funding series.
- Same-bar ambiguity (TP and SL both inside one 5m candle): assume **SL first** (pessimistic).
- Entry happens at the **open of the bar after** the signal bar (no look-ahead).

## Profit protection (user-approved 2026-10-19; `tb/protect.py`, shared by engine, shadow and `tb.sim`)
Incident: +32 % equity that was mostly OPEN profit on ~12 positions (10 correlated alt longs) was erased in 8 minutes, and one
coin was stopped out twice within 3 minutes. Five settings (in `/api/settings`, validated), plus the trailing stop above:
- **Break-even** (`breakeven_r`, 1.0; 0 = off). Once the favourable move >= `breakeven_r` × initial risk (|entry − initial SL|)
  the stop moves to entry ± round-trip fees (`entry × (1 ± 2·fee_taker)`). The stop only ever tightens; break-even and trailing
  are combined by `protect.next_stop` (the tighter wins).
- **Moving the exchange stop** (engine, every mode): the NEW reduce-only stop is placed first, the old one is cancelled afterwards
  (`Exchange.replace_conditional`; algo orders are cancelled through the algo endpoint). If placing the new stop fails (3 tries, never
  retried on -2021) the old stop stays live (the position is never without one) and the move is retried after 30 s; a stop that
  could not be cancelled is remembered and cancelled later. Throttle: a move needs an improvement >= 0.1 % of the entry price and
  happens at most once per 5m bar per position. Price already through the new level → market exit (`sl` / `trail`). Applies to the
  Settings exits and to model exits (`pct`, `atr:<tf>`; the trail of an `atr:<tf>` position uses that timeframe's ATR; `pct` exits and
  cross-sectional legs are not trailed, legs still get break-even).
- **Cooldown after a losing exit** (`sl_cooldown_bars`, 24 = 2 h of 5m bars). No new entry on the same coin after an exit with reason
  `sl` / `trail` / `kill` / `timeout` / `liquidation` AND net pnl <= 0 (a stop that locked in profit does not count); in addition to
  `cooldown_bars`. Rebuilt from the `trades` table at start (survives restarts); shadow books and `tb.sim` use the same rule.
- **One-direction cap** (`max_side_risk_pct`, 60). `risk.check_entry(..., side)` rule `side_risk`: the open stop-out risk of ONE side
  (longs or shorts, incl. the new entry) may not exceed `max_side_risk_pct` % of the open-risk BUDGET (`max_open_risk_pct` % × bot
  equity); e.g. 35 % of $200 = $70 budget → at most $42 of long risk. Log: `blocked by rule side_risk (...)`. Cross-sectional legs
  are checked per leg. A stop moved to break-even stops counting as risk, which frees capacity. Shadow books apply it per model against a
  static equity proxy (`capital_usd` or `paper_start_equity`); `tb.sim` against `SimConfig.max_open_risk_pct` × equity (100 = off in both).
- **Partial take-profit** (`partial_tp_r` 1.5, `partial_tp_frac` 0.5; `partial_tp_r` 0 = off). Once per position, when price reaches
  entry ± `partial_tp_r` × initial risk (and a TP, if any, is further than 0.1 % beyond that level): a reduce-only MARKET order closes
  `frac` of the quantity (lot step respected; skipped for good when either part would be below min qty / min notional). It is a trade
  row with `exit_reason: "partial"` (NET pnl, fees = entry-fee share + exit fee, funding share as a cost; the remainder's row carries the
  rest, so the two rows sum to the position's net). Then stop and TP are re-placed for the remaining quantity (stop at least at
  break-even; if the new stop cannot be placed the old, larger one stays and the resize is retried). Win rate / trade counts
  (`trade_stats`, drift) count a scaled-out position once. Not used for cross-sectional legs. Exchange close-outs rebuilt from fills
  (`last_close(..., booked=)`) exclude the booked partial. State per position lives in kv `protect`.
- **Simulators.** `SimConfig` has the same five flags with the same defaults (and `trail_enabled` True / `trail_activate_r` 1.5);
  `tb.sim.run` exposes them (symmetric defaults: `--shorts` on, `--thr-long` = `--thr-short` = 0.70, `--side both`) (`--no-trail`, `--breakeven-r`, `--sl-cooldown-bars`, `--max-side-risk`, `--partial-r`, ...). Break-even /
  trail moves take effect from the NEXT bar; the partial fires after the stop check (SL first) and before the TP, at its level (gap: the
  open) with slippage + taker fee. Sim and shadow fold a partial into the position's single trade row. Labels (`tb.labels`) and
  `tb.sim.xsec` do not model any of this: the label-parity tests run with the protection off.

## Dollar floor (user 2026-10-19; `tb.risk.dollar_floor`, `size_trade`, shadow, `tb.sim`)
"TP was taken on coins for just cents": drawdown derating (mult down to 0.07) shrank positions and the 50 % partial booked tiny amounts.
- `min_tp_usd` / `min_sl_usd` (2.0 each; 0 = off). At entry, AFTER all sizing (derating, drift, slippage / staged multipliers, caps, lot
  rounding) the dollar move to the TP and to the SL for the ACTUAL quantity (`qty × |level − entry|`) must be >= its floor, else the trade is
  SKIPPED (`skip ...: target below $ floor: SL $1.60 < $2`), never shrunk below it. A stop-only position (cross-sectional leg, `tp_enabled`
  false) is checked on the stop only. Live champion (`size_trade`), shadow (`ShadowBook._make_pos`, sizing at mult 1.0 still applies it) and
  `tb.sim.backtest` (`skip_dollar_floor`) apply the same rule. No cross-validation against `tp_usd` / `sl_usd`: a `tp_usd` below the floor
  simply never trades.
- **Partial take-profit**: only when the closed piece gains >= `min_tp_usd` at the partial level (`protect.partial_gain_ok`); otherwise there
  is no partial (`partial` state "skipped") and the position runs to its full TP.

## Reversal protection (user-approved 2026-10-19; `tb/reversal.py`, shared by engine, shadow and `tb.sim`)
Symmetric trading "in such a way that if the market reverses, the bots don't lose money on their other edges". Together with the
60 % `max_side_risk_pct` cap (unchanged):
1. **Beta-weighted net exposure cap** (`max_net_exposure_pct`, 150; 0 = off). Beta of each coin vs BTC = cov/var of the last 7 days of 1h returns
   (up to 168; needs >= 48; fallback 1.0; clipped to [0.25, 3]); the engine refreshes it hourly from 1h klines (`refresh_betas`, <= 3 requests
   per bars cycle; the sim derives it rolling from its own data, BTC or the equal-weight basket as the reference). Net exposure = Σ side × notional ×
   beta (+1 long, −1 short), book = open positions + in-flight maker entries. Meaning: `|net| <= max_net_exposure_pct % × equity`. An entry that
   would push |net| beyond the cap is rejected (`blocked by rule net_exposure`); one that REDUCES |net| is always allowed, and so is a lone position
   into a flat book (otherwise a $160 slice on a $100 stake could never trade). NOTE: at 150 % and a $100–200 stake that is about one slice
   of net exposure: more positions need the other side to hedge them. `Book.open_net`, `risk.net_exposure_check`.
2. **Basket reversal detector.** On every new closed 5m bar the equal-weight return of the current universe and BTC over 15 min (3 bars) and 1 h
   (12 bars) is turned into a z-score against its own volatility (RMS of the earlier, non-overlapping windows of the last 576 bars; needs 96;
   `reversal.market_signal`). The book's net side = sign of net exposure (needs |net| >= 5 % of equity). When the move is AGAINST that side with
   z >= `reversal_z` (2.5; 0 = off) for any of the four series: (a) every open position of that side in profit >= `reversal_lock_r` (0.5) initial risks
   gets its exchange-side stop tightened to keep half of the open profit but at least break-even + round-trip fees (`reversal.lock_stop`: never
   loosens, never through the market, nothing is closed in a panic; the move goes through the normal stop-replace path of `_protect`, so the
   stop is replaced, not duplicated, and the usual 0.1 % / once-per-bar throttle applies; winners that appear while the window is open are locked
   too); (b) new entries on that side are blocked for `reversal_block_bars` (6) bars (`skip ...: entries are blocked after a market reversal`);
   (c) an event `reversal` + a line in the alert digest. Losing positions keep their stops. The window is persisted (kv `reversal`).
   `/api/status` adds `net_exposure_pct`, `max_net_exposure_pct` and `reversal {blocked, last, z}`.
3. **Shadow books** run the same rules per model book (`ShadowBook.apply_reversal`, `betas`, a static equity proxy like the side cap; counters in
   `ShadowBook.skips`). **Simulator**: `SimConfig.max_net_exposure_pct / reversal_z / reversal_lock_r / reversal_block_bars` and
   `tb.sim.run --max-net-exposure --reversal-z --reversal-lock-r --reversal-block-bars` (0 = off), stats `skip_net_exposure`,
   `skip_reversal_block`, `reversal_triggers`, `reversal_locks`; the lock applies from the next bar. `tb.sim.model_backtest` and the labels do not model it.

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
   **Per-symbol emergency paths (`tb/emergency.py`, incident 2026-10-19 testnet).** One misbehaving coin never escalates the
   account: a missing stop is re-placed; if the price is already through its level (or the venue refuses it) THAT coin is closed,
   and the account only stays `DEGRADED` (no new entries) while the problem persists. `FLATTEN` is for account-wide causes only
   (drawdown / kill, positions or orders unreadable, repeated network errors with unconfirmable stops). A position whose mark is
   through its `liquidationPrice` (long mark ≤ liq, short mark ≥ liq) or whose isolated margin is ≤ 0 is *in liquidation*: no
   orders are sent for it, one Google Chat alert, re-checked every cycle (kv `liquidating`); once it vanishes the engine records
   the trade with `exit_reason: liquidation`. Emergency close order: cancel the symbol's reduce-only conditional orders (legacy
   and algo ids), reduce-only market close; on -2022 with the position still shown and not liquidating, ONE plain MARKET of exactly
   the freshly re-read quantity (one-way mode), else give up for that symbol. Every such action (guard flatten / no-stop close,
   engine invariant close, "beyond its stop" close, stop re-place, untracked-position close) backs off per symbol 5 s → 10 s → …
   5 min (reset on success), logs once per step and alerts once per symbol per incident. `BinanceExchange.market_order` raises
   `OrderNotFilled` (and cancels the order) when the venue acks a market order that never fills, instead of reporting the
   requested quantity as executed.
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
**Whitelist** (`universe_whitelist: list[str] | null`, default null = the ranking above): when set, the universe is exactly the
whitelist ∩ symbols TRADING on prod (signal data) and on the order venue, in whitelist order; `universe_size`, the ranking and the
volume / funding / listing filters are ignored. Changing it rebuilds the universe within ~5 s. Open positions in coins outside the
universe are still watched, priced, protected (stops, trailing, partials) and closed normally.

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
Trade if `prob_long ≥ prob_threshold_long` (default 0.70) or, when `short_enabled` (default on),
`prob_short ≥ prob_threshold_short` (default 0.70, equal to the long one). When both qualify, the larger margin over its OWN threshold
wins and an exact tie trades neither (`adaptive.pick_side`, also used for adaptive events). Thresholds are configurable and the Lab
reports which value is break-even after fees for the chosen exit mode.

### V3 engine (docs/V3_EXPERIMENTS.md is the model-side contract)
- **Scorer v2.** `Scorer.kind`, `Scorer.history_bars_5m`, `Scorer.score_universe(frames: {symbol: FrameBundle})`.
  `FrameBundle(symbol, kl, metrics, funding, depth, btc_kl)` (aliases `klines`, `btc`) = closed 5m bars + derivative frames.
  `scorer.load_model` dispatches on the artifact `kind`: `per_symbol_5m`/`mtf` -> `tb.model_np.load`, `gbm` ->
  `tb.model_gbm.load`, `xsec` -> `tb.model_xsec.load`; an unknown kind, a missing module or a broken artifact is logged and
  skipped. Classic per-symbol models are still scored through `score(symbol, window)` exactly as before.
- **History.** The engine keeps `max(720, history_bars_5m of every loaded model (+ ATR-timeframe / vol-stop need))` 5m bars per
  symbol: the first fetch takes 1500, then ONE older REST page (`endTime`) per symbol per bars cycle (max 3 symbols/cycle, none
  while the minute's request weight > 1200) until enough; a young listing marks itself exhausted. Metrics / funding lookback
  follows the same need (<= 28 days, incremental top-ups).
- **Decision points / limits** (`entry_cfg` of the model, `Settings` defaults per key: `decision_every_bars`, `cooldown_bars`,
  `max_trades_per_day`, `entry_mode`, `maker_timeout_s`, `maker_reprices`, `maker_on_timeout`): a model decides on the 5m bar that
  closes on a multiple of `decision_every_bars` on the UTC grid (12 = every full hour); cooldown counts from the coin's last entry
  or exit, max trades per UTC day counts entries (all coins). Restart-safe (seeded from the DB). Thresholds: classic models use the
  Settings, V3 kinds use the artifact's `thresholds` (Settings if absent).
- **Adaptive cadence** (`entry_cfg.decision_mode: "adaptive"`, `backend/tb/adaptive.py`; default `"fixed"` = the grid above). The model
  is SCORED on every closed 5m bar (signals recorded every bar) but a coin may only ENTER at a decision point: (a) REGULAR, `gap`
  bars after its last regular look, `gap = clamp(round(base_gap_bars * median_vol / current_vol), min_regular_gap_bars,
  max_regular_gap_bars)` (defaults 12 / 3 / 24; vol = std of the last 12 5m log-returns vs its trailing-7d median: volatile = looks
  more often); normal threshold. (b) EVENT on the coin's newest closed bar: 5m range >= `ev_range_atr` (1.5) x ATR14, volume
  z-score (48 bars) >= `ev_volume_z` (2.5), close beyond the prior closed 1h bar's high/low (`ev_breakout`), OI moved >=
  `ev_oi_jump_pct` (1.0 %) over 3 bars, or the side's probability rose >= `ev_prob_jump` (0.10) since the coin's last decision
  point; the entry then needs `prob >= threshold + ev_extra_conf` (0.05). `min_gap_bars` (1) spaces any two decision points of a
  coin. A threshold <= 0 switches that event off. The live champion and every shadow model call the same `AdaptiveState.step`
  (the shadow stats measure the live behaviour); state is in memory, first sight after a restart counts the previous
  `base_gap_bars` grid close as the last regular look. Event entries are logged (`adaptive entry ETHUSDT long: event
  volume_z=3.1, p=0.71`), regular decisions log the coin's current gap. `/api/status` `decisions` and `models[]` add
  `decision_mode`, the gap settings, `last_decision` {t, symbol, kind, action, prob, reason} and `decision_counts`. Training
  (`ml/mtf_dataset.py`) still labels on the fixed hourly grid.
- **Time stop / max_open.** `exit_cfg.horizon_bars` (any exit mode): the position is closed at market that many 5m bars after the
  entry (exit reason `timeout`; shadow: at that bar's close, adverse slippage, taker fee), persisted in kv `time_stops`.
  `entry_cfg.max_open`: concurrent positions allowed (live: all open positions + in-flight entries; shadow: that model's own).
- **Exits.** (profit protection: see above.) Only `exit_cfg.exit_mode = "pct"` (`tp_pct`/`sl_pct` in PERCENT, like `tb.labels`) or a non-5m `atr_tf` (`1h|4h|1d`; the SMA-ATR14
  of the last CLOSED bar of that timeframe, computed by `tb.labels.exit_atr` = exactly the labels' ATR) make the artifact drive the exits (`tb.exits`); classic `fixed`/`atr` on 5m stay the Settings
  template. Sizing is still risk-capped (`sl_usd`) and every cap applies. xsec artifacts use `stop_vol_mult` x trailing-7d daily
  vol (clipped `stop_min..stop_max`), stop only (legs close on the rebalance). `pct` positions and xsec legs are not trailed.
- **Maker entries** (`entry_mode: maker`). Post-only GTX LIMIT at the venue's best bid (long) / ask (short) (`/fapi/v1/ticker/bookTicker`);
  -5022 (would take) re-prices; every `maker_timeout_s` the order is cancelled and re-posted, at most `maker_reprices` times; the
  FIRST fill (partial or full) ends the attempt, the remainder is cancelled and the stop/TP are placed for exactly the filled
  quantity only AFTER the fill (never an unprotected resting position; a fill that arrives when entries are no longer allowed is
  closed at once). Nothing filled -> skip, or taker per `maker_on_timeout`. The order rests OUTSIDE the trade lock (kill / close_all /
  stop checks keep running); the in-flight order is reserved against every cap and blocks a duplicate. Stale `tbm...` resting orders
  are cancelled at engine start. Paper venue: fills only when the mark / a later bar trades strictly through the limit, at the limit,
  maker fee, no slippage, optional partial fills. Shadow: the same conservative `tb.sim.execution.fill_order` the labels use.
- **Shadow.** Every loaded model (any kind) is simulated at its own decision points with its own entry/exit config, gate and
  execution model into `shadow_trades`; xsec challengers open/close legs on their rebalance (exit reason `rebalance`).
  Champion = the existing promotion rule; "PAPER TEST ONLY" is refused in live for champions and challengers of every kind.
- **xsec champion.** On each `rebalance_every_h` slot (UTC grid, executed within 15 min of the boundary, first sight waits for
  the next boundary) the engine ranks the universe with `score_universe`, goes long the `top_k` highest `prob_long` and short the
  `bottom_k` highest `prob_short` (shorts need `short_enabled`; `0/0` scores = not tradable), closes legs no longer targeted
  (`exit_reason: rebalance`), keeps unchanged legs, opens new ones through `try_open(neutral=True)`: all caps, kill, guard state,
  derating and the per-leg exchange stop apply, only the correlation cap does not. The leg book persists in kv `xsec_book`.
- **Status.** `/api/status` adds `models[]` (name, kind, role, history, cadence, entry/exit mode, next_decision_t), `decisions`
  (next decision time, trades today vs max, cadence, history ready), `maker` (entries, filled, partial, timeouts, gtx_rejects,
  reprices, taker_fallback, skipped, aborted, fill_rate, in_flight), `xsec_legs`; `/api/shadow` rows add `kind`, `entry_mode`,
  `loaded`. `/api/settings` adds the seven entry-control keys above. `exit_reason` may also be `rebalance` or `timeout`.

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
- `GET /api/orderbook?symbol=` → `{symbol, t, mid, spread, bands:[{pct, bid_depth, bid_notional,
  ask_depth, ask_notional}], bids:[[price, qty]], asks:[[price, qty]]}`; bands at
  pct = 0.2, 1, 2, 3, 4, 5 are cumulative depth within ±pct % of mid, the same definition as the
  data.binance.vision bookDepth archive; bids/asks = top-20 ladder. Engine refreshes per universe
  symbol (REST `/fapi/v1/depth?limit=1000`, every ~5-10 s), best-effort, never blocks trading.
- `GET /api/signals?symbol=&limit=200` → `[{t, symbol, prob_long, prob_short,
  action:"long"|"short"|"none", reasons:[{feature,label,value,contribution}]}]`
- `GET /api/positions` → `[{id, symbol, side, entry_price, qty, notional, leverage, margin,
  sl_price, tp_price, trail_active, trail_price, mark_price, upnl, opened_at, prob, reasons}]`
- `GET /api/trades?limit=200` → `[{id, symbol, side, entry_price, exit_price, qty, opened_at,
  closed_at, pnl, fees, funding, exit_reason:"tp"|"sl"|"trail"|"kill"|"manual"|"liquidation"|"partial"|"timeout"|"rebalance",
  prob}]`
  **PnL convention (everywhere: sim, engine, API, UI):** `pnl` = NET realised PnL
  (gross − fees − funding). `fees` ≥ 0 is a cost. `funding` is a cost: positive = paid,
  negative = received. `win` means net pnl > 0. `win_rate` is a 0–1 fraction.
- `GET /api/shadow/equity?limit=5000` → `[{model, status, kind, points:[{t, pnl}]}]`: cumulative NET PnL (USDT) of each registry
  model's closed shadow trades (champion + challengers + retired; a model with no trades gets one `pnl: 0` point). `points[0]` is
  `{t: first entry (or stake start / now), pnl: 0}`, then one point per close time; only trades opened since the stake began
  (same cut-off as `/api/equity`). The BOTS page draws it with the account curve (`equity − start_equity`) in one chart.
- `GET /api/equity?limit=2000` → `[{t, equity}]`
- `GET /api/logs?limit=200` → `[{t, level, msg}]`
- `GET /api/settings` / `PUT /api/settings` (partial merge) → `{slice_margin_usd, leverage,
  max_positions, exit_mode, tp_usd, sl_usd, tp_atr, sl_atr, trail:{enabled, activate_r,
  distance_atr}, breakeven_r, sl_cooldown_bars, max_side_risk_pct, partial_tp_r, partial_tp_frac, min_tp_usd, min_sl_usd,
  max_net_exposure_pct, reversal_z, reversal_lock_r, reversal_block_bars, short_enabled, prob_threshold_long, prob_threshold_short,
  corr_cap:{max_corr, max_correlated}, universe_size, universe_refresh_s, universe_whitelist (null | [symbol]),
  kill_dd_pct, fee_taker, sound}`
- `POST /api/control` `{action:"start"|"stop"|"kill"|"reset_kill"|"close_position"|"close_all"|
  "pause_symbol"|"unpause_symbol"|"refresh_universe", position_id?, symbol?}` → `{ok, error?}`
- `GET /api/lab/reports` → `[{id, created_at, kind:"backtest"|"train", title, summary:{...},
  equity:[{t,equity}]}]`
- `WS /ws` → messages `{type, data}`; types: `status`, `tick` `{symbol,t,price}`, `candle`
  `{symbol,tf,candle}`, `signal`, `position_opened`, `position_updated`, `position_closed`,
  `universe`, `equity` `{t,equity}`, `log`, `kill`, `orderbook`.
