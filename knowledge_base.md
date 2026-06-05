# TradingBot — Knowledge Base

A cryptocurrency trading bot written in Python with a Tkinter desktop GUI. It connects to two
crypto-derivatives exchanges — **Binance Futures** and **BitMEX** — to stream live prices, view
account balances, manage a symbol watchlist, and (via the connector layer) place/cancel/track
orders.

- **Repository:** https://github.com/vishu1038/TradingBot.git
- **Language:** Python 3.10 (`.pyc` artifacts are `cpython-310`)
- **GUI:** Tkinter (standard library)
- **Status:** Work-in-progress / educational project. Watchlist + live price streaming +
  logging UI are functional. Order placement exists in the connectors but is **not yet wired
  to the GUI**; the trades table is display-only.

---

## 1. High-Level Architecture

```
                          main.py  (entry point, logging setup)
                             │
                 ┌───────────┴────────────┐
                 ▼                         ▼
   BinanceFuturesClient          BitmexClient          ← connectors/ (REST + WebSocket)
                 │                         │
                 └───────────┬─────────────┘
                             ▼
                     Root (tk.Tk)                       ← interface/root_component.py
                             │   (orchestrates UI, 1.5s refresh loop)
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                   ▼
      Watchlist           Logging            TradesWatch       ← interface/ components
   (live bid/ask)      (log console)       (trade table)
                             │
                             ▼
                         models.py                            ← data classes shared by all
            (Balance, Candle, Contract, OrderStatus)
```

**Data flow:**
1. `main.py` instantiates both exchange clients with API keys (testnet enabled).
2. Each client, on construction, fetches contracts + balances over REST, then spawns a
   **background thread** running a WebSocket that continuously updates `self.prices`.
3. `Root` builds the UI and starts a recurring `after(1500, ...)` loop that pulls the latest
   prices/logs from the clients and pushes them into the widgets.

---

## 2. Directory / File Map

| Path | Role |
|------|------|
| `main.py` | Entry point. Configures logging (stream + `info.log` file handlers), creates the two exchange clients, launches the Tk `Root` mainloop. **Contains hard-coded testnet API keys.** |
| `models.py` | Exchange-agnostic data classes (`Balance`, `Candle`, `Contract`, `OrderStatus`) plus the `tick_todecimals()` helper. Each class normalizes raw JSON from either exchange into a common shape. |
| `connectors/binance_futures.py` | `BinanceFuturesClient` — REST + WebSocket client for Binance Futures. |
| `connectors/bitmex.py` | `BitmexClient` — REST + WebSocket client for BitMEX. |
| `interface/root_component.py` | `Root(tk.Tk)` — top-level window; lays out the three UI frames and runs the periodic UI-update loop. |
| `interface/watchlist_component.py` | `Watchlist(tk.Frame)` — add/remove symbols per exchange; live bid/ask table. |
| `interface/logging_component.py` | `Logging(tk.Frame)` — scrolling, read-only log console. |
| `interface/trades_component.py` | `TradesWatch(tk.Frame)` — table of trades (time/exchange/strategy/side/qty/status/pnl). Display scaffold only. |
| `interface/styling.py` | Shared color and font constants. |
| `info.log` | Runtime log file produced by the file handler. |
| `.idea/` | JetBrains (PyCharm) project metadata. |
| `README.md` | One-line project description. |

---

## 3. Data Model (`models.py`)

All classes take a raw `info`/`*_info` dict plus an `exchange` string (`"binance"` or
`"bitmex"`) and branch on it to populate a uniform set of attributes.

- **`Balance`** — `initial_margin`, `maintenance_margin`, `margin_balance`, `wallet_balance`,
  `unrealized_pnl`. BitMEX values are scaled by `BITMEX_MULTIPLIER = 1e-8` (satoshi → BTC).
- **`Candle`** — OHLCV (`open/high/low/close/volume`) + `timestamp` (ms epoch). BitMEX
  timestamps are parsed from ISO strings and shifted back by one timeframe (`BITMEX_TF_MINUTES`)
  to align bucket-start convention with Binance.
- **`Contract`** — `symbol`, `base_asset`, `quote_asset`, `price_decimals`,
  `quantity_decimals`, `tick_size`, `lot_size`. Binance derives tick/lot size from precision
  fields; BitMEX reads them directly and uses `tick_todecimals()` to count decimals.
- **`OrderStatus`** — `order_id`, `status`, `avg_price`.
- **`tick_todecimals(tick_size)`** — formats a float to 8 decimals, strips trailing zeros, and
  returns the number of significant decimal places.

---

## 4. Connector Layer

Both clients follow the same structure, differing mainly in REST endpoints, auth signing, and
WebSocket message shape.

### Common pattern
- `__init__(public_key, secret_key, testnet)` — selects base/WSS URLs by `testnet` flag,
  fetches `contracts` and `balances`, initializes `self.prices = {}` and `self.logs = []`,
  then starts the WebSocket thread.
- `_make_request(method, endpoint, data)` — wraps `requests` GET/POST/DELETE with
  try/except + status-code checking; returns parsed JSON or `None`.
- `_add_log(msg)` — logs and appends `{"log": msg, "displayed": False}` so the GUI can render
  it once.
- `_start_ws()` — runs `WebSocketApp.run_forever()` in a loop with reconnect-on-error and a 2s
  backoff. Callbacks: `_on_open` (subscribes to a channel), `_on_close`, `_on_error`,
  `_on_message` (updates `self.prices`).
- REST trading methods: `get_contracts`, `get_balances`, `get_historical_candles`,
  `place_order`, `cancel_order`, `get_order_status`.

### Binance-specific (`binance_futures.py`)
- Base: `testnet.binancefuture.com` / `fapi.binance.com`; WSS `stream.binancefuture.com/ws`.
- Auth: HMAC-SHA256 over URL-encoded params → `signature`, plus `X-MBX-APIKEY` header.
- Has `get_bid_ask()` for an on-demand REST order-book snapshot.
- Subscribes to the `bookTicker` channel; `_on_message` reads `data['e'] == "bookTicker"`.
- Endpoints: `/fapi/v1/exchangeInfo`, `/klines`, `/ticker/bookTicker`, `/account`, `/order`.

### BitMEX-specific (`bitmex.py`)
- Base: `testnet.bitmex.com` / `www.bitmex.com`; WSS `.../realtime`.
- Auth: HMAC-SHA256 over `method + endpoint + ?query + expires`, sent via
  `api-key` / `api-signature` / `api-expires` headers (5s expiry window).
- Subscribes to the `instrument` topic; `_on_message` reads `data['table'] == "instrument"`.
- Endpoints: `/api/v1/instrument/active`, `/user/margin`, `/trade/bucketed`, `/order`.

---

## 5. GUI Layer (`interface/`)

- **`Root`** — `tk.Tk` window titled "Trading Bot". Splits into a left frame (Watchlist +
  Logging stacked) and a right frame (TradesWatch). `_update_ui()` runs every **1500 ms**:
  flushes undisplayed logs from both clients into the log console, then refreshes each
  watchlist row's bid/ask using the matching exchange's `contracts`/`prices` (formatted to the
  contract's `price_decimals`). Wrapped in try/except for `RuntimeError` (dict mutated during
  iteration when rows are added/removed).
- **`Watchlist`** — Two entry boxes (Binance / BitMEX). Pressing `<Return>` validates the typed
  symbol against the known contract list and, if valid, adds a row via `_add_symbol()`. Rows
  hold symbol/exchange labels, bid/ask `StringVar`-backed labels, and a red "x" remove button.
  Rows are keyed by an incrementing `_body_index`; `_remove_symbol()` does `grid_forget()` +
  `del`.
- **`Logging`** — Read-only `tk.Text`; `add_log()` prepends a timestamped line.
- **`TradesWatch`** — Builds a 7-column header (time/exchange/strategy/side/quantity/status/pnl).
  `add_trade(data)` adds a row. **Not yet called anywhere** — trading is not connected to the UI.
- **`styling.py`** — Dark theme: `BG_COLOR="gray12"`, accent `#1c2c5c`, white/`SteelBlue1` text,
  Calibri 11 fonts.

---

## 6. Running the Project

### Prerequisites
- Python 3.10+ with Tkinter available (bundled on Windows/macOS python.org installers; on Linux
  may need `python3-tk`).
- Dependencies (no `requirements.txt` in repo — install manually):
  ```
  pip install requests websocket-client python-dateutil
  ```
  (`hmac`, `hashlib`, `json`, `threading`, `tkinter`, `typing` are standard library.)

### Run
```
python main.py
```
This opens the GUI. It connects to **testnet** by default (`testnet=True` in `main.py`).

---

## 7. Notable Observations / Gotchas

- **⚠️ Hard-coded API keys** in `main.py` (testnet keys, but still should be moved to env vars
  / config and rotated). Do not commit production keys.
- **`bitmex.py` bug (line 34):** `self_ws = None` is a typo for `self._ws = None`; it creates a
  local variable that is discarded. `self._ws` is only set later in `_start_ws()`, so a very
  early `subscribe_channel` call could `AttributeError`. Binance correctly uses `self._ws = None`.
- **`subscribe_channel` logging (bitmex.py line 231):** format string has more `%s` placeholders
  than arguments.
- **Trading is not wired to the UI** — `place_order`/`cancel_order` exist in connectors and
  `TradesWatch.add_trade` exists, but nothing connects user actions to order placement. No
  strategy engine yet (the "strategy" column is a placeholder).
- **No tests, no `requirements.txt`, no `.gitignore`** for `__pycache__`/`info.log` (these are
  committed in the repo).
- **`get_order_status` (bitmex.py):** iterates orders to find a match but then returns
  `order_status[0]` (the first order) rather than the matched `order` — likely a bug.

---

## 8. Roadmap → Self-Learning Automated Trading Bot

**End goal:** an automated bot that learns from historical data, takes paper trades on
testnet, uses a reward/penalty (reinforcement-learning) loop to actively adjust its strategy
for higher ROI, exposes a GUI for local start/stop + monitoring, and pushes performance alerts
to the user's phone — with the success criterion being a consistently profitable paper-trading
account.

> **Reality check (read first):** The hard part is *not* the AI library — it's avoiding
> overfitting, modeling transaction costs/slippage honestly, and surviving market
> non-stationarity (a policy trained on one regime decays in the next). Paper-trading profit
> does **not** guarantee live profit. Treat every backtest result as suspect until validated
> walk-forward and out-of-sample. The connectors already support `testnet=True`, and
> **Binance/BitMEX testnet accounts ARE paper-trading accounts** (fake money, real market
> data) — so the learning loop can run with zero financial risk.

### Phase 0 — Foundation / hygiene (prerequisite)
1. Move credentials to environment variables or a `.env` (with `python-dotenv`); rotate the
   committed testnet keys.
2. Add `requirements.txt` and `.gitignore` (`__pycache__/`, `info.log`, `.env`, model
   checkpoints, the SQLite DB).
3. Fix the `self_ws` typo and the `get_order_status` return bug in `bitmex.py`.
4. Add a persistence layer (**SQLite**) for trades, equity curve, and per-episode metrics
   (extends the existing "trades table" idea). One `data/` table per: trades, candles cache,
   model checkpoints metadata.

### Phase 1 — Data & backtesting infrastructure
5. **Historical data pipeline:** use existing `get_historical_candles()` to bulk-download and
   cache OHLCV to SQLite/Parquet. Add pagination (Binance caps 1000 candles/req).
6. **Feature engineering:** compute technical indicators (RSI, MACD, EMA, Bollinger, ATR,
   returns, volatility, order-book imbalance) with `pandas` + `pandas-ta` or `ta`.
7. **Backtester / replay engine:** deterministically replay cached candles bar-by-bar through
   a strategy, applying realistic **fees + slippage**. This is the single most important piece
   for trustworthy results. Support walk-forward (train on window N, test on N+1).

### Phase 2 — Strategy abstraction & manual trading
8. Define a `Strategy` base class (`on_candle()` / `on_tick()` → signal) so rule-based, ML, and
   RL strategies are interchangeable. Wire `place_order` → `TradesWatch.add_trade` so trades
   appear in the UI.
9. **Risk manager:** position sizing, stop-loss/take-profit, max-drawdown kill-switch,
   max-open-positions. This guards every strategy regardless of how "smart" it thinks it is.
10. Ship a simple baseline strategy first (e.g. EMA crossover) end-to-end on testnet. This
    validates the whole pipeline before any AI is added.

### Phase 3 — The learning agent (see §9 for AI options)
11. Wrap the backtester as a **Gymnasium environment** (state → action → reward).
12. Train an RL agent (start with the FinRL / Stable-Baselines3 stack), validate walk-forward,
    then run it live on testnet with online/periodic retraining ("active adjustment").
13. Add online learning: retrain or fine-tune on a rolling window (e.g. nightly) so the agent
    adapts to new regimes.

### Phase 4 — GUI control & phone alerts
14. **GUI:** Start/Stop bot button, strategy selector + hyperparameters, live equity-curve
    chart (`matplotlib` embedded in Tk), open-positions panel, and a "training status" panel.
15. **Phone alerts:** push notifications on fills, daily PnL summary, drawdown breaches, and
    bot start/stop. **Recommended channel: a Telegram bot** (free, trivial API, two-way
    control possible). Alternatives: Pushover, Discord webhook, Twilio SMS.
16. **(Optional) two-way control:** Telegram commands to pause/resume the bot or query status
    remotely.

### Suggested target structure
```
TradingBot/
├── connectors/            # existing
├── interface/             # existing GUI + new control/chart panels
├── strategies/            # base.py, ema_cross.py, ml_strategy.py, rl_strategy.py
├── data/                  # data_manager.py (download/cache), database.py (SQLite)
├── backtesting/           # engine.py (replay + fees/slippage), metrics.py (Sharpe, DD)
├── learning/              # env.py (Gym), train.py, agent checkpoints
├── risk/                  # risk_manager.py
├── alerts/                # telegram_bot.py
├── models.py              # existing
└── main.py
```

### Implemented so far (this scaffold)
Phases 0–2 are scaffolded and runnable. Status:

| Item | File(s) | Status |
|------|---------|--------|
| Env-based config (no hard-coded keys) | `config.py`, `.env.example` | ✅ |
| `requirements.txt`, `.gitignore` | — | ✅ |
| BitMEX bug fixes (`self_ws`, `get_order_status`, log fmt) | `connectors/bitmex.py` | ✅ |
| SQLite persistence (candles/trades/equity) | `data/database.py` | ✅ |
| Historical data download + cache | `data/data_manager.py` | ✅ |
| Indicator feature engineering | `strategies/features.py` | ✅ |
| Strategy base class | `strategies/base.py` | ✅ |
| Baseline EMA-crossover strategy | `strategies/ema_cross.py` | ✅ |
| Supervised-ML strategy (GBM) | `strategies/ml_strategy.py` | ✅ |
| Backtester (fees/slippage + walk-forward) | `backtesting/engine.py` | ✅ |
| Performance metrics | `backtesting/metrics.py` | ✅ |
| Risk manager (sizing + drawdown kill switch) | `risk/risk_manager.py` | ✅ |
| End-to-end demo runner | `run_backtest.py` | ✅ |
| Wire strategies into GUI + live testnet executor | — | ⬜ Phase 2/3 |
| RL agent + Gym env | `learning/` | ⬜ Phase 3 |
| Telegram phone alerts | `alerts/` | ⬜ Phase 4 |

**Run it:** `pip install -r requirements.txt` then `python run_backtest.py` (offline synthetic
demo) or `python run_backtest.py BTCUSDT 1h` (real testnet data, requires keys in `.env`).
The synthetic run is *expected* to show ~0/negative net return after costs — random-walk data
has no edge to find; it only proves the pipeline, cost model, and walk-forward guards work.

---

## 9. AI Integration Options (answer to "what AI can improve this bot?")

Ordered from most-reliable / least-effort to most-ambitious. A real system usually **combines**
several layers rather than betting everything on one model.

### A. Supervised ML for signal prediction *(recommended first AI step)*
- **What:** Predict next-bar direction (up/down/flat) or expected return from engineered
  features. More stable and debuggable than RL; gives you a working signal quickly.
- **Models:** Gradient-boosted trees (**XGBoost / LightGBM**) on tabular indicator features —
  strong baseline. Or sequence models (**LSTM / Temporal CNN / small Transformer**) on raw
  OHLCV windows when you have lots of data.
- **Output:** a probability/score the strategy and risk manager consume for entries and sizing.

### B. Reinforcement Learning *(this is the reward/penalty loop you asked for)*
- **Framing (MDP):**
  - **State:** window of OHLCV + indicators + current position + unrealized PnL + time features.
  - **Action:** discrete `{long, short, flat}` (start here) or continuous position size `[-1,1]`.
  - **Reward:** change in portfolio value, **minus** transaction-cost & holding penalties, often
    shaped toward **risk-adjusted return** (differential Sharpe / Sortino) and penalizing
    drawdown — this is the "penalty" half of your reward/penalty design.
- **Algorithms:** **DQN** (discrete actions), **PPO / A2C / SAC** (Stable-Baselines3) for more
  stability or continuous sizing. **Contextual bandits** are a simpler stepping stone if full RL
  is too noisy.
- **Libraries:** **FinRL** (finance-specific RL framework — closest fit to this project),
  **Stable-Baselines3 + Gymnasium** (the general engine underneath), `quantstats` for tear
  sheets.
- **Caveat:** RL is sample-hungry and overfits to backtests easily. Always validate
  walk-forward and on a held-out period; expect to discard many runs.

### C. LLM integration (Claude API) — for the things numbers can't do
LLMs should **not** make tick-level trade decisions (too slow/costly, not numeric). But they add
real value at the edges:
- **News / sentiment as a feature:** periodically fetch headlines or social posts and have
  Claude score sentiment per asset → feed that score in as an extra model feature.
- **Natural-language alerts:** turn raw metrics into the human-readable phone summaries
  ("Bot is up 2.3% today, 6/9 trades profitable, max drawdown 1.1%"). Great fit for the
  Telegram alerts in Phase 4.
- **Trade explanations:** "explain why the bot opened this position" for the GUI / logs —
  useful for trust and debugging.
- **Strategy config via chat:** describe a strategy in English; Claude emits the parameter set.
- Use the **Anthropic Python SDK** with the latest model (e.g. `claude-opus-4-8` for analysis,
  `claude-haiku-4-5` for cheap high-frequency sentiment scoring).

### D. Hybrid / ensemble *(realistic production shape)*
- **ML signal (B-A) → RL or rules for sizing/risk → LLM for sentiment feature + alerts.** Each
  layer does what it's best at; the risk manager (Phase 2) sits on top of all of them as a
  hard safety floor.

### Honest expectation setting
- Begin with **supervised ML + a solid backtester**, prove an edge survives fees and
  walk-forward validation, *then* layer RL on top for adaptive sizing.
- Define "making money" concretely up front: positive return **and** acceptable Sharpe / max
  drawdown over a multi-week out-of-sample paper run — not a single lucky backtest.
- Keep the kill-switch and paper-only constraint until results are stable across regimes.

---

## 10. Git History (most recent first)

```
0e2a1ef  Added the trades table
a9b6a28  Adding Updating Prices in watchlist
44b3036  Developed Add symbol method
b1b829a  Added watchlist component
9aeee64  Create watchlist_component.py
```

The commit history shows the project was built incrementally: connectors first, then the
watchlist UI, live price updates, and finally the (display-only) trades table.
