# TradingBot v2 backend

Trading engine, safety guard, API and Binance exchange layer (`backend/tb/`). Contracts live in
`docs/ARCHITECTURE.md`. Python 3.14, `.venv` at the repo root.

## Processes (all separate, one SQLite WAL DB `var/tb.sqlite`)

| process | entry | role |
|---|---|---|
| engine | `python -m tb.engine` | polls 5m klines, scores bars, opens/manages trades, writes state + events + heartbeat, polls `commands` |
| guard | `python -m tb.guard` | independent safety monitor: own exchange client, never imports the engine; publishes `safety_state` |
| api | `python -m tb.api` | FastAPI on `127.0.0.1:8787`, reads SQLite only, writes `commands`; serves `ui/` at `/` |

Killing the API (or closing the browser) changes nothing for open trades: stops live on the exchange
(or in the paper simulation inside the engine).

## Run

```sh
scripts/dev.sh                       # engine + guard + API (+ UI at http://127.0.0.1:8787/), logs in var/
scripts/run_engine.sh [--mode paper_local|binance_demo|live] [--reset-paper]
scripts/run_guard.sh  [--mode ...]   # keep the mode identical to the engine's
scripts/run_api.sh    [--port 8787]
scripts/install_launchd.sh [install|uninstall]   # KeepAlive launchd agents (templates in deploy/)
cd backend && ../.venv/bin/python -m pytest -q   # tests
```

Env (`.env` at the repo root, never logged): `BINANCE_API_KEY`, `BINANCE_API_SECRET`,
`GOOGLE_CHAT_WEBHOOK_URL` (alerts; unset = no alerts), `TB_MODE`, `TB_DB`, `TB_MODELS`, `TB_API_PORT`.

### Modes
- `paper_local` (default): real prod market data, simulated fills/fees/funding/liquidation. No keys.
  Starts trading immediately (`running=true`) with `paper_start_equity` (100 USD).
- `binance_demo`: real orders on `demo-fapi.binance.com` (or `BINANCE_ENV=testnet`). Needs keys. Starts
  **stopped**; send `start`. The engine refuses entries unless the guard process is alive.
- `live`: needs `ALLOW_LIVE_TRADING=true` in the env AND `--confirm-live` (or `TB_CONFIRM_LIVE=true`). The
  first 20 live trades run at 0.5x size. Starts stopped.

Only one engine can run per DB (flock on `var/tb.sqlite.engine.lock`).

## Engine behaviour in short
- Universe refresh hourly (hot list); watched symbols = universe + open positions + BTC (reference).
- On each closed 5m bar: plausibility check -> features -> champion score -> signal row/event -> entry if
  threshold met, engine running, no kill, safety state NORMAL, symbol not fault-excluded, positions < cap,
  correlation cap OK, derated size above min notional.
- Entry fills, then the reduce-only STOP_MARKET (MARK_PRICE) is placed at once; if that fails the position is
  closed at market. TP is also an exchange order (the engine closes at TP itself if that order is missing).
- Every loop: no open position without a live stop (else close); trailing stop only tightens; kill switch at
  drawdown >= `kill_dd_pct` (paper 50, demo/live capped at 25) closes everything until `reset_kill`.
- A symbol leaving the universe never force-closes its trade.
- Conditional orders use `/fapi/v1/order`; on error -4120 (algo switch) the client moves to
  `/fapi/v1/algoOrder` permanently. Order ids are namespaced strings `o:<id>` / `a:<id>`.

## Safety layer
- `safety_state` NORMAL -> DEGRADED -> FLATTEN -> HALTED (HALTED only cleared by `reset_kill`).
- Plausibility checks reject bad bars/ticks; faults mature after 3 consecutive cycles and clear after 5 good
  ones (`fault_mature_n`, `fault_recover_m`).
- `risk.derate_multiplier(dd_pct, kill_dd_pct)`: 1.0 until 40% of the kill limit, 0.5 at 80%, 0 at 100%.
- Drift (z of realised vs predicted win rate over the last 30 trades): z < -2 size x0.5, z < -3 entries blocked,
  clears at z > -1. Entry slippage > 2x the assumption: alert + size x0.5.
- Shadow mode: `models/*.json` `status` (champion/challenger/retired); challengers (and the champion) are
  simulated each bar into `shadow_trades`. `promote_model` needs >= 50 shadow trades, higher expectancy than
  the champion over the same period, no drift alarm. `rollback_model` restores the previous champion.

## API notes (shapes follow the contract in docs/ARCHITECTURE.md)
- `trades[].pnl` is the **gross** price PnL; `fees` is a positive cost; `funding` is a signed cashflow
  (+ received). Net = `pnl - fees + funding`. `status.realized_pnl` is net.
- `status.win_rate` is a 0..1 ratio (net-positive trades / all trades).
- `POST /api/control` also takes `model` (for `promote_model` / `rollback_model`). It waits up to 4 s for the
  engine's answer; `{ok:true, queued:true}` means the engine had not answered yet. If the engine heartbeat is
  older than 15 s the command is **not** queued (`ok:false`).
- `/ws`: first message is `status`, then the tail of the `events` table (`tick`, `candle`, `signal`,
  `position_opened`, `position_updated`, `position_closed`, `universe`, `equity`, `log`, `kill`, `safety`,
  `shadow_trade`); `status` events are expanded to the full status object.
