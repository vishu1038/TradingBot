# TradingBot — session resume notes (2026-06-05)

Drop this in a fresh Claude Code conversation opened in this repo to continue. The global
project memory (`MEMORY.md` + `tradingbot-project`, `claude-cli-advisor-no-api-key`) also
auto-loads. Source of truth for architecture is `knowledge_base.md`.

## Where we are
Branch `feature/automated-trading-scaffold` (uncommitted work being committed alongside this
file). The bot is a multi-symbol live/paper dashboard (`run_live.py`) with continuous RL
retraining that hot-swaps policies into live engines.

### Done & committed earlier
- **EMA base strategy de-churned** (`strategies/ema_cross.py`, commit `d653185`, pushed):
  volatility deadband + hysteresis (`band_mult=1.0`, `trend_filter=0`). On real Binance 1m data
  this cut summed loss from −91% to −9% and trades ~80%.
- **Multi-core training offload + MLP policy + Claude CLI advisor** (commit `fda6a32`, pushed).

### Done this session, IN the commit that includes this file
1. **5-minute default timeframe** (`run_live.py --timeframe` default `5m`). Validated on real
   data: base strategy −6% (1m) → +16.5% (5m). 15m looked better (+69%) but only 4 trades →
   not trusted. 5m is the sweet spot (good economics + enough activity).
2. **Active-token universe selector** `select_active_universe()` — ranks Binance USDⓈ-M perps by
   `abs(24h %chg) * sqrt(quoteVolume)` with a liquidity floor. Flags `--auto-universe` /
   `--universe-size` pick the hot set at startup. Verified live (returned ZEC/HYPE/WLD/etc.).
3. **Live token CYCLING** `UniverseManager` (`run_live.py`) — every `--cycle-interval` (default
   1800s) re-ranks and adds/removes engines. Decisions: **fully dynamic (no anchors)**, **size 8**,
   anti-thrash via top-`ceil(size*1.5)` retention buffer + 1-cycle min-dwell. `MultiEngineController`
   made mutation-safe (RLock around `get_state`/add/remove) and **equity-continuous** (retired
   engines' equity/PnL banked into `_retired_equity/_realized` so the portfolio total never jumps).
   `make_engine` factory extracted for reuse. Flag `--cycle-universe` (live-only, needs trainer).
4. **RL churn penalty** (`learning/env.py` `TradingEnv.churn_penalty`, threaded through
   `learning/train.py` `train_policy` and `ContinuousTrainer`). **Default 0.0 (off)** — a single
   SOL OOS split was inconclusive (CEM seed-noise dominated; see below). Tunable via `--churn-penalty`.
5. **TESTNET execution mode** (`engine/trading_engine.py` + `run_live.py --testnet`) — see next.

## ⏭️ NEXT TASK (was in progress): real paper trading on Binance testnet
User: *"do paper trading on the binance platform itself not locally."* Implemented a `testnet`
engine mode that places **real orders on the Binance Futures testnet** (fake money, but real
fills/slippage/min-notional/lot-step/leverage/liquidation — exactly the realism gap for the
$100 goal). Code is in place and compiles but is **UNTESTED** because it needs credentials.

**Blocker (user action required — assistant must NOT create/handle API keys):**
1. Create Binance **Futures testnet** API keys at https://testnet.binancefuture.com/ (log in
   with GitHub/Google → API Key).
2. Put them in `.env` (gitignored): `BINANCE_PUBLIC_KEY=...`, `BINANCE_SECRET_KEY=...`,
   `USE_TESTNET=true`.
3. Run: `python run_live.py --testnet --cycle-universe --no-browser` (testnet was reachable on
   the user's current network: ping 200).

**To verify once keys are in:** confirm engines start in `testnet` mode, an order actually fills
on testnet (check the testnet account UI), `/api/state` stays responsive, and min-notional
rejections are handled gracefully.

**Known design tension to discuss with user:** dynamic hot-token cycling wants PROD market
activity, but testnet has its own thin/synthetic books and a smaller symbol list. A symbol hot
on prod may be illiquid/absent on testnet. Recommendation: for testnet, prefer a small fixed set
of liquid pairs (BTC/ETH/SOL…) that exist on testnet, OR accept that cycling ranks testnet
activity. Local-paper-on-prod-data remains the better venue for the dynamic-cycling research;
testnet is for execution realism / pre-real-money validation. Decide which the user wants.

## $100 demo capital (user decision 2026-06-05)
Since testnet is demo money, train/run the demo at the real-money target size: **$100 per-symbol
capital**. Implemented: `--testnet` defaults `--capital` to 100 (override allowed). NUANCE to
handle in the testnet impl: 8 cycling engines each *locally* assume their own $100, but they share
ONE real testnet margin account — so for a faithful $100 *total* test, run a small set (1–3
symbols) or add per-symbol sub-accounting/margin reconciliation. With size-8 cycling, treat each
engine's $100 as independent demo sizing (fine for learning; not a single-$100-account model).

## Rust/Go execution-layer rewrite (planned — see `.claude/plans/rust-go-execution-plan.md`)
User wants order EXECUTION moved out of Python into a compiled service (latency/reliability/
exchange-arithmetic; Python keeps strategies/RL/dashboard). Plan written: Python sends *target
positions*, a **Go (recommended) or Rust** service owns the exchange (place/cancel, fills WS,
min-notional/lot rounding, autonomous risk kill-switch) over a gRPC/JSON contract. Migration is
phased: contract → testnet shadow w/ reconciliation → cut over `_place_real_order` → move risk
gate → harden. Not started; it's the next architecture epic after testnet is verified.

## GPU/tensor dual-mode training (planned — see `.claude/plans/gpu-tensor-training-plan.md`)
User wants training to auto-detect PC specs and shift onto GPU tensors when available, else stay
on numpy CPU. Plan written: a `learning/backend.py` `xp` abstraction (numpy | CuPy | torch),
auto-select by GPU/CPU/RAM, tensor-batched CEM first (same algo, big speedup), optional PPO later.
Key caveat: torch/CuPy may lack Python 3.14 wheels → likely run GPU training in a separate 3.11/12
venv as the worker interpreter (trainer already uses worker processes). numpy stays the guaranteed
default. Not started.

## Pending / also-asked
- User asked to "open a terminal and give live model updates." Not yet done. Easiest: launch
  `python run_live.py --cycle-universe --no-browser` in the background and stream training-cycle /
  hot-swap / universe-rotation log lines. (An OLD-code instance PID was still on :8765 — start a
  fresh one, possibly on a different `--port`, after stopping the old.)
- Verify the live cycling end-to-end (short smoke: `--cycle-interval 120`): engines spin up/down,
  equity doesn't jump on removal, trainer auto-trains added symbols.
- Real-money realism still unmodeled in the LOCAL backtester/engine (min-notional/lot/leverage/
  liquidation) — testnet is the answer to that; local sim won't transfer to $100 faithfully.

## Hard constraints (carry forward)
- Real-money LIVE orders require BOTH env `ALLOW_LIVE_TRADING=true` AND `confirm_live=True`.
  Testnet mode deliberately bypasses that gate (fake money) but only ever hits testnet endpoints.
- Never expose port 8765 to the open internet (no auth on the dashboard).
- Assistant must not guess/transmit credentials; user sets keys in `.env` themselves.
- Python 3.14 (`C:\Program Files\Python314\python.exe`); numpy/pandas/sklearn only (no gym/SB3).

## Churn-penalty validation (for reference; inconclusive)
SOL, train[:6000]/test[6000:] OOS, one CEM seed each:
`churn 0.0→15 trades/−14.5% · 0.0005→21/−15.8% · 0.001→104/−18.9% · 0.002→15/−14.1% · 0.004→103/−9.7%`
Non-monotonic = seed noise dominates. Left off by default; 5m already reduces structural churn.
