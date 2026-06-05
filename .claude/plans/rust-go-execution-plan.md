# Plan: move order EXECUTION to a Rust/Go service (Python keeps research/ML)

## Context / why
Today everything is one Python process: data, features, RL/CEM training, the dashboard, AND
order execution (`engine/trading_engine.py` → `connectors/binance_futures.py`). Python is the
right tool for the ML/RL/research half, but the **execution hot-path** wants a compiled, low-
latency, reliable service:
- Tight, predictable latency on order place/cancel/replace (no GIL pauses, no training thread
  contention — the GIL-starvation we already hit with in-process training is the canary).
- Rock-solid exchange connectivity: WS order/position streams, REST with rate-limit budgeting,
  reconnect/resync, idempotent retries, clock-skew handling.
- A risk kill-switch that lives *close to the exchange* and can flatten independently of Python.
- Correct exchange arithmetic: min-notional, lot/tick step rounding, leverage, reduce-only,
  liquidation-aware sizing — the things the local sim can't model and the $100 target needs.

Keep Python for: strategies, RL/CEM training, the dashboard/EventBus, universe selection.

## Recommended split
**Python (unchanged role)** computes a *target position* per symbol (it already does — strategies
emit {-1,0,+1} → desired qty). It sends **desired state**, not raw orders.

**Execution service (Rust or Go)** owns the exchange. It:
1. Receives `set_target(symbol, target_qty, max_slippage, reduce_only)` from Python.
2. Diffs against the live exchange position (its own source of truth via user-data WS).
3. Places/cancels the minimal orders to converge, with lot/tick rounding + min-notional checks.
4. Streams fills/positions/errors back to Python for the dashboard and RL reward bookkeeping.
5. Enforces a local risk gate (max position, max drawdown, kill-switch → flatten) autonomously.

## Interface contract (the key design artifact)
A small, versioned API between Python and the service. Options, simplest first:
- **gRPC** (recommended): typed `.proto` shared by both sides; bidi stream for fills/positions.
  Rust `tonic` / Go `google.golang.org/grpc`. Python `grpcio`.
- REST + WebSocket (simpler to start, less typed).
- Local IPC (stdin/stdout JSON lines or a Unix/named pipe) for a first cut with no network dep.

Proto sketch:
```
service Execution {
  rpc SetTarget(TargetRequest) returns (Ack);          // desired position per symbol
  rpc Flatten(FlattenRequest) returns (Ack);           // kill-switch
  rpc StreamEvents(Empty) returns (stream ExecEvent);  // fills, position, errors, heartbeats
}
TargetRequest { string symbol; double target_qty; double max_slippage_bps; bool reduce_only; }
ExecEvent { oneof { Fill fill; Position pos; ExecError err; Heartbeat hb; } }
```

## Rust vs Go (pick one)
- **Go** — faster to ship: goroutines/channels map cleanly to "one WS reader + N symbol workers";
  great Binance SDK ecosystem (`go-binance`); GC pauses are sub-ms and irrelevant at our cadence
  (5m bars, not HFT). **Recommended** unless sub-100µs latency or zero-GC is a hard requirement.
- **Rust** — maximal safety + no GC; `tokio` + `binance-rs`/hand-rolled. More upfront cost
  (ownership, async ergonomics). Choose if this grows toward HFT or must be ultra-robust.
Given the bot is minutes-cadence, **Go is the pragmatic choice**; revisit Rust only if latency
becomes the bottleneck.

## Migration phases (incremental, never a big-bang rewrite)
1. **Contract first** — define the proto/JSON and a Python client stub. No behavior change.
2. **Shadow service** — Go/Rust service connects to **testnet**, receives targets, places orders,
   streams fills. Python keeps its local accounting in parallel and *reconciles* against the
   service's fills (detect drift). Run alongside the existing path.
3. **Cut over execution** — `TradingEngine` in `testnet`/`live` mode delegates fills to the
   service instead of calling `client.place_order` directly. Python's `_apply_fill` becomes
   "apply the fill the service reported." Local sim (`paper`) stays pure-Python for research.
4. **Move the risk gate** into the service (autonomous flatten/kill-switch).
5. **Harden** — rate-limit budgeter, reconnect/resync, idempotency keys, structured logging,
   metrics; only then consider real money behind the existing `ALLOW_LIVE_TRADING` gate.

## Touch points in this repo
- `engine/trading_engine.py` — `_place_real_order`/`_execute` become a thin client call in
  `live`/`testnet` modes (behind a flag like `--executor grpc://...`); `paper` unchanged.
- `connectors/binance_futures.py` — its order methods move (conceptually) into the service; the
  Python connector keeps market-data/history use.
- New top-level dir, e.g. `executor/` (Go module) or `executor-rs/` (Cargo crate), + the shared
  `proto/execution.proto`.

## Constraints
- Real money still gated by `ALLOW_LIVE_TRADING=true` + `confirm_live=True`; testnet first.
- Credentials live in env/.env only; never in code or the proto. The service reads its own keys.
- Don't start real-money routing until the testnet shadow phase shows zero reconciliation drift.
