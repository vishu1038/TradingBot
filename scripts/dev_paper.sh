#!/usr/bin/env bash
# Second, independent stack: paper trading on REAL Binance prices (paper_local) next to the testnet bot.
# Own DB (var/tb_paper.sqlite), own API/UI port (8788), own logs (var/paper_{engine,guard,api}.log).
# Ctrl-C stops all three; only the engine dying ends this script (same rule as dev.sh).
set -uo pipefail
source "$(dirname "$0")/_env.sh"
cd "$ROOT/backend"
export TB_DB="${TB_DB:-$ROOT/var/tb_paper.sqlite}"
export TB_API_PORT="${TB_API_PORT:-8788}"
export TB_MODE=paper_local
pids=()
cleanup() {
  trap - INT TERM EXIT
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
}
trap cleanup INT TERM EXIT
"$PY" -m tb.engine --mode paper_local >>"$ROOT/var/paper_engine.log" 2>&1 &
pids+=($!)
"$PY" -m tb.guard --mode paper_local >>"$ROOT/var/paper_guard.log" 2>&1 &
pids+=($!)
"$PY" -m tb.api >>"$ROOT/var/paper_api.log" 2>&1 &
pids+=($!)
echo "paper engine pid ${pids[0]}, guard ${pids[1]}, api ${pids[2]}  UI: http://127.0.0.1:${TB_API_PORT}/  DB: $TB_DB"
while kill -0 "${pids[0]}" 2>/dev/null; do sleep 2; done
echo "paper engine exited; stopping the rest (see var/paper_engine.log)" >&2
exit 1
