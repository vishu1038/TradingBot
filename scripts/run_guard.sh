#!/usr/bin/env bash
# Independent safety guard (own exchange client, never imports the engine).
# Usage: scripts/run_guard.sh [--mode paper_local|binance_demo|live] [--confirm-live]
set -euo pipefail
source "$(dirname "$0")/_env.sh"
cd "$ROOT/backend"
exec "$PY" -m tb.guard "$@"
