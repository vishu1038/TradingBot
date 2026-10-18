#!/usr/bin/env bash
# Trading engine. Usage: scripts/run_engine.sh [--mode paper_local|binance_demo|live] [--confirm-live] [--reset-paper]
set -euo pipefail
source "$(dirname "$0")/_env.sh"
cd "$ROOT/backend"
exec "$PY" -m tb.engine "$@"
