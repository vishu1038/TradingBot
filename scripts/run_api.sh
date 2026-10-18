#!/usr/bin/env bash
# API + UI (http://127.0.0.1:8787). Usage: scripts/run_api.sh [--port N]
set -euo pipefail
source "$(dirname "$0")/_env.sh"
cd "$ROOT/backend"
exec "$PY" -m tb.api "$@"
