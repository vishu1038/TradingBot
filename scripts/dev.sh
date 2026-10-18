#!/usr/bin/env bash
# Run engine + guard + API (the API also serves ui/ at /) together; Ctrl-C stops all three.
# Extra args (e.g. --mode binance_demo) go to the engine and the guard. Mode otherwise comes from TB_MODE.
# Logs: var/engine.log, var/guard.log, var/api.log
set -uo pipefail
source "$(dirname "$0")/_env.sh"
cd "$ROOT/backend"
pids=()
cleanup() {
  trap - INT TERM EXIT
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
}
trap cleanup INT TERM EXIT
"$PY" -m tb.engine "$@" >>"$ROOT/var/engine.log" 2>&1 &
pids+=($!)
"$PY" -m tb.guard "$@" >>"$ROOT/var/guard.log" 2>&1 &
pids+=($!)
"$PY" -m tb.api >>"$ROOT/var/api.log" 2>&1 &
pids+=($!)
echo "engine pid ${pids[0]}, guard pid ${pids[1]}, api pid ${pids[2]}"
echo "UI + API: http://127.0.0.1:${TB_API_PORT:-8787}/   logs: $ROOT/var/{engine,guard,api}.log"
# The UI/API must never be able to stop trading: only the engine dying ends this script.
# (portable bash 3.2 polling; no `wait -n`)
reported=""
while kill -0 "${pids[0]}" 2>/dev/null; do
  for p in "${pids[1]}" "${pids[2]}"; do
    if ! kill -0 "$p" 2>/dev/null && [[ "$reported" != *" $p "* ]]; then
      reported="$reported $p "
      echo "process $p (guard/api) exited; the engine keeps running (see var/*.log)" >&2
    fi
  done
  sleep 2
done
echo "engine exited; stopping the rest (see var/engine.log)" >&2
exit 1
