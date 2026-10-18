#!/bin/sh
# Chronos-Bolt zero-shot forecasts on a time-stratified random subsample (CPU, 3 threads, nice 10).
cd "$(dirname "$0")/.." || exit 1
PY=/Users/dungeonmaster/.local/share/tradingbot/ml-env/bin/python
for m in tiny small; do
  nice -n 10 "$PY" ml/chronos_probe.py forecast --model "amazon/chronos-bolt-$m" --ctx 256 --per-symbol 1500 > "var/chronos_$m.log" 2>&1
done
