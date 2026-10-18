#!/bin/sh
# Launch independent walk-forward jobs (one process per exit mode) that share the GPU.
# Useful for GRU/transformer, which leave the GPU mostly idle (python-loop bound, ~20% util); the CNN already
# saturates it (~90%+ at batch 2048) so run CNN jobs one at a time.  Check `memory_pressure` first: each
# process holds ~1.2 GB RSS.
#   ml/run_parallel.sh gru "fixed atr32 atr63"
cd "$(dirname "$0")/.." || exit 1
ARCH=${1:-gru}
MODES=${2:-"fixed atr32 atr63"}
export PYTHONPATH=backend
mkdir -p var
for m in $MODES; do
  nohup nice -n 10 .venv-ml/bin/python ml/walkforward.py wf --archs "$ARCH" --modes "$m" > "var/ml_${ARCH}_${m}.log" 2>&1 &
done
wait
