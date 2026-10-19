#!/bin/sh
# Unattended (no Claude): V3 experiment A (cross-sectional ranking, market neutral) + D (maker execution).
# Walk-forward (GBM + MLP x horizons 12/24/48/72) -> validation-only selection -> pooled out-of-sample vs random and
# 7d-momentum rankings -> ONE locked-holdout evaluation -> export a challenger -> post the summary to Google Chat.
#   nohup sh ml/run_v3_xsec.sh > var/ml/v3_xsec_launch.log 2>&1 &
# Extra arguments are passed to ml/xsec_walkforward.py (e.g. --resume --tag v1_20261019_1100 to continue a run).
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=backend
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
STAMP=$(date +%Y%m%d_%H%M)
TAG="v1_$STAMP"
case " $* " in *" --tag "*) TAG=$(echo "$*" | sed -n 's/.*--tag \([^ ]*\).*/\1/p') ;; esac
LOG="var/ml/v3_xsec_$TAG.log"
REPORT="var/ml/v3_xsec_${TAG}_report.txt"
mkdir -p var/ml var/lab

# at most 2 GPU trainings system-wide: wait while two other jobs already use the GPU
gpu_jobs() {
  ps -axo command | grep -E '\.venv-ml/bin/python.*(walkforward\.py|trainer\.py|chronos|retrain|train_|bench\.py)' \
    | grep -v grep | grep -v xsec_walkforward | wc -l | tr -d ' '
}
WAITED=0
while [ "$(gpu_jobs)" -ge 2 ]; do
  [ $WAITED -eq 0 ] && .venv/bin/python ml/notify_text.py "V3 xsec ($TAG): waiting for a free GPU slot ($(gpu_jobs) other jobs)"
  WAITED=1
  sleep 120
done

.venv/bin/python ml/notify_text.py "V3 xsec ($TAG) started: walk-forward cross-sectional ranker (GBM+MLP), taker vs maker, 229 coins x 24 months; log $LOG"
nice -n 10 .venv-ml/bin/python ml/xsec_walkforward.py run --tag "$TAG" "$@" > "$LOG" 2>&1
RC=$?

if [ -f "$REPORT" ]; then
  .venv/bin/python ml/notify_text.py "V3 xsec ($TAG) finished rc=$RC:
$(cut -c1-170 "$REPORT" | head -c 3300)"
else
  .venv/bin/python ml/notify_text.py "V3 xsec ($TAG) FAILED rc=$RC, last log lines:" "$LOG"
fi
exit $RC
