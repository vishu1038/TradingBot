#!/bin/sh
# Unattended (no Claude): V3 experiments B (multi-timeframe net), C (gradient-boosted trees), E (Kalman features).
#   1. dataset   229 coins x 24 months, decision every closed 1h bar, 1h/4h/1d + kalman features, pct 3/6 and 1h-ATR 3/2 labels (CPU)
#   2. GBM walk-forward: 4 variants {pct36, atr1h32} x {kalman off, on}                                              (CPU, nice)
#   3. WAIT until no other GPU job (ml/run_all229.sh, ml/walkforward.py, retrain, chronos) runs, then MLP walk-forward (mps, 1 GPU job)
#   4. finish: qualify (beats random AND the ATR rule, z>=2), ONE holdout evaluation per exported variant, export as
#      challenger (or the single best as "PAPER TEST ONLY - no edge"), lab report, text report, post to Google Chat.
# Results: var/ml/v3_mtf_wf_<variant>.json/.npz, var/ml/v3_mtf_report_latest.txt, var/ml/v3_mtf_summary.json,
#          var/lab/train-v3_*.json, models/v3_*.json (+ .npz/.pkl).  Log: var/v3_mtf_<stamp>.log
# Re-running resumes: finished variants (var/ml/v3_mtf_wf_<variant>.json) are skipped.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=backend
STAMP=$(date +%Y%m%d_%H%M)
LOG="var/v3_mtf_$STAMP.log"
PY=.venv/bin/python
PYML=.venv-ml/bin/python
mkdir -p var/ml

note() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
notify() { $PY ml/notify_text.py "$@" > /dev/null 2>&1 || true; }
gpu_busy() { ps -axo command | grep -E "ml/walkforward\.py|ml/run_all229\.sh|ml/run_retrain_challenger\.sh|ml/chronos_|ml/run_chronos\.sh|ml/run_phaseA\.sh|ml/run_depth" | grep -v grep > /dev/null; }

note "v3 mtf pipeline start (pid $$)"
notify "V3 multi-timeframe experiments started (B mtf-net, C gbm, E kalman): dataset -> 4 GBM variants -> MLP variants (waits for a free GPU) -> holdout + export. Log: $LOG"

# ---- 1. dataset ----
TB_DS_WORKERS=2 nice -n 10 $PYML ml/mtf_walkforward.py build >> "$LOG" 2>&1
RC=$?
note "dataset rc=$RC"
if [ "$RC" -ne 0 ]; then
  notify "V3 mtf: dataset build FAILED (rc=$RC)" "$LOG"
  exit "$RC"
fi

# ---- 2. GBM walk-forward (CPU) ----
OMP_NUM_THREADS=3 nice -n 10 $PYML ml/mtf_walkforward.py wf --models gbm >> "$LOG" 2>&1
RC_GBM=$?
note "gbm walk-forward rc=$RC_GBM"
notify "V3 mtf: GBM walk-forward finished (rc=$RC_GBM); next the MLP variants once the GPU is free" "$LOG"

# ---- 3. wait for a free GPU, then the MLP walk-forward ----
N=0
while gpu_busy && [ "$N" -lt 300 ]; do
  [ $((N % 15)) -eq 0 ] && note "GPU busy (other ml job running), waiting ... ($N)"
  N=$((N + 1))
  sleep 120
done
note "GPU free (waited $N x 120 s); MLP walk-forward"
nice -n 10 $PYML ml/mtf_walkforward.py wf --models mlp >> "$LOG" 2>&1
RC_MLP=$?
note "mlp walk-forward rc=$RC_MLP"

# ---- 4. qualify, ONE holdout evaluation, export, report ----
while gpu_busy && [ "$N" -lt 330 ]; do
  N=$((N + 1))
  sleep 120
done
OMP_NUM_THREADS=3 nice -n 10 $PYML ml/mtf_walkforward.py finish >> "$LOG" 2>&1
RC_FIN=$?
note "finish rc=$RC_FIN"

if [ -f var/ml/v3_mtf_chat_latest.txt ] && [ "$RC_FIN" -eq 0 ]; then
  notify "V3 mtf DONE (gbm rc=$RC_GBM, mlp rc=$RC_MLP, finish rc=$RC_FIN). Full report: var/ml/v3_mtf_report_latest.txt" var/ml/v3_mtf_chat_latest.txt
else
  notify "V3 mtf finished WITHOUT a report (gbm rc=$RC_GBM, mlp rc=$RC_MLP, finish rc=$RC_FIN)" "$LOG"
fi
exit "$RC_FIN"
