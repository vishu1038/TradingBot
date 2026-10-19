#!/bin/sh
# Unattended (no Claude): finish the metrics archive, walk-forward the CNN on every archived perp
# (229 coins x 24 months, phase-A settings), compare against the 25-coin phase A run, post to Google Chat.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=backend
STAMP=$(date +%Y%m%d_%H%M)

.venv/bin/python ml/notify_text.py "229-coin training started: step 1/3 metrics download"
nice -n 10 .venv/bin/python scripts/download_archive.py --all-perps --kinds metrics --months 24 --workers 48 --no-verify --no-topup > "var/all229_download_$STAMP.log" 2>&1

.venv/bin/python ml/notify_text.py "229-coin training: step 2/3 walk-forward (CNN, fixed + atr32, 6 folds x 45 days)"
nice -n 10 .venv-ml/bin/python ml/walkforward.py wf --archs cnn --modes fixed,atr32 --folds 6 --test-days 45 \
  --aux 0 --seeds 1 --tag a0s1_229 > "var/all229_wf_$STAMP.log" 2>&1
RC=$?

.venv/bin/python ml/compare.py --arch cnn --modes fixed,atr32 --tags a0s1_25,a0s1_229 > "var/ml/all229_result_$STAMP.txt" 2>&1
.venv/bin/python ml/notify_text.py "229-coin training finished (walk-forward exit $RC). 25 vs 229 coins:" "var/ml/all229_result_$STAMP.txt"
