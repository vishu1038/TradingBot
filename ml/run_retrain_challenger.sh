#!/bin/sh
# Drift-triggered retrain (3-hourly review rule: champion drift z < -2). Exports a CHALLENGER only;
# the engine shadows it and promotes it only through its own rules. Posts the outcome to Google Chat.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=backend
SYMS=1000BONKUSDT,1000PEPEUSDT,ADAUSDT,APTUSDT,ARBUSDT,AVAXUSDT,BNBUSDT,BTCUSDT,DOGEUSDT,ENAUSDT,ETHUSDT,FETUSDT,INJUSDT,LINKUSDT,NEARUSDT,OPUSDT,ORDIUSDT,SEIUSDT,SOLUSDT,SUIUSDT,TAOUSDT,TIAUSDT,WIFUSDT,WLDUSDT,XRPUSDT
LOG=var/retrain_$(date +%Y%m%d_%H%M).log
nice -n 10 .venv-ml/bin/python ml/walkforward.py final --symbols "$SYMS" --archs cnn --modes atr32 > "$LOG" 2>&1
RC=$?
.venv/bin/python ml/notify_text.py "Drift-triggered retrain finished (exit $RC). Exported as CHALLENGER (shadow mode only). Log: $LOG" "$LOG"
