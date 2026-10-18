#!/bin/sh
# Paired depth test: identical rows / splits / seeds, with vs without the depth feature group.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=backend
SYMS=1000BONKUSDT,1000PEPEUSDT,ADAUSDT,APTUSDT,ARBUSDT,AVAXUSDT,BNBUSDT,BTCUSDT,DOGEUSDT,ENAUSDT,ETHUSDT,FETUSDT,INJUSDT,LINKUSDT,NEARUSDT,OPUSDT,ORDIUSDT,SEIUSDT,SOLUSDT,SUIUSDT,TAOUSDT,TIAUSDT,WIFUSDT,WLDUSDT,XRPUSDT
COMMON="--symbols $SYMS --groups depth --start 2025-10-02 --archs cnn --modes fixed,atr32 --folds 4 --test-days 30 --min-train-days 25"
nice -n 10 .venv-ml/bin/python ml/walkforward.py wf $COMMON --use-cols all --tag depth_on > var/depth_on.log 2>&1
nice -n 10 .venv-ml/bin/python ml/walkforward.py wf $COMMON --use-cols base --tag depth_off > var/depth_off.log 2>&1
