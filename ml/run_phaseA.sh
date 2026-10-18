#!/bin/sh
# Phase A on the original 25 symbols (24 months of klines): single-seed CNN, auxiliary heads off vs on.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=backend
SYMS=1000BONKUSDT,1000PEPEUSDT,ADAUSDT,APTUSDT,ARBUSDT,AVAXUSDT,BNBUSDT,BTCUSDT,DOGEUSDT,ENAUSDT,ETHUSDT,FETUSDT,INJUSDT,LINKUSDT,NEARUSDT,OPUSDT,ORDIUSDT,SEIUSDT,SOLUSDT,SUIUSDT,TAOUSDT,TIAUSDT,WIFUSDT,WLDUSDT,XRPUSDT
COMMON="--symbols $SYMS --archs cnn --modes fixed,atr32,atr63 --folds 6 --test-days 45 --tag"
nice -n 10 .venv-ml/bin/python ml/walkforward.py wf $COMMON a0s1_25 --aux 0 --seeds 1 > var/phaseA_a0.log 2>&1
nice -n 10 .venv-ml/bin/python ml/walkforward.py wf $COMMON a1s1_25 --aux 1 --seeds 1 > var/phaseA_a1.log 2>&1
