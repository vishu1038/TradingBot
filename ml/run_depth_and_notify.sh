#!/bin/sh
# Paired depth test, then post the comparison to Google Chat. Runs unattended (no Claude needed).
cd "$(dirname "$0")/.." || exit 1
sh ml/run_depth_pair.sh
.venv/bin/python ml/notify_depth_result.py
