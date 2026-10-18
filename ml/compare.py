"""Compare walk-forward result files (var/ml/wf_{arch}_{mode}_{tag}.json).

    python ml/compare.py --arch cnn --modes fixed,atr32,atr63 --tags a0s1_25,a1s1_25,a1s5_25

Prints, per exit mode and side: pooled OOS trades / win rate / expectancy of the network at its val-chosen
threshold next to the random-entry baseline and the ATR%-rule baseline, mean per-fold AUC (network vs
ATR-only), calibrated log-loss, calibration error, and drift false-alarm rates; then paired per-fold
differences between consecutive tags (AUC and log-loss: positive AUC diff / negative log-loss diff = the later
tag is better).  numpy only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VAR = Path(__file__).resolve().parents[1] / "var" / "ml"


def f(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cnn")
    ap.add_argument("--modes", default="fixed,atr32,atr63")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--dir", default=str(VAR))
    a = ap.parse_args()
    tags = a.tags.split(",")
    for mode in a.modes.split(","):
        R = {}
        for t in tags:
            p = Path(a.dir) / f"wf_{a.arch}_{mode}_{t}.json"
            if p.exists():
                R[t] = json.loads(p.read_text())
        if not R:
            continue
        print(f"\n### {a.arch} / {mode}")
        for side in ("long", "short"):
            print(f"\n{side}: pooled out-of-sample, thresholds chosen on validation only")
            print("| tag | trades | win % | exp $/trade | random exp | ATR-rule exp | z vs random | AUC | ATR-only AUC | logloss | ECE cal | drift z<-2 cal / raw |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for t, r in R.items():
                p = r[f"pooled_{side}"]
                c, ru, d = p["chosen"], p["atr_rule"], p["drift"]
                print(f"| {t} | {c['n']} | {f(100 * c['win_rate'] if c['win_rate'] is not None else None, 1)} | {f(c['expectancy'])} | "
                      f"{f(c['baseline_expectancy'])} | {f(ru['expectancy'])} (n={ru['n']}) | {f(c['z_vs_baseline'], 1)} | {f(p['auc'])} | "
                      f"{f(p['auc_atr_only'])} | {f(p['logloss_cal'], 4)} | {f(p['ece_cal'])} | "
                      f"{f(d['calibrated']['frac_z_lt_m2'])} / {f(d['uncalibrated']['frac_z_lt_m2'])} |")
            names = list(R)
            for t0, t1 in zip(names, names[1:]):
                a0 = {fo["fold"]: fo[side] for fo in R[t0]["folds"]}
                a1 = {fo["fold"]: fo[side] for fo in R[t1]["folds"]}
                common = sorted(set(a0) & set(a1))
                if not common:
                    continue
                dauc = np.array([a1[k]["auc"] - a0[k]["auc"] for k in common])
                dll = np.array([a1[k]["logloss_cal"] - a0[k]["logloss_cal"] for k in common])
                dece = np.array([a1[k]["calibration"]["ece_cal"] - a0[k]["calibration"]["ece_cal"] for k in common])
                se = lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")  # noqa: E731
                print(f"\npaired per-fold {t1} minus {t0} ({len(common)} folds): AUC {dauc.mean():+.4f} +- {se(dauc):.4f} "
                      f"(later better in {(dauc > 0).sum()}/{len(common)}), logloss {dll.mean():+.5f} +- {se(dll):.5f} "
                      f"(better in {(dll < 0).sum()}/{len(common)}), ECE {dece.mean():+.4f} +- {se(dece):.4f}")


if __name__ == "__main__":
    main()
