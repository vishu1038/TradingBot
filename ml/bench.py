"""Throughput / regularisation probe on the FIRST walk-forward fold's train + val blocks only (never test or
holdout).  Prints ms/step, GPU "Device Utilization %" sampled during training, and best validation loss.

    PYTHONPATH=backend nice -n 10 .venv-ml/bin/python ml/bench.py --mode fixed --configs b512,b2048,b4096
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

import dataset as D  # noqa: E402
import trainer as T  # noqa: E402
import walkforward as W  # noqa: E402
from tb import features as ft  # noqa: E402

CONFIGS = {
    # name: (TrainCfg kwargs, arch_cfg)
    "b512": (dict(batch=512, lr=2e-3, eval_every=100, steps=600), {}),
    "b2048": (dict(batch=2048, lr=4e-3, eval_every=50, steps=300), {}),
    "b4096": (dict(batch=4096, lr=6e-3, eval_every=50, steps=300), {}),
    "b2048_reg": (dict(batch=2048, lr=3e-3, weight_decay=5e-2, eval_every=50, steps=300),
                  {"dropout": 0.3, "channels": 48}),
    "b2048_small": (dict(batch=2048, lr=3e-3, weight_decay=5e-2, eval_every=50, steps=300),
                    {"dropout": 0.4, "channels": 24, "hidden": 32, "dilations": [1, 2, 4, 8]}),
    "gru_b1024": (dict(batch=1024, lr=2e-3, weight_decay=5e-2, eval_every=50, steps=200), {}),
    "gru_b2048": (dict(batch=2048, lr=2e-3, weight_decay=5e-2, eval_every=50, steps=200), {}),
    "b2048_small_lowlr": (dict(batch=2048, lr=1e-3, weight_decay=1e-1, eval_every=50, steps=400),
                          {"dropout": 0.4, "channels": 24, "hidden": 32, "dilations": [1, 2, 4, 8]}),
}


def gpu_util_sampler(samples: list, stop: threading.Event):
    while not stop.is_set():
        try:
            out = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"], capture_output=True, text=True,
                                 timeout=5).stdout
            m = re.findall(r'"Device Utilization %"=(\d+)', out)
            if m:
                samples.append(max(int(x) for x in m))
        except Exception:
            pass
        stop.wait(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fixed")
    ap.add_argument("--arch", default="cnn")
    ap.add_argument("--configs", default="b512,b2048,b4096")
    ap.add_argument("--no-gpu-data", action="store_true", help="use the old numpy-prefetch path (before numbers)")
    a = ap.parse_args()
    ds = D.build_dataset(log=lambda *x: None)
    sp = W.make_splits(ds.t, purge_bars=int(ds.meta["horizon"]) + ft.WINDOW)
    f = sp["folds"][0]
    if a.no_gpu_data:
        T.GPU_MAX_BYTES = 0
    ok = ds.sample_ok & (np.isfinite(ds.lab[f"{a.mode}_long"]) | np.isfinite(ds.lab[f"{a.mode}_short"]))
    tr, va = W.rows_in(ds, f["train"], ok), W.rows_in(ds, f["val"], ok)
    norm = ft.robust_norm_stats(ds.X[tr[::5]].astype("float32"))
    for name in a.configs.split(","):
        kw, ac = CONFIGS[name]
        cfg = T.TrainCfg(arch=a.arch, arch_cfg=ac, patience=99, val_max=60_000, **kw)
        samples, stop = [], threading.Event()
        th = threading.Thread(target=gpu_util_sampler, args=(samples, stop), daemon=True)
        th.start()
        t0 = time.time()
        net, info = T.train_model(ds, a.mode, tr, va, norm, cfg, lambda *x: None)
        dt = time.time() - t0
        stop.set()
        th.join()
        h = info["history"]
        best = min(h, key=lambda r: r["val_loss"])
        util = np.mean(samples) if samples else float("nan")
        print(f"{name:18s} steps={cfg.steps} batch={cfg.batch} {1000 * dt / cfg.steps:6.1f} ms/step (incl. val) "
              f"util mean {util:4.1f}% max {max(samples) if samples else 'n/a'}  best val {best['val_loss']:.4f} @ "
              f"step {best['step']}  last val {h[-1]['val_loss']:.4f}  samples/s {cfg.steps * cfg.batch / dt:,.0f}",
              flush=True)
        del net
        T.free_gpu()


if __name__ == "__main__":
    main()
