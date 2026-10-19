"""Rankers for the cross-sectional experiment: HistGradientBoostingRegressor (CPU) and a small MLP (torch on `mps`).

Both are trained on the gaussianised cross-sectional rank of the 24h-relative return (regression) and early-stopped
on the validation rank-IC (mean per-hour Spearman correlation between score and realised relative return).
Inference is numpy (MLP) / sklearn (GBM) via the objects in ``tb.model_xsec`` - torch is only needed to TRAIN.

CRITICAL (docs/ARCHITECTURE.md): torch CPU math crashes on this machine (MKL on AMD).  Every tensor op here runs on
``mps``; ``require_mps`` refuses to train otherwise.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from tb import model_xsec as mx  # noqa: E402


def rows_ic(pred: np.ndarray, rel: np.ndarray, k: np.ndarray, s: np.ndarray, S: int, min_n: int = 10) -> float:
    """Mean per-hour Spearman IC between ``pred`` and ``rel`` for rows identified by (hour k, coin s)."""
    import pandas as pd
    uk, inv = np.unique(k, return_inverse=True)
    A = np.full((len(uk), S), np.nan)
    B = np.full((len(uk), S), np.nan)
    A[inv, s] = pred
    B[inv, s] = rel
    ok = np.isfinite(A) & np.isfinite(B)
    A = np.where(ok, A, np.nan)
    B = np.where(ok, B, np.nan)
    ra = pd.DataFrame(A).rank(axis=1).to_numpy()
    rb = pd.DataFrame(B).rank(axis=1).to_numpy()
    da = ra - np.nanmean(ra, axis=1, keepdims=True)
    db = rb - np.nanmean(rb, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nansum(da * db, axis=1) / np.sqrt(np.nansum(da * da, axis=1) * np.nansum(db * db, axis=1))
    c[ok.sum(1) < min_n] = np.nan
    return float(np.nanmean(c)) if np.isfinite(c).any() else float("nan")


# --------------------------------------------------------------------------------------------
# gradient boosting
# --------------------------------------------------------------------------------------------
GBM_DEFAULT = dict(learning_rate=0.05, max_depth=4, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=5.0,
                   max_features=0.5, max_bins=63)


def train_gbm(tr: dict, va: dict, S: int, seed: int = 0, max_iter: int = 240, step: int = 10, **kw):
    """-> (GbmRanker, info).  The number of boosting stages is chosen on the validation rank-IC (every ``step``)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    params = dict(GBM_DEFAULT, **kw)
    t0 = time.time()
    m = HistGradientBoostingRegressor(loss="squared_error", max_iter=max_iter, early_stopping=False,
                                      random_state=seed, **params)
    m.fit(tr["X"], tr["y"])
    best, best_n, curve = -9.0, max_iter, []
    for n, p in enumerate(m.staged_predict(va["X"]), start=1):
        if n % step == 0 or n == max_iter:
            ic = rows_ic(p, va["rel"], va["k"], va["s"], S)
            curve.append((n, ic))
            if np.isfinite(ic) and ic > best:
                best, best_n = ic, n
    info = {"val_ic": best, "n_stages": best_n, "curve": curve, "fit_s": time.time() - t0, "n_train": int(len(tr["y"]))}
    return mx.GbmRanker(m, best_n), info


# --------------------------------------------------------------------------------------------
# MLP (torch / mps)
# --------------------------------------------------------------------------------------------
GPU_JOB_PAT = r"\.venv-ml/bin/python.*(walkforward\.py|trainer\.py|chronos|retrain|train_|bench\.py)"


def other_gpu_jobs() -> int:
    """Number of other GPU training jobs running system-wide (by command line), excluding this process tree."""
    import re
    import subprocess
    try:
        out = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return 0
    me = {os.getpid(), os.getppid()}
    n = 0
    for line in out.splitlines()[1:]:
        pid, _, cmd = line.strip().partition(" ")
        if not pid.isdigit() or int(pid) in me or "xsec_walkforward" in cmd:
            continue
        if re.search(GPU_JOB_PAT, cmd):
            n += 1
    return n


def wait_for_gpu(max_others: int = 1, poll_s: int = 60, max_wait_s: int = 6 * 3600, log=print):
    """At most 2 GPU jobs system-wide: wait while two OTHER jobs already hold the GPU."""
    t0 = time.time()
    while other_gpu_jobs() > max_others and time.time() - t0 < max_wait_s:
        log(f"[gpu] {other_gpu_jobs()} other GPU jobs running - waiting")
        time.sleep(poll_s)


def require_mps():
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("mps not available: refusing to run model math on CPU (MKL crashes on this box)")


def _train_one_mlp(Xt, yt, Xv, rel_v, kv, sv, S, mean, std, seed, hidden, dropout, lr, wd, epochs, batch, patience):
    import torch
    import torch.nn as nn
    dev = "mps"
    torch.manual_seed(seed)
    F = Xt.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Linear(F, hidden[0])
            self.l1 = nn.Linear(hidden[0], hidden[1])
            self.l2 = nn.Linear(hidden[1], 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.drop(nn.functional.silu(self.l0(x)))
            h = self.drop(nn.functional.silu(self.l1(h)))
            return self.l2(h)[:, 0]

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    xt = torch.from_numpy(Xt).to(dev)
    yt_ = torch.from_numpy(yt).to(dev)
    xv = torch.from_numpy(Xv).to(dev)
    rng = np.random.default_rng(seed)
    N = len(Xt)
    best, best_state, bad, curve = -9.0, None, 0, []
    for ep in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(N)).to(dev)
        for a in range(0, N, batch):
            idx = perm[a:a + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xt[idx]), yt_[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(xv[a:a + 65536]) for a in range(0, len(xv), 65536)]).cpu().numpy()
        ic = rows_ic(pv, rel_v, kv, sv, S)
        curve.append((ep + 1, ic))
        if np.isfinite(ic) and ic > best:
            best, bad = ic, 0
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    del xt, yt_, xv, net, opt
    torch.mps.empty_cache()
    return best_state, best, curve


def train_mlp(tr: dict, va: dict, S: int, seeds: int = 3, hidden=(96, 48), dropout: float = 0.2, lr: float = 2e-3,
              wd: float = 1e-2, epochs: int = 40, batch: int = 4096, patience: int = 4, seed0: int = 0):
    """-> (MlpRanker ensemble of ``seeds`` networks, info).  Inputs are standardised with train mean/std."""
    require_mps()
    wait_for_gpu()
    t0 = time.time()
    mean = tr["X"].mean(0).astype("float32")
    std = np.maximum(tr["X"].std(0), 1e-3).astype("float32")
    nz = lambda X: np.clip((X - mean) / std, -10.0, 10.0).astype("float32")  # noqa: E731
    Xt, Xv = nz(tr["X"]), nz(va["X"])
    members, ics, curves = [], [], []
    for i in range(seeds):
        st, ic, cv = _train_one_mlp(Xt, tr["y"].astype("float32"), Xv, va["rel"], va["k"], va["s"], S, mean, std,
                                    seed0 + i, hidden, dropout, lr, wd, epochs, batch, patience)
        if st is None:
            continue
        members.append(st)
        ics.append(ic)
        curves.append(cv)
    ranker = mx.MlpRanker(members, mean, std)
    pv = ranker.predict(va["X"])
    info = {"val_ic": rows_ic(pv, va["rel"], va["k"], va["s"], S), "member_val_ic": ics, "epochs": [len(c) for c in curves],
            "fit_s": time.time() - t0, "n_train": int(len(tr["y"])), "hidden": list(hidden)}
    return ranker, info
