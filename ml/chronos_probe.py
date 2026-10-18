"""Chronos-Bolt zero-shot probe (runs in the modern CPU env, NOT .venv-ml):

    nice -n 10 /Users/dungeonmaster/.local/share/tradingbot/ml-env/bin/python -I ml/chronos_probe.py cost
    nice -n 10 .../python ml/chronos_probe.py forecast --per-symbol 600 --model amazon/chronos-bolt-tiny

``cost``     loads the model, times forecasts for several batch sizes / context lengths (<= 3 threads).
``forecast`` samples bars (time-stratified) from the archive, forecasts quantiles of future log-price at
             +12 bars for each sampled bar from the previous ``ctx`` log closes (strictly past data),
             and writes var/ml/chronos_{model}.parquet with the quantiles and the realised targets so the
             evaluation script can compare against ATR.  The context ends at the signal bar's close.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
ROOT = Path(__file__).resolve().parents[1]
ARCH = Path(os.environ.get("TB_ARCHIVE") or ROOT / "data" / "archive")
OUT = Path(os.environ.get("TB_VAR") or ROOT / "var") / "ml"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
H = 12                                    # forecast horizon in bars (1 hour)


def load_pipeline(name: str):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)


def forecast_quantiles(pipe, ctx_batch: list[np.ndarray], h: int = H) -> np.ndarray:
    """-> [B, h, 9] quantile forecasts (log-price) for the 9 default Chronos-Bolt quantiles."""
    ctx = [torch.tensor(c, dtype=torch.float32) for c in ctx_batch]
    with torch.no_grad():
        q = pipe.predict(ctx, prediction_length=h)                 # [B, n_q, h] (or list)
    q = torch.as_tensor(q) if not torch.is_tensor(q) else q
    return q.permute(0, 2, 1).numpy()


def cmd_cost(a):
    t = time.time()
    pipe = load_pipeline(a.model)
    print(f"load {a.model}: {time.time() - t:.1f}s threads={torch.get_num_threads()}", flush=True)
    rng = np.random.default_rng(0)
    for ctx_len in (128, 256, 512):
        for bs in (1, 16, 64):
            ctx = [np.cumsum(rng.normal(0, 0.002, ctx_len)) for _ in range(bs)]
            forecast_quantiles(pipe, ctx)                           # warm-up
            t = time.time()
            forecast_quantiles(pipe, ctx)
            dt = time.time() - t
            print(f"ctx {ctx_len:3d} batch {bs:3d}: {dt:6.2f}s  = {1000 * dt / bs:7.1f} ms/forecast", flush=True)


def cmd_forecast(a):
    pipe = load_pipeline(a.model)
    rng = np.random.default_rng(a.seed)
    syms = sorted(p.stem for p in (ARCH / "klines" / "5m").glob("*.parquet"))
    if a.symbols:
        syms = [s for s in syms if s in a.symbols.split(",")]
    rows = []
    t0 = time.time()
    n_fc = 0
    for s in syms:
        kl = pd.read_parquet(ARCH / "klines" / "5m" / f"{s}.parquet")
        c = np.log(kl["close"].to_numpy("float64"))
        hi, lo = kl["high"].to_numpy("float64"), kl["low"].to_numpy("float64")
        pc = np.concatenate([[kl["close"].iloc[0]], kl["close"].to_numpy()[:-1]])
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
        atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
        atr_pct = atr / kl["close"].to_numpy()
        n = len(c)
        cand = np.arange(a.ctx, n - H - 1)
        idx = np.sort(rng.choice(cand, size=min(a.per_symbol, len(cand)), replace=False))
        for b in range(0, len(idx), a.batch):
            ii = idx[b:b + a.batch]
            ctx = [c[i - a.ctx + 1:i + 1] - c[i] for i in ii]        # relative to the last close
            q = forecast_quantiles(pipe, ctx)                         # [B, H, 9] log-price offsets
            n_fc += len(ii)
            for k, i in enumerate(ii):
                fut = c[i + 1:i + H + 1] - c[i]
                rows.append(dict(symbol=s, open_time=int(kl["open_time"].iloc[i]), atr_pct=float(atr_pct[i]),
                                 **{f"q{int(100 * qq)}": float(q[k, -1, j]) for j, qq in enumerate(QUANTILES)},
                                 ret_h=float(fut[-1]), rv_h=float(np.sqrt(np.sum(np.diff(np.r_[0.0, fut]) ** 2))),
                                 absret_h=float(abs(fut[-1])), mfe_h=float(fut.max()), mae_h=float(fut.min())))
        print(f"{s}: {len(idx)} forecasts  total {n_fc}  {time.time() - t0:.0f}s ({1000 * (time.time() - t0) / max(n_fc, 1):.0f} ms/forecast)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"chronos_{a.model.split('/')[-1]}.parquet"
    pd.DataFrame(rows).to_parquet(path)
    print(f"wrote {path} ({len(rows)} rows); {1000 * (time.time() - t0) / max(n_fc, 1):.1f} ms/forecast end-to-end")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cost", "forecast"])
    ap.add_argument("--model", default="amazon/chronos-bolt-tiny")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--per-symbol", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"cost": cmd_cost, "forecast": cmd_forecast}[a.cmd](a)
