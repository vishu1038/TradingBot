"""Training on mps: streaming fp16 batches, background prefetch, early stopping on validation.

CRITICAL: torch CPU math crashes on this machine (MKL on AMD).  Everything that touches a model runs on
``mps``; tensors come from ``torch.from_numpy(x).to("mps")`` and modules are moved to mps before any
forward.  Importing this module sets the MPS watermark env vars (must happen before torch loads).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import copy  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import models as M  # noqa: E402
from dataset import Dataset  # noqa: E402
from tb import features as ft  # noqa: E402

DEVICE = "mps"
OFFS = np.arange(-ft.WINDOW + 1, 1, dtype="int64")


def require_mps():
    if not torch.backends.mps.is_available():
        raise RuntimeError("mps not available: refusing to run model math on CPU (MKL crashes on this box)")


def gather(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """[len(idx), WINDOW, F] fp16 windows ending at rows idx."""
    return X[idx[:, None] + OFFS[None, :]]


@dataclass
class TrainCfg:
    arch: str = "cnn"
    arch_cfg: dict = field(default_factory=dict)
    steps: int = 600
    batch: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-1
    eval_every: int = 50
    patience: int = 5
    val_max: int = 60_000
    pos_weight: str = "auto"          # "auto" | "none"
    seed: int = 0
    clip_grad: float = 1.0


class Prefetcher:
    """One background thread producing (x_fp16, y, m) numpy batches; queue depth 3."""

    def __init__(self, X, idx_pool, Y, Mk, batch, steps, seed, depth=3):
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, args=(X, idx_pool, Y, Mk, batch, steps, seed), daemon=True)
        self.t.start()

    def _run(self, X, pool, Y, Mk, batch, steps, seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(steps):
                if self._stop.is_set():
                    return
                ii = pool[rng.integers(0, len(pool), batch)]
                self.q.put((gather(X, ii), Y[ii], Mk[ii]))
        except Exception as e:  # surfaced in the consumer
            self.q.put(e)

    def get(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self._stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def _norm_t(mean, std):
    return (torch.from_numpy(mean.astype("float32")).to(DEVICE), torch.from_numpy(std.astype("float32")).to(DEVICE))


def to_dev(xb: np.ndarray, mean_t, std_t, clip: float = ft.NORM_CLIP):
    x = torch.from_numpy(np.ascontiguousarray(xb)).to(DEVICE).float()
    return ((x - mean_t) / std_t).clamp_(-clip, clip)


GPU_MAX_BYTES = 2 << 30


class GpuData:
    """The whole fp16 feature matrix resident on mps (uploaded once per process); [B, WINDOW, F] windows are
    built on the GPU by index gather, so a training step has no numpy -> mps copies and no Python batch prep."""

    def __init__(self, X: np.ndarray):
        self.src = X                                   # keep the array alive so id() stays unique
        self.X = torch.from_numpy(np.ascontiguousarray(X)).to(DEVICE)
        self.offs = torch.from_numpy(OFFS).to(DEVICE)
        self.labels: dict = {}

    def windows(self, idx_t):
        return self.X[idx_t[:, None] + self.offs[None, :]]


_GPU: dict[int, GpuData] = {}


def gpu_data(X: np.ndarray) -> "GpuData | None":
    """Cached GPU copy of X, or None when it would not fit the VRAM budget (-> prefetch-thread fallback)."""
    if X.nbytes > GPU_MAX_BYTES:
        return None
    g = _GPU.get(id(X))
    if g is None or g.src is not X:
        _GPU.clear()
        free_gpu()
        g = _GPU[id(X)] = GpuData(X)
    return g


def predict_logits(net, X, idx, norm, batch: int = 4096) -> np.ndarray:
    """Raw logits [len(idx), 2] on mps, eval mode."""
    net.eval()
    mean_t, std_t = _norm_t(norm["mean"], norm["std"])
    g = gpu_data(X)
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            ii = idx[i:i + batch]
            if g is not None:
                x = g.windows(torch.from_numpy(ii.astype("int64")).to(DEVICE)).float()
                xb = ((x - mean_t) / std_t).clamp_(-ft.NORM_CLIP, ft.NORM_CLIP)
            else:
                xb = to_dev(gather(X, ii), mean_t, std_t)
            out.append(net(xb).to("cpu").numpy())
    return np.concatenate(out, axis=0).astype("float64") if out else np.zeros((0, 2))


def masked_bce(logits, y, m, pos_w):
    l = Fn.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = m * (1.0 + (pos_w - 1.0) * y)
    return (l * w).sum() / m.sum().clamp_min(1.0)


def val_loss(logits: np.ndarray, Y: np.ndarray, Mk: np.ndarray) -> float:
    z = logits
    l = np.maximum(z, 0) - z * Y + np.log1p(np.exp(-np.abs(z)))
    return float((l * Mk).sum() / max(Mk.sum(), 1.0))


def train_model(ds: Dataset, mode: str, train_idx: np.ndarray, val_idx: np.ndarray, norm: dict, cfg: TrainCfg,
                log=print):
    """Train one model for ``mode`` ("fixed"/"atr").  idx = sample row positions (valid windows).
    Returns (net on mps, history dict).  Early stopping restores the best-validation weights."""
    require_mps()
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    Y = np.stack([ds.lab[f"{mode}_long"], ds.lab[f"{mode}_short"]], axis=1)
    Mk = np.isfinite(Y).astype("float32")
    Y = np.nan_to_num(Y, nan=0.0).astype("float32")
    net, arch_cfg = M.build(cfg.arch, ft.F, ft.WINDOW, cfg.arch_cfg)
    net = net.to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    if cfg.pos_weight == "auto":
        pr = (Y[train_idx] * Mk[train_idx]).sum(0) / np.maximum(Mk[train_idx].sum(0), 1)
        pw = np.clip((1 - pr) / np.maximum(pr, 1e-3), 0.5, 4.0)
    else:
        pw = np.ones(2)
    pos_w = torch.from_numpy(pw.astype("float32")).to(DEVICE)
    mean_t, std_t = _norm_t(norm["mean"], norm["std"])

    if len(val_idx) > cfg.val_max:
        val_idx = np.sort(np_rng.choice(val_idx, cfg.val_max, replace=False))
    g = gpu_data(ds.X)
    if g is not None:
        pool_t = torch.from_numpy(train_idx.astype("int64")).to(DEVICE)
        Yt = torch.from_numpy(Y).to(DEVICE)
        Mt = torch.from_numpy(Mk).to(DEVICE)
        pre = None
    else:
        pre = Prefetcher(ds.X, train_idx, Y, Mk, cfg.batch, cfg.steps, cfg.seed + 1)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    try:
        for step in range(1, cfg.steps + 1):
            net.train()
            if g is not None:
                bi = pool_t[torch.randint(0, len(pool_t), (cfg.batch,), device=DEVICE)]
                x = ((g.windows(bi).float() - mean_t) / std_t).clamp_(-ft.NORM_CLIP, ft.NORM_CLIP)
                y, m = Yt[bi], Mt[bi]
            else:
                xb, yb, mb = pre.get()
                x = to_dev(xb, mean_t, std_t)
                y = torch.from_numpy(yb).to(DEVICE)
                m = torch.from_numpy(mb).to(DEVICE)
            loss = masked_bce(net(x), y, m, pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad)
            opt.step()
            sched.step()
            if step % cfg.eval_every == 0 or step == cfg.steps:
                lv = float(loss.item())              # sync point (only here)
                if not np.isfinite(lv):
                    raise FloatingPointError("training loss is not finite")
                lg = predict_logits(net, ds.X, val_idx, norm)
                vl = val_loss(lg, Y[val_idx], Mk[val_idx])
                hist.append(dict(step=step, train_loss=lv, val_loss=vl, elapsed_s=round(time.time() - t0, 1)))
                log(f"  [{cfg.arch}/{mode}] step {step:5d} train {lv:.4f} val {vl:.4f} ({hist[-1]['elapsed_s']}s)")
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        log(f"  early stop at step {step} (best val {best:.4f})")
                        break
    finally:
        if pre is not None:
            pre.close()
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    total = time.time() - t0
    n_steps = hist[-1]["step"] if hist else 0
    return net, dict(history=hist, best_val_loss=float(best), arch_cfg=arch_cfg, seconds=round(total, 1),
                     steps_run=n_steps, ms_per_step=round(1000 * total / max(n_steps, 1), 1),
                     pos_weight=pw.tolist())


def free_gpu():
    import gc
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
