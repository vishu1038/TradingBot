"""MLP training on mps for the multi-timeframe feature vector (flat [R, F] rows, two heads long / short).

CRITICAL (see ml/README.md): torch CPU math crashes on this machine (MKL on AMD) - every model op runs on ``mps``;
tensors are built in numpy and moved with ``torch.from_numpy(x).to("mps")``.  The whole normalised training matrix
lives on the GPU (fp16, <= ~0.4 GB).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import models as M  # noqa: E402
from tb import features as ft  # noqa: E402

DEVICE = "mps"


def require_mps():
    if not torch.backends.mps.is_available():
        raise RuntimeError("mps not available: refusing to run model math on CPU (MKL crashes on this box)")


@dataclass
class MlpCfg:
    arch_cfg: dict = field(default_factory=lambda: {"hidden": 128, "hidden2": 64, "dropout": 0.3})
    steps: int = 2500
    batch: int = 4096
    lr: float = 2e-3
    weight_decay: float = 5e-2
    eval_every: int = 100
    patience: int = 6
    seed: int = 0
    clip_grad: float = 1.0


def free_gpu():
    import gc
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass


def normalise_np(X: np.ndarray, mean: np.ndarray, std: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """(X - mean) / std clipped to +-NORM_CLIP, fp16, in chunks (X may be a big memmap-derived array)."""
    out = np.empty(X.shape, dtype="float16")
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = ft.normalize(X[i:i + chunk], mean, std).astype("float16")
    return out


def predict_logits(net, Xn: np.ndarray, batch: int = 16384) -> np.ndarray:
    """Raw logits [R, n_out] (float64) of normalised fp16 rows, eval mode."""
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            x = torch.from_numpy(np.ascontiguousarray(Xn[i:i + batch])).to(DEVICE).float()
            out.append(net(x).to("cpu").numpy())
    return np.concatenate(out, axis=0).astype("float64") if out else np.zeros((0, 2))


def _masked_bce(logits, y, m):
    l = Fn.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (l * m).sum() / m.sum().clamp_min(1.0)


def _val_loss(lg: np.ndarray, Y: np.ndarray, Mk: np.ndarray) -> float:
    z = lg[:, :2]
    l = np.maximum(z, 0) - z * Y + np.log1p(np.exp(-np.abs(z)))
    return float((l * Mk).sum() / max(Mk.sum(), 1.0))


def train_mlp(Xtr: np.ndarray, Ytr: np.ndarray, Mtr: np.ndarray, Xva: np.ndarray, Yva: np.ndarray, Mva: np.ndarray,
              cfg: MlpCfg, log=print):
    """Inputs are NORMALISED fp16 feature rows with float32 targets [R, 2] and masks [R, 2] (0 where the label is
    n.a.).  Early stopping on the validation BCE; the best-validation weights are restored."""
    require_mps()
    torch.manual_seed(cfg.seed)
    net, arch_cfg = M.build("mlp", Xtr.shape[1], 1, dict(cfg.arch_cfg))
    net = net.to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    Xg = torch.from_numpy(np.ascontiguousarray(Xtr)).to(DEVICE)
    Yg = torch.from_numpy(np.ascontiguousarray(Ytr, dtype="float32")).to(DEVICE)
    Mg = torch.from_numpy(np.ascontiguousarray(Mtr, dtype="float32")).to(DEVICE)
    R = len(Xtr)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        net.train()
        bi = torch.randint(0, R, (cfg.batch,), device=DEVICE)
        out = net(Xg[bi].float())
        loss = _masked_bce(out[:, :2], Yg[bi], Mg[bi])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad)
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            lv = float(loss.item())
            if not np.isfinite(lv):
                raise FloatingPointError("training loss is not finite")
            vl = _val_loss(predict_logits(net, Xva), Yva, Mva)
            hist.append(dict(step=step, train_loss=lv, val_loss=vl, elapsed_s=round(time.time() - t0, 1)))
            log(f"  [mlp/s{cfg.seed}] step {step:5d} train {lv:.4f} val {vl:.4f} ({hist[-1]['elapsed_s']}s)")
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    log(f"  early stop at step {step} (best val {best:.4f})")
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    del Xg, Yg, Mg
    free_gpu()
    return net, dict(history=hist, best_val_loss=float(best), arch_cfg=arch_cfg, seconds=round(time.time() - t0, 1),
                     steps_run=hist[-1]["step"] if hist else 0, seed=cfg.seed)
