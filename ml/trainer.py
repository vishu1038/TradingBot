"""Training on mps: GPU-resident data (small sets) or streamed GPU chunks (large sets), multi-task heads,
early stopping on the main-head validation loss.

CRITICAL: torch CPU math crashes on this machine (MKL on AMD).  Everything that touches a model runs on
``mps``; tensors come from ``torch.from_numpy(x).to("mps")`` and modules are moved to mps before any
forward.  Importing this module sets the MPS watermark env vars (must happen before torch loads).

Data paths
    resident : the whole fp16 feature matrix (<= GPU_MAX_BYTES) lives on the GPU; [B, 64, F] windows are
               gathered on the GPU by index - no host->device copy per step.
    streaming: the matrix (several GB, memmapped on disk) is read in contiguous blocks; a chunk of ~0.8 M rows
               is uploaded, trained on for ``chunk_steps`` steps, and the next chunk is prepared by a
               background thread meanwhile (numpy only; uploads happen on the main thread).

Multi-task (cfg.aux): the net has ``n_out = 2 + 5`` outputs.  Column 0/1 = P(TP first) long/short logits (BCE,
the only thing used at inference); columns 2.. = standardised regression targets (smooth-L1, weight aux_w):
forward realised vol, max favourable / adverse excursion, time-to-exit long / short.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

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
GPU_MAX_BYTES = 2 << 30
AUX_COLS = ["fvol", "up_exc", "dn_exc", "ttl_long", "ttl_short"]


def require_mps():
    if not torch.backends.mps.is_available():
        raise RuntimeError("mps not available: refusing to run model math on CPU (MKL crashes on this box)")


def gather(X, idx: np.ndarray) -> np.ndarray:
    """[len(idx), WINDOW, F] fp16 windows ending at rows idx (X may be a memmap)."""
    return np.asarray(X[idx[:, None] + OFFS[None, :]])


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
    aux: bool = False
    aux_w: float = 0.3
    chunk_blocks: int = 400           # streaming: blocks per chunk x block_len rows
    block_len: int = 2048
    chunk_steps: int = 100


# --------------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------------
def label_block(ds: Dataset, mode: str, rows: np.ndarray):
    """(Y [M,2], Mk [M,2], A [M,5]) float32 for global rows.  A has NaN where a target is n.a."""
    yl, ys = ds.lab[f"{mode}_long"][rows], ds.lab[f"{mode}_short"][rows]
    Y = np.stack([yl, ys], axis=1)
    Mk = np.isfinite(Y).astype("float32")
    Y = np.nan_to_num(Y, nan=0.0).astype("float32")
    A = np.empty((len(rows), len(AUX_COLS)), dtype="float32")
    for j, nm in enumerate(("fvol", "up_exc", "dn_exc")):
        A[:, j] = ds.aux[nm][rows]
    for j, side in ((3, "long"), (4, "short")):
        off = ds.off[f"{mode}_{side}"][rows].astype("float32")
        A[:, j] = np.where(off > 0, np.log1p(off), np.nan)
    return Y, Mk, A


def aux_stats(ds: Dataset, mode: str, anchors: np.ndarray, rng, n: int = 200_000) -> dict:
    sel = anchors if len(anchors) <= n else np.sort(rng.choice(anchors, n, replace=False))
    _, _, A = label_block(ds, mode, sel)
    mean = np.nanmean(A, axis=0)
    std = np.nanstd(A, axis=0)
    return dict(mean=np.nan_to_num(mean).astype("float32"), std=np.where(std > 1e-6, std, 1.0).astype("float32"))


def select_rows(ds: Dataset, rng_ms, mode: str, stride: int = 1, block: int = 4_000_000) -> np.ndarray:
    """Global rows with t in [lo, hi), a full window behind them and at least one label for ``mode``;
    every ``stride``-th row (strided on the global row id, so each symbol keeps its regular grid)."""
    lo, hi = rng_ms
    out = []
    N = len(ds)
    for s in range(0, N, block):
        e = min(s + block, N)
        t = np.asarray(ds.t[s:e])
        m = (t >= lo) & (t < hi) & (np.asarray(ds.pos[s:e]) >= ft.WINDOW - 1)
        if stride > 1:
            m &= (np.arange(s, e) % stride) == 0
        idx = np.nonzero(m)[0] + s
        if len(idx):
            idx = idx[ds.any_label(mode, idx)]
        out.append(idx)
    return np.concatenate(out) if out else np.zeros(0, dtype="int64")


# --------------------------------------------------------------------------------------------
# GPU residency + chunks
# --------------------------------------------------------------------------------------------
class GpuData:
    """The whole fp16 feature matrix resident on mps (uploaded once per process)."""

    def __init__(self, X):
        self.src = X
        self.X = torch.from_numpy(np.ascontiguousarray(X)).to(DEVICE)
        self.offs = torch.from_numpy(OFFS).to(DEVICE)

    def windows(self, idx_t):
        return self.X[idx_t[:, None] + self.offs[None, :]]


_GPU: dict[int, GpuData] = {}


def gpu_data(X) -> "GpuData | None":
    """Cached GPU copy of X, or None when it would not fit the VRAM budget (-> streaming)."""
    if X.nbytes > GPU_MAX_BYTES:
        return None
    g = _GPU.get(id(X))
    if g is None or g.src is not X:
        _GPU.clear()
        free_gpu()
        g = _GPU[id(X)] = GpuData(X)
    return g


def _norm_t(mean, std):
    return (torch.from_numpy(np.asarray(mean, "float32")).to(DEVICE), torch.from_numpy(np.asarray(std, "float32")).to(DEVICE))


def _normalise(x, mean_t, std_t):
    return ((x.float() - mean_t) / std_t).clamp_(-ft.NORM_CLIP, ft.NORM_CLIP)


class _HostChunk:
    __slots__ = ("X", "anchors", "Y", "Mk", "A")


class ChunkSource:
    """Yields device chunks (X rows on mps, local anchor indices, targets).  Resident: one chunk, forever."""

    def __init__(self, ds: Dataset, mode: str, anchors: np.ndarray, cfg: TrainCfg, astat: dict | None):
        self.ds, self.mode, self.anchors, self.cfg, self.astat = ds, mode, anchors, cfg, astat
        self.g = gpu_data(ds.X)
        self.rng = np.random.default_rng(cfg.seed + 11)
        self.q: queue.Queue | None = None
        self._stop = threading.Event()
        self._fixed = None
        if self.g is None:
            self.mask = np.zeros(len(ds), dtype=bool)
            self.mask[anchors] = True
            self.q = queue.Queue(maxsize=2)
            self.thread = threading.Thread(target=self._produce, daemon=True)
            self.thread.start()

    @property
    def resident(self) -> bool:
        return self.g is not None

    # -- host side ---------------------------------------------------------------------------
    def _prep_aux(self, A):
        if self.astat is None:
            return np.zeros((len(A), 0), "float32"), np.zeros((len(A), 0), "float32")
        z = (A - self.astat["mean"]) / self.astat["std"]
        m = np.isfinite(z).astype("float32")
        return np.nan_to_num(z).astype("float32"), m

    def _make_host_chunk(self) -> "_HostChunk":
        ds, cfg = self.ds, self.cfg
        N, L, W = len(ds), cfg.block_len, ft.WINDOW
        pick = self.anchors[self.rng.integers(0, len(self.anchors), cfg.chunk_blocks)]
        r0s = np.clip(pick - (W - 1) - self.rng.integers(0, L - W, len(pick)), 0, N - L)
        Xs = np.empty((len(r0s) * L, ds.n_feat), dtype="float16")
        rows, loc = [], []
        for b, r0 in enumerate(r0s):
            Xs[b * L:(b + 1) * L] = ds.X[r0:r0 + L]
            m = self.mask[r0:r0 + L].copy()
            m[:W - 1] = False
            j = np.nonzero(m)[0]
            rows.append(r0 + j)
            loc.append(b * L + j)
        rows, loc = np.concatenate(rows), np.concatenate(loc)
        h = _HostChunk()
        h.X, h.anchors = Xs, loc.astype("int64")
        h.Y, h.Mk, A = label_block(ds, self.mode, rows)
        h.A = self._prep_aux(A)
        return h

    def _produce(self):
        try:
            while not self._stop.is_set():
                self.q.put(self._make_host_chunk())
        except Exception as e:                       # surfaced in next()
            self.q.put(e)

    # -- device side -------------------------------------------------------------------------
    def next(self):
        if self.resident:
            if self._fixed is None:
                Y, Mk, A = label_block(self.ds, self.mode, self.anchors)
                Az, Am = self._prep_aux(A)
                self._fixed = (self.g.X, torch.from_numpy(self.anchors.astype("int64")).to(DEVICE),
                               torch.from_numpy(Y).to(DEVICE), torch.from_numpy(Mk).to(DEVICE),
                               torch.from_numpy(Az).to(DEVICE), torch.from_numpy(Am).to(DEVICE))
            return self._fixed
        h = self.q.get()
        if isinstance(h, Exception):
            raise h
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE)  # noqa: E731
        return (t(h.X), t(h.anchors), t(h.Y), t(h.Mk), t(h.A[0]), t(h.A[1]))

    def close(self):
        self._stop.set()
        if self.q is not None:
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass


# --------------------------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------------------------
def predict_logits(net, ds_or_X, idx, norm, batch: int = 4096) -> np.ndarray:
    """Raw logits [len(idx), n_out] on mps, eval mode.  ``ds_or_X``: Dataset or the feature matrix."""
    X = ds_or_X.X if hasattr(ds_or_X, "X") else ds_or_X
    net.eval()
    mean_t, std_t = _norm_t(norm["mean"], norm["std"])
    g = gpu_data(X)
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            ii = idx[i:i + batch]
            if g is not None:
                x = g.windows(torch.from_numpy(ii.astype("int64")).to(DEVICE))
            else:
                x = torch.from_numpy(gather(X, ii)).to(DEVICE)
            out.append(net(_normalise(x, mean_t, std_t)).to("cpu").numpy())
    return np.concatenate(out, axis=0).astype("float64") if out else np.zeros((0, 2))


def predict_ensemble(nets, ds_or_X, idx, norm) -> np.ndarray:
    """Mean of the members' [long, short] logits -> [len(idx), 2]."""
    return sum(predict_logits(n, ds_or_X, idx, norm)[:, :2] for n in nets) / len(nets)


def masked_bce(logits, y, m, pos_w):
    l = Fn.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = m * (1.0 + (pos_w - 1.0) * y)
    return (l * w).sum() / m.sum().clamp_min(1.0)


def val_loss(logits: np.ndarray, Y: np.ndarray, Mk: np.ndarray) -> float:
    z = logits[:, :2]
    l = np.maximum(z, 0) - z * Y + np.log1p(np.exp(-np.abs(z)))
    return float((l * Mk).sum() / max(Mk.sum(), 1.0))


# --------------------------------------------------------------------------------------------
def train_model(ds: Dataset, mode: str, train_idx: np.ndarray, val_idx: np.ndarray, norm: dict, cfg: TrainCfg,
                log=print):
    """Train one model for ``mode``.  idx = global rows (valid windows + a label).  Early stopping restores
    the best-validation weights (validation = main-head BCE only, so aux and non-aux runs compare)."""
    require_mps()
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    ac = dict(cfg.arch_cfg)
    ac["n_out"] = 2 + len(AUX_COLS) if cfg.aux else 2
    net, arch_cfg = M.build(cfg.arch, ds.n_feat, ft.WINDOW, ac)
    net = net.to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    ysamp = label_block(ds, mode, train_idx if len(train_idx) < 300_000 else
                        np.sort(np_rng.choice(train_idx, 300_000, replace=False)))
    if cfg.pos_weight == "auto":
        pr = (ysamp[0] * ysamp[1]).sum(0) / np.maximum(ysamp[1].sum(0), 1)
        pw = np.clip((1 - pr) / np.maximum(pr, 1e-3), 0.5, 4.0)
    else:
        pw = np.ones(2)
    pos_w = torch.from_numpy(pw.astype("float32")).to(DEVICE)
    mean_t, std_t = _norm_t(norm["mean"], norm["std"])
    astat = aux_stats(ds, mode, train_idx, np_rng) if cfg.aux else None

    if len(val_idx) > cfg.val_max:
        val_idx = np.sort(np_rng.choice(val_idx, cfg.val_max, replace=False))
    Yv, Mv, _ = label_block(ds, mode, val_idx)
    src = ChunkSource(ds, mode, train_idx, cfg, astat)
    chunk_steps = cfg.steps if src.resident else cfg.chunk_steps
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    step = 0
    stop = False
    try:
        while step < cfg.steps and not stop:
            Xg, anc, Yg, Mg, Ag, Amg = src.next()
            for _ in range(min(chunk_steps, cfg.steps - step)):
                step += 1
                net.train()
                bi = torch.randint(0, len(anc), (cfg.batch,), device=DEVICE)
                rows = anc[bi]
                x = _normalise(Xg[rows[:, None] + src_offs(Xg)], mean_t, std_t)
                out = net(x)
                loss = masked_bce(out[:, :2], Yg[bi], Mg[bi], pos_w)
                if cfg.aux:
                    d = Fn.smooth_l1_loss(out[:, 2:], Ag[bi], reduction="none") * Amg[bi]
                    loss = loss + cfg.aux_w * d.sum() / Amg[bi].sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad)
                opt.step()
                sched.step()
                if step % cfg.eval_every == 0 or step == cfg.steps:
                    lv = float(loss.item())              # sync point (only here)
                    if not np.isfinite(lv):
                        raise FloatingPointError("training loss is not finite")
                    lg = predict_logits(net, ds, val_idx, norm)
                    vl = val_loss(lg, Yv, Mv)
                    hist.append(dict(step=step, train_loss=lv, val_loss=vl, elapsed_s=round(time.time() - t0, 1)))
                    log(f"  [{cfg.arch}/{mode}/s{cfg.seed}{'/aux' if cfg.aux else ''}] step {step:5d} "
                        f"train {lv:.4f} val {vl:.4f} ({hist[-1]['elapsed_s']}s)")
                    if vl < best - 1e-5:
                        best, bad = vl, 0
                        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= cfg.patience:
                            log(f"  early stop at step {step} (best val {best:.4f})")
                            stop = True
                            break
    finally:
        src.close()
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    total = time.time() - t0
    n_steps = hist[-1]["step"] if hist else 0
    return net, dict(history=hist, best_val_loss=float(best), arch_cfg=arch_cfg, seconds=round(total, 1),
                     steps_run=n_steps, ms_per_step=round(1000 * total / max(n_steps, 1), 1),
                     pos_weight=pw.tolist(), streaming=not src.resident, aux=cfg.aux, seed=cfg.seed)


_OFFS_T: dict = {}


def src_offs(Xg):
    """Window offsets as a device tensor (cached)."""
    o = _OFFS_T.get("o")
    if o is None:
        o = _OFFS_T["o"] = torch.from_numpy(OFFS).to(DEVICE)[None, :]
    return o


def free_gpu():
    import gc
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
