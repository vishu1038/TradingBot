"""Pure-numpy inference for the exported entry models (no torch needed on the trading box).

Artifact (``models/{name}.npz`` + ``models/{name}.json``)::

    json : name, version, created_at, status ("champion"|"challenger"|"retired"), parent,
           arch ("cnn"|"gru"|"transformer"), arch_cfg {...}, feature_names, window,
           norm {mean[F], std[F], clip}, calibration {long:{...}, short:{...}},
           thresholds {long, short}, exit_cfg, metrics {...}
    npz  : the torch ``state_dict`` of the network as float32 arrays (key names == torch names).  For an
           ensemble (``json.ensemble = N > 1``) member k's arrays are stored as ``e{k}/<torch name>`` and the
           logits of the members are averaged before calibration.

The network may have extra multi-task outputs (``arch_cfg.n_out > 2``: volatility / excursion / time-to-hit
auxiliary heads, training-time only); inference uses output columns 0 and 1 = ``[long, short]`` logits.
Heads are two logits ``[long, short]``; ``prob = calibrate(sigmoid(logit))``.  Calibration is fitted
on validation data only and is part of the artifact so the live drift detector (realised win rate
vs mean predicted probability) is comparing like with like.

Calibration methods (per head), all numpy:
    {"method": "none"}
    {"method": "platt", "a": a, "b": b}              p = sigmoid(a * logit + b)
    {"method": "isotonic", "x": [...], "y": [...]}   p = interp(p_raw, x, y)   (PAV, monotone)

``load(path) -> Scorer``; ``Scorer.score(symbol, feats)`` returns ``Score(prob_long, prob_short,
reasons)`` where reasons are the top-5 features by |occlusion contribution| for the side with the
higher probability (feature i is "occluded" by replacing it with its training median over the whole
window; contribution = change in the calibrated logit, positive = pushes toward trading).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:                                            # the engine's dataclasses (same shape as below)
    from .scorer import Reason, Score
except Exception:                               # pragma: no cover - scorer.py not importable standalone
    @dataclass
    class Reason:                               # type: ignore[no-redef]
        feature: str
        label: str
        value: float
        contribution: float

    @dataclass
    class Score:                                # type: ignore[no-redef]
        prob_long: float = 0.0
        prob_short: float = 0.0
        reasons: list = field(default_factory=list)

from .features import WINDOW, feature_label, feature_names, normalize

STATUSES = ("champion", "challenger", "retired")


# --------------------------------------------------------------------------------------------
# math helpers
# --------------------------------------------------------------------------------------------
def sigmoid(x):
    x = np.asarray(x, dtype="float64")
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def gelu(x):                                    # tanh approximation (== torch gelu(approximate="tanh"))
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


# --------------------------------------------------------------------------------------------
# calibration (fit on validation only)
# --------------------------------------------------------------------------------------------
def fit_platt(logits: np.ndarray, y: np.ndarray, iters: int = 100, l2: float = 1e-6) -> dict:
    """Logistic regression of y on the raw logit: p = sigmoid(a*logit + b) (temperature scaling plus
    a bias term).  Newton iterations; falls back to identity on degenerate input."""
    z = np.asarray(logits, "float64")
    y = np.asarray(y, "float64")
    ok = np.isfinite(z) & np.isfinite(y)
    z, y = z[ok], y[ok]
    if len(z) < 50 or y.min() == y.max():
        return {"method": "none"}
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = sigmoid(a * z + b)
        g = np.array([((p - y) * z).sum() + l2 * (a - 1.0), (p - y).sum()])
        w = np.maximum(p * (1 - p), 1e-9)
        h = np.array([[(w * z * z).sum() + l2, (w * z).sum()], [(w * z).sum(), w.sum() + l2]])
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    if not (np.isfinite(a) and np.isfinite(b)):
        return {"method": "none"}
    return {"method": "platt", "a": float(a), "b": float(b)}


def fit_isotonic(p_raw: np.ndarray, y: np.ndarray, max_knots: int = 200) -> dict:
    """Pool-adjacent-violators regression of y on the raw probability, exported as <= max_knots
    piecewise-linear knots (x ascending, y non-decreasing)."""
    p = np.asarray(p_raw, "float64")
    y = np.asarray(y, "float64")
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < 200 or y.min() == y.max():
        return {"method": "none"}
    o = np.argsort(p, kind="stable")
    p, y = p[o], y[o]
    # PAV with blocks
    vals, wts, xs = [], [], []
    for xi, yi in zip(p, y):
        vals.append(yi)
        wts.append(1.0)
        xs.append([xi])
        while len(vals) > 1 and vals[-2] >= vals[-1]:
            w2 = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w2
            x2 = xs[-2] + xs[-1]
            vals[-2:], wts[-2:], xs[-2:] = [v2], [w2], [x2]
    kx = np.array([np.mean(b) for b in xs])
    ky = np.array(vals)
    if len(kx) > max_knots:
        sel = np.unique(np.linspace(0, len(kx) - 1, max_knots).astype(int))
        kx, ky = kx[sel], ky[sel]
    ky = np.clip(np.maximum.accumulate(ky), 1e-4, 1 - 1e-4)
    return {"method": "isotonic", "x": kx.tolist(), "y": ky.tolist()}


def fit_calibration(logits: np.ndarray, y: np.ndarray, method: str = "platt") -> dict:
    if method == "none":
        return {"method": "none"}
    if method == "isotonic":
        return fit_isotonic(sigmoid(logits), y)
    return fit_platt(logits, y)


def apply_calibration(cal: dict | None, logits) -> np.ndarray:
    """raw logits -> calibrated probabilities (float64)."""
    z = np.asarray(logits, "float64")
    m = (cal or {}).get("method", "none")
    if m == "platt":
        return sigmoid(cal["a"] * z + cal["b"])
    if m == "isotonic":
        return np.interp(sigmoid(z), np.asarray(cal["x"]), np.asarray(cal["y"]))
    return sigmoid(z)


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10, lo: float = 0.0, hi: float = 1.0) -> list[dict]:
    """Equal-width reliability table: [{lo, hi, n, mean_pred, frac_pos}] (empty bins omitted)."""
    p = np.asarray(p, "float64")
    y = np.asarray(y, "float64")
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        s = idx == b
        if s.any():
            out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(s.sum()),
                        "mean_pred": float(p[s].mean()), "frac_pos": float(y[s].mean())})
    return out


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (weighted mean |mean_pred - frac_pos|)."""
    bins = reliability_bins(p, y, n_bins)
    tot = sum(b["n"] for b in bins)
    return float(sum(b["n"] * abs(b["mean_pred"] - b["frac_pos"]) for b in bins) / tot) if tot else float("nan")


# --------------------------------------------------------------------------------------------
# forward passes.  x: [B, T, F] float32 (already normalised); returns raw logits [B, 2] float64
# --------------------------------------------------------------------------------------------
def _head(h, W):
    h = silu(h @ W["head1.weight"].T + W["head1.bias"])
    return h @ W["head2.weight"].T + W["head2.bias"]          # [B, n_out]; columns 0/1 = long/short logits


def forward_cnn(W: dict, cfg: dict, x: np.ndarray) -> np.ndarray:
    h = x.transpose(0, 2, 1).astype("float32")                    # [B,F,T]
    T = h.shape[2]
    h = np.einsum("oc,bct->bot", W["inp.weight"][:, :, 0], h) + W["inp.bias"][None, :, None]
    k = cfg["kernel"]
    for i, d in enumerate(cfg["dilations"]):
        w = W[f"blocks.{i}.weight"]                              # [C,C,k]
        pad = (k - 1) * d
        hp = np.pad(h, ((0, 0), (0, 0), (pad, 0)))
        y = np.zeros_like(h)
        for j in range(k):
            y += np.matmul(w[:, :, j], hp[:, :, j * d:j * d + T])
        y += W[f"blocks.{i}.bias"][None, :, None]
        h = h + silu(y)
    feat = np.concatenate([h[:, :, -1], h.mean(2), h.max(2)], axis=1)
    return _head(feat, W)


def forward_gru(W: dict, cfg: dict, x: np.ndarray) -> np.ndarray:
    B, T, _ = x.shape
    H = cfg["hidden"]
    seq = x.astype("float32")
    for l in range(cfg["layers"]):
        wi, wh = W[f"gru.weight_ih_l{l}"], W[f"gru.weight_hh_l{l}"]
        bi, bh = W[f"gru.bias_ih_l{l}"], W[f"gru.bias_hh_l{l}"]
        gi = seq @ wi.T + bi                                     # [B,T,3H] input part, all steps at once
        h = np.zeros((B, H), dtype="float32")
        outs = np.empty((B, T, H), dtype="float32")
        for t in range(T):
            gh = h @ wh.T + bh
            r = sigmoid(gi[:, t, :H] + gh[:, :H]).astype("float32")
            z = sigmoid(gi[:, t, H:2 * H] + gh[:, H:2 * H]).astype("float32")
            n = np.tanh(gi[:, t, 2 * H:] + r * gh[:, 2 * H:])
            h = ((1 - z) * n + z * h).astype("float32")
            outs[:, t] = h
        seq = outs
    feat = np.concatenate([seq[:, -1], seq.mean(1)], axis=1)
    return _head(feat, W)


def forward_transformer(W: dict, cfg: dict, x: np.ndarray) -> np.ndarray:
    B, T, _ = x.shape
    D, nh = cfg["d_model"], cfg["heads"]
    dh = D // nh
    h = x.astype("float32") @ W["embed.weight"].T + W["embed.bias"] + W["pos"][None, :T]
    for i in range(cfg["layers"]):
        p = f"layers.{i}."
        a = layer_norm(h, W[p + "ln1.weight"], W[p + "ln1.bias"])
        qkv = a @ W[p + "qkv.weight"].T + W[p + "qkv.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
        att = softmax(np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(dh))
        o = np.matmul(att, v).transpose(0, 2, 1, 3).reshape(B, T, D)
        h = h + o @ W[p + "proj.weight"].T + W[p + "proj.bias"]
        a = layer_norm(h, W[p + "ln2.weight"], W[p + "ln2.bias"])
        a = gelu(a @ W[p + "ff1.weight"].T + W[p + "ff1.bias"])
        h = h + a @ W[p + "ff2.weight"].T + W[p + "ff2.bias"]
    h = layer_norm(h, W["lnf.weight"], W["lnf.bias"])
    feat = np.concatenate([h[:, -1], h.mean(1)], axis=1)
    return _head(feat, W)


FORWARD = {"cnn": forward_cnn, "gru": forward_gru, "transformer": forward_transformer}


def forward(arch: str, W: dict, cfg: dict, x: np.ndarray, batch: int = 512) -> np.ndarray:
    """Raw logits [B,2] (float64).  Chunked so large calls stay memory-friendly."""
    fn = FORWARD[arch]
    outs = [fn(W, cfg, x[i:i + batch]) for i in range(0, len(x), batch)] or [np.zeros((0, 2))]
    return np.concatenate(outs, axis=0).astype("float64")


# --------------------------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------------------------
class NpScorer:
    def __init__(self, meta: dict, weights: dict[str, np.ndarray]):
        self.meta = meta
        self.name = str(meta.get("name", "model"))
        self.version = str(meta.get("version", ""))
        self.status = meta.get("status", "challenger")
        self.parent = meta.get("parent")
        self.arch = meta["arch"]
        if self.arch not in FORWARD:
            raise ValueError(f"unknown arch {self.arch!r}")
        self.arch_cfg = meta["arch_cfg"]
        self.n_members = int(meta.get("ensemble", 1) or 1)
        W32 = {k: np.asarray(v, dtype="float32") for k, v in weights.items()}
        if self.n_members > 1:
            self.members = [{k.split("/", 1)[1]: v for k, v in W32.items() if k.startswith(f"e{m}/")}
                            for m in range(self.n_members)]
            if not all(self.members):
                raise ValueError("ensemble weights missing for some member")
        else:
            self.members = [W32]
        self.W = self.members[0]
        self.feature_groups = list(meta.get("feature_groups", []))
        if "depth" in self.feature_groups:
            from . import features_depth  # noqa: F401  (registers the group)
        self.feature_names = list(meta.get("feature_names", feature_names(self.feature_groups)))
        if self.feature_names != feature_names(self.feature_groups):
            raise ValueError("model feature_names differ from tb.features (retrain, or register the model's "
                             "extra feature groups before loading)")
        self.window = int(meta.get("window", WINDOW))
        norm = meta["norm"]
        self.mean = np.asarray(norm["mean"], dtype="float32")
        self.std = np.asarray(norm["std"], dtype="float32")
        self.clip = float(norm.get("clip", 5.0))
        self.calibration = meta.get("calibration") or {}
        self.thresholds = meta.get("thresholds", {})
        self.exit_cfg = meta.get("exit_cfg", {})
        self.metrics = meta.get("metrics", {})
        self.n_reasons = 5
        self._F = len(self.feature_names)

    # -- batch API (backtester / evaluation) ------------------------------------------------
    def logits_raw_windows(self, x_raw: np.ndarray) -> np.ndarray:
        """Raw (unnormalised) feature windows [B, T, F] -> uncalibrated logits [B, 2]."""
        z = normalize(x_raw, self.mean, self.std, self.clip)
        return self._logits(z)

    def _logits(self, z: np.ndarray) -> np.ndarray:
        """Ensemble-averaged raw [long, short] logits for normalised windows [B, T, F]."""
        lg = sum(forward(self.arch, W, self.arch_cfg, z)[:, :2] for W in self.members) / len(self.members)
        return lg

    def predict_windows(self, x_raw: np.ndarray) -> np.ndarray:
        """Raw feature windows [B, T, F] -> calibrated probabilities [B, 2] (long, short)."""
        lg = self.logits_raw_windows(x_raw)
        return np.stack([self._cal("long", lg[:, 0]), self._cal("short", lg[:, 1])], axis=1)

    def _cal(self, side: str, lg):
        return apply_calibration(self.calibration.get(side), lg)

    def _cal_slope(self, side: str) -> float:
        c = self.calibration.get(side) or {}
        return float(c.get("a", 1.0)) if c.get("method") == "platt" else 1.0

    # -- live API -----------------------------------------------------------------------------
    def _window(self, feats: np.ndarray) -> np.ndarray:
        f = np.asarray(feats, dtype="float32")
        if f.ndim != 2 or f.shape[1] != self._F:
            raise ValueError(f"feats must be [T, {self._F}], got {f.shape}")
        if len(f) < self.window:                          # short history: repeat the oldest row
            f = np.concatenate([np.repeat(f[:1], self.window - len(f), axis=0), f], axis=0)
        return f[-self.window:]

    def score(self, symbol: str, feats: np.ndarray, with_reasons: bool = True) -> Score:
        w = self._window(feats)
        z = normalize(w, self.mean, self.std, self.clip)
        if not with_reasons:
            lg = self._logits(z[None])[0]
            return Score(float(self._cal("long", lg[0])), float(self._cal("short", lg[1])), [])
        # batch = the window itself + one copy per feature with that feature occluded (set to the
        # normalised median, 0) across the whole window.
        batch = np.repeat(z[None], self._F + 1, axis=0)
        for j in range(self._F):
            batch[j + 1, :, j] = 0.0
        lg = self._logits(batch)
        p_long = float(self._cal("long", lg[0, 0]))
        p_short = float(self._cal("short", lg[0, 1]))
        side = 0 if p_long >= p_short else 1
        a = self._cal_slope("long" if side == 0 else "short")
        contrib = a * (lg[0, side] - lg[1:, side])
        order = np.argsort(-np.abs(contrib))[: self.n_reasons]
        last = np.asarray(feats[-1], dtype="float32")
        reasons = [Reason(self.feature_names[j], feature_label(self.feature_names[j], float(last[j])),
                          float(last[j]), float(contrib[j])) for j in order]
        return Score(p_long, p_short, reasons)


def load(path) -> NpScorer:
    """``path`` may be the ``.json``, the ``.npz`` or the bare model path (no suffix)."""
    p = Path(path)
    if p.suffix in (".json", ".npz"):
        p = p.with_suffix("")
    meta = json.loads(Path(str(p) + ".json").read_text())
    with np.load(str(p) + ".npz") as z:
        weights = {k: z[k] for k in z.files}
    return NpScorer(meta, weights)
