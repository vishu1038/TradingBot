"""Torch architectures (train on `mps` ONLY - see README: CPU math crashes on this machine).

Parameter names are the export contract with ``tb.model_np`` (state_dict keys == npz keys), and the
forward maths here must match ``tb.model_np.forward_*`` to ~1e-4.

All models: input [B, T, F] normalised float32 -> logits [B, 2] (long, short).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEVICE = "mps"

DEFAULT_CFG = {
    "cnn": {"channels": 24, "dilations": [1, 2, 4, 8], "kernel": 3, "hidden": 32, "dropout": 0.4},
    "gru": {"hidden": 32, "layers": 1, "dropout": 0.3, "head_hidden": 32},
    "transformer": {"d_model": 48, "heads": 4, "layers": 2, "ff": 96, "dropout": 0.1, "head_hidden": 64},
    "mlp": {"hidden": 128, "hidden2": 64, "dropout": 0.3},
}


class Head(nn.Module):
    """Two-layer MLP head, parameters exposed as head1/head2 on the parent."""


class CNN(nn.Module):
    def __init__(self, n_feat: int, channels=64, dilations=(1, 2, 4, 8, 16), kernel=3, hidden=64, dropout=0.1, n_out=2):
        super().__init__()
        self.kernel, self.dilations = kernel, list(dilations)
        self.inp = nn.Conv1d(n_feat, channels, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(channels, channels, kernel, dilation=d) for d in self.dilations])
        self.drop = nn.Dropout(dropout)
        self.head1 = nn.Linear(3 * channels, hidden)
        self.head2 = nn.Linear(hidden, n_out)

    def forward(self, x):                                  # [B,T,F]
        h = self.inp(x.transpose(1, 2))
        for conv, d in zip(self.blocks, self.dilations):
            y = conv(Fn.pad(h, ((self.kernel - 1) * d, 0)))      # causal
            h = h + self.drop(Fn.silu(y))
        feat = torch.cat([h[:, :, -1], h.mean(2), h.amax(2)], dim=1)
        return self.head2(self.drop(Fn.silu(self.head1(feat))))


class GRU(nn.Module):
    def __init__(self, n_feat: int, hidden=64, layers=1, dropout=0.1, head_hidden=64, n_out=2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head1 = nn.Linear(2 * hidden, head_hidden)
        self.head2 = nn.Linear(head_hidden, n_out)

    def forward(self, x):
        out, _ = self.gru(x)
        feat = torch.cat([out[:, -1], out.mean(1)], dim=1)
        return self.head2(self.drop(Fn.silu(self.head1(feat))))


class Block(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        B, T, D = h.shape
        nh = self.heads
        dh = D // nh
        q, k, v = self.qkv(self.ln1(h)).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, T, nh, dh).transpose(1, 2) for t in (q, k, v))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, D)
        h = h + self.drop(self.proj(o))
        return h + self.drop(self.ff2(Fn.gelu(self.ff1(self.ln2(h)), approximate="tanh")))


class Transformer(nn.Module):
    def __init__(self, n_feat: int, window=64, d_model=48, heads=4, layers=2, ff=96, dropout=0.1, head_hidden=64,
                 n_out=2):
        super().__init__()
        self.embed = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(window, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([Block(d_model, heads, ff, dropout) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head1 = nn.Linear(2 * d_model, head_hidden)
        self.head2 = nn.Linear(head_hidden, n_out)

    def forward(self, x):
        h = self.embed(x) + self.pos[None, : x.shape[1]]
        for blk in self.layers:
            h = blk(h)
        h = self.lnf(h)
        feat = torch.cat([h[:, -1], h.mean(1)], dim=1)
        return self.head2(self.drop(Fn.silu(self.head1(feat))))


class MLP(nn.Module):
    """Multi-timeframe net: flat feature vector [B, F] -> [B, n_out] (long, short logits [+ aux]).
    Parameter names (fc1/fc2/head) are the export contract with ``tb.model_np.forward_mlp``."""

    def __init__(self, n_feat: int, hidden=128, hidden2=64, dropout=0.3, n_out=2):
        super().__init__()
        self.fc1 = nn.Linear(n_feat, hidden)
        self.fc2 = nn.Linear(hidden, hidden2)
        self.head = nn.Linear(hidden2, n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1]
        h = self.drop(Fn.silu(self.fc1(x)))
        h = self.drop(Fn.silu(self.fc2(h)))
        return self.head(h)


def build(arch: str, n_feat: int, window: int = 64, cfg: dict | None = None) -> tuple[nn.Module, dict]:
    """Construct on CPU (no math is run), return (module, full arch_cfg).  Caller does `.to("mps")`."""
    c = dict(DEFAULT_CFG[arch])
    c.update(cfg or {})
    if arch == "cnn":
        return CNN(n_feat, **c), c
    if arch == "gru":
        return GRU(n_feat, **c), c
    if arch == "mlp":
        return MLP(n_feat, **c), c
    if arch == "transformer":
        m = Transformer(n_feat, window=window, **c)
        c["window"] = window
        return m, c
    raise ValueError(arch)


def export_state(model: nn.Module) -> dict:
    """float32 numpy state dict (device->cpu copy, no math)."""
    return {k: v.detach().to("cpu").numpy().astype("float32") for k, v in model.state_dict().items()}
