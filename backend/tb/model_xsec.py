"""Cross-sectional ranker (experiment A): artifact, numpy/sklearn inference and the ``Scorer`` for ``kind="xsec"``.

Artifact (``models/{name}.json`` + weights)::

    json : name, version, created_at, status ("challenger" on export), kind "xsec", arch ("mlp"|"gbm"),
           timeframes ["5m","1h"], history_bars_5m, feature_names (tb.features_xsec.MODEL_FEATURES),
           rebalance_every_h, top_k, bottom_k, horizon_h, entry_cfg {entry_mode, maker_timeout_s, maker_reprices,
           decision_every_bars, ...}, xsec_cfg (copy of the four xsec fields + entry_mode), exit_cfg {exit_mode "pct",
           stop_vol_mult, stop_min, stop_max}, universe {min_daily_qv_usd, min_age_h, ...}, norm, metrics, note
    npz  : mlp weights (torch names: l0.weight [H, F], l0.bias, l1.weight, l1.bias, l2.weight [1, H2], l2.bias)
    pkl  : gbm = pickled sklearn HistGradientBoostingRegressor (loaded only from our own models/ directory)

``load(path) -> XsecScorer``.  ``score_universe(frames)`` ranks the whole tradable universe at once: ``frames`` maps
symbol -> FrameBundle (an object with ``klines``/``metrics``/``funding`` frames, a dict with those keys, or a
``(klines, metrics, funding)`` tuple) holding at least ``history_bars_5m`` five-minute bars (BTCUSDT should be among
them: it is the residual-momentum reference).  The result maps symbol -> Score where, for tradable symbols,
``prob_long`` = cross-sectional percentile of the predicted relative return (1.0 = best), ``prob_short`` = 1 -
percentile; symbols that are not tradable (too young, too illiquid, no fresh data) get Score(0, 0).  The engine goes
long the ``top_k`` highest ``prob_long`` and short the ``bottom_k`` highest ``prob_short`` (``target_portfolio``).

numpy-only at import (sklearn is imported lazily when a gbm artifact is loaded).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features_xsec as fx

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

KIND = "xsec"
DEFAULT_UNIVERSE = {"min_daily_qv_usd": fx.MIN_DAILY_QV_USD, "min_age_h": fx.MIN_AGE_H,
                    "age_ok": fx.AGE_WINDOW_OK, "liq_window_h": fx.LIQ_WINDOW_H}


def silu(x):
    return x / (1.0 + np.exp(-x))


class MlpRanker:
    """Small MLP (F -> H -> H2 -> 1), numpy forward; weights in torch layout.  ``members``: one weight dict or a list
    (seed ensemble, scores averaged).  Inputs are standardised with ``mean``/``std`` (training statistics)."""

    def __init__(self, members, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        if isinstance(members, dict):
            members = [members]
        self.members = [{k: np.asarray(v, dtype="float32") for k, v in W.items()} for W in members]
        self.mean = None if mean is None else np.asarray(mean, dtype="float32")
        self.std = None if std is None else np.asarray(std, dtype="float32")

    def _one(self, W: dict, x: np.ndarray) -> np.ndarray:
        h = silu(x @ W["l0.weight"].T + W["l0.bias"])
        h = silu(h @ W["l1.weight"].T + W["l1.bias"])
        return (h @ W["l2.weight"].T + W["l2.bias"])[:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype="float32")
        if self.mean is not None:
            x = (x - self.mean) / self.std
        x = np.clip(x, -10.0, 10.0)
        return (sum(self._one(W, x) for W in self.members) / len(self.members)).astype("float64")


class GbmRanker:
    """sklearn HistGradientBoostingRegressor evaluated at its validation-chosen number of stages."""

    def __init__(self, model, n_stages: int | None = None):
        self.model = model
        self.n_stages = n_stages

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        n = self.n_stages
        total = getattr(self.model, "n_iter_", None)
        if n is None or total is None or n >= total:
            return np.asarray(self.model.predict(X), dtype="float64")
        out = None
        for i, p in enumerate(self.model.staged_predict(X), start=1):
            out = p
            if i >= n:
                break
        return np.asarray(out, dtype="float64")


def entry_cfg_dict(rebalance_every_h: int, entry_mode: str, timeout_bars: int = 3, reprices: int = 2,
                   on_timeout: str = "skip") -> dict:
    return {"decision_every_bars": int(rebalance_every_h) * 12, "cooldown_bars": 0, "max_trades_per_day": None,
            "entry_mode": entry_mode, "maker_timeout_s": int(timeout_bars) * 300, "maker_reprices": int(reprices),
            "maker_on_timeout": on_timeout}


def make_meta(name: str, arch: str, rebalance_every_h: int, top_k: int, bottom_k: int, horizon_h: int,
              entry_mode: str, version: str | None = None, created_at: str | None = None, metrics: dict | None = None,
              note: str | None = None, stop_cfg: dict | None = None, norm: dict | None = None,
              extra: dict | None = None, timeout_bars: int = 3, reprices: int = 2) -> dict:
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    meta = {
        "name": name, "version": version or now.strftime("%Y%m%d.%H%M"),
        "created_at": created_at or now.isoformat(), "status": "challenger", "kind": KIND, "arch": arch,
        "timeframes": ["5m", "1h"], "history_bars_5m": fx.HISTORY_BARS_5M,
        "feature_names": list(fx.MODEL_FEATURES), "base_features": list(fx.BASE_FEATURES),
        "rebalance_every_h": int(rebalance_every_h), "top_k": int(top_k), "bottom_k": int(bottom_k),
        "horizon_h": int(horizon_h), "entry_mode": entry_mode,
        "xsec_cfg": {"rebalance_every_h": int(rebalance_every_h), "top_k": int(top_k), "bottom_k": int(bottom_k),
                     "horizon_h": int(horizon_h), "entry_mode": entry_mode},
        "entry_cfg": entry_cfg_dict(rebalance_every_h, entry_mode, timeout_bars, reprices),
        "exit_cfg": {"exit_mode": "pct", "stop_vol_mult": 2.0, "stop_min": 0.02, "stop_max": 0.20,
                     "note": "stop distance = stop_vol_mult x trailing 7d daily vol, clipped; legs are closed by the "
                             "next rebalance (no take-profit)"},
        "universe": dict(DEFAULT_UNIVERSE), "norm": norm or {},
        "thresholds": {}, "metrics": metrics or {},
        "score_semantics": "prob_long = cross-sectional percentile of the predicted 24h-relative return; "
                           "prob_short = 1 - prob_long; trade the top_k / bottom_k",
    }
    if stop_cfg:
        meta["exit_cfg"].update(stop_cfg)
    if note:
        meta["note"] = note
    if extra:
        meta.update(extra)
    return meta


def save_mlp(base: str | Path, meta: dict, ranker: MlpRanker) -> Path:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, arch="mlp", ensemble=len(ranker.members))
    arrs = {f"e{i}/{k}": np.asarray(v, dtype="float32") for i, W in enumerate(ranker.members) for k, v in W.items()}
    if ranker.mean is not None:
        arrs["in.mean"] = ranker.mean
        arrs["in.std"] = ranker.std
    np.savez(str(base) + ".npz", **arrs)
    Path(str(base) + ".json").write_text(json.dumps(meta, indent=1))
    return Path(str(base) + ".json")


def save_gbm(base: str | Path, meta: dict, ranker: GbmRanker) -> Path:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, arch="gbm", n_stages=ranker.n_stages)
    with open(str(base) + ".pkl", "wb") as f:
        pickle.dump(ranker.model, f, protocol=4)
    Path(str(base) + ".json").write_text(json.dumps(meta, indent=1))
    return Path(str(base) + ".json")


# --------------------------------------------------------------------------------------------
@dataclass
class Inputs:
    syms: list[str]                 # all symbols in the panel
    tradable: np.ndarray            # [S] bool at the decision hour
    X: np.ndarray                   # [S, F_MODEL] (zeros where not tradable)
    decision_ms: int
    k: int
    panel: fx.Panel


class XsecScorer:
    kind = KIND

    def __init__(self, meta: dict, ranker):
        if meta.get("kind", KIND) != KIND:
            raise ValueError(f"not an xsec artifact: kind={meta.get('kind')!r}")
        self.meta = meta
        self.name = str(meta.get("name", "xsec"))
        self.version = str(meta.get("version", ""))
        self.status = meta.get("status", "challenger")
        self.parent = meta.get("parent")
        self.arch = meta["arch"]
        self.ranker = ranker
        self.history_bars_5m = int(meta.get("history_bars_5m", fx.HISTORY_BARS_5M))
        self.timeframes = list(meta.get("timeframes", ["5m", "1h"]))
        self.feature_names = list(meta.get("feature_names", fx.MODEL_FEATURES))
        if self.feature_names != fx.MODEL_FEATURES:
            raise ValueError("model feature_names differ from tb.features_xsec.MODEL_FEATURES (retrain)")
        self.rebalance_every_h = int(meta["rebalance_every_h"])
        self.top_k = int(meta["top_k"])
        self.bottom_k = int(meta["bottom_k"])
        self.horizon_h = int(meta["horizon_h"])
        self.entry_mode = meta.get("entry_mode") or (meta.get("entry_cfg") or {}).get("entry_mode", "taker")
        self.entry_cfg = meta.get("entry_cfg", {})
        self.exit_cfg = meta.get("exit_cfg", {})
        self.universe = dict(DEFAULT_UNIVERSE, **(meta.get("universe") or {}))
        self.thresholds = meta.get("thresholds", {})
        self.metrics = meta.get("metrics", {})

    # -- per-symbol API is not meaningful for a cross-sectional model
    def score(self, symbol: str, feats: np.ndarray) -> Score:
        return Score(0.0, 0.0, [])

    # -- feature building (ONE code path shared with training: tb.features_xsec) --------------------
    def build_inputs(self, frames, asof_ms: int | None = None) -> Inputs:
        syms = sorted(frames)
        if not syms:
            raise ValueError("no frames")
        lasts = []
        cut = {}
        for s in syms:
            kl, m, f = fx._unpack(frames[s])
            if kl is None or len(kl) == 0:
                cut[s] = (None, m, f)
                continue
            t = fx._ms_col(kl["open_time"])
            if asof_ms is not None:
                kl = kl.loc[t + fx.BAR_MS <= asof_ms]
                t = t[t + fx.BAR_MS <= asof_ms]
            if len(t) == 0:
                cut[s] = (None, m, f)
                continue
            lasts.append(int(t.max()))
            cut[s] = (kl, m, f)
        if not lasts:
            raise ValueError("no klines")
        end = (max(lasts) + fx.BAR_MS) // fx.HOUR_MS * fx.HOUR_MS          # end of the last COMPLETE hour
        T = max(1, self.history_bars_5m // fx.SUB)
        t0 = end - T * fx.HOUR_MS
        panel = fx.build_panel(cut, t0=t0, T=T, symbols=syms)
        cube = fx.compute_raw_features(panel)
        u = self.universe
        mask = fx.tradable_mask(panel, u["min_daily_qv_usd"], u["min_age_h"], u["age_ok"], u["liq_window_h"])
        k = T - 1
        X = fx.xs_transform(cube[k:k + 1], mask[k:k + 1], fx.btc_index(panel.syms))[0]
        return Inputs(panel.syms, mask[k], X, int(panel.decision_ms(k)), k, panel)

    def _scores(self, inp: Inputs, with_reasons: bool = True) -> dict[str, Score]:
        idx = np.nonzero(inp.tradable)[0]
        out = {s: Score(0.0, 0.0, []) for s in inp.syms}
        if len(idx) == 0:
            return out
        Xt = inp.X[idx]
        pred = self.ranker.predict(Xt)
        n = len(idx)
        pct = (pd_rank(pred) - 0.5) / n                         # percentile of the predicted relative return
        reasons = self._reasons(Xt, pred) if with_reasons else [[] for _ in range(n)]
        for j, i in enumerate(idx):
            out[inp.syms[i]] = Score(float(pct[j]), float(1.0 - pct[j]), reasons[j])
        return out

    def score_universe(self, frames, asof_ms: int | None = None, with_reasons: bool = True) -> dict[str, Score]:
        return self._scores(self.build_inputs(frames, asof_ms), with_reasons)

    def target_portfolio(self, frames, asof_ms: int | None = None) -> dict:
        """{"long": [...], "short": [...], "n_universe": n, "decision_ms": ms, "scores": {sym: pct}}"""
        inp = self.build_inputs(frames, asof_ms)
        sc = self._scores(inp, with_reasons=False)
        ranked = sorted(((v.prob_long, s) for s, v in sc.items() if v.prob_long > 0 or v.prob_short > 0), reverse=True)
        longs = [s for _, s in ranked[:self.top_k]]
        shorts = [s for _, s in ranked[::-1][:self.bottom_k] if s not in longs]
        return {"long": longs, "short": shorts, "n_universe": len(ranked), "decision_ms": inp.decision_ms,
                "scores": {s: p for p, s in ranked}}

    def _reasons(self, Xt: np.ndarray, pred: np.ndarray, top: int = 3) -> list[list[Reason]]:
        """Occlusion by feature group: zero (= the universe median) the group's columns and measure the score change."""
        names = self.feature_names
        col = {n: i for i, n in enumerate(names)}
        deltas = []
        gnames = []
        for g, members in fx.FEATURE_GROUPS.items():
            cols = [col[f"{m}{suf}"] for m in members for suf in ("_z", "_r") if f"{m}{suf}" in col]
            if not cols:
                continue
            Xo = Xt.copy()
            Xo[:, cols] = 0.0
            deltas.append(pred - self.ranker.predict(Xo))
            gnames.append((g, members))
        D = np.stack(deltas, axis=1) if deltas else np.zeros((len(pred), 0))
        res = []
        for j in range(len(pred)):
            order = np.argsort(-np.abs(D[j]))[:top]
            rs = []
            for gi in order:
                g, members = gnames[gi]
                zc = [col[f"{m}_z"] for m in members]
                dom = int(np.argmax(np.abs(Xt[j, zc])))
                z = float(Xt[j, zc[dom]])
                rs.append(Reason(g, f"{g}: {members[dom]} z={z:+.1f}", z, float(D[j, gi])))
            res.append(rs)
        return res


def pd_rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1..n) of x ascending, ties averaged."""
    import pandas as pd
    return pd.Series(np.asarray(x, dtype="float64")).rank(method="average").to_numpy()


def load(path) -> XsecScorer:
    """``path`` may be the .json, the weights file or the bare model path."""
    p = Path(path)
    if p.suffix in (".json", ".npz", ".pkl"):
        p = p.with_suffix("")
    meta = json.loads(Path(str(p) + ".json").read_text())
    arch = meta["arch"]
    if arch == "mlp":
        with np.load(str(p) + ".npz") as z:
            n = int(meta.get("ensemble", 1))
            members = [{k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(f"e{i}/")} for i in range(n)]
            mean = z["in.mean"] if "in.mean" in z.files else None
            std = z["in.std"] if "in.std" in z.files else None
        ranker = MlpRanker(members, mean, std)
    elif arch == "gbm":
        with open(str(p) + ".pkl", "rb") as f:
            ranker = GbmRanker(pickle.load(f), meta.get("n_stages"))
    else:
        raise ValueError(f"unknown xsec arch {arch!r}")
    return XsecScorer(meta, ranker)
