"""Scorer interface between the engine and the model, plus the champion/challenger registry.

``NullScorer`` (probabilities 0, never trades) is used when no ``models/*.json`` exists or when
loading fails. Model status lives in ``models/*.json`` (``status``: champion | challenger |
retired) and can be overridden at runtime through the store (promote / rollback), see ``tb.shadow``.
"""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger("tb.scorer")


@dataclass
class Reason:
    feature: str
    label: str
    value: float
    contribution: float


@dataclass
class Score:
    prob_long: float = 0.0
    prob_short: float = 0.0
    reasons: list[Reason] = field(default_factory=list)


KIND_PER_SYMBOL = "per_symbol_5m"
KINDS = (KIND_PER_SYMBOL, "mtf", "gbm", "xsec")


@dataclass
class FrameBundle:
    """Everything one symbol's model input is built from (live and training share ``build_for_model``).

    ``kl`` holds CLOSED 5m bars in the archive schema (latest last, at least the model's ``history_bars_5m``
    when the exchange has that much); ``metrics`` / ``funding`` are the archive-schema derivative frames;
    ``depth`` is the 5m order-book-band frame when a loaded model uses the depth group; ``btc_kl`` is
    BTCUSDT's closed 5m bars (market-context features). Field names match ``tb.features_mtf.FrameBundle``;
    ``klines`` / ``btc`` / ``btc_klines`` are read-only aliases (``tb.features_xsec`` reads ``klines``)."""
    symbol: str
    kl: "pd.DataFrame"
    metrics: "pd.DataFrame | None" = None
    funding: "pd.DataFrame | None" = None
    depth: "pd.DataFrame | None" = None
    btc_kl: "pd.DataFrame | None" = None

    @property
    def klines(self) -> "pd.DataFrame":
        return self.kl

    @property
    def btc(self) -> "pd.DataFrame | None":
        return self.btc_kl

    @property
    def btc_klines(self) -> "pd.DataFrame | None":
        return self.btc_kl


class Scorer(Protocol):
    """``score`` is the per-symbol interface (a ``[T, F]`` window). Scorer interface v2 adds the optional
    ``kind``, ``history_bars_5m`` and ``score_universe(frames)`` (cross-sectional models need all coins at
    once); a scorer without them is treated as a per-symbol 5m model (see ``score_frames``)."""
    name: str
    version: str

    def score(self, symbol: str, feats: np.ndarray) -> Score: ...


class NullScorer:
    name = "null"
    version = "0"
    kind = KIND_PER_SYMBOL
    history_bars_5m = 0

    def score(self, symbol: str, feats: np.ndarray) -> Score:
        return Score(0.0, 0.0, [])

    def score_universe(self, frames: Mapping[str, FrameBundle]) -> dict[str, Score]:
        return {s: Score(0.0, 0.0, []) for s in frames}


@dataclass
class ModelInfo:
    name: str
    version: str
    trained_at: int | None
    threshold_long: float | None
    threshold_short: float | None
    status: str = "champion"
    # ---- V3 artifact fields (defaults == a legacy per-symbol 5m model)
    kind: str = KIND_PER_SYMBOL
    history_bars_5m: int = 0              # 0 = the engine default window
    timeframes: tuple = ("5m",)
    entry_cfg: dict = field(default_factory=dict)
    exit_cfg: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass
class ModelEntry:
    """One ``models/*.json`` artifact and its resolved status."""
    name: str
    path: Path
    status: str                      # champion | challenger | retired
    created_at: int | None
    meta: dict


def reasons_to_dicts(reasons: list[Any]) -> list[dict]:
    out = []
    for r in reasons or []:
        if isinstance(r, dict):
            out.append({"feature": r.get("feature", ""), "label": r.get("label", ""),
                        "value": float(r.get("value", 0.0)), "contribution": float(r.get("contribution", 0.0))})
        else:
            out.append({"feature": r.feature, "label": r.label, "value": float(r.value),
                        "contribution": float(r.contribution)})
    return out


def _created_ts(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v / 1000) if v > 1e12 else int(v)
    try:
        from datetime import datetime

        return int(datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def scan_models(models_dir: str | Path, overrides: Mapping[str, str] | None = None) -> list[ModelEntry]:
    """All model artifacts, newest first, with resolved status.

    Status = runtime override > ``status`` in the JSON > derived. When no model is explicitly a
    champion, the newest model without an explicit status becomes champion and the other
    status-less (older) models are retired."""
    overrides = overrides or {}
    entries: list[ModelEntry] = []
    for p in sorted(Path(models_dir).glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(p.read_text())
        except Exception as e:
            log.error("cannot read model metadata %s: %s", p, e)
            continue
        if not isinstance(meta, dict):
            continue
        raw = overrides.get(meta.get("name", p.stem)) or meta.get("status")
        entries.append(ModelEntry(str(meta.get("name", p.stem)), p, raw if raw in ("champion", "challenger", "retired") else "",
                                  _created_ts(meta.get("created_at")), meta))
    if not any(e.status == "champion" for e in entries):
        for e in entries:
            if e.status == "":
                e.status = "champion"
                break
    for e in entries:
        if e.status == "":
            e.status = "retired"
    return entries


PAPER_ONLY_MARKER = "PAPER TEST ONLY"


def is_paper_only(entry: ModelEntry) -> bool:
    """Models whose JSON ``note`` contains 'PAPER TEST ONLY' must never run with real money (live mode); testnet/demo is allowed (user, 2026-10-18)."""
    return PAPER_ONLY_MARKER.lower() in str(entry.meta.get("note", "")).lower()


_KIND_MODULE = {KIND_PER_SYMBOL: "model_np", "mtf": "model_np", "gbm": "model_gbm", "xsec": "model_xsec"}


def model_kind(meta: Mapping[str, Any]) -> str:
    k = meta.get("kind")
    return KIND_PER_SYMBOL if k in (None, "") else str(k)


def _as_dict(x: Any) -> dict:
    return dict(x) if isinstance(x, Mapping) else {}


def _default_history() -> int:
    try:
        from . import features

        return int(features.WARMUP + features.WINDOW)
    except Exception:
        return 0


def _history_bars(meta: Mapping[str, Any], scorer: Any, kind: str) -> int:
    for v in (meta.get("history_bars_5m"), getattr(scorer, "history_bars_5m", None)):
        try:
            n = int(v)           # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return _default_history() if kind == KIND_PER_SYMBOL else 0


def load_model(entry: ModelEntry) -> tuple[Scorer, ModelInfo] | None:
    """Load one artifact, dispatching on its ``kind`` (per_symbol_5m / mtf -> ``tb.model_np``, gbm ->
    ``tb.model_gbm``, xsec -> ``tb.model_xsec``). An unknown kind, a missing module or a broken artifact is
    logged and skipped (None): a model can never crash the engine."""
    kind = model_kind(entry.meta)
    mod_name = _KIND_MODULE.get(kind)
    if mod_name is None:
        log.error("model %s has unknown kind %r (known: %s); skipped", entry.name, kind, ", ".join(KINDS))
        return None
    try:
        mod = importlib.import_module(f".{mod_name}", __package__)
    except Exception as e:
        log.error("tb.%s unavailable for model %s (%s: %s); skipped", mod_name, entry.name, type(e).__name__, e)
        return None
    loader = getattr(mod, "load", None)
    if loader is None:
        log.error("tb.%s has no load(); model %s skipped", mod_name, entry.name)
        return None
    scorer = None
    errs = []
    targets = [entry.path, entry.path.with_suffix(".npz"), entry.path.with_suffix(".pkl"), entry.path.with_suffix("")]
    for target in targets:
        try:
            scorer = loader(target)
            break
        except Exception as e:  # try the next path convention
            errs.append(f"{target.name}: {type(e).__name__}: {e}")
    if scorer is None:
        log.error("%s.load failed for %s: %s", mod_name, entry.path.name, "; ".join(errs))
        return None
    if not (hasattr(scorer, "score") or hasattr(scorer, "score_universe")):
        log.error("model %s loaded but has neither score() nor score_universe(); skipped", entry.name)
        return None
    if kind == "xsec" and not hasattr(scorer, "score_universe"):
        log.error("xsec model %s has no score_universe(); skipped", entry.name)
        return None
    th = _as_dict(entry.meta.get("thresholds"))
    tfs = entry.meta.get("timeframes")
    info = ModelInfo(entry.name, str(entry.meta.get("version", "")), entry.created_at,
                     th.get("long", th.get("threshold_long")), th.get("short", th.get("threshold_short")),
                     entry.status, kind, _history_bars(entry.meta, scorer, kind),
                     tuple(tfs) if isinstance(tfs, (list, tuple)) and tfs else ("5m",),
                     _as_dict(entry.meta.get("entry_cfg")), _as_dict(entry.meta.get("exit_cfg")), dict(entry.meta))
    return scorer, info


def per_symbol_window(bundle: FrameBundle, now_ms: int | None = None) -> "np.ndarray | None":
    """The legacy per-symbol input: the last ``features.WINDOW`` rows of ``features.build_features`` (None when
    the history is too short). Shared by the engine's own per-symbol path and ``score_frames``."""
    from . import features

    closed = bundle.kl
    if len(closed) < features.WARMUP + features.WINDOW:
        return None
    btc = bundle.btc_kl if bundle.btc_kl is not None else closed
    feats = features.build_features(closed.reset_index(drop=True), bundle.metrics, bundle.funding,
                                    btc.reset_index(drop=True))
    return np.nan_to_num(np.asarray(feats[-features.WINDOW:], dtype="float32"))


def score_frames(scorer: Any, info: ModelInfo | None, frames: Mapping[str, FrameBundle]) -> dict[str, Score]:
    """Scorer interface v2 for ANY scorer: ``score_universe`` when the model has it, otherwise per symbol
    (``features_mtf.build_for_model`` for mtf/gbm, the classic window for per_symbol_5m). Symbols whose
    inputs cannot be built are simply missing from the result."""
    fn = getattr(scorer, "score_universe", None)
    if fn is not None:
        return dict(fn(frames))
    kind = info.kind if info else KIND_PER_SYMBOL
    out: dict[str, Score] = {}
    for sym, b in frames.items():
        if kind == KIND_PER_SYMBOL:
            w = per_symbol_window(b)
        else:
            from . import features_mtf

            w = features_mtf.build_for_model(info.meta if info else {}, b)       # [N, F]: the newest bar's vector is the last row
            if w is not None and getattr(w, "ndim", 1) == 2:
                w = w[-1]
        if w is not None:
            out[sym] = scorer.score(sym, w)
    return out


def load_scorer(models_dir: str | Path, overrides: Mapping[str, str] | None = None) -> tuple[Scorer, ModelInfo | None]:
    """Champion model (see ``scan_models``) or NullScorer."""
    for e in scan_models(models_dir, overrides):
        if e.status == "champion":
            got = load_model(e)
            if got:
                return got
            break
    return NullScorer(), None
