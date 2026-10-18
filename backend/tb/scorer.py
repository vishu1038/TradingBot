"""Scorer interface between the engine and the model, plus the champion/challenger registry.

``NullScorer`` (probabilities 0, never trades) is used when no ``models/*.json`` exists or when
loading fails. Model status lives in ``models/*.json`` (``status``: champion | challenger |
retired) and can be overridden at runtime through the store (promote / rollback), see ``tb.shadow``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

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


class Scorer(Protocol):
    name: str
    version: str

    def score(self, symbol: str, feats: np.ndarray) -> Score: ...


class NullScorer:
    name = "null"
    version = "0"

    def score(self, symbol: str, feats: np.ndarray) -> Score:
        return Score(0.0, 0.0, [])


@dataclass
class ModelInfo:
    name: str
    version: str
    trained_at: int | None
    threshold_long: float | None
    threshold_short: float | None
    status: str = "champion"


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


def load_model(entry: ModelEntry) -> tuple[Scorer, ModelInfo] | None:
    """``tb.model_np.load`` for one artifact; None (logged) when it cannot be loaded."""
    try:
        from . import model_np
    except Exception as e:
        log.error("tb.model_np unavailable (%s)", e)
        return None
    scorer = None
    errs = []
    for target in (entry.path, entry.path.with_suffix(".npz"), entry.path.with_suffix("")):
        try:
            scorer = model_np.load(target)
            break
        except Exception as e:  # try the next path convention
            errs.append(f"{target.name}: {type(e).__name__}: {e}")
    if scorer is None:
        log.error("model_np.load failed for %s: %s", entry.path.name, "; ".join(errs))
        return None
    th = entry.meta.get("thresholds") or {}
    info = ModelInfo(entry.name, str(entry.meta.get("version", "")), entry.created_at,
                     th.get("long", th.get("threshold_long")), th.get("short", th.get("threshold_short")),
                     entry.status)
    return scorer, info


def load_scorer(models_dir: str | Path, overrides: Mapping[str, str] | None = None) -> tuple[Scorer, ModelInfo | None]:
    """Champion model (see ``scan_models``) or NullScorer."""
    for e in scan_models(models_dir, overrides):
        if e.status == "champion":
            got = load_model(e)
            if got:
                return got
            break
    return NullScorer(), None
