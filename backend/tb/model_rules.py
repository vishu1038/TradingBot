"""Scorer kind ``rules``: transparent rule signals (``tb.rules``) as a model, so a rule bot runs next to the ML models.

Artifact: ``models/{name}.json`` only (no weights)::

    kind "rules", rule_cfg {mode, setup, tf, ... fixed constants of tb.rules.RuleCfg}, exit_cfg (ATR 1h, TP closes the whole
    position), entry_cfg, thresholds {long: 0.5, short: 0.5}, history_bars_5m, status, note

``load(path) -> RulesScorer`` implements the Scorer interface v2 (``kind``, ``history_bars_5m``, ``score_universe``).  A signal is
reported as ``Score(prob_long=1, prob_short=0)`` (or the mirror), no signal as ``Score(0, 0)``: the engine's thresholds / cadence /
gates / exits / caps then treat it like any other model.  The computation is ``tb.rules.rule_signals`` - the SAME function the
research backtest (``tb.sim.rules_research``) used - on the newest closed bars; because its filters are finite-memory the result
for a bar does not depend on how much older history the engine holds (see the ``tb.rules`` docstring), so research == live.
BTC's bars for the ``btc`` mode come from ``FrameBundle.btc_kl`` (the engine always watches BTCUSDT); BTCUSDT itself uses its own.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import rules
from .scorer import Reason, Score

log = logging.getLogger("tb.model_rules")

ZERO = (0.0, 0.0)


def _attr(b: Any, name: str, alt: str | None = None):
    if isinstance(b, Mapping):
        return b.get(name, b.get(alt) if alt else None)
    v = getattr(b, name, None)
    return v if v is not None or alt is None else getattr(b, alt, None)


class RulesScorer:
    kind = "rules"

    def __init__(self, meta: dict):
        self.meta = dict(meta)
        self.name = str(meta.get("name", "rules"))
        self.version = str(meta.get("version", ""))
        self.cfg = rules.RuleCfg.from_dict(meta["rule_cfg"])
        self.history_bars_5m = int(meta.get("history_bars_5m") or self.cfg.history_bars_5m())
        self._warned = False

    # the per-symbol window interface is not used by this kind (the engine goes through score_frames -> score_universe)
    def score(self, symbol: str, feats) -> Score:
        return Score(0.0, 0.0, [])

    def score_one(self, symbol: str, kl: pd.DataFrame, btc_kl: pd.DataFrame | None) -> Score:
        if kl is None or len(kl) == 0:
            return Score(0.0, 0.0, [])
        cfg = self.cfg
        if cfg.tf == "1h" and (int(kl["open_time"].iloc[-1]) + rules.BAR_MS) % rules.HOUR_MS:
            return Score(0.0, 0.0, [])                       # a 1h variant decides on the 5m bar that closes the hour only
        n = self.history_bars_5m
        kl = kl.tail(n)
        if cfg.mode == "btc":
            if symbol == "BTCUSDT":
                btc_kl = kl
            elif btc_kl is None or len(btc_kl) == 0:
                if not self._warned:
                    self._warned = True
                    log.warning("rules model %s: no BTCUSDT bars, no signals", self.name)
                return Score(0.0, 0.0, [])
            else:
                btc_kl = btc_kl.tail(n)
        side, regime = rules.last_signal(kl, btc_kl, cfg)
        if side is None:
            return Score(0.0, 0.0, [])
        label = f"{cfg.mode} regime {'up' if regime > 0 else 'down'}: {rules.SETUP_NAMES[cfg.setup]} ({cfg.tf})"
        reasons = [Reason("regime", label, float(regime), 1.0), Reason("setup", rules.describe(cfg), 1.0, 1.0)]
        return Score(1.0, 0.0, reasons) if side == "long" else Score(0.0, 1.0, reasons)

    def score_universe(self, frames: Mapping[str, Any]) -> dict[str, Score]:
        out: dict[str, Score] = {}
        for sym, b in frames.items():
            kl = _attr(b, "kl", "klines")
            if kl is None:
                continue
            out[sym] = self.score_one(sym, kl, _attr(b, "btc_kl", "btc"))
        return out


def load(path) -> RulesScorer:
    """``path``: the ``.json`` or the bare model path."""
    p = Path(path)
    if p.suffix in (".json", ".pkl", ".npz"):
        p = p.with_suffix("")
    return RulesScorer(json.loads(Path(str(p) + ".json").read_text()))
