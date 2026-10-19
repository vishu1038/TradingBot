"""Model-driven exits (V3): ``exit_cfg.exit_mode = "pct"`` and ``exit_cfg.atr_tf`` (ATR from resampled 1h/4h bars).

A model whose ``exit_cfg`` is the classic ``fixed``/``atr`` on 5m keeps using the engine ``Settings`` exactly as
before (``ExitSpec.source == "settings"``): the exit template is the trader's setting, not the model's.
Only ``pct`` and a non-5m ``atr_tf`` make the artifact drive the exits.

Both reuse the sizing / level maths of ``tb.risk`` through an *effective* ``Settings``:

* ``pct``   (``tp_pct`` / ``sl_pct`` are PERCENT in the artifact, like ``tb.labels``: 3.0 = 3 %; ``ExitSpec`` holds fractions)
            -> ``exit_mode = "atr"`` with ``atr := entry price`` and ``tp_atr/sl_atr := tp_pct/sl_pct`` (so the
  stop is ``sl_pct * entry`` away and the position is sized to risk ``sl_usd``, capped by the slice notional);
* ``atr``+tf -> ``exit_mode = "atr"`` with ``atr := Wilder ATR14 of the resampled bars`` and the model's multiples.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

import pandas as pd

from .config import Settings
from .risk import atr14

log = logging.getLogger("tb.exits")

BAR_MS = 300_000
TF_BARS = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "6h": 72, "8h": 96, "12h": 144, "1d": 288}


@dataclass(frozen=True)
class ExitSpec:
    mode: str = "settings"                 # settings | pct | atr
    atr_tf: str = "5m"
    tp_pct: float | None = None
    sl_pct: float | None = None
    tp_atr: float | None = None
    sl_atr: float | None = None
    tp_enabled: bool = True                # False: stop only (cross-sectional legs close on the rebalance)
    # volatility-scaled pct stop (xsec artifacts): sl_pct = clip(vol_mult * trailing-7d daily vol, stop_min, stop_max);
    # resolved per coin by ``bind`` right before sizing
    vol_mult: float | None = None
    stop_min: float = 0.02
    stop_max: float = 0.20
    # time stop: close at market this many 5m bars after the entry (labels count a timeout as a loss closed at the
    # last bar's close); None = none. Valid on any spec, also on the Settings exits.
    horizon_bars: int | None = None

    @property
    def source(self) -> str:
        return "settings" if self.mode == "settings" else "model"

    def effective(self, cfg: Settings) -> Settings:
        """The Settings the shared sizing / levels maths run with."""
        if self.mode == "pct":
            sl = float(self.sl_pct or 0.0)
            tp = float(self.tp_pct) if self.tp_pct else sl * 3.0       # (a stop-only spec still needs a positive target to size)
            return replace(cfg, exit_mode="atr", sl_atr=sl, tp_atr=tp)
        if self.mode == "atr":
            return replace(cfg, exit_mode="atr", sl_atr=float(self.sl_atr or cfg.sl_atr),
                           tp_atr=float(self.tp_atr or cfg.tp_atr))
        return cfg

    def atr_arg(self, atr: float | None, price: float) -> float | None:
        """The ``atr`` argument for ``tb.risk.size_trade`` / ``levels`` (the price itself for pct mode)."""
        return price if self.mode == "pct" else atr

    def history_bars(self) -> int:
        """5m bars needed to compute this spec's ATR (ATR14 needs 15 complete bars + alignment slack)."""
        if self.vol_mult is not None:
            return 176 * 12                       # 168 h of hourly returns + slack
        if self.mode != "atr":
            return 0
        n = TF_BARS.get(self.atr_tf, 1)
        return 0 if n == 1 else (15 + 2) * n


SETTINGS_SPEC = ExitSpec()


def _pos(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _pct(v: Any) -> float | None:
    f = _pos(v)
    return None if f is None else f / 100.0


def resolve_exit(exit_cfg: Mapping[str, Any] | None, kind: str = "per_symbol_5m", name: str = "") -> ExitSpec:
    """ExitSpec for a model's ``exit_cfg`` (invalid combinations are logged and fall back to the Settings exits);
    ``horizon_bars`` (time stop) is honoured in every case."""
    spec = _resolve_exit(exit_cfg, kind, name)
    hb = _pos((exit_cfg or {}).get("horizon_bars"))
    return replace(spec, horizon_bars=int(hb)) if hb else spec


ATR_TFS = ("5m", "1h", "4h", "1d")          # the timeframes tb.labels supports for exit_cfg.atr_tf


def _resolve_exit(exit_cfg: Mapping[str, Any] | None, kind: str, name: str) -> ExitSpec:
    ec = dict(exit_cfg or {})
    mode = ec.get("exit_mode")
    tf = str(ec.get("atr_tf") or "5m")
    if tf not in ATR_TFS:
        log.error("model %s: unsupported exit_cfg.atr_tf %r (use %s); using the Settings exits", name, tf, "|".join(ATR_TFS))
        return SETTINGS_SPEC
    if mode == "pct":
        sl, tp = _pct(ec.get("sl_pct")), _pct(ec.get("tp_pct"))     # artifact units: percent (3.0 = 3 %)
        vm = _pos(ec.get("stop_vol_mult"))
        if sl is None and vm is not None:             # volatility-scaled stop, no take-profit (legs close on the rebalance)
            return ExitSpec("pct", "5m", tp_pct=tp, sl_pct=None, tp_enabled=tp is not None, vol_mult=vm,
                            stop_min=_pos(ec.get("stop_min")) or 0.02, stop_max=_pos(ec.get("stop_max")) or 0.20)
        if sl is None or (tp is None and kind != "xsec"):
            log.error("model %s: exit_mode pct needs sl_pct%s > 0; using the Settings exits", name,
                      "" if kind == "xsec" else " and tp_pct")
            return SETTINGS_SPEC
        return ExitSpec("pct", "5m", tp_pct=tp, sl_pct=sl, tp_enabled=tp is not None)
    if tf != "5m" and mode in (None, "atr"):
        return ExitSpec("atr", tf, tp_atr=_pos(ec.get("tp_atr")), sl_atr=_pos(ec.get("sl_atr")),
                        tp_enabled=not (kind == "xsec" and _pos(ec.get("tp_atr")) is None))
    return SETTINGS_SPEC


def resample_closed(closed: pd.DataFrame, tf: str, now_ms: int, max_bars: int = 40) -> pd.DataFrame:
    """Complete ``tf`` bars (UTC-aligned) built from closed 5m bars; the still-forming tf bar is dropped."""
    n = TF_BARS.get(tf, 1)
    if n == 1:
        return closed
    ms = n * BAR_MS
    k = closed.tail(n * max_bars)
    g = (k["open_time"] // ms).to_numpy()
    df = pd.DataFrame({"g": g, "open": k["open"].to_numpy(), "high": k["high"].to_numpy(), "low": k["low"].to_numpy(),
                       "close": k["close"].to_numpy()})
    agg = df.groupby("g", sort=True).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                          close=("close", "last"), cnt=("close", "size"))
    agg = agg[(agg["cnt"] >= n) & ((agg.index.to_numpy() + 1) * ms <= now_ms)].copy()
    agg["open_time"] = agg.index.to_numpy() * ms
    return agg.reset_index(drop=True)[["open_time", "open", "high", "low", "close"]]


def bind(spec: ExitSpec, closed: pd.DataFrame, now_ms: int) -> ExitSpec | None:
    """Resolve a volatility-scaled stop to a concrete ``sl_pct`` for this coin (trailing 168 h of hourly log returns,
    scaled to a day, exactly as ``tb.sim.xsec.sigma_day``). None = not enough data (the coin is skipped)."""
    if spec.vol_mult is None:
        return spec
    hourly = resample_closed(closed, "1h", now_ms, 175)
    c = hourly["close"].to_numpy(dtype="float64")[-169:]
    if len(c) < 121 or (c <= 0).any():
        return None
    lr = pd.Series(c).pipe(lambda s: (s / s.shift(1)).apply(math.log)).dropna()
    sg = float(lr.std(ddof=1)) * math.sqrt(24.0)
    if not math.isfinite(sg) or sg <= 0:
        return None
    return replace(spec, sl_pct=min(max(spec.vol_mult * sg, spec.stop_min), spec.stop_max), vol_mult=None)


def exit_atr(spec: ExitSpec, closed: pd.DataFrame, now_ms: int) -> float | None:
    """The ATR value ``spec`` sizes its stop with (5m ATR for the Settings exits, as before)."""
    if spec.mode == "pct":
        return None
    if spec.mode == "settings" or spec.atr_tf == "5m":
        return atr14(closed["high"].to_numpy(), closed["low"].to_numpy(), closed["close"].to_numpy())
    # EXACTLY the ATR the labels were built with (tb.labels.exit_atr): SMA-ATR14 of the last CLOSED tf bar
    from .labels import exit_atr as label_exit_atr

    n = TF_BARS[spec.atr_tf]
    tail = closed.tail(n * 24).reset_index(drop=True)
    a = label_exit_atr(tail, {"atr_tf": spec.atr_tf})
    v = float(a[-1]) if len(a) else float("nan")
    return v if math.isfinite(v) and v > 0 else None
