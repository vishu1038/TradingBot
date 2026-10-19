"""Reversal protection maths shared by the engine, the shadow simulator and the backtester (pure functions, no I/O).

User 2026-10-19: trade long AND short symmetrically, "in such a way that if the market reverses, the bots don't lose money on their
other edges". Two mechanisms (plus the existing ``max_side_risk_pct`` cap):

* **Beta-weighted net exposure cap** (``max_net_exposure_pct``). Every coin gets a beta vs BTC from the last 7 days of 1h returns
  (``beta_from_returns``, fallback 1.0, clipped to [0.25, 3]); the book's net exposure is ``sum(side x notional x beta)``
  (+1 long, -1 short). An entry that would push ``|net|`` beyond the cap is rejected, one that REDUCES ``|net|`` is always allowed
  (``tb.risk.net_exposure_check``).
* **Basket reversal detector** (``market_signal`` -> ``RevSignal``, ``ReversalBook``). The equal-weight return of the universe and of
  BTC over 15 min (3 x 5m bars) and 1 h (12 bars) as a z-score against its own recent volatility (RMS of the earlier, non-overlapping
  windows of the last 2 days). When the move is AGAINST the book's net side with ``z >= reversal_z``: that side's winners (profit >=
  ``reversal_lock_r`` x initial risk) get their stop tightened (``lock_stop``: half of the open profit, at least break-even + round-trip
  fees, never loosened, never through the market) and new entries on that side are blocked for ``reversal_block_bars`` 5m bars.
  Losing positions keep their stops; nothing is closed in a panic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .protect import breakeven_price, tighter

BAR_MS = 300_000
HOUR_MS = 3_600_000
Z_WINDOWS = {"15m": 3, "1h": 12}      # window length in 5m bars
Z_LOOKBACK = 576                       # bars of history (2 days) the volatility is measured on
Z_MIN_OBS = 96                         # minimum earlier windows for a z-score (else NaN = no signal)
MIN_COVER = 0.6                        # share of the basket that must have a bar for it to count
BETA_RETURNS = 168                     # 7 days of 1h returns
BETA_MIN_OBS = 48
BETA_LO, BETA_HI = 0.25, 3.0
NET_SIDE_MIN_PCT = 5.0                 # |net| below this % of equity = no net side (nothing to protect)
LOCK_FRAC = 0.5                        # a reversal lock keeps this share of the open profit (at least break-even + fees)


@dataclass(frozen=True)
class ReversalCfg:
    max_net_exposure_pct: float = 150.0   # 0 = off
    z: float = 2.5                        # 0 = off
    lock_r: float = 0.5
    block_bars: int = 6

    @classmethod
    def from_settings(cls, cfg: Any) -> "ReversalCfg":
        return cls(cfg.max_net_exposure_pct, cfg.reversal_z, cfg.reversal_lock_r, int(cfg.reversal_block_bars))

    @classmethod
    def from_sim(cls, cfg: Any) -> "ReversalCfg":
        return cls(cfg.max_net_exposure_pct, cfg.reversal_z, cfg.reversal_lock_r, int(cfg.reversal_block_bars))

    @property
    def detector_on(self) -> bool:
        return self.z > 0


# ----------------------------------------------------------------------------- beta vs BTC
def beta_from_returns(x: np.ndarray, y: np.ndarray, min_obs: int = BETA_MIN_OBS) -> float:
    """cov(x, y) / var(y) over the finite pairs of the last ``BETA_RETURNS`` returns; 1.0 when there is too little data."""
    x = np.asarray(x, dtype="float64")[-BETA_RETURNS:]
    y = np.asarray(y, dtype="float64")[-BETA_RETURNS:]
    n = min(len(x), len(y))
    x, y = x[-n:], y[-n:]
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < min_obs:
        return 1.0
    x, y = x[ok], y[ok]
    vy = float(np.var(y))
    if vy <= 1e-18:
        return 1.0
    b = float(np.mean((x - x.mean()) * (y - y.mean())) / vy)
    return float(min(max(b, BETA_LO), BETA_HI)) if math.isfinite(b) else 1.0


def hourly_returns(times_ms: np.ndarray, closes: np.ndarray) -> pd.Series:
    """1h returns indexed by the open time of the hour they END (consecutive hours only) from 1h closes."""
    s = pd.Series(np.asarray(closes, dtype="float64"), index=np.asarray(times_ms, dtype="int64"))
    r = s / s.shift(1) - 1.0
    r[(s.index.to_series().diff() != HOUR_MS).to_numpy()] = np.nan
    return r


def beta_vs_btc(sym_times: np.ndarray, sym_closes: np.ndarray, btc_times: np.ndarray, btc_closes: np.ndarray) -> float:
    """Beta of a coin vs BTC from 1h closes (last 7 days); 1.0 when either series is too short / flat."""
    if len(sym_times) < 3 or len(btc_times) < 3:
        return 1.0
    a, b = hourly_returns(sym_times, sym_closes), hourly_returns(btc_times, btc_closes)
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < BETA_MIN_OBS:
        return 1.0
    return beta_from_returns(j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy())


def rolling_beta(x: np.ndarray, y: np.ndarray, window: int = BETA_RETURNS, min_obs: int = BETA_MIN_OBS) -> np.ndarray:
    """Rolling beta of ``x`` on ``y`` (hourly returns on a common grid; NaN gaps allowed), the vectorised twin of
    ``beta_from_returns`` for the backtester. NaN-free output: 1.0 where there is not enough data."""
    dfx = pd.DataFrame({"x": np.asarray(x, dtype="float64"), "y": np.asarray(y, dtype="float64")})
    ok = dfx["x"].notna() & dfx["y"].notna()
    xx, yy = dfx["x"].where(ok), dfx["y"].where(ok)
    cov = xx.rolling(window, min_periods=min_obs).cov(yy, ddof=0)        # population form = beta_from_returns
    var = yy.rolling(window, min_periods=min_obs).var(ddof=0)
    b = (cov / var).to_numpy()
    b = np.where(np.isfinite(b) & (var.to_numpy() > 1e-18), b, 1.0)
    return np.clip(b, BETA_LO, BETA_HI)


# ----------------------------------------------------------------------------- net exposure
def signed_net(positions: Iterable[tuple[str, float, float]]) -> float:
    """sum(+1 long / -1 short x notional x beta) over ``(side, notional, beta)``."""
    return float(sum((1.0 if s == "long" else -1.0) * n * b for s, n, b in positions))


def net_side(net: float, equity: float) -> str | None:
    """The side the book is net exposed on, or None when |net| is below ``NET_SIDE_MIN_PCT`` % of equity."""
    if equity <= 0 or abs(net) < NET_SIDE_MIN_PCT / 100.0 * equity:
        return None
    return "long" if net > 0 else "short"


# ----------------------------------------------------------------------------- basket z-score
def rolling_z(r: np.ndarray, w: int, lookback: int = Z_LOOKBACK, min_obs: int = Z_MIN_OBS) -> np.ndarray:
    """z[i] = (sum of r[i-w+1..i]) / RMS of the EARLIER non-overlapping w-windows (those ending at <= i-w, the last ``lookback``).
    NaN where fewer than ``min_obs`` earlier windows exist. ``r`` = per-bar returns (NaN counts as 0)."""
    r = np.nan_to_num(np.asarray(r, dtype="float64"), nan=0.0)
    n = len(r)
    out = np.full(n, np.nan)
    if n < w + min_obs:
        return out
    rw = pd.Series(r).rolling(w).sum()
    sq = (rw ** 2).rolling(lookback, min_periods=min_obs).mean().shift(w)
    sd = np.sqrt(sq.to_numpy())
    cur = rw.to_numpy()
    ok = np.isfinite(sd) & (sd > 1e-12) & np.isfinite(cur)
    out[ok] = cur[ok] / sd[ok]
    return out


def window_z(r: np.ndarray, w: int, lookback: int = Z_LOOKBACK, min_obs: int = Z_MIN_OBS) -> float:
    """The newest ``rolling_z`` value (NaN = no signal)."""
    z = rolling_z(r, w, lookback, min_obs)
    return float(z[-1]) if len(z) else float("nan")


@dataclass(frozen=True)
class RevSignal:
    t_ms: int                                   # open time of the newest 5m bar used; decisions are taken at t_ms + 5 min
    z: Mapping[str, float] = field(default_factory=dict)   # basket_15m / basket_1h / btc_15m / btc_1h: signed z (+ up, - down)

    def adverse(self, side: str) -> tuple[float, str]:
        """(largest z AGAINST ``side``, its source). Against a long = the basket falling (negative z), against a short = rising."""
        d = 1.0 if side == "long" else -1.0
        best, src = float("-inf"), ""
        for k, v in self.z.items():
            if v == v and -d * v > best:
                best, src = -d * float(v), k
        return best, src


def basket_returns(frames: Mapping[str, tuple[np.ndarray, np.ndarray]], min_cover: float = MIN_COVER) -> tuple[np.ndarray, int] | None:
    """Equal-weight 5m return of the basket on a continuous 5m grid. ``frames`` = {symbol: (open_times_ms, closes)} of CLOSED bars.
    Bars at the end covered by fewer than ``min_cover`` of the symbols are dropped (a cycle where only some symbols have the newest
    bar yet); gaps inside count as 0. Returns (returns, open_time of the last bar) or None."""
    series = {s: pd.Series(np.asarray(c, dtype="float64"), index=np.asarray(t, dtype="int64"))
              for s, (t, c) in frames.items() if len(t) > 2}
    if not series:
        return None
    px = pd.DataFrame(series).sort_index()
    ret = px / px.shift(1) - 1.0
    cnt = ret.notna().sum(axis=1)
    need = max(1, math.ceil(min_cover * len(series)))
    good = cnt >= need
    if not good.any():
        return None
    last = good[good].index[-1]
    ret = ret.loc[:last]
    b = ret.mean(axis=1, skipna=True).where(cnt.loc[:last] >= need)
    grid = np.arange(int(b.index[0]), int(last) + BAR_MS, BAR_MS, dtype="int64")
    return b.reindex(grid).fillna(0.0).to_numpy(), int(last)


def market_signal(universe: Mapping[str, tuple[np.ndarray, np.ndarray]],
                  btc: tuple[np.ndarray, np.ndarray] | None = None) -> RevSignal | None:
    """z-scores of the equal-weight universe basket and of BTC over 15 min and 1 h (None when no data at all)."""
    z: dict[str, float] = {}
    t_last = 0
    bk = basket_returns(universe)
    if bk is not None:
        b, t_last = bk
        for name, w in Z_WINDOWS.items():
            z[f"basket_{name}"] = window_z(b, w)
    if btc is not None and len(btc[0]) > Z_MIN_OBS:
        bb = basket_returns({"BTC": btc}, min_cover=1.0)
        if bb is not None:
            r, t_b = bb
            t_last = max(t_last, t_b) if t_last else t_b
            for name, w in Z_WINDOWS.items():
                z[f"btc_{name}"] = window_z(r, w)
    if not z:
        return None
    return RevSignal(int(t_last), z)


# ----------------------------------------------------------------------------- the lock
def lock_stop(side: str, entry: float, init_sl: float, cur_sl: float, mark: float, fee_rate: float, lock_r: float,
              frac: float = LOCK_FRAC) -> float | None:
    """Stop level that locks profit during a reversal, or None when the position is not eligible: in profit >= ``lock_r`` x initial
    risk at ``mark``; the level keeps ``frac`` of the open profit but never less than break-even + round-trip fees; never loosens
    ``cur_sl``; a level that is not behind the market is not used (no panic closes)."""
    risk = abs(entry - init_sl)
    if risk <= 0 or mark <= 0:
        return None
    d = 1.0 if side == "long" else -1.0
    fav = (mark - entry) * d
    if fav <= 0 or fav < lock_r * risk - 1e-12:
        return None
    cand = tighter(side, entry + d * frac * fav, breakeven_price(side, entry, fee_rate))
    if (cand - mark) * d >= 0:
        return None
    if (cand - cur_sl) * d <= 0:
        return None
    return cand


# ----------------------------------------------------------------------------- per-book state
class ReversalBook:
    """Block / lock windows of ONE book (the live account, one shadow model, or the backtest)."""

    def __init__(self) -> None:
        self.block_until: dict[str, int] = {}     # side -> ms before which NEW entries on that side are refused
        self.lock_until: dict[str, int] = {}      # side -> ms before which winners of that side keep being tightened
        self.last: dict[str, Any] | None = None

    def step(self, sig: RevSignal | None, side: str | None, rc: ReversalCfg) -> dict[str, Any] | None:
        """Feed the newest signal and the book's net side. Returns the trigger ({side, z, source, until_ms, new}) when the move is
        against ``side`` with z >= reversal_z (the windows are extended), else None."""
        if not rc.detector_on or sig is None or side is None:
            return None
        z, src = sig.adverse(side)
        if not (z >= rc.z):
            return None
        t_dec = int(sig.t_ms) + BAR_MS
        new = not self.locking(side, t_dec)
        self.block_until[side] = max(self.block_until.get(side, 0), t_dec + rc.block_bars * BAR_MS)
        self.lock_until[side] = max(self.lock_until.get(side, 0), t_dec + max(rc.block_bars, 1) * BAR_MS)
        self.last = {"side": side, "z": float(z), "source": src, "t_ms": t_dec, "until_ms": self.block_until[side], "new": new}
        return dict(self.last)

    def blocked(self, side: str, now_ms: int) -> bool:
        return now_ms < self.block_until.get(side, 0)

    def locking(self, side: str, now_ms: int) -> bool:
        return now_ms < self.lock_until.get(side, 0)

    def to_dict(self) -> dict[str, Any]:
        return {"block_until": dict(self.block_until), "lock_until": dict(self.lock_until), "last": self.last}

    def load(self, d: Mapping[str, Any] | None) -> None:
        d = d or {}
        self.block_until = {k: int(v) for k, v in (d.get("block_until") or {}).items()}
        self.lock_until = {k: int(v) for k, v in (d.get("lock_until") or {}).items()}
        self.last = d.get("last")
