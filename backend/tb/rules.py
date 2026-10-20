"""Transparent rule-based entry signals (user 2026-10-20: rules first, ML at most a filter): EMA / RSI / MACD / volume
z-score / N-bar breakouts, a market REGIME that chooses the side, and three fixed setups.

Pure numpy / pandas, no look-ahead: every value at 5m row ``i`` uses rows ``<= i`` only, 1h values come from COMPLETE
1h bars only (an hour is complete at the close of its 12th 5m bar) and a 1h signal is placed on that 12th 5m row.

Window independence (what makes research == live): the recursive smoothers (EMA, Wilder RSI) are implemented as
FINITE-MEMORY filters, i.e. the same exponential weights truncated to ``K = 3 * (n + 1)`` taps (tail weight e**-6 ~ 0.25 %)
and renormalised.  An output depends on exactly the last ``K`` inputs, so the live scorer (which sees the last
``history_bars_5m`` bars) produces BIT-IDENTICAL signals to the research arrays computed over two years.  Nothing else in
this module has memory beyond its window (rolling max / min / mean / std are exact windows).

Direction modes (two separate bots, the regime picks the side, no fixed bias):
  * ``btc``  - BTC's trend decides for every coin: up -> only longs, down -> only shorts, chop -> flat;
  * ``coin`` - the coin's own trend decides.
Setups (decided before looking at results):
  * S1 trend pullback : in the regime direction the price touched the 1h/5m EMA20 and RSI went < 40 (long) / > 60 (short)
                        within the last 6 bars, entry when the MACD histogram turns in the regime direction;
  * S2 breakout       : close beyond the prior 20-bar high / low, volume z-score > 2, MACD histogram on the same side;
  * S3 momentum       : MACD line crosses its signal line in the regime direction with RSI 50-70 (long) / 30-50 (short).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import numpy as np
import pandas as pd

BAR_MS = 300_000
HOUR_MS = 3_600_000
BARS_PER_HOUR = 12

MODES = ("btc", "coin")
SETUPS = ("S1", "S2", "S3")
TFS = ("5m", "1h")
SETUP_NAMES = {"S1": "trend pullback", "S2": "breakout", "S3": "momentum continuation"}


# ----------------------------------------------------------------------------------------------- indicators
def _fir_weights(alpha: float, taps: int) -> np.ndarray:
    w = alpha * (1.0 - alpha) ** np.arange(taps)
    return w / w.sum()


def _fir(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """y[t] = sum_k w[k] * x[t-k]; NaN until a full window of finite inputs exists (NaN inputs poison only their own windows)."""
    x = np.asarray(x, dtype="float64")
    n, k = len(x), len(w)
    out = np.full(n, np.nan)
    if n >= k:
        out[k - 1:] = np.convolve(x, w)[k - 1:n]
    return out


def ema(x, n: int) -> np.ndarray:
    """Exponential moving average, span ``n`` (alpha = 2 / (n + 1)), finite memory of ``3 * (n + 1)`` inputs."""
    return _fir(x, _fir_weights(2.0 / (n + 1.0), 3 * (n + 1)))


def ema_taps(n: int) -> int:
    return 3 * (n + 1)


def rsi(close, n: int = 14) -> np.ndarray:
    """RSI with Wilder smoothing (alpha = 1 / n), finite memory of ``6 * n`` inputs. 100 when there were no losses, 50 when flat."""
    c = np.asarray(close, dtype="float64")
    d = np.diff(c, prepend=np.nan)
    w = _fir_weights(1.0 / n, 6 * n)
    g = _fir(np.where(d > 0, d, 0.0) + (d * 0.0), w)               # (the +d*0 keeps the leading NaN)
    l = _fir(np.where(d < 0, -d, 0.0) + (d * 0.0), w)
    tot = g + l
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(tot > 0, 100.0 * g / tot, 50.0)
    return np.where(np.isfinite(tot), out, np.nan)


def macd(close, fast: int = 12, slow: int = 26, sig: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(macd line, signal line, histogram = macd - signal)."""
    c = np.asarray(close, dtype="float64")
    line = ema(c, fast) - ema(c, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def macd_taps(fast: int = 12, slow: int = 26, sig: int = 9) -> int:
    return ema_taps(slow) + ema_taps(sig) - 1


def true_range(high, low, close) -> np.ndarray:
    h, l, c = (np.asarray(a, dtype="float64") for a in (high, low, close))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def atr_sma(high, low, close, n: int = 14) -> np.ndarray:
    """ATR as the simple moving average of the true range (the ATR the exits / labels use, ``features_mtf.sma_atr``)."""
    return rolling_mean(true_range(high, low, close), n)


def _windows(x: np.ndarray, n: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(x, n)


def rolling_mean(x, n: int) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = _windows(x, n).mean(axis=1)
    return out


def rolling_std(x, n: int) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = _windows(x, n).std(axis=1)
    return out


def rolling_max(x, n: int) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = _windows(x, n).max(axis=1)
    return out


def rolling_min(x, n: int) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = _windows(x, n).min(axis=1)
    return out


def shift(x, k: int = 1) -> np.ndarray:
    """x[t-k] (NaN for the first k rows)."""
    x = np.asarray(x, dtype="float64")
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def volume_z(volume, n: int = 48) -> np.ndarray:
    """Volume z-score against the PREVIOUS ``n`` bars (the current bar is not part of its own baseline)."""
    v = np.asarray(volume, dtype="float64")
    mu, sd = shift(rolling_mean(v, n)), shift(rolling_std(v, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (v - mu) / sd, np.nan)


def breakout_levels(high, low, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """(highest high, lowest low) of the PREVIOUS ``n`` bars."""
    return shift(rolling_max(high, n)), shift(rolling_min(low, n))


# ----------------------------------------------------------------------------------------------- resampling
@dataclass
class Bars:
    """OHLCV arrays of one timeframe; ``t_end`` = close time (ms) of every bar."""
    t_end: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray

    def __len__(self) -> int:
        return len(self.c)


def bars_5m(kl: pd.DataFrame) -> Bars:
    return Bars(kl["open_time"].to_numpy("int64") + BAR_MS, kl["open"].to_numpy("float64"), kl["high"].to_numpy("float64"),
                kl["low"].to_numpy("float64"), kl["close"].to_numpy("float64"), kl["volume"].to_numpy("float64"))


def bars_1h(kl: pd.DataFrame) -> Bars:
    """COMPLETE UTC-aligned 1h bars from 5m rows (an hour with fewer than 12 rows is dropped)."""
    t = kl["open_time"].to_numpy("int64")
    if len(t) == 0:
        z = np.zeros(0)
        return Bars(np.zeros(0, "int64"), z, z, z, z, z)
    g = t // HOUR_MS
    df = pd.DataFrame({"g": g, "o": kl["open"].to_numpy("float64"), "h": kl["high"].to_numpy("float64"),
                       "l": kl["low"].to_numpy("float64"), "c": kl["close"].to_numpy("float64"),
                       "v": kl["volume"].to_numpy("float64")})
    a = df.groupby("g", sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"),
                                       n=("c", "size"))
    a = a[a["n"] >= BARS_PER_HOUR]
    return Bars((a.index.to_numpy("int64") + 1) * HOUR_MS, a["o"].to_numpy(), a["h"].to_numpy(), a["l"].to_numpy(),
                a["c"].to_numpy(), a["v"].to_numpy())


# ----------------------------------------------------------------------------------------------- regimes (on 1h closes)
def btc_regime(close_1h, ema_hours: int = 240, ret_hours: int = 168, ret_thr: float = 0.02) -> np.ndarray:
    """BTC trend per 1h bar: +1 up (close above its 10-day EMA and 7-day return > +2 %), -1 down (mirror), 0 chop."""
    c = np.asarray(close_1h, dtype="float64")
    e = ema(c, ema_hours)
    r = c / shift(c, ret_hours) - 1.0
    up, dn = (c > e) & (r > ret_thr), (c < e) & (r < -ret_thr)
    return np.where(up, 1, np.where(dn, -1, 0)).astype("int8")


def coin_regime(close_1h, fast: int = 50, slow: int = 200, slope_hours: int = 6) -> np.ndarray:
    """A coin's own trend per 1h bar: +1 up (close above the 1h EMA200 and the EMA50 rising over 6 h), -1 down (mirror), 0 chop."""
    c = np.asarray(close_1h, dtype="float64")
    ef, es = ema(c, fast), ema(c, slow)
    slope = ef - shift(ef, slope_hours)
    up, dn = (c > es) & (slope > 0), (c < es) & (slope < 0)
    return np.where(up, 1, np.where(dn, -1, 0)).astype("int8")


def asof(t_end_src: np.ndarray, values: np.ndarray, t_query: np.ndarray, fill: float = 0) -> np.ndarray:
    """``values`` of the last source bar whose close time is <= ``t_query`` (``fill`` before the first)."""
    j = np.searchsorted(t_end_src, t_query, side="right") - 1
    out = np.full(len(t_query), fill, dtype=values.dtype)
    ok = j >= 0
    out[ok] = values[j[ok]]
    return out


# ----------------------------------------------------------------------------------------------- setups
@dataclass(frozen=True)
class RuleCfg:
    """One rule variant. The numbers are FIXED (decided before any backtest); only mode / setup / tf vary."""
    mode: str = "btc"                  # btc | coin
    setup: str = "S1"                  # S1 | S2 | S3
    tf: str = "1h"                     # 5m | 1h
    ema_pullback: int = 20
    pullback_bars: int = 6
    rsi_n: int = 14
    rsi_pull_long: float = 40.0
    rsi_pull_short: float = 60.0
    brk_n: int = 20
    vol_z_n: int = 48
    vol_z_min: float = 2.0
    s3_long: tuple = (50.0, 70.0)
    s3_short: tuple = (30.0, 50.0)
    btc_ema_hours: int = 240
    btc_ret_hours: int = 168
    btc_ret_thr: float = 0.02
    coin_fast: int = 50
    coin_slow: int = 200
    coin_slope_hours: int = 6

    def __post_init__(self):
        if self.mode not in MODES or self.setup not in SETUPS or self.tf not in TFS:
            raise ValueError(f"bad rule variant {self.mode}/{self.setup}/{self.tf}")

    @property
    def key(self) -> str:
        return f"{self.mode}_{self.setup}_{self.tf}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["s3_long"], d["s3_short"] = list(self.s3_long), list(self.s3_short)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RuleCfg":
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if k in ("s3_long", "s3_short") else v) for k, v in dict(d).items() if k in names}
        return cls(**kw)

    def history_bars_5m(self) -> int:
        """5m rows after which every output is final (window-independent): the longest filter chain on its timeframe + slack."""
        hours = max(ema_taps(self.btc_ema_hours) if self.mode == "btc" else ema_taps(self.coin_slow) + self.coin_slope_hours,
                    self.btc_ret_hours + 1 if self.mode == "btc" else 0)
        regime_5m = (hours + 2) * BARS_PER_HOUR
        chain = max(macd_taps() + 2, self.rsi_n * 6 + self.pullback_bars + 4, self.vol_z_n + 4, self.brk_n + 2, ema_taps(self.ema_pullback) + 8)
        setup_5m = chain if self.tf == "5m" else (chain + 2) * BARS_PER_HOUR
        return int(max(regime_5m, setup_5m) + BARS_PER_HOUR)


def _setup_raw(cfg: RuleCfg, b: Bars) -> tuple[np.ndarray, np.ndarray]:
    """(long, short) setup conditions per bar WITHOUT the regime filter."""
    c, h, l, v = b.c, b.h, b.l, b.v
    _, _, hist = macd(c)
    h1, h2 = shift(hist, 1), shift(hist, 2)
    with np.errstate(invalid="ignore"):
        if cfg.setup == "S1":
            e = ema(c, cfg.ema_pullback)
            r = rsi(c, cfg.rsi_n)
            n = cfg.pullback_bars
            touch_dn = rolling_min(l - e, n) <= 0                  # price touched / crossed below the EMA20 within the last n bars
            touch_up = rolling_max(h - e, n) >= 0
            r_lo, r_hi = rolling_min(r, n), rolling_max(r, n)
            turn_up = (hist > h1) & (h1 <= h2)
            turn_dn = (hist < h1) & (h1 >= h2)
            return touch_dn & (r_lo < cfg.rsi_pull_long) & turn_up, touch_up & (r_hi > cfg.rsi_pull_short) & turn_dn
        if cfg.setup == "S2":
            hi_prev, lo_prev = breakout_levels(h, l, cfg.brk_n)
            z = volume_z(v, cfg.vol_z_n)
            return (c > hi_prev) & (z > cfg.vol_z_min) & (hist > 0), (c < lo_prev) & (z > cfg.vol_z_min) & (hist < 0)
        r = rsi(c, cfg.rsi_n)                                     # S3
        lo, hi = cfg.s3_long, cfg.s3_short
        return ((hist > 0) & (h1 <= 0) & (r >= lo[0]) & (r <= lo[1]),
                (hist < 0) & (h1 >= 0) & (r >= hi[0]) & (r <= hi[1]))


@dataclass
class RuleSignals:
    long: np.ndarray                   # bool per 5m row: enter long at this row's close
    short: np.ndarray
    regime: np.ndarray                 # int8 per 5m row: the regime in force at this row's close (+1 up, -1 down, 0 chop)
    due: np.ndarray                    # bool per 5m row: a decision point of the variant's timeframe (every row for 5m, hour closes for 1h)


def regime_for(cfg: RuleCfg, own_1h: Bars, btc_1h: Bars | None, t_query: np.ndarray) -> np.ndarray:
    """The regime of ``cfg.mode`` as of each query close time (0 = chop / not enough history / no BTC data)."""
    if cfg.mode == "btc":
        if btc_1h is None or not len(btc_1h):
            return np.zeros(len(t_query), "int8")
        reg = btc_regime(btc_1h.c, cfg.btc_ema_hours, cfg.btc_ret_hours, cfg.btc_ret_thr)
        return asof(btc_1h.t_end, reg, t_query)
    if not len(own_1h):
        return np.zeros(len(t_query), "int8")
    reg = coin_regime(own_1h.c, cfg.coin_fast, cfg.coin_slow, cfg.coin_slope_hours)
    return asof(own_1h.t_end, reg, t_query)


def rule_signals(kl: pd.DataFrame, btc_kl: pd.DataFrame | None, cfg: RuleCfg) -> RuleSignals:
    """Signals for every 5m row of ``kl`` (closed bars, oldest first); ``btc_kl`` = BTCUSDT's 5m bars for the btc mode
    (None -> no trades; for BTCUSDT itself pass ``kl``). Row ``i`` is decided at the CLOSE of bar ``i``."""
    n = len(kl)
    t_close = kl["open_time"].to_numpy("int64") + BAR_MS
    own_1h = bars_1h(kl)
    btc_1h = None
    if cfg.mode == "btc" and btc_kl is not None:
        btc_1h = own_1h if btc_kl is kl else bars_1h(btc_kl)
    reg = regime_for(cfg, own_1h, btc_1h, t_close)
    if cfg.tf == "5m":
        lo, sh = _setup_raw(cfg, bars_5m(kl))
        due = np.ones(n, bool)
    else:
        lo_h, sh_h = _setup_raw(cfg, own_1h)
        j = np.searchsorted(t_close, own_1h.t_end)                # the 5m row that closes each complete hour
        ok = (j < n) & (t_close[np.minimum(j, max(n - 1, 0))] == own_1h.t_end) if n else np.zeros(len(own_1h), bool)
        lo, sh = np.zeros(n, bool), np.zeros(n, bool)
        lo[j[ok]], sh[j[ok]] = lo_h[ok], sh_h[ok]
        due = (t_close % HOUR_MS) == 0
    return RuleSignals(lo & (reg > 0) & due, sh & (reg < 0) & due, reg, due)


def last_signal(kl: pd.DataFrame, btc_kl: pd.DataFrame | None, cfg: RuleCfg) -> tuple[str | None, int]:
    """('long' | 'short' | None, regime) for the newest closed row of ``kl`` (what the live scorer reports)."""
    if len(kl) == 0:
        return None, 0
    s = rule_signals(kl, btc_kl, cfg)
    side = "long" if s.long[-1] else "short" if s.short[-1] else None
    return side, int(s.regime[-1])


def describe(cfg: RuleCfg) -> str:
    who = "BTC's trend" if cfg.mode == "btc" else "the coin's own trend"
    return f"{SETUP_NAMES[cfg.setup]} on {cfg.tf} bars, side chosen by {who}"
