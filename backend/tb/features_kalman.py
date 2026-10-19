"""Kalman-filter state-estimation features per timeframe (experiment E; pure numpy).

"Vehicle-style" state estimation: a local linear trend model on log price per timeframe,

    state  x = [level, slope]            level_t = level_{t-1} + slope_{t-1} + eta1,   slope_t = slope_{t-1} + eta2
    obs    y_t = level_t + eps           (y = ln close of the CLOSED timeframe bars)

filtered over a FIXED window of the last ``L`` bars (``TF_SPEC[tf]["kal_L"]``), started from a diffuse prior at the
window's first bar.  A fixed window makes the value at a bar independent of how much older history exists, so
training (full archive) and live (``history_bars_5m`` bars) agree exactly - the usual recursive filter would only
agree up to a convergence tail.

Scale-free: inside each window the log prices are expressed in units of the window's own realised bar volatility
``s = sqrt(mean(r^2))`` and the noise variances are fixed fractions of it (``Q_LEVEL``, ``Q_SLOPE``, ``R_OBS``).  The
covariance recursion is therefore the SAME for every window - it is computed once - and only the (cheap,
vectorised over windows) mean recursion depends on the data.  The parameters are fixed a-priori, NOT fitted.

Features per timeframe (prefix ``kal_<h1|h4|d1>_``):
    gap        (price - filtered level) / s        >0: price above the smoothed level (stretched up)
    slope      filtered slope / s                  trend per bar in volatility units
    slope_z    slope / sqrt(P_slope)               signal-to-noise of the trend (t-statistic)
    innov_var  ln(mean of normalised innovation^2 over the last ``NIS_BARS`` bars)  ~0 when the model fits, >0 regime break
    post_sd    100 * sqrt(P_level) * s             posterior level uncertainty (% of price)
    sigma      100 * s                             realised bar volatility (% per bar)

Importing this module registers the group ``kalman`` with ``tb.features`` (``groups=["kalman"]`` on
``build_features``: 3 timeframes mapped onto the 5m rows) - the multi-timeframe builder uses ``compute`` directly.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import features as ft
from .features import BAR_MS, register_feature_group
from .features_mtf import DEFAULT_TFS, TF_SPEC, htf_index, resample_ohlcv

Q_LEVEL = 0.40      # level process-noise variance, in units of the window's bar variance
Q_SLOPE = 0.003     # slope random-walk variance
R_OBS = 0.30        # observation noise variance
P0_SLOPE = 1.0      # diffuse-ish prior on the slope
NIS_BARS = 12       # innovation-variance averaging window (bars)

_KEYS = ("gap", "slope", "slope_z", "innov_var", "post_sd", "sigma")

LABELS = {}
for _tf, _s in TF_SPEC.items():
    _p = _s["p"]
    LABELS.update({
        f"kal_{_p}_gap": f"price {{:+.2f}} vol-units from the {_tf} Kalman level",
        f"kal_{_p}_slope": f"{_tf} Kalman slope {{:+.2f}} vol-units/bar",
        f"kal_{_p}_slope_z": f"{_tf} Kalman trend t-stat {{:+.1f}}",
        f"kal_{_p}_innov_var": f"{_tf} Kalman innovation variance ln {{:+.2f}}",
        f"kal_{_p}_post_sd": f"{_tf} Kalman level uncertainty {{:.2f}}%",
        f"kal_{_p}_sigma": f"{_tf} bar volatility {{:.2f}}%",
    })


def names(timeframes=DEFAULT_TFS) -> list[str]:
    return [f"kal_{TF_SPEC[tf]['p']}_{k}" for tf in timeframes for k in _KEYS]


def _riccati(L: int, q1=Q_LEVEL, q2=Q_SLOPE, r=R_OBS, p0s=P0_SLOPE):
    """Data-independent gains / innovation variances / posterior covariances for steps 1..L-1."""
    Fm = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q1, q2])
    P = np.diag([r, p0s])
    K = np.zeros((L, 2))
    S = np.ones(L)
    Pll = np.zeros(L)
    Pss = np.zeros(L)
    Pll[0], Pss[0] = P[0, 0], P[1, 1]
    for t in range(1, L):
        Pp = Fm @ P @ Fm.T + Q
        S[t] = Pp[0, 0] + r
        K[t] = Pp[:, 0] / S[t]
        P = Pp - np.outer(K[t], Pp[0, :])
        Pll[t], Pss[t] = P[0, 0], P[1, 1]
    return K, S, Pll, Pss


def filter_windows(logp: np.ndarray, L: int, q1=Q_LEVEL, q2=Q_SLOPE, r=R_OBS) -> dict[str, np.ndarray]:
    """Run the fixed-window filter ending at every bar ``i >= L-1`` of ``logp`` (ln prices).  Returns arrays of
    length ``len(logp)`` (NaN for the first ``L-1`` bars and wherever the window contains a non-finite price)."""
    n = len(logp)
    out = {k: np.full(n, np.nan) for k in _KEYS}
    if n < L or L < 3:
        return out
    Y = sliding_window_view(np.asarray(logp, "float64"), L)                     # [M, L], oldest first
    M = len(Y)
    good = np.isfinite(Y).all(axis=1)
    Y = np.where(good[:, None], Y, 0.0)
    d = np.diff(Y, axis=1)
    s = np.sqrt(np.maximum((d * d).mean(axis=1), 1e-12))                         # window bar volatility (log units)
    Z = (Y - Y[:, :1]) / s[:, None]                                              # unit-free, first obs = 0
    K, S, Pll, Pss = _riccati(L, q1, q2, r)
    lvl = np.zeros(M)
    slp = np.zeros(M)
    nis = np.zeros((M, L - 1))
    for t in range(1, L):
        pl = lvl + slp
        nu = Z[:, t] - pl
        lvl = pl + K[t, 0] * nu
        slp = slp + K[t, 1] * nu
        nis[:, t - 1] = nu * nu / S[t]
    last = L - 1
    o = {}
    o["gap"] = Z[:, last] - lvl
    o["slope"] = slp
    o["slope_z"] = slp / np.sqrt(Pss[last])
    o["innov_var"] = np.log(np.maximum(nis[:, -NIS_BARS:].mean(axis=1), 1e-6))
    o["post_sd"] = 100.0 * np.sqrt(Pll[last]) * s
    o["sigma"] = 100.0 * s
    for k in _KEYS:
        v = o[k].copy()
        v[~good] = np.nan
        out[k][L - 1:] = v
    return out


def compute(kl, timeframes=DEFAULT_TFS, rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Kalman features of the CLOSED bars of each timeframe, as-of the close of every requested 5m row.
    ``kl``: 5m klines on a regular grid.  Returns ``{feature name: array over rows}`` (NaN where undefined)."""
    t_open = ft._ms(kl["open_time"])
    rows = np.arange(len(kl)) if rows is None else np.asarray(rows, "int64")
    rc = t_open[rows] + BAR_MS
    res: dict[str, np.ndarray] = {}
    for tf in timeframes:
        s = TF_SPEC[tf]
        bars = resample_ohlcv(kl, tf)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.log(bars["close"])
        f = filter_windows(lp, s["kal_L"])
        j = htf_index(rc, bars["end_ms"])
        ok = j >= 0
        for k in _KEYS:
            v = np.full(len(rows), np.nan)
            v[ok] = f[k][j[ok]]
            res[f"kal_{s['p']}_{k}"] = v
    return res


def _group_fn(kl, t_open, extra):
    cols = compute(kl, DEFAULT_TFS)
    return np.stack([cols[n] for n in names(DEFAULT_TFS)], axis=1)


if "kalman" not in ft._EXTRA:
    register_feature_group("kalman", names(DEFAULT_TFS), _group_fn, LABELS)
