"""Out-of-sample backtest of a saved V3 model, trading it the way the live engine does (``tb.engine``).

What is reused (nothing of the engine / entry / adaptive / scorer / protect / risk code is modified):

* features + probabilities: ``tb.model_gbm.load`` -> ``GbmScorer.predict`` over ``tb.features_mtf.build_for_model`` (the SAME
  builder the live scorer calls through ``score_universe``); the model's own calibration and thresholds;
* entry cadence: ``tb.adaptive.AdaptiveState.step`` (regular gap, events, ``ev_extra_conf``) with the model's ``entry_cfg``
  resolved by ``tb.entry.EntryCfg.resolve`` and the Settings stored in the paper DB (opened read-only);
* per-coin gates: ``tb.entry.EntryGate`` semantics (cooldown_bars from the last entry / exit, ``sl_cooldown_bars`` after a losing exit);
* exits / sizing: ``tb.exits.resolve_exit`` (pct mode, time stop) -> ``tb.risk.size_trade``; caps: ``tb.risk.check_entry`` (open risk,
  side risk, total leverage, margin ratio, free margin), ``correlation_gate``, ``derate_multiplier`` (kill 50 %);
* profit protection: ``tb.protect`` (break-even / partial TP / trailing).  With a 3 % TP and a 6 % stop (pct mode) every
  protection level sits beyond the TP, so it is inert; ``protection_inert`` asserts that from the real functions.

Documented simplifications (see the report): hourly point-in-time universe (the engine refreshes every 10 min); the OI-jump event
never fires (no OI history for the archive bars; ``metrics=None`` is passed to ``bar_events``); entries fill at the decision bar's
close +/- slippage (the engine's market order right after the close); stops fill at the stop (or the bar open on a gap) with
slippage + stop slippage (``stop_fill="trigger"``, the ``tb.sim`` convention; ``"extreme"`` reproduces ``PaperExchange._eval``,
which fills at the bar's low / high); exits are evaluated on 5m OHLC with the stop first on a tie; kill / equity are checked at
decision and exit times; no liquidation modelling (the stop is 6 % away, liquidation ~9 %); the trade-level drift/slippage
multipliers are 1.0; lot-size rounding is ignored (continuous quantity).
"""
from __future__ import annotations

import heapq
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .. import adaptive, features_mtf as fm, model_gbm
from ..config import Settings, settings_from_rows
from ..data import frames, live, loader
from ..data.paths import archive_root, repo_root
from ..entry import EntryCfg
from ..exits import resolve_exit
from ..exchange.base import SymbolFilters
from ..protect import Protect, is_loss_exit, next_stop, partial_level
from ..risk import Book, check_entry, choose_leverage, correlation_gate, derate_multiplier, maintenance_margin, size_trade, sl_loss

BAR = 300_000
HOUR = 3_600_000
DAY = 86_400_000
MODEL_NAME = "v3_gbm_pct36_kal_20261019_0607"
PROB_START = pd.Timestamp("2026-04-01", tz="UTC").value // 10**6        # probabilities are computed from here on
CACHE = Path(os.environ.get("TB_BT_CACHE", "/Users/dungeonmaster/.claude/jobs/f7beae83/tmp/cache"))

PERIODS = {
    "A_holdout": ("2026-08-15", "2026-10-14"),
    "B_post": ("2026-10-14", "2026-10-20"),
    "C_dev_insample": ("2026-04-01", "2026-08-15"),
}


def ms(s: str) -> int:
    return pd.Timestamp(s, tz="UTC").value // 10**6


# ----------------------------------------------------------------------------- settings / model
def load_settings(db: str | Path | None = None) -> Settings:
    """Settings of the paper engine, read from its sqlite file in READ-ONLY mode."""
    db = Path(db or repo_root() / "var" / "tb_paper.sqlite")
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = {k: json.loads(v) for k, v in c.execute("SELECT key, value FROM settings")}
    finally:
        c.close()
    return settings_from_rows(rows)


def load_model(name: str = MODEL_NAME):
    sc = model_gbm.load(repo_root() / "models" / f"{name}.json")
    return sc


def protection_inert(cfg: Settings, spec) -> bool:
    """True when break-even / partial TP / trailing can never fire for this exit spec (pct mode, 3 % TP vs 6 % stop)."""
    pc = Protect.from_settings(cfg)
    entry, sl0, tp = 100.0, 100.0 * (1 - spec.sl_pct), 100.0 * (1 + spec.tp_pct)
    if partial_level("long", entry, sl0, tp, pc) is not None:
        return False
    # the favourable extreme needed to move the stop is >= the TP, which closes the position first
    _, new_sl, why = next_stop("long", entry, sl0, sl0, tp - 1e-6, None, False, pc)
    return why is None


# ----------------------------------------------------------------------------- data
def _cat(old: pd.DataFrame, new: pd.DataFrame | None, key: str) -> pd.DataFrame:
    if new is None or not len(new):
        return old
    return pd.concat([old, new], ignore_index=True).drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)


def _post(kind: str, sym: str) -> pd.DataFrame | None:
    p = CACHE / "post" / kind / f"{sym}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def load_sym(sym: str, with_post: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(klines, metrics, funding): the archive (read only) + the post-archive public-REST top-up kept in the scratch cache.
    A symbol that is not in the archive (a recent listing) is read from the REST cache ``CACHE/ext`` (``fetch_ext``)."""
    from ..data.paths import kline_path
    if not kline_path(sym).exists():
        e = CACHE / "ext"
        return (pd.read_parquet(e / "klines" / f"{sym}.parquet"), pd.read_parquet(e / "metrics" / f"{sym}.parquet"),
                pd.read_parquet(e / "funding" / f"{sym}.parquet"))
    kl, me, fu = loader.load(sym)
    if with_post:
        kl = _cat(kl, _post("klines", sym), "open_time")
        me = _cat(me, _post("metrics", sym), "create_time")
        fu = _cat(fu, _post("funding", sym), "calc_time")
    return kl, me, fu


def _rest(path: str, params: dict, pace: float = 0.25):
    time.sleep(pace)                        # <= 4 req/s
    return live.rest_get(path, params)


def fetch_post_klines(symbols: Iterable[str], now: int | None = None) -> dict[str, int]:
    """Public prod 5m klines after the archive's last bar of every symbol (limit 499 = request weight 2). Cached under CACHE/post."""
    now = now or int(time.time() * 1000)
    out = {}
    d = CACHE / "post" / "klines"
    d.mkdir(parents=True, exist_ok=True)
    for s in symbols:
        last = int(loader.load_klines(s)["open_time"].iloc[-1])
        cur, parts = last + BAR, []
        while cur + BAR <= now:
            rows = _rest("/fapi/v1/klines", {"symbol": s, "interval": "5m", "startTime": cur, "limit": 499})
            if not rows:
                break
            parts.append(live.parse_klines(rows))
            cur = int(rows[-1][0]) + BAR
            if len(rows) < 499:
                break
        if parts:
            df = frames.merge_frames(None, pd.concat(parts, ignore_index=True), "open_time")
            df = df[df["open_time"] + BAR <= now].reset_index(drop=True)      # closed bars only
            df.to_parquet(d / f"{s}.parquet")
            out[s] = len(df)
    return out


def fetch_post_derivs(symbols: Iterable[str], now: int | None = None) -> dict[str, tuple[int, int]]:
    now = now or int(time.time() * 1000)
    out = {}
    for k in ("metrics", "funding"):
        (CACHE / "post" / k).mkdir(parents=True, exist_ok=True)
    for s in symbols:
        _, me, fu = loader.load(s)
        m_last = int(me["create_time"].iloc[-1]) if len(me) else now - 3 * DAY
        nm = live.fetch_metrics(s, m_last + BAR, None, get=_rest, now=now, pace=0.25)
        nf = live.fetch_funding(s, int(fu["calc_time"].iloc[-1]) + 1000 if len(fu) else now - 3 * DAY, now, get=_rest, pace=0.25)
        nm.to_parquet(CACHE / "post" / "metrics" / f"{s}.parquet")
        nf.to_parquet(CACHE / "post" / "funding" / f"{s}.parquet")
        out[s] = (len(nm), len(nf))
    return out


def fetch_ext(sym: str, start_ms: int, now: int | None = None, derivs: bool = True) -> int:
    """Full public-REST history of a symbol that is not in the archive: closed 5m bars from ``start_ms`` (limit 1000 = weight 5),
    metrics (last <= 29 days, what REST serves and what the live engine has) and funding. Cached under CACHE/ext."""
    now = now or int(time.time() * 1000)
    for k in ("klines", "metrics", "funding"):
        (CACHE / "ext" / k).mkdir(parents=True, exist_ok=True)
    cur, parts = (start_ms // BAR) * BAR, []
    while cur + BAR <= now:
        rows = _rest("/fapi/v1/klines", {"symbol": sym, "interval": "5m", "startTime": cur, "limit": 1000}, pace=0.45)
        if not rows:
            break
        parts.append(live.parse_klines(rows))
        cur = int(rows[-1][0]) + BAR
        if len(rows) < 1000:
            break
    if not parts:
        return 0
    df = frames.merge_frames(None, pd.concat(parts, ignore_index=True), "open_time")
    df = df[df["open_time"] + BAR <= now].reset_index(drop=True)
    df.to_parquet(CACHE / "ext" / "klines" / f"{sym}.parquet")
    if derivs:
        live.fetch_metrics(sym, now - 28 * DAY, None, get=_rest, now=now, pace=0.25).to_parquet(CACHE / "ext" / "metrics" / f"{sym}.parquet")
        live.fetch_funding(sym, now - 40 * DAY, now, get=_rest, pace=0.25).to_parquet(CACHE / "ext" / "funding" / f"{sym}.parquet")
    return len(df)


# ----------------------------------------------------------------------------- probabilities
def compute_probs(sym: str, scorer, btc: pd.DataFrame, t_from: int = PROB_START, hist: int = 9600) -> pd.DataFrame:
    """Calibrated (prob_long, prob_short) for every closed 5m bar of ``sym`` with open_time >= t_from. The frame handed to the
    builder starts ``hist`` bars before t_from (the live window is the last 9600 bars); finite-memory features make it equal."""
    kl, me, fu = load_sym(sym)
    i0 = int(np.searchsorted(kl["open_time"].to_numpy(), t_from))
    if i0 >= len(kl):
        return pd.DataFrame({"open_time": np.zeros(0, "int64"), "pl": np.zeros(0, "float32"), "ps": np.zeros(0, "float32")})
    k = kl.iloc[max(0, i0 - hist):].reset_index(drop=True)
    b = btc[btc["open_time"] >= k["open_time"].iloc[0] - 300 * BAR]
    X, t_open = fm.build_for_model(scorer.meta, dict(kl=k, metrics=me, funding=fu, btc_kl=b.reset_index(drop=True)), return_index=True)
    P = scorer.predict(X)
    real = np.isin(t_open, k["open_time"].to_numpy())      # regularize() may add filled bars: skip them
    m = (t_open >= t_from) & real
    return pd.DataFrame({"open_time": t_open[m], "pl": P[m, 0].astype("float32"), "ps": P[m, 1].astype("float32")})


# ----------------------------------------------------------------------------- universe (point in time)
def build_universe(panel: dict[str, pd.DataFrame], g0: int, g1: int, cfg: Settings, funding: dict[str, pd.DataFrame],
                   first_bar: dict[str, int], archive_start: int, refresh_ms: int = HOUR) -> dict[str, np.ndarray]:
    """Point-in-time hot list per ``refresh_ms`` (the engine refreshes every 10 min; hourly here): eligible = 24h quote volume
    >= ``min_universe_volume_usd``, listed >= ``min_listing_days``, |last funding| <= ``max_abs_funding``; the 150 most liquid
    are ranked by ``atr_pct(12 x 5m) * sqrt(24h quote volume)`` (``tb.universe.score``) and the top ``universe_size`` kept
    (retention 1.0 = pure top-N). Uses only bars CLOSED at or before the refresh time; the list then applies to the bars that
    close in (T, T+refresh]. Returns {symbol: bool mask over the grid g0..g1 step 5m (by bar open_time)}."""
    grid = np.arange(g0, g1 + 1, BAR, dtype="int64")
    n = len(grid)
    syms = sorted(panel)
    ix = {s: i for i, s in enumerate(syms)}
    C = np.full((n, len(syms)), np.nan)
    H = C.copy(); Lw = C.copy(); Q = C.copy()
    for s in syms:
        k = panel[s]
        j = ((k["open_time"].to_numpy() - g0) // BAR).astype(int)
        ok = (j >= 0) & (j < n)
        for arr, col in ((C, "close"), (H, "high"), (Lw, "low"), (Q, "quote_volume")):
            arr[j[ok], ix[s]] = k[col].to_numpy()[ok]
    prev = np.vstack([np.full((1, len(syms)), np.nan), C[:-1]])
    tr = np.maximum(H - Lw, np.maximum(np.abs(H - prev), np.abs(Lw - prev)))
    nb = int(getattr(cfg, "universe_atr_bars", 12) or 12)
    atr = pd.DataFrame(tr).rolling(nb, min_periods=nb).mean().to_numpy()
    atrp = atr / C * 100.0
    vol = pd.DataFrame(np.nan_to_num(Q)).rolling(288, min_periods=200).sum().to_numpy()
    mask = {s: np.zeros(n, bool) for s in syms}
    ft = {s: (f["calc_time"].to_numpy(), f["funding_rate"].to_numpy()) for s, f in funding.items() if len(f)}
    size = int(cfg.universe_size)
    step = max(1, refresh_ms // BAR)
    first = np.array([first_bar.get(s, archive_start) for s in syms])
    for i in range(288, n - 1):
        close_t = grid[i] + BAR                       # the bar open_time grid[i] has closed at close_t
        if close_t % refresh_ms:
            continue
        v, a, c = vol[i], atrp[i], C[i]
        elig = np.isfinite(v) & np.isfinite(a) & np.isfinite(c) & (v >= cfg.min_universe_volume_usd)
        old = (first <= archive_start + BAR) | (close_t - first >= cfg.min_listing_days * DAY)
        elig &= old
        if cfg.max_abs_funding:
            for s in np.flatnonzero(elig):
                f = ft.get(syms[s])
                if f is not None:
                    p = np.searchsorted(f[0], close_t, side="right") - 1
                    if p >= 0 and abs(f[1][p]) > cfg.max_abs_funding:
                        elig[s] = False
        idx = np.flatnonzero(elig)
        if not len(idx):
            continue
        idx = idx[np.argsort(-v[idx])][:150]
        sc = a[idx] * np.sqrt(v[idx])
        top = idx[np.argsort(-sc)][:size]
        hi = min(n, i + 1 + step)
        for s in top:
            mask[syms[s]][i + 1:hi] = True
    return mask


# ----------------------------------------------------------------------------- decisions
def gen_decisions(sym: str, kl: pd.DataFrame, probs: pd.DataFrame, member: np.ndarray, member_t0: int, ent: EntryCfg,
                  thr: tuple[float, float], short_enabled: bool = True) -> pd.DataFrame:
    """All decision points of ``sym`` (regular + event) while it is in the hot list, from ``AdaptiveState.step`` itself.
    ``member`` is a bool mask over bars of the 5m grid starting at ``member_t0`` (by open_time)."""
    st = adaptive.AdaptiveState()
    t = kl["open_time"].to_numpy("int64")
    p = probs.set_index("open_time")
    pl_all = p["pl"].reindex(t).to_numpy()
    ps_all = p["ps"].reindex(t).to_numpy()
    sel = np.flatnonzero((t >= member_t0) & (t < member_t0 + len(member) * BAR))
    sel = sel[member[((t[sel] - member_t0) // BAR).astype(int)] & np.isfinite(pl_all[sel])]
    rows = []
    keep = adaptive.VOL_HISTORY + adaptive.VOL_WINDOW + 8
    for i in sel:
        closed = kl.iloc[max(0, i - keep + 1): i + 1]
        d = st.step(sym, closed, None, float(pl_all[i]), float(ps_all[i]), ent, thr[0], thr[1], short_enabled)
        if d.kind:
            rows.append((sym, int(t[i]), d.kind, d.side or "", float(d.prob), float(pl_all[i]), float(ps_all[i]),
                         ",".join(d.events)))
    return pd.DataFrame(rows, columns=["sym", "bar", "kind", "side", "prob", "pl", "ps", "events"])


# ----------------------------------------------------------------------------- simulation
@dataclass
class SimCfg:
    caps: bool = True                     # False: only the per-coin gates (cooldown, one position / coin); no portfolio caps / kill
    stop_fill: str = "trigger"            # "trigger" (tb.sim convention) | "extreme" (PaperExchange._eval: bar low / high)
    funding: bool = True
    start_equity: float = 200.0
    horizon_bars: int | None = None


class Market:
    """OHLC + funding arrays per symbol, shared by every simulation run (exit lookups are cached)."""

    def __init__(self, data: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]], t_from: int | None = None):
        self.a: dict[str, tuple] = {}
        for s, (kl, fu) in data.items():
            if t_from is not None:
                kl = kl[kl["open_time"] >= t_from]
            self.a[s] = (kl["open_time"].to_numpy("int64"), kl["open"].to_numpy(), kl["high"].to_numpy(), kl["low"].to_numpy(),
                         kl["close"].to_numpy(), fu["calc_time"].to_numpy("int64"), fu["funding_rate"].to_numpy("float64"))
        self.cache: dict[tuple, tuple] = {}
        self.ccache: dict[tuple, float | None] = {}

    def mark(self, sym: str, t_ms: int) -> float | None:
        """Close of the last bar closed at or before t_ms."""
        T, _, _, _, C, _, _ = self.a[sym]
        j = int(np.searchsorted(T, t_ms - BAR, side="right")) - 1
        return float(C[j]) if j >= 0 else None

    def exit_of(self, sym: str, side: str, bar_open: int, fill: float, sl: float, tp: float, horizon: int, slip: float,
                sslip: float, stop_fill: str) -> tuple:
        """(exit_ms, price, reason, hit_end) for a position opened on the close of the bar ``bar_open``: first bar after the entry
        whose adverse extreme reaches the stop (stop first on a tie) or the favourable extreme reaches the TP; else the time stop
        at the close of the bar ``horizon`` bars after the entry (market exit with slippage)."""
        key = (sym, side, bar_open, round(fill, 12), round(sl, 12), round(tp, 12), horizon, stop_fill)
        got = self.cache.get(key)
        if got is not None:
            return got
        T, O, H, L, C, _, _ = self.a[sym]
        t0 = bar_open + BAR
        j0 = int(np.searchsorted(T, t0))
        deadline = t0 + horizon * BAR
        j_to = int(np.searchsorted(T, deadline - BAR, side="left"))
        j1 = min(j_to, len(T) - 1)
        long = side == "long"
        res = None
        if j0 <= j1:
            lo, hi = L[j0:j1 + 1], H[j0:j1 + 1]
            sh = np.flatnonzero(lo <= sl) if long else np.flatnonzero(hi >= sl)
            th = np.flatnonzero(hi >= tp) if long else np.flatnonzero(lo <= tp)
            js = int(sh[0]) if len(sh) else 1 << 60
            jt = int(th[0]) if len(th) else 1 << 60
            if js < (1 << 60) or jt < (1 << 60):
                if js <= jt:
                    j = j0 + js
                    ext = L[j] if long else H[j]
                    base = (min(sl, O[j]) if long else max(sl, O[j])) if stop_fill == "trigger" else (min(sl, ext) if long else max(sl, ext))
                    s = slip + sslip
                    res = (int(T[j]) + BAR, base * (1 - s) if long else base * (1 + s), "sl", False)
                else:
                    j = j0 + jt
                    res = (int(T[j]) + BAR, tp, "tp", False)
            elif j_to <= len(T) - 1 and int(T[j1]) + BAR >= deadline:
                px = C[j1] * (1 - slip) if long else C[j1] * (1 + slip)
                res = (int(T[j1]) + BAR, px, "timeout", False)
            else:                                     # data ends first: still open, closed at the last close for the books
                px = C[j1] * (1 - slip) if long else C[j1] * (1 + slip)
                res = (int(T[j1]) + BAR, px, "open_end", True)
        else:
            res = (bar_open + BAR, fill, "open_end", True)
        self.cache[key] = res
        return res

    def corr(self, a: str, b: str, t_dec: int, bars: int = 288, min_overlap: int = 100) -> float | None:
        """``tb.risk.return_corr`` on numpy: 5m return correlation of the last ``bars`` aligned bars up to the decision close."""
        key = (a, b, t_dec) if a < b else (b, a, t_dec)
        if key in self.ccache:
            return self.ccache[key]
        rs = []
        for s in (a, b):
            T, _, _, _, C, _, _ = self.a[s]
            j = int(np.searchsorted(T, t_dec - BAR, side="right"))
            lo = max(0, j - 320)
            c, t = C[lo:j], T[lo:j]
            rs.append((t[1:], c[1:] / c[:-1] - 1.0))
        t1, r1 = rs[0]; t2, r2 = rs[1]
        _, i1, i2 = np.intersect1d(t1, t2, assume_unique=True, return_indices=True)
        x, y = r1[i1][-bars:], r2[i2][-bars:]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok][-bars:], y[ok][-bars:]
        out = None
        if len(x) >= min_overlap and np.std(x) > 0 and np.std(y) > 0:
            out = float(np.corrcoef(x, y)[0, 1])
        self.ccache[key] = out
        return out

    def funding_paid(self, sym: str, side: str, t_in: int, t_out: int, notional: float) -> float:
        _, _, _, _, _, ft, fr = self.a[sym]
        if not len(ft):
            return 0.0
        lo, hi = np.searchsorted(ft, t_in, side="right"), np.searchsorted(ft, t_out, side="right")
        s = float(fr[lo:hi].sum()) * notional
        return s if side == "long" else -s


def _stub_filters() -> SymbolFilters:
    return SymbolFilters(tick_size=1e-12, step_size=1e-12, min_qty=0.0, min_notional=5.0)


def simulate(cands: pd.DataFrame, mkt: Market, cfg: Settings, spec, sc: SimCfg, t_end: int, sim_info: dict | None = None) -> list[dict]:
    """Run candidate entries (columns sym, bar, side, prob, kind) through the engine's gates and the exit template.
    Returns the closed trades (dicts)."""
    sim_info = {} if sim_info is None else sim_info
    eff = spec.effective(cfg)
    f = _stub_filters()
    ent = EntryCfg.resolve(cfg, None)
    cooldown = 6 * BAR
    slc = cfg.sl_cooldown_bars * BAR
    horizon = int(sc.horizon_bars or spec.horizon_bars or 0) or 10**9
    fee, slip, sslip = cfg.fee_taker, cfg.slippage_pct, cfg.stop_slippage_pct
    kill_dd = cfg.kill_limit("paper_local")
    cands = cands.sort_values(["bar", "prob"], ascending=[True, False], kind="stable")
    wallet = float(sc.start_equity)
    open_: list[dict] = []             # open positions
    heap: list[tuple] = []             # (exit_ms, uid)
    pos_by_uid: dict[int, dict] = {}
    last: dict[str, int] = {}          # sym -> last entry / exit (ms)
    last_loss: dict[str, int] = {}
    trades: list[dict] = []
    killed = False
    kill_t = None
    uid = 0
    cl = cfg.corr_cap
    cfgs = (fee, slip, sslip)

    def close_pos(p: dict, t_x: int) -> None:
        nonlocal wallet
        open_.remove(p)
        p["done"] = True
        d = 1.0 if p["side"] == "long" else -1.0
        gross = (p["px"] - p["fill"]) * p["qty"] * d
        exit_fee = p["qty"] * p["px"] * fee
        fund = mkt.funding_paid(p["sym"], p["side"], p["t_in"], t_x, p["notional"]) if sc.funding else 0.0
        net = gross - p["entry_fee"] - exit_fee - fund
        wallet += gross - exit_fee - fund
        last[p["sym"]] = max(last.get(p["sym"], 0), t_x)
        if is_loss_exit(p["reason"], net):
            last_loss[p["sym"]] = max(last_loss.get(p["sym"], 0), t_x)
        trades.append(dict(sym=p["sym"], side=p["side"], kind=p["kind"], prob=p["prob"], t_in=p["t_in"], t_out=t_x,
                           fill=p["fill"], px=p["px"], qty=p["qty"], notional=p["notional"], pnl=net,
                           fees=p["entry_fee"] + exit_fee, funding=fund, reason=p["reason"], end=p["end"]))

    def settle(until: int) -> None:
        while heap and heap[0][0] <= until:
            t_x, u = heapq.heappop(heap)
            p = pos_by_uid.pop(u)
            if not p.get("done"):
                close_pos(p, t_x)

    def equity(at: int) -> float:
        e = wallet
        for p in open_:
            m = mkt.mark(p["sym"], at)
            if m is not None:
                e += (m - p["fill"]) * p["qty"] * (1.0 if p["side"] == "long" else -1.0)
        return e

    for r in cands.itertuples(index=False):
        t_dec = int(r.bar) + BAR                           # the decision bar's close = the moment of the entry
        if t_dec > t_end:
            continue
        settle(t_dec)
        if killed:
            break
        sym, side = r.sym, r.side
        if sc.caps:
            eq = equity(t_dec)
            if (sc.start_equity - eq) / sc.start_equity * 100.0 >= kill_dd:       # dd_basis "start"
                killed = True
                kill_t = t_dec
                for p in list(open_):
                    m = mkt.mark(p["sym"], t_dec) or p["fill"]
                    p.update(px=m * (1 - (1.0 if p["side"] == "long" else -1.0) * slip), reason="kill", end=False)
                    close_pos(p, t_dec)
                break
        if sym in {p["sym"] for p in open_}:
            continue
        if cooldown and t_dec - last.get(sym, -10**18) < cooldown:           # EntryGate.check: cooldown_bars from entry / exit
            continue
        if slc and t_dec - last_loss.get(sym, -10**18) < slc:
            continue
        ref = mkt.mark(sym, t_dec)
        if ref is None:
            continue
        mult = 1.0
        if sc.caps:
            dd = max(0.0, (sc.start_equity - eq) / sc.start_equity * 100.0)
            mult = derate_multiplier(dd, kill_dd)
            if mult <= 0:
                continue
            same = [p for p in open_ if p["side"] == side]
            if len(same) >= cl.max_correlated:                 # == tb.risk.correlation_gate (numpy version, equal to return_corr)
                hot = 0
                for p in same:
                    c = mkt.corr(sym, p["sym"], t_dec)
                    if c is not None and c > cl.max_corr:
                        hot += 1
                if hot >= cl.max_correlated:
                    continue
        sz = size_trade(eff, side, ref, spec.atr_arg(None, ref), f, None, risk_mult=mult, brackets=None)
        if not sz.ok:
            continue
        fill = ref * (1 + slip) if side == "long" else ref * (1 - slip)
        d = 1.0 if side == "long" else -1.0
        qty = sz.qty
        sl_p = fill * (1 - d * spec.sl_pct)
        tp_p = fill * (1 + d * spec.tp_pct)
        notional = qty * fill
        efee = notional * fee
        if sc.caps:
            lev = choose_leverage(side, fill, sl_p, notional, cfg.max_leverage, None, cfg.maint_margin_rate)
            if lev is None:
                continue
            margin = notional / lev
            used = sum(p["margin"] for p in open_)
            risk_l = sum(p["risk"] for p in open_ if p["side"] == "long")
            risk_s = sum(p["risk"] for p in open_ if p["side"] == "short")
            new_risk = sl_loss(cfg, qty, fill, sl_p, side, entry_fee=True)
            bk = Book(eq, eq - used, sum(p["notional"] for p in open_),
                      sum(maintenance_margin(p["notional"], None, cfg.maint_margin_rate) for p in open_),
                      risk_l + risk_s, risk_l, risk_s)
            if not check_entry(cfg, bk, notional, maintenance_margin(notional, None, cfg.maint_margin_rate), new_risk, margin, efee, side).ok:
                continue
        else:
            margin, new_risk = notional / cfg.leverage, 0.0
        t_x, px, why, end = mkt.exit_of(sym, side, int(r.bar), fill, sl_p, tp_p, horizon, slip, sslip, sc.stop_fill)
        uid += 1
        wallet -= efee
        p = dict(uid=uid, sym=sym, side=side, kind=r.kind, prob=float(r.prob), t_in=t_dec, fill=fill, qty=qty, notional=notional,
                 margin=margin, risk=new_risk - efee, entry_fee=efee, px=px, reason=why, end=end)
        open_.append(p)
        pos_by_uid[uid] = p
        heapq.heappush(heap, (t_x, uid))
        last[sym] = t_dec
    settle(10**18)
    sim_info["kill_t"] = kill_t
    return trades


_close_cache: dict[tuple, pd.Series] = {}


def _closes(mkt: Market, sym: str, t_dec: int, n: int = 320) -> pd.Series:
    T, _, _, _, C, _, _ = mkt.a[sym]
    j = int(np.searchsorted(T, t_dec - BAR, side="right"))
    return pd.Series(C[max(0, j - n):j], index=T[max(0, j - n):j])


# ----------------------------------------------------------------------------- statistics
def stats(trades: list[dict], days: float, start_equity: float = 200.0) -> dict:
    n = len(trades)
    if not n:
        return dict(n=0, win=float("nan"), per=float("nan"), total=0.0, maxdd=0.0, per_day=0.0, n_event=0, n_regular=0)
    df = pd.DataFrame(trades).sort_values("t_out")
    pnl = df["pnl"].to_numpy()
    eq = start_equity + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[start_equity, eq])[1:]
    return dict(n=n, win=float((pnl > 0).mean()), per=float(pnl.mean()), total=float(pnl.sum()), maxdd=float((peak - eq).max()),
                per_day=n / days, n_event=int((df["kind"] == "event").sum()), n_regular=int((df["kind"] == "regular").sum()),
                sl=int((df["reason"] == "sl").sum()), tp=int((df["reason"] == "tp").sum()), to=int((df["reason"] == "timeout").sum()))


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 1) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ----------------------------------------------------------------------------- baselines
def random_cands(dec: pd.DataFrame, model_entries: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Random entries with the model's per-(kind, side) counts, drawn uniformly from the SAME pool of decision points
    (regular points for regular entries, event points for event entries)."""
    rng = np.random.default_rng(seed)
    parts = []
    for kind in ("regular", "event"):
        pool = dec[dec["kind"] == kind]
        if not len(pool):
            continue
        for side in ("long", "short"):
            k = int(((model_entries["kind"] == kind) & (model_entries["side"] == side)).sum())
            if k <= 0:
                continue
            pick = pool.iloc[rng.choice(len(pool), size=min(k, len(pool)), replace=False)].copy()
            pick["side"], pick["prob"] = side, rng.random(len(pick))
            parts.append(pick)
    return pd.concat(parts, ignore_index=True) if parts else dec.iloc[0:0].copy()


def always_regular(dec: pd.DataFrame, side_mode: str = "both") -> pd.DataFrame:
    """Prob threshold 0 on the regular cadence: both sides qualify, the higher calibrated probability wins (``_pick``); ``long``
    forces long."""
    r = dec[dec["kind"] == "regular"].copy()
    r["side"] = "long" if side_mode == "long" else np.where(r["pl"] >= r["ps"], "long", "short")
    r["prob"] = np.where(r["side"] == "long", r["pl"], r["ps"])
    return r
