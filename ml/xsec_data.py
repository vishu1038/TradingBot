"""Data layer for the cross-sectional experiment: panel + cached feature cube + training rows.

    ctx = get_context()                      # panel (hourly + 5m memmaps), raw feature cube (memmap), tradable mask
    rows = build_rows(ctx, ks, horizon_h)    # X [N, F], y (gauss-rank of the 24h-relative return), k, s, rel
    S = score_matrix(ctx, ranker, ks)        # [T, S] model scores (NaN outside ks / the tradable universe)

numpy / pandas only (no torch).  Everything is keyed by the DECISION hour k (decision on the close of hour k).
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from tb import features_xsec as fx  # noqa: E402

CHUNK = 192


def archive_root() -> Path:
    return Path(os.environ.get("TB_ARCHIVE") or os.environ.get("TB_ARCHIVE_DIR") or ROOT / "data" / "archive")


def var_dir() -> Path:
    return Path(os.environ.get("TB_VAR") or os.environ.get("TB_VAR_DIR") or ROOT / "var")


@dataclass
class Ctx:
    panel: fx.Panel
    cube: np.ndarray              # [T, S, F_RAW] float32 memmap
    mask: np.ndarray              # [T, S] tradable at the decision on the close of hour k
    btc: int | None
    dec_ms: np.ndarray            # [T] decision time of every hour index

    @property
    def T(self) -> int:
        return self.panel.T

    @property
    def S(self) -> int:
        return self.panel.S


def get_context(root: Path | None = None, cache_root: Path | None = None, symbols: list[str] | None = None,
                log=print) -> Ctx:
    root = Path(root or archive_root())
    cache_root = Path(cache_root or var_dir() / "ml_cache")
    panel = fx.build_panel_from_archive(root, cache_root, symbols, progress=log)
    src = hashlib.sha1(Path(fx.__file__).read_bytes()).hexdigest()[:10]
    cube_path = cache_root / f"xsec_cube_{panel.meta.get('key', 'x')}_{src}.npy"
    if not cube_path.exists():
        log("[xsec] computing the raw feature cube ...")
        cube = fx.compute_raw_features(panel)
        tmp = cube_path.with_suffix(".tmp.npy")
        np.save(tmp, cube)
        os.replace(tmp, cube_path)
        del cube
    cube = np.load(cube_path, mmap_mode="r")
    mask = fx.tradable_mask(panel)
    return Ctx(panel, cube, mask, fx.btc_index(panel.syms), np.asarray(panel.decision_ms(np.arange(panel.T))))


def hour_range(ctx: Ctx, lo_ms: int, hi_ms: int) -> tuple[int, int]:
    """[k_lo, k_hi): hour indices whose decision time lies in [lo_ms, hi_ms)."""
    return int(np.searchsorted(ctx.dec_ms, lo_ms, side="left")), int(np.searchsorted(ctx.dec_ms, hi_ms, side="left"))


def targets(ctx: Ctx, horizon_h: int, risk_scale: bool = True):
    """(rel [T,S], y [T,S]) relative forward return and its gaussianised cross-sectional rank (training target).

    The return is the SIMPLE return from the open after the decision to the open h hours later, divided by the coin's
    trailing daily volatility (an equal-risk leg earns PnL proportional to that), minus the universe mean.  Without
    the scaling a rank target rewards low-volatility coins for a purely statistical reason (median < mean of a skewed
    return) that earns nothing when positions are sized per unit of risk."""
    from tb.sim.xsec import sigma_day
    fwd = fx.forward_returns(ctx.panel, horizon_h, "simple")
    if risk_scale:
        fwd = fx.risk_scaled(fwd, sigma_day(ctx.panel))
    rel = fx.relative_returns(fwd, ctx.mask)
    return rel, fx.rank_target(rel)


def transform_hours(ctx: Ctx, ks: np.ndarray) -> np.ndarray:
    """[len(ks), S, F_MODEL] model inputs for decision hours ``ks`` (chunked reads of the memmapped cube)."""
    out = np.zeros((len(ks), ctx.S, fx.F_MODEL), dtype="float32")
    for a in range(0, len(ks), CHUNK):
        kk = ks[a:a + CHUNK]
        out[a:a + CHUNK] = fx.xs_transform(np.asarray(ctx.cube[kk]), ctx.mask[kk], ctx.btc)
    return out


def build_rows(ctx: Ctx, ks: np.ndarray, horizon_h: int, tgt=None, need_label: bool = True) -> dict:
    """Training / evaluation rows for decision hours ``ks``: every tradable (hour, coin) with a known label."""
    if need_label:
        rel, y = tgt if tgt is not None else targets(ctx, horizon_h)
    else:
        rel = y = None
    Xs, ys, rs, kk_, ss_ = [], [], [], [], []
    for a in range(0, len(ks), CHUNK):
        kk = ks[a:a + CHUNK]
        X = fx.xs_transform(np.asarray(ctx.cube[kk]), ctx.mask[kk], ctx.btc)
        ok = ctx.mask[kk]
        if need_label:
            ok = ok & np.isfinite(y[kk])
        ii, ss = np.nonzero(ok)
        Xs.append(X[ii, ss])
        ys.append(y[kk][ii, ss] if need_label else np.zeros(len(ii), "float32"))
        rs.append(rel[kk][ii, ss] if need_label else np.zeros(len(ii), "float32"))
        kk_.append(kk[ii])
        ss_.append(ss)
    if not Xs:
        return dict(X=np.zeros((0, fx.F_MODEL), "float32"), y=np.zeros(0, "float32"), rel=np.zeros(0, "float32"),
                    k=np.zeros(0, "int64"), s=np.zeros(0, "int64"))
    return dict(X=np.concatenate(Xs), y=np.concatenate(ys).astype("float32"), rel=np.concatenate(rs).astype("float32"),
                k=np.concatenate(kk_).astype("int64"), s=np.concatenate(ss_).astype("int64"))


def score_matrix(ctx: Ctx, predict, ks: np.ndarray) -> np.ndarray:
    """[T, S] model scores at decision hours ``ks`` (NaN elsewhere and where the coin is not tradable)."""
    out = np.full((ctx.T, ctx.S), np.nan)
    for a in range(0, len(ks), CHUNK):
        kk = ks[a:a + CHUNK]
        X = fx.xs_transform(np.asarray(ctx.cube[kk]), ctx.mask[kk], ctx.btc)
        ii, ss = np.nonzero(ctx.mask[kk])
        if len(ii) == 0:
            continue
        out[kk[ii], ss] = predict(X[ii, ss])
    return out


def ic_matrix(scores: np.ndarray, rel: np.ndarray, ks: np.ndarray, min_n: int = 10) -> np.ndarray:
    """Spearman rank correlation between score and realised relative return, one value per hour in ``ks``."""
    import pandas as pd
    a = scores[ks]
    b = rel[ks]
    ok = np.isfinite(a) & np.isfinite(b)
    a = np.where(ok, a, np.nan)
    b = np.where(ok, b, np.nan)
    ra = pd.DataFrame(a).rank(axis=1).to_numpy()
    rb = pd.DataFrame(b).rank(axis=1).to_numpy()
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.nanmean(ra, axis=1, keepdims=True)
        mb = np.nanmean(rb, axis=1, keepdims=True)
        da, db = ra - ma, rb - mb
        c = np.nansum(da * db, axis=1) / np.sqrt(np.nansum(da * da, axis=1) * np.nansum(db * db, axis=1))
    c[n < min_n] = np.nan
    return c


def ic_summary(ic: np.ndarray, horizon_h: int) -> dict:
    """Mean IC and a t-statistic that respects the overlap of the labels (one sample per ``horizon_h`` hours)."""
    v = ic[np.isfinite(ic)]
    if len(v) == 0:
        return {"ic": float("nan"), "t": float("nan"), "n": 0}
    sub = ic[::max(1, int(horizon_h))]
    sub = sub[np.isfinite(sub)]
    t = float(sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub)))) if len(sub) > 2 and sub.std(ddof=1) > 0 else float("nan")
    return {"ic": float(v.mean()), "t": t, "n": int(len(sub)), "pos_frac": float((v > 0).mean())}


def with_labels(rows: dict, rel: np.ndarray, y: np.ndarray) -> dict:
    """Attach labels of one horizon to label-free rows and drop the rows whose label is unknown."""
    yy = y[rows["k"], rows["s"]]
    ok = np.isfinite(yy)
    return dict(X=rows["X"][ok], y=yy[ok].astype("float32"), rel=rel[rows["k"], rows["s"]][ok].astype("float32"),
                k=rows["k"][ok], s=rows["s"][ok])
