"""Evaluate zero-shot Chronos-Bolt forecasts (written by chronos_probe.py forecast).

    .venv/bin/python ml/chronos_eval.py vol  var/ml/chronos_chronos-bolt-tiny.parquet           # (a) vs ATR
    ml-env/bin/python ml/chronos_eval.py feat var/ml/chronos_...parquet --cache var/ml_cache/<key>  # (b) as features

(a) vol : is the forecast spread (q90 - q10 of the +12-bar log price) a better predictor of the realised 12-bar
    volatility than ATR%?  Rank correlations pooled and within symbol, plus the incremental R^2 of
    ln(spread) on top of ln(ATR%) in a strict time split (fit on the first 70 % of the sampled bars).
(b) feat: does adding Chronos-derived features (spread / ATR, median drift, skew) to the network's last-bar
    features improve the out-of-sample AUC of the trade labels?  Small MLP (torch CPU, 3 threads), time split,
    3 seeds, with vs without the 3 extra features, on the SAMPLED bars only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

Z90 = 1.2815515655


def spearman(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def load(path):
    df = pd.read_parquet(path)
    df = df[np.isfinite(df[["q10", "q50", "q90", "atr_pct", "rv_h"]]).all(axis=1) & (df["rv_h"] > 0)].copy()
    df["sig"] = (df["q90"] - df["q10"]) / (2 * Z90)
    df = df[df["sig"] > 0].sort_values("open_time").reset_index(drop=True)
    return df


def r2(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta, X1


def cmd_vol(a):
    df = load(a.path)
    n = len(df)
    y = np.log(df["rv_h"].to_numpy())
    la, lc = np.log(df["atr_pct"].to_numpy()), np.log(df["sig"].to_numpy())
    out = {"n_forecasts": n, "symbols": int(df["symbol"].nunique())}
    out["spearman_pooled_vs_rv"] = dict(chronos=spearman(df["sig"], df["rv_h"]), atr=spearman(df["atr_pct"], df["rv_h"]))
    out["spearman_pooled_vs_absret"] = dict(chronos=spearman(df["sig"], df["absret_h"]), atr=spearman(df["atr_pct"], df["absret_h"]))
    # within-symbol: demean the logs per symbol (removes the cross-sectional level, which both get right)
    g = df["symbol"]
    yd, ad, cd = (pd.Series(v).groupby(g).transform(lambda s: s - s.mean()).to_numpy() for v in (y, la, lc))
    out["within_symbol_corr_ln"] = dict(chronos=float(np.corrcoef(cd, yd)[0, 1]), atr=float(np.corrcoef(ad, yd)[0, 1]))
    cut = int(0.7 * n)
    res = {}
    # partial correlation within symbol: does ln(sigma_chronos) explain what ln(ATR%) leaves unexplained?
    ba = (ad * yd).sum() / (ad * ad).sum()
    bc = (ad * cd).sum() / (ad * ad).sum()
    out["within_symbol_partial_corr_chronos_given_atr"] = float(np.corrcoef(yd - ba * ad, cd - bc * ad)[0, 1])
    # strict time split R^2 (pooled ln rv ~ ln atr [+ ln sigma])
    for name, X in (("atr", la[:, None]), ("chronos", lc[:, None]), ("atr+chronos", np.column_stack([la, lc]))):
        beta, X1 = r2(y[:cut], X[:cut])
        pred = np.column_stack([np.ones(n - cut), X[cut:]]) @ beta
        ss = ((y[cut:] - pred) ** 2).sum()
        st = ((y[cut:] - y[cut:].mean()) ** 2).sum()
        res[name] = float(1 - ss / st)
    out["oos_r2_ln_rv"] = res
    # direction: does the median forecast carry sign information?
    d = df["q50"].to_numpy()
    out["median_sign_accuracy"] = float(np.mean(np.sign(d[d != 0]) == np.sign(df["ret_h"].to_numpy()[d != 0])))
    out["median_corr_with_ret"] = float(np.corrcoef(d, df["ret_h"].to_numpy())[0, 1])
    out["median_spearman_with_ret"] = spearman(d, df["ret_h"])
    qs = (df["ret_h"] - df["q50"]).abs() / df["sig"]
    out["coverage_q10_q90"] = float(((df["ret_h"] >= df["q10"]) & (df["ret_h"] <= df["q90"])).mean())      # nominal 0.80
    out["atr_scaled_coverage_q10_q90"] = None
    print(json.dumps(out, indent=1))
    Path(a.path).with_suffix(".vol.json").write_text(json.dumps(out, indent=1))


def cmd_feat(a):
    import dataset as D
    import torch
    torch.set_num_threads(3)
    cdir = Path(a.cache)
    ds = D.load_cached(cdir)
    df = load(a.path)
    sym_id = {s: i for i, s in enumerate(ds.symbols)}
    df = df[df["symbol"].isin(sym_id)].reset_index(drop=True)
    sym = np.asarray(ds.sym)
    # contiguous symbol blocks -> start offsets
    starts = {i: int(np.searchsorted(sym, i, side="left")) for i in range(len(ds.symbols))}
    ends = {i: int(np.searchsorted(sym, i, side="right")) for i in range(len(ds.symbols))}
    rows = np.full(len(df), -1)
    for s, i in sym_id.items():
        m = (df["symbol"] == s).to_numpy()
        if not m.any():
            continue
        tt = np.asarray(ds.t[starts[i]:ends[i]])
        pos = np.searchsorted(tt, df.loc[m, "open_time"].to_numpy())
        ok = (pos < len(tt)) & (tt[np.clip(pos, 0, len(tt) - 1)] == df.loc[m, "open_time"].to_numpy())
        idx = np.where(m)[0]
        rows[idx[ok]] = starts[i] + pos[ok]
    keep = rows >= 0
    df, rows = df[keep].reset_index(drop=True), rows[keep]
    o = np.argsort(df["open_time"].to_numpy(), kind="stable")
    df, rows = df.iloc[o].reset_index(drop=True), rows[o]
    X = np.asarray(ds.X[rows]).astype("float32")
    atr_i = ds.feature_names.index("atr_pct")
    ex = np.column_stack([np.log(df["sig"] / (X[:, atr_i] / 100.0 + 1e-9)), df["q50"] / df["sig"],
                          ((df["q90"] - df["q50"]) - (df["q50"] - df["q10"])) / (df["q90"] - df["q10"])]).astype("float32")
    ex = np.nan_to_num(ex, nan=0.0, posinf=0.0, neginf=0.0)
    cut = int(0.7 * len(df))
    res = {"n_rows": int(len(df)), "n_train": cut, "tasks": {}}
    for key in a.keys.split(","):
        y = ds.lab[key][rows]
        ok = np.isfinite(y)
        tr, te = ok.copy(), ok.copy()
        tr[cut:] = False
        te[:cut] = False
        r = {}
        for name, F_ in (("base", X), ("base+chronos", np.column_stack([X, ex])), ("chronos_only", np.column_stack([X[:, atr_i:atr_i + 1], ex]))):
            mu = np.median(F_[tr], 0)
            sd = 1.4826 * np.median(np.abs(F_[tr] - mu), 0)
            sd = np.where(sd > 1e-6, sd, F_[tr].std(0) + 1e-6)
            Z = np.clip((F_ - mu) / sd, -5, 5).astype("float32")
            aucs = []
            for seed in range(3):
                aucs.append(_mlp_auc(Z[tr], y[tr], Z[te], y[te], seed))
            r[name] = dict(auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs)))
        res["tasks"][key] = r
        print(key, json.dumps(r), flush=True)
    Path(a.path).with_suffix(".feat.json").write_text(json.dumps(res, indent=1))


def _mlp_auc(Xtr, ytr, Xte, yte, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.SiLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype("float32"))
    n = len(Xt)
    for ep in range(12):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[b])[:, 0], yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = net(torch.from_numpy(Xte))[:, 0].numpy()
    from evaluate import auc
    return auc(yte, p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vol", "feat"])
    ap.add_argument("path")
    ap.add_argument("--cache", default="")
    ap.add_argument("--keys", default="fixed_long,fixed_short,atr32_long,atr32_short")
    a = ap.parse_args()
    {"vol": cmd_vol, "feat": cmd_feat}[a.cmd](a)
