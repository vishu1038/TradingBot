"""Time-based walk-forward splits with purge gaps (numpy only).

Layout (all boundaries in ms, shared by every symbol):

    |---- train ----|purge|-- val --|purge|---- test ----|        (one block per fold, expanding train)
    ...                                      |purge|---- HOLDOUT (last N days, locked) ----|

``purge`` >= the label horizon (+ window) so no training/validation label overlaps the period after it.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

DAY_MS = 86_400_000
BAR_MS = 300_000
HORIZON = 1152
WINDOW = 64


def make_splits(t: np.ndarray, n_folds=4, test_days=30, val_days=14, holdout_days=60,
                purge_bars=HORIZON + WINDOW, min_train_days=45) -> dict:
    """Time boundaries (ms).  purge >= label horizon between train|val, val|test and dev|holdout."""
    purge = purge_bars * BAR_MS
    t_min, t_end = int(t.min()), int(t.max()) + BAR_MS
    hold_start = t_end - holdout_days * DAY_MS
    dev_end = hold_start - purge
    folds = []
    for k in range(n_folds):
        te = dev_end - (n_folds - 1 - k) * test_days * DAY_MS
        ts = te - test_days * DAY_MS
        ve = ts - purge
        vs = ve - val_days * DAY_MS
        tr_end = vs - purge
        if tr_end - t_min < min_train_days * DAY_MS:
            continue
        folds.append(dict(fold=k, train=(t_min, tr_end), val=(vs, ve), test=(ts, te)))
    ve = hold_start - purge
    vs = ve - val_days * DAY_MS
    final = dict(train=(t_min, vs - purge), val=(vs, ve), holdout=(hold_start, t_end))
    return dict(purge_bars=purge_bars, folds=folds, final=final)


def rows_in(ds, rng_ms: tuple[int, int], mask: np.ndarray) -> np.ndarray:
    lo, hi = rng_ms
    return np.nonzero(mask & (ds.t >= lo) & (ds.t < hi))[0]


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")
