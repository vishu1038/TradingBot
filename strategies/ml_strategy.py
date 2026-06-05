"""Supervised-ML strategy — the recommended first AI step.

Trains a gradient-boosted classifier to predict the sign of the next bar's return from the
engineered indicator features, then trades in the predicted direction (subject to a
confidence threshold). LightGBM is used if available, otherwise it falls back to
scikit-learn's GradientBoostingClassifier so the scaffold runs out of the box.

Critical discipline (enforced by convention here, by the harness in run_backtest.py):
  * Labels look one bar into the FUTURE — features must not. `features.add_indicators`
    only uses past/current data, and we shift labels, so there is no look-ahead leak.
  * NEVER call `train()` and `generate_signals()` on the same rows you report metrics on.
    Use a chronological train/test split (walk-forward), never a random shuffle.
"""

import logging
import os

import numpy as np
import pandas as pd

from strategies.base import Strategy
from strategies.features import add_indicators, FEATURE_COLUMNS

logger = logging.getLogger()

try:
    from lightgbm import LGBMClassifier as _Model
    _BACKEND = "lightgbm"
except ImportError:  # pragma: no cover - fallback path
    from sklearn.ensemble import GradientBoostingClassifier as _Model
    _BACKEND = "sklearn"


def make_labels(df: pd.DataFrame, horizon: int = 1, deadband: float = 0.0) -> pd.Series:
    """Label = sign of forward return over `horizon` bars.

    `deadband` (e.g. 0.001) maps tiny moves to 0/flat so the model isn't forced to call
    noise. Returns a Series aligned to df (last `horizon` rows are NaN -> drop before fit).
    """
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    label = pd.Series(0, index=df.index, dtype="float")
    label[fwd_ret > deadband] = 1
    label[fwd_ret < -deadband] = -1
    label[fwd_ret.isna()] = np.nan
    return label


class MLStrategy(Strategy):
    name = "ml_gbm"

    def __init__(self, horizon: int = 1, deadband: float = 0.0,
                 confidence: float = 0.55, model_params: dict = None):
        self.horizon = horizon
        self.deadband = deadband
        self.confidence = confidence  # min predicted prob to act; else stay flat
        self.model_params = model_params or {}
        self.model = None
        self.classes_ = None
        logger.info("MLStrategy using %s backend", _BACKEND)

    # --- Training ----------------------------------------------------------
    def train(self, df: pd.DataFrame) -> dict:
        feats = add_indicators(df)
        labels = make_labels(feats, self.horizon, self.deadband)
        data = feats.assign(_label=labels).dropna(subset=FEATURE_COLUMNS + ["_label"])

        X = data[FEATURE_COLUMNS].values
        y = data["_label"].astype(int).values

        self.model = _Model(**self.model_params)
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        train_acc = float((self.model.predict(X) == y).mean())
        logger.info("MLStrategy trained on %s rows, in-sample acc=%.3f", len(y), train_acc)
        return {"n_samples": len(y), "train_accuracy": train_acc, "backend": _BACKEND}

    # --- Inference ---------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("MLStrategy.train() must be called before generate_signals()")

        feats = add_indicators(df)
        valid = feats[FEATURE_COLUMNS].dropna()
        signal = pd.Series(0, index=df.index, dtype=int)
        if valid.empty:
            return signal

        proba = self.model.predict_proba(valid.values)
        best_idx = proba.argmax(axis=1)
        best_prob = proba.max(axis=1)
        preds = np.array([self.classes_[i] for i in best_idx])

        # Act only when confident enough; otherwise flat.
        preds[best_prob < self.confidence] = 0
        signal.loc[valid.index] = preds.astype(int)

        # Decision uses current-bar features -> execute next bar (no look-ahead).
        return signal.shift(1).fillna(0).astype(int)

    # --- Persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "classes_": self.classes_,
                     "confidence": self.confidence, "horizon": self.horizon,
                     "deadband": self.deadband}, path)
        logger.info("MLStrategy saved to %s", path)

    def load(self, path: str) -> None:
        import joblib
        blob = joblib.load(path)
        self.model = blob["model"]
        self.classes_ = blob["classes_"]
        self.confidence = blob["confidence"]
        self.horizon = blob["horizon"]
        self.deadband = blob["deadband"]
        logger.info("MLStrategy loaded from %s", path)
