"""Gradient-boosted trees entry model (experiment C): sklearn ``HistGradientBoostingClassifier`` on the
multi-timeframe feature vector (``tb.features_mtf``).  CPU only; no torch.

Artifact: ``models/{name}.json`` (the usual metadata with ``kind: "gbm"``, ``feature_names``, ``timeframes``,
``history_bars_5m``, ``calibration``, ``thresholds``, ``exit_cfg``, ``entry_cfg``, ``metrics`` ...) plus
``models/{name}.pkl``: a pickle of ``{"long": clf, "short": clf, "sklearn": version}``.

SECURITY: a pickle executes code when loaded - only ever load artifacts this repo's training wrote.  Pickles are
also sklearn-version specific: the json records ``sklearn_version`` and the same pinned version
(``scikit-learn==1.7.2``) is installed in both venvs.

``load(path) -> GbmScorer`` implements the Scorer interface v2 (``score``, ``score_universe``, ``kind``,
``history_bars_5m``, see ``tb.model_np.MtfBase``).  Trees are scale invariant, so the features are fed raw; the
artifact's ``norm.mean`` holds the training MEDIAN of every feature (used for the occlusion "reasons").
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

from .model_np import MtfBase, logit

DEFAULT_PARAMS = dict(
    learning_rate=0.05, max_iter=250, max_depth=4, max_leaf_nodes=15, min_samples_leaf=800,
    l2_regularization=5.0, max_bins=63, early_stopping=False, random_state=0,
)


def make_estimator(params: dict | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    return HistGradientBoostingClassifier(**p)


class GbmScorer(MtfBase):
    """``kind: "gbm"``: one classifier per side, probabilities -> logits -> the artifact's calibration."""
    kind = "gbm"

    def __init__(self, meta: dict, models: dict):
        super().__init__(meta)
        self.models = {"long": models["long"], "short": models["short"]}

    def _raw_logits(self, X_raw: np.ndarray) -> np.ndarray:
        X = np.asarray(X_raw, "float32")
        pl = self.models["long"].predict_proba(X)[:, 1]
        ps = self.models["short"].predict_proba(X)[:, 1]
        return np.stack([logit(pl), logit(ps)], axis=1)


def save(models_dir: str | Path, name: str, meta: dict, models: dict) -> None:
    """Write ``{name}.pkl`` then ``{name}.json`` atomically (json last: the scanner keys on it)."""
    import sklearn
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, kind="gbm", sklearn_version=sklearn.__version__, weights_file=f"{name}.pkl")
    tmp = d / f".{name}.pkl.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"long": models["long"], "short": models["short"], "sklearn": sklearn.__version__}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, d / f"{name}.pkl")
    tmpj = d / f".{name}.json.tmp"
    tmpj.write_text(json.dumps(meta, indent=1, default=float))
    os.replace(tmpj, d / f"{name}.json")


def load(path) -> GbmScorer:
    """``path``: the ``.json``, the ``.pkl`` or the bare model path."""
    p = Path(path)
    if p.suffix in (".json", ".pkl", ".npz"):
        p = p.with_suffix("")
    meta = json.loads(Path(str(p) + ".json").read_text())
    with open(str(p) + ".pkl", "rb") as fh:
        blob = pickle.load(fh)                                  # noqa: S301 - our own artifact, see module doc
    return GbmScorer(meta, blob)
