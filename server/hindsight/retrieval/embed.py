"""PCA reduction and the learned-metric pair feature (rationale: ingest/synthetic.py)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA


class PCAReducer:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCAReducer":
        k = min(self.dim, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=k, whiten=False, random_state=0).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X).astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump({"dim": self.dim, "pca": self.pca}, path)

    def load(self, path: str | Path) -> "PCAReducer":
        d = joblib.load(path)
        self.dim, self.pca = d["dim"], d["pca"]
        return self


def pair_features(q: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Features for query q (D,) against candidates C (n, D) -> (n, 2D).

    Summing the first half recovers the dot product, so cosine is one linear
    readout of this feature; a fitted scorer beats it by reweighting dims."""
    q = q.reshape(1, -1)
    return np.hstack([q * C, np.abs(q - C)]).astype(np.float32)
