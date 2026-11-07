"""Match scorers over the shared pair feature; CosineScorer is the baseline."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .embed import pair_features


class Scorer:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Scorer":
        self._fit(X, y)
        s = self._score(X)
        # floor splits positives from negatives; used by the service refusal gate.
        self.floor = float(0.5 * (s[y == 1].mean() + s[y == 0].mean()))
        return self

    def rank(self, q: np.ndarray, C: np.ndarray) -> np.ndarray:
        return self._score(pair_features(q, C))

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def _score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CosineScorer(Scorer):
    def _score(self, X: np.ndarray) -> np.ndarray:
        d = X.shape[1] // 2
        return X[:, :d].sum(axis=1)  # == q . C


class _ClfScorer(Scorer):
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf = self._build().fit(X, y)

    def _score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def _build(self): ...


class LogRegScorer(_ClfScorer):
    def _build(self):
        return LogisticRegression(max_iter=2000, C=1.0)


class RandomForestScorer(_ClfScorer):
    def _build(self):
        return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)


class MLPScorer(_ClfScorer):
    def _build(self):
        return MLPClassifier(hidden_layer_sizes=(128,), max_iter=800, random_state=0)


_SCORERS = {
    "cosine": CosineScorer,
    "logreg": LogRegScorer,
    "rf": RandomForestScorer,
    "mlp": MLPScorer,
}


def build_scorer(name: str) -> Scorer:
    return _SCORERS[name]()
