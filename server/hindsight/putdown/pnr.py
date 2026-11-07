"""Put-down point-of-no-return detection.

Two small gradient-boosted classifiers over the per-second hand-object window
features: one fires on the PNR window (the instant an object has definitively
left the hand), the other confirms the contact -> no-contact state change. The
PNR label is rare (a handful of windows per video), so both heads train with
balanced sample weights.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


class PNRDetector:
    def __init__(self, random_state: int = 0):
        self.pnr = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=random_state
        )
        self.state = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=random_state
        )

    def fit(self, video) -> "PNRDetector":
        X = np.asarray(video.ho_feat, dtype=np.float32)
        y_pnr = np.asarray(video.ho_pnr).astype(int)
        y_state = np.asarray(video.ho_state).astype(int)
        self.pnr.fit(X, y_pnr, sample_weight=compute_sample_weight("balanced", y_pnr))
        self.state.fit(X, y_state, sample_weight=compute_sample_weight("balanced", y_state))
        return self

    def predict(self, ho_feat) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(ho_feat, dtype=np.float32))
        return self._prob(self.pnr, X), self._prob(self.state, X)

    @staticmethod
    def _prob(clf, X) -> np.ndarray:
        # a degenerate single-class fit (shouldn't happen on real data) has no column 1
        if len(clf.classes_) < 2:
            return np.full(len(X), float(clf.classes_[0]))
        return clf.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"pnr": self.pnr, "state": self.state}, f)

    @classmethod
    def load(cls, path: str | Path) -> "PNRDetector":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        det = cls()
        det.pnr, det.state = blob["pnr"], blob["state"]
        return det
