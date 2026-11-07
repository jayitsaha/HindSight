"""Window-level action labeling.

An MLP over the (W, 96) action-window features ("InternVideo2-like"). Windows
carry the signal; segments are labeled by majority vote over the windows that
fall inside them, so a few noisy windows don't flip a segment.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..ingest.synthetic import SyntheticVideo


class ActionClassifier:
    def __init__(self, vocab: list[str] | None = None):
        self.vocab = vocab
        self.scaler = StandardScaler()
        # tight tolerance + patience so training reliably reaches the (well
        # separated) solution; otherwise BLAS thread noise stops it early at a
        # poor minimum and val scores swing run to run.
        self.net = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4,
                                 max_iter=3000, tol=1e-6, n_iter_no_change=40,
                                 random_state=0)

    def fit(self, video: SyntheticVideo) -> "ActionClassifier":
        self.vocab = list(video.actions_vocab)
        X = self.scaler.fit_transform(video.win_feat)
        self.net.fit(X, video.win_label)
        return self

    def predict_windows(self, win_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self.scaler.transform(win_feat)
        probs = self.net.predict_proba(X)
        labels = self.net.classes_[probs.argmax(1)]
        return labels, probs

    def segment_label(self, seg_win_indices: np.ndarray, win_feat: np.ndarray) -> tuple[int, float]:
        """Aggregate the windows of one segment into a single (label_id, confidence)."""
        idx = np.asarray(seg_win_indices, dtype=int)
        if idx.size == 0:
            return -1, 0.0
        labels, probs = self.predict_windows(win_feat[idx])
        # average the class posteriors over the segment, then take the top class.
        mean_p = probs.mean(0)
        top = int(mean_p.argmax())
        return int(self.net.classes_[top]), float(mean_p[top])

    def label_name(self, label_id: int) -> str:
        return self.vocab[label_id] if self.vocab and 0 <= label_id < len(self.vocab) else "unknown"

    def save(self, path: str | Path) -> None:
        joblib.dump({"vocab": self.vocab, "scaler": self.scaler, "net": self.net}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ActionClassifier":
        blob = joblib.load(path)
        obj = cls(blob["vocab"])
        obj.scaler = blob["scaler"]
        obj.net = blob["net"]
        return obj
