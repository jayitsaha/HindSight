"""Did this action just happen again?

Repeating the same activity within minutes is a dementia red flag, so we score
pairs of action segments for "is this a repeat of that". Features come only from
the two segment embeddings — cosine, elementwise product, abs-diff — plus a
same-action indicator. The time gap is deliberately left out: the proposal
forbids it so the model can't just learn "close in time == repeat". Candidate
gating by time lives in the service, not here.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from ..config import settings
from ..ingest.synthetic import ActionSeg, SyntheticVideo


def is_repeat_pair(a: ActionSeg, b: ActionSeg) -> bool:
    # b repeats a directly, or a and b are both repeats of the same earlier action.
    if b.repeat_of == a.idx or a.repeat_of == b.idx:
        return True
    return a.repeat_of is not None and a.repeat_of == b.repeat_of


def pair_features(emb_a: np.ndarray, emb_b: np.ndarray, same_class: float) -> np.ndarray:
    cos = float(emb_a @ emb_b)
    return np.concatenate([[cos], emb_a * emb_b, np.abs(emb_a - emb_b), [same_class]]).astype(np.float32)


def _training_pairs(video: SyntheticVideo) -> tuple[np.ndarray, np.ndarray]:
    segs, emb = video.segments, video.seg_emb
    X, y = [], []
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            if b.t_start - a.t_start > settings.repeat_gap_s:
                break  # segments are time-ordered; nothing past the gap is a candidate
            same = float(a.label == b.label)
            X.append(pair_features(emb[a.seg_emb_idx], emb[b.seg_emb_idx], same))
            y.append(int(is_repeat_pair(a, b)))
    return np.asarray(X, dtype=np.float32), np.asarray(y)


class RepetitionScorer:
    def __init__(self):
        # strong L2 + a small net on purpose: there are ~130 embedding-derived
        # features but only a handful of positive pairs per video, so a loose
        # net memorizes the training embeddings and collapses on held-out data.
        # Heavy regularization keeps it leaning on the signal that generalizes.
        self.net = MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0,
                                 max_iter=4000, tol=1e-6, n_iter_no_change=40,
                                 random_state=0)

    def fit(self, video: SyntheticVideo) -> "RepetitionScorer":
        X, y = _training_pairs(video)
        self.net.fit(X, y)
        return self

    def score(self, seg_a, seg_b, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        feat = pair_features(emb_a, emb_b, float(seg_a.label == seg_b.label))
        return float(self.net.predict_proba(feat[None])[0, 1])


class SameClassRule:
    """Baseline: a pair repeats iff the two segments carry the same action label."""

    def fit(self, video: SyntheticVideo) -> "SameClassRule":
        return self

    def score(self, seg_a, seg_b, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return 1.0 if seg_a.label == seg_b.label else 0.0
