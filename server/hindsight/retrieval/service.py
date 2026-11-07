"""Object retrieval: "where is my X?" -> latest frame + box of X."""

from __future__ import annotations

import numpy as np

from ..config import settings
from ..ingest.synthetic import LOOKALIKES
from ..timeline.models import RetrievalHit
from .embed import PCAReducer, pair_features
from .scorers import build_scorer

_LOOK: dict[str, list[str]] = {}
for _a, _b in LOOKALIKES:
    _LOOK.setdefault(_a, []).append(_b)
    _LOOK.setdefault(_b, []).append(_a)


class ObjectRetrieval:
    def __init__(self, scorer_name: str = "logreg"):
        self.scorer = build_scorer(scorer_name)
        self._embedder = None
        self.reducer: PCAReducer | None = None
        self._raw_dim: int | None = None       # crop width PCA was fit on
        self._expected_dim: int | None = None   # feature width the scorer trained on

    def fit(self, video, seed: int = 0) -> "ObjectRetrieval":
        # The proposal calls for a 128-D PCA; only worth it when the crops are
        # wider than that (the 64-D synthetic data passes straight through).
        self._raw_dim = int(video.det_emb.shape[1])
        self.reducer = None
        if self._raw_dim > settings.pca_dim:
            self.reducer = PCAReducer(settings.pca_dim).fit(video.det_emb)
        det_emb = self._reduce(video.det_emb)
        query_emb = self._reduce(video.query_emb)
        self._expected_dim = int(det_emb.shape[1])
        X, y = _training_pairs(video, np.random.default_rng(seed),
                               det_emb=det_emb, query_emb=query_emb)
        self.scorer.fit(X, y)
        return self

    def _reduce(self, X: np.ndarray) -> np.ndarray:
        if self.reducer is not None and X.shape[1] == self._raw_dim:
            return self.reducer.transform(X)
        return X

    def score_candidates(self, query_emb, C) -> np.ndarray:
        """Rank crops C (n, D) against the query. Applies the fitted PCA when the
        width matches; falls back to native-space cosine when it doesn't (e.g. a
        live CLIP query against a scorer trained on the synthetic space)."""
        q = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        C = np.asarray(C, dtype=np.float32)
        C = self._reduce(C)
        if self.reducer is not None and q.shape[0] == self._raw_dim:
            q = self.reducer.transform(q.reshape(1, -1))[0]
        if self._expected_dim is not None and C.shape[1] == self._expected_dim \
                and q.shape[0] == self._expected_dim:
            return self.scorer.rank(q, C)
        return _cosine(q, C)

    def locate(self, query_emb, dets, det_emb, t_asked, query_text: str = "") -> RetrievalHit:
        cand = np.array([i for i, d in enumerate(dets) if d.t <= t_asked])
        if cand.size == 0:
            return RetrievalHit(query=query_text, frame_id=-1, t=t_asked,
                                xyxy=(0, 0, 0, 0), score=0.0, margin=0.0, found=False)
        scores = self.score_candidates(query_emb, det_emb[cand])
        order = np.argsort(-scores)
        top1 = float(scores[order[0]])
        top2 = float(scores[order[1]]) if order.size > 1 else top1
        margin = top1 - top2

        floor = getattr(self.scorer, "floor", -np.inf)
        keep = scores >= floor
        times = np.array([dets[i].t for i in cand])
        if keep.any():
            kept = np.where(keep)[0]
            j = int(kept[np.argmax(times[kept])])  # latest high-scoring detection
            found = True
        else:
            j = int(order[0])
            found = False

        d = dets[int(cand[j])]
        return RetrievalHit(
            query=query_text,
            frame_id=int(getattr(d, "frame_id", getattr(d, "frame_idx", -1))),
            t=float(d.t), xyxy=tuple(float(v) for v in d.xyxy),
            score=float(scores[j]), margin=float(margin), found=found,
        )

    def from_timeline(self, store, query_text: str, t: float) -> RetrievalHit:
        if self._embedder is None:
            from ..providers import get_embedder
            self._embedder = get_embedder()
        q = self._embedder.embed_texts([query_text])[0]
        dets, det_emb = store.embeddings()
        return self.locate(q, dets, det_emb, t, query_text=query_text)


def _cosine(q: np.ndarray, C: np.ndarray) -> np.ndarray:
    if q.shape[0] != C.shape[1]:
        return np.full(C.shape[0], -np.inf)  # incomparable widths -> refuse
    qn = q / max(float(np.linalg.norm(q)), 1e-8)
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-8)
    return Cn @ qn


def _crops_by_label(video) -> dict[str, np.ndarray]:
    by: dict[str, list[int]] = {}
    for d in video.dets:
        by.setdefault(d.label, []).append(d.emb_idx)
    return {k: np.array(v) for k, v in by.items()}


def _training_pairs(video, rng, det_emb=None, query_emb=None,
                    n_pos: int = 25, n_look: int = 15, n_other: int = 8):
    det_emb = video.det_emb if det_emb is None else det_emb
    query_emb = video.query_emb if query_emb is None else query_emb
    by = _crops_by_label(video)
    feats, labels = [], []
    for q in video.queries:
        if q.kind != "object" or q.query_emb_idx is None:
            continue
        qv = query_emb[q.query_emb_idx]
        tgt = q.target_label
        pos = _sample(by[tgt], n_pos, rng)
        feats.append(pair_features(qv, det_emb[pos]))
        labels += [1] * pos.size

        neg = []
        for lk in _LOOK.get(tgt, []):
            neg += list(_sample(by[lk], n_look, rng))
        others = [l for l in by if l != tgt and l not in _LOOK.get(tgt, [])]
        for l in rng.choice(others, min(6, len(others)), replace=False):
            neg += list(_sample(by[l], n_other, rng))
        neg = np.array(neg)
        feats.append(pair_features(qv, det_emb[neg]))
        labels += [0] * neg.size
    return np.vstack(feats), np.array(labels)


def _sample(idx: np.ndarray, n: int, rng) -> np.ndarray:
    return rng.choice(idx, min(n, idx.size), replace=False)
