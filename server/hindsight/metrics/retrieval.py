"""Ranking metrics for object retrieval.

Conventions used throughout:
- `ranked_relevant_bool` is relevance in rank order (best score first). A 1D
  sequence is one query; a 2D array/list-of-lists is (queries, ranks). Most of
  our queries have a single correct target, so recall@k is the hit rate.
- `average_precision` takes raw `scores` + binary `labels`, not a sorted list.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def recall_at_k(ranked_relevant_bool, k: int) -> float:
    """Fraction of queries with a relevant item in the top k."""
    rows = np.atleast_2d(np.asarray(ranked_relevant_bool, dtype=bool))
    return float(rows[:, :k].any(axis=1).mean())


def mrr(ranked) -> float:
    """Mean reciprocal rank.

    Accepts either a list of 1-indexed rank positions of the first relevant item
    (0 or <=0 means "not found"), or a list of per-query boolean rank lists.
    """
    first = ranked[0]
    if isinstance(first, (Sequence, np.ndarray)) and not np.isscalar(first):
        ranks = []
        for row in ranked:
            hits = np.flatnonzero(np.asarray(row, dtype=bool))
            ranks.append(hits[0] + 1 if hits.size else 0)
        ranks = np.asarray(ranks, dtype=float)
    else:
        ranks = np.asarray(ranked, dtype=float)
    rr = np.where(ranks > 0, 1.0 / ranks, 0.0)
    return float(rr.mean())


def average_precision(scores, labels) -> float:
    """AP for one query: higher score should mean more relevant."""
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, np.asarray(scores, dtype=float)))
