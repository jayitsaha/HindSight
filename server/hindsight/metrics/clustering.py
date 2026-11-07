"""Clustering agreement metrics for the object/identity clustering."""

from __future__ import annotations

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def nmi(labels_true, labels_pred) -> float:
    return float(normalized_mutual_info_score(labels_true, labels_pred))


def ari(labels_true, labels_pred) -> float:
    return float(adjusted_rand_score(labels_true, labels_pred))
