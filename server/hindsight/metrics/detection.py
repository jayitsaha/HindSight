"""Detection metrics for put-down / misplacement."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(scores, labels) -> float:
    """ROC AUC, with a neutral 0.5 fallback when only one class is present."""
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        return 0.5
    return float(roc_auc_score(labels, np.asarray(scores, dtype=float)))


def pnr_timing_error(pred_times, true_times) -> dict:
    """Absolute timing error (seconds) of predicted point-of-no-return events.

    `pred_times` and `true_times` are already matched element-wise.
    """
    err = np.abs(np.asarray(pred_times, dtype=float) - np.asarray(true_times, dtype=float))
    return {"mean": float(err.mean()), "median": float(np.median(err))}
