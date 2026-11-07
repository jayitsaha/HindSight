"""Action metrics: macro-F1 for classification, temporal mAP for localization.

Segments are tuples `(t_start, t_end, label, score)`; score is ignored for gts.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def temporal_iou(seg, segs: np.ndarray) -> np.ndarray:
    """1D IoU of one (start, end) against an (n, 2) array of segments."""
    s, e = seg
    inter = np.minimum(e, segs[:, 1]) - np.maximum(s, segs[:, 0])
    inter = np.clip(inter, 0.0, None)
    union = (e - s) + (segs[:, 1] - segs[:, 0]) - inter
    return np.where(union > 0, inter / union, 0.0)


def _average_precision(tp: np.ndarray, n_gt: int) -> float:
    # VOC all-points AP from a score-sorted TP/FP sequence.
    fp = 1 - tp
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_at_iou(preds, gts, thresh: float) -> float:
    n_gt = len(gts)
    if n_gt == 0:
        return 0.0
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda p: p[3], reverse=True)
    gt_spans = np.asarray([[g[0], g[1]] for g in gts], dtype=float)
    matched = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(len(preds))
    for i, p in enumerate(preds):
        ious = temporal_iou((p[0], p[1]), gt_spans)
        ious[matched] = -1.0  # each gt matches at most one prediction
        j = int(np.argmax(ious))
        if ious[j] >= thresh:
            tp[i] = 1.0
            matched[j] = True
    return _average_precision(tp, n_gt)


def temporal_map(preds, gts, iou_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)) -> dict:
    """Temporal action-localization mAP, averaged over classes and IoU thresholds.

    Returns per-threshold mAP plus the overall mean under key "mAP".
    """
    labels = {p[2] for p in preds} | {g[2] for g in gts}
    per_thresh = {}
    for t in iou_thresholds:
        aps = [
            _ap_at_iou([p for p in preds if p[2] == c], [g for g in gts if g[2] == c], t)
            for c in labels
        ]
        per_thresh[t] = float(np.mean(aps)) if aps else 0.0
    out = {t: per_thresh[t] for t in iou_thresholds}
    out["mAP"] = float(np.mean(list(per_thresh.values()))) if per_thresh else 0.0
    return out
