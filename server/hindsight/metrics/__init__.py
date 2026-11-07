"""Evaluation metrics shared across HINDSIGHT modules."""

from .action import macro_f1, temporal_iou, temporal_map
from .clustering import ari, nmi
from .detection import auroc, pnr_timing_error
from .retrieval import average_precision, mrr, recall_at_k
from .selective import aurc, coverage_at_accuracy, risk_coverage_curve

__all__ = [
    "recall_at_k",
    "mrr",
    "average_precision",
    "auroc",
    "pnr_timing_error",
    "macro_f1",
    "temporal_iou",
    "temporal_map",
    "risk_coverage_curve",
    "aurc",
    "coverage_at_accuracy",
    "nmi",
    "ari",
]
