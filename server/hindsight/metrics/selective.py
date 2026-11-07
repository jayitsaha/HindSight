"""Selective-prediction metrics for the calibrated assistant.

`confidences` is a per-answer confidence; `correct` is a matching bool array of
whether that answer was right. We accept the most-confident answers first and
refuse the rest.
"""

from __future__ import annotations

import numpy as np


def _sort_desc(confidences, correct):
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=float)
    order = np.argsort(-conf, kind="stable")
    return ok[order]


def risk_coverage_curve(confidences, correct):
    """Risk (error rate among accepted) vs coverage (fraction accepted).

    Returns (coverage, risk), both length n, for coverage 1/n .. 1.
    """
    ok = _sort_desc(confidences, correct)
    n = len(ok)
    accepted = np.arange(1, n + 1)
    coverage = accepted / n
    risk = 1.0 - np.cumsum(ok) / accepted
    return coverage, risk


def aurc(confidences, correct) -> float:
    """Area under the risk-coverage curve; lower is better."""
    coverage, risk = risk_coverage_curve(confidences, correct)
    return float(np.trapezoid(risk, coverage))


def coverage_at_accuracy(confidences, correct, target_acc: float) -> float:
    """Largest coverage whose accepted set still has accuracy >= target_acc."""
    ok = _sort_desc(confidences, correct)
    n = len(ok)
    acc = np.cumsum(ok) / np.arange(1, n + 1)
    feasible = np.flatnonzero(acc >= target_acc)
    return float((feasible[-1] + 1) / n) if feasible.size else 0.0
