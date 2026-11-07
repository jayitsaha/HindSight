"""Selective classifier over answer confidence.

The gate accepts the most confident answers and refuses the rest. `fit` picks
the threshold with the largest coverage whose accepted answers still clear the
target accuracy — then pulls it up above any confidence at which calibration was
actually wrong. That second step is the restraint the module exists for: a user
with memory loss trusts whatever we say, so we never commit at a confidence
level we've already been burned at, even when the overall accuracy budget would
let a few errors through.
"""

from __future__ import annotations

import numpy as np

from ..metrics.selective import aurc as _aurc
from ..metrics.selective import coverage_at_accuracy as _cov_at_acc


class RefusalGate:
    def __init__(self, target_acc: float = 0.80):
        self.target_acc = target_acc
        self.threshold = np.inf  # refuse everything until fit

    def fit(self, confidences, correct) -> "RefusalGate":
        conf = np.asarray(confidences, dtype=float)
        ok = np.asarray(correct, dtype=bool)
        order = np.argsort(-conf, kind="stable")
        cs, oks = conf[order], ok[order]
        acc = np.cumsum(oks) / np.arange(1, len(cs) + 1)

        feasible = np.flatnonzero(acc >= self.target_acc)
        if feasible.size == 0:
            self.threshold = np.inf  # can't reach target at any coverage
            return self

        k = int(feasible[-1])  # top k+1 is max coverage at target accuracy
        acc_conf, acc_ok = cs[: k + 1], oks[: k + 1]
        wrong = acc_conf[~acc_ok]
        if wrong.size == 0:
            self.threshold = float(cs[k])
            return self
        # sit the threshold in the gap above every confidence we got wrong, at the
        # midpoint to the lowest right answer still above them — so a slightly
        # more confident wrong answer at test time still lands on the refuse side.
        w = float(wrong.max())
        right_above = acc_conf[acc_ok & (acc_conf > w)]
        self.threshold = 0.5 * (w + float(right_above.min())) if right_above.size \
            else float(np.nextafter(w, np.inf))
        return self

    def decide(self, confidence: float) -> bool:
        """True = refuse."""
        return float(confidence) < self.threshold

    def aurc(self, confidences, correct) -> float:
        return _aurc(confidences, correct)

    def coverage_at_accuracy(self, confidences, correct, target_acc: float | None = None) -> float:
        return _cov_at_acc(confidences, correct, target_acc or self.target_acc)
