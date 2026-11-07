"""Turn a stream of action segments into labels and repeat alerts.

The classifier names each segment; the repetition scorer compares it against
recent segments. Time is used only to decide *which* priors are even worth
comparing (anything older than repeat_window_s is out), never as a feature of
the scorer itself.
"""

from __future__ import annotations

import numpy as np

from ..config import settings
from ..ingest.synthetic import SyntheticVideo
from ..timeline.models import ActionSegment, Alert
from .classifier import ActionClassifier
from .repetition import RepetitionScorer


class ActionDetector:
    # repeats are a minority of candidate pairs, so the scorer's calibrated
    # probability for a genuine repeat sits well below 0.5 but far above the
    # ~0 it gives unrelated activities; 0.15 is the gap between the two.
    def __init__(self, classifier: ActionClassifier, repetition: RepetitionScorer,
                 repeat_threshold: float = 0.15):
        self.classifier = classifier
        self.repetition = repetition
        self.repeat_threshold = repeat_threshold
        self.reset()

    def reset(self) -> None:
        self._history: list[tuple[ActionSegment, np.ndarray]] = []
        self.alerts: list[Alert] = []

    def process(self, video: SyntheticVideo) -> list[ActionSegment]:
        self.reset()
        out = []
        for seg in video.segments:
            win_idx = np.nonzero((video.win_t >= seg.t_start) & (video.win_t < seg.t_end))[0]
            label_id, conf = self.classifier.segment_label(win_idx, video.win_feat)
            emb = video.seg_emb[seg.seg_emb_idx]
            segment = ActionSegment(id=seg.idx, label=self.classifier.label_name(label_id),
                                    t_start=seg.t_start, t_end=seg.t_end, score=conf)
            self._ingest(segment, emb)
            out.append(segment)
        return out

    def step(self, label_id: int, emb: np.ndarray, t_start: float, t_end: float,
             score: float) -> tuple[ActionSegment, Alert | None]:
        """Streaming entry point: add one finished segment, get it back labeled + flagged."""
        segment = ActionSegment(id=len(self._history), label=self.classifier.label_name(label_id),
                                t_start=t_start, t_end=t_end, score=score)
        alert = self._ingest(segment, np.asarray(emb, dtype=np.float32))
        return segment, alert

    def _ingest(self, segment: ActionSegment, emb: np.ndarray) -> Alert | None:
        best_score, best_prior = 0.0, None
        for prior, prior_emb in self._history:
            if segment.t_start - prior.t_start > settings.repeat_window_s:
                continue
            p = self.repetition.score(prior, segment, prior_emb, emb)
            if p > best_score:
                best_score, best_prior = p, prior

        alert = None
        if best_prior is not None and best_score >= self.repeat_threshold:
            segment.is_repeat = True
            segment.repeat_of = best_prior.id
            segment.repeat_score = best_score
            gap = segment.t_start - best_prior.t_start
            alert = Alert(
                kind="repeat", t=segment.t_start, severity=best_score,
                message=f"You already did '{segment.label}' about {int(gap // 60)} min ago.",
                ref_id=best_prior.id,
            )
            self.alerts.append(alert)

        self._history.append((segment, emb))
        return alert
