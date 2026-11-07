"""Wires the PNR detector and the misplacement model into timeline events.

`process` runs the learned detector over a whole video's hand-object windows,
keeps the put-downs it confirms, attaches the object/location the perception
layer recorded, and scores each for misplacement. `step` is the same logic for
one live window at a time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..timeline.models import Alert, PutDownEvent
from .misplacement import MisplacementModel
from .pnr import PNRDetector


class PutDownDetector:
    def __init__(self, pnr: PNRDetector, misplacement: MisplacementModel,
                 pnr_threshold: float = 0.5, alert_threshold: float = 0.5):
        self.pnr = pnr
        self.misplacement = misplacement
        self.pnr_threshold = pnr_threshold
        self.alert_threshold = alert_threshold
        self._live_t = 0.0

    def process(self, video) -> list[PutDownEvent]:
        pnr_prob, state_prob = self.pnr.predict(video.ho_feat)
        ho_t = np.asarray(video.ho_t, dtype=np.float32)
        events = []
        for i, (p, scene_emb) in enumerate(zip(video.putdowns, video.pd_scene_emb)):
            w = int(np.argmin(np.abs(ho_t - p.t)))
            if pnr_prob[w] < self.pnr_threshold:
                continue  # detector didn't confirm this put-down
            events.append(PutDownEvent(
                id=i,
                object_label=p.object_label,
                t=p.t,
                frame_id=p.frame_idx,
                location_id=p.location_id,
                unusualness=self.misplacement.score(p.object_label, scene_emb),
                state_change=bool(state_prob[w] >= 0.5),
            ))
        return events

    def alerts(self, events: list[PutDownEvent]) -> list[Alert]:
        out = []
        for e in events:
            if e.unusualness < self.alert_threshold:
                continue
            out.append(Alert(
                kind="misplacement",
                t=e.t,
                severity=e.unusualness,
                message=f"{e.object_label} left in an unusual spot",
                ref_id=e.id,
            ))
        return out

    def step(self, ho_window, object_label: str, scene_emb) -> Optional[PutDownEvent]:
        pnr_prob, state_prob = self.pnr.predict(ho_window)
        self._live_t += 1.0
        if pnr_prob[0] < self.pnr_threshold:
            return None
        return PutDownEvent(
            id=int(self._live_t),
            object_label=object_label,
            t=self._live_t,
            frame_id=int(self._live_t),
            location_id=-1,  # live: discrete room id unknown, scene embedding drives scoring
            unusualness=self.misplacement.score(object_label, scene_emb),
            state_change=bool(state_prob[0] >= 0.5),
        )
