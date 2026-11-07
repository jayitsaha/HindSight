"""Per-session state and the shared, fit-once model bundle.

The heavy scorers are trained once at startup on train_01 and shared across
every session; each session owns a TimelineStore (and, live, its own action
detector so repeat history is per-camera). Demo sessions keep the materialized
SyntheticVideo so the assistant can answer against the planted embedding space;
the live path instead reads detections back out of the store.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Optional

import numpy as np
from PIL import Image

from ..action import ActionClassifier, ActionDetector, RepetitionScorer
from ..assistant import Assistant
from ..config import settings
from ..ingest import synthetic as S
from ..putdown import MisplacementModel, PNRDetector, PutDownDetector
from ..timeline.models import Alert, Answer, Detection, PutDownEvent
from ..timeline.store import TimelineStore

_SYNTH = settings.data_dir / "synthetic"
_MAX_SESSIONS = 64  # keep the most-recently-used N; evicted stores are closed


class Session:
    def __init__(self, video_id: str, store: TimelineStore, video=None):
        self.video_id = video_id
        self.store = store
        self.video = video  # SyntheticVideo for demo sessions, else None (live camera)
        self.action: Optional[ActionDetector] = None  # live repeat tracker, built on demand


class SessionManager:
    _instance: Optional["SessionManager"] = None
    _init_lock = threading.Lock()

    def __init__(self):
        train = S.load(_SYNTH, "train_01")
        self.putdown = PutDownDetector(PNRDetector().fit(train), MisplacementModel().fit(train))
        self._clf = ActionClassifier().fit(train)
        self._rep = RepetitionScorer().fit(train)
        self.assistant = Assistant(scorer_name="logreg").fit(train)
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "SessionManager":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # --- lifecycle --------------------------------------------------------
    def new_session(self) -> str:
        vid = uuid.uuid4().hex[:12]
        self._register(Session(vid, TimelineStore()))
        return vid

    def get(self, vid: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(vid)
            if session is not None:
                self._sessions.move_to_end(vid)  # mark most-recently-used
            return session

    def demo(self, dataset: str) -> str:
        video = S.load(_SYNTH, dataset)  # FileNotFoundError if the dataset isn't materialized
        vid = f"{dataset}-{uuid.uuid4().hex[:6]}"
        session = Session(vid, TimelineStore(), video)
        self._populate(session, video)
        self._register(session)
        return vid

    def _register(self, session: Session) -> None:
        """Insert a session as most-recently-used, evicting + closing the oldest past the cap."""
        with self._lock:
            self._sessions[session.video_id] = session
            self._sessions.move_to_end(session.video_id)
            while len(self._sessions) > _MAX_SESSIONS:
                _, evicted = self._sessions.popitem(last=False)
                evicted.store.close()

    def answer(self, session: Session, question: str, t: float) -> Answer:
        ans = self.assistant.answer(
            question, t,
            video=session.video,
            store=None if session.video is not None else session.store,
        )
        session.store.add_answer(ans)
        return ans

    # --- frame ingestion --------------------------------------------------
    def ingest_image(self, session: Session, image: Image.Image, t: float):
        """Camera path: ground boxes, embed the crops, store them as detections.

        Best-effort — if no detector/embedder is available the frame is recorded
        with no detections rather than failing. A bare image carries no
        hand-object window or finished segment, so it yields no alerts.
        """
        from ..providers import get_embedder, get_grounder

        boxes, embs = [], None
        try:
            boxes = get_grounder().detect(image, "object")
            crops = [b.crop(image) for b in boxes]
            if crops:
                embs = get_embedder().embed_images(crops)
        except Exception:
            boxes, embs = [], None

        frame_id = session.store.add_frame(session.video_id, t, "", image.width, image.height)
        dets = []
        for i, b in enumerate(boxes):
            emb = embs[i] if embs is not None else None
            did = session.store.add_detection(
                frame_id, t, b.label or "object", float(b.score), b.xyxy, emb=emb)
            dets.append(Detection(id=did, frame_id=frame_id, t=t,
                                  label=b.label or "object", score=float(b.score), xyxy=b.xyxy))
        return dets, []

    def ingest_json(self, session: Session, body: dict):
        """Client-side perception path: a timestamp plus precomputed detections /
        embeddings, and optionally a hand-object window or a finished action
        segment the client already cut — those drive live misplacement / repeat
        alerts through the same fitted detectors the demo path uses."""
        t = float(body.get("t", 0.0))
        store = session.store
        frame_id = store.add_frame(session.video_id, t, "", 0, 0)
        out = []
        for d in body.get("detections") or []:
            label = d.get("label", "object")
            score = float(d.get("score", 1.0))
            xyxy = tuple(float(v) for v in d.get("xyxy", (0, 0, 0, 0)))
            emb = np.asarray(d["embedding"], dtype=np.float32) if d.get("embedding") else None
            did = store.add_detection(frame_id, t, label, score, xyxy, emb=emb)
            out.append(Detection(id=did, frame_id=frame_id, t=t, label=label, score=score, xyxy=xyxy))
        emb = body.get("embedding")
        if emb is not None:
            did = store.add_detection(frame_id, t, "object", 1.0, (0, 0, 0, 0),
                                      emb=np.asarray(emb, dtype=np.float32))
            out.append(Detection(id=did, frame_id=frame_id, t=t, label="object",
                                 score=1.0, xyxy=(0, 0, 0, 0)))

        alerts = self._putdown_alerts(session, body, t, frame_id)
        alerts += self._repeat_alerts(session, body)
        return out, alerts

    # --- live alerts ------------------------------------------------------
    def _putdown_alerts(self, session: Session, body: dict, t: float, frame_id: int) -> list[Alert]:
        """Run the fitted PNR + misplacement detectors on a live hand-object window."""
        window, scene = body.get("ho_window"), body.get("scene_embedding")
        if window is None or scene is None:
            return []
        try:
            X = np.atleast_2d(np.asarray(window, dtype=np.float32))
            scene_emb = np.asarray(scene, dtype=np.float32)
            object_label = str(body.get("object_label", "object"))
            pnr_prob, state_prob = self.putdown.pnr.predict(X)
            if float(pnr_prob[0]) < self.putdown.pnr_threshold:
                return []
            event = PutDownEvent(
                id=0, object_label=object_label, t=t, frame_id=frame_id, location_id=-1,
                unusualness=self.putdown.misplacement.score(object_label, scene_emb),
                state_change=bool(state_prob[0] >= 0.5))
            event.id = session.store.add_putdown(event)
            alerts = self.putdown.alerts([event])
        except Exception:
            return []
        for a in alerts:
            session.store.add_alert(a)
        return alerts

    def _repeat_alerts(self, session: Session, body: dict) -> list[Alert]:
        """Feed a finished action segment to the session's repeat detector."""
        seg = body.get("segment")
        if not seg or "embedding" not in seg:
            return []
        try:
            if session.action is None:
                session.action = ActionDetector(self._clf, self._rep)
            segment, alert = session.action.step(
                int(seg.get("label_id", -1)),
                np.asarray(seg["embedding"], dtype=np.float32),
                float(seg.get("t_start", 0.0)), float(seg.get("t_end", 0.0)),
                float(seg.get("score", 1.0)))
            session.store.add_action(segment)
        except Exception:
            return []
        if alert is None:
            return []
        session.store.add_alert(alert)
        return [alert]

    # --- demo population --------------------------------------------------
    def _populate(self, session: Session, video) -> None:
        store = session.store
        frames_dir = _SYNTH / video.video_id / "frames"

        frame_ids: dict[int, int] = {}
        for d in video.dets:
            if d.frame_idx not in frame_ids:
                path = str(frames_dir / f"{d.frame_idx:06d}.png")
                frame_ids[d.frame_idx] = store.add_frame(
                    session.video_id, d.t, path, S.FRAME_W, S.FRAME_H)
            store.add_detection(frame_ids[d.frame_idx], d.t, d.label, 1.0, d.xyxy,
                                object_id=d.object_id, emb=video.det_emb[d.emb_idx])

        events = self.putdown.process(video)
        for e in events:
            store.add_putdown(e)
        for a in self.putdown.alerts(events):
            store.add_alert(a)

        action = ActionDetector(self._clf, self._rep)
        for seg in action.process(video):
            store.add_action(seg)
        for a in action.alerts:
            store.add_alert(a)
