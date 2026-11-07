from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .models import ActionSegment, Alert, Answer, Detection, Frame, PutDownEvent, TimelineSnapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames(
  id INTEGER PRIMARY KEY, video_id TEXT, t REAL, path TEXT, width INT, height INT);
CREATE TABLE IF NOT EXISTS detections(
  id INTEGER PRIMARY KEY, frame_id INT, t REAL, label TEXT, score REAL,
  x0 REAL, y0 REAL, x1 REAL, y1 REAL, object_id INT, emb BLOB);
CREATE TABLE IF NOT EXISTS putdowns(
  id INTEGER PRIMARY KEY, object_label TEXT, t REAL, frame_id INT,
  location_id INT, unusualness REAL, state_change INT);
CREATE TABLE IF NOT EXISTS actions(
  id INTEGER PRIMARY KEY, label TEXT, t_start REAL, t_end REAL, score REAL,
  is_repeat INT, repeat_of INT, repeat_score REAL);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY, kind TEXT, t REAL, severity REAL, message TEXT, ref_id INT);
CREATE TABLE IF NOT EXISTS answers(
  id INTEGER PRIMARY KEY, question TEXT, t_asked REAL, refused INT,
  confidence REAL, text TEXT, window TEXT, support TEXT);
CREATE INDEX IF NOT EXISTS det_label ON detections(label);
CREATE INDEX IF NOT EXISTS det_frame ON detections(frame_id);
"""


class TimelineStore:
    def __init__(self, path: str | Path = ":memory:"):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._db.executescript(_SCHEMA)

    def _exec(self, sql: str, args=()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.lastrowid

    # --- writes -----------------------------------------------------------
    def add_frame(self, video_id: str, t: float, path: str, width: int, height: int) -> int:
        return self._exec(
            "INSERT INTO frames(video_id,t,path,width,height) VALUES(?,?,?,?,?)",
            (video_id, t, path, width, height),
        )

    def add_detection(
        self, frame_id: int, t: float, label: str, score: float,
        xyxy, object_id: Optional[int] = None, emb: Optional[np.ndarray] = None,
    ) -> int:
        blob = np.asarray(emb, dtype=np.float32).tobytes() if emb is not None else None
        return self._exec(
            "INSERT INTO detections(frame_id,t,label,score,x0,y0,x1,y1,object_id,emb)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (frame_id, t, label, score, *map(float, xyxy), object_id, blob),
        )

    def add_putdown(self, e: PutDownEvent) -> int:
        return self._exec(
            "INSERT INTO putdowns(object_label,t,frame_id,location_id,unusualness,state_change)"
            " VALUES(?,?,?,?,?,?)",
            (e.object_label, e.t, e.frame_id, e.location_id, e.unusualness, int(e.state_change)),
        )

    def add_action(self, s: ActionSegment) -> int:
        return self._exec(
            "INSERT INTO actions(label,t_start,t_end,score,is_repeat,repeat_of,repeat_score)"
            " VALUES(?,?,?,?,?,?,?)",
            (s.label, s.t_start, s.t_end, s.score, int(s.is_repeat), s.repeat_of, s.repeat_score),
        )

    def add_alert(self, a: Alert) -> int:
        return self._exec(
            "INSERT INTO alerts(kind,t,severity,message,ref_id) VALUES(?,?,?,?,?)",
            (a.kind, a.t, a.severity, a.message, a.ref_id),
        )

    def add_answer(self, a: Answer) -> int:
        return self._exec(
            "INSERT INTO answers(question,t_asked,refused,confidence,text,window,support)"
            " VALUES(?,?,?,?,?,?,?)",
            (a.question, a.t_asked, int(a.refused), a.confidence, a.text,
             json.dumps(a.window), json.dumps(a.support_frames)),
        )

    # --- reads ------------------------------------------------------------
    # Reads share the connection with writes on a check_same_thread=False handle,
    # so they take the same lock; fetch rows inside it, build pydantic outside.
    def _fetchall(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def frames(self, video_id: Optional[str] = None) -> list[Frame]:
        if video_id:
            rows = self._fetchall("SELECT * FROM frames WHERE video_id=? ORDER BY t", (video_id,))
        else:
            rows = self._fetchall("SELECT * FROM frames ORDER BY t")
        return [Frame(**dict(r)) for r in rows]

    def detections(self, label: Optional[str] = None) -> list[Detection]:
        if label:
            rows = self._fetchall("SELECT * FROM detections WHERE label=? ORDER BY t", (label,))
        else:
            rows = self._fetchall("SELECT * FROM detections ORDER BY t")
        return [self._det(r) for r in rows]

    def embeddings(self, label: Optional[str] = None) -> tuple[list[Detection], np.ndarray]:
        """Detections and their embedding matrix, positionally aligned.

        Rows with a NULL or wrong-width emb are dropped from both the detection
        list and the matrix so callers can index det_emb[i] <-> dets[i] safely.
        Returns (dets, (N, D)); (all_dets, (N, 0)) when nothing has an embedding.
        """
        rows = self._fetchall(
            "SELECT * FROM detections" + (" WHERE label=?" if label else "") + " ORDER BY id",
            (label,) if label else (),
        )
        vecs = [np.frombuffer(r["emb"], dtype=np.float32) if r["emb"] else None for r in rows]
        widths = [v.size for v in vecs if v is not None and v.size]
        if not widths:
            return [self._det(r) for r in rows], np.zeros((len(rows), 0), np.float32)
        width = max(set(widths), key=widths.count)  # dominant width; deterministic
        dets, keep = [], []
        for r, v in zip(rows, vecs):
            if v is not None and v.size == width:
                dets.append(self._det(r))
                keep.append(v)
        return dets, np.vstack(keep)

    def putdowns(self) -> list[PutDownEvent]:
        rows = self._fetchall("SELECT * FROM putdowns ORDER BY t")
        return [PutDownEvent(
            id=r["id"], object_label=r["object_label"], t=r["t"], frame_id=r["frame_id"],
            location_id=r["location_id"], unusualness=r["unusualness"],
            state_change=bool(r["state_change"])) for r in rows]

    def actions(self) -> list[ActionSegment]:
        rows = self._fetchall("SELECT * FROM actions ORDER BY t_start")
        return [ActionSegment(
            id=r["id"], label=r["label"], t_start=r["t_start"], t_end=r["t_end"], score=r["score"],
            is_repeat=bool(r["is_repeat"]), repeat_of=r["repeat_of"], repeat_score=r["repeat_score"])
            for r in rows]

    def alerts(self) -> list[Alert]:
        rows = self._fetchall("SELECT * FROM alerts ORDER BY t")
        return [Alert(kind=r["kind"], t=r["t"], severity=r["severity"],
                      message=r["message"], ref_id=r["ref_id"]) for r in rows]

    def _count(self, sql: str, args=()) -> int:
        with self._lock:
            return int(self._db.execute(sql, args).fetchone()[0])

    def snapshot(self, video_id: str) -> TimelineSnapshot:
        return TimelineSnapshot(
            video_id=video_id,
            frames=self._count("SELECT COUNT(*) FROM frames WHERE video_id=?", (video_id,)),
            detections=self._count("SELECT COUNT(*) FROM detections"),
            putdowns=self.putdowns(),
            actions=self.actions(),
            alerts=self.alerts(),
        )

    @staticmethod
    def _det(r: sqlite3.Row) -> Detection:
        return Detection(
            id=r["id"], frame_id=r["frame_id"], t=r["t"], label=r["label"], score=r["score"],
            xyxy=(r["x0"], r["y0"], r["x1"], r["y1"]), object_id=r["object_id"])

    def close(self) -> None:
        self._db.close()
