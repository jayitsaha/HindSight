"""Feed frame-curation output (hindsight.video.curation) to the VLM evaluation.

The curation batch job writes one folder per session:

    <root>/<video_id>/kmeans_K100.mp4   K keyframes, in time order
    <root>/<video_id>/kmeans_K100.json  manifest: frames[i].t_sec = source time of output frame i

`root` is a local folder or a Hugging Face dataset written as ``hf:<repo_id>``
(e.g. ``hf:hindsight-cs7641/hindsight-curated-frames``); only the files for the chosen method/K are downloaded.

For each question, CuratedFrames picks the sessions to show, keeps only frames recorded before the question
(when a session is the question's own), orders sessions by recording order and returns labelled frames:

    cf = CuratedFrames("out", method="kmeans", k=100)
    frames = cf.frames_for(question)          # [(label, PIL.Image), ...]
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from hindsight.data.schema import QuestionRecord

SESSION_MODES = ("evidence", "primary")


def session_number(video_id: str) -> int:
    m = re.search(r"session_(\d+)", video_id)
    return int(m.group(1)) if m else 0


def fmt_time(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def resolve_root(root: str, method: str, k: int) -> str:
    """Local folder as-is; ``hf:<repo>`` is downloaded (only this method/K) and its cache folder returned."""
    if not root.startswith("hf:"):
        return root
    from huggingface_hub import snapshot_download
    return snapshot_download(root[3:], repo_type="dataset",
                             allow_patterns=[f"*/{method}_K{k}.mp4", f"*/{method}_K{k}.json"])


class CuratedFrames:
    def __init__(self, root: str, method: str = "kmeans", k: int = 100, sessions: str = "evidence",
                 label_frames: bool = True):
        if sessions not in SESSION_MODES:
            raise ValueError(f"sessions must be one of {SESSION_MODES}")
        self.method, self.k, self.sessions, self.label_frames = method, k, sessions, label_frames
        self.root = resolve_root(root, method, k)
        self._manifests: Dict[str, dict] = {}
        stem = f"{method}_K{k}"
        for vid in sorted(os.listdir(self.root)) if os.path.isdir(self.root) else []:
            j, v = (os.path.join(self.root, vid, f"{stem}.{ext}") for ext in ("json", "mp4"))
            if os.path.isfile(j) and os.path.isfile(v):
                self._manifests[vid] = {"json": j, "mp4": v}
        if not self._manifests:
            raise FileNotFoundError(f"No <video_id>/{stem}.mp4 + .json pairs under {self.root}")

    # ------------------------------------------------------------------ selection
    @property
    def video_ids(self) -> List[str]:
        return list(self._manifests)

    def sessions_for(self, q: QuestionRecord) -> List[str]:
        """'evidence': the sessions holding the answer evidence (the question's own session for
        unanswerable questions) - the same oracle session choice the coverage analysis uses.
        'primary': only the session the question is asked in."""
        if self.sessions == "primary" or not q.evidence_clips:
            vids = [q.video_id] if q.video_id else []
        else:
            vids = list(dict.fromkeys(c["video_id"] for c in q.evidence_clips))
        return sorted(vids, key=session_number)

    def covers(self, q: QuestionRecord) -> bool:
        vids = self.sessions_for(q)
        return bool(vids) and all(v in self._manifests for v in vids)

    # ------------------------------------------------------------------ frames
    def _read(self, vid: str) -> Tuple[List[float], List[Image.Image]]:
        import cv2
        files = self._manifests[vid]
        with open(files["json"]) as f:
            times = [fr["t_sec"] for fr in json.load(f)["frames"]]
        cap, images = cv2.VideoCapture(files["mp4"]), []
        while len(images) < len(times):
            ok, fr = cap.read()
            if not ok:
                break
            images.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        if len(images) != len(times):
            raise ValueError(f"{files['mp4']}: {len(images)} frames but manifest lists {len(times)}")
        return times, images

    def frames_for(self, q: QuestionRecord) -> List[Tuple[Optional[str], Image.Image]]:
        """Labelled frames for one question, oldest first. In the question's own session, frames
        recorded after the question time are dropped (the answer must come from the past)."""
        out = []
        for vid in self.sessions_for(q):
            times, images = self._read(vid)
            cut = q.cutoff_timestamp if (vid == q.video_id and q.cutoff_timestamp is not None) else float("inf")
            for t, im in zip(times, images):
                if t <= cut:
                    label = f"[session {session_number(vid)}, {fmt_time(t)}]" if self.label_frames else None
                    out.append((label, im))
        return out


def filter_questions(questions: Sequence[QuestionRecord], cf: CuratedFrames) -> List[QuestionRecord]:
    """Questions whose sessions are all in the curated set (the rest cannot be answered from it)."""
    return [q for q in questions if cf.covers(q)]
