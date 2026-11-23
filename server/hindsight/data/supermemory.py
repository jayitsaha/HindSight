"""SuperMemory-VQA annotation helpers: load the QA file, parse evidence time spans, map video ids to repo paths.

Used by hindsight.curation (make_sessions, analyze).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

REPO_ID = "OSU-AIoT-MLSys-Lab/SuperMemory-VQA"
SKILL = "object_location_memory"


def load_qa(path: Optional[str] = None) -> List[dict]:
    """Load all_qa.json (downloads it from Hugging Face if no path is given)."""
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO_ID, "data/json/all_qa.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def to_sec(x) -> float:
    """Times are usually seconds, but a few records use 'MM:SS' / 'HH:MM:SS' strings."""
    if isinstance(x, (int, float)):
        return float(x)
    sec = 0.0
    for part in str(x).strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


def spans_of(ev: dict) -> List[Tuple[float, float]]:
    """question_evidence uses 'time_spans' (list) in most records and 'time_span' (dict) in a few."""
    sp = ev.get("time_spans") or ([ev["time_span"]] if ev.get("time_span") else [])
    return [(to_sec(s["start_time"]), to_sec(s["end_time"])) for s in sp]


def evidence_rows(qa: List[dict], skill: str = SKILL) -> List[dict]:
    """One row per answer-evidence span of every answerable question of `skill`.
    q_cut = question time if the evidence is in the question's own video (frames after it must not count)."""
    rows = []
    for r in qa:
        if r["metadata"]["skill"] != skill or not r["is_answerable"]:
            continue
        prim = r["metadata"]["primary_video_id"]
        qs = spans_of(r["question_evidence"])
        q_time = min(s for s, _ in qs) if qs else float("inf")
        for e in r["answer_evidence"]["evidence_list"]:
            for s, en in spans_of(e):
                rows.append(dict(qid=r["question_id"], person=r["subject"], video_id=e["video_id"],
                                 s=s, e=en, room=e.get("room", ""),
                                 q_cut=(q_time if e["video_id"] == prim else float("inf")),
                                 question=r["question"]))
    return rows


def video_paths() -> Dict[str, str]:
    """video_id -> path of its mp4 inside the dataset repo."""
    from huggingface_hub import list_repo_files
    return {os.path.basename(f)[:-4]: f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".mp4")}
