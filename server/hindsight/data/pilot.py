"""
Pilot subset selection and video resolution for SuperMemory-VQA.

The full dataset is ~228 GB of session videos (2-4 GB each), so a pilot is chosen
to touch as few distinct videos as possible, with the natural answerable /
unanswerable mix preserved.
"""

from typing import Dict, Iterable, List, Optional, Set
import random

from hindsight.data.schema import QuestionRecord

from hindsight.data.supermemory import REPO_ID as HF_REPO  # one source of truth for the repo id


def clip_video_ids(q: QuestionRecord) -> Set[str]:
    return {c["video_id"] for c in (q.evidence_clips or [])}


def hf_video_path(video_id: str) -> str:
    """Repo path of a session video, e.g. data/video/Person_1/Person_1_session_8_..._1264.mp4."""
    person = "_".join(video_id.split("_")[:2])
    return f"data/video/{person}/{video_id}.mp4"


def select_pilot(
    questions: List[QuestionRecord],
    n: int = 40,
    max_videos: int = 6,
    seed: int = 0,
    available_videos: Optional[Iterable[str]] = None,
) -> List[QuestionRecord]:
    """
    Picks up to `n` questions that only need clips from at most `max_videos` videos.
    Videos are chosen greedily by how many questions they unlock; the sample keeps
    the answerable/unanswerable ratio of the candidates. `available_videos` drops
    questions whose evidence video is not in the repo (e.g. Person_1 session 2).
    """
    usable = [q for q in questions if q.evidence_clips]
    if available_videos is not None:
        avail = set(available_videos)
        usable = [q for q in usable if clip_video_ids(q) <= avail]
    all_videos = sorted({v for q in usable for v in clip_video_ids(q)})
    chosen: List[str] = []
    for _ in range(max_videos):
        def gain(v: str) -> int:
            return sum(1 for q in usable if clip_video_ids(q) <= set(chosen) | {v})
        best = max((v for v in all_videos if v not in chosen), key=gain, default=None)
        if best is None:
            break
        chosen.append(best)
    candidates = [q for q in usable if clip_video_ids(q) <= set(chosen)]

    rng = random.Random(seed)
    unans = [q for q in candidates if q.is_answerable is False]
    ans = [q for q in candidates if q.is_answerable is not False]
    rng.shuffle(unans)
    rng.shuffle(ans)
    n = min(n, len(candidates))
    n_un = min(len(unans), round(n * len(unans) / len(candidates))) if candidates else 0
    picked = ans[: n - n_un] + unans[:n_un]
    picked.sort(key=lambda q: str(q.question_id))
    return picked
