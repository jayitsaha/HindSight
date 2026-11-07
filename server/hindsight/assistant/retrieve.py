"""Route a question to a proposal; refusal.py decides whether we commit."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..timeline.models import RetrievalHit

_OBJECT_CUE = re.compile(r"\bwhere(?:'s|s| is| are| did| can| could)?\b")
_ACTION_CUE = re.compile(r"\b(when did|how many times|did i|have i|last time|how often)\b")


@dataclass
class Proposal:
    kind: str                                  # "object" | "action"
    target: str                                # asked noun or verb
    found: bool
    conf_base: float                           # retrieval-only confidence in [0, 1]
    text: str
    window: tuple[float, float] | None
    support_frames: list[int] = field(default_factory=list)
    label: str | None = None                   # label of the top-ranked detection
    hit: RetrievalHit | None = None


def classify(question: str) -> str:
    q = question.lower()
    if _OBJECT_CUE.search(q):
        return "object"
    if _ACTION_CUE.search(q):
        return "action"
    return "object"


def extract_target(question: str, kind: str) -> str:
    q = question.lower()
    if kind == "object":
        m = re.search(r"\bmy\s+([a-z][a-z ]*?)\s*\?*$", q)
        if m:
            return m.group(1).strip()
        m = re.search(r"where(?:'s| is| are| did)?\s+(?:the\s+|my\s+)?([a-z ]+?)\s*\?*$", q)
        return m.group(1).strip() if m else ""
    m = re.search(r"\blast\s+([a-z_ ]+?)\s*\?*$", q)
    if m:
        return m.group(1).strip()
    m = re.search(r"\bdid i\s+([a-z_ ]+?)\s*\?*$", q)
    return m.group(1).strip() if m else ""


def retrieve_object(retrieval, query_emb, dets, det_emb, t_asked: float,
                    target: str, query_text: str = "") -> Proposal:
    cand = np.array([i for i, d in enumerate(dets) if d.t <= t_asked])
    if cand.size == 0:
        return Proposal("object", target, False, 0.0, "", None)

    scores = retrieval.score_candidates(query_emb, det_emb[cand])
    order = np.argsort(-scores)
    top1 = float(np.clip(scores[order[0]], 0.0, 1.0))
    top_label = dets[int(cand[order[0]])].label
    # the top match score says "how sure the best crop matches the query"; if the
    # best-matching detection isn't even the asked object, halve that: strong
    # evidence the thing isn't in view at all (the unanswerable case).
    match = target == "" or top_label == target
    conf_base = top1 if match else top1 * 0.5

    hit = retrieval.locate(query_emb, dets, det_emb, t_asked, query_text=query_text)
    support = [hit.frame_id] if hit.frame_id >= 0 else []
    text = f"Your {target or top_label} was last here around {int(hit.t)}s in."
    return Proposal("object", target or top_label, hit.found, conf_base, text,
                    (hit.t - 1.0, hit.t), support, top_label, hit)


def retrieve_action(segments, t_asked: float, target: str) -> Proposal:
    prior = [s for s in segments if s.t_end <= t_asked and s.label == target]
    if not prior:
        return Proposal("action", target, False, 0.2, "", None)
    last = max(prior, key=lambda s: s.t_end)
    text = f"You last {target} around {int(last.t_start)}-{int(last.t_end)}s in."
    return Proposal("action", target, True, 0.9, text, (last.t_start, last.t_end))
