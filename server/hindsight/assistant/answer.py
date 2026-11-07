"""Turn a proposal + the refusal gate into a spoken Answer.

Confidence is the retrieval signal, optionally reconciled with a VLM verdict.
The VLM is a bonus check, not a dependency: if it's the offline HeuristicVLM or
the call fails, we fall back to the retrieval confidence alone.
"""

from __future__ import annotations

import numpy as np

from ..providers import VLMVerdict
from ..timeline.models import Answer
from .retrieve import Proposal
from .refusal import RefusalGate

_DONT_KNOW = "I don't know for sure."


def combine_confidence(base: float, vlm: VLMVerdict | None = None) -> float:
    c = float(np.clip(base, 0.0, 1.0))
    if vlm is not None:
        a = float(np.clip(vlm.confidence, 0.0, 1.0))
        c = 0.5 * c + 0.5 * a if vlm.agrees else 0.5 * c * (1.0 - a)
    return float(np.clip(c, 0.0, 1.0))


def build_answer(question: str, t: float, prop: Proposal, gate: RefusalGate,
                 vlm: VLMVerdict | None = None) -> Answer:
    conf = combine_confidence(prop.conf_base, vlm)
    if not prop.found or gate.decide(conf):
        return Answer(question=question, t_asked=t, refused=True, confidence=conf,
                      text=_DONT_KNOW, window=None, support_frames=[])
    return Answer(question=question, t_asked=t, refused=False, confidence=conf,
                  text=prop.text, window=prop.window, support_frames=prop.support_frames)
