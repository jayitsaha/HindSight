"""The calibrated assistant: train retrieval, calibrate the refusal gate, answer."""

from __future__ import annotations

import numpy as np

from ..retrieval import ObjectRetrieval
from ..timeline.models import Answer
from .answer import build_answer, combine_confidence
from .refusal import RefusalGate
from .retrieve import Proposal, classify, extract_target, retrieve_action, retrieve_object


class Assistant:
    def __init__(self, scorer_name: str = "logreg", target_acc: float = 0.80,
                 seed: int = 0, enable_vlm: bool = False):
        self.retrieval = ObjectRetrieval(scorer_name)
        self.gate = RefusalGate(target_acc)
        self.seed = seed
        self.enable_vlm = enable_vlm

    def fit(self, video) -> "Assistant":
        self.retrieval.fit(video, self.seed)
        conf, correct = self._calibration_set(video)
        self.gate.fit(conf, correct)
        return self

    def answer(self, question: str, t: float, video=None, store=None) -> Answer:
        kind = classify(question)
        target = extract_target(question, kind)
        if kind == "object":
            prop = self._object_proposal(question, target, t, video, store)
        else:
            segments = video.segments if video is not None else (store.actions() if store else [])
            prop = retrieve_action(segments, t, target)
        return build_answer(question, t, prop, self.gate, self._maybe_vlm(question, prop, store))

    # --- calibration ------------------------------------------------------
    def _calibration_set(self, video):
        conf, correct = [], []
        for q in video.queries:
            prop = self._query_proposal(video, q)
            conf.append(combine_confidence(prop.conf_base))
            correct.append(self._is_correct(q, prop))
        return np.array(conf), np.array(correct, dtype=bool)

    def _query_proposal(self, video, q) -> Proposal:
        if q.kind == "object":
            qv = video.query_emb[q.query_emb_idx]
            return retrieve_object(self.retrieval, qv, video.dets, video.det_emb,
                                   q.t_asked, q.target_label, q.text)
        return retrieve_action(video.segments, q.t_asked, q.target_label)

    @staticmethod
    def _is_correct(q, prop: Proposal) -> bool:
        # Committing is right iff the top match is the asked object, or (action)
        # the proposed window overlaps the true prior occurrence.
        if q.kind == "object":
            return prop.label == q.target_label
        truth = getattr(q, "answer_window", None)
        if not prop.found or prop.window is None or truth is None:
            return False
        return max(prop.window[0], truth[0]) < min(prop.window[1], truth[1])

    def judge_queries(self, video) -> list[dict]:
        """Per-query judgement for selective-classification metrics, on any video.

        `correct` is the real answer correctness measured pre-refusal (does
        committing give the right answer?), not "answerable and not refused"."""
        out = []
        for q in video.queries:
            prop = self._query_proposal(video, q)
            conf = combine_confidence(prop.conf_base)
            out.append({
                "conf": conf,
                "correct": self._is_correct(q, prop),
                "refused": self.gate.decide(conf),
                "answerable": bool(q.answerable),
            })
        return out

    # --- live/synthetic answering ----------------------------------------
    def _object_proposal(self, question, target, t, video, store) -> Proposal:
        if video is not None:
            qv = self._planted_query_vec(video, question, t)
            if qv is None:
                # free-text question with no planted query: build the query in the
                # same synthetic space as the centroid of this object's seen crops.
                qv = self._crop_centroid(video, target, t)
            if qv is None:
                return Proposal("object", target, False, 0.0, "", None)
            return retrieve_object(self.retrieval, qv, video.dets, video.det_emb,
                                   t, target, question)
        if store is not None:
            from ..providers import get_embedder
            qv = get_embedder().embed_texts([question])[0]
            dets, det_emb = store.embeddings()
            return retrieve_object(self.retrieval, qv, dets, det_emb, t, target, question)
        return Proposal("object", target, False, 0.0, "", None)

    @staticmethod
    def _planted_query_vec(video, question, t):
        for q in video.queries:
            if q.kind == "object" and q.query_emb_idx is not None \
                    and q.text == question and abs(q.t_asked - t) < 1e-3:
                return video.query_emb[q.query_emb_idx]
        return None

    @staticmethod
    def _crop_centroid(video, target, t):
        idx = [d.emb_idx for d in video.dets if d.label == target and d.t <= t]
        if not idx:
            return None
        v = video.det_emb[idx].mean(axis=0)
        return (v / max(float(np.linalg.norm(v)), 1e-8)).astype(np.float32)

    def _maybe_vlm(self, question, prop, store):
        if not self.enable_vlm or store is None or not prop.found or not prop.support_frames:
            return None
        try:
            from PIL import Image

            from ..providers import get_vlm
            frames = {f.id: f for f in store.frames()}
            imgs = [Image.open(frames[i].path) for i in prop.support_frames if i in frames]
            if not imgs:
                return None
            return get_vlm().verify(imgs, question, prop.text)
        except Exception:
            return None  # VLM is a bonus; never let it break an answer
