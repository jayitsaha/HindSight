from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from hindsight.data.supermemory import SKILL, load_qa, spans_of, video_paths


def _l2(a: np.ndarray, axis: int) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=axis, keepdims=True) + 1e-12)


def pair_features(frame_emb: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    f = _l2(np.asarray(frame_emb, dtype=np.float64), axis=1)
    q = np.asarray(query_emb, dtype=np.float64).reshape(-1)
    q = q / (np.linalg.norm(q) + 1e-12)
    qt = np.tile(q, (f.shape[0], 1))
    return np.concatenate([f, qt, f * qt, np.abs(f - qt)], axis=1)


def cosine_from_features(X: np.ndarray) -> np.ndarray:
    d = X.shape[1] // 4
    return X[:, 2 * d:3 * d].sum(axis=1)


def _make_estimator(name: str, class_weight=None):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0, class_weight=class_weight)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=0)
    if name == "gbm":
        # class_weight on HistGradientBoosting is ~20x slower (hours at 800k rows) for no AP gain;
        # its loss already handles imbalance, so we keep the fast default path.
        return HistGradientBoostingClassifier(random_state=0)
    raise ValueError(f"unknown model {name!r}; options: logreg|mlp|gbm")


class FrameScorer:
    def __init__(self, model: str = "logreg", class_weight=None):
        self.model_name = model
        self.class_weight = class_weight
        self.pipeline = Pipeline([("scaler", StandardScaler()),
                                  ("model", _make_estimator(model, class_weight))])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FrameScorer":
        self.pipeline.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]

    def score_frames(self, frame_embs: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
        return self.predict_proba(pair_features(frame_embs, query_emb))

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model_name, "pipeline": self.pipeline}, path)

    @classmethod
    def load(cls, path: str) -> "FrameScorer":
        d = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model_name = d["model"]
        obj.pipeline = d["pipeline"]
        return obj


@dataclass
class SessionFrames:
    emb: np.ndarray
    times: np.ndarray


@dataclass
class MemoryQuestion:
    qid: str
    subject: str
    video_ids: List[str]
    primary: str
    cutoff: Optional[float]
    evidence: List[Tuple[str, float, float]]
    is_answerable: bool = True
    text: str = ""


class EmbeddingStore:
    def __init__(self, frames: Dict[str, SessionFrames], queries: Dict[str, np.ndarray]):
        self._frames = frames
        self._queries = queries

    def frames(self, vid: str) -> Optional[SessionFrames]:
        return self._frames.get(vid)

    def query(self, qid: str) -> np.ndarray:
        return self._queries[qid]


def _evidence_map(q: MemoryQuestion) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    for vid, s, e in q.evidence:
        out.setdefault(vid, []).append((float(s), float(e)))
    return out


def assert_unique_subjects(questions: List[MemoryQuestion]) -> None:
    owners: Dict[str, set] = {}
    for q in questions:
        for vid in q.video_ids:
            owners.setdefault(vid, set()).add(str(q.subject))
    shared = {vid: sorted(subs) for vid, subs in owners.items() if len(subs) > 1}
    if shared:
        raise ValueError(f"video_ids shared across subjects (LOPO leak): {shared}")


def build_rows(sessions_emb: EmbeddingStore, questions: List[MemoryQuestion],
               max_frames_per_session: int = 0):
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    for q in questions:
        if not q.is_answerable:
            continue
        query_emb = sessions_emb.query(q.qid)
        ev = _evidence_map(q)
        for vid in q.video_ids:
            sf = sessions_emb.frames(vid)
            if sf is None:
                continue
            emb = np.asarray(sf.emb, dtype=np.float64)
            times = np.asarray(sf.times, dtype=np.float64)
            if vid == q.primary and q.cutoff is not None:
                keep = times < float(q.cutoff)
                emb, times = emb[keep], times[keep]
            if emb.shape[0] == 0:
                continue
            if max_frames_per_session and emb.shape[0] > max_frames_per_session:
                sel = np.unique(np.linspace(0, emb.shape[0] - 1, max_frames_per_session).round().astype(int))
                emb, times = emb[sel], times[sel]
            spans = ev.get(vid, [])
            y = np.array([int(any(s <= t <= e for s, e in spans)) for t in times], dtype=int)
            feats.append(pair_features(emb, query_emb).astype(np.float32))
            labels.append(y)
            groups.append(np.full(emb.shape[0], str(q.subject), dtype=object))
    if not feats:
        raise ValueError("no rows built from the given questions")
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(groups)


def _eval_one(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cos: np.ndarray,
              model: str, class_weight=None) -> dict:
    participants = sorted(set(groups.tolist()))
    per: Dict[str, Dict[str, float]] = {}
    for p in participants:
        test = groups == p
        train = ~test
        yte = y[test]
        if yte.sum() == 0 or train.sum() == 0:
            continue
        scorer = FrameScorer(model, class_weight=class_weight).fit(X[train], y[train])
        probs = scorer.predict_proba(X[test])
        per[p] = {
            "scorer": float(average_precision_score(yte, probs)),
            "cosine": float(average_precision_score(yte, cos[test])),
            "base_rate": float(yte.mean()),
            "n": int(test.sum()),
            "pos": int(yte.sum()),
        }
    sc = [v["scorer"] for v in per.values()]
    co = [v["cosine"] for v in per.values()]
    mean = {
        "scorer": float(np.mean(sc)) if sc else 0.0,
        "cosine": float(np.mean(co)) if co else 0.0,
        "base_rate": float(np.mean([v["base_rate"] for v in per.values()])) if per else 0.0,
        "scorer_median": float(median(sc)) if sc else 0.0,
        "cosine_median": float(median(co)) if co else 0.0,
        "scorer_wins": int(sum(s > c for s, c in zip(sc, co))),
        "n_participants": len(sc),
    }
    return {
        "model": model,
        "class_weight": class_weight if model == "logreg" else None,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "per_participant": per,
        "mean": mean,
    }


def evaluate_lopo(sessions_emb: EmbeddingStore, questions: List[MemoryQuestion],
                  model: str = "logreg", max_frames_per_session: int = 0, class_weight=None):
    X, y, groups = build_rows(sessions_emb, questions, max_frames_per_session)
    cos = cosine_from_features(X)
    return _eval_one(X, y, groups, cos, model, class_weight)


def evaluate_models(sessions_emb: EmbeddingStore, questions: List[MemoryQuestion],
                    models: List[str], max_frames_per_session: int = 0, class_weight=None) -> List[dict]:
    X, y, groups = build_rows(sessions_emb, questions, max_frames_per_session)
    cos = cosine_from_features(X)
    return [_eval_one(X, y, groups, cos, m, class_weight) for m in models]


def load_real_questions(qa_path: Optional[str] = None) -> List[MemoryQuestion]:
    qa = load_qa(qa_path)
    out: List[MemoryQuestion] = []
    for r in qa:
        meta = r.get("metadata") or {}
        if meta.get("skill") != SKILL or not r.get("is_answerable"):
            continue
        primary = meta.get("primary_video_id")
        q_spans = spans_of(r.get("question_evidence") or {})
        cutoff = min(s for s, _ in q_spans) if q_spans else None
        evidence: List[Tuple[str, float, float]] = []
        for e in (r.get("answer_evidence") or {}).get("evidence_list") or []:
            if e.get("video_id"):
                for s, en in spans_of(e):
                    evidence.append((e["video_id"], s, en))
        vids = r.get("video_ids")
        if not vids:
            vids = sorted({primary, *(v for v, _, _ in evidence)} - {None})
        out.append(MemoryQuestion(
            qid=str(r.get("question_id")),
            subject=str(r.get("subject")),
            video_ids=list(vids),
            primary=primary,
            cutoff=cutoff,
            evidence=evidence,
            is_answerable=True,
            text=r.get("question", ""),
        ))
    return out


def build_real_store(questions: List[MemoryQuestion], cache_dir: str, data_dir: str,
                     config: Optional[str] = None, embed_fps: float = 1.0,
                     delete_video: bool = False, allow: Optional[set] = None) -> EmbeddingStore:
    from huggingface_hub import hf_hub_download
    from hindsight.video.curation import FrameCurator, probe_video

    assert_unique_subjects(questions)
    cur = FrameCurator(config if (config and os.path.exists(config)) else None,
                       **{"kmeans.cache_dir": cache_dir})
    emb_cfg = cur.config["kmeans"]["embedder"]
    paths = video_paths()
    candidates = sorted({v for q in questions for v in q.video_ids})
    needed = [v for v in candidates if v in paths and (allow is None or v in allow)]
    missing = [v for v in candidates if v not in paths]
    if allow is not None:
        skipped = [v for v in candidates if v in paths and v not in allow]
        print(f"session allowlist: embedding {len(needed)} of {len(candidates)} candidate sessions "
              f"({len(skipped)} in-repo sessions outside the allowlist are skipped)")
    if missing:
        print(f"{len(missing)} of {len(candidates)} candidate video_ids have no path in "
              f"video_paths() and are skipped: {', '.join(missing)}")
    tag = hashlib.md5(json.dumps([emb_cfg, float(embed_fps)], sort_keys=True).encode()).hexdigest()[:10]
    reused = 0
    frames: Dict[str, SessionFrames] = {}
    for vid in needed:
        store_npz = os.path.join(cache_dir, f"store_{vid}_{tag}.npz") if cache_dir else None
        if store_npz and os.path.exists(store_npz):
            d = np.load(store_npz)
            frames[vid] = SessionFrames(np.asarray(d["emb"], dtype=np.float64), np.asarray(d["times"], dtype=np.float64))
            reused += 1
            continue
        mp4 = os.path.join(data_dir, paths[vid])
        downloaded = not os.path.exists(mp4)
        if downloaded:
            from hindsight.data.supermemory import REPO_ID
            mp4 = hf_hub_download(REPO_ID, paths[vid], repo_type="dataset", local_dir=data_dir)
        info = probe_video(mp4)
        idx = cur._sample_indices(info, float(embed_fps))
        emb = cur.embed_frames(info, idx, emb_cfg, cache_dir)
        times = idx / info.fps
        frames[vid] = SessionFrames(np.asarray(emb, dtype=np.float64), times)
        if store_npz:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(store_npz, emb=np.asarray(emb, dtype=np.float32), times=np.asarray(times, dtype=np.float64))
        if delete_video and downloaded and os.path.exists(mp4):
            os.remove(mp4)
    if reused:
        print(f"reused {reused} of {len(needed)} sessions from the vid-keyed store cache (no re-download/embed)")
    queries: Dict[str, np.ndarray] = {}
    for q in questions:
        queries[q.qid] = cur.embed_query([q.text], emb_cfg)[0]
    return EmbeddingStore(frames, queries)


def _print_table(res: dict) -> None:
    cw = res.get("class_weight")
    print(f"\n=== model={res['model']}  class_weight={cw}  n={res['n']}  positive_rate={res['positive_rate']:.4f} ===")
    print(f"{'participant':<16}{'n':>8}{'pos':>8}{'base':>10}{'cosine':>10}{'scorer':>10}")
    for p, v in sorted(res["per_participant"].items()):
        print(f"{p:<16}{v['n']:>8}{v['pos']:>8}{v['base_rate']:>10.4f}"
              f"{v['cosine']:>10.4f}{v['scorer']:>10.4f}")
    m = res["mean"]
    print(f"{'MEAN AP':<16}{'':>8}{'':>8}{m['base_rate']:>10.4f}"
          f"{m['cosine']:>10.4f}{m['scorer']:>10.4f}")
    print(f"{'MEDIAN AP':<16}{'':>8}{'':>8}{'':>10}"
          f"{m['cosine_median']:>10.4f}{m['scorer_median']:>10.4f}")
    print(f"scorer beats cosine on {m['scorer_wins']}/{m['n_participants']} participants")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Supervised evidence-relevance frame scorer (LOPO).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        sp = sub.add_parser(name)
        sp.add_argument("--qa", default=None)
        sp.add_argument("--cache", default="cache")
        sp.add_argument("--data", default="data")
        sp.add_argument("--config", default=None)
        sp.add_argument("--embed-fps", type=float, default=1.0)
        sp.add_argument("--model", default="logreg",
                        help="logreg|mlp|gbm, a comma list, or 'all' (eval trains them in one embedding pass)")
        sp.add_argument("--balanced", action="store_true",
                        help="class_weight='balanced' for logreg (gbm is ~20x slower with it and mlp has none, so both ignore it)")
        sp.add_argument("--max-questions", type=int, default=None)
        sp.add_argument("--delete-video", action="store_true",
                        help="delete each session mp4 right after embedding (peak disk stays ~one video)")
        sp.add_argument("--sessions", default=None,
                        help="restrict candidate sessions to the video_ids in this sessions.txt (col 2)")
        sp.add_argument("--max-frames-per-session", type=int, default=0,
                        help="uniformly subsample each session's candidate frames to this many (0 = all); bounds memory")
        if name == "train":
            sp.add_argument("--out", default="frame_scorer.joblib")
    a = ap.parse_args(argv)

    all_models = ["logreg", "gbm", "mlp"]
    models = all_models if a.model == "all" else [m.strip() for m in a.model.split(",") if m.strip()]
    bad = [m for m in models if m not in all_models]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; options: {all_models} or 'all'")
    class_weight = "balanced" if a.balanced else None

    allow = None
    if a.sessions:
        with open(a.sessions) as f:
            allow = {line.split("\t")[1].strip() for line in f if line.strip() and "\t" in line}
        if not allow:
            raise SystemExit(f"--sessions {a.sessions} has no '<path>\\t<video_id>' lines")

    questions = load_real_questions(a.qa)
    if a.max_questions:
        questions = questions[:a.max_questions]
    store = build_real_store(questions, a.cache, a.data, a.config, a.embed_fps, a.delete_video, allow)

    if a.cmd == "train":
        X, y, _ = build_rows(store, questions, a.max_frames_per_session)
        FrameScorer(models[0], class_weight=class_weight).fit(X, y).save(a.out)
        print(f"trained {models[0]} (class_weight={class_weight}) on {len(y)} frames "
              f"({y.mean():.4f} positive); saved {a.out}")
    else:
        results = evaluate_models(store, questions, models, a.max_frames_per_session, class_weight)
        for res in results:
            _print_table(res)
        if len(results) > 1:
            print(f"\n=== summary (mean / median AP; cosine baseline shared) ===")
            cm = results[0]["mean"]
            print(f"{'zero-shot cosine':<22}{cm['cosine']:>10.4f}{cm['cosine_median']:>10.4f}   (mean / median)")
            for res in results:
                m = res["mean"]
                print(f"{res['model']:<22}{m['scorer']:>10.4f}{m['scorer_median']:>10.4f}   "
                      f"beats cosine {m['scorer_wins']}/{m['n_participants']}")


if __name__ == "__main__":
    main()
