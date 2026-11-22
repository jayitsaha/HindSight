"""Learned retriever: features, labels, training rows, leave-one-participant-out run and report."""
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from hindsight.curation import grounded as gr  # noqa: E402
from hindsight.curation import learned_retriever as lr  # noqa: E402

D = 16


def unit_rows(m):
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def make_world(n_people=3, n_q=4, n_frames=900, seed=0):
    """Each question's object is a different axis; its frames light up at the evidence span (weakly, so
    zero-shot is imperfect) and also at a decoy sighting with a stronger cosine but no scene change."""
    rng = np.random.default_rng(seed)
    videos, questions, texts, starts = {}, [], {}, {}
    k = 3
    for p in range(1, n_people + 1):
        for s in (1, 2):
            vid = f"Person_{p}_session_{s}_x"
            videos[vid] = {"t": np.arange(n_frames, dtype=float),
                           "emb": unit_rows(rng.normal(0, 0.05, (n_frames, D)) + np.eye(D)[0])}
            starts[vid] = 1.7e9 + p * 1e6 + s * 1e4
        for i in range(n_q):
            ev, decoy = 100 + 150 * i, 120 + 150 * i + 60
            emb = videos[f"Person_{p}_session_1_x"]["emb"]
            emb[ev:ev + 8] = unit_rows(emb[ev:ev + 8] + 0.6 * np.eye(D)[k] + 0.8 * np.eye(D)[1])   # scene change
            emb[decoy:decoy + 8] = unit_rows(emb[decoy:decoy + 8] + 0.9 * np.eye(D)[k])
            qtext = f"Where did I leave the object{k}?"
            for form in gr.query_texts(qtext).values():
                texts[form] = unit_rows(np.eye(D)[k][None])[0]
            questions.append(dict(qid=f"{p}-{i}", question=qtext, primary=f"Person_{p}_session_2_x", q_time=500.0,
                                  answerable=True, spans=[(f"Person_{p}_session_1_x", ev, ev + 7)],
                                  evidence_videos=[f"Person_{p}_session_1_x"]))
            k = 3 + (k - 2) % (D - 3)
    questions.append(dict(qid="u", question="Where did I leave the object3?", primary="Person_1_session_2_x",
                          q_time=500.0, answerable=False, spans=[], evidence_videos=[]))
    return videos, questions, texts, starts


def test_video_starts():
    qa = [{"metadata": {"primary_video_id": "A", "primary_video_start_time": 1769988618},
           "question_evidence": {"video_id": "A", "start_time": 1769988618,
                                 "time_spans": [{"video_id": "A", "video_start_time_unix": 1769988618}]},
           "answer_evidence": {"evidence_list": [{"video_id": "B", "start_time": 1769973961}]}}]
    assert lr.video_starts(qa) == {"A": 1769988618.0, "B": 1769973961.0}


def test_features_labels_and_training_rows():
    videos, qs, texts, starts = make_world(n_people=1, n_q=1)
    q = qs[0]
    cands = gr.candidates(q, videos, gr.sessions_for(q, videos, "evidence"), "all")
    qemb = {k: texts[t] for k, t in gr.query_texts(q["question"]).items()}
    X, prod, y, vids, ts = lr.question_features(q, cands, qemb, starts, {"Person_1": min(starts.values())})
    assert X.shape == (900, len(lr.BASE_FEATURES)) and prod.shape == (900, D)
    assert y.sum() == 12                                               # span 100..107 +- 2 s
    assert X[:, lr.BASE_FEATURES.index("back_1")].all()                 # evidence session is one back
    rows = lr.training_rows((X, prod, y, vids, ts), n_hard=20, n_random=30, rng=np.random.default_rng(0))
    assert set(np.where(y == 1)[0]) <= set(rows) and len(rows) == int(y.sum()) + 20 + 30
    hard = [r for r in rows if y[r] == 0][:20]
    assert all(np.min(np.abs(ts[r] - ts[y == 1])) > 10 for r in hard)   # hard negatives stay away from evidence


def test_evidence_rank_pct():
    assert lr.evidence_rank_pct(np.array([0.9, 0.1, 0.5]), np.array([1, 0, 0])) == 0.0
    assert lr.evidence_rank_pct(np.array([0.9, 0.1, 0.5]), np.array([0, 1, 0])) == 100.0
    assert lr.evidence_rank_pct(np.array([0.9, 0.1]), np.array([0, 0])) is None


def test_run_lopo_and_report(tmp_path):
    videos, qs, texts, starts = make_world()
    rows, fm, info, windows = lr.run(qs, videos, texts, starts, xs=(1, 2), ws=(15,), product_dims=4,
                                     n_hard=30, n_random=60, log=lambda s: None)
    get = lambda m, x: next(r["question_recall"] for r in rows if r["method"] == m and r["x"] == x)
    assert info["folds"] == 3 and info["answerable"] == 12
    assert get("logreg", 1) >= get("random", 1) and get("hgb", 1) >= get("random", 1)
    assert get("logreg", 2) >= get("zeroshot", 1)
    assert set(fm) == {"zeroshot", "logreg", "hgb"} and 0 <= fm["logreg"]["auc"] <= 1
    assert len(windows["logreg"]) == len(qs) and len(windows["logreg"][0]["windows"]) == 2
    lr.write_report(rows, fm, info, windows, str(tmp_path))
    text = (tmp_path / "RETRIEVER.md").read_text()
    assert "Learned: logistic regression" in text and "Frame ranking" in text
    assert json.loads((tmp_path / "windows_hgb.jsonl").read_text().splitlines()[0])["qid"]
