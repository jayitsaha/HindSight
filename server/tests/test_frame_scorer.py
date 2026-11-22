import numpy as np

from hindsight.curation.frame_scorer import (
    EmbeddingStore,
    FrameScorer,
    MemoryQuestion,
    SessionFrames,
    build_rows,
    evaluate_lopo,
    pair_features,
)

D = 16


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1.0, 1.0], size=D)
    frames = {}
    queries = {}
    questions = []
    qid = 0
    for pi in range(6):
        subject = f"P{pi}"
        for _ in range(3):
            q = rng.standard_normal(D)
            q /= np.linalg.norm(q)
            key = f"q{qid}"
            queries[key] = q
            vids = [f"{key}_s0", f"{key}_s1"]
            ev_vid = vids[int(rng.integers(2))]
            span = (10.0, 14.0)
            for vid in vids:
                n = 30
                times = np.arange(n, dtype=float)
                emb = rng.standard_normal((n, D))
                if vid == ev_vid:
                    mask = (times >= span[0]) & (times <= span[1])
                    emb[mask] = sign * q + 0.15 * rng.standard_normal((int(mask.sum()), D))
                frames[vid] = SessionFrames(emb, times)
            questions.append(MemoryQuestion(
                qid=key, subject=subject, video_ids=vids, primary=vids[0],
                cutoff=25.0, evidence=[(ev_vid, span[0], span[1])], is_answerable=True))
            qid += 1
    return EmbeddingStore(frames, queries), questions


def test_pair_features_shape():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((7, D))
    q = rng.standard_normal(D)
    X = pair_features(f, q)
    assert X.shape == (7, 4 * D)


def test_cutoff_filters_primary():
    store, questions = _synthetic()
    X, y, groups = build_rows(store, questions)
    assert X.shape[1] == 4 * D
    assert len(y) == len(groups) == X.shape[0]
    assert set(groups.tolist()) == {f"P{i}" for i in range(6)}
    assert 0 < y.mean() < 0.5


def test_lopo_beats_cosine_and_base_rate():
    store, questions = _synthetic()
    res = evaluate_lopo(store, questions, model="logreg")
    scorer = res["mean"]["scorer"]
    cosine = res["mean"]["cosine"]
    base = res["mean"]["base_rate"]
    assert scorer > base + 0.3
    assert scorer > cosine + 0.3
    assert len(res["per_participant"]) == 6


def test_score_frames_uses_only_frame_and_query():
    store, questions = _synthetic()
    X, y, _ = build_rows(store, questions)
    scorer = FrameScorer("logreg").fit(X, y)
    rng = np.random.default_rng(99)
    sign = np.random.default_rng(0).choice([-1.0, 1.0], size=D)
    q = rng.standard_normal(D)
    q /= np.linalg.norm(q)
    pos = sign * q + 0.15 * rng.standard_normal((5, D))
    neg = rng.standard_normal((20, D))
    frame_embs = np.vstack([pos, neg])
    probs = scorer.score_frames(frame_embs, q)
    assert probs.shape == (25,)
    assert np.all((probs >= 0) & (probs <= 1))
    assert probs[:5].mean() > probs[5:].mean() + 0.2


def test_save_load_round_trip(tmp_path):
    store, questions = _synthetic()
    X, y, _ = build_rows(store, questions)
    scorer = FrameScorer("logreg").fit(X, y)
    path = str(tmp_path / "scorer.joblib")
    scorer.save(path)
    loaded = FrameScorer.load(path)
    a = scorer.predict_proba(X)
    b = loaded.predict_proba(X)
    assert np.allclose(a, b)
    assert loaded.model_name == "logreg"
