"""Question-grounded retrieval (phase 1): object extraction, anchors, windows, recall, report."""
import json
import os

import numpy as np

from hindsight.curation import grounded as gr

D = 16


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def fake_encode(texts):
    """Texts mentioning 'car' point along axis 0, everything else along axis 1."""
    return np.stack([unit(np.eye(D)[0] if "car" in t else np.eye(D)[1]) for t in texts])


def make_video(n, car_at=(), seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 0.05, (n, D)) + np.eye(D)[2]
    for a, b in car_at:
        emb[a:b] += 3 * np.eye(D)[0]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return {"t": np.arange(n, dtype=float), "emb": emb.astype(np.float32)}


def question(qid="1", text="Where did I leave the yellow toy car?", primary="Person_1_session_3_x", q_time=500.0,
             spans=(("Person_1_session_2_x", 300, 310),), answerable=True):
    spans = list(spans) if answerable else []
    return dict(qid=qid, question=text, primary=primary, q_time=q_time, answerable=answerable, spans=spans,
                evidence_videos=sorted({v for v, _, _ in spans}, key=gr.session_number))


def test_object_phrase():
    assert gr.object_phrase("I'm cleaning up. Where did I leave that yellow toy car?") == "yellow toy car"
    assert gr.object_phrase("Where did Person D leave his smartphone just before we paused?") == "smartphone"
    assert gr.object_phrase("Do you know where I left the green watering can?") == "green watering can"
    assert gr.object_phrase("Where are the eggs?") == "eggs"
    assert gr.object_phrase("I want to use the black spatula again. Where did I leave it?") == "black spatula"
    assert gr.object_phrase("Where did I set down that plate?") == "plate"
    q = gr.query_texts("Where did I leave my sunglasses?")
    assert q["object"] == "sunglasses" and q["prompt"] == "a photo of sunglasses"
    assert gr.query_texts("What did I say?")["prompt"] == "What did I say?"      # falls back to the question


def test_anchors_respect_gap_and_question_time():
    videos = {"Person_1_session_3_x": make_video(1000, car_at=[(100, 110), (600, 700)])}
    q = question(primary="Person_1_session_3_x", q_time=500.0, spans=(("Person_1_session_3_x", 100, 110),))
    c = gr.candidates(q, videos, ["Person_1_session_3_x"], "all")
    assert c[0][1].max() <= 500                                                  # nothing after the question
    qe = fake_encode(["car"])[0]
    a = gr.select_anchors([(v, t, gr.smooth(e @ qe, 5)) for v, t, e in c], 3, min_gap=60)
    assert len(a) == 3 and 95 <= a[0][1] <= 115
    ts = sorted(t for _, t, _ in a)
    assert all(b - a_ >= 60 for a_, b in zip(ts, ts[1:]))
    wins = gr.windows_of(a, q, 15)
    assert all(hi <= 500 for _, _, hi in wins)
    assert gr.span_hits(q, wins, 2.0).all()
    assert gr.first_hit_rank(q, a, 15, 2.0) == 1


def test_window_hit_edges():
    q = question(spans=(("Person_1_session_2_x", 300, 310),))
    assert gr.span_hits(q, [("Person_1_session_2_x", 250, 299)], 2.0).all()     # within tolerance
    assert not gr.span_hits(q, [("Person_1_session_2_x", 250, 296)], 2.0).any()
    assert not gr.span_hits(q, [("Person_1_session_1_x", 290, 320)], 2.0).any()  # other session


def test_sessions_modes():
    videos = {v: make_video(10) for v in ["Person_1_session_1_x", "Person_1_session_2_x", "Person_1_session_3_x",
                                          "Person_1_session_4_x", "Person_2_session_1_x"]}
    q = question()
    assert gr.sessions_for(q, videos, "evidence") == ["Person_1_session_2_x"]
    assert gr.sessions_for(q, videos, "prior") == ["Person_1_session_1_x", "Person_1_session_2_x", "Person_1_session_3_x"]
    assert gr.sessions_for(question(answerable=False), videos, "evidence") == ["Person_1_session_3_x"]


def test_evaluate_and_report(tmp_path):
    videos = {"Person_1_session_2_x": make_video(2000, car_at=[(300, 310)], seed=1),
              "Person_1_session_3_x": make_video(1000, seed=2)}
    qs = [question(qid="1"), question(qid="2", text="Where are the plates?", answerable=False)]
    rows, windows, info = gr.evaluate(qs, videos, fake_encode, xs=(1, 4), ws=(15,), pools=("all", "uniform100"),
                                      session_modes=("evidence", "prior"), n_random=3)
    get = lambda **k: next(r for r in rows if all(r[a] == b for a, b in k.items()))
    assert get(sessions="evidence", pool="all", query="prompt", x=1)["question_recall"] == 100.0
    assert get(sessions="evidence", pool="all", query="random", x=1)["question_recall"] < 100.0
    assert info["answerable"] == 1 and info["object_extracted"] == 2
    assert len(windows) == 2 and windows[0]["first_hit_rank"] == 1 and len(windows[0]["windows"]) == 4
    gr.write_report(rows, windows, info, str(tmp_path))
    assert "Recall@X" in (tmp_path / "RECALL.md").read_text()
    assert json.loads((tmp_path / "windows.jsonl").read_text().splitlines()[0])["qid"] == "1"


def test_load_cache(tmp_path):
    cache, out = tmp_path / "cache", tmp_path / "out"
    os.makedirs(cache); os.makedirs(out / "Person_1_session_2_x")
    np.savez_compressed(cache / "Person_1_session_2_x_abcd1234.npz", emb=np.ones((3, D)), indices=np.array([60, 0, 30]))
    (out / "Person_1_session_2_x" / "kmeans_K100.json").write_text(json.dumps({"source": {"fps": 30.0}}))
    v = gr.load_cache(str(cache), str(out))["Person_1_session_2_x"]
    assert v["t"].tolist() == [0.0, 1.0, 2.0] and np.allclose(np.linalg.norm(v["emb"], axis=1), 1)


def test_text_cache_skips_encoding(tmp_path):
    """Texts already in the cache are returned without loading open_clip."""
    f = tmp_path / "text_emb.npz"
    np.savez_compressed(f, texts=np.array(["a", "b"]), emb=np.eye(2, dtype=np.float32))
    e = gr.encode_texts_open_clip(["b", "a", "b"], "ViT-B-32", "x", cache_file=str(f))
    assert e.tolist() == [[0, 1], [1, 0], [0, 1]]
