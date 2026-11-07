"""Drives the FastAPI app end to end with the TestClient: health, the no-camera
demo path (populated timeline + alerts), a calibrated ask, and a JSON frame post."""

import threading

import numpy as np
from fastapi.testclient import TestClient

from hindsight.api.app import app
from hindsight.timeline.models import Alert
from hindsight.timeline.store import TimelineStore


def test_api_contract():
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok"
        assert h["provider"]["active"] in ("local", "nim")
        assert h["provider"]["embedder"]

        # no-camera demo: load + run the whole pipeline over demo_01
        vid = c.post("/demo/demo_01").json()["video_id"]
        assert vid

        tl = c.get(f"/session/{vid}/timeline").json()
        assert tl["video_id"] == vid
        assert tl["detections"] > 0
        assert tl["frames"] > 0
        assert len(tl["alerts"]) >= 1
        assert {a["kind"] for a in tl["alerts"]} <= {"misplacement", "repeat"}

        al = c.get(f"/session/{vid}/alerts").json()
        assert al and all("message" in a for a in al)

        # ask an object question against the demo session
        ans = c.post(f"/session/{vid}/ask", json={"question": "where is my mug?", "t": 900.0}).json()
        for key in ("question", "t_asked", "refused", "confidence", "text"):
            assert key in ans
        assert isinstance(ans["refused"], bool)
        assert 0.0 <= ans["confidence"] <= 1.0

        # a JSON frame post on a fresh live session returns detections + alerts
        live = c.post("/session", json={}).json()["video_id"]
        body = {"t": 1.0, "detections": [
            {"label": "wallet", "xyxy": [10, 10, 50, 50], "score": 0.9, "embedding": [0.1] * 64}]}
        fr = c.post(f"/session/{live}/frame", json=body).json()
        assert "detections" in fr and "alerts" in fr
        assert len(fr["detections"]) == 1
        assert fr["detections"][0]["label"] == "wallet"


def test_unknown_session_404():
    with TestClient(app) as c:
        assert c.get("/session/nope/timeline").status_code == 404
        assert c.post("/demo/does_not_exist").status_code == 404


def test_live_repeat_alert():
    """A live session that posts the same action segment twice fires a repeat alert."""
    with TestClient(app) as c:
        vid = c.post("/session", json={}).json()["video_id"]
        emb = [0.1] * 64

        def seg_frame(ts):
            return {"t": ts, "segment": {"label_id": 0, "embedding": emb,
                                         "t_start": ts, "t_end": ts + 5.0, "score": 1.0}}

        first = c.post(f"/session/{vid}/frame", json=seg_frame(0.0)).json()
        assert first["alerts"] == []  # nothing to repeat yet

        second = c.post(f"/session/{vid}/frame", json=seg_frame(30.0)).json()
        assert any(a["kind"] == "repeat" for a in second["alerts"])

        stored = c.get(f"/session/{vid}/alerts").json()
        assert any(a["kind"] == "repeat" for a in stored)


def test_frame_validation():
    with TestClient(app) as c:
        vid = c.post("/session", json={}).json()["video_id"]
        # multipart without the required 't' field -> 422, not a 500
        r = c.post(f"/session/{vid}/frame", files={"image": ("f.txt", b"x")})
        assert r.status_code == 422
        # unknown session is a 404 on the frame route too
        assert c.post("/session/nope/frame", json={"t": 1.0}).status_code == 404


def test_store_concurrent_read_write():
    """Readers and writers hammer one check_same_thread=False store without racing."""
    store = TimelineStore()
    errors: list[Exception] = []

    def writer():
        try:
            for i in range(150):
                fid = store.add_frame("v", float(i), "", 10, 10)
                store.add_detection(fid, float(i), "mug", 0.9, (0, 0, 1, 1),
                                    emb=np.full(64, 0.1, np.float32))
                store.add_alert(Alert(kind="repeat", t=float(i), severity=0.5, message="x"))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(150):
                store.frames("v"); store.detections(); store.embeddings()
                store.alerts(); store.putdowns(); store.actions(); store.snapshot("v")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.close()
    assert not errors, errors[:3]


def test_store_embeddings_mixed_width():
    """Mixed-width and NULL emb rows don't crash embeddings(); output stays aligned."""
    store = TimelineStore()
    fid = store.add_frame("v", 0.0, "", 10, 10)
    store.add_detection(fid, 0.0, "a", 1.0, (0, 0, 1, 1), emb=np.ones(64, np.float32))
    store.add_detection(fid, 0.0, "b", 1.0, (0, 0, 1, 1), emb=None)  # NULL
    store.add_detection(fid, 0.0, "c", 1.0, (0, 0, 1, 1), emb=np.ones(32, np.float32))  # ragged
    store.add_detection(fid, 0.0, "d", 1.0, (0, 0, 1, 1), emb=np.ones(64, np.float32))
    dets, arr = store.embeddings()
    assert arr.shape == (2, 64)  # only the two consistent-width rows survive
    assert len(dets) == arr.shape[0]
    assert [d.label for d in dets] == ["a", "d"]
    store.close()
