"""Frame curation (hindsight.video.curation) and SuperMemory helpers, on a tiny synthetic video.

No model downloads: KMeans runs with a toy colour-histogram embedder registered for the test.
"""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hindsight.curation.analyze import hits  # noqa: E402
from hindsight.data.supermemory import evidence_rows, spans_of, to_sec  # noqa: E402
from hindsight.video import curation  # noqa: E402
from hindsight.video.curation import FrameCurator, Selection, curation_method, probe_video  # noqa: E402

FPS, SECONDS, SCENES = 30, 12, 3          # 3 scenes of 4 s each, distinct colours


class ToyEmbedder:
    """Mean RGB of each frame: separates the synthetic scenes perfectly."""
    name = "toy"

    def __init__(self, **_):
        pass

    def embed(self, images):
        return np.stack([np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(0) + 1.0 for im in images])


@pytest.fixture(autouse=True)
def toy_backend(monkeypatch):
    monkeypatch.setitem(curation.EMBEDDERS, "toy", ToyEmbedder)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "scenes.mp4"
    colours = [(30, 30, 200), (30, 200, 30), (200, 30, 30)]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 64))
    for i in range(FPS * SECONDS):
        vw.write(np.full((64, 64, 3), colours[i // (FPS * SECONDS // SCENES)], np.uint8))
    vw.release()
    return str(path)


def _frames_in(path):
    return int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))


def test_uniform_rate_and_manifest(video, tmp_path):
    out = str(tmp_path / "u.mp4")
    res = FrameCurator().curate(video, out, method="uniform", fps=3)
    assert len(res.indices) == 3 * SECONDS
    assert np.all(np.diff(res.indices) == FPS // 3)
    assert _frames_in(out) == len(res.indices)
    man = json.load(open(res.manifest))
    assert man["output"]["fps"] == 3.0
    assert [f["src_index"] for f in man["frames"]] == res.indices.tolist()


def test_time_window(video, tmp_path):
    cur = FrameCurator(**{"io.start_s": 4, "io.end_s": 8})
    res = cur.curate(video, str(tmp_path / "w.mp4"), method="uniform", fps=1)
    assert res.timestamps.min() >= 4 and res.timestamps.max() < 8


def test_kmeans_one_frame_per_scene(video, tmp_path):
    cur = FrameCurator({"kmeans": {"embedder": {"backend": "toy"}, "embed_fps": 2.0}})
    res = cur.curate(video, str(tmp_path / "k.mp4"), method="kmeans", k=SCENES)
    scene_of = (res.timestamps // (SECONDS / SCENES)).astype(int)
    assert sorted(scene_of.tolist()) == list(range(SCENES))       # one keyframe per scene
    assert np.all(np.diff(res.indices) > 0)                       # stitched in time order
    frames = json.load(open(res.manifest))["frames"]
    assert sum(f["cluster_size"] for f in frames) == 2 * SECONDS  # every candidate assigned


def test_custom_method_registration(video, tmp_path):
    class MyCurator(FrameCurator):
        @curation_method("first_n")
        def _first_n(self, info, cfg):
            return Selection(indices=np.arange(cfg.get("n", 4)))

    assert {"uniform", "kmeans", "first_n"} <= set(MyCurator.methods())
    res = MyCurator().curate(video, str(tmp_path / "f.mp4"), method="first_n", n=5)
    assert res.indices.tolist() == [0, 1, 2, 3, 4]


def test_probe(video):
    info = probe_video(video)
    assert info.fps == pytest.approx(FPS) and info.n_frames == FPS * SECONDS


def test_supermemory_time_parsing():
    assert to_sec(12.5) == 12.5
    assert to_sec("16:32") == 992.0
    assert to_sec("1:00:05") == 3605.0
    assert spans_of({"time_spans": [{"start_time": 1, "end_time": 2}]}) == [(1.0, 2.0)]
    assert spans_of({"time_span": {"start_time": "0:10", "end_time": "0:20"}}) == [(10.0, 20.0)]


def test_evidence_rows_question_cutoff():
    qa = [{"question_id": 1, "subject": 1, "question": "where?", "is_answerable": True,
           "metadata": {"skill": "object_location_memory", "primary_video_id": "A"},
           "question_evidence": {"time_spans": [{"start_time": 100, "end_time": 110}]},
           "answer_evidence": {"evidence_list": [
               {"video_id": "A", "time_span": {"start_time": 40, "end_time": 50}},     # same session
               {"video_id": "B", "time_span": {"start_time": 5, "end_time": 9}}]}}]   # earlier session
    rows = {r["video_id"]: r for r in evidence_rows(qa)}
    assert rows["A"]["q_cut"] == 100.0           # frames after the question must not count
    assert rows["B"]["q_cut"] == float("inf")


def test_coverage_hits():
    spans = [(10, 20, float("inf")), (50, 60, 55)]
    assert hits([15], spans, tol=0).tolist() == [True, False]
    assert hits([58], spans, tol=0).tolist() == [False, False]   # after the question time
    assert hits([22], spans, tol=2).tolist() == [True, False]    # within tolerance


def test_make_sessions_all_skip_done(tmp_path, monkeypatch, capsys):
    from hindsight.curation import make_sessions as ms

    def rec(qid, prim, ev, answerable=True):
        return {"question_id": qid, "subject": 1, "question": "where?", "is_answerable": answerable,
                "metadata": {"skill": "object_location_memory", "primary_video_id": prim},
                "question_evidence": {"time_spans": [{"start_time": 100, "end_time": 110}]},
                "answer_evidence": {"evidence_list": [{"video_id": v, "time_span": {"start_time": 1, "end_time": 2}}
                                                      for v in ev]}}
    qa = [rec(1, "P_1_session_2", ["P_1_session_1", "P_1_session_2"]), rec(2, "P_1_session_3", [], answerable=False),
          rec(3, "P_1_session_2", ["P_1_session_9"])]                       # session 9 is not in the repo
    monkeypatch.setattr(ms, "load_qa", lambda path: qa)
    monkeypatch.setattr(ms, "video_paths", lambda: {v: f"data/video/P_1/{v}.mp4" for v in
                                                    ("P_1_session_1", "P_1_session_2", "P_1_session_3")})
    out = tmp_path / "out"
    for m in ("kmeans", "uniform"):                                         # session 1 already curated
        for ext in ("mp4", "json"):
            (out / "P_1_session_1").mkdir(parents=True, exist_ok=True)
            (out / "P_1_session_1" / f"{m}_K100.{ext}").write_text("")
    assert ms.is_done(str(out), "P_1_session_1", [100]) and not ms.is_done(str(out), "P_1_session_1", [32, 100])
    sessions = tmp_path / "rest.txt"
    monkeypatch.setattr("sys.argv", ["x", "--all", "--skip-done", str(out), "--ks", "100", "--out", str(sessions)])
    ms.main()
    vids = [line.split("\t")[1] for line in sessions.read_text().splitlines()]
    assert sorted(vids) == ["P_1_session_2", "P_1_session_3"]             # session 3: only an unanswerable question
    assert "P_1_session_9" in capsys.readouterr().out                     # reported as missing from the repo
