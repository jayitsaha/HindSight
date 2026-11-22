import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hindsight.video.curation_temporal import TemporalCurator

FPS, SECONDS = 10, 60
EVENTS = {15, 30, 45}
K = 8


class PlantedCurator(TemporalCurator):
    def embed_frames(self, info, indices, emb_cfg, cache_dir=None):
        secs = np.round(np.asarray(indices) / info.fps).astype(int)
        emb = np.zeros((len(secs), 4), np.float32)
        emb[:, 0] = 1.0 + secs / float(SECONDS) * 4.0
        for i, s in enumerate(secs):
            if int(s) in EVENTS:
                emb[i] = np.array([10.0, 50.0, 0.0, 0.0], np.float32)
        return emb


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "scene.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 64))
    for i in range(FPS * SECONDS):
        sec = i // FPS
        val = 200 if sec in EVENTS else 40 + sec
        vw.write(np.full((64, 64, 3), val, np.uint8))
    vw.release()
    return str(path)


def _config():
    emb = {"backend": "toy"}
    return {"temporal": {"embedder": emb, "embed_fps": 1.0, "connectivity_neighbors": 2, "linkage": "ward"},
            "kmeans": {"embedder": emb, "embed_fps": 1.0, "normalize": True}}


def _secs(res):
    return set(np.round(res.timestamps).astype(int).tolist())


def test_temporal_registered():
    assert "temporal" in TemporalCurator.methods()
    assert {"uniform", "kmeans", "temporal"} <= set(TemporalCurator.methods())


def test_exact_count_and_sorted(video, tmp_path):
    cur = PlantedCurator(_config())
    res = cur.curate(video, str(tmp_path / "t.mp4"), method="temporal", k=K)
    assert len(res.indices) == K
    assert np.all(np.diff(res.indices) > 0)
    assert res.extra["n_candidates"] >= K
    assert res.extra["linkage"] == "ward"


def test_temporal_beats_kmeans_on_short_events(video, tmp_path):
    cur = PlantedCurator(_config())
    temporal = cur.curate(video, str(tmp_path / "tmp.mp4"), method="temporal", k=K)
    kmeans = cur.curate(video, str(tmp_path / "km.mp4"), method="kmeans", k=K)

    temporal_cov = EVENTS & _secs(temporal)
    kmeans_cov = EVENTS & _secs(kmeans)

    assert temporal_cov == EVENTS
    assert len(kmeans_cov) < len(EVENTS)
    assert len(temporal_cov) > len(kmeans_cov)


def test_k_larger_than_candidates(video, tmp_path):
    cur = PlantedCurator(_config())
    res = cur.curate(video, str(tmp_path / "big.mp4"), method="temporal", k=100000)
    n = res.extra["n_candidates"]
    assert len(res.indices) == n
    assert res.extra["k"] == n
    assert np.all(np.diff(res.indices) > 0)
