"""Temporal agglomerative selection and cache reuse, without decoding video (runs where OpenCV is missing;
tests/test_curation.py covers the same method end to end on a real synthetic video)."""
import sys
import types

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
_stub = False
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")               # only needed for decoding, unused here
    _stub = True

from hindsight.video import curation  # noqa: E402
from hindsight.video.curation import FrameCurator, VideoInfo  # noqa: E402

if _stub:                                                      # keep other tests' importorskip("cv2") honest
    del sys.modules["cv2"]


def info(path="/x/aba.mp4", n_frames=360, fps=30.0):
    return VideoInfo(path=path, fps=fps, n_frames=n_frames, width=64, height=64)


def aba_embeddings(n):
    """Scene A, B, A again (thirds of the video), tiny noise."""
    rng = np.random.default_rng(0)
    e = np.zeros((n, 3), np.float32)
    e[: n // 3, 0] = 1; e[n // 3: 2 * n // 3, 1] = 1; e[2 * n // 3:, 0] = 1
    return e + rng.normal(0, 0.01, e.shape).astype(np.float32)


def test_temporal_keeps_returning_scene(monkeypatch):
    cur = FrameCurator({"kmeans": {"embed_fps": 1.0}})
    monkeypatch.setattr(cur, "embed_frames", lambda inf, idx, cfg, cache=None: aba_embeddings(len(idx)))
    sel = cur._select_agglomerative(info(), {"k": 3, "temporal": True, "linkage": "ward"})
    t = np.sort(sel.indices) / 30.0
    assert (t // 4).astype(int).tolist() == [0, 1, 2]          # one frame per stretch, A twice
    assert all(f["cluster_t_max"] - f["cluster_t_min"] < 4 for f in sel.per_frame)
    plain = cur._select_agglomerative(info(), {"k": 3, "temporal": False, "linkage": "ward"})
    assert len(plain.indices) == 3 and sel.extra["temporal"] is True and plain.extra["temporal"] is False


def test_falls_back_to_kmeans_settings(monkeypatch):
    seen = {}
    cur = FrameCurator({"kmeans": {"embed_fps": 0.5, "cache_dir": "cachedir", "embedder": {"backend": "toy"}}})

    def fake(inf, idx, cfg, cache=None):
        seen.update(n=len(idx), cfg=cfg, cache=cache)
        return aba_embeddings(len(idx))
    monkeypatch.setattr(cur, "embed_frames", fake)
    cur._select_agglomerative(info(), cur.config["agglomerative"] | {"k": 3})
    assert seen["n"] == 6 and seen["cache"] == "cachedir" and seen["cfg"]["backend"] == "toy"


def test_cache_reused_across_file_keys(tmp_path):
    video = tmp_path / "Person_1_session_1_x.mp4"
    video.write_bytes(b"0")
    idx = np.array([0, 30, 60])
    np.savez_compressed(tmp_path / "Person_1_session_1_x_oldkey1234.npz", emb=np.eye(3), indices=idx)
    np.savez_compressed(tmp_path / "Person_1_session_1_x_otherkey99.npz", emb=np.zeros((2, 3)), indices=idx[:2])

    class Boom:
        name = "boom"

        def __init__(self, **_):
            pass

        def embed(self, images):
            raise AssertionError("should reuse the cache")
    curation.EMBEDDERS["boom"] = Boom
    try:
        emb = FrameCurator().embed_frames(info(path=str(video)), idx, {"backend": "boom"}, str(tmp_path))
    finally:
        del curation.EMBEDDERS["boom"]
    assert np.array_equal(emb, np.eye(3))
