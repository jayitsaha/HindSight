"""Ego4D loader tests on tiny v2-shaped fixtures (no real data needed)."""

import json

import numpy as np
import pytest
import torch

from hindsight.ingest import ego4d


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def _wrap(clip_uid, ann):
    return {"videos": [{"video_uid": "vid_a", "clips": [
        {"clip_uid": clip_uid, "video_start_sec": 120.0, "annotations": [ann]}]}]}


def test_vq2d(tmp_path):
    ann = {"query_sets": {
        "1": {"is_valid": True, "object_title": "mug", "query_frame": 900,
              "visual_crop": {"frame_number": 400, "x": 10, "y": 20, "width": 30, "height": 40},
              "response_track": [
                  {"frame_number": 700, "x": 0, "y": 0, "width": 10, "height": 10},
                  {"frame_number": 702, "x": 5, "y": 5, "width": 10, "height": 10}]},
        "2": {"is_valid": False, "query_sets": {}},
    }}
    qs = ego4d.load_vq2d(_write(tmp_path, "vq_val.json", _wrap("clip_1", ann)))
    assert len(qs) == 1
    q = qs[0]
    assert (q.clip_uid, q.object_title, q.query_frame) == ("clip_1", "mug", 900)
    assert q.crop_xyxy == (10.0, 20.0, 40.0, 60.0)
    assert q.response_frame == 702 and q.response_xyxy == (5.0, 5.0, 15.0, 15.0)
    assert [f for f, _ in q.response_track] == [700, 702]


def test_mq(tmp_path):
    ann = {"labels": [
        {"label": "use_phone", "start_time": 3.5, "end_time": 9.0},
        {"label": "clean_dishes", "moment_start_sec": 12.0, "moment_end_sec": 20.5},
        {"label": "broken", "start_time": None, "end_time": 5.0},
    ]}
    segs = ego4d.load_mq(_write(tmp_path, "moments_val.json", _wrap("clip_2", ann)))
    assert [(s.label, s.t_start, s.t_end) for s in segs] == [
        ("use_phone", 3.5, 9.0), ("clean_dishes", 12.0, 20.5)]
    assert segs[0].video_uid == "vid_a"


def test_nlq(tmp_path):
    ann = {"language_queries": [
        {"query": "where did I put my keys?", "clip_start_sec": 41.2, "clip_end_sec": 44.8},
        {"query": "what did I pour?"},  # unannotated test query, no window
    ]}
    qs = ego4d.load_nlq(_write(tmp_path, "nlq_val.json", _wrap("clip_3", ann)))
    assert len(qs) == 1
    assert qs[0].query == "where did I put my keys?"
    assert (qs[0].t_start, qs[0].t_end) == (41.2, 44.8)


def test_pnr_flat_and_nested(tmp_path):
    flat = {"clips": [{"clip_uid": "clip_4", "video_uid": "vid_a",
                       "pnr_frame": 55, "state_change": True}]}
    nested = {"clips": [{"clip_uid": "clip_5", "video_uid": "vid_b", "frames": [
        {"pnr_frame": {"frame": 120}, "pre_frame": {"frame": 90},
         "post_frame": {"frame": 150}, "state_change": False}]}]}
    a = ego4d.load_pnr(_write(tmp_path, "fho_oscc-pnr_val.json", flat))
    b = ego4d.load_pnr(_write(tmp_path, "fho_hands_val.json", nested))
    assert a[0].pnr_frame == 55 and a[0].state_change and a[0].pre_frame is None
    assert (b[0].pnr_frame, b[0].pre_frame, b[0].post_frame) == (120, 90, 150)
    assert not b[0].state_change


def test_read_features(tmp_path):
    torch.save(torch.ones(8, 1536), tmp_path / "clip_pt.pt")
    np.save(tmp_path / "clip_npy.npy", np.zeros((5, 256)))
    pt = ego4d.read_features(tmp_path, "clip_pt")
    npy = ego4d.read_features(tmp_path, "clip_npy")
    assert pt.shape == (8, 1536) and pt.dtype == np.float32
    assert npy.shape == (5, 256) and npy.dtype == np.float32


def test_missing_paths_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="ego4d --output_directory"):
        ego4d.load_nlq(tmp_path / "nlq_val.json")
    with pytest.raises(FileNotFoundError, match="omnivore"):
        ego4d.read_features(tmp_path, "nope")


def test_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("EGO4D_ROOT", str(tmp_path))
    p = ego4d.Ego4DPaths.from_env()
    assert p.vq2d("train") == tmp_path / "v2" / "annotations" / "vq_train.json"
    assert p.mq() == tmp_path / "v2" / "annotations" / "moments_val.json"
    assert p.pnr().name == "fho_oscc-pnr_val.json"
    assert p.features() == tmp_path / "v2" / "omnivore_video_swinl"
    monkeypatch.delenv("EGO4D_ROOT")
    with pytest.raises(FileNotFoundError, match="EGO4D_ROOT"):
        ego4d.Ego4DPaths.from_env()
