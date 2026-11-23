"""100-frame selectors: per-session selectors, spaced top frames, coverage rule, full run and report."""
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from hindsight.curation import frame_selectors as fs  # noqa: E402
from tests.test_learned_retriever import make_world  # noqa: E402


def test_top_spaced_respects_gap_and_taken():
    vids = np.array(["a"] * 10 + ["b"] * 10)
    ts = np.r_[np.arange(10.0), np.arange(10.0)]
    score = np.r_[np.linspace(1, 0.5, 10), np.linspace(0.4, 0.1, 10)]
    out = fs.top_spaced(vids, ts, score, 3, gap=4, taken=[("b", 0.0)])
    assert out[:2] == [("a", 0.0), ("a", 4.0)]
    assert out[2] in {("a", 8.0), ("b", 4.0)}
    assert all(not (v == "b" and t < 4) for v, t in out)


def test_agglomerative_temporal_is_contiguous():
    rng = np.random.default_rng(0)
    emb = np.repeat(np.eye(4), 25, axis=0) + rng.normal(0, 0.01, (100, 4))
    emb = np.vstack([emb, emb[:25]])                       # the first scene comes back at the end
    t = np.arange(len(emb), dtype=float)
    plain = fs.agglomerative_times(emb, t, k=5)
    temporal = fs.agglomerative_times(emb, t, k=5, temporal=True)
    assert len(temporal) == 5 and len(plain) <= 5
    assert any(x >= 100 for x in temporal)                 # the returning scene gets its own frame


def test_span_coverage_rule():
    q = dict(primary="P_s2", q_time=50.0, spans=[("P_s1", 10, 20), ("P_s2", 40, 45), ("P_s2", 60, 70)])
    got = fs.span_coverage(q, [("P_s1", 22.0), ("P_s2", 46.5), ("P_s2", 65.0)], tol=2.0)
    assert got.tolist() == [True, True, False]             # last span is after the question


def test_manifest_times(tmp_path):
    (tmp_path / "v").mkdir()
    (tmp_path / "v" / "kmeans_K100.json").write_text(json.dumps({"frames": [{"t_sec": 5.0}, {"t_sec": 1.0}]}))
    assert fs.manifest_times(str(tmp_path), "v", "kmeans").tolist() == [1.0, 5.0]
    assert fs.manifest_times(str(tmp_path), "v", "uniform") is None


def test_run_and_report(tmp_path):
    videos, qs, texts, starts = make_world(n_frames=600)
    rows, info, sel = fs.run(qs, videos, texts, starts, k=20, product_dims=4, n_random=2, log=lambda s: None)
    got = {r["selector"]: r for r in rows}
    assert set(got) == set(fs.AGNOSTIC + fs.AWARE) and info["answerable"] == 12
    assert got["clip_top"]["question_coverage"] >= got["random"]["question_coverage"]
    q0 = qs[0]["qid"]
    assert len(sel["kmeans"][q0]) == 20 and len(sel["kmeans+clip"][q0]) == 20
    assert len(sel["clip_top"][q0]) == 20
    fs.write_report(rows, info, sel, str(tmp_path))
    assert "Agglomerative, temporal" in (tmp_path / "SELECTORS.md").read_text()
    first = json.loads((tmp_path / "selections_hgb_top.jsonl").read_text().splitlines()[0])
    assert first["qid"] and len(first["frames"]) == 20
