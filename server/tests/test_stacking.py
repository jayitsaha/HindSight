import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from hindsight.evaluation import abstention as ab
from hindsight.evaluation import stacking as st


def _mock(tmp_path, n_participants=5, n_per=40, seed=0):
    root = tmp_path / "views"
    questions, view_dirs = st.make_mock_lp(str(root), n_participants=n_participants, n_per=n_per, seed=seed)
    return questions, view_dirs


def test_runs_end_to_end(tmp_path):
    questions, (km, uni, blind) = _mock(tmp_path)
    res = st.analyse_stacking(km, uni, blind, questions, str(tmp_path / "out"), meta="gbm", seed=0)
    st.write_report(res, str(tmp_path / "out"))
    assert res["n"] == 5 * 40 and res["n_participants"] == 5
    for name in ["stacker", "cross_view_ablated", "single_view", "threshold"]:
        m = res["methods"][name]
        assert np.isfinite(m["aurc"]) and 0.0 <= m["aurc"] <= 1.0
        assert np.isfinite(m["answerability_macro_f1"]) and 0.0 <= m["answerability_macro_f1"] <= 1.0
    assert "p_vs_single_view" in res["methods"]["stacker"]
    assert "p_vs_cross_view_ablated" in res["methods"]["stacker"]
    p = res["methods"]["stacker"]["p_vs_cross_view_ablated"]
    assert np.isfinite(p) and 0.0 <= p <= 1.0
    assert res["methods"]["cross_view_ablated"]["p_vs_cross_view_ablated"] == 1.0
    for f in ["STACKING.md", "stacking_results.json", "stacking_decisions.jsonl"]:
        assert (tmp_path / "out" / f).is_file(), f
    decisions = [json.loads(l) for l in open(tmp_path / "out" / "stacking_decisions.jsonl")]
    assert len(decisions) == 200
    assert {"stacker", "single_view", "cross_view_ablated", "threshold", "y"} <= set(decisions[0])


def test_feature_width(tmp_path):
    questions, (km, uni, blind) = _mock(tmp_path)
    kmv = st.load_view(km, questions)
    univ = st.load_view(uni, questions)
    blindv = st.load_view(blind, questions)
    aligned = st.align_views(kmv, univ, blindv)
    X, mask = st.build_stacked(aligned)
    n_options = 4
    expected = 3 * len(ab.FEATURES) + 2 * n_options + st.N_AGREEMENT
    assert X.shape[1] == expected
    assert int(mask.sum()) == 2 * n_options + st.N_AGREEMENT
    assert not mask[: 3 * len(ab.FEATURES)].any()


def test_blind_ablation_zeros_cross_view(tmp_path):
    questions, (km, uni, blind) = _mock(tmp_path)
    kmv = st.load_view(km, questions)
    univ = st.load_view(uni, questions)
    blindv = st.load_view(blind, questions)
    aligned = st.align_views(kmv, univ, blindv)
    X, mask = st.build_stacked(aligned)
    X_ablation = X.copy()
    X_ablation[:, mask] = 0.0
    assert float(np.abs(X_ablation[:, mask]).max()) == 0.0
    assert float(X_ablation[:, mask].std()) == 0.0
    assert float(X[:, mask].std()) > 0.0
    assert np.allclose(X_ablation[:, ~mask], X[:, ~mask])


def test_mismatched_question_ids_raises(tmp_path):
    questions, (km, uni, blind) = _mock(tmp_path)
    kmv = st.load_view(km, questions)
    univ = st.load_view(uni, questions)
    blindv = st.load_view(blind, questions)
    shuffled = ([blindv[0][1]] + blindv[0][:1] + blindv[0][2:], blindv[1], blindv[2])
    with pytest.raises(ValueError):
        st.align_views(kmv, univ, shuffled)


def test_no_logprobs_raises_legibly(tmp_path):
    questions, (km, uni, blind) = _mock(tmp_path, n_participants=3, n_per=10)
    for d in (km, uni, blind):
        raw = [json.loads(l) for l in open(tmp_path / "views" / d.split("/")[-1] / "raw_predictions.jsonl")]
        for r in raw:
            r["option_logprobs"] = None
        open(tmp_path / "views" / d.split("/")[-1] / "raw_predictions.jsonl", "w").write(
            "\n".join(json.dumps(r) for r in raw) + "\n")
    with pytest.raises(ValueError, match="option_logprobs"):
        st.analyse_stacking(km, uni, blind, questions, str(tmp_path / "out"), meta="logreg", seed=0)


def test_cli_end_to_end(tmp_path):
    root = tmp_path / "cli"
    questions, (km, uni, blind) = st.make_mock_lp(str(root), n_participants=4, n_per=30, seed=1)
    st.main([km, uni, blind, "--out", str(tmp_path / "cliout"), "--meta", "logreg", "--seed", "1"])
    res = json.load(open(tmp_path / "cliout" / "stacking_results.json"))
    assert res["n"] == 120 and res["feature_width"] == 3 * len(ab.FEATURES) + 2 * 4 + st.N_AGREEMENT
