"""The consolidated harness: train all four modules, eval, sanity-check the table.

Runs the real train_01 -> val_01 pipeline (a few seconds) and asserts the report
has every module's keys and that the headline numbers land where the individual
module tests say they should.
"""

import pytest

from hindsight.training import eval_all, load_split


@pytest.fixture(scope="module")
def report():
    return eval_all.run(save=False, verbose=False)


def test_report_has_every_module(report):
    assert set(report) == {"retrieval", "putdown", "action", "assistant", "clustering"}


def test_split_loads(report):
    train, val = load_split()
    assert train.video_id == "train_01" and val.video_id == "val_01"


def test_retrieval_trained_beats_cosine(report):
    sc = report["retrieval"]["scorers"]
    cos = sc["cosine"]
    assert sc["logreg"]["auc"] >= cos["auc"]
    assert sc["logreg"]["recall@1"] >= cos["recall@1"]
    km = report["retrieval"]["kmeans_index"]
    assert abs(km["recall@1"] - km["brute_recall@1"]) <= 0.05


def test_putdown(report):
    pd = report["putdown"]
    assert pd["pnr"]["auroc"] > 0.9
    assert pd["pnr"]["state_acc"] > 0.9
    # clean synthetic rooms make counting strong; clustering is only competitive
    assert pd["misplacement"]["model_auroc"] >= pd["misplacement"]["baseline_auroc"] - 0.08


def test_action(report):
    ac = report["action"]
    assert ac["classifier"]["macro_f1"] > 0.8
    assert ac["classifier"]["window_acc"] > 0.8
    assert ac["localization"]["mAP"] > 0.5
    # repetition beats the same-class rule by a clear margin, without any time feature
    assert ac["repetition"]["scorer_auroc"] >= ac["repetition"]["rule_auroc"] + 0.05


def test_clustering(report):
    cl = report["clustering"]
    # action windows cluster cleanly; object crops don't (appearance noise buries
    # identity), which is the motivation for a supervised retrieval metric
    assert cl["action_type"]["nmi"] > 0.5
    assert cl["object_identity"]["nmi"] < cl["action_type"]["nmi"]


def test_assistant_refuses_when_unsure(report):
    a = report["assistant"]
    assert a["refusal_recall"] > 0.5
    assert 0.0 <= a["coverage_at_80"] <= 1.0
    assert a["aurc_model"] < a["aurc_random"]
    assert a["accuracy_answered"] >= 0.8  # high accuracy on what it commits to
