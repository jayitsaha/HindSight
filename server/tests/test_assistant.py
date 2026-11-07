import numpy as np
import pytest

from hindsight.assistant import Assistant, RefusalGate, classify, extract_target
from hindsight.assistant.answer import combine_confidence
from hindsight.config import settings
from hindsight.ingest import synthetic as S
from hindsight.providers import VLMVerdict


@pytest.fixture(scope="module")
def data():
    root = settings.data_dir / "synthetic"
    return S.load(root, "train_01"), S.load(root, "val_01")


@pytest.fixture(scope="module")
def fitted(data):
    train, _ = data
    return Assistant(scorer_name="logreg").fit(train)


def _eval(asst, video):
    """Per-query confidence, real answer correctness, and answerable truth.

    Correctness is what the assistant COMMITS to being right (object label ==
    target / action window overlaps the true occurrence), measured pre-refusal
    via judge_queries -- not the dishonest "answerable and not refused"."""
    judged = asst.judge_queries(video)
    conf = np.array([j["conf"] for j in judged])
    correct = np.array([j["correct"] for j in judged], bool)
    answerable = np.array([j["answerable"] for j in judged], bool)
    return conf, correct, answerable


def test_intent_and_target_parsing():
    assert classify("where is my wallet?") == "object"
    assert classify("when did I last drinking?") == "action"
    assert extract_target("where is my wallet?", "object") == "wallet"
    assert extract_target("when did I last drinking?", "action") == "drinking"


def test_refusal_gate_maximizes_coverage_at_target():
    # clean separation: all correct answers above 0.8, all wrong ones below 0.4
    conf = np.array([0.95, 0.9, 0.88, 0.85, 0.82, 0.3, 0.25, 0.2, 0.1])
    correct = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    gate = RefusalGate(target_acc=0.80).fit(conf, correct)
    assert not gate.decide(0.85)        # commit to the confident, correct ones
    assert gate.decide(0.3)             # refuse the low-confidence wrong ones
    assert gate.coverage_at_accuracy(conf, correct) >= 5 / 9


def test_vlm_disagreement_lowers_confidence():
    base = 0.8
    agree = combine_confidence(base, VLMVerdict("yes", True, 0.9))
    disagree = combine_confidence(base, VLMVerdict("no", False, 0.9))
    assert disagree < base < agree
    assert combine_confidence(base, None) == base


def test_demo_free_text_answers_present_objects(fitted):
    # the API answers demo sessions with free-text questions whose (text, t) won't
    # match a planted query; a present object must still be answerable.
    demo = S.load(settings.data_dir / "synthetic", "demo_01")
    t = demo.minutes * 60
    present = demo.dets[0].label  # an object that was actually seen

    ans = fitted.answer(f"where is my {present}?", t, video=demo)
    assert not ans.refused
    assert ans.confidence > 0.0
    assert ans.window is not None

    absent = fitted.answer("where is my submarine?", t, video=demo)
    assert absent.refused
    assert absent.confidence == 0.0


def test_calibrated_assistant_refuses_when_unsure(fitted, data):
    _, val = data
    conf, correct, answerable = _eval(fitted, val)

    refused = np.array([fitted.answer(q.text, q.t_asked, video=val).refused
                        for q in val.queries])
    answered = ~refused

    coverage = answered.mean()
    acc_answered = correct[answered].mean() if answered.any() else float("nan")
    cov_at_80 = fitted.gate.coverage_at_accuracy(conf, correct, 0.80)

    aurc_model = fitted.gate.aurc(conf, correct)
    rng = np.random.default_rng(0)
    aurc_random = fitted.gate.aurc(rng.random(len(conf)), correct)

    unans = ~answerable
    refusal_recall = (refused & unans).sum() / unans.sum()

    print(f"\ncoverage (answered)      {coverage:.3f}")
    print(f"accuracy on answered     {acc_answered:.3f}")
    print(f"coverage @ 80% accuracy  {cov_at_80:.3f}")
    print(f"AURC model / random      {aurc_model:.4f} / {aurc_random:.4f}")
    print(f"refusal recall (unans.)  {refusal_recall:.3f}")

    # (1) the gate holds the target accuracy on committed answers, non-trivially
    assert coverage >= 0.15
    assert acc_answered >= 0.80
    # (2) the confidence ranks answers far better than chance
    assert aurc_model < 0.6 * aurc_random
    # (3) unanswerable questions are refused at a high rate
    assert refusal_recall >= 0.80
