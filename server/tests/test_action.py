"""Action labeling + repetition detection, trained on train_01, judged on val_01.

Repetition is scored from ground-truth segment labels and the segment
embeddings only: [cosine, emb_a*emb_b, |emb_a-emb_b|, same_class], never the
time gap. The hard part is telling a genuine repeat from two unrelated actions
that merely share a label — the same-class rule is blind to that (0.5 AUROC on
same-label pairs), while the learned scorer leans on the instance embedding a
repeat inherits from the action it repeats.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from hindsight.config import settings
from hindsight.ingest import synthetic as S
from hindsight.action import ActionClassifier, ActionDetector, RepetitionScorer, SameClassRule
from hindsight.action.repetition import is_repeat_pair, pair_features
from hindsight.timeline.models import ActionSegment, Alert

ROOT = settings.data_dir / "synthetic"


@pytest.fixture(scope="module")
def data():
    return S.load(ROOT, "train_01"), S.load(ROOT, "val_01")


@pytest.fixture(scope="module")
def trained(data):
    train, _ = data
    return ActionClassifier().fit(train), RepetitionScorer().fit(train)


def _pair_scores(video, scorer, same_label_only):
    """Candidate pairs within the repeat window -> (truth, scores) using true labels."""
    emb = video.seg_emb
    y, p = [], []
    for i, a in enumerate(video.segments):
        for b in video.segments[i + 1:]:
            if b.t_start - a.t_start > settings.repeat_window_s:
                continue
            if same_label_only and a.label != b.label:
                continue
            y.append(int(is_repeat_pair(a, b)))
            p.append(scorer.score(a, b, emb[a.seg_emb_idx], emb[b.seg_emb_idx]))
    return np.array(y), np.array(p)


def test_action_classification_beats_chance(trained, data):
    clf, _ = trained
    train, val = data
    labels, probs = clf.predict_windows(val.win_feat)
    assert probs.shape == (len(val.win_label), len(clf.net.classes_))

    macro_f1 = f1_score(val.win_label, labels, average="macro")
    acc = accuracy_score(val.win_label, labels)
    present = np.array([y in set(train.win_label.tolist()) for y in val.win_label])
    present_f1 = f1_score(val.win_label[present], labels[present], average="macro")

    print(f"\naction macro-F1 (val present classes): {present_f1:.3f}")
    print(f"action macro-F1 (all val): {macro_f1:.3f}")
    print(f"per-window accuracy: {acc:.3f}")

    assert present_f1 > 0.8
    assert macro_f1 > 0.5


def test_segment_label_aggregates_windows(trained, data):
    clf, _ = trained
    _, val = data
    seg = val.segments[0]
    win_idx = np.nonzero((val.win_t >= seg.t_start) & (val.win_t < seg.t_end))[0]
    lid, conf = clf.segment_label(win_idx, val.win_feat)
    assert 0 <= lid < len(val.actions_vocab)
    assert 0.0 <= conf <= 1.0


def test_classifier_save_load_roundtrip(trained, data, tmp_path):
    clf, _ = trained
    _, val = data
    path = tmp_path / "action.joblib"
    clf.save(path)
    reloaded = ActionClassifier.load(path)
    a, _ = clf.predict_windows(val.win_feat)
    b, _ = reloaded.predict_windows(val.win_feat)
    assert np.array_equal(a, b)
    assert reloaded.vocab == clf.vocab


def test_repetition_omits_time_gap():
    # 1 cosine + 2*emb_dim + 1 same-class, no slot for a time gap.
    e = np.ones(64, dtype=np.float32) / 8
    assert pair_features(e, e, 1.0).shape == (1 + 2 * 64 + 1,)


def test_repetition_beats_same_class_rule(trained, data):
    _, rep = trained
    _, val = data
    rule = SameClassRule()

    # the discriminative task: repeats vs coincidental same-label actions.
    y, model_p = _pair_scores(val, rep, same_label_only=True)
    _, rule_p = _pair_scores(val, rule, same_label_only=True)
    auroc_model = roc_auc_score(y, model_p)
    auroc_rule = roc_auc_score(y, rule_p)

    # the full candidate set, for context (cross-label pairs are easy for both).
    ya, ma = _pair_scores(val, rep, same_label_only=False)
    _, ra = _pair_scores(val, rule, same_label_only=False)
    print(f"\nrepetition AUROC (same-label pairs)  model={auroc_model:.3f}  rule={auroc_rule:.3f}"
          f"  (+{auroc_model - auroc_rule:.3f})")
    print(f"repetition AUROC (all candidates)    model={roc_auc_score(ya, ma):.3f}"
          f"  rule={roc_auc_score(ya, ra):.3f}")

    assert auroc_rule == pytest.approx(0.5, abs=0.1)  # blind to within-label structure
    assert auroc_model > 0.7
    assert auroc_model >= auroc_rule + 0.05


def test_detector_flags_and_alerts(trained, data):
    clf, rep = trained
    _, val = data
    det = ActionDetector(clf, rep)
    segments = det.process(val)

    assert all(isinstance(s, ActionSegment) for s in segments)
    assert len(segments) == len(val.segments)

    truth = {s.idx for s in val.segments if s.is_repeat}
    flagged = {s.id for s in segments if s.is_repeat}
    recall = len(flagged & truth) / len(truth)
    precision = len(flagged & truth) / max(len(flagged), 1)
    print(f"\ndetector: {len(det.alerts)} repeat alerts, recall={recall:.2f}, precision={precision:.2f}")

    assert det.alerts and all(isinstance(a, Alert) and a.kind == "repeat" for a in det.alerts)
    for s in segments:
        if s.is_repeat:
            assert s.repeat_of is not None and s.repeat_of < s.id and s.repeat_score > 0
    assert recall >= 0.6


def test_streaming_step_matches_batch(trained, data):
    clf, rep = trained
    _, val = data
    det = ActionDetector(clf, rep)
    det.reset()

    alerts = []
    for seg in val.segments:
        win_idx = np.nonzero((val.win_t >= seg.t_start) & (val.win_t < seg.t_end))[0]
        lid, conf = clf.segment_label(win_idx, val.win_feat)
        out, alert = det.step(lid, val.seg_emb[seg.seg_emb_idx], seg.t_start, seg.t_end, conf)
        assert isinstance(out, ActionSegment)
        if alert is not None:
            alerts.append(alert)
    assert alerts
