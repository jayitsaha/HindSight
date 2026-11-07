import numpy as np
import pytest

from hindsight.metrics import (
    ari,
    aurc,
    auroc,
    average_precision,
    coverage_at_accuracy,
    macro_f1,
    mrr,
    nmi,
    pnr_timing_error,
    recall_at_k,
    risk_coverage_curve,
    temporal_iou,
    temporal_map,
)


def test_recall_at_k():
    # perfect ranker: relevant item first for every query
    perfect = [[True, False, False], [True, False, False]]
    assert recall_at_k(perfect, 1) == 1.0
    # one query hits at rank 1, one at rank 3
    mixed = [[True, False, False], [False, False, True]]
    assert recall_at_k(mixed, 1) == 0.5
    assert recall_at_k(mixed, 3) == 1.0
    # single-query 1D input
    assert recall_at_k([False, True, False], 2) == 1.0
    assert recall_at_k([False, True, False], 1) == 0.0


def test_mrr_both_forms():
    # rank positions: 1 and 2 -> (1 + 1/2) / 2
    assert mrr([1, 2]) == pytest.approx(0.75)
    # not-found encoded as 0
    assert mrr([0, 2]) == pytest.approx(0.25)
    # boolean rank lists: first True at rank 1 and rank 3 -> (1 + 1/3)/2
    assert mrr([[True, False], [False, False, True]]) == pytest.approx((1 + 1 / 3) / 2)


def test_average_precision():
    # perfectly ordered -> AP 1.0
    assert average_precision([0.9, 0.8, 0.1], [1, 1, 0]) == pytest.approx(1.0)
    # one negative ranked above a positive
    ap = average_precision([0.9, 0.8, 0.1], [1, 0, 1])
    assert 0.0 < ap < 1.0
    assert average_precision([0.5, 0.4], [0, 0]) == 0.0


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert auroc([0.9, 0.1], [0, 1]) == pytest.approx(0.0)
    # single class -> neutral fallback, no crash
    assert auroc([0.3, 0.7], [1, 1]) == 0.5


def test_pnr_timing_error():
    err = pnr_timing_error([10.0, 20.0, 30.0], [11.0, 18.0, 30.0])
    assert err["mean"] == pytest.approx(1.0)
    assert err["median"] == pytest.approx(1.0)


def test_macro_f1():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    # 3 classes, class 2 fully wrong -> macro mean drops below 1
    assert macro_f1([0, 0, 1, 2], [0, 0, 1, 1]) < 1.0


def test_temporal_iou():
    gts = np.array([[0.0, 10.0], [20.0, 30.0], [0.0, 5.0]])
    iou = temporal_iou((0.0, 10.0), gts)
    assert iou[0] == pytest.approx(1.0)  # identical
    assert iou[1] == pytest.approx(0.0)  # disjoint
    assert iou[2] == pytest.approx(0.5)  # [0,5] inside [0,10]


def test_temporal_map_perfect_and_partial():
    gts = [(0.0, 10.0, "walk", 1.0), (20.0, 30.0, "sit", 1.0)]
    perfect = [(0.0, 10.0, "walk", 0.9), (20.0, 30.0, "sit", 0.8)]
    res = temporal_map(perfect, gts)
    assert res["mAP"] == pytest.approx(1.0)
    # a badly localized prediction fails the high-IoU thresholds
    shifted = [(5.0, 12.0, "walk", 0.9), (20.0, 30.0, "sit", 0.8)]
    res2 = temporal_map(shifted, gts)
    assert res2["mAP"] < res["mAP"]


def test_risk_coverage_and_aurc():
    conf = [0.9, 0.8, 0.7, 0.6]
    # calibrated: confidence order matches correctness
    calibrated = [1, 1, 0, 0]
    cov, risk = risk_coverage_curve(conf, calibrated)
    assert cov[-1] == pytest.approx(1.0)
    assert risk[0] == pytest.approx(0.0)  # top answer is correct
    assert risk[-1] == pytest.approx(0.5)  # 2/4 wrong overall
    # anti-calibrated: wrong answers are the most confident
    bad = [0, 0, 1, 1]
    assert aurc(conf, calibrated) < aurc(conf, bad)


def test_coverage_at_accuracy():
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [1, 1, 1, 0]
    # top 3 are correct -> can cover 75% at 100% accuracy
    assert coverage_at_accuracy(conf, correct, 1.0) == pytest.approx(0.75)
    # relax target -> full coverage (75% accuracy overall)
    assert coverage_at_accuracy(conf, correct, 0.75) == pytest.approx(1.0)


def test_clustering():
    labels = [0, 0, 1, 1, 2, 2]
    assert nmi(labels, labels) == pytest.approx(1.0)
    assert ari(labels, labels) == pytest.approx(1.0)
    # permuted labels are the same clustering
    permuted = [2, 2, 0, 0, 1, 1]
    assert nmi(labels, permuted) == pytest.approx(1.0)
    assert ari(labels, permuted) == pytest.approx(1.0)
    # random split of a single true cluster -> low agreement
    assert ari([0, 0, 0, 0], [0, 1, 0, 1]) == pytest.approx(0.0, abs=1e-9)
