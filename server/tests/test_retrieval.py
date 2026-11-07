import dataclasses

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from hindsight.assistant.retrieve import retrieve_object
from hindsight.config import settings
from hindsight.ingest import synthetic as S
from hindsight.retrieval import (
    KMeansIndex,
    ObjectRetrieval,
    PCAReducer,
    build_scorer,
    pair_features,
)
from hindsight.retrieval.service import _training_pairs


@pytest.fixture(scope="module")
def data():
    root = settings.data_dir / "synthetic"
    return S.load(root, "train_01"), S.load(root, "val_01")


def _answerable(video):
    return [q for q in video.queries if q.kind == "object" and q.answerable]


def _eval(scorer, video):
    """Recall@1, MRR, ROC-AUC on query->correct-crop matching."""
    C = video.det_emb
    crop_labels = np.array([d.label for d in video.dets])
    r1, rr, ys, ss = [], [], [], []
    for q in _answerable(video):
        qv = video.query_emb[q.query_emb_idx]
        s = scorer.rank(qv, C)
        order = np.argsort(-s)
        ranked = crop_labels[order]
        r1.append(ranked[0] == q.target_label)
        rr.append(1.0 / (np.where(ranked == q.target_label)[0][0] + 1))
        ys.append((crop_labels == q.target_label).astype(int))
        ss.append(s)
    auc = roc_auc_score(np.concatenate(ys), np.concatenate(ss))
    return float(np.mean(r1)), float(np.mean(rr)), float(auc)


def test_trained_scorers_beat_cosine(data):
    train, val = data
    X, y = _training_pairs(train, np.random.default_rng(0))

    metrics = {}
    for name in ("cosine", "logreg", "rf", "mlp"):
        scorer = build_scorer(name).fit(X, y)
        metrics[name] = _eval(scorer, val)

    print("\nscorer   R@1     MRR     AUC")
    for name, (r1, mrr, auc) in metrics.items():
        print(f"{name:7s} {r1:.3f}   {mrr:.3f}   {auc:.3f}")

    cos_r1, _, cos_auc = metrics["cosine"]
    for name in ("logreg", "mlp"):
        r1, _, auc = metrics[name]
        assert r1 > cos_r1, f"{name} R@1 {r1} did not beat cosine {cos_r1}"
        assert auc > cos_auc, f"{name} AUC {auc} did not beat cosine {cos_auc}"


def test_kmeans_index_matches_brute_force(data):
    _, val = data
    C = val.det_emb
    crop_labels = np.array([d.label for d in val.dets])
    index = KMeansIndex(n_clusters=32).build(C)

    bf, ix = [], []
    for q in _answerable(val):
        qv = val.query_emb[q.query_emb_idx]
        bf.append(crop_labels[int(np.argmax(C @ qv))] == q.target_label)
        ix.append(crop_labels[int(index.query(qv, topk=1)[0])] == q.target_label)
    bf_r1, ix_r1 = float(np.mean(bf)), float(np.mean(ix))
    print(f"\nbrute R@1={bf_r1:.3f}  kmeans R@1={ix_r1:.3f}")
    assert abs(bf_r1 - ix_r1) <= 0.02


def test_service_locate_returns_latest(data):
    train, val = data
    svc = ObjectRetrieval("logreg").fit(train)
    q = _answerable(val)[0]
    qv = val.query_emb[q.query_emb_idx]
    hit = svc.locate(qv, val.dets, val.det_emb, q.t_asked, query_text=q.text)

    assert hit.found
    assert hit.t <= q.t_asked
    # the returned detection should carry the asked label at or near the latest sighting
    labels = [d.label for d in val.dets if d.t <= q.t_asked and d.label == q.target_label]
    assert labels, "target never seen before ask time"
    latest_t = max(d.t for d in val.dets if d.t <= q.t_asked and d.label == q.target_label)
    assert hit.t <= latest_t + 1e-6


def test_live_clip_width_falls_back_to_cosine(data):
    # A scorer trained on the 64-D synthetic space, then asked live: the provider
    # CLIP embedder produces a 512-D query and the store holds 512-D crops. The
    # feature widths no longer match the scorer, so retrieval must fall back to
    # native cosine and still return a well-formed result -- never raise / 500.
    train, val = data
    svc = ObjectRetrieval("logreg").fit(train)
    assert svc._expected_dim == 64

    rng = np.random.default_rng(3)
    dets = val.dets
    C = rng.standard_normal((len(dets), 512)).astype(np.float32)   # 512-D crops
    q = rng.standard_normal(512).astype(np.float32)                 # 512-D query
    t = max(d.t for d in dets)

    hit = svc.locate(q, dets, C, t)
    assert isinstance(hit.found, bool)  # answered or refused, but no exception

    prop = retrieve_object(svc, q, dets, C, t, target="wallet")
    assert 0.0 <= prop.conf_base <= 1.0
    assert prop.hit is not None

    # a 512-D query against 64-D crops is incomparable -> refuse, still no raise
    bad = svc.locate(q, dets, val.det_emb, t)
    assert bad.found is False


def test_pca_reduces_wide_embeddings_and_retrieval_works(data):
    # Lift the synthetic space to 256-D (> pca_dim=128) so fit() must engage the
    # PCA reducer; retrieval should keep working through the reduced space.
    train, val = data
    assert settings.pca_dim == 128
    P = np.random.default_rng(7).standard_normal((train.det_emb.shape[1], 256)).astype(np.float32)
    wide_train = dataclasses.replace(train, det_emb=(train.det_emb @ P).astype(np.float32),
                                     query_emb=(train.query_emb @ P).astype(np.float32))
    wide_val = dataclasses.replace(val, det_emb=(val.det_emb @ P).astype(np.float32),
                                   query_emb=(val.query_emb @ P).astype(np.float32))

    svc = ObjectRetrieval("logreg").fit(wide_train)
    assert svc.reducer is not None
    assert svc._expected_dim == 128
    assert svc.reducer.transform(wide_train.det_emb).shape[1] == 128

    # retrieval over the 256-D (internally reduced) space still ranks the asked
    # object first for most queries, and locate returns a well-formed hit.
    labels = np.array([d.label for d in wide_val.dets])
    answerable = _answerable(wide_val)
    r1 = []
    for q in answerable:
        qv = wide_val.query_emb[q.query_emb_idx]
        cand = np.array([i for i, d in enumerate(wide_val.dets) if d.t <= q.t_asked])
        scores = svc.score_candidates(qv, wide_val.det_emb[cand])
        r1.append(labels[cand[int(np.argmax(scores))]] == q.target_label)
    assert np.mean(r1) >= 0.7

    q0 = answerable[0]
    hit = svc.locate(wide_val.query_emb[q0.query_emb_idx], wide_val.dets,
                     wide_val.det_emb, q0.t_asked, query_text=q0.text)
    assert hit.found and hit.t <= q0.t_asked


def test_pca_roundtrip(tmp_path, data):
    train, _ = data
    red = PCAReducer(dim=32).fit(train.det_emb)
    Z = red.transform(train.det_emb)
    assert Z.shape == (train.det_emb.shape[0], 32)
    p = tmp_path / "pca.joblib"
    red.save(p)
    Z2 = PCAReducer().load(p).transform(train.det_emb)
    assert np.allclose(Z, Z2)
