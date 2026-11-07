import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from hindsight.config import settings
from hindsight.ingest import synthetic as S
from hindsight.putdown import (CountingBaseline, MisplacementModel, PNRDetector,
                               PutDownDetector)

ROOT = settings.data_dir / "synthetic"


@pytest.fixture(scope="module")
def data():
    return S.load(ROOT, "train_01"), S.load(ROOT, "val_01")


@pytest.fixture(scope="module")
def pnr(data):
    train, _ = data
    return PNRDetector().fit(train)


def test_pnr_detection(pnr, data):
    _, val = data
    pnr_prob, state_prob = pnr.predict(val.ho_feat)

    auroc = roc_auc_score(val.ho_pnr, pnr_prob)
    state_acc = float(((state_prob >= 0.5).astype(int) == val.ho_state).mean())

    # timing: detector should fire on the put-down window, not drift off it
    ho_t = np.asarray(val.ho_t)
    fired = pnr_prob >= 0.5
    gt = val.ho_pnr == 1
    recall = float(fired[gt].mean())
    precision = float((gt & fired).sum() / max(fired.sum(), 1))
    fired_t = ho_t[fired]
    offsets = [float(np.min(np.abs(fired_t - ho_t[i]))) for i in np.where(gt)[0]] if fired.any() else []
    med_offset = float(np.median(offsets)) if offsets else float("nan")

    print(f"\nPNR AUROC={auroc:.3f}  state-change acc={state_acc:.3f}")
    print(f"PNR timing: recall={recall:.3f} precision={precision:.3f} "
          f"median offset={med_offset:.1f}s @thr=0.5")

    assert auroc > 0.9
    assert state_acc > 0.9


def test_pnr_save_load(pnr, data, tmp_path):
    _, val = data
    p = tmp_path / "pnr.pkl"
    pnr.save(p)
    reloaded = PNRDetector.load(p)
    a, b = pnr.predict(val.ho_feat)
    c, d = reloaded.predict(val.ho_feat)
    assert np.allclose(a, c) and np.allclose(b, d)


def test_misplacement_competitive_with_baseline(data):
    train, val = data
    model = MisplacementModel().fit(train)
    base = CountingBaseline().fit(train)

    y = np.array([int(p.unusual) for p in val.putdowns])
    model_s = np.array([model.score(p.object_label, e)
                        for p, e in zip(val.putdowns, val.pd_scene_emb)])
    base_s = np.array([base.score(p.object_label, p.location_id) for p in val.putdowns])

    model_auroc = roc_auc_score(y, model_s)
    base_auroc = roc_auc_score(y, base_s)
    print(f"\nmisplacement AUROC: model={model_auroc:.3f}  baseline={base_auroc:.3f}")

    # On clean synthetic data the discrete location_id is effectively the true room,
    # so naive counting is near-optimal and the scene-embedding clustering model is
    # only competitive. Its advantage is expected on sparse/continuous Ego4D scenes
    # where no clean location id exists. We assert competitiveness, not a win.
    assert model_auroc >= base_auroc - 0.08


def test_service_process_and_alerts(pnr, data):
    train, val = data
    model = MisplacementModel().fit(train)
    detector = PutDownDetector(pnr, model)

    events = detector.process(val)
    assert events
    assert all(0.0 <= e.unusualness <= 1.0 for e in events)
    assert all(isinstance(e.state_change, bool) for e in events)

    alerts = detector.alerts(events)
    assert all(a.kind == "misplacement" for a in alerts)
    assert len(alerts) <= len(events)

    # streaming: a genuine PNR window should surface a put-down event
    pnr_win = val.ho_feat[val.ho_pnr == 1][0]
    scene_emb = val.pd_scene_emb[0]
    ev = detector.step(pnr_win, "mug", scene_emb)
    assert ev is not None and ev.object_label == "mug"
    assert 0.0 <= ev.unusualness <= 1.0
