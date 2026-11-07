"""Smoke-mode tests for the PNR/OSCC trainer: it builds, a step cuts the loss, and
the learned PNR localization beats a random-frame baseline."""

import numpy as np
import torch

from hindsight.benchmarks.config import FEATURE_FPS, TrainConfig
from hindsight.benchmarks.registry import available
from hindsight.benchmarks.trainers.pnr import PNRTask


def _cfg():
    return TrainConfig(task="pnr").smoke_mode()


def test_registry_has_pnr():
    assert "pnr" in available()


def test_builds_and_trains():
    task = PNRTask(_cfg())
    model = task.build_model()
    assert isinstance(model, torch.nn.Module)

    cfg = _cfg()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = task.train_batches(cfg)
    first = task.loss(model, next(batches), "cpu")[0].item()
    for _ in range(60):
        opt.zero_grad()
        loss, _ = task.loss(model, next(batches), "cpu")
        loss.backward()
        opt.step()
    last = task.loss(model, next(batches), "cpu")[0].item()
    assert last < first


def test_evaluate_keys_and_beats_random():
    cfg = _cfg()
    task = PNRTask(cfg)
    model = task.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = task.train_batches(cfg)
    for _ in range(120):
        opt.zero_grad()
        loss, _ = task.loss(model, next(batches), "cpu")
        loss.backward()
        opt.step()

    metrics, monitor = task.evaluate(model, cfg)
    assert set(metrics) == {"pnr_err_mean", "pnr_err_median", "pnr_recall@1s",
                            "state_acc", "state_auroc"}
    assert monitor == -metrics["pnr_err_mean"]
    assert 0.0 <= metrics["pnr_recall@1s"] <= 1.0

    # random-frame baseline: expected |rand - gt| error in seconds
    rng = np.random.default_rng(0)
    rand_err = float(np.mean([
        abs(rng.integers(0, c.feat.size(0)) - c.pnr_idx) / FEATURE_FPS for c in task.val
    ]))
    assert metrics["pnr_err_mean"] < rand_err
    assert metrics["state_acc"] >= 0.8
