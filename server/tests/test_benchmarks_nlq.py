"""NLQ smoke task: it is discovered, builds, a few steps drop the loss, and
evaluate returns the expected span-retrieval keys."""

import torch

from hindsight.benchmarks.config import TrainConfig
from hindsight.benchmarks.registry import available
from hindsight.benchmarks.trainers.nlq import NLQTask


def _cfg():
    return TrainConfig(task="nlq").smoke_mode()


def test_registered():
    assert "nlq" in available()


def test_builds_and_shapes():
    cfg = _cfg()
    task = NLQTask(cfg)
    model = task.build_model()
    batch = next(task.train_batches(cfg))
    loss, logs = task.loss(model, batch, cfg.device)
    assert loss.requires_grad and "span" in logs
    start, end = model(batch["feat"], batch["mask"], batch["tok"], batch["tmask"])
    assert start.shape == batch["mask"].shape == end.shape


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    cfg = _cfg()
    task = NLQTask(cfg)
    model = task.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = task.train_batches(cfg)
    first = last = None
    for i in range(40):
        batch = next(batches)
        opt.zero_grad()
        loss, _ = task.loss(model, batch, cfg.device)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first


def test_evaluate_sane_keys():
    cfg = _cfg()
    task = NLQTask(cfg)
    model = task.build_model()
    metrics, monitor = task.evaluate(model, cfg)
    assert set(metrics) == {"R@1@0.3", "R@1@0.5", "mIoU"}
    assert monitor == metrics["R@1@0.5"]
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
