"""VQ2D (visual-queries temporal localization) smoke tests: the task builds, a few
train steps cut the loss, and evaluate localizes the planted response window well
above a random-model baseline — all on CPU with synthetic features, no download."""

import torch

from hindsight.benchmarks.config import TrainConfig
from hindsight.benchmarks.models.vq2d import localize
from hindsight.benchmarks.registry import available
from hindsight.benchmarks.trainers.vq2d import VQ2DTask


def _cfg():
    cfg = TrainConfig(task="vq2d").smoke_mode()
    cfg.steps = 200
    return cfg


def test_registered():
    assert "vq2d" in available()


def test_builds_and_shapes():
    task = VQ2DTask(_cfg())
    model = task.build_model()
    batch = next(task.train_batches(task.cfg))
    logits = model(batch["feat"], batch["query"], batch["mask"])
    assert logits.shape == batch["tgt"].shape           # (B, T) per-frame match logit


def test_train_step_reduces_loss():
    task = VQ2DTask(_cfg())
    model = task.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = task.train_batches(task.cfg)
    first = task.loss(model, next(batches), "cpu")[0].item()
    for _ in range(60):
        opt.zero_grad()
        loss, _ = task.loss(model, next(batches), "cpu")
        loss.backward()
        opt.step()
    assert loss.item() < first                          # loss drops with training


def test_evaluate_keys_and_beats_random():
    cfg = _cfg()
    task = VQ2DTask(cfg)

    # random untrained model is the chance baseline
    base_metrics, base_mon = task.evaluate(task.build_model(), cfg)
    assert set(base_metrics) == {"tIoU", "R@1@0.3", "tAP@0.3", "R@1@0.5", "tAP@0.5"}
    for v in base_metrics.values():
        assert 0.0 <= v <= 1.0

    model = task.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batches = task.train_batches(cfg)
    for _ in range(cfg.steps):
        opt.zero_grad()
        loss, _ = task.loss(model, next(batches), "cpu")
        loss.backward()
        opt.step()

    metrics, monitor = task.evaluate(model, cfg)
    assert monitor == metrics["R@1@0.5"]
    assert metrics["R@1@0.5"] > base_metrics["R@1@0.5"]
    assert metrics["tIoU"] > 0.5                        # localizes the window, well above chance
    assert metrics["R@1@0.3"] >= metrics["R@1@0.5"]


def test_localize_picks_contiguous_span():
    prob = torch.tensor([0.1, 0.1, 0.9, 0.8, 0.95, 0.2, 0.1])
    s, e, score = localize(prob, thresh=0.5)
    assert (s, e) == (2, 4) and score > 0.5
    # nothing above threshold -> fall back to the single best frame
    s, e, _ = localize(torch.tensor([0.1, 0.3, 0.2]), thresh=0.5)
    assert s == e == 1
