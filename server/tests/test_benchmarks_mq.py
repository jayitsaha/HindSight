import torch

from hindsight.benchmarks.config import TrainConfig
from hindsight.benchmarks.registry import available
from hindsight.benchmarks.trainers.mq import MQTask


def _smoke_task():
    cfg = TrainConfig(task="mq").smoke_mode()
    return cfg, MQTask(cfg)


def test_mq_is_discovered():
    assert "mq" in available()


def test_mq_trains_and_evaluates():
    cfg, task = _smoke_task()
    model = task.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = task.train_batches(cfg)

    first, last = None, None
    for i in range(80):
        batch = next(batches)
        opt.zero_grad()
        loss, _ = task.loss(model, batch, "cpu")
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first  # the loop actually learns on the planted signal

    metrics, monitor = task.evaluate(model, cfg)
    assert "mAP" in metrics and 0.0 <= metrics["mAP"] <= 1.0
    assert monitor == metrics["mAP"]


def test_mq_decode_shapes():
    cfg, task = _smoke_task()
    model = task.build_model()
    clip = task.val[0]
    feat = clip.feat.unsqueeze(0)
    mask = torch.zeros(1, feat.size(1), dtype=torch.bool)
    cls, reg = model(feat, mask)
    assert cls.shape[-1] == task.num_classes and reg.shape[-1] == 2
