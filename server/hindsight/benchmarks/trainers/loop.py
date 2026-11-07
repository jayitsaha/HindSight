"""Task-agnostic training loop: warmup+cosine LR, bf16 autocast, periodic eval, and
checkpoint/resume so a job killed at the 8h wall limit picks up where it left off.

A task supplies build_model / train_batches / loss / evaluate; the loop owns the
optimizer, schedule, logging and checkpoints.
"""

from __future__ import annotations

import json
import math
import time
from typing import Iterator, Protocol

import torch
from torch import nn

from ..config import TrainConfig


class Task(Protocol):
    name: str

    def build_model(self) -> nn.Module: ...
    def train_batches(self, cfg: TrainConfig) -> Iterator[dict]: ...
    def loss(self, model: nn.Module, batch: dict, device: str) -> tuple[torch.Tensor, dict]: ...
    def evaluate(self, model: nn.Module, cfg: TrainConfig) -> tuple[dict, float]: ...


def _lr_scale(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _resume(model, opt, cfg: TrainConfig) -> tuple[int, float]:
    path = cfg.ckpt_dir / "latest.pt"
    if not path.exists():
        return 0, -math.inf
    ckpt = torch.load(path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    print(f"resumed {cfg.task} from step {ckpt['step']} (best {ckpt['best']:.4f})")
    return ckpt["step"], ckpt["best"]


def _save(model, opt, step: int, best: float, cfg: TrainConfig, tag: str) -> None:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "best": best, "task": cfg.task},
               cfg.ckpt_dir / f"{tag}.pt")


def run_training(task: Task, cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    model = task.build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step, best = _resume(model, opt, cfg)
    batches = task.train_batches(cfg)
    use_amp = cfg.amp and device == "cuda"

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    log = (cfg.run_dir / "metrics.jsonl").open("a")
    metrics: dict = {}
    t0 = time.monotonic()

    while step < cfg.steps:
        model.train()
        batch = next(batches)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * _lr_scale(step, cfg.warmup, cfg.steps)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, logs = task.loss(model, batch, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 50 == 0:
            rate = step / max(1e-9, time.monotonic() - t0)
            print(f"[{cfg.task}] step {step}/{cfg.steps} loss {loss.item():.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e} ({rate:.1f} it/s) " +
                  " ".join(f"{k}={v:.3f}" for k, v in logs.items()))

        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics, monitor = task.evaluate(model, cfg)
            log.write(json.dumps({"step": step, **metrics}) + "\n")
            log.flush()
            print(f"[{cfg.task}] eval @ {step}: {metrics}")
            if monitor >= best:
                best = monitor
                _save(model, opt, step, best, cfg, "best")
        if step % cfg.ckpt_every == 0:
            _save(model, opt, step, best, cfg, "latest")

    _save(model, opt, step, best, cfg, "latest")
    log.close()
    return metrics
