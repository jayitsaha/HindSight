"""Entry point: python -m hindsight.benchmarks.train <task> [--smoke] [--steps N]

Resume is automatic — a run reads ckpt_dir/latest.pt if present, so resubmitting a
job after the 8h wall limit continues the same run.
"""

from __future__ import annotations

import argparse

import torch

from .config import TrainConfig
from .registry import available
from .trainers.loop import run_training

TASKS = available()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="hindsight.benchmarks.train")
    p.add_argument("task", choices=sorted(TASKS))
    p.add_argument("--smoke", action="store_true", help="10 CPU steps on synthetic features")
    p.add_argument("--steps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    args = p.parse_args(argv)

    cfg = TrainConfig(task=args.task)
    if args.smoke:
        cfg.smoke_mode()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("no CUDA; falling back to CPU")
        cfg.device, cfg.amp = "cpu", False

    task = TASKS[args.task](cfg)
    metrics = run_training(task, cfg)
    print("final:", metrics)


if __name__ == "__main__":
    main()
