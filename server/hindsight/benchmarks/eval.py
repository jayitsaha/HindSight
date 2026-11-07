"""Evaluate a trained checkpoint on the val split and write metrics.json.

    python -m hindsight.benchmarks.eval mq --checkpoint .../ckpt/mq/best.pt
"""

from __future__ import annotations

import argparse
import json

import torch

from .config import TrainConfig
from .registry import available

TASKS = available()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="hindsight.benchmarks.eval")
    p.add_argument("task", choices=sorted(TASKS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    cfg = TrainConfig(task=args.task)
    if args.smoke:
        cfg.smoke_mode()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    task = TASKS[args.task](cfg)
    model = task.build_model().to(cfg.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=cfg.device)["model"])
    metrics, monitor = task.evaluate(model, cfg)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{args.task}: {metrics}  (monitor {monitor:.4f})")


if __name__ == "__main__":
    main()
