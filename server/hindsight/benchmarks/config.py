"""Shared config for the Ego4D benchmark trainers.

The heads here train on Ego4D's precomputed clip features (frozen backbone), so a
run is defined by: where the data lives, the feature width, and a few optimizer
knobs. `--smoke` swaps the real dataset for random tensors of the same shape so the
whole loop runs on CPU in seconds before anything hits a GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

EGO4D_ROOT = Path(os.environ.get("EGO4D_ROOT", str(Path.home() / "scratch/hindsight/ego4d")))
RUN_ROOT = Path(os.environ.get("HINDSIGHT_RUNS", str(Path.home() / "scratch/hindsight/runs")))
CKPT_ROOT = Path(os.environ.get("HINDSIGHT_CKPT", str(Path.home() / "scratch/hindsight/ckpt")))

FEATURE_DIM = int(os.environ.get("EGO4D_FEATURE_DIM", "1536"))  # omnivore_video_swinl
FEATURE_FPS = 1.875  # omnivore stride: features per second
MQ_CLASSES = 110     # Ego4D Moment Queries taxonomy


@dataclass
class TrainConfig:
    task: str
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    steps: int = 20_000
    warmup: int = 500
    eval_every: int = 1000
    ckpt_every: int = 1000
    amp: bool = True            # bf16 autocast on GPU
    max_len: int = 2304         # cap clip length (timesteps) for batching
    seed: int = 1
    smoke: bool = False
    device: str = "cuda"

    def smoke_mode(self) -> "TrainConfig":
        self.smoke = True
        self.device = "cpu"
        self.steps = 10
        self.warmup = 2
        self.eval_every = 5
        self.ckpt_every = 100
        self.amp = False
        self.batch_size = 4
        return self

    @property
    def ckpt_dir(self) -> Path:
        return CKPT_ROOT / self.task

    @property
    def run_dir(self) -> Path:
        return RUN_ROOT / self.task
