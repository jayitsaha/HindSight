"""Precomputed Ego4D clip features, real or smoke.

Real: `omnivore_video_swinl/<clip_uid>.pt` -> (T, D). Smoke: a deterministic random
(T, D) per uid so the trainers run with no data. A smoke feature optionally carries
planted class structure (a label-dependent mean) so a head can actually learn
something during a smoke run and the loss visibly drops.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ..config import EGO4D_ROOT, FEATURE_DIM, FEATURE_FPS


def sec_to_idx(sec: float) -> int:
    return int(round(sec * FEATURE_FPS))


class FeatureStore:
    def __init__(self, smoke: bool = False, dim: int = FEATURE_DIM, root: Path | None = None):
        self.smoke = smoke
        self.dim = dim
        self.root = (root or EGO4D_ROOT) / "v2" / "omnivore_video_swinl"
        self._cache: dict[str, torch.Tensor] = {}

    def __call__(self, clip_uid: str) -> torch.Tensor:
        if clip_uid in self._cache:
            return self._cache[clip_uid]
        feat = self._smoke(clip_uid) if self.smoke else self._load(clip_uid)
        self._cache[clip_uid] = feat
        return feat

    def _load(self, clip_uid: str) -> torch.Tensor:
        path = self.root / f"{clip_uid}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"missing feature {path}. Download with:\n"
                f"  ego4d --output_directory $EGO4D_ROOT --datasets omnivore_video_swinl --version v2 -y"
            )
        feat = torch.load(path, map_location="cpu")
        if isinstance(feat, dict):           # some dumps wrap the tensor
            feat = next(iter(feat.values()))
        feat = torch.as_tensor(feat, dtype=torch.float32)
        return feat if feat.ndim == 2 else feat.reshape(1, -1)

    def _smoke(self, clip_uid: str) -> torch.Tensor:
        seed = int(hashlib.md5(clip_uid.encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        T = int(torch.randint(120, 600, (1,), generator=g).item())
        return torch.randn(T, self.dim, generator=g)
