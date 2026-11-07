"""Load the materialized synthetic splits, generating them if they're missing.

Same seeds/render flags the CLI's `gen` command uses, so a split built here is
byte-identical to one built with `python -m hindsight.cli gen`.
"""

from __future__ import annotations

from ..config import settings
from ..ingest import synthetic as S

# video_id -> (seed, render); mirrors cli._gen
_SPECS = {"train_01": 7, "val_01": 11, "demo_01": 23}
_RENDER = {"demo_01"}


def synthetic_root():
    return settings.data_dir / "synthetic"


def load(video_id: str, minutes: float = 30.0) -> S.SyntheticVideo:
    root = synthetic_root()
    if not (root / video_id / "meta.json").exists():
        generate(video_id, minutes)
    return S.load(root, video_id)


def generate(video_id: str, minutes: float = 30.0) -> S.SyntheticVideo:
    render = video_id in _RENDER
    if render:
        minutes = min(minutes, 20.0)
    v = S.generate(video_id, minutes=minutes, seed=_SPECS[video_id], render=render)
    v.save(synthetic_root(), render=render)
    return v


def load_split(minutes: float = 30.0):
    return load("train_01", minutes), load("val_01", minutes)
