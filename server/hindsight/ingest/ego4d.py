"""Ego4D v2 loaders: episodic memory (VQ2D / MQ / NLQ) and hands-and-objects PNR.

We don't ship the data; these parse the official annotation JSONs and the
per-clip precomputed-feature layout from the `ego4d` CLI download
(https://ego4d-data.org/docs/start-here/#download-data — see docs/EGO4D.md).
Field names follow the published v2 schema, with defensive fallbacks where
splits differ (fho_hands nests frames per action, fho_oscc-pnr is flat;
pnr/pre/post frames arrive either as ints or as {"frame": ...} dicts).
Missing files raise FileNotFoundError with the download command, never a
bare KeyError deep in JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DOWNLOAD_HINT = ("download with `ego4d --output_directory <EGO4D_ROOT> --datasets annotations "
                 "--version v2` and point EGO4D_ROOT at it (docs/EGO4D.md)")

Box = tuple[float, float, float, float]  # xyxy, matches synthetic.Det


@dataclass
class VisualQuery:
    video_uid: str
    clip_uid: str
    object_title: str
    query_frame: int
    crop_frame: int
    crop_xyxy: Box | None
    response_frame: int          # last appearance before the query frame
    response_xyxy: Box
    response_track: list[tuple[int, Box]]


@dataclass
class MomentSegment:
    video_uid: str
    clip_uid: str
    label: str                   # one of the 110 MQ action classes
    t_start: float
    t_end: float


@dataclass
class NLQuery:
    video_uid: str
    clip_uid: str
    query: str
    t_start: float               # answer window, clip-relative seconds
    t_end: float


@dataclass
class PNRClip:
    video_uid: str
    clip_uid: str
    pnr_frame: int
    pre_frame: int | None
    post_frame: int | None
    state_change: bool


@dataclass
class Ego4DPaths:
    """Standard `ego4d` CLI download layout under a root directory."""

    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> "Ego4DPaths":
        root = root or os.environ.get("EGO4D_ROOT")
        if not root:
            raise FileNotFoundError(f"no Ego4D root: set EGO4D_ROOT or pass one; {DOWNLOAD_HINT}")
        return cls(Path(root))

    @property
    def annotations(self) -> Path:
        return self.root / "v2" / "annotations"

    def vq2d(self, split: str = "val") -> Path:
        return self.annotations / f"vq_{split}.json"

    def mq(self, split: str = "val") -> Path:
        return self.annotations / f"moments_{split}.json"

    def nlq(self, split: str = "val") -> Path:
        return self.annotations / f"nlq_{split}.json"

    def pnr(self, split: str = "val") -> Path:
        return self.annotations / f"fho_oscc-pnr_{split}.json"

    def hands(self, split: str = "val") -> Path:
        return self.annotations / f"fho_hands_{split}.json"

    def features(self, model: str = "omnivore_video_swinl") -> Path:
        return self.root / "v2" / model


def load_vq2d(ann_path: str | Path) -> list[VisualQuery]:
    """Visual queries 2D: object crop + response track (last-seen frame and box)."""
    queries = []
    for video, clip, ann in _clip_annotations(_read_json(ann_path)):
        for qset in (ann.get("query_sets") or {}).values():
            if not qset.get("is_valid", True):
                continue
            track = sorted(((_frame_no(f), _box_xyxy(f)) for f in qset.get("response_track") or []),
                           key=lambda fb: fb[0])
            if not track:
                continue
            crop = qset.get("visual_crop") or {}
            queries.append(VisualQuery(
                video_uid=video.get("video_uid", ""),
                clip_uid=clip.get("clip_uid", ""),
                object_title=qset.get("object_title", ""),
                query_frame=int(qset.get("query_frame", -1)),
                crop_frame=_frame_no(crop) if crop else -1,
                crop_xyxy=_box_xyxy(crop) if crop else None,
                response_frame=track[-1][0],
                response_xyxy=track[-1][1],
                response_track=track,
            ))
    return queries


def load_mq(ann_path: str | Path) -> list[MomentSegment]:
    """Moment queries: labeled action segments with clip-relative (t_start, t_end)."""
    segments = []
    for video, clip, ann in _clip_annotations(_read_json(ann_path)):
        for m in ann.get("labels") or ann.get("moments") or []:
            t0 = _pick(m, "start_time", "moment_start_sec")
            t1 = _pick(m, "end_time", "moment_end_sec")
            if t0 is None or t1 is None or "label" not in m:
                continue
            segments.append(MomentSegment(
                video_uid=video.get("video_uid", ""),
                clip_uid=clip.get("clip_uid", ""),
                label=m["label"], t_start=float(t0), t_end=float(t1),
            ))
    return segments


def load_nlq(ann_path: str | Path) -> list[NLQuery]:
    """Natural-language queries with their answer window. Unannotated test
    queries (no window) are skipped."""
    queries = []
    for video, clip, ann in _clip_annotations(_read_json(ann_path)):
        for lq in ann.get("language_queries") or []:
            text = lq.get("query")
            t0, t1 = lq.get("clip_start_sec"), lq.get("clip_end_sec")
            if not text or t0 is None or t1 is None:
                continue
            queries.append(NLQuery(
                video_uid=video.get("video_uid", ""),
                clip_uid=clip.get("clip_uid", ""),
                query=text, t_start=float(t0), t_end=float(t1),
            ))
    return queries


def load_pnr(ann_path: str | Path) -> list[PNRClip]:
    """Hands-and-objects point-of-no-return: PNR frame plus pre/post state frames.
    Accepts both fho_oscc-pnr (flat clips) and fho_hands (frames nested per action)."""
    data = _read_json(ann_path)
    clips = []
    for clip in data.get("clips") or []:
        for fr in clip.get("frames") or [clip]:
            pnr = _frame_field(fr, "pnr_frame", "parent_pnr_frame")
            if pnr is None:
                continue
            clips.append(PNRClip(
                video_uid=clip.get("video_uid", ""),
                clip_uid=clip.get("clip_uid") or clip.get("clip_id", ""),
                pnr_frame=pnr,
                pre_frame=_frame_field(fr, "pre_frame", "pre_45"),
                post_frame=_frame_field(fr, "post_frame"),
                state_change=bool(_pick(fr, "state_change", default=_pick(clip, "state_change", default=True))),
            ))
    return clips


def read_features(feature_dir: str | Path, clip_uid: str) -> np.ndarray:
    """Per-clip precomputed features (omnivore / InternVideo2) as (T, D) float32.
    Ego4D ships torch tensors ({uid}.pt); we also take {uid}.npy."""
    d = Path(feature_dir)
    for ext in (".pt", ".npy"):
        p = d / f"{clip_uid}{ext}"
        if not p.exists():
            continue
        if ext == ".pt":
            import torch
            feat = torch.load(p, map_location="cpu")
            arr = feat.detach().numpy() if hasattr(feat, "detach") else np.asarray(feat)
        else:
            arr = np.load(p)
        arr = np.asarray(arr, dtype=np.float32)
        return arr[None, :] if arr.ndim == 1 else arr
    raise FileNotFoundError(
        f"no features for clip {clip_uid!r} under {d} (tried .pt, .npy); download with "
        f"`ego4d --output_directory <EGO4D_ROOT> --datasets omnivore_video_swinl --version v2`")


def _read_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ego4D annotation file {path} not found; {DOWNLOAD_HINT}")
    return json.loads(path.read_text())


def _clip_annotations(data: dict):
    """Yield (video, clip, annotation) for the shared videos->clips->annotations shape."""
    for video in data.get("videos") or []:
        for clip in video.get("clips") or []:
            for ann in clip.get("annotations") or []:
                yield video, clip, ann


def _pick(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _frame_no(d: dict) -> int:
    return int(_pick(d, "frame_number", "frame", default=-1))


def _frame_field(d: dict, *keys) -> int | None:
    v = _pick(d, *keys)
    if v is None:
        return None
    return _frame_no(v) if isinstance(v, dict) else int(v)


def _box_xyxy(d: dict) -> Box:
    x, y = float(d.get("x", 0)), float(d.get("y", 0))
    return (x, y, x + float(d.get("width", 0)), y + float(d.get("height", 0)))
