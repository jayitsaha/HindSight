"""Moment Queries (MQ) task: train MQLocalizer on clip features, score temporal mAP.

Real data comes from `v2/annotations/moments_{split}.json` + omnivore features. Smoke
data is synthetic clips with planted class-biased segments so the loop demonstrably
learns (loss drops, mAP climbs) with no download.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from ...metrics import temporal_map
from ..config import FEATURE_FPS, EGO4D_ROOT, TrainConfig
from ..data.features import FeatureStore, sec_to_idx
from ..models.mq import MQLocalizer, focal_bce, iou_1d_loss

SMOKE_CLASSES = 5


@dataclass
class Clip:
    feat: torch.Tensor                 # (T, D)
    segs: list[tuple[int, int, int]]   # (start_idx, end_idx, label_id)


class MQTask:
    name = "mq"

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if cfg.smoke:
            self.num_classes = SMOKE_CLASSES
            self.dim = 128
            self.train = _smoke_clips(48, self.dim, self.num_classes, seed=1)
            self.val = _smoke_clips(16, self.dim, self.num_classes, seed=99)
        else:
            self.store = FeatureStore()
            self.dim = self.store.dim
            self.label2id = _label_vocab(EGO4D_ROOT, "train")
            self.num_classes = len(self.label2id)
            self.train = self._real("train")
            self.val = self._real("val")

    # --- Task protocol ---------------------------------------------------
    def build_model(self):
        return MQLocalizer(self.dim, self.num_classes, max_len=self.cfg.max_len)

    def train_batches(self, cfg):
        g = torch.Generator().manual_seed(cfg.seed)
        while True:
            idx = torch.randint(0, len(self.train), (cfg.batch_size,), generator=g).tolist()
            yield _collate([self.train[i] for i in idx], cfg.max_len, self.num_classes)

    def loss(self, model, batch, device):
        feat, mask, cls_t, reg_t, pos = (batch[k].to(device) for k in
                                         ("feat", "mask", "cls", "reg", "pos"))
        cls, reg = model(feat, mask)
        valid = ~mask
        lc = focal_bce(cls[valid], cls_t[valid])
        lr = iou_1d_loss(reg[pos], reg_t[pos]) if pos.any() else reg.sum() * 0.0
        loss = lc + lr
        return loss, {"cls": lc.item(), "reg": float(lr.detach())}

    @torch.no_grad()
    def evaluate(self, model, cfg):
        model.eval()
        preds, gts = [], []
        for ci, clip in enumerate(self.val):
            feat = clip.feat.unsqueeze(0).to(cfg.device)
            mask = torch.zeros(1, feat.size(1), dtype=torch.bool, device=cfg.device)
            cls, reg = model(feat, mask)
            preds += _decode(cls[0], reg[0], ci)
            gts += [(s / FEATURE_FPS, e / FEATURE_FPS, lab, 1.0, ci) for s, e, lab in clip.segs]
        m = temporal_map([p[:4] for p in _per_video(preds)], [g[:4] for g in _per_video(gts)])
        return {"mAP": m["mAP"], **{f"mAP@{k}": v for k, v in m.items() if k != "mAP"}}, m["mAP"]

    def _real(self, split):
        moments = _load_moments(EGO4D_ROOT, split)
        clips = []
        for clip_uid, segs in moments.items():
            try:
                feat = self.store(clip_uid)
            except FileNotFoundError:
                continue
            T = feat.size(0)
            idx_segs = [(max(0, sec_to_idx(s)), min(T - 1, sec_to_idx(e)), self.label2id[lab])
                        for lab, s, e in segs if lab in self.label2id and sec_to_idx(e) > sec_to_idx(s)]
            if idx_segs:
                clips.append(Clip(feat, idx_segs))
        if not clips:
            raise RuntimeError(f"no MQ clips with features for split={split}; check EGO4D_ROOT")
        return clips


# --- targets / batching ---------------------------------------------------
def _collate(clips, max_len, num_classes):
    T = min(max_len, max(c.feat.size(0) for c in clips))
    B, D, C = len(clips), clips[0].feat.size(1), num_classes
    feat = torch.zeros(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    cls = torch.zeros(B, T, C)
    reg = torch.zeros(B, T, 2)
    pos = torch.zeros(B, T, dtype=torch.bool)
    for b, c in enumerate(clips):
        t = min(T, c.feat.size(0))
        feat[b, :t] = c.feat[:t]
        mask[b, :t] = False
        for s, e, lab in c.segs:
            s, e = min(s, t - 1), min(e, t - 1)
            if e <= s:
                continue
            # center sampling: supervise only the central part of the segment, where
            # the distance-to-boundary regression is least ambiguous (ActionFormer-style)
            center = (s + e) / 2
            half = max(1.0, 0.25 * (e - s))
            lo, hi = max(s, int(center - half)), min(e, int(center + half))
            cls[b, lo:hi + 1, lab] = 1.0
            for ti in range(lo, hi + 1):
                reg[b, ti] = torch.tensor([ti - s, e - ti], dtype=torch.float32)
                pos[b, ti] = True
    return {"feat": feat, "mask": mask, "cls": cls, "reg": reg, "pos": pos}


@torch.no_grad()
def _decode(cls, reg, video_id, topk=100):
    # keep the top-scoring timesteps as candidates (mAP ranks by score, so we don't
    # hard-threshold); each becomes a segment via its boundary regression, then NMS.
    prob = torch.sigmoid(cls)                      # (T, C)
    score, label = prob.max(dim=1)
    order = torch.argsort(score, descending=True)[:topk]
    out = []
    for t in order.tolist():
        left, right = reg[t].tolist()
        s = (t - left) / FEATURE_FPS
        e = (t + right) / FEATURE_FPS
        if e > s:
            out.append((s, e, int(label[t]), float(score[t]), video_id))
    return _soft_nms(out)


def _soft_nms(segs, iou_thr=0.4):
    kept = []
    for s in sorted(segs, key=lambda r: -r[3]):
        if all(s[2] != k[2] or _tiou(s, k) < iou_thr for k in kept):
            kept.append(s)
    return kept


def _tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _per_video(items):
    # temporal_map matches within a video; encode video id into the label space-free
    # tuple by keeping 5th element — our temporal_map treats each list globally, so we
    # offset times per video to avoid cross-video matches.
    out = []
    for s, e, lab, sc, vid in items:
        off = vid * 1e6
        out.append((s + off, e + off, lab, sc))
    return out


# --- annotation / smoke loaders ------------------------------------------
def _load_moments(root, split):
    path = root / "v2" / "annotations" / f"moments_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; download MQ annotations (see docs/PACE.md)")
    data = json.loads(path.read_text())
    out: dict[str, list] = {}
    for video in data.get("videos", []):
        for clip in video.get("clips", []):
            uid = clip["clip_uid"]
            segs = []
            for ann in clip.get("annotations", []):
                for lab in ann.get("labels", []):
                    s = lab.get("start_time", lab.get("video_start_time"))
                    e = lab.get("end_time", lab.get("video_end_time"))
                    name = lab.get("label")
                    if s is not None and e is not None and name:
                        segs.append((name, float(s), float(e)))
            if segs:
                out.setdefault(uid, []).extend(segs)
    return out


def _label_vocab(root, split):
    labels = sorted({lab for segs in _load_moments(root, split).values() for lab, _, _ in segs})
    return {name: i for i, name in enumerate(labels)}


TASK = MQTask


def _smoke_clips(n, dim, num_classes, seed):
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(num_classes, dim, generator=g)
    clips = []
    for _ in range(n):
        T = int(torch.randint(60, 200, (1,), generator=g).item())
        feat = torch.randn(T, dim, generator=g) * 0.5
        segs = []
        for _ in range(int(torch.randint(1, 4, (1,), generator=g).item())):
            lab = int(torch.randint(0, num_classes, (1,), generator=g).item())
            s = int(torch.randint(0, T - 15, (1,), generator=g).item())
            e = min(T - 1, s + int(torch.randint(8, 30, (1,), generator=g).item()))
            feat[s:e + 1] += protos[lab]            # plant a learnable class signal
            segs.append((s, e, lab))
        clips.append(Clip(feat, segs))
    return clips
