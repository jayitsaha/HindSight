"""Visual Queries 2D (VQ2D) task: train VQ2DMatcher on clip features, score temporal
localization of the response window (tIoU / temporal-AP / Recall@1).

Real data comes from `v2/annotations/vq_{split}.json` (via ingest.ego4d.load_vq2d) +
omnivore features. For each visual query we use the feature at the crop frame as the
query embedding (we have no separate crop encoder here — documented simplification) and
the response-track frames as the GT response window. Smoke data is synthetic clips with
a PLANTED window where the frame features carry the query's signal, so the matcher
demonstrably learns on CPU in a few hundred steps (loss drops, R@1/tAP climb above chance).

SIMPLIFICATION: no box regression (see models/vq2d.py) — we localize *when*, not *where*.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...ingest.ego4d import load_vq2d
from ...metrics import average_precision, temporal_iou
from ..config import FEATURE_FPS, EGO4D_ROOT, TrainConfig
from ..data.features import FeatureStore, sec_to_idx
from ..models.vq2d import VQ2DMatcher, localize, match_bce

# Ego4D VQ2D annotation frame numbers are in the clip's canonical frame space; convert
# frame -> seconds -> feature index. 5 fps is the Ego4D VQ canonical annotation rate.
CLIP_FPS = 5.0


@dataclass
class VQClip:
    feat: torch.Tensor                  # (T, D) clip features
    query: torch.Tensor                 # (D,)   query-object embedding
    window: tuple[int, int]             # (start_idx, end_idx) response window, inclusive


class VQ2DTask:
    name = "vq2d"

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if cfg.smoke:
            self.dim = 128
            self.train = _smoke_clips(192, self.dim, seed=1)
            self.val = _smoke_clips(32, self.dim, seed=99)
        else:
            self.store = FeatureStore()
            self.dim = self.store.dim
            self.train = self._real("train")
            self.val = self._real("val")

    # --- Task protocol ---------------------------------------------------
    def build_model(self):
        return VQ2DMatcher(self.dim, max_len=self.cfg.max_len)

    def train_batches(self, cfg):
        g = torch.Generator().manual_seed(cfg.seed)
        while True:
            idx = torch.randint(0, len(self.train), (cfg.batch_size,), generator=g).tolist()
            yield _collate([self.train[i] for i in idx], cfg.max_len)

    def loss(self, model, batch, device):
        feat, query, mask, tgt = (batch[k].to(device) for k in ("feat", "query", "mask", "tgt"))
        logits = model(feat, query, mask)
        valid = ~mask
        loss = match_bce(logits[valid], tgt[valid])
        return loss, {"bce": loss.item()}

    @torch.no_grad()
    def evaluate(self, model, cfg):
        model.eval()
        scores, ious = [], []
        for clip in self.val:
            feat = clip.feat.unsqueeze(0).to(cfg.device)
            query = clip.query.unsqueeze(0).to(cfg.device)
            mask = torch.zeros(1, feat.size(1), dtype=torch.bool, device=cfg.device)
            prob = torch.sigmoid(model(feat, query, mask)[0]).cpu()
            s, e, sc = localize(prob)
            pred = (s / FEATURE_FPS, (e + 1) / FEATURE_FPS)      # span in seconds
            gt = clip.window
            gt_sec = (gt[0] / FEATURE_FPS, (gt[1] + 1) / FEATURE_FPS)
            iou = float(temporal_iou(pred, torch.tensor([[gt_sec[0], gt_sec[1]]]).numpy())[0])
            scores.append(sc)
            ious.append(iou)
        ious_t = torch.tensor(ious)
        out = {"tIoU": float(ious_t.mean())}
        for thr in (0.3, 0.5):
            hit = (ious_t >= thr)
            out[f"R@1@{thr}"] = float(hit.float().mean())
            # temporal AP: rank the single top span per query by its score, label = tIoU>=thr
            out[f"tAP@{thr}"] = average_precision(scores, hit.int().tolist())
        return out, out["R@1@0.5"]

    def _real(self, split):
        queries = load_vq2d(EGO4D_ROOT / "v2" / "annotations" / f"vq_{split}.json")
        clips = []
        for q in queries:
            try:
                feat = self.store(q.clip_uid)
            except FileNotFoundError:
                continue
            T = feat.size(0)
            frames = [f for f, _ in q.response_track]
            s = max(0, min(T - 1, sec_to_idx(min(frames) / CLIP_FPS)))
            e = max(0, min(T - 1, sec_to_idx(max(frames) / CLIP_FPS)))
            # query embedding = feature at the crop frame (no separate crop encoder here)
            qf = q.crop_frame if q.crop_frame >= 0 else q.query_frame
            qi = max(0, min(T - 1, sec_to_idx(qf / CLIP_FPS)))
            if e >= s:
                clips.append(VQClip(feat, feat[qi].clone(), (s, e)))
        if not clips:
            raise RuntimeError(f"no VQ2D clips with features for split={split}; check EGO4D_ROOT")
        return clips


# --- targets / batching ---------------------------------------------------
def _collate(clips, max_len):
    T = min(max_len, max(c.feat.size(0) for c in clips))
    B, D = len(clips), clips[0].feat.size(1)
    feat = torch.zeros(B, T, D)
    query = torch.zeros(B, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    tgt = torch.zeros(B, T)
    for b, c in enumerate(clips):
        t = min(T, c.feat.size(0))
        feat[b, :t] = c.feat[:t]
        query[b] = c.query
        mask[b, :t] = False
        s, e = min(c.window[0], t - 1), min(c.window[1], t - 1)
        if e >= s:
            tgt[b, s:e + 1] = 1.0
    return {"feat": feat, "query": query, "mask": mask, "tgt": tgt}


def _smoke_clips(n, dim, seed):
    """Synthetic clips with a planted response window: in that window the frame
    features carry the query vector's signal, so a matcher that learns query<->frame
    similarity can localize it. Each clip has its own query, so the model must learn a
    general matching function rather than memorize a fixed prototype."""
    g = torch.Generator().manual_seed(seed)
    clips = []
    for _ in range(n):
        T = int(torch.randint(60, 200, (1,), generator=g).item())
        query = torch.randn(dim, generator=g)
        feat = torch.randn(T, dim, generator=g) * 0.8
        s = int(torch.randint(5, T - 25, (1,), generator=g).item())
        e = min(T - 1, s + int(torch.randint(8, 25, (1,), generator=g).item()))
        feat[s:e + 1] += query * 0.6                 # plant the query signal in the window
        clips.append(VQClip(feat, query, (s, e)))
    return clips


TASK = VQ2DTask
