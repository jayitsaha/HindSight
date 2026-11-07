"""PNR/OSCC task: localize the point-of-no-return frame and predict clip-level
state change on Ego4D's hands-and-objects features.

Real data comes from `v2/annotations/fho_oscc-pnr_{split}.json` + omnivore features;
PNR frame numbers are mapped through the clip's video fps to a feature index. Smoke
data plants a feature spike at the PNR frame and a direction shift tied to the
state-change label, so the loop demonstrably learns (loss drops, PNR error shrinks)
with no download.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ...ingest.ego4d import load_pnr
from ...metrics import auroc, pnr_timing_error
from ..config import EGO4D_ROOT, FEATURE_FPS, TrainConfig
from ..data.features import FeatureStore, sec_to_idx
from ..models.pnr import PNRNet, pnr_soft_ce

VIDEO_FPS = 30.0        # Ego4D canonical clip fps (frame numbers -> seconds)
SOFT_SIGMA = 1.0        # Gaussian width (feature steps) of the soft PNR target


@dataclass
class PClip:
    feat: torch.Tensor   # (T, D)
    pnr_idx: int         # feature index of the PNR frame
    state: float         # 1.0 if the object's state changed


class PNRTask:
    name = "pnr"

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if cfg.smoke:
            self.dim = 128
            self.train = _smoke_clips(48, self.dim, seed=1)
            self.val = _smoke_clips(16, self.dim, seed=99)
        else:
            self.store = FeatureStore()
            self.dim = self.store.dim
            self.train = self._real("train")
            self.val = self._real("val")

    # --- Task protocol ---------------------------------------------------
    def build_model(self):
        return PNRNet(self.dim, max_len=self.cfg.max_len)

    def train_batches(self, cfg):
        g = torch.Generator().manual_seed(cfg.seed)
        while True:
            idx = torch.randint(0, len(self.train), (cfg.batch_size,), generator=g).tolist()
            yield _collate([self.train[i] for i in idx], cfg.max_len)

    def loss(self, model, batch, device):
        feat, mask, target, state = (batch[k].to(device) for k in
                                     ("feat", "mask", "target", "state"))
        pnr, state_logit = model(feat, mask)
        lp = pnr_soft_ce(pnr, target)
        ls = F.binary_cross_entropy_with_logits(state_logit, state)
        return lp + ls, {"pnr": lp.item(), "state": ls.item()}

    @torch.no_grad()
    def evaluate(self, model, cfg):
        model.eval()
        pred_t, gt_t, within, scores, labels = [], [], [], [], []
        for clip in self.val:
            feat = clip.feat.unsqueeze(0).to(cfg.device)
            mask = torch.zeros(1, feat.size(1), dtype=torch.bool, device=cfg.device)
            pnr, state_logit = model(feat, mask)
            pi = int(pnr[0].argmax())
            pred_t.append(pi / FEATURE_FPS)
            gt_t.append(clip.pnr_idx / FEATURE_FPS)
            within.append(abs(pi - clip.pnr_idx) / FEATURE_FPS <= 1.0)
            scores.append(torch.sigmoid(state_logit[0]).item())
            labels.append(int(clip.state))
        err = pnr_timing_error(pred_t, gt_t)
        acc = float(np.mean([(s >= 0.5) == bool(l) for s, l in zip(scores, labels)]))
        metrics = {
            "pnr_err_mean": err["mean"],
            "pnr_err_median": err["median"],
            "pnr_recall@1s": float(np.mean(within)),
            "state_acc": acc,
            "state_auroc": auroc(scores, labels),
        }
        return metrics, -err["mean"]   # higher (less error) is better

    def _real(self, split):
        path = EGO4D_ROOT / "v2" / "annotations" / f"fho_oscc-pnr_{split}.json"
        clips = []
        for pc in load_pnr(path):
            try:
                feat = self.store(pc.clip_uid)
            except FileNotFoundError:
                continue
            idx = max(0, min(feat.size(0) - 1, sec_to_idx(pc.pnr_frame / VIDEO_FPS)))
            clips.append(PClip(feat, idx, float(pc.state_change)))
        if not clips:
            raise RuntimeError(f"no PNR clips with features for split={split}; check EGO4D_ROOT")
        return clips


# --- targets / batching ---------------------------------------------------
def _collate(clips, max_len):
    T = min(max_len, max(c.feat.size(0) for c in clips))
    B, D = len(clips), clips[0].feat.size(1)
    feat = torch.zeros(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    target = torch.zeros(B, T)
    state = torch.zeros(B)
    for b, c in enumerate(clips):
        t = min(T, c.feat.size(0))
        feat[b, :t] = c.feat[:t]
        mask[b, :t] = False
        idx = min(c.pnr_idx, t - 1)
        pos = torch.arange(t, dtype=torch.float32)
        soft = torch.exp(-0.5 * ((pos - idx) / SOFT_SIGMA) ** 2)
        target[b, :t] = soft / soft.sum()   # Gaussian around the PNR frame
        state[b] = c.state
    return {"feat": feat, "mask": mask, "target": target, "state": state}


TASK = PNRTask


def _smoke_clips(n, dim, seed):
    g = torch.Generator().manual_seed(seed)
    # fixed across splits so the state-change signature is the same signal to learn
    direction = torch.randn(dim, generator=torch.Generator().manual_seed(0))
    clips = []
    for _ in range(n):
        T = int(torch.randint(60, 200, (1,), generator=g).item())
        feat = torch.randn(T, dim, generator=g) * 0.5
        pnr = int(torch.randint(5, T - 5, (1,), generator=g).item())
        feat[pnr] += 4.0                         # spike marks the PNR frame
        state = int(torch.randint(0, 2, (1,), generator=g).item())
        if state:
            feat[pnr:] += direction * 0.8        # post-PNR shift iff state changed
        clips.append(PClip(feat, pnr, float(state)))
    return clips
