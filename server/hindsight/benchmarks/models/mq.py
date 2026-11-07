"""ActionFormer-lite: an anchor-free temporal action localizer over clip features.

A transformer encodes the feature sequence; each timestep predicts a class
distribution (actionness x class) and the distance to the segment's left/right
boundaries. Single scale — faithful enough to train and score temporal mAP on MQ,
without the full feature-pyramid machinery.
"""

from __future__ import annotations

import torch
from torch import nn


class MQLocalizer(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, hidden: int = 256,
                 layers: int = 4, heads: int = 4, max_len: int = 2304):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.cls_head = nn.Linear(hidden, num_classes)
        self.reg_head = nn.Linear(hidden, 2)  # softplus -> (left, right) distances in steps

    def forward(self, feat: torch.Tensor, mask: torch.Tensor):
        # feat (B, T, D), mask (B, T) True where padded
        T = feat.size(1)
        x = self.proj(feat) + self.pos[:, :T]
        x = self.encoder(x, src_key_padding_mask=mask)
        cls = self.cls_head(x)                       # (B, T, C) logits
        reg = torch.nn.functional.softplus(self.reg_head(x))  # (B, T, 2) >= 0
        return cls, reg


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5,
              gamma: float = 2.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    return (w * (1 - pt).pow(gamma) * ce).sum() / targets.sum().clamp(min=1.0)


def iou_1d_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target are (N, 2) = (left, right) distances; 1D IoU on the implied interval
    pl, pr = pred[:, 0], pred[:, 1]
    tl, tr = target[:, 0], target[:, 1]
    inter = torch.min(pl, tl) + torch.min(pr, tr)
    union = (pl + pr) + (tl + tr) - inter
    iou = inter / union.clamp(min=1e-6)
    return (1 - iou).mean()
