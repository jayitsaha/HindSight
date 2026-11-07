"""PNR/OSCC head: localize the point-of-no-return frame and flag a state change.

A transformer encodes the feature sequence; a per-timestep head scores each frame
as the PNR (argmax = predicted frame) and a pooled head emits one clip-level
state-change logit. Trained with a soft (Gaussian) cross-entropy over the PNR frame
plus BCE on the state-change label.
"""

from __future__ import annotations

import torch
from torch import nn


class PNRNet(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, layers: int = 4,
                 heads: int = 4, max_len: int = 2304):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.pnr_head = nn.Linear(hidden, 1)
        self.state_head = nn.Linear(hidden, 1)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor):
        # feat (B, T, D), mask (B, T) True where padded
        T = feat.size(1)
        x = self.proj(feat) + self.pos[:, :T]
        x = self.encoder(x, src_key_padding_mask=mask)
        pnr = self.pnr_head(x).squeeze(-1).masked_fill(mask, float("-inf"))  # (B, T)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        state = self.state_head(pooled).squeeze(-1)                          # (B,)
        return pnr, state


def pnr_soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # soft cross-entropy: -sum(target * log_softmax); padded steps carry no target
    # and their -inf logits vanish from log_softmax, so zero them before the product.
    lp = torch.nan_to_num(torch.log_softmax(logits, dim=1), neginf=0.0)
    return -(target * lp).sum(1).mean()
