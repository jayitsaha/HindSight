"""VQ2D-lite: a per-frame "is the query object visible now" match head.

Given a query object embedding and a clip's precomputed per-frame features, score
every frame by how well it matches the query, then localize the response window
(the last contiguous high-score span). We condition the clip features on the query
with FiLM (query -> per-channel scale/shift) and a learned similarity MLP, optionally
after a small temporal transformer that lets a frame see its neighbours.

SIMPLIFICATION (honest note): full VQ2D is detection + tracking + box regression —
for each response frame you must also predict the object's bounding box. That is too
heavy for a CPU-trainable course model, so we solve only the TEMPORAL-LOCALIZATION
core: *when* does the object most recently appear, not *where* in the frame. Box
regression is a documented extension (add a 4-dim reg head off the matched frames and
an L1/GIoU term); the matching + windowing trained here is the real, load-bearing part.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class VQ2DMatcher(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, layers: int = 2,
                 heads: int = 4, max_len: int = 2304, use_transformer: bool = True):
        super().__init__()
        # ONE projection for both clip frames and the query: the match score is a
        # dot-product, so query and frame must land in the same space. A shared encoder
        # makes similarity the path of least resistance and the matcher generalizes to
        # unseen queries instead of memorizing training windows by position.
        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        self.use_transformer = use_transformer
        if use_transformer:
            enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1,
                                             batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers)
        self.film = nn.Linear(hidden, 2 * hidden)                 # query -> (gamma, beta)
        self.score = nn.Sequential(                               # [frame, query, frame*query] -> logit
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.scale = hidden ** 0.5

    def forward(self, feat: torch.Tensor, query: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # feat (B, T, D), query (B, D), mask (B, T) True where padded -> logit (B, T)
        T = feat.size(1)
        x = self.proj(feat) + self.pos[:, :T]
        if self.use_transformer:
            x = self.encoder(x, src_key_padding_mask=mask)
        q = self.proj(query)                                      # (B, H), shared encoder
        gamma, beta = self.film(q).chunk(2, dim=-1)               # (B, H) each
        xm = x * (1 + gamma[:, None]) + beta[:, None]             # FiLM-condition each frame
        qe = q[:, None].expand_as(xm)                             # broadcast query over time
        sim = (xm * qe).sum(-1) / self.scale                      # (B, T) cosine-ish similarity
        mlp = self.score(torch.cat([xm, qe, xm * qe], dim=-1)).squeeze(-1)
        return sim + mlp                                          # (B, T) per-frame match logit


def match_bce(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.75,
              gamma: float = 2.0) -> torch.Tensor:
    """Focal BCE over valid frames: positives (inside the GT window) are rare, so we
    up-weight them (alpha) and focus on hard examples (gamma), normalising by positives."""
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    return (w * (1 - pt).pow(gamma) * ce).sum() / targets.sum().clamp(min=1.0)


@torch.no_grad()
def localize(prob: torch.Tensor, thresh: float = 0.5) -> tuple[int, int, float]:
    """Pick the top-scoring contiguous span above `thresh` (the predicted response
    window). Falls back to the single argmax frame if nothing clears the threshold."""
    above = (prob >= thresh).tolist()
    best, cur_s = None, None
    for t, hi in enumerate(above):
        if hi and cur_s is None:
            cur_s = t
        if cur_s is not None and (not hi or t == len(above) - 1):
            e = t if hi else t - 1
            s = float(prob[cur_s:e + 1].sum())
            if best is None or s > best[0]:
                best = (s, cur_s, e)
            cur_s = None
    if best is None:
        t = int(torch.argmax(prob))
        return t, t, float(prob[t])
    _, s, e = best
    return s, e, float(prob[s:e + 1].mean())
