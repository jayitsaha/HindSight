"""VSLNet-lite: natural-language moment retrieval over clip features.

A question is encoded from scratch (hashed token embeddings + a tiny transformer,
mean-pooled to a query vector); the clip feature sequence is encoded by its own
transformer; the query is fused into every timestep (concat + MLP) and two
per-timestep heads predict the answer span's start and end distributions. Loss is
cross-entropy of the start/end logits against the ground-truth boundary indices.
"""

from __future__ import annotations

import torch
from torch import nn

VOCAB = 30_000
NEG = -1e4  # mask fill for padded timesteps before softmax/argmax


class NLQSpanNet(nn.Module):
    def __init__(self, feat_dim: int, vocab: int = VOCAB, hidden: int = 256,
                 layers: int = 4, heads: int = 4, max_len: int = 2304, max_tok: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.tok_pos = nn.Parameter(torch.zeros(1, max_tok, hidden))
        qenc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1,
                                          batch_first=True, norm_first=True)
        self.qencoder = nn.TransformerEncoder(qenc, 2)

        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)

        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.start_head = nn.Linear(hidden, 1)
        self.end_head = nn.Linear(hidden, 1)

    def encode_query(self, tok: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
        # tok (B, L), tok_mask (B, L) True where padded -> query vector (B, H)
        L = tok.size(1)
        q = self.embed(tok) + self.tok_pos[:, :L]
        q = self.qencoder(q, src_key_padding_mask=tok_mask)
        valid = (~tok_mask).unsqueeze(-1).float()
        return (q * valid).sum(1) / valid.sum(1).clamp(min=1.0)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor,
                tok: torch.Tensor, tok_mask: torch.Tensor):
        # feat (B, T, D), mask (B, T) True where padded
        T = feat.size(1)
        x = self.proj(feat) + self.pos[:, :T]
        x = self.encoder(x, src_key_padding_mask=mask)        # (B, T, H)
        q = self.encode_query(tok, tok_mask)                  # (B, H)
        x = self.fuse(torch.cat([x, q.unsqueeze(1).expand(-1, T, -1)], dim=-1))
        start = self.start_head(x).squeeze(-1)                # (B, T)
        end = self.end_head(x).squeeze(-1)
        start = start.masked_fill(mask, NEG)
        end = end.masked_fill(mask, NEG)
        return start, end


def span_ce(start_logits: torch.Tensor, end_logits: torch.Tensor,
            s_idx: torch.Tensor, e_idx: torch.Tensor) -> torch.Tensor:
    ls = torch.nn.functional.cross_entropy(start_logits, s_idx)
    le = torch.nn.functional.cross_entropy(end_logits, e_idx)
    return 0.5 * (ls + le)
