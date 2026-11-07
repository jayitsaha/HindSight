"""Natural Language Queries (NLQ) task: train NLQSpanNet to localize the answer
window for a question over a clip, score Recall@1 at tIoU thresholds and mean IoU.

Real data comes from `v2/annotations/nlq_{split}.json` (via ingest.load_nlq) +
omnivore features. Smoke data is synthetic clips whose planted answer window carries
a question-correlated signal, so the loop demonstrably learns on CPU in a few
hundred steps with no download.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from ...ingest.ego4d import load_nlq
from ..config import FEATURE_FPS, EGO4D_ROOT, TrainConfig
from ..data.features import FeatureStore, sec_to_idx
from ..models.nlq import VOCAB, NLQSpanNet, span_ce

MAX_TOK = 64
SMOKE_CONCEPTS = 8


@dataclass
class Item:
    feat: torch.Tensor      # (T, D)
    tokens: list[int]       # hashed question tokens
    s_idx: int              # answer start timestep
    e_idx: int              # answer end timestep


def _tok(text: str) -> list[int]:
    ids = [int(hashlib.md5(w.encode()).hexdigest(), 16) % VOCAB for w in text.lower().split()]
    return ids[:MAX_TOK] or [0]


class NLQTask:
    name = "nlq"

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if cfg.smoke:
            self.dim = 128
            self.train = _smoke_items(64, self.dim, seed=1)
            self.val = _smoke_items(24, self.dim, seed=99)
        else:
            self.store = FeatureStore()
            self.dim = self.store.dim
            self.train = self._real("train")
            self.val = self._real("val")

    # --- Task protocol ---------------------------------------------------
    def build_model(self):
        return NLQSpanNet(self.dim, max_len=self.cfg.max_len, max_tok=MAX_TOK)

    def train_batches(self, cfg):
        g = torch.Generator().manual_seed(cfg.seed)
        while True:
            idx = torch.randint(0, len(self.train), (cfg.batch_size,), generator=g).tolist()
            yield _collate([self.train[i] for i in idx], cfg.max_len)

    def loss(self, model, batch, device):
        feat, mask, tok, tmask, s, e = (batch[k].to(device) for k in
                                        ("feat", "mask", "tok", "tmask", "s", "e"))
        start, end = model(feat, mask, tok, tmask)
        loss = span_ce(start, end, s, e)
        return loss, {"span": loss.item()}

    @torch.no_grad()
    def evaluate(self, model, cfg):
        model.eval()
        ious = []
        for it in self.val:
            feat = it.feat.unsqueeze(0).to(cfg.device)
            mask = torch.zeros(1, feat.size(1), dtype=torch.bool, device=cfg.device)
            tok, tmask = _pad_tokens([it.tokens])
            start, end = model(feat, mask, tok.to(cfg.device), tmask.to(cfg.device))
            ps, pe = _decode(start[0], end[0])
            ious.append(_tiou((ps, pe), (it.s_idx, it.e_idx)))
        ious = torch.tensor(ious)
        m = {"R@1@0.3": float((ious >= 0.3).float().mean()),
             "R@1@0.5": float((ious >= 0.5).float().mean()),
             "mIoU": float(ious.mean())}
        return m, m["R@1@0.5"]

    def _real(self, split):
        items = []
        for q in load_nlq(EGO4D_ROOT / "v2" / "annotations" / f"nlq_{split}.json"):
            try:
                feat = self.store(q.clip_uid)
            except FileNotFoundError:
                continue
            T = feat.size(0)
            s = max(0, min(T - 1, sec_to_idx(q.t_start)))
            e = max(0, min(T - 1, sec_to_idx(q.t_end)))
            if e < s:
                s, e = e, s
            items.append(Item(feat, _tok(q.query), s, e))
        if not items:
            raise RuntimeError(f"no NLQ clips with features for split={split}; check EGO4D_ROOT")
        return items


# --- batching / decode ----------------------------------------------------
def _pad_tokens(toks):
    L = max(len(t) for t in toks)
    tok = torch.zeros(len(toks), L, dtype=torch.long)
    tmask = torch.ones(len(toks), L, dtype=torch.bool)
    for b, t in enumerate(toks):
        tok[b, :len(t)] = torch.tensor(t, dtype=torch.long)
        tmask[b, :len(t)] = False
    return tok, tmask


def _collate(items, max_len):
    T = min(max_len, max(it.feat.size(0) for it in items))
    B, D = len(items), items[0].feat.size(1)
    feat = torch.zeros(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    s = torch.zeros(B, dtype=torch.long)
    e = torch.zeros(B, dtype=torch.long)
    for b, it in enumerate(items):
        t = min(T, it.feat.size(0))
        feat[b, :t] = it.feat[:t]
        mask[b, :t] = False
        s[b] = min(it.s_idx, t - 1)
        e[b] = min(it.e_idx, t - 1)
    tok, tmask = _pad_tokens([it.tokens for it in items])
    return {"feat": feat, "mask": mask, "tok": tok, "tmask": tmask, "s": s, "e": e}


@torch.no_grad()
def _decode(start_logits, end_logits):
    # argmax start, then best end at or after start (VSLNet-style)
    ps = int(start_logits.argmax())
    pe = ps + int(end_logits[ps:].argmax())
    return ps, pe


def _tiou(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


TASK = NLQTask


# --- smoke loader ---------------------------------------------------------
def _smoke_items(n, dim, seed):
    # prototypes are shared across splits (fixed seed) so the question->signal
    # association the model learns on train actually transfers to val.
    protos = torch.randn(SMOKE_CONCEPTS, dim, generator=torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(seed)
    items = []
    for _ in range(n):
        T = int(torch.randint(60, 200, (1,), generator=g).item())
        feat = torch.randn(T, dim, generator=g) * 0.5
        c = int(torch.randint(0, SMOKE_CONCEPTS, (1,), generator=g).item())
        s = int(torch.randint(0, T - 20, (1,), generator=g).item())
        e = min(T - 1, s + int(torch.randint(8, 20, (1,), generator=g).item()))
        feat[s:e + 1] += protos[c]                       # plant a query-correlated signal
        items.append(Item(feat, _tok(f"where is concept {c}"), s, e))
    return items
