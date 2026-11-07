"""Local fallbacks used when NIM is unreachable or unconfigured."""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Sequence

import numpy as np
from PIL import Image

from .base import Box, VLMVerdict


class HashEmbedder:
    """Deterministic image/text vectors. Not semantic, but stable and aligned in dim.

    Only hit when open_clip weights are unavailable; synthetic data carries its
    own planted structure so retrieval still learns something meaningful.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._proj = np.random.default_rng(0).standard_normal((48 * 48 * 3, dim)).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True)

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        rows = []
        for im in images:
            a = np.asarray(im.convert("RGB").resize((48, 48)), dtype=np.float32).reshape(-1) / 255.0
            rows.append(a @ self._proj)
        return _l2(np.asarray(rows, dtype=np.float32))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            rows.append(v)
        return _l2(np.asarray(rows, dtype=np.float32))


class OpenClipEmbedder:
    """open_clip CLIP embedder. Weights load lazily on first embed (keeping
    startup and status fast); if the load fails it swaps in a HashEmbedder so
    offline runs degrade instead of dying."""

    def __init__(self, name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.name, self.pretrained = name, pretrained
        self.dim = 512
        self._ready = False
        self._fallback: HashEmbedder | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> bool:
        """One-time, thread-safe load. Returns False once we've fallen back to hashing."""
        if self._ready:
            return True
        if self._fallback is not None:
            return False
        with self._lock:
            if self._ready:
                return True
            if self._fallback is None:
                try:
                    self._load()
                except Exception:
                    self._fallback = HashEmbedder(self.dim)
        return self._fallback is None

    def _load(self) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.name, pretrained=self.pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.name)
        self.dim = self.model.text_projection.shape[1]
        self._ready = True

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not self._ensure():
            return self._fallback.embed_images(images)
        batch = self._torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        with self._torch.no_grad():
            feats = self.model.encode_image(batch)
        return _l2(feats.cpu().numpy().astype(np.float32))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not self._ensure():
            return self._fallback.embed_texts(texts)
        toks = self.tokenizer(list(texts))
        with self._torch.no_grad():
            feats = self.model.encode_text(toks)
        return _l2(feats.cpu().numpy().astype(np.float32))


def build_local_embedder() -> OpenClipEmbedder:
    # The embedder loads open_clip lazily and swaps itself to a HashEmbedder if
    # the weights can't be fetched, so the fallback happens at first use.
    return OpenClipEmbedder()


class WindowGrounder:
    """Multi-scale box proposals, scored later by the embedder against the prompt.

    A stand-in for Grounding DINO when no detector is hosted. Synthetic frames
    ship ground-truth boxes, so this only matters for real camera frames.
    """

    def __init__(self, scales=(0.3, 0.5, 0.7), stride_frac: float = 0.5):
        self.scales = scales
        self.stride_frac = stride_frac

    def detect(self, image: Image.Image, prompt: str) -> list[Box]:
        w, h = image.size
        boxes = []
        for s in self.scales:
            bw, bh = int(w * s), int(h * s)
            sx, sy = max(1, int(bw * self.stride_frac)), max(1, int(bh * self.stride_frac))
            for y0 in range(0, max(1, h - bh + 1), sy):
                for x0 in range(0, max(1, w - bw + 1), sx):
                    boxes.append(Box(prompt, 0.0, (x0, y0, min(w, x0 + bw), min(h, y0 + bh))))
        boxes.append(Box(prompt, 0.0, (0, 0, w, h)))
        return boxes


class HeuristicVLM:
    """Confirms or refuses from retrieval evidence alone, with templated phrasing."""

    def verify(self, images: Sequence[Image.Image], question: str, proposed: str) -> VLMVerdict:
        has_support = len(list(images)) > 0 and bool(proposed)
        return VLMVerdict(
            answer=proposed if has_support else "I don't know.",
            agrees=has_support,
            confidence=0.6 if has_support else 0.2,
        )


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-8, None)
