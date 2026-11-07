from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image

Boxed = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


@dataclass
class Box:
    label: str
    score: float
    xyxy: Boxed

    def crop(self, img: Image.Image) -> Image.Image:
        x0, y0, x1, y1 = (int(round(v)) for v in self.xyxy)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.width, max(x0 + 1, x1)), min(img.height, max(y0 + 1, y1))
        return img.crop((x0, y0, x1, y1))


@dataclass
class VLMVerdict:
    answer: str
    agrees: bool
    confidence: float


@runtime_checkable
class Embedder(Protocol):
    """Maps images and text into one shared space (CLIP-style)."""

    dim: int

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class Grounder(Protocol):
    """Open-vocabulary detector: text prompt -> boxes."""

    def detect(self, image: Image.Image, prompt: str) -> list[Box]: ...


@runtime_checkable
class VLM(Protocol):
    """Checks a proposed answer against frames and phrases the reply."""

    def verify(self, images: Sequence[Image.Image], question: str, proposed: str) -> VLMVerdict: ...
