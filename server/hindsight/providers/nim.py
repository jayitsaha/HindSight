from __future__ import annotations

import base64
import io
import itertools
import re
import threading
import time
from typing import Sequence

import httpx
import numpy as np
from PIL import Image

from ..config import settings
from .base import Box, VLMVerdict


def _to_data_uri(img: Image.Image, fmt: str = "JPEG", max_side: int = 512) -> str:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


class KeyPool:
    """Round-robins across NIM keys and benches a key for a while after it fails."""

    def __init__(self, keys: Sequence[str], cooldown: float = 60.0):
        self._keys = list(keys)
        self._cooldown = cooldown
        self._benched: dict[str, float] = {}
        self._cycle = itertools.cycle(self._keys) if self._keys else None
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return bool(self._keys)

    def next(self) -> str | None:
        if not self._keys:
            return None
        with self._lock:
            now = time.monotonic()
            for _ in range(len(self._keys)):
                key = next(self._cycle)
                if self._benched.get(key, 0.0) <= now:
                    return key
            # everything benched; return the one freeing up soonest
            return min(self._keys, key=lambda k: self._benched.get(k, 0.0))

    def bench(self, key: str) -> None:
        with self._lock:
            self._benched[key] = time.monotonic() + self._cooldown


class NimClient:
    def __init__(self):
        self.pool = KeyPool(settings.api_keys)
        self.base = settings.nim_base.rstrip("/")
        self.dim = 1024  # NV-CLIP width; overwritten by the actual embedding width downstream
        self._http = httpx.Client(timeout=60.0)

    def _post(self, path: str, payload: dict) -> dict:
        last_err: Exception | None = None
        tried: set[str] = set()
        for _ in range(max(1, len(settings.api_keys))):
            key = self.pool.next()
            if key is None:
                raise RuntimeError("no NIM keys configured")
            if key in tried:
                break  # pool only has benched keys left; stop hammering the same one
            tried.add(key)
            try:
                r = self._http.post(
                    f"{self.base}{path}",
                    headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                    json=payload,
                )
                if r.status_code in (401, 403, 429):
                    self.pool.bench(key)
                    last_err = httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                    continue
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:  # network / 5xx / parse
                last_err = e
                self.pool.bench(key)
        raise RuntimeError(f"NIM request failed: {last_err}")

    # --- embeddings -------------------------------------------------------
    def embed(self, inputs: list[str], model: str) -> np.ndarray:
        data = self._post("/embeddings", {"input": inputs, "model": model, "encoding_format": "float"})
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return np.asarray([d["embedding"] for d in rows], dtype=np.float32)

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        return self.embed([_to_data_uri(im) for im in images], settings.embed_model)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(list(texts), settings.embed_model)

    # --- vision-language --------------------------------------------------
    def chat(self, messages: list[dict], model: str, max_tokens: int = 256) -> str:
        data = self._post(
            "/chat/completions",
            {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
        )
        return data["choices"][0]["message"]["content"]

    def verify(self, images: Sequence[Image.Image], question: str, proposed: str) -> VLMVerdict:
        content = [{"type": "text", "text": _VERIFY_PROMPT.format(q=question, a=proposed)}]
        for im in list(images)[:4]:
            content.append({"type": "image_url", "image_url": {"url": _to_data_uri(im)}})
        reply = self.chat([{"role": "user", "content": content}], settings.vlm_model)
        return _parse_verdict(reply, proposed)

    def detect(self, image: Image.Image, prompt: str) -> list[Box]:
        content = [
            {"type": "text", "text": f"Find every '{prompt}'. Reply with boxes as x0,y0,x1,y1 in pixels, one per line."},
            {"type": "image_url", "image_url": {"url": _to_data_uri(image)}},
        ]
        reply = self.chat([{"role": "user", "content": content}], settings.grounder_model, max_tokens=256)
        return _parse_boxes(reply, prompt, image.size)

    def healthy(self) -> bool:
        if not self.pool:
            return False
        try:
            self.embed_texts(["ping"])
            return True
        except Exception:
            return False


_VERIFY_PROMPT = (
    "You are checking an answer against first-person photos.\n"
    "Question: {q}\nProposed answer: {a}\n"
    "Do the photos support the proposed answer? Start your reply with YES or NO, "
    "then a confidence from 0 to 1, then one short sentence."
)


def _parse_verdict(reply: str, proposed: str) -> VLMVerdict:
    head = reply.strip().lower()
    agrees = head.startswith("yes")
    m = re.search(r"(\d?\.\d+|\d+)", reply)
    conf = float(m.group(1)) if m else (0.7 if agrees else 0.3)
    conf = min(1.0, max(0.0, conf if conf <= 1 else conf / 100))
    text = reply.strip().split("\n")[-1].strip() or proposed
    return VLMVerdict(answer=text, agrees=agrees, confidence=conf)


def _parse_boxes(reply: str, label: str, size: tuple[int, int]) -> list[Box]:
    w, h = size
    out = []
    for line in reply.splitlines():
        nums = re.findall(r"-?\d+\.?\d*", line)
        if len(nums) >= 4:
            x0, y0, x1, y1 = (float(v) for v in nums[:4])
            if max(x0, y0, x1, y1) <= 1.5:  # normalized coords
                x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
            out.append(Box(label=label, score=0.5, xyxy=(x0, y0, x1, y1)))
    return out
