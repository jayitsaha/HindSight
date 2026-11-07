from __future__ import annotations

import functools

from ..config import settings
from .base import Embedder, Grounder, VLM
from . import local


@functools.lru_cache(maxsize=1)
def _nim():
    from .nim import NimClient

    return NimClient()


@functools.lru_cache(maxsize=1)
def _nim_ok() -> bool:
    if settings.provider == "local":
        return False
    if not settings.has_keys:
        return False
    try:
        return _nim().healthy()
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.provider in ("auto", "nim") and _nim_ok():
        return _nim()
    return local.build_local_embedder()


@functools.lru_cache(maxsize=1)
def get_grounder() -> Grounder:
    if settings.provider in ("auto", "nim") and _nim_ok():
        return _nim()
    return local.WindowGrounder()


@functools.lru_cache(maxsize=1)
def get_vlm() -> VLM:
    if settings.provider in ("auto", "nim") and _nim_ok():
        return _nim()
    return local.HeuristicVLM()


def provider_status() -> dict:
    active = "nim" if (settings.provider in ("auto", "nim") and _nim_ok()) else "local"
    return {
        "configured": settings.provider,
        "active": active,
        "keys_loaded": len(settings.api_keys),
        "embedder": type(get_embedder()).__name__,
        "vlm": type(get_vlm()).__name__,
        "grounder": type(get_grounder()).__name__,
    }
