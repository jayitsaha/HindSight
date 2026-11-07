"""Task discovery. Each trainer module exposes a module-level `TASK` (its Task
class); this imports the ones that are built and skips the rest, so adding a task
is just dropping in `trainers/<name>.py` — no edits here or in train.py."""

from __future__ import annotations

import importlib

_MODULES = ["mq", "nlq", "pnr", "vq2d"]


def available() -> dict:
    tasks = {}
    for name in _MODULES:
        try:
            mod = importlib.import_module(f".trainers.{name}", __package__)
        except ModuleNotFoundError:
            continue
        task = getattr(mod, "TASK", None)
        if task is not None:
            tasks[name] = task
    return tasks
