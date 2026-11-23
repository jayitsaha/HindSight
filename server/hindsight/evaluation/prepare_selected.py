"""Cut per-question frames (e.g. the learned frame picker's choice) out of the raw session videos, for Gemma.

Input: a selections file from hindsight.curation.frame_selectors (one line per question:
{"qid": ..., "frames": [[video_id, t_sec], ...]}). Frames are styled exactly like the curated videos
(448x448, source time drawn in the corner) and stored once per (video, time), since questions share frames:

    <out>/selections.jsonl                 copy of the input (which frames each question gets)
    <out>/frames/<video_id>/<ms>.jpg       one JPEG per distinct frame, <ms> = source time in milliseconds

Each session video is downloaded, cut and deleted in turn (one 2-4 GB file on disk at a time). Run as a Slurm array:
--task i --tasks N handles every N-th video, so tasks can run in parallel and re-running skips finished frames.

    python -m hindsight.evaluation.prepare_selected --selections results/full/selectors/selections_logreg_top.jsonl \
        --out selected/logreg_top --task 0 --tasks 1

Then: python -m hindsight.evaluation.run --condition selected --selected-dir selected/logreg_top ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ID = "OSU-AIoT-MLSys-Lab/SuperMemory-VQA"


def frame_name(t_sec: float) -> str:
    return f"{int(round(t_sec * 1000))}.jpg"


def load_selections(path: str) -> Dict[str, List[tuple]]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["qid"])] = [(v, float(t)) for v, t in r["frames"]]
    return out


def frames_by_video(selections: Dict[str, List[tuple]]) -> Dict[str, List[float]]:
    by = defaultdict(set)
    for frames in selections.values():
        for v, t in frames:
            by[v].add(round(t, 3))
    return {v: sorted(ts) for v, ts in by.items()}


def my_videos(videos: Sequence[str], task: int, tasks: int) -> List[str]:
    return [v for i, v in enumerate(sorted(videos)) if i % tasks == task]


def cut_frames(video_path: str, times: Sequence[float], out_dir: str, size: int = 448, overlay: bool = True) -> int:
    """Write one styled JPEG per time; returns how many were written."""
    import cv2

    from hindsight.video.curation import draw_timestamp, iter_frames, probe_video
    info = probe_video(video_path)
    want = {}
    for t in times:
        want.setdefault(min(int(round(t * info.fps)), info.n_frames - 1), []).append(t)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for idx, frame in iter_frames(video_path, sorted(want)):
        if size:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        for t in want[idx]:
            img = frame.copy()
            if overlay:
                draw_timestamp(img, t)
            cv2.imwrite(os.path.join(out_dir, frame_name(t)), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
    return n


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--data", default="data_selected",
                    help="download folder (separate from curation's data/, so parallel jobs never delete each other's videos)")
    ap.add_argument("--size", type=int, default=448, help="frame size (square); 0 = keep the source size")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--keep-video", action="store_true", help="do not delete each video after cutting it")
    a = ap.parse_args(argv)

    sel = load_selections(a.selections)
    os.makedirs(a.out, exist_ok=True)
    if a.task == 0:
        shutil.copyfile(a.selections, os.path.join(a.out, "selections.jsonl"))
    by_vid = frames_by_video(sel)
    mine = my_videos(by_vid, a.task, a.tasks)
    print(f"{len(sel)} questions, {sum(map(len, by_vid.values()))} distinct frames in {len(by_vid)} videos; "
          f"task {a.task}/{a.tasks}: {len(mine)} videos", flush=True)
    from hindsight.data.supermemory import video_paths
    paths = video_paths()
    failed = 0
    for v in mine:
        out_dir = os.path.join(a.out, "frames", v)
        todo = [t for t in by_vid[v] if not os.path.isfile(os.path.join(out_dir, frame_name(t)))]
        if not todo:
            print(f"{v}: done already", flush=True)
            continue
        if v not in paths:
            print(f"!! {v}: not in the dataset repo", flush=True); failed += 1
            continue
        t0 = time.time()
        try:
            from huggingface_hub import hf_hub_download
            mp4 = hf_hub_download(REPO_ID, paths[v], repo_type="dataset", local_dir=a.data)
            t1 = time.time()
            n = cut_frames(mp4, todo, out_dir, a.size, not a.no_overlay)
            print(f"{v}: {n}/{len(todo)} frames (download {t1 - t0:.0f}s, cut {time.time() - t1:.0f}s)", flush=True)
            if n < len(todo):
                failed += 1
            if not a.keep_video:
                os.remove(mp4)
        except Exception as e:                                       # keep going with the other videos
            print(f"!! {v}: {type(e).__name__}: {e}", flush=True); failed += 1
    print(f"task {a.task} finished, {failed} videos with problems", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
