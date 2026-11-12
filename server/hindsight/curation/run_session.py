"""Process ONE session: download it, then write KMeans and budget-matched uniform videos for each K.

Run by run_array.sbatch with the Slurm array index (one session per task):
    python -m hindsight.curation.run_session --index $SLURM_ARRAY_TASK_ID --sessions sessions.txt --ks 32 64 100 150

Outputs (per session):  out/<video_id>/kmeans_K<k>.mp4 + .json
                        out/<video_id>/uniform_K<k>.mp4 + .json   (same number of frames as KMeans)
Embeddings are cached in cache/, so each extra K only costs one write pass.
"""
import argparse
import os
import time
from pathlib import Path

from hindsight.video.curation import FrameCurator, probe_video

REPO_ID = "OSU-AIoT-MLSys-Lab/SuperMemory-VQA"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "frame_curation.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, help="line of sessions.txt to process (0-based); default $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--sessions", default="sessions.txt")
    ap.add_argument("--video", help="process this local mp4 instead of a sessions.txt line")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--ks", type=int, nargs="+", default=[100])
    ap.add_argument("--data", default="data", help="download folder")
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--embed-fps", type=float, default=1.0)
    ap.add_argument("--resize", type=int, nargs=2, default=[448, 448], help="output frame size; 0 0 = keep source size")
    ap.add_argument("--no-overlay", action="store_true", help="don't draw timestamps on output frames")
    ap.add_argument("--delete-video", action="store_true", help="delete the downloaded mp4 afterwards")
    a = ap.parse_args()

    if a.video:
        mp4, vid = a.video, os.path.splitext(os.path.basename(a.video))[0]
    else:
        idx = a.index if a.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        with open(a.sessions) as f:
            lines = [l.rstrip("\n").split("\t") for l in f if l.strip()]
        repo_path, vid = lines[idx][0], lines[idx][1]
        mp4 = os.path.join(a.data, repo_path)
        if not os.path.exists(mp4):
            from huggingface_hub import hf_hub_download
            t = time.time()
            mp4 = hf_hub_download(REPO_ID, repo_path, repo_type="dataset", local_dir=a.data)
            print(f"downloaded {vid} in {time.time() - t:.0f}s", flush=True)

    out_dir = os.path.join(a.out, vid)
    os.makedirs(out_dir, exist_ok=True)
    overrides = {"kmeans.embed_fps": a.embed_fps, "kmeans.cache_dir": a.cache,
                 "io.overlay_timestamp": not a.no_overlay,
                 "io.resize": (a.resize if a.resize[0] > 0 else None)}
    cur = FrameCurator(a.config if os.path.exists(a.config) else None, **overrides)
    info = probe_video(mp4)
    print(f"{vid}: {info.duration / 60:.1f} min, {info.n_frames} frames @ {info.fps:.2f} fps", flush=True)

    for k in a.ks:
        t = time.time()
        r = cur.curate(mp4, os.path.join(out_dir, f"kmeans_K{k}.mp4"), method="kmeans", k=k)
        u = cur.curate(mp4, os.path.join(out_dir, f"uniform_K{k}.mp4"), method="uniform", fps=k / info.duration)
        print(f"  K={k}: kmeans {len(r.indices)} frames, uniform {len(u.indices)} frames ({time.time() - t:.0f}s)", flush=True)

    if a.delete_video and not a.video:
        os.remove(mp4)
    print("done", vid, flush=True)


if __name__ == "__main__":
    main()
