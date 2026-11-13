"""
Build the SuperMemory-VQA pilot: pick questions, then download each needed session
video one at a time, cut evidence-clip frames to JPEGs, and delete the video.

Only the (small) frames and the selected question JSON stay on disk, so the pilot
needs room for one 2-4 GB video at a time instead of the 703 GB repo.

Usage (on PACE, from the project dir):
    python -m hindsight.evaluation.prepare_pilot --subject S1 --n 40 --dry-run   # list videos + sizes
    python -m hindsight.evaluation.prepare_pilot --subject S1 --n 40             # download + cut frames
"""

import argparse
import json
import os
import shutil
import sys


from hindsight.vlm.config import DataConfig
from hindsight.data.loader import SuperMemoryDataLoader
from hindsight.data.pilot import HF_REPO, clip_video_ids, hf_video_path, select_pilot
from hindsight.video.clips import clip_frame_times, extract_frames, video_duration


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="S1", help="Participant, e.g. S1")
    ap.add_argument("--qa-json", default=None, help="Local qa_person_N.json (downloaded from the HF repo if omitted)")
    ap.add_argument("--n", type=int, default=40, help="Number of pilot questions")
    ap.add_argument("--max-videos", type=int, default=6, help="Max distinct session videos to download")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=8, help="Frames per question (split across its clips)")
    ap.add_argument("--pad", type=float, default=5.0, help="Seconds of context around each evidence span")
    ap.add_argument("--pilot-dir", default="pilot")
    ap.add_argument("--download-dir", default=None, help="Where the full video is staged (default: <pilot-dir>/_videos)")
    ap.add_argument("--keep-videos", action="store_true", help="Do not delete videos after cutting frames")
    ap.add_argument("--dry-run", action="store_true", help="Select questions and print the download plan only")
    args = ap.parse_args()

    person = int(args.subject.lstrip("Ss"))
    os.makedirs(args.pilot_dir, exist_ok=True)
    qa_path = args.qa_json
    if not qa_path:
        from huggingface_hub import hf_hub_download
        qa_path = hf_hub_download(HF_REPO, f"data/json/qa_person_{person}.json", repo_type="dataset")

    loader = SuperMemoryDataLoader(DataConfig(subject_filter=[args.subject]))
    questions = loader.load_questions(qa_path)
    available = None
    try:
        from huggingface_hub import HfApi
        available = {os.path.basename(e.path)[:-4] for e in HfApi().list_repo_tree(
            HF_REPO, path_in_repo=f"data/video/Person_{person}", repo_type="dataset") if e.path.endswith(".mp4")}
    except Exception as e:
        print(f"[Pilot] (could not list repo videos, assuming all exist: {e})")
    pilot = select_pilot(questions, n=args.n, max_videos=args.max_videos, seed=args.seed, available_videos=available)
    videos = sorted({v for q in pilot for v in clip_video_ids(q)})
    n_un = sum(1 for q in pilot if q.is_answerable is False)
    print(f"[Pilot] {len(pilot)} questions ({n_un} unanswerable) from {len(videos)} videos")

    if args.dry_run:
        try:
            from huggingface_hub import HfApi
            infos = HfApi().get_paths_info(HF_REPO, [hf_video_path(v) for v in videos], repo_type="dataset")
            sizes = {i.path: i.size for i in infos}
        except Exception as e:
            print(f"[Pilot] (could not fetch sizes: {e})")
            sizes = {}
        total = 0
        for v in videos:
            sz = sizes.get(hf_video_path(v), 0)
            total += sz
            print(f"  {hf_video_path(v)}  {sz / 1e9:.2f} GB")
        print(f"[Pilot] peak scratch needed: {max(sizes.values(), default=0) / 1e9:.2f} GB (one video at a time); "
              f"total download {total / 1e9:.2f} GB")
        return

    # Self-contained question file: the pilot needs no other copy of the dataset.
    with open(os.path.join(args.pilot_dir, "pilot_questions.json"), "w", encoding="utf-8") as f:
        json.dump([q.raw for q in pilot], f, indent=1)

    frames_dir = os.path.join(args.pilot_dir, "frames")
    stage = args.download_dir or os.path.join(args.pilot_dir, "_videos")
    os.makedirs(stage, exist_ok=True)

    # Plan: every (question, clip) -> frame paths, grouped by video.
    plan = {}  # video_id -> [(times, paths)]
    frames_by_q = {}
    for q in pilot:
        per_clip = max(1, args.frames // len(q.evidence_clips))
        frames_by_q[str(q.question_id)] = []
        for k, c in enumerate(q.evidence_clips):
            paths = [os.path.join(frames_dir, str(q.question_id), f"c{k}_f{i}.jpg") for i in range(per_clip)]
            frames_by_q[str(q.question_id)] += paths
            plan.setdefault(c["video_id"], []).append((c, paths))

    from huggingface_hub import hf_hub_download
    for vid in videos:
        todo = [(c, p) for c, p in plan[vid] if not all(os.path.isfile(x) for x in p)]
        if not todo:
            print(f"[Pilot] {vid}: frames already extracted, skipping")
            continue
        print(f"[Pilot] Downloading {hf_video_path(vid)} ...")
        local = hf_hub_download(HF_REPO, hf_video_path(vid), repo_type="dataset", local_dir=stage)
        dur = video_duration(local)
        for c, paths in todo:
            times = clip_frame_times(c["start"], c["end"], len(paths), pad=args.pad, duration=dur)
            extract_frames(local, times, paths)
        print(f"[Pilot] {vid}: wrote frames for {len(todo)} clips ({dur / 60:.0f} min video)")
        if not args.keep_videos:
            os.remove(local)

    if not args.keep_videos:
        shutil.rmtree(stage, ignore_errors=True)

    manifest = {
        "subject": args.subject,
        "questions": "pilot_questions.json",
        "frames": {qid: [os.path.relpath(p, args.pilot_dir) for p in ps] for qid, ps in frames_by_q.items()},
        "videos": videos,
        "seed": args.seed,
        "pad_seconds": args.pad,
    }
    with open(os.path.join(args.pilot_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"[Pilot] Done. Manifest: {os.path.join(args.pilot_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
