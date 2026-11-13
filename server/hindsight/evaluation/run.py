"""Evaluate a VLM on SuperMemory-VQA Object & Location Memory questions under one visual condition.

    python -m hindsight.evaluation.run --backend hf --model-name google/gemma-4-E4B-it --condition <c> ...

Conditions (what the model sees):
  curated  K frames chosen by frame curation (uniform or CLIP+KMeans) from the evidence sessions
           --curated-dir out|hf:<repo> --method kmeans|uniform --k 100 [--sessions evidence|primary]
  oracle   frames cut around the answer-evidence span        --pilot-dir pilot   (prepare_pilot.py)
  blind    no frames (text-only control for option priors)    with --pilot-dir or --curated-dir to use
           the same questions as the oracle / curated run
  video    one raw/any mp4 per question: <video-dir>/<primary_video_id>.mp4, sampled to --max-frames

Results go to --output-dir (default results/vlm/<condition>[_<method>_K<k>]): metrics.json, eval_report.md,
predictions.jsonl, and raw_predictions.jsonl, which makes a re-run resume where it stopped.
"""
import argparse
import functools
import json
import os

from hindsight.vlm.config import VLMConfig, DataConfig, EvalConfig
from hindsight.data.loader import SuperMemoryDataLoader
from hindsight.data.supermemory import REPO_ID
from hindsight.vlm.pipeline import VLMPipeline
from hindsight.evaluation.evaluator import Evaluator

CONDITIONS = ("curated", "oracle", "blind", "video")


def default_qa_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, "data/json/all_qa.json", repo_type="dataset")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", default="curated", choices=CONDITIONS)
    p.add_argument("--backend", default="mock", choices=["mock", "hf"], help="'hf' = Hugging Face model on GPU")
    p.add_argument("--model-name", default="google/gemma-4-E4B-it")
    p.add_argument("--device", default="auto", help="auto/cuda/cpu/mps (hf backend)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap on frames per question (default: none for curated/oracle, 16 for video)")
    p.add_argument("--image-tokens", type=int, default=None, choices=[70, 140, 280, 560, 1120],
                   help="Gemma 4 visual tokens per frame (default: model default, 280)")
    # questions
    p.add_argument("--data-path", default=None, help="QA json/jsonl (default: all_qa.json from the HF dataset)")
    p.add_argument("--task-category", default="Object and Location Memory")
    p.add_argument("--subject", default=None, help="only this participant, e.g. S1")
    p.add_argument("--answerable-only", action="store_true")
    p.add_argument("--max-questions", type=int, default=None)
    # condition inputs
    p.add_argument("--curated-dir", default=None, help="curation output folder, or hf:<dataset repo>")
    p.add_argument("--method", default="kmeans", help="curation method (file prefix): kmeans | uniform")
    p.add_argument("--k", type=int, default=100, help="frames per session in the curated videos")
    p.add_argument("--sessions", default="evidence", choices=["evidence", "primary"],
                   help="curated: sessions shown per question")
    p.add_argument("--no-frame-labels", action="store_true", help="curated: do not label frames with session/time")
    p.add_argument("--pilot-dir", default=None, help="from python -m hindsight.evaluation.prepare_pilot")
    p.add_argument("--video-dir", default=None, help="video condition: folder of <video_id>.mp4")
    p.add_argument("--output-dir", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.condition == "curated" and not args.curated_dir:
        raise SystemExit("--condition curated needs --curated-dir")
    if args.condition == "oracle" and not args.pilot_dir:
        raise SystemExit("--condition oracle needs --pilot-dir")
    if args.condition == "video" and not args.video_dir:
        raise SystemExit("--condition video needs --video-dir")

    max_frames = args.max_frames if args.max_frames is not None else (16 if args.condition == "video" else None)
    vlm_config = VLMConfig(backend=args.backend, model_name=args.model_name, device=args.device,
                           max_frames=max_frames, image_token_budget=args.image_tokens)
    tag = args.condition + (f"_{args.method}_K{args.k}" if args.condition == "curated" else "")
    out_dir = args.output_dir or os.path.join("results", "vlm", tag)

    manifest = None
    if args.pilot_dir:
        with open(os.path.join(args.pilot_dir, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        data_path = os.path.join(args.pilot_dir, manifest["questions"])
    else:
        data_path = args.data_path or default_qa_path()

    data_config = DataConfig(dataset_name_or_path=data_path, task_category=args.task_category,
                             subject_filter=[args.subject] if args.subject else None)
    questions = SuperMemoryDataLoader(data_config).load_questions(source_path=data_path)
    if args.answerable_only:
        questions = [q for q in questions if q.is_answerable is not False]

    if args.curated_dir:  # also for blind: score the text-only control on the same questions as curated
        from hindsight.evaluation.curated import CuratedFrames, filter_questions
        cf = CuratedFrames(args.curated_dir, method=args.method, k=args.k, sessions=args.sessions,
                           label_frames=not args.no_frame_labels)
        n_all = len(questions)
        questions = filter_questions(questions, cf)
        print(f"[RunEval] Curated {args.method} K={args.k}: {len(cf.video_ids)} sessions; "
              f"{len(questions)}/{n_all} questions have all their sessions curated")
    if args.max_questions:
        questions = questions[: args.max_questions]
    if not questions:
        raise SystemExit("[RunEval] No questions matched. Check --subject / --curated-dir / --data-path.")

    visual_inputs = []
    for q in questions:
        if args.condition == "blind":
            visual_inputs.append([])                                    # text-only control
        elif args.condition == "oracle":
            visual_inputs.append([os.path.join(args.pilot_dir, r) for r in manifest["frames"][str(q.question_id)]])
        elif args.condition == "curated":
            visual_inputs.append(functools.partial(cf.frames_for, q))    # loaded lazily, one question at a time
        else:
            visual_inputs.append(os.path.join(args.video_dir, f"{q.video_id}.mp4"))

    print(f"[RunEval] {tag} | backend {args.backend} ({args.model_name}) | {len(questions)} questions | "
          f"max_frames={max_frames} | tokens/frame={args.image_tokens or 'default'} -> {out_dir}")
    pipeline = VLMPipeline(config=vlm_config)
    summary = Evaluator(EvalConfig(output_dir=out_dir)).evaluate_pipeline(pipeline, questions, visual_inputs)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "max_frames": max_frames, "n_questions": len(questions)}, f, indent=1)
    print("\n" + summary.to_markdown())
    return summary


if __name__ == "__main__":
    main()
