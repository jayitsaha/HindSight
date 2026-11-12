"""
CLI script to run batch evaluation of VLM pipeline on SuperMemory-VQA.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hindsight.config import VLMConfig, DataConfig, EvalConfig
from hindsight.data.loader import SuperMemoryDataLoader
from hindsight.vlm.pipeline import VLMPipeline
from hindsight.evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM Pipeline on SuperMemory-VQA")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "hf", "gemini"],
                        help="VLM backend: 'mock', 'hf' (open-source), or 'gemini'")
    parser.add_argument("--model-name", type=str, default="mistralai/Pixtral-12B-2409",
                        help="HuggingFace model name or path (e.g. mistralai/Pixtral-12B-2409 or HuggingFaceTB/SmolVLM-Instruct)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to questions JSON/JSONL file, or Hugging Face dataset name")
    parser.add_argument("--task-category", type=str, default="Object and Location Memory",
                        help="Task category filter")
    parser.add_argument("--subject", type=str, default=None,
                        help="Filter by participant/subject (e.g., S1, S9)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions to evaluate (for testing on Kaggle)")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory containing video files or frame folders")
    parser.add_argument("--pilot-dir", type=str, default=None,
                        help="Pilot directory from scripts/prepare_pilot.py (questions + pre-cut evidence frames)")
    parser.add_argument("--condition", type=str, default="oracle", choices=["oracle", "blind"],
                        help="With --pilot-dir: 'oracle' = frames cut around the evidence span, "
                             "'blind' = no frames (text-only control for option priors)")
    parser.add_argument("--device", type=str, default="auto", help="Device for the hf backend (auto/cuda/cpu/mps)")
    parser.add_argument("--output-dir", type=str, default="results/eval",
                        help="Directory to save evaluation reports and predictions")

    args = parser.parse_args()

    # Setup configs
    vlm_config = VLMConfig(
        backend=args.backend,
        model_name=args.model_name,
        device=args.device,
    )
    data_config = DataConfig(
        dataset_name_or_path=args.data_path or "OSU-AIoT-MLSys-Lab/SuperMemory-VQA",
        task_category=args.task_category,
        subject_filter=[args.subject] if args.subject else None,
        max_questions=args.max_questions
    )
    eval_config = EvalConfig(output_dir=args.output_dir)

    print(f"[RunEval] Backend: {vlm_config.backend} | Model: {vlm_config.model_name}")
    print(f"[RunEval] Task: {data_config.task_category} | Subject: {args.subject or 'All'}")

    manifest = None
    if args.pilot_dir:
        with open(os.path.join(args.pilot_dir, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        args.data_path = os.path.join(args.pilot_dir, manifest["questions"])
        data_config.dataset_name_or_path = args.data_path
        print(f"[RunEval] Pilot: {args.pilot_dir} | Condition: {args.condition}")

    loader = SuperMemoryDataLoader(data_config)
    questions = loader.load_questions(source_path=args.data_path)

    if not questions:
        print("[RunEval] No questions matched criteria. Exiting.")
        return

    # Visual inputs
    visual_inputs = []
    for q in questions:
        if manifest is not None:
            # [] = text-only control; otherwise the pre-cut JPEG frames for this question
            rel = manifest["frames"][str(q.question_id)] if args.condition == "oracle" else []
            visual_inputs.append([os.path.join(args.pilot_dir, r) for r in rel])
            continue
        v_path = None
        if args.video_dir and q.video_id:
            candidate = os.path.join(args.video_dir, f"{q.video_id}.mp4")
            if os.path.exists(candidate):
                v_path = candidate
        visual_inputs.append(v_path or f"simulated_video_{q.question_id}")

    pipeline = VLMPipeline(config=vlm_config)
    evaluator = Evaluator(config=eval_config)

    summary = evaluator.evaluate_pipeline(pipeline, questions, visual_inputs)
    print("\n" + summary.to_markdown())


if __name__ == "__main__":
    main()
