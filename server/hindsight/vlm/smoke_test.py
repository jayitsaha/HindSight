"""
Smoke test: python -m hindsight.vlm.smoke_test --model-name google/gemma-4-E4B-it

End-to-end sample video execution for Hindsight on PACE GPU (H100 / H200 / A100).
Generates a synthetic egocentric video (.mp4) and feeds it directly into
an open-source VLM (e.g. Pixtral-12B, SmolVLM, Llama-3.2-Vision) using CUDA.
"""

import os
import sys
import json
import time
import argparse


from hindsight.vlm.config import VLMConfig
from hindsight.data.schema import QuestionRecord
from hindsight.video.demo_video import create_demo_video
from hindsight.vlm.pipeline import VLMPipeline


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Sample Video on PACE GPU")
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/SmolVLM-Instruct",
        help="Model name (e.g. 'HuggingFaceTB/SmolVLM-Instruct' [2.2B] or 'mistralai/Pixtral-12B-2409')"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("     HINDSIGHT: END-TO-END SAMPLE VIDEO INFERENCE (PACE GPU)")
    print(f"     Target Model: {args.model_name}")
    print("=" * 70)

    # 1. Output directory: $SCRATCH_PROJECT if set, else the current directory
    scratch_dir = os.environ.get("SCRATCH_PROJECT") or os.getcwd()
    os.makedirs(scratch_dir, exist_ok=True)
    results_dir = os.path.join(scratch_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    video_path = os.path.join(scratch_dir, "sample_aria_session.mp4")

    # 2. Generate Sample Egocentric Video
    print(f"\n[Step 1] Generating sample egocentric video at: {video_path}")
    create_demo_video(
        output_path=video_path,
        duration_seconds=3.0,
        fps=10,
        object_label="golden keys",
        location_label="wooden console table in entryway",
        action_label="placed on table"
    )
    print(f"Generated video: {video_path} ({os.path.getsize(video_path)} bytes)")

    # 3. Create Sample SuperMemory-VQA Question
    question = QuestionRecord(
        question_id="sm_vqa_pace_sample_001",
        category="Object and Location Memory",
        subject_id="S1",
        session_id="session_01",
        question="Where did I put down my keys after entering the house?",
        choices=[
            "A) On the wooden console table in the entryway",
            "B) In the entryway area",
            "C) On the kitchen dining table",
            "D) Unanswerable"
        ],
        answer="A",
        answer_idx=0
    )

    print("\n[Step 2] Prepared Question:")
    print(f"  ID:       {question.question_id}")
    print(f"  Question: {question.question}")
    print("  Options:")
    print(question.formatted_options())
    print(f"  Ground Truth: {question.answer}")

    # 4. Initialize VLM Pipeline (Auto-detect GPU / CUDA / CPU)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "None (CPU Mode)"
    except Exception:
        has_cuda = False
        gpu_name = "None (CPU Mode)"

    device = "cuda" if has_cuda else "cpu"
    print(f"\n[Step 3] Initializing Open-Source VLM Backend ({args.model_name})...")
    print(f"  Target Device: {device.upper()} | GPU: {gpu_name}")

    vlm_config = VLMConfig(
        backend="hf",
        model_name=args.model_name,
        device=device
    )
    pipeline = VLMPipeline(config=vlm_config)

    # 5. Run Inference
    print("\n[Step 4] Running End-to-End Video Inference...")
    start_time = time.time()
    output = pipeline.answer_question(
        question=question,
        video_input=video_path
    )
    elapsed = time.time() - start_time

    # 6. Display Output
    print("\n" + "=" * 70)
    print("                       VLM PREDICTION OUTPUT")
    print("=" * 70)
    print(f"Model:            {args.model_name}")
    print(f"Predicted Choice: {output.predicted_choice}")
    print(f"Predicted Text:   {output.predicted_text}")
    print(f"Abstain Flag:     {output.abstain}")
    print(f"Confidence:       {output.confidence:.4f}")
    print(f"Reasoning:\n{output.reasoning}")
    print(f"Inference Latency: {elapsed:.2f} seconds")
    print("=" * 70)

    # 7. Save result
    out_file = os.path.join(results_dir, "pace_sample_output.json")
    result_data = {
        "model_name": args.model_name,
        "question_id": question.question_id,
        "question": question.question,
        "ground_truth": question.answer,
        "prediction": output.to_dict(),
        "video_path": video_path,
        "elapsed_seconds": elapsed
    }
    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nSaved detailed prediction JSON to: {out_file}")


if __name__ == "__main__":
    main()
