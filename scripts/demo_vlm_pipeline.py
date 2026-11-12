"""
Demonstration script for the Hindsight VLM Pipeline and Evaluation.
Runs end-to-end on sample egocentric visual questions without external API/GPU requirements.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Add repo to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hindsight.config import VLMConfig, EvalConfig
from hindsight.data.loader import SuperMemoryDataLoader, DEMO_SAMPLES
from hindsight.vlm.pipeline import VLMPipeline
from hindsight.evaluation.evaluator import Evaluator


def generate_sample_frame(label: str, color: tuple = (70, 130, 180)) -> Image.Image:
    """Generates a synthetic egocentric video frame with visual labels."""
    img = Image.new("RGB", (640, 480), color=color)
    draw = ImageDraw.Draw(img)

    # Draw border simulating camera view
    draw.rectangle([(20, 20), (620, 460)], outline=(255, 255, 255), width=3)

    # Draw text label
    text = f"Egocentric View: {label}"
    draw.text((40, 50), text, fill=(255, 255, 255))
    draw.text((40, 90), "Simulated Aria Glasses Frame [30fps]", fill=(200, 200, 200))

    # Draw bounding box representing detected object / interaction
    draw.rectangle([(180, 180), (460, 360)], outline=(255, 215, 0), width=4)
    draw.text((190, 190), f"Object Region: {label}", fill=(255, 215, 0))

    return img


def main():
    print("=" * 70)
    print("       HINDSIGHT: VLM PIPELINE & EVALUATION DEMONSTRATION")
    print("=" * 70)

    # 1. Load questions (filtered for Object and Location Memory)
    print("\n[Step 1] Loading Object and Location Memory questions...")
    loader = SuperMemoryDataLoader()
    questions = loader.load_questions()
    print(f"Loaded {len(questions)} test questions.")

    # 2. Initialize VLM Pipeline (Mock for fast demonstration)
    print("\n[Step 2] Initializing VLM Pipeline (backend='mock')...")
    vlm_config = VLMConfig(backend="mock")
    pipeline = VLMPipeline(config=vlm_config)

    # 3. Simulate visual inputs for each question
    print("\n[Step 3] Preparing visual inputs (frames/clips from teammates)...")
    visual_inputs = []
    for q in questions:
        # Generate 2 synthetic frames depicting the scenario
        frame1 = generate_sample_frame(f"Action start for {q.question_id}")
        frame2 = generate_sample_frame(f"Evidence for '{q.question}'", color=(46, 139, 87))
        visual_inputs.append([frame1, frame2])
    print(f"Prepared visual evidence for {len(visual_inputs)} queries.")

    # 4. Demonstrate single-query inference
    demo_q = questions[0]
    demo_v = visual_inputs[0]
    print("\n" + "-" * 70)
    print("[Step 4] Demonstrating single-query VLM inference:")
    print(f"Question ID: {demo_q.question_id} (Subject: {demo_q.subject_id})")
    print(f"Question:    {demo_q.question}")
    print("Options:")
    print(demo_q.formatted_options())
    print("-" * 70)

    answer = pipeline.answer_question(demo_q, demo_v)
    print(">>> VLM Pipeline Output:")
    print(f"  Selected Choice:  {answer.predicted_choice}")
    print(f"  Selected Text:    {answer.predicted_text}")
    print(f"  Abstain (Unans):  {answer.abstain}")
    print(f"  Confidence:       {answer.confidence}")
    print(f"  Reasoning:        {answer.reasoning}")
    print(f"  Inference Time:   {answer.latency_seconds:.4f}s")
    print("-" * 70)

    # 5. Run Batch Evaluation Pipeline
    print("\n[Step 5] Running Batch Evaluation Pipeline...")
    eval_config = EvalConfig(output_dir="results/demo_eval")
    evaluator = Evaluator(config=eval_config)
    summary = evaluator.evaluate_pipeline(pipeline, questions, visual_inputs)

    # 6. Display evaluation report
    print("\n" + "=" * 70)
    print("                EVALUATION REPORT SUMMARY")
    print("=" * 70)
    print(summary.to_markdown())
    print("=" * 70)
    print(f"Full results, metrics.json, and predictions.jsonl saved in: {eval_config.output_dir}/")
if __name__ == "__main__":
    main()
