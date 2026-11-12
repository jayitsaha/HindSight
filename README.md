# Hindsight: VLM Pipeline & Evaluation for SuperMemory-VQA

Egocentric Object & Location Memory system with calibrated recall for wearable camera streams (Meta Aria Glasses).

## 1. Scope & Task Analysis

### Do all questions involve all videos?
**No.** SuperMemory-VQA is partitioned across **10 distinct human participants (S1 to S10)**.
- Each question represents an episodic personal memory query (e.g., *"Where did I put my keys?"*) anchored to that **specific participant's stream**. S1's questions never reference S2's or S10's videos.
- Within a participant's timeline, questions are either **local** (anchored to a single session or event) or **global** (spanning multiple sessions of that participant).
- Questions adhere to **temporal precedence**: an answering agent is only evaluated against visual context occurring *prior* to the question cutoff.

### How long is the data for the "Object and Location Memory" task?
- **Total Dataset**: 52.9 hours across 10 participants (~3 to 12 hours per participant).
- **Total Questions**: 4,853 human-verified multiple-choice questions.
- **Object and Location Memory**: Accounts for **~19% of the dataset** (approximately **900–925 questions**).
- **Split Breakdown**:
  - **Train (S1–S6)**: ~32 hours total video, ~580 object/location questions.
  - **Validation (S7–S8)**: ~8 hours total video, ~140 object/location questions.
  - **Test (S9–S10)**: ~12.9 hours total video, ~180 object/location questions.
- **For running on Kaggle**: You do **not** need to process 53 hours! When evaluating on a test participant (e.g., S9), you only need to process **~3 to 5 hours of video** (or even a single 30–60 minute session during development).

---

## 2. Architecture & Teammate Integration

Your teammates working on frame sampling, unsupervised memory retrieval (Plans A/B/C), or supervised retrieval can interface directly with this pipeline:

```
[ Teammates: Video File / Sampled Frames ] + [ SuperMemory-VQA Question & Options ]
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │       VLMPipeline         │
                        │  (Qwen2-VL / SmolVLM /    │
                        │   Mock for local testing) │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │      VLMAnswerOutput      │
                        │ - predicted_choice: "A"   │
                        │ - predicted_text: "..."   │
                        │ - abstain: false          │
                        │ - reasoning: "..."        │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │        Evaluator          │
                        │ - Exact Match Accuracy    │
                        │ - Abstention & Recall     │
                        │ - Hallucination Rate      │
                        │ - Diagnostic Reports      │
                        └───────────────────────────┘
```

### Python API Usage
```python
from hindsight.data.schema import QuestionRecord
from hindsight.vlm.pipeline import VLMPipeline

# 1. Initialize Pipeline (use "hf" for open-source model on GPU, or "mock" for tests)
pipeline = VLMPipeline(backend="hf", model_name="Qwen/Qwen2-VL-2B-Instruct")

# 2. Teammates pass either:
#    a) Video file path: "session_01.mp4"
#    b) Sampled PIL Images: [frame1, frame2, ...]
#    c) List of frame filepaths: ["frame_01.jpg", "frame_02.jpg"]
output = pipeline.answer_question(
    question=question_record,
    video_input=sampled_frames
)

print(output.predicted_choice) # "A", "B", "C", or "D"
print(output.predicted_text)   # Text of selected option
print(output.abstain)          # True if chose "Unanswerable"
print(output.reasoning)        # Visual explanation
```

---

## 3. Running the Demo

A fully self-contained demonstrative script runs locally without external downloads or GPU requirements:

```bash
python3 scripts/demo_vlm_pipeline.py
```

### Running Unit Tests
```bash
python3 -m unittest discover tests
```

---

## 4. Running on Kaggle with Open-Source VLMs

See [`kaggle_setup.md`](./kaggle_setup.md) for full Kaggle setup instructions, recommended open-source models, and memory optimization flags.
