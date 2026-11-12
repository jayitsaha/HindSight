# Kaggle Setup & Open-Source VLM Guide

This guide details how to run the Hindsight VLM pipeline on Kaggle using free GPU instances (NVIDIA T4 or P100).

## 1. Kaggle Environment Settings
- **Accelerator**: GPU T4 x2 or GPU P100
- **Internet**: Enabled (for downloading Hugging Face models and dataset)
- **Persistence**: Files / Output

## 2. Dependencies Installation
In your Kaggle notebook, run:

```bash
!pip install -q torch torchvision transformers accelerate qwen-vl-utils opencv-python pillow
```

## 3. Recommended Open-Source VLMs for Kaggle

| Model | Size | VRAM Usage | Multi-Frame Support | Best For |
|---|---|---|---|---|
| **`Qwen/Qwen2-VL-2B-Instruct`** | ~2 Billion | ~4.5 GB (bfloat16) | Native video & multi-image | **Fastest, highly accurate, easily fits 16GB GPU** |
| **`Qwen/Qwen2-VL-7B-Instruct`** | ~7 Billion | ~14.5 GB (4-bit) | Native video & multi-image | Highest accuracy if using 4-bit quantization |
| **`HuggingFaceTB/SmolVLM-Instruct`** | ~2.2 Billion | ~4 GB (bfloat16) | Native multi-image | Extremely lightweight |

> **Recommendation**: Start with `Qwen/Qwen2-VL-2B-Instruct`. It is fast, consumes less than 5 GB of VRAM, handles multi-frame egocentric streams natively, and does not exhaust GPU quotas.

---

## 4. Running the VLM Pipeline on Kaggle

```python
from hindsight.vlm.pipeline import VLMPipeline
from hindsight.data.loader import SuperMemoryDataLoader
from hindsight.evaluation.evaluator import Evaluator

# 1. Initialize Open-Source VLM Backend
pipeline = VLMPipeline(
    backend="hf",
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    device="cuda"
)

# 2. Load Questions for Object and Location Memory
loader = SuperMemoryDataLoader()
# To evaluate on a subset (e.g. test subject S9, first 20 questions):
loader.config.subject_filter = ["S9"]
loader.config.max_questions = 20
questions = loader.load_questions("path/to/supermemory_vqa.json")

# 3. Teammates provide sampled frames or video paths:
# visual_inputs = [list_of_frames_q1, list_of_frames_q2, ...]

# 4. Run Evaluation
evaluator = Evaluator()
summary = evaluator.evaluate_pipeline(pipeline, questions, visual_inputs)
print(summary.to_markdown())
```

---

## 5. Running via CLI on Kaggle
```bash
python scripts/run_evaluation.py \
  --backend hf \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --data-path /kaggle/input/supermemory-vqa/questions.json \
  --task-category "Object and Location Memory" \
  --subject S9 \
  --max-questions 50 \
  --output-dir /kaggle/working/results
```
