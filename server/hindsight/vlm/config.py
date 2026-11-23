"""
Configuration module for the Hindsight VLM pipeline and evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VLMConfig:
    """Configuration for VLM backend."""
    backend: str = "mock"  # "mock" or "hf"
    model_name: str = "google/gemma-4-E4B-it"  # any Hugging Face VLM id (SmolVLM, Pixtral, ...) for backend="hf"
    temperature: float = 0.0
    max_output_tokens: int = 512
    system_instruction: Optional[str] = None
    allow_abstention: bool = True
    device: str = "cpu"  # "auto", "cuda", "mps", "cpu"
    # Frames handed to the model. A raw video path is sampled down to this many evenly spaced frames;
    # a list of frames longer than this is subsampled. None = no cap (use for already-curated K-frame inputs).
    max_frames: Optional[int] = 16
    # Gemma 4 visual token budget per image (max_soft_tokens): one of 70, 140, 280, 560, 1120.
    # None = the processor default (280). Ignored by other models.
    image_token_budget: Optional[int] = None


@dataclass
class DataConfig:
    """Configuration for SuperMemory-VQA data loader."""
    dataset_name_or_path: str = "OSU-AIoT-MLSys-Lab/SuperMemory-VQA"
    task_category: str = "Object and Location Memory"  # target task
    split: str = "test"
    subject_filter: Optional[List[str]] = None  # e.g., ["S9", "S10"] or ["S1"]
    max_questions: Optional[int] = None  # limit for dev/testing on Kaggle
    cache_dir: str = ".cache/supermemory"


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics and logging."""
    output_dir: str = "results/eval"
    save_predictions: bool = True
    partial_credit_vague: float = 0.5  # SuperMemory-VQA diagnostic credit for vague option
    penalize_hallucination: bool = True
    verbose: bool = True


@dataclass
class AppConfig:
    """Top-level application config."""
    vlm: VLMConfig = field(default_factory=VLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
