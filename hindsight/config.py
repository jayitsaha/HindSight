"""
Configuration module for the Hindsight VLM pipeline and evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class VLMConfig:
    """Configuration for VLM backend."""
    backend: str = "mock"  # "mock", "gemini", "hf"
    model_name: str = "gemini-1.5-flash"  # e.g., "gemini-1.5-flash", "Qwen/Qwen2-VL-7B-Instruct", "mock"
    temperature: float = 0.0
    max_output_tokens: int = 512
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    system_instruction: Optional[str] = None
    allow_abstention: bool = True
    device: str = "cpu"  # "cuda", "mps", "cpu"


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
