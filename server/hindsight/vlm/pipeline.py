"""
Unified VLM Pipeline Orchestrator.
Exposes a clean, simple API for teammates to send video/frames + question and get back answers.
"""

from typing import Any, Dict, List, Optional, Union
import os
import json
from PIL import Image

from hindsight.vlm.config import VLMConfig
from hindsight.data.schema import QuestionRecord, VLMAnswerOutput
from hindsight.vlm.base import BaseVLM
from hindsight.vlm.mock_vlm import MockVLM


def create_vlm_backend(config: VLMConfig) -> BaseVLM:
    """Factory function to instantiate the configured VLM backend."""
    backend_type = config.backend.lower()

    if backend_type == "mock":
        return MockVLM()
    elif backend_type in ["hf", "huggingface", "qwen", "qwen2", "smolvlm", "gemma"]:
        from hindsight.vlm.hf_vlm import HuggingFaceVLM
        return HuggingFaceVLM(
            model_name_or_path=config.model_name,
            device=config.device,
            temperature=config.temperature,
            max_new_tokens=config.max_output_tokens,
            max_frames=config.max_frames,
            image_token_budget=config.image_token_budget,
        )
    else:
        raise ValueError(f"Unknown VLM backend: '{config.backend}'. Supported: 'mock', 'hf'.")


class VLMPipeline:
    """
    Main VLM Pipeline interface.

    Usage:
        pipeline = VLMPipeline(backend="mock")  # or "hf" with "Qwen/Qwen2-VL-2B-Instruct"
        answer = pipeline.answer_question(question_record, video_input="session_1.mp4")
        print(answer.predicted_choice, answer.reasoning)
    """

    def __init__(
        self,
        config: Optional[VLMConfig] = None,
        backend: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any
    ):
        if config is None:
            config = VLMConfig()
        if backend is not None:
            config.backend = backend
        if model_name is not None:
            config.model_name = model_name

        self.config = config
        self.backend: BaseVLM = create_vlm_backend(config)
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, question_id: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in question_id)
        return os.path.join(self.cache_dir, f"{safe_id}.json")

    def answer_question(
        self,
        question: QuestionRecord,
        video_input: Union[str, List[str], List[Image.Image], Image.Image],
        use_cache: bool = True,
        **kwargs: Any
    ) -> VLMAnswerOutput:
        """
        Answers a single question given visual evidence (video or sampled frames).

        Args:
            question: QuestionRecord containing question text, choices, etc.
            video_input: Path to video file, list of image paths, list of PIL Images, or single Image.
            use_cache: Whether to return cached predictions if available.

        Returns:
            VLMAnswerOutput with predicted choice, text, reasoning, and abstention status.
        """
        cache_path = self._get_cache_path(question.question_id)
        if use_cache and cache_path and os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return VLMAnswerOutput(**data)
            except Exception:
                pass

        # Invoke model
        output = self.backend.predict(video_input=video_input, question=question, **kwargs)

        # Cache output if configured
        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(output.to_dict(), f, indent=2)
            except Exception:
                pass

        return output

    def batch_answer(
        self,
        questions: List[QuestionRecord],
        video_inputs: List[Union[str, List[str], List[Image.Image], Image.Image]],
        use_cache: bool = True,
        **kwargs: Any
    ) -> List[VLMAnswerOutput]:
        """
        Batch answers a list of questions with their corresponding video inputs.
        """
        if len(questions) != len(video_inputs):
            raise ValueError(f"Length mismatch: {len(questions)} questions vs {len(video_inputs)} video inputs.")

        results = []
        for q, v in zip(questions, video_inputs):
            ans = self.answer_question(q, v, use_cache=use_cache, **kwargs)
            results.append(ans)
        return results
