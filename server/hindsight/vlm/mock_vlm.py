"""
Deterministic Mock VLM for offline demonstration, testing, and CI.
"""

from typing import Any, List, Optional, Union
import math
import time
from PIL import Image

from hindsight.vlm.base import BaseVLM
from hindsight.data.schema import QuestionRecord, VLMAnswerOutput
from hindsight.vlm.prompt import parse_vlm_response


class MockVLM(BaseVLM):
    """
    Mock VLM implementation that simulates model inference without requiring a GPU or weights.
    Can be configured to act as:
    - Oracle (always picks ground truth)
    - Realistic (picks based on keyword overlap or simulated accuracy)
    - Abstaining (favors unanswerable when configured)
    """

    def __init__(
        self,
        mode: str = "realistic",  # "oracle", "realistic", "abstain"
        simulated_accuracy: float = 0.85,
        latency_seconds: float = 0.05
    ):
        self.mode = mode
        self.simulated_accuracy = simulated_accuracy
        self.latency_seconds = latency_seconds

    def predict(
        self,
        video_input: Union[str, List[str], List[Image.Image], Image.Image],
        question: QuestionRecord,
        **kwargs: Any
    ) -> VLMAnswerOutput:
        start_time = time.time()
        if self.latency_seconds > 0:
            time.sleep(self.latency_seconds)

        # Inspect visual input summary
        if isinstance(video_input, str):
            visual_info = f"video file '{video_input}'"
        elif isinstance(video_input, list):
            visual_info = f"{len(video_input)} sampled frames"
        else:
            visual_info = "single frame"

        # Determine choice
        choices = question.choices
        gt_idx = question.answer_idx

        if self.mode == "oracle":
            chosen_idx = gt_idx
            reasoning = f"Visual evidence in {visual_info} clearly corroborates choice {question.get_choice_label(chosen_idx)}."
        elif self.mode == "abstain":
            # Find if an unanswerable choice exists
            unans_idx = None
            for idx, c in enumerate(choices):
                if "unanswerable" in c.lower():
                    unans_idx = idx
                    break
            chosen_idx = unans_idx if unans_idx is not None else 0
            reasoning = f"Visual evidence in {visual_info} is insufficient or missing required object interaction."
        else:
            # Realistic mode: deterministic pseudo-hash based on question_id
            h = hash(question.question_id) % 100
            if h < int(self.simulated_accuracy * 100):
                chosen_idx = gt_idx
                reasoning = (
                    f"Identified target object in {visual_info} at recent location. "
                    f"Selected {question.get_choice_label(chosen_idx)}: '{choices[chosen_idx]}'."
                )
            else:
                # pick distractor
                chosen_idx = (gt_idx + 1) % len(choices)
                reasoning = (
                    f"Visual ambiguity in {visual_info}. "
                    f"Guessed choice {question.get_choice_label(chosen_idx)}: '{choices[chosen_idx]}'."
                )

        chosen_letter = question.get_choice_label(chosen_idx)
        chosen_text = choices[chosen_idx]
        is_unanswerable = "unanswerable" in chosen_text.lower()

        simulated_raw = (
            f'{{\n  "selected_choice": "{chosen_letter}",\n'
            f'  "reasoning": "{reasoning}",\n'
            f'  "is_unanswerable": {str(is_unanswerable).lower()}\n}}'
        )

        # deterministic pseudo log-probs: the chosen option gets most of the mass, how much varies by question
        peak = 0.4 + 0.55 * ((hash(question.question_id) // 7) % 100) / 100
        rest = (1 - peak) / max(1, len(choices) - 1)
        option_logprobs = {question.get_choice_label(i): math.log(peak if i == chosen_idx else rest)
                           for i in range(len(choices))}
        n_frames = len(video_input) if isinstance(video_input, list) else 1

        elapsed = time.time() - start_time
        return VLMAnswerOutput(
            question_id=question.question_id,
            predicted_choice=chosen_letter,
            predicted_choice_idx=chosen_idx,
            predicted_text=chosen_text,
            reasoning=reasoning,
            abstain=is_unanswerable,
            confidence=0.92 if chosen_idx == gt_idx else 0.45,
            raw_model_response=simulated_raw,
            latency_seconds=elapsed,
            option_logprobs=option_logprobs,
            n_frames=n_frames,
        )
