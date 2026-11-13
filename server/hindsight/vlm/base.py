"""
Abstract Base Class for Vision-Language Model (VLM) backends.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
from PIL import Image

from hindsight.data.schema import QuestionRecord, VLMAnswerOutput


class BaseVLM(ABC):
    """Abstract base class that all VLM backends implement."""

    @abstractmethod
    def predict(
        self,
        video_input: Union[str, List[str], List[Image.Image], Image.Image],
        question: QuestionRecord,
        **kwargs: Any
    ) -> VLMAnswerOutput:
        """
        Takes video/frames input and a QuestionRecord with multiple choices,
        and returns a structured VLMAnswerOutput.

        Args:
            video_input: Path to video file, list of frame image paths,
                         list of PIL images, or a single PIL image.
            question: QuestionRecord containing question, choices, etc.
            **kwargs: Extra backend-specific parameters (e.g. max_frames, temperature).

        Returns:
            VLMAnswerOutput containing predicted_choice, reasoning, confidence, etc.
        """
        pass
