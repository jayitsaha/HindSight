"""
Core data schemas for the Hindsight VLM pipeline and evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


def is_unanswerable_text(text: str) -> bool:
    """True for the dataset's abstention option ("This question can not be answered.") or demo 'Unanswerable'."""
    t = text.lower()
    return "unanswerable" in t or "can not be answered" in t or "cannot be answered" in t


@dataclass
class QuestionRecord:
    """Represents a question from SuperMemory-VQA or a user query."""
    question_id: str
    question: str
    choices: List[str]  # ordered options (e.g. [A, B, C, D])
    answer: str  # ground truth answer label (e.g. "A" or option index/text)
    answer_idx: int  # 0-indexed correct option
    category: str = "Object and Location Memory"
    subject_id: Optional[str] = None  # e.g., "S1", "S9"
    video_id: Optional[str] = None
    session_id: Optional[str] = None
    video_path: Optional[str] = None  # path to source video file if local
    evidence: Optional[List[Dict[str, Any]]] = None
    cutoff_timestamp: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None
    # Real SuperMemory-VQA fields (None for hand-written demo records)
    is_answerable: Optional[bool] = None
    choice_types: Optional[List[str]] = None  # per option: correct / incorrect / vague
    video_ids: Optional[List[str]] = None  # sessions available as memory for this question
    skill: Optional[str] = None  # e.g. "object_location_memory"
    # Oracle evidence clips [{"video_id", "start", "end"}] in seconds. Only the
    # location is used to pick frames; answer_evidence text is never shown to the model.
    evidence_clips: Optional[List[Dict[str, Any]]] = None

    def get_choice_label(self, idx: int) -> str:
        """Returns option letter e.g. 'A', 'B', 'C', 'D'."""
        return chr(ord('A') + idx)

    def formatted_options(self) -> str:
        """Returns formatted string of options with letter prefixes."""
        lines = []
        for i, choice in enumerate(self.choices):
            label = self.get_choice_label(i)
            # If choice already starts with 'A)' or 'A.', don't double prefix
            if choice.strip().startswith(f"{label})") or choice.strip().startswith(f"{label}."):
                lines.append(choice.strip())
            else:
                lines.append(f"{label}) {choice.strip()}")
        return "\n".join(lines)


@dataclass
class VLMAnswerOutput:
    """Structured output returned by the VLM pipeline."""
    question_id: str
    predicted_choice: str  # e.g. "A", "B", "C", "D"
    predicted_choice_idx: int  # 0, 1, 2, 3
    predicted_text: str  # Full string of the chosen option
    reasoning: str  # Model explanation / evidence cited
    abstain: bool = False  # True if chosen option is "Unanswerable" or abstains
    confidence: Optional[float] = None
    raw_model_response: str = ""
    latency_seconds: float = 0.0
    # log-probability the model gave each option letter at the point it wrote "selected_choice"
    # ({"A": -0.1, "B": -3.2, ...}); None when the backend cannot provide it. Input to the abstention classifier.
    option_logprobs: Optional[Dict[str, float]] = None
    n_frames: Optional[int] = None  # images the model actually saw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "predicted_choice": self.predicted_choice,
            "predicted_choice_idx": self.predicted_choice_idx,
            "predicted_text": self.predicted_text,
            "reasoning": self.reasoning,
            "abstain": self.abstain,
            "confidence": self.confidence,
            "raw_model_response": self.raw_model_response,
            "latency_seconds": round(self.latency_seconds, 4),
            "option_logprobs": self.option_logprobs,
            "n_frames": self.n_frames,
        }


@dataclass
class EvaluationSample:
    """Single evaluated sample comparison."""
    question_id: str
    category: str
    subject_id: Optional[str]
    question: str
    choices: List[str]
    ground_truth_idx: int
    ground_truth_label: str
    predicted_idx: int
    predicted_label: str
    is_correct: bool
    is_unanswerable_ground_truth: bool
    did_abstain: bool
    score: float  # 1.0 for correct, 0.5 for vague (if partial credit), 0.0 for wrong
    reasoning: str = ""
    is_answerable: bool = True
    predicted_answerable: bool = True
    is_vague_choice: bool = False


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics."""
    total_samples: int
    accuracy: float
    score_with_partial_credit: float
    abstention_accuracy: float  # accuracy on questions where GT is unanswerable
    answerable_accuracy: float  # accuracy on questions where GT is answerable
    hallucination_rate: float  # answered an option when question was actually unanswerable
    over_abstention_rate: float  # abstained when question was actually answerable
    per_subject_accuracy: Dict[str, float] = field(default_factory=dict)
    per_category_accuracy: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    answerability_f1: float = 0.0  # F1 of the "answerable" class
    answerability_macro_f1: float = 0.0  # mean of answerable/unanswerable F1
    vague_rate: float = 0.0  # fraction of answerable questions where the vague option was chosen
    invalid_rate: float = 0.0  # fraction of unparseable predictions

    def to_markdown(self) -> str:
        lines = [
            "# VLM Pipeline Evaluation Report",
            "",
            f"- **Total Questions Evaluated**: {self.total_samples}",
            f"- **Exact Match Accuracy**: {self.accuracy * 100:.2f}%",
            f"- **Score (with partial credit)**: {self.score_with_partial_credit * 100:.2f}%",
            f"- **Answerable Question Accuracy**: {self.answerable_accuracy * 100:.2f}%",
            f"- **Abstention Accuracy (Calibrated Recall)**: {self.abstention_accuracy * 100:.2f}%",
            f"- **Answerability F1 (answerable class / macro)**: {self.answerability_f1 * 100:.2f}% / {self.answerability_macro_f1 * 100:.2f}%",
            f"- **Vague-option Rate**: {self.vague_rate * 100:.2f}%",
            f"- **Invalid (unparseable) Rate**: {self.invalid_rate * 100:.2f}%",
            f"- **Hallucination Rate**: {self.hallucination_rate * 100:.2f}%",
            f"- **Over-Abstention Rate**: {self.over_abstention_rate * 100:.2f}%",
            "",
            "## Accuracy by Subject",
        ]
        if self.per_subject_accuracy:
            for subj, acc in sorted(self.per_subject_accuracy.items()):
                lines.append(f"- **{subj}**: {acc * 100:.2f}%")
        else:
            lines.append("- (No subject partition)")

        lines.extend(["", "## Accuracy by Task Category"])
        if self.per_category_accuracy:
            for cat, acc in sorted(self.per_category_accuracy.items()):
                lines.append(f"- **{cat}**: {acc * 100:.2f}%")
        else:
            lines.append("- (Single category)")

        return "\n".join(lines)
