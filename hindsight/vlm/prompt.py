"""
Prompt formatting and parsing utilities for open-source VLMs on SuperMemory-VQA.
"""

from typing import Dict, List, Optional, Tuple
import re
import json

from hindsight.data.schema import is_unanswerable_text


SYSTEM_PROMPT_QA = """You are an expert egocentric visual memory assistant.
You are given visual evidence from a first-person wearable camera recording and a multiple-choice question about objects and their locations or interactions.

Your instructions:
1. Examine the visual evidence (video frames) carefully.
2. Determine which multiple-choice option (A, B, C, D, etc.) is factually supported by the visual evidence.
3. If the visual evidence is missing, ambiguous, or does NOT contain enough information to confirm the answer, select the option stating that the question cannot be answered (e.g. 'Unanswerable' or 'This question can not be answered.'). Do NOT guess or hallucinate.
4. Output your answer in valid JSON format with the following keys:
{
  "selected_choice": "<LETTER, e.g. A, B, C, or D>",
  "reasoning": "<Short explanation citing specific visual cues/timestamps seen>",
  "is_unanswerable": <true or false>
}
"""

USER_PROMPT_TEMPLATE = """Question: {question}

Options:
{options}

{evidence_note}select the best option.
Respond ONLY with the JSON object:
"""


def build_qa_prompt(question: str, formatted_options: str, has_frames: bool = True) -> str:
    """Builds the textual user prompt for the VLM (has_frames=False for the text-only control)."""
    note = ("Based strictly on the provided video frames, " if has_frames
            else "No video is available for this question; ")
    return USER_PROMPT_TEMPLATE.format(
        question=question,
        options=formatted_options,
        evidence_note=note,
    )


def parse_vlm_response(raw_text: str, choices: List[str]) -> Tuple[str, int, str, str, bool]:
    """
    Parses the raw VLM text response to extract:
    - selected_letter (e.g. "A", "B", "C", "D")
    - selected_idx (0-indexed integer)
    - selected_text (text of the chosen option)
    - reasoning (explanation)
    - abstain (bool, True if unanswerable or abstained)
    """
    cleaned = raw_text.strip()
    selected_letter = None
    reasoning = ""
    is_unanswerable = False

    # 1. Try JSON parsing
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if "selected_choice" in data:
                candidate = str(data["selected_choice"]).strip().upper()
                # Clean up e.g. "A)" -> "A"
                candidate = re.sub(r'[^A-Z]', '', candidate)
                if candidate and len(candidate) == 1:
                    selected_letter = candidate
            if "reasoning" in data:
                reasoning = str(data["reasoning"]).strip()
            if "is_unanswerable" in data:
                is_unanswerable = bool(data["is_unanswerable"])
        except Exception:
            pass

    # 2. Fallback regex for "Answer: A" or "Choice A" or single letter
    if not selected_letter:
        patterns = [
            r'(?:selected_choice|answer|choice|option)[\s\:\"]*([A-D])\b',
            r'\b([A-D])\)',
            r'^\s*([A-D])\b',
            r'\b([A-D])\s*$'
        ]
        for pat in patterns:
            match = re.search(pat, cleaned, re.IGNORECASE | re.MULTILINE)
            if match:
                selected_letter = match.group(1).upper()
                break

    # 3. Fallback: match text of choices directly in output
    if not selected_letter:
        for i, choice in enumerate(choices):
            clean_choice = re.sub(r'^[A-D][\.\)]\s*', '', choice).strip().lower()
            if clean_choice and len(clean_choice) > 3 and clean_choice in cleaned.lower():
                selected_letter = chr(ord('A') + i)
                break

    # Do not turn malformed output into a plausible answer. The caller records
    # INVALID as an incorrect prediction; silently choosing A can create a
    # false correct result when the ground truth happens to be A.
    if not selected_letter:
        reasoning = f"Could not parse valid choice from response: '{cleaned[:100]}...'"
        return "INVALID", -1, "", reasoning, False

    # Convert letter to 0-based index
    idx = ord(selected_letter) - ord('A')
    if idx < 0 or idx >= len(choices):
        reasoning = f"Parsed choice {selected_letter} is outside the {len(choices)} available choices."
        return "INVALID", -1, "", reasoning, False

    selected_text = choices[idx]

    # Check if choice is unanswerable
    if is_unanswerable_text(selected_text):
        is_unanswerable = True

    return selected_letter, idx, selected_text, reasoning, is_unanswerable
