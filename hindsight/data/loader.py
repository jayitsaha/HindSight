"""
Dataset loader and normalizer for SuperMemory-VQA.
Filters specifically for Object and Location Memory tasks.
"""

from typing import Any, Dict, List, Optional
import os
import json
import re

from hindsight.config import DataConfig
from hindsight.data.schema import QuestionRecord


def _clip(video_id: str, span: Dict[str, Any]) -> Dict[str, Any]:
    return {"video_id": video_id, "start": float(span["start_time"]), "end": float(span["end_time"])}


def extract_evidence_clips(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Oracle evidence locations for a real SuperMemory-VQA item: the answer-evidence
    spans when present, otherwise the span where the question was asked.
    """
    clips = []
    ans_ev = item.get("answer_evidence") or {}
    for ev in ans_ev.get("evidence_list") or []:
        if ev.get("video_id") and ev.get("time_span"):
            clips.append(_clip(ev["video_id"], ev["time_span"]))
    if not clips:
        q_ev = item.get("question_evidence") or {}
        for span in q_ev.get("time_spans") or []:
            if span.get("video_id"):
                clips.append(_clip(span["video_id"], span))
    return clips


# Built-in demonstrative sample data for rapid testing without dataset download
DEMO_SAMPLES = [
    {
        "question_id": "sm_vqa_obj_001",
        "category": "Object and Location Memory",
        "subject_id": "S1",
        "session_id": "session_01",
        "question": "Where did I put down my keys after entering the house?",
        "options": [
            "A) On the wooden console table in the entryway",
            "B) In the entryway area",
            "C) On the kitchen dining table",
            "D) Unanswerable"
        ],
        "answer": "A",
        "evidence": [{"start": "02:15", "end": "02:30", "session": "session_01", "text": "Keys placed on console table"}]
    },
    {
        "question_id": "sm_vqa_obj_002",
        "category": "Object and Location Memory",
        "subject_id": "S1",
        "session_id": "session_01",
        "question": "Where was my coffee mug left before I walked into the home office?",
        "options": [
            "A) On the kitchen counter next to the electric kettle",
            "B) In the kitchen",
            "C) On the living room coffee table",
            "D) Unanswerable"
        ],
        "answer": "A",
        "evidence": [{"start": "14:40", "end": "15:05", "session": "session_01", "text": "Coffee mug set by kettle"}]
    },
    {
        "question_id": "sm_vqa_obj_003",
        "category": "Object and Location Memory",
        "subject_id": "S2",
        "session_id": "session_03",
        "question": "Where did I place my reading glasses after sitting on the sofa?",
        "options": [
            "A) On the glass side table next to the floor lamp",
            "B) Somewhere in the living room",
            "C) Inside the bookshelf drawer",
            "D) Unanswerable"
        ],
        "answer": "A",
        "evidence": [{"start": "08:12", "end": "08:25", "session": "session_03", "text": "Glasses on side table"}]
    },
    {
        "question_id": "sm_vqa_obj_004",
        "category": "Object and Location Memory",
        "subject_id": "S2",
        "session_id": "session_04",
        "question": "Where did I leave my blue water bottle on Thursday morning?",
        "options": [
            "A) In the car cup holder",
            "B) Near the vehicle",
            "C) Inside the gym backpack",
            "D) Unanswerable"
        ],
        "answer": "D",  # Ground truth unanswerable (testing epistemic calibration)
        "evidence": []
    },
    {
        "question_id": "sm_vqa_obj_005",
        "category": "Object and Location Memory",
        "subject_id": "S9",
        "session_id": "session_09",
        "question": "Where was the TV remote placed during the evening movie?",
        "options": [
            "A) On the central coffee table between the magazines",
            "B) In the media room",
            "C) Under the sofa cushion",
            "D) Unanswerable"
        ],
        "answer": "A",
        "evidence": [{"start": "35:10", "end": "35:25", "session": "session_09", "text": "Remote placed on coffee table"}]
    }
]


class SuperMemoryDataLoader:
    """Loads and filters questions from SuperMemory-VQA."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    def _normalize_item(self, item: Dict[str, Any], idx: int) -> QuestionRecord:
        """Normalizes a raw dictionary into a QuestionRecord."""
        q_id = str(item.get("question_id", item.get("id", f"q_{idx}")))
        question_text = item.get("question", item.get("query", ""))
        metadata = item.get("metadata") or {}
        skill = metadata.get("skill")
        category = item.get("category", item.get("task", None))
        if category is None:
            category = skill.replace("_", " ").title().replace("Object Location", "Object and Location") if skill else "Object and Location Memory"
        subject_id = item.get("subject_id", item.get("participant", item.get("subject", None)))
        if isinstance(subject_id, int):
            subject_id = f"S{subject_id}"
        session_id = item.get("session_id", item.get("session", None))
        video_id = item.get("video_id", metadata.get("primary_video_id", session_id))
        evidence_clips = extract_evidence_clips(item) if "answer_evidence" in item or "question_evidence" in item else None

        # Handle choices / options
        raw_options = item.get("options", item.get("choices", []))
        if isinstance(raw_options, dict):
            # Sort by key e.g. "A", "B", "C", "D"
            choices = [f"{k}) {v}" for k, v in sorted(raw_options.items())]
        elif isinstance(raw_options, list):
            choices = [str(c) for c in raw_options]
        else:
            choices = ["Option A", "Option B", "Option C", "Unanswerable"]

        # Handle answer
        raw_ans = item.get("correct_option_index", item.get("answer", item.get("correct_choice", item.get("label", "A"))))
        answer_idx = 0
        if isinstance(raw_ans, int):
            answer_idx = raw_ans
            ans_str = chr(ord('A') + answer_idx)
        else:
            ans_str = str(raw_ans).strip().upper()
            if len(ans_str) == 1 and 'A' <= ans_str <= 'Z':
                answer_idx = ord(ans_str) - ord('A')
            else:
                # Search if text matches one of the choices
                for c_i, c_text in enumerate(choices):
                    if ans_str.lower() in c_text.lower():
                        answer_idx = c_i
                        ans_str = chr(ord('A') + c_i)
                        break

        # Ensure index in bounds
        if answer_idx >= len(choices):
            answer_idx = 0
            ans_str = "A"

        return QuestionRecord(
            question_id=q_id,
            question=question_text,
            choices=choices,
            answer=ans_str,
            answer_idx=answer_idx,
            category=category,
            subject_id=subject_id,
            session_id=session_id,
            video_id=video_id,
            evidence=item.get("evidence", item.get("evidence_timestamps", None)),
            cutoff_timestamp=item.get("cutoff_timestamp", item.get("start_time", None)),
            raw=item,
            is_answerable=item.get("is_answerable"),
            choice_types=item.get("choice_types"),
            video_ids=item.get("video_ids"),
            skill=skill,
            evidence_clips=evidence_clips,
        )

    def load_questions(
        self,
        source_path: Optional[str] = None,
        use_demo_if_missing: bool = True
    ) -> List[QuestionRecord]:
        """
        Loads questions from file path or directly from Hugging Face dataset.
        Filters for the configured task category ('Object and Location Memory').
        """
        # If source_path is None or a placeholder string like "path/to/..."
        if not source_path or "path/to/" in source_path:
            path = self.config.dataset_name_or_path
        else:
            path = source_path

        raw_items: List[Dict[str, Any]] = []

        if path and os.path.exists(path):
            print(f"[DataLoader] Loading from local file: {path}")
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        raw_items = data
                    elif isinstance(data, dict) and "questions" in data:
                        raw_items = data["questions"]
            elif path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    raw_items = [json.loads(line) for line in f if line.strip()]
        else:
            # Try loading directly via HuggingFace datasets library
            hf_repo = path if "/" in path and not os.path.isabs(path) else self.config.dataset_name_or_path
            try:
                from datasets import load_dataset
                print(f"[DataLoader] Fetching questions directly from Hugging Face: '{hf_repo}'...")
                ds = load_dataset(hf_repo)
                if hasattr(ds, "keys"):
                    # DatasetDict: pick split
                    split_name = self.config.split if self.config.split in ds else list(ds.keys())[0]
                    print(f"[DataLoader] Using split: '{split_name}' (total records in split: {len(ds[split_name])})")
                    raw_items = [dict(x) for x in ds[split_name]]
                else:
                    raw_items = [dict(x) for x in ds]
                print(f"[DataLoader] Successfully retrieved {len(raw_items)} questions from Hugging Face!")
            except Exception as e:
                print(f"[DataLoader] Note: Could not load from Hugging Face ({e}).")
                if use_demo_if_missing:
                    print("[DataLoader] Using built-in demonstrative benchmark dataset (5 sample questions).")
                    raw_items = DEMO_SAMPLES
                else:
                    raise FileNotFoundError(f"Could not load SuperMemory-VQA from: {path}") from e

        records: List[QuestionRecord] = []
        for i, item in enumerate(raw_items):
            rec = self._normalize_item(item, i)

            # Filter by task category
            if self.config.task_category:
                if self.config.task_category.lower() not in rec.category.lower():
                    continue

            # Filter by subject if specified
            if self.config.subject_filter:
                if rec.subject_id and rec.subject_id not in self.config.subject_filter:
                    continue

            records.append(rec)

            if self.config.max_questions and len(records) >= self.config.max_questions:
                break

        print(f"[DataLoader] Loaded {len(records)} questions (category='{self.config.task_category}')")
        return records
