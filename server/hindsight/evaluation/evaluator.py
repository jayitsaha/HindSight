"""
Evaluator orchestrator for running evaluations, logging metrics, and generating reports.
"""

from typing import Any, Dict, List, Optional, Union
import os
import json
from dataclasses import asdict

from hindsight.vlm.config import EvalConfig
from hindsight.data.schema import EvaluationSample, EvaluationSummary, QuestionRecord, VLMAnswerOutput
from hindsight.vlm.pipeline import VLMPipeline
from hindsight.evaluation.metrics import compute_sample_evaluation, aggregate_metrics


class Evaluator:
    """
    Orchestrates the evaluation of a VLM Pipeline on SuperMemory-VQA questions.
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def evaluate_pipeline(
        self,
        pipeline: VLMPipeline,
        questions: List[QuestionRecord],
        visual_inputs: List[Any],
        **kwargs: Any
    ) -> EvaluationSummary:
        """
        Runs the VLM pipeline on each question and its visual input, evaluates predictions,
        and saves reports. A visual input may be a zero-argument callable; it is called only for
        questions not already in raw_predictions.jsonl, so frames are never all held in memory.
        """
        if len(questions) != len(visual_inputs):
            raise ValueError(f"Mismatch: {len(questions)} questions vs {len(visual_inputs)} visual inputs.")

        samples: List[EvaluationSample] = []
        predictions: List[VLMAnswerOutput] = []

        # Predictions are appended to raw_predictions.jsonl as they are produced, so a
        # job killed by the Slurm time limit resumes where it stopped.
        raw_path = os.path.join(self.config.output_dir, "raw_predictions.jsonl")
        done = self._load_raw_predictions(raw_path)
        print(f"[Evaluator] Starting evaluation on {len(questions)} questions ({len(done)} already done)...")
        for i, (q, v_input) in enumerate(zip(questions, visual_inputs)):
            if q.question_id in done:
                pred = done[q.question_id]
            else:
                if self.config.verbose and (i % 10 == 0 or i == len(questions) - 1):
                    print(f"[Evaluator] Processing question {i + 1}/{len(questions)}: ID '{q.question_id}'")
                failed = False
                try:
                    if callable(v_input):  # lazy input (e.g. curated frames): load only when needed
                        v_input = v_input()
                    pred = pipeline.answer_question(q, v_input, **kwargs)
                except Exception as e:  # one bad clip must not sink the whole run
                    failed = True
                    print(f"[Evaluator] Question '{q.question_id}' failed: {e!r}")
                    pred = VLMAnswerOutput(
                        question_id=q.question_id, predicted_choice="INVALID", predicted_choice_idx=-1,
                        predicted_text="", reasoning=f"Inference error: {e!r}", abstain=False,
                    )
                if not failed:  # errors are not persisted, so a rerun retries them
                    with open(raw_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(pred.to_dict()) + "\n")
            predictions.append(pred)

            sample = compute_sample_evaluation(
                question=q,
                prediction=pred,
                partial_credit_vague=self.config.partial_credit_vague
            )
            samples.append(sample)

        summary = aggregate_metrics(samples)
        self._save_results(summary, samples, predictions)
        return summary

    @staticmethod
    def _load_raw_predictions(path: str) -> Dict[str, VLMAnswerOutput]:
        done: Dict[str, VLMAnswerOutput] = {}
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        done[str(d["question_id"])] = VLMAnswerOutput(**d)
        return done

    def evaluate_predictions(
        self,
        questions: List[QuestionRecord],
        predictions: List[VLMAnswerOutput]
    ) -> EvaluationSummary:
        """Evaluates already-computed predictions against ground truth questions."""
        samples = []
        for q, p in zip(questions, predictions):
            s = compute_sample_evaluation(q, p, partial_credit_vague=self.config.partial_credit_vague)
            samples.append(s)
        summary = aggregate_metrics(samples)
        self._save_results(summary, samples, predictions)
        return summary

    def _save_results(
        self,
        summary: EvaluationSummary,
        samples: List[EvaluationSample],
        predictions: List[VLMAnswerOutput]
    ):
        """Saves evaluation summary, detailed samples, and predictions to disk."""
        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # 1. Save Markdown report
        md_path = os.path.join(out_dir, "eval_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary.to_markdown())

        # 2. Save JSON metrics
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)

        # 3. Save detailed predictions JSONL
        if self.config.save_predictions:
            preds_path = os.path.join(out_dir, "predictions.jsonl")
            with open(preds_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(asdict(sample)) + "\n")

        print(f"[Evaluator] Results saved to: {out_dir}")
        print(f"[Evaluator] Accuracy: {summary.accuracy * 100:.2f}% | Score w/ partial credit: {summary.score_with_partial_credit * 100:.2f}%")
