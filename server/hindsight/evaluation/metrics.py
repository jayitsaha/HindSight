"""
Metrics calculation for SuperMemory-VQA and Hindsight evaluation.
"""

from typing import Dict, List, Tuple
from collections import defaultdict

from hindsight.data.schema import (
    EvaluationSample, EvaluationSummary, QuestionRecord, VLMAnswerOutput, is_unanswerable_text
)


def compute_sample_evaluation(
    question: QuestionRecord,
    prediction: VLMAnswerOutput,
    partial_credit_vague: float = 0.5
) -> EvaluationSample:
    """Evaluates a single question against a model prediction."""
    gt_idx = question.answer_idx
    pred_idx = prediction.predicted_choice_idx

    gt_text = question.choices[gt_idx] if 0 <= gt_idx < len(question.choices) else ""
    if question.is_answerable is not None:
        is_gt_unanswerable = not question.is_answerable
    else:
        is_gt_unanswerable = is_unanswerable_text(gt_text)
    did_abstain = prediction.abstain or is_unanswerable_text(prediction.predicted_text)

    is_valid_prediction = 0 <= pred_idx < len(question.choices)
    is_correct = is_valid_prediction and (pred_idx == gt_idx)

    # An unparseable prediction is wrong on answerability too (never rewarded as a lucky abstain/answer).
    predicted_answerable = (not did_abstain) if is_valid_prediction else is_gt_unanswerable

    is_vague = False
    if is_valid_prediction:
        if question.choice_types and pred_idx < len(question.choice_types):
            is_vague = question.choice_types[pred_idx] == "vague"
        else:
            is_vague = "vague" in question.choices[pred_idx].lower()

    # Compute score with optional partial credit for "vague" option
    score = 0.0
    if is_correct:
        score = 1.0
    elif is_vague:
        # SuperMemory-VQA diagnostic partial credit
        score = partial_credit_vague

    return EvaluationSample(
        question_id=question.question_id,
        category=question.category,
        subject_id=question.subject_id,
        question=question.question,
        choices=question.choices,
        ground_truth_idx=gt_idx,
        ground_truth_label=question.get_choice_label(gt_idx),
        predicted_idx=pred_idx,
        predicted_label=prediction.predicted_choice,
        is_correct=is_correct,
        is_unanswerable_ground_truth=is_gt_unanswerable,
        did_abstain=did_abstain,
        score=score,
        reasoning=prediction.reasoning,
        is_answerable=not is_gt_unanswerable,
        predicted_answerable=predicted_answerable,
        is_vague_choice=is_vague,
    )


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def answerability_f1(samples: List[EvaluationSample]) -> Tuple[float, float]:
    """(F1 of the 'answerable' class, macro F1 over both classes)."""
    tp = sum(1 for s in samples if s.is_answerable and s.predicted_answerable)
    fp = sum(1 for s in samples if not s.is_answerable and s.predicted_answerable)
    fn = sum(1 for s in samples if s.is_answerable and not s.predicted_answerable)
    tn = sum(1 for s in samples if not s.is_answerable and not s.predicted_answerable)
    f1_ans = _f1(tp, fp, fn)
    f1_unans = _f1(tn, fn, fp)
    return f1_ans, (f1_ans + f1_unans) / 2


def aggregate_metrics(samples: List[EvaluationSample]) -> EvaluationSummary:
    """Aggregates individual evaluation samples into an EvaluationSummary."""
    if not samples:
        return EvaluationSummary(
            total_samples=0,
            accuracy=0.0,
            score_with_partial_credit=0.0,
            abstention_accuracy=0.0,
            answerable_accuracy=0.0,
            hallucination_rate=0.0,
            over_abstention_rate=0.0
        )

    total = len(samples)
    correct_count = sum(1 for s in samples if s.is_correct)
    total_score = sum(s.score for s in samples)

    # Unanswerable vs Answerable breakdowns
    unans_samples = [s for s in samples if s.is_unanswerable_ground_truth]
    ans_samples = [s for s in samples if not s.is_unanswerable_ground_truth]

    abstention_acc = (
        sum(1 for s in unans_samples if s.is_correct) / len(unans_samples)
        if unans_samples else 1.0
    )
    answerable_acc = (
        sum(1 for s in ans_samples if s.is_correct) / len(ans_samples)
        if ans_samples else 1.0
    )

    # Hallucination: question was unanswerable, but model answered something else
    hallucinations = sum(1 for s in unans_samples if not s.did_abstain)
    hallucination_rate = hallucinations / len(unans_samples) if unans_samples else 0.0

    # Over-abstention: question was answerable, but model abstained
    over_abstentions = sum(1 for s in ans_samples if s.did_abstain)
    over_abstention_rate = over_abstentions / len(ans_samples) if ans_samples else 0.0

    # Per-subject accuracy
    subject_counts = defaultdict(int)
    subject_correct = defaultdict(int)
    for s in samples:
        subj = s.subject_id or "unknown"
        subject_counts[subj] += 1
        if s.is_correct:
            subject_correct[subj] += 1
    per_subject_acc = {
        subj: subject_correct[subj] / subject_counts[subj]
        for subj in subject_counts
    }

    # Per-category accuracy
    category_counts = defaultdict(int)
    category_correct = defaultdict(int)
    for s in samples:
        cat = s.category or "default"
        category_counts[cat] += 1
        if s.is_correct:
            category_correct[cat] += 1
    per_category_acc = {
        cat: category_correct[cat] / category_counts[cat]
        for cat in category_counts
    }

    # Confusion matrix
    conf_matrix = defaultdict(lambda: defaultdict(int))
    for s in samples:
        conf_matrix[s.ground_truth_label][s.predicted_label] += 1
    # Convert to normal dict
    clean_conf_matrix = {k: dict(v) for k, v in conf_matrix.items()}

    f1_ans, f1_macro = answerability_f1(samples)
    vague_rate = (sum(1 for s in ans_samples if s.is_vague_choice) / len(ans_samples)) if ans_samples else 0.0
    invalid_rate = sum(1 for s in samples if s.predicted_idx < 0) / total

    return EvaluationSummary(
        total_samples=total,
        accuracy=round(correct_count / total, 4),
        score_with_partial_credit=round(total_score / total, 4),
        abstention_accuracy=round(abstention_acc, 4),
        answerable_accuracy=round(answerable_acc, 4),
        hallucination_rate=round(hallucination_rate, 4),
        over_abstention_rate=round(over_abstention_rate, 4),
        per_subject_accuracy={k: round(v, 4) for k, v in per_subject_acc.items()},
        per_category_accuracy={k: round(v, 4) for k, v in per_category_acc.items()},
        confusion_matrix=clean_conf_matrix,
        answerability_f1=round(f1_ans, 4),
        answerability_macro_f1=round(f1_macro, 4),
        vague_rate=round(vague_rate, 4),
        invalid_rate=round(invalid_rate, 4),
    )
