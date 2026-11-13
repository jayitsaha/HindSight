"""
Unit tests for evaluation metrics and evaluator.
"""

import unittest
from hindsight.data.schema import QuestionRecord, VLMAnswerOutput
from hindsight.evaluation.metrics import compute_sample_evaluation, aggregate_metrics


class TestEvaluationMetrics(unittest.TestCase):

    def test_metrics_calculation(self):
        q1 = QuestionRecord(
            question_id="q1",
            question="Where are the keys?",
            choices=["A) Desk", "B) Table", "C) Shelf", "D) Unanswerable"],
            answer="A",
            answer_idx=0,
            subject_id="S1"
        )
        pred1 = VLMAnswerOutput(
            question_id="q1",
            predicted_choice="A",
            predicted_choice_idx=0,
            predicted_text="A) Desk",
            reasoning="Observed on desk",
            abstain=False
        )

        q2 = QuestionRecord(
            question_id="q2",
            question="Where is the bottle?",
            choices=["A) Desk", "B) Table", "C) Shelf", "D) Unanswerable"],
            answer="D",
            answer_idx=3,
            subject_id="S1"
        )
        pred2 = VLMAnswerOutput(
            question_id="q2",
            predicted_choice="D",
            predicted_choice_idx=3,
            predicted_text="D) Unanswerable",
            reasoning="No evidence seen",
            abstain=True
        )

        sample1 = compute_sample_evaluation(q1, pred1)
        sample2 = compute_sample_evaluation(q2, pred2)

        self.assertTrue(sample1.is_correct)
        self.assertTrue(sample2.is_correct)
        self.assertTrue(sample2.is_unanswerable_ground_truth)

        summary = aggregate_metrics([sample1, sample2])
        self.assertEqual(summary.total_samples, 2)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.abstention_accuracy, 1.0)
        self.assertEqual(summary.hallucination_rate, 0.0)

    def test_invalid_prediction_is_counted_wrong_without_index_error(self):
        q = QuestionRecord(
            question_id="q_invalid",
            question="Where are the keys?",
            choices=["A) Desk", "B) Table", "C) Shelf", "D) Unanswerable"],
            answer="A",
            answer_idx=0,
        )
        prediction = VLMAnswerOutput(
            question_id="q_invalid",
            predicted_choice="INVALID",
            predicted_choice_idx=-1,
            predicted_text="",
            reasoning="Could not parse valid choice",
        )

        sample = compute_sample_evaluation(q, prediction)
        self.assertFalse(sample.is_correct)
        self.assertEqual(sample.score, 0.0)
        self.assertEqual(sample.predicted_label, "INVALID")


if __name__ == "__main__":
    unittest.main()
