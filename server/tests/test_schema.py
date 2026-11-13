"""
Unit tests for data schemas.
"""

import unittest
from hindsight.data.schema import QuestionRecord, VLMAnswerOutput


class TestDataSchemas(unittest.TestCase):

    def test_question_record_formatting(self):
        q = QuestionRecord(
            question_id="test_01",
            question="Where are the keys?",
            choices=["Kitchen counter", "Dining table", "Living room", "Unanswerable"],
            answer="A",
            answer_idx=0
        )
        self.assertEqual(q.get_choice_label(0), "A")
        self.assertEqual(q.get_choice_label(3), "D")

        formatted = q.formatted_options()
        self.assertIn("A) Kitchen counter", formatted)
        self.assertIn("D) Unanswerable", formatted)

    def test_vlm_answer_output(self):
        ans = VLMAnswerOutput(
            question_id="test_01",
            predicted_choice="A",
            predicted_choice_idx=0,
            predicted_text="A) Kitchen counter",
            reasoning="Observed on counter at 02:15",
            abstain=False
        )
        d = ans.to_dict()
        self.assertEqual(d["question_id"], "test_01")
        self.assertEqual(d["predicted_choice"], "A")
        self.assertFalse(d["abstain"])


if __name__ == "__main__":
    unittest.main()
