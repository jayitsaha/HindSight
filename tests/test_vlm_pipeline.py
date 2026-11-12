"""
Unit tests for VLM prompt parsing and pipeline inference.
"""

import unittest
from PIL import Image

from hindsight.config import VLMConfig
from hindsight.data.schema import QuestionRecord
from hindsight.vlm.prompt import parse_vlm_response
from hindsight.vlm.pipeline import VLMPipeline


class TestVLMPipeline(unittest.TestCase):

    def test_parse_vlm_json_response(self):
        raw_json = '{"selected_choice": "B", "reasoning": "Seen on sofa", "is_unanswerable": false}'
        choices = ["A) Desk", "B) Sofa", "C) Counter", "D) Unanswerable"]
        letter, idx, text, reasoning, abstain = parse_vlm_response(raw_json, choices)

        self.assertEqual(letter, "B")
        self.assertEqual(idx, 1)
        self.assertEqual(text, "B) Sofa")
        self.assertEqual(reasoning, "Seen on sofa")
        self.assertFalse(abstain)

    def test_parse_vlm_unanswerable_response(self):
        raw_json = '{"selected_choice": "D", "reasoning": "Not observed", "is_unanswerable": true}'
        choices = ["A) Desk", "B) Sofa", "C) Counter", "D) Unanswerable"]
        letter, idx, text, reasoning, abstain = parse_vlm_response(raw_json, choices)

        self.assertEqual(letter, "D")
        self.assertEqual(idx, 3)
        self.assertTrue(abstain)

    def test_unparseable_response_is_not_silently_scored_as_a(self):
        raw = '{"targetEntity": {"text": "Golden keys at a wooden console table"}}'
        choices = ["A) Shelf", "B) Entryway area", "C) Kitchen table", "D) Unanswerable"]

        letter, idx, text, reasoning, abstain = parse_vlm_response(raw, choices)

        self.assertEqual(letter, "INVALID")
        self.assertEqual(idx, -1)
        self.assertEqual(text, "")
        self.assertIn("Could not parse valid choice", reasoning)
        self.assertFalse(abstain)

    def test_mock_pipeline_prediction(self):
        config = VLMConfig(backend="mock")
        pipeline = VLMPipeline(config=config)

        q = QuestionRecord(
            question_id="demo_1",
            question="Where was the phone?",
            choices=["A) Desk", "B) Bed", "C) Kitchen", "D) Unanswerable"],
            answer="A",
            answer_idx=0
        )
        dummy_img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        output = pipeline.answer_question(q, dummy_img)

        self.assertIn(output.predicted_choice, ["A", "B", "C", "D"])
        self.assertGreater(len(output.reasoning), 0)
        self.assertIsNotNone(output.latency_seconds)


if __name__ == "__main__":
    unittest.main()
