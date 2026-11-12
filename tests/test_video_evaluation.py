"""Smoke test for synthetic video decoding and evaluation output generation."""

import os
import tempfile
import unittest

from hindsight.config import EvalConfig
from hindsight.data.loader import DEMO_SAMPLES, SuperMemoryDataLoader
from hindsight.evaluation.evaluator import Evaluator
from hindsight.vlm.hf_vlm import HuggingFaceVLM
from hindsight.vlm.mock_vlm import MockVLM
from hindsight.vlm.pipeline import VLMPipeline
from hindsight.video.demo_video import create_demo_video


class VideoDecodingMockVLM(MockVLM):
    """Decode video inputs like the HF backend, then answer with the oracle mock."""

    def __init__(self):
        super().__init__(mode="oracle", latency_seconds=0)
        self.reader = HuggingFaceVLM(max_frames=4)
        self.decoded_frame_counts = []

    def predict(self, video_input, question, **kwargs):
        frames = self.reader._prepare_visual_inputs(video_input)
        self.decoded_frame_counts.append(len(frames))
        return super().predict(frames, question, **kwargs)


class TestVideoEvaluationSmoke(unittest.TestCase):

    def test_mp4_video_is_decoded_and_evaluated(self):
        with tempfile.TemporaryDirectory(prefix="hindsight-video-test-") as tmp:
            video_path = create_demo_video(
                os.path.join(tmp, "smoke.mp4"),
                duration_seconds=1,
                fps=4,
            )
            self.assertTrue(video_path.endswith(".mp4"), video_path)
            self.assertGreater(os.path.getsize(video_path), 0)

            loader = SuperMemoryDataLoader()
            questions = [
                loader._normalize_item(item, i)
                for i, item in enumerate(DEMO_SAMPLES)
            ]
            pipeline = VLMPipeline(backend="mock")
            backend = VideoDecodingMockVLM()
            pipeline.backend = backend

            output_dir = os.path.join(tmp, "evaluation")
            evaluator = Evaluator(EvalConfig(output_dir=output_dir, verbose=False))
            summary = evaluator.evaluate_pipeline(
                pipeline,
                questions,
                [video_path] * len(questions),
            )

            self.assertEqual(summary.total_samples, len(questions))
            self.assertEqual(summary.accuracy, 1.0)
            self.assertEqual(backend.decoded_frame_counts, [4] * len(questions))
            for filename in ("eval_report.md", "metrics.json", "predictions.jsonl"):
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)))


if __name__ == "__main__":
    unittest.main()
