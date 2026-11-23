import json
import os
import tempfile
import unittest

from hindsight.vlm.config import DataConfig, EvalConfig, VLMConfig
from hindsight.data.loader import SuperMemoryDataLoader
from hindsight.data.pilot import hf_video_path, select_pilot
from hindsight.data.schema import VLMAnswerOutput
from hindsight.evaluation.evaluator import Evaluator
from hindsight.evaluation.metrics import aggregate_metrics, compute_sample_evaluation
from hindsight.video.clips import clip_frame_times, extract_frames
from hindsight.vlm.pipeline import VLMPipeline


def real_item(qid, answerable=True, video="Person_1_session_8_03102026_glasses_1264"):
    choices = ["Wrong place", "Right place", "This question can not be answered.", "Somewhere in the kitchen"]
    types = ["incorrect", "correct", "incorrect", "vague"]
    idx = 1
    if not answerable:
        types = ["incorrect", "incorrect", "correct", "incorrect"]
        idx = 2
    ev = [{"room": "Kitchen", "time_span": {"start_time": 10.0, "end_time": 20.0}, "video_id": video}]
    return {
        "question_id": qid, "question": "Where is the mug?", "choices": choices,
        "correct_answer": choices[idx], "correct_option_index": idx, "choice_types": types,
        "subject": 1, "metadata": {"skill": "object_location_memory", "primary_video_id": video},
        "video_ids": [video], "start_time": 123,
        "question_evidence": {"video_id": video, "time_spans": [{"start_time": 1.0, "end_time": 2.0, "video_id": video}]},
        "is_answerable": answerable,
        "answer_evidence": {"evidence_list": ev if answerable else [], "is_answerable": answerable, "text": choices[idx]},
    }


def load(items):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(items, f)
    return SuperMemoryDataLoader(DataConfig(subject_filter=["S1"])).load_questions(f.name)


class RealSchemaTest(unittest.TestCase):
    def test_loader_maps_real_fields(self):
        q = load([real_item(1), real_item(2, answerable=False)])
        self.assertEqual(len(q), 2)
        self.assertEqual(q[0].subject_id, "S1")
        self.assertEqual(q[0].category, "Object and Location Memory")
        self.assertEqual(q[0].answer_idx, 1)
        self.assertEqual(q[0].evidence_clips, [{"video_id": "Person_1_session_8_03102026_glasses_1264", "start": 10.0, "end": 20.0}])
        # unanswerable question falls back to the span where it was asked
        self.assertEqual(q[1].evidence_clips[0]["start"], 1.0)
        self.assertFalse(q[1].is_answerable)

    def test_metrics_vague_and_answerability(self):
        qs = load([real_item(1), real_item(2, answerable=False), real_item(3)])

        def pred(q, idx):
            return VLMAnswerOutput(q.question_id, "ABCD"[idx] if idx >= 0 else "INVALID", idx,
                                   q.choices[idx] if idx >= 0 else "", "", abstain=idx == 2)
        samples = [
            compute_sample_evaluation(qs[0], pred(qs[0], 3)),   # vague on answerable -> 0.5
            compute_sample_evaluation(qs[1], pred(qs[1], 2)),   # correct abstain
            compute_sample_evaluation(qs[2], pred(qs[2], -1)),  # invalid
        ]
        self.assertEqual(samples[0].score, 0.5)
        self.assertTrue(samples[0].is_vague_choice)
        self.assertTrue(samples[1].is_correct and not samples[1].predicted_answerable)
        self.assertEqual(samples[2].predicted_idx, -1)
        self.assertFalse(samples[2].predicted_answerable)  # invalid counts as wrong on answerability
        s = aggregate_metrics(samples)
        self.assertAlmostEqual(s.invalid_rate, 1 / 3, places=3)
        self.assertAlmostEqual(s.vague_rate, 0.5)
        self.assertAlmostEqual(s.answerability_f1, 2 / 3, places=3)  # tp=1 (vague answer), fn=1 (invalid)

    def test_select_pilot_limits_videos(self):
        items = [real_item(i, video=f"Person_1_session_{i % 5}_x") for i in range(1, 51)]
        pilot = select_pilot(load(items), n=10, max_videos=2)
        vids = {c["video_id"] for q in pilot for c in q.evidence_clips}
        self.assertLessEqual(len(vids), 2)
        self.assertEqual(len(pilot), 10)

    def test_video_path(self):
        self.assertEqual(hf_video_path("Person_1_session_8_03102026_glasses_1264"),
                         "data/video/Person_1/Person_1_session_8_03102026_glasses_1264.mp4")

    def test_clip_times_clamped(self):
        t = clip_frame_times(2.0, 4.0, 4, pad=5.0, duration=8.0)
        self.assertEqual(t[0], 0.0)
        self.assertLessEqual(t[-1], 7.5)

    def test_extract_frames_and_resumable_eval(self):
        from hindsight.video.demo_video import create_demo_video
        with tempfile.TemporaryDirectory() as d:
            video = create_demo_video(os.path.join(d, "v.mp4"), duration_seconds=3.0)
            outs = [os.path.join(d, f"f{i}.jpg") for i in range(3)]
            extract_frames(video, [0.0, 1.0, 2.5], outs)
            self.assertTrue(all(os.path.getsize(o) > 0 for o in outs))

            qs = load([real_item(1), real_item(2, answerable=False)])
            out = os.path.join(d, "eval")
            pipe = VLMPipeline(config=VLMConfig(backend="mock"))
            Evaluator(EvalConfig(output_dir=out, verbose=False)).evaluate_pipeline(pipe, qs[:1], [outs])
            with open(os.path.join(out, "raw_predictions.jsonl")) as f:
                self.assertEqual(len(f.readlines()), 1)

            calls = []
            orig = pipe.answer_question
            pipe.answer_question = lambda q, v, **k: (calls.append(q.question_id), orig(q, v, **k))[1]
            summary = Evaluator(EvalConfig(output_dir=out, verbose=False)).evaluate_pipeline(pipe, qs, [outs, []])
            self.assertEqual(calls, ["2"])  # question 1 resumed from disk, question 2 text-only
            self.assertEqual(summary.total_samples, 2)


if __name__ == "__main__":
    unittest.main()
