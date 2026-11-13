"""Glue between frame curation and the VLM evaluation: curated frames per question, the question-time cutoff,
labelled frames, the frame cap, lazy inputs, and an end-to-end run with the mock backend."""
import json
import os

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

from hindsight.data.loader import SuperMemoryDataLoader  # noqa: E402
from hindsight.data.schema import VLMAnswerOutput  # noqa: E402
from hindsight.evaluation import run as run_mod  # noqa: E402
from hindsight.evaluation.curated import CuratedFrames, filter_questions  # noqa: E402
from hindsight.evaluation.evaluator import Evaluator  # noqa: E402
from hindsight.vlm.config import DataConfig, EvalConfig, VLMConfig  # noqa: E402
from hindsight.vlm.hf_vlm import HuggingFaceVLM  # noqa: E402
from hindsight.vlm.pipeline import VLMPipeline  # noqa: E402
from hindsight.vlm.prompt import build_qa_prompt  # noqa: E402

S6 = "Person_1_session_6_01312026_glasses_1275"
S8 = "Person_1_session_8_03102026_glasses_1264"
S9 = "Person_1_session_9_03122026_glasses_1300"   # never curated
TIMES = [10.0, 40.0, 70.0, 100.0]                  # source times of the 4 curated frames per session


def write_session(root, vid, method="kmeans", k=4):
    d = root / vid
    d.mkdir(parents=True)
    vw = cv2.VideoWriter(str(d / f"{method}_K{k}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1, (32, 32))
    for i in range(len(TIMES)):
        vw.write(np.full((32, 32, 3), 40 * (i + 1), np.uint8))
    vw.release()
    json.dump({"frames": [{"out_index": i, "t_sec": t} for i, t in enumerate(TIMES)],
               "output": {"path": f"/somewhere/else/{method}_K{k}.mp4"}}, open(d / f"{method}_K{k}.json", "w"))


def item(qid, evidence, primary=S8, q_time="0:50", answerable=True):
    choices = ["Desk", "Kitchen counter", "This question can not be answered.", "Somewhere"]
    idx = 1 if answerable else 2
    return {
        "question_id": qid, "question": "Where did I leave my keys?", "choices": choices,
        "correct_answer": choices[idx], "correct_option_index": idx,
        "choice_types": ["incorrect", "correct", "incorrect", "vague"] if answerable else
                        ["incorrect", "incorrect", "correct", "incorrect"],
        "subject": 1, "metadata": {"skill": "object_location_memory", "primary_video_id": primary},
        "video_ids": [S6, S8], "start_time": 1773180268, "is_answerable": answerable,
        "question_evidence": {"video_id": primary, "time_spans": [
            {"start_time": q_time, "end_time": 60, "video_id": primary}]},
        "answer_evidence": {"evidence_list": [{"video_id": v, "time_span": {"start_time": s, "end_time": e}}
                                              for v, s, e in evidence] if answerable else []},
    }


@pytest.fixture
def curated(tmp_path):
    root = tmp_path / "out"
    write_session(root, S6)
    write_session(root, S8)
    return root


@pytest.fixture
def qa_file(tmp_path):
    items = [item(1, [(S8, 30, 35), (S6, 5, 12)]),          # evidence in an earlier session + own session
             item(2, [(S9, 1, 2)]),                         # needs a session we never curated
             item(3, [], answerable=False)]                 # unanswerable -> its own session
    p = tmp_path / "qa.json"
    p.write_text(json.dumps(items))
    return str(p)


def load(path):
    return SuperMemoryDataLoader(DataConfig()).load_questions(path)


def test_loader_times_and_cutoff(qa_file):
    q = load(qa_file)[0]
    assert q.cutoff_timestamp == 50.0                      # from question_evidence "0:50", not the unix start_time
    assert {c["video_id"] for c in q.evidence_clips} == {S6, S8}
    assert q.subject_id == "S1" and q.video_id == S8


def test_curated_sessions_cutoff_and_labels(curated, qa_file):
    q1, q2, q3 = load(qa_file)
    cf = CuratedFrames(str(curated), method="kmeans", k=4)
    assert cf.sessions_for(q1) == [S6, S8]                 # recording order, not evidence order
    frames = cf.frames_for(q1)
    labels = [lab for lab, _ in frames]
    # all 4 frames of session 6; session 8 only up to the question at 0:50
    assert labels == ["[session 6, 0:10]", "[session 6, 0:40]", "[session 6, 1:10]", "[session 6, 1:40]",
                      "[session 8, 0:10]", "[session 8, 0:40]"]
    assert all(isinstance(im, Image.Image) for _, im in frames)
    assert [q.question_id for q in filter_questions([q1, q2, q3], cf)] == ["1", "3"]
    assert cf.sessions_for(q3) == [S8]
    assert CuratedFrames(str(curated), k=4, sessions="primary").sessions_for(q1) == [S8]


def test_missing_curated_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        CuratedFrames(str(tmp_path), method="kmeans", k=100)


def test_labelled_inputs_and_frame_cap():
    frames = [(f"[session 1, 0:{i:02d}]", Image.new("RGB", (8, 8))) for i in range(10)]
    uncapped = HuggingFaceVLM(max_frames=None)._prepare_labelled_inputs(frames)
    assert len(uncapped[0]) == 10 and uncapped[1][3] == "[session 1, 0:03]"
    imgs, labels = HuggingFaceVLM(max_frames=4)._prepare_labelled_inputs(frames)
    assert len(imgs) == 4 and labels == ["[session 1, 0:00]", "[session 1, 0:02]", "[session 1, 0:05]",
                                         "[session 1, 0:07]"]
    content = HuggingFaceVLM._image_content(imgs[:2], labels[:2])
    assert [c["type"] for c in content] == ["text", "image", "text", "image"]
    with pytest.raises(ValueError):
        HuggingFaceVLM(image_token_budget=100)


def test_prompt_mentions_labels():
    assert "[session N" in build_qa_prompt("q?", "A) x", has_frames=True, labelled=True)
    assert "[session N" not in build_qa_prompt("q?", "A) x", has_frames=True)


def test_lazy_inputs_skip_finished_questions(qa_file, tmp_path):
    qs = load(qa_file)[:2]
    out = tmp_path / "eval"
    out.mkdir()
    done = VLMAnswerOutput(question_id="1", predicted_choice="B", predicted_choice_idx=1,
                           predicted_text=qs[0].choices[1], reasoning="cached")
    (out / "raw_predictions.jsonl").write_text(json.dumps(done.to_dict()) + "\n")
    calls = []

    def lazy(i):
        return lambda: calls.append(i) or []

    Evaluator(EvalConfig(output_dir=str(out), verbose=False)).evaluate_pipeline(
        VLMPipeline(VLMConfig(backend="mock")), qs, [lazy(0), lazy(1)])
    assert calls == [1]                                    # question 1 was resumed, its frames never loaded


def test_run_curated_and_blind_end_to_end(curated, qa_file, tmp_path):
    common = ["--backend", "mock", "--data-path", qa_file, "--curated-dir", str(curated), "--k", "4"]
    run_mod.main(common + ["--condition", "curated", "--output-dir", str(tmp_path / "cur")])
    run_mod.main(common + ["--condition", "blind", "--output-dir", str(tmp_path / "blind")])
    for d in ("cur", "blind"):
        m = json.load(open(tmp_path / d / "metrics.json"))
        assert m["total_samples"] == 2                     # question 2 dropped: session 9 not curated
    cfg = json.load(open(tmp_path / "cur" / "run_config.json"))
    assert cfg["max_frames"] is None and cfg["n_questions"] == 2
    with pytest.raises(SystemExit):
        run_mod.main(["--condition", "curated", "--data-path", qa_file])
