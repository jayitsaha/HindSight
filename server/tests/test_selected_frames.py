"""Per-question frames: selection bookkeeping, task split and the SelectedFrames loader used by run.py."""
import json

from PIL import Image

from hindsight.data.schema import QuestionRecord
from hindsight.evaluation import prepare_selected as ps
from hindsight.evaluation.curated import SelectedFrames


def write_selection(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "selections.jsonl", "w") as f:
        for qid, frames in rows.items():
            f.write(json.dumps({"qid": qid, "frames": frames}) + "\n")


def test_frames_by_video_dedupes_and_tasks_split(tmp_path):
    rows = {"1": [["Person_1_session_2_x", 10.0], ["Person_1_session_1_x", 5.0]],
            "2": [["Person_1_session_2_x", 10.0], ["Person_1_session_2_x", 20.5]]}
    write_selection(tmp_path, rows)
    sel = ps.load_selections(str(tmp_path / "selections.jsonl"))
    by = ps.frames_by_video(sel)
    assert by == {"Person_1_session_2_x": [10.0, 20.5], "Person_1_session_1_x": [5.0]}
    vids = [f"v{i}" for i in range(7)]
    parts = [ps.my_videos(vids, i, 3) for i in range(3)]
    assert sorted(sum(parts, [])) == sorted(vids) and all(parts)
    assert ps.frame_name(20.5) == "20500.jpg"


def test_selected_frames_loader(tmp_path):
    rows = {"7": [["Person_1_session_2_x", 30.0], ["Person_1_session_1_x", 90.0], ["Person_1_session_2_x", 600.0]]}
    write_selection(tmp_path, rows)
    for v, t in rows["7"]:
        d = tmp_path / "frames" / v
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(d / ps.frame_name(t))
    q = QuestionRecord(question_id="7", question="Where did I leave the car?", choices=["a", "b", "c", "d"],
                       answer="A", answer_idx=0, video_id="Person_1_session_2_x", cutoff_timestamp=300.0)
    sf = SelectedFrames(str(tmp_path))
    assert sf.covers(q)
    frames = sf.frames_for(q)
    assert [lab for lab, _ in frames] == ["[session 1, 1:30]", "[session 2, 0:30]"]   # ordered; 600 s is after the question
    (tmp_path / "frames" / "Person_1_session_1_x" / "90000.jpg").unlink()
    assert not sf.covers(q)


def test_cut_frames_on_a_real_video(tmp_path):
    import numpy as np
    import pytest
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    for i in range(90):
        vw.write(np.full((64, 64, 3), i, np.uint8))
    vw.release()
    n = ps.cut_frames(str(path), [0.5, 1.0, 2.9], str(tmp_path / "out"), size=448)
    assert n == 3
    img = cv2.imread(str(tmp_path / "out" / "1000.jpg"))
    assert img.shape == (448, 448, 3)
