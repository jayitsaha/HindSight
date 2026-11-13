"""Option log-probs from generation, and the leave-one-participant-out abstention classifier."""
import json
import math
import random

import numpy as np
import pytest

pytest.importorskip("sklearn")

from hindsight.data.schema import QuestionRecord  # noqa: E402
from hindsight.evaluation import abstention as ab  # noqa: E402
from hindsight.vlm.hf_vlm import option_logprobs_from_generation  # noqa: E402

ABSTAIN = "This question can not be answered."


class FakeTokenizer:
    """SentencePiece-like: pieces joined, '▁' is a space."""

    def __init__(self, pieces):
        self.vocab = {p: i for i, p in enumerate(pieces)}
        self.inv = {i: p for p, i in self.vocab.items()}
        self.unk_token_id = -1

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.inv[i] for i in ids).replace("▁", " ")

    def convert_ids_to_tokens(self, i):
        return self.inv[i]

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)


def test_option_logprobs_merged_quote_token():
    pieces = ['{', '\n', '▁▁"', 'selected', '_choice', '":', '▁', '"A', '"B', '"C', '"D', '",', 'x']
    tok = FakeTokenizer(pieces)
    gen = [tok.vocab[p] for p in ['{', '\n', '▁▁"', 'selected', '_choice', '":', '▁', '"C', '",']]
    lp = np.full(len(pieces), -20.0)
    lp[tok.vocab['"A']], lp[tok.vocab['"B']], lp[tok.vocab['"C']], lp[tok.vocab['"D']] = -3, -4, -0.1, -2.5
    seen = []
    out = option_logprobs_from_generation(tok, gen, lambda i: seen.append(i) or lp, 4)
    assert out == {"A": -3, "B": -4, "C": -0.1, "D": -2.5}
    assert seen == [7]                                          # the step that wrote "C"


def test_option_logprobs_missing():
    tok = FakeTokenizer(['{', 'no', 'answer', '"A'])
    assert option_logprobs_from_generation(tok, [0, 1, 2], lambda i: np.zeros(4), 4) is None
    # letter written, but a swapped piece ('"B') is not in the vocabulary -> None, never a wrong guess
    tok = FakeTokenizer(['"selected_choice":', '▁', '"A'])
    assert option_logprobs_from_generation(tok, [0, 1, 2], lambda i: np.zeros(3), 4) is None


def synthetic(n_subjects=10, per=40, seed=0):
    """Gemma-like data: the model always abstains, but when its best real option is confident it is right."""
    rnd = random.Random(seed)
    qs, raw = {}, []
    for s in range(1, n_subjects + 1):
        for j in range(per):
            qid = f"{s}-{j}"
            a = rnd.randrange(4)
            answerable = rnd.random() < 0.65
            real = [i for i in range(4) if i != a]
            gt = rnd.choice(real) if answerable else a
            choices = [f"option {i}" for i in range(4)]
            choices[a] = ABSTAIN
            qs[qid] = QuestionRecord(question_id=qid, question="where?", choices=choices, answer=chr(65 + gt),
                                     answer_idx=gt, subject_id=f"S{s}", is_answerable=answerable,
                                     choice_types=["correct" if i == gt else "incorrect" for i in range(4)],
                                     cutoff_timestamp=60.0 * j)
            conf = answerable and rnd.random() < 0.7               # confident and right
            best = gt if answerable else rnd.choice(real)
            p = np.full(4, 0.05)
            p[a] = 0.55
            p[best] = 0.40 if conf else 0.12
            p /= p.sum()
            raw.append({"question_id": qid, "predicted_choice_idx": a, "predicted_text": ABSTAIN, "abstain": True,
                        "option_logprobs": {chr(65 + i): math.log(p[i]) for i in range(4)}})
    return qs, raw


def test_classifier_beats_always_abstain(tmp_path):
    qs, raw = synthetic()
    run = tmp_path / "curated_kmeans_K100_lp"
    run.mkdir()
    (run / "raw_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in raw) + "\n")
    res = ab.analyse(str(run), qs, str(tmp_path / "out"), models=["threshold", "logreg"])
    g, lr = res["methods"]["gemma"], res["methods"]["logreg"]
    assert res["n"] == 400 and res["n_participants"] == 10
    assert g["answered"] == 0 and g["over_abstention_rate"] == 1.0
    assert lr["accuracy"] > g["accuracy"] + 0.1                 # learns to answer the confident ones
    assert lr["over_abstention_rate"] < 0.6 and lr["p_vs_gemma"] < 0.01
    assert res["methods"]["threshold"]["accuracy"] > g["accuracy"]
    assert 0 <= lr["aurc"] <= 1 and 0 <= res["mrr"] <= 1
    ab.write_report({"curated_kmeans_K100_lp": res}, str(tmp_path / "out"))
    for f in ["ABSTENTION.md", "abstention_results.json", "curated_kmeans_K100_lp_decisions.jsonl",
              "curated_kmeans_K100_lp_risk_coverage.png"]:
        assert (tmp_path / "out" / f).is_file(), f
    d = [json.loads(l) for l in open(tmp_path / "out" / "curated_kmeans_K100_lp_decisions.jsonl")]
    assert len(d) == 400 and {"logreg", "threshold", "y"} <= set(d[0])


def test_no_logprobs_is_skipped(tmp_path):
    qs, raw = synthetic(n_subjects=3, per=10)
    for r in raw:
        r["option_logprobs"] = None
    (tmp_path / "raw_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in raw))
    assert ab.analyse(str(tmp_path), qs, str(tmp_path / "out")) is None


def test_features_do_not_use_answers():
    # same log-probs and question, different ground truth -> identical features
    qs, raw = synthetic(n_subjects=1, per=1)
    q = next(iter(qs.values()))
    f1 = ab.build_rows(qs, raw)[0]["features"]
    q.answer_idx = (q.answer_idx + 1) % 4
    q.is_answerable = not q.is_answerable
    q.choice_types = list(reversed(q.choice_types))
    assert ab.build_rows(qs, raw)[0]["features"] == f1


def test_mcnemar():
    a = np.array([1] * 10 + [0] * 10)
    assert ab.mcnemar_p(a, a) == 1.0
    assert ab.mcnemar_p(np.zeros(12), np.ones(12)) < 0.001
