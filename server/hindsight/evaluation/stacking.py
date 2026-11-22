import argparse
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from hindsight.data.schema import QuestionRecord
from hindsight.evaluation import abstention as ab

ABSTAIN_TEXT = "This question can not be answered."
N_AGREEMENT = 10
VIEW_NAMES = ["curated_kmeans_K100_lp", "curated_uniform_K100_lp", "blind_lp"]
METHOD_NAMES = {
    "stacker": "Cross-view stacker",
    "cross_view_ablated": "3-view stack, cross-view features removed",
    "single_view": "Single-view logistic regression (kmeans)",
    "threshold": "Tuned threshold (1 parameter)",
}


def _load_raw(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, "raw_predictions.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


def load_view(run_dir: str, questions: Dict[str, QuestionRecord]):
    raw = _load_raw(run_dir)
    rows = ab.build_rows(questions, raw)
    lp = {str(r["question_id"]): r.get("option_logprobs") for r in raw}
    return raw, rows, lp


def align_views(km, uni, blind):
    ids_km = [str(r["question_id"]) for r in km[0]]
    ids_uni = [str(r["question_id"]) for r in uni[0]]
    ids_blind = [str(r["question_id"]) for r in blind[0]]
    if not (ids_km == ids_uni == ids_blind):
        raise ValueError("views are not aligned by question_id (differing ids or order)")
    rmap_uni = {r["question_id"]: r for r in uni[1]}
    rmap_blind = {r["question_id"]: r for r in blind[1]}
    aligned = []
    for r in km[1]:
        qid = r["question_id"]
        if qid not in rmap_uni or qid not in rmap_blind:
            raise ValueError(f"question_id {qid} missing from a view after row building")
        aligned.append((r, rmap_uni[qid], rmap_blind[qid], km[2][qid], uni[2][qid], blind[2][qid]))
    return aligned


def _dist(lp, letters):
    v = np.array([lp[c] for c in letters], dtype=float)
    m = float(np.logaddexp.reduce(v))
    return v, np.exp(v - m)


def _kl(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float((p * np.log(p / q)).sum())


def _js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _agreement(p_km, p_uni, p_blind):
    top = [int(np.argmax(p_km)), int(np.argmax(p_uni)), int(np.argmax(p_blind))]
    same = [float(top[0] == top[1]), float(top[0] == top[2]), float(top[1] == top[2])]
    kls = [_kl(p_km, p_uni), _kl(p_km, p_blind), _kl(p_uni, p_blind)]
    jss = [_js(p_km, p_uni), _js(p_km, p_blind), _js(p_uni, p_blind)]
    counts = Counter(top)
    probs = np.array([c / 3.0 for c in counts.values()])
    vote_entropy = float(-(probs * np.log(probs)).sum())
    return same + kls + jss + [vote_entropy]


def build_stacked(aligned) -> Tuple[np.ndarray, np.ndarray]:
    base = len(ab.FEATURES)
    vectors = []
    for km_r, uni_r, blind_r, lp_km, lp_uni, lp_blind in aligned:
        letters = [chr(ord("A") + i) for i in range(len(km_r["question"].choices))]
        logp_km, p_km = _dist(lp_km, letters)
        logp_uni, p_uni = _dist(lp_uni, letters)
        logp_blind, p_blind = _dist(lp_blind, letters)
        contrast = np.concatenate([logp_km - logp_blind, logp_uni - logp_blind]).tolist()
        agreement = _agreement(p_km, p_uni, p_blind)
        vec = list(km_r["features"]) + list(uni_r["features"]) + list(blind_r["features"]) + contrast + agreement
        vectors.append([float(x) for x in vec])
    X = np.array(vectors, dtype=float)
    mask = np.arange(X.shape[1]) >= 3 * base
    return X, mask


def meta_make_model(kind: str, seed: int = 0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    if kind == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier
        return make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                       subsample=1.0, random_state=seed),
        )
    raise ValueError(kind)


def meta_fit_score(kind: str, X_tr, y_tr, X_te, seed: int) -> np.ndarray:
    if len(set(y_tr)) < 2:
        return np.full(len(X_te), float(np.mean(y_tr)))
    m = meta_make_model(kind, seed).fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def meta_lopo_scores(kind: str, X, y, unans, groups, seed: int):
    score, tau = np.zeros(len(y)), np.zeros(len(y))
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        Xtr, ytr, utr, gtr = X[tr], y[tr], unans[tr], groups[tr]
        inner = np.zeros(int(tr.sum()))
        for h in np.unique(gtr):
            ite = gtr == h
            inner[ite] = meta_fit_score(kind, Xtr[~ite], ytr[~ite], Xtr[ite], seed)
        t = ab.best_threshold(inner, ytr, utr)
        score[te] = meta_fit_score(kind, Xtr, ytr, X[te], seed)
        tau[te] = t
    return score, tau


def _method_metrics(rows, best, score, tau, y):
    answer = score >= tau
    s = ab.summarise(rows, best, ~answer)
    sel = ab.selective_metrics(score, y)
    sel.pop("curve_coverage")
    sel.pop("curve_risk")
    s.update(sel)
    return s, answer


def analyse_stacking(km_dir, uni_dir, blind_dir, questions, out_dir, meta="gbm", seed=0) -> dict:
    km = load_view(km_dir, questions)
    uni = load_view(uni_dir, questions)
    blind = load_view(blind_dir, questions)
    aligned = align_views(km, uni, blind)
    rows = [t[0] for t in aligned]
    if not rows:
        raise ValueError("no rows have option_logprobs - point at the _lp runs (results/full/vlm/*_lp)")
    y = np.array([r["y"] for r in rows])
    unans = np.array([float(r["unanswerable"]) for r in rows])
    groups = np.array([r["subject"] for r in rows])
    best = [r["best_idx"] for r in rows]
    X_km = np.array([r["features"] for r in rows], dtype=float)
    X_stack, mask = build_stacked(aligned)
    X_ablation = X_stack.copy()
    X_ablation[:, mask] = 0.0

    sv_score, sv_tau = ab.lopo_scores("logreg", X_km, y, unans, groups, seed)
    th_score, th_tau = ab.lopo_scores("threshold", X_km, y, unans, groups, seed)
    st_score, st_tau = meta_lopo_scores(meta, X_stack, y, unans, groups, seed)
    ba_score, ba_tau = meta_lopo_scores(meta, X_ablation, y, unans, groups, seed)

    sv, _ = _method_metrics(rows, best, sv_score, sv_tau, y)
    th, _ = _method_metrics(rows, best, th_score, th_tau, y)
    st, _ = _method_metrics(rows, best, st_score, st_tau, y)
    ba, _ = _method_metrics(rows, best, ba_score, ba_tau, y)
    ref = sv["_correct"]
    st["p_vs_single_view"] = ab.mcnemar_p(ref, st["_correct"])
    ba["p_vs_single_view"] = ab.mcnemar_p(ref, ba["_correct"])
    th["p_vs_single_view"] = ab.mcnemar_p(ref, th["_correct"])
    sv["p_vs_single_view"] = 1.0
    st["p_vs_cross_view_ablated"] = ab.mcnemar_p(ba["_correct"], st["_correct"])
    ba["p_vs_cross_view_ablated"] = 1.0
    sv["p_vs_cross_view_ablated"] = ab.mcnemar_p(ba["_correct"], sv["_correct"])
    th["p_vs_cross_view_ablated"] = ab.mcnemar_p(ba["_correct"], th["_correct"])

    scores = {"stacker": st_score, "single_view": sv_score, "cross_view_ablated": ba_score, "threshold": th_score}
    taus = {"stacker": st_tau, "single_view": sv_tau, "cross_view_ablated": ba_tau, "threshold": th_tau}

    res = {
        "n": len(rows),
        "n_participants": int(len(set(groups))),
        "meta": meta,
        "label_rate": float(y.mean()),
        "mrr": float(np.mean([r["rr"] for r in rows])),
        "feature_width": int(X_stack.shape[1]),
        "n_single_features": int(len(ab.FEATURES)),
        "n_contrast_features": int(2 * len([c for c in [chr(ord("A") + i) for i in range(len(rows[0]["question"].choices))]])),
        "n_agreement_features": N_AGREEMENT,
        "methods": {"stacker": st, "cross_view_ablated": ba, "single_view": sv, "threshold": th},
    }

    os.makedirs(out_dir, exist_ok=True)
    dec_path = os.path.join(out_dir, "stacking_decisions.jsonl")
    with open(dec_path, "w") as f:
        for i, r in enumerate(rows):
            rec = {"question_id": r["question_id"], "subject": r["subject"], "y": int(r["y"]),
                   "unanswerable": bool(r["unanswerable"])}
            for m in scores:
                rec[m] = {"score": float(scores[m][i]), "tau": float(taus[m][i]),
                          "answer": bool(scores[m][i] >= taus[m][i])}
            f.write(json.dumps(rec) + "\n")

    for v in res["methods"].values():
        v.pop("_correct", None)
    return res


def write_report(res: dict, out_dir: str):
    ms = ["stacker", "cross_view_ablated", "single_view", "threshold"]
    L = ["# Cross-view stacking (leave-one-participant-out)", "",
         "A meta-learner stacks three Gemma views (curated kmeans, curated uniform, blind) with cross-view "
         "contrast and agreement features, deciding answer vs abstain. The cross-view-ablated control is the same "
         "3-view stack with the contrast and agreement features removed, so it keeps every view's single-view "
         "features and isolates what the cross-view signal adds. Every number is on held-out participants; "
         "the threshold is tuned on the training participants only.", "",
         f"{res['n']} questions, {res['n_participants']} participants. Meta-learner: `{res['meta']}`. "
         f"Feature width {res['feature_width']} = 3x{res['n_single_features']} single-view + "
         f"{res['n_contrast_features']} contrast + {res['n_agreement_features']} agreement. "
         f"Gemma QA-MRR {100 * res['mrr']:.1f}%. Answering with the best real option would be right on "
         f"{100 * res['label_rate']:.1f}% of questions.", "",
         "| metric | " + " | ".join(METHOD_NAMES[m] for m in ms) + " |", "|---|" + "---|" * len(ms)]
    rows = [("QA accuracy", "accuracy"), ("Answerable accuracy", "answerable_accuracy"),
            ("Abstention accuracy", "abstention_accuracy"), ("Answerability macro-F1", "answerability_macro_f1"),
            ("Hallucination rate", "hallucination_rate"), ("Over-abstention rate", "over_abstention_rate"),
            ("Answered (coverage)", "answered"), ("Coverage at 80% selective accuracy", "coverage_at_80"),
            ("AURC (lower is better)", "aurc")]
    for label, k in rows:
        fmt = (lambda v: f"{v:.3f}") if k == "aurc" else (lambda v: f"{100 * v:.1f}%")
        L.append(f"| {label} | " + " | ".join(fmt(res["methods"][m][k]) for m in ms) + " |")
    L.append("| McNemar p vs single-view | " +
             " | ".join(f"{res['methods'][m]['p_vs_single_view']:.3g}" for m in ms) + " |")
    L.append("| McNemar p vs cross-view-ablated | " +
             " | ".join(f"{res['methods'][m]['p_vs_cross_view_ablated']:.3g}" for m in ms) + " |")
    L += ["",
          "Honesty note: no claim that the stacker beats the cross-view-ablated or single-view controls is made. "
          "The stacker-vs-ablated McNemar column is the direct test of whether the cross-view features add anything; "
          "the expected result is a small or near-null gain. Read the AURC, coverage, and McNemar columns straight.",
          "", f"Per-question decisions: `stacking_decisions.jsonl`.", ""]
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "STACKING.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(out_dir, "stacking_results.json"), "w"), indent=1)
    print("\n".join(L))


def _mock_view_logprobs(rnd, a, gt, answerable, conf_rate, n=4):
    real = [i for i in range(n) if i != a]
    conf = answerable and rnd.random() < conf_rate
    best = gt if conf else rnd.choice(real)
    p = np.full(n, 0.05)
    p[a] = 0.55
    p[best] = 0.40 if conf else 0.12
    p = p * (1.0 + 0.15 * np.array([rnd.random() for _ in range(n)]))
    p = p / p.sum()
    return {chr(ord("A") + i): math.log(float(p[i])) for i in range(n)}


def _q_to_json(q: QuestionRecord) -> dict:
    return {"question_id": q.question_id, "question": q.question, "choices": q.choices, "answer": q.answer,
            "answer_idx": q.answer_idx, "subject_id": q.subject_id, "is_answerable": q.is_answerable,
            "choice_types": q.choice_types, "cutoff_timestamp": q.cutoff_timestamp}


def load_questions_jsonl(path: str) -> Dict[str, QuestionRecord]:
    qs = {}
    for l in open(path):
        if not l.strip():
            continue
        d = json.loads(l)
        qs[d["question_id"]] = QuestionRecord(**d)
    return qs


def make_mock_lp(dir: str, n_participants: int = 5, n_per: int = 40, seed: int = 0):
    rnd = random.Random(seed)
    view_dirs = [os.path.join(dir, v) for v in VIEW_NAMES]
    for d in view_dirs:
        os.makedirs(d, exist_ok=True)
    questions = {}
    recs = {v: [] for v in VIEW_NAMES}
    conf_rates = {VIEW_NAMES[0]: 0.74, VIEW_NAMES[1]: 0.71, VIEW_NAMES[2]: 0.50}
    for s in range(1, n_participants + 1):
        for j in range(n_per):
            qid = f"{s}-{j}"
            a = rnd.randrange(4)
            answerable = rnd.random() < 0.65
            real = [i for i in range(4) if i != a]
            gt = rnd.choice(real) if answerable else a
            choices = [f"option {i}" for i in range(4)]
            choices[a] = ABSTAIN_TEXT
            questions[qid] = QuestionRecord(
                question_id=qid, question="where was it?", choices=choices, answer=chr(ord("A") + gt),
                answer_idx=gt, subject_id=f"S{s}", is_answerable=answerable,
                choice_types=["correct" if i == gt else "incorrect" for i in range(4)],
                cutoff_timestamp=60.0 * j)
            for v in VIEW_NAMES:
                lp = _mock_view_logprobs(rnd, a, gt, answerable, conf_rates[v])
                recs[v].append({"question_id": qid, "predicted_choice_idx": a, "predicted_text": ABSTAIN_TEXT,
                                "abstain": True, "option_logprobs": lp})
    for v, d in zip(VIEW_NAMES, view_dirs):
        with open(os.path.join(d, "raw_predictions.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in recs[v]) + "\n")
    with open(os.path.join(dir, "questions.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(_q_to_json(q)) for q in questions.values()) + "\n")
    return questions, view_dirs


def _resolve_questions(km_dir, questions_path, data_path):
    if questions_path:
        return load_questions_jsonl(questions_path)
    guess = os.path.join(os.path.dirname(os.path.normpath(km_dir)), "questions.jsonl")
    if os.path.isfile(guess):
        return load_questions_jsonl(guess)
    from hindsight.data.loader import SuperMemoryDataLoader
    from hindsight.evaluation.run import default_qa_path
    from hindsight.vlm.config import DataConfig
    path = data_path or default_qa_path()
    qs = SuperMemoryDataLoader(DataConfig(dataset_name_or_path=path)).load_questions(source_path=path)
    return {q.question_id: q for q in qs}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-view stacking over three Gemma log-prob views.")
    ap.add_argument("kmeans_lp_dir")
    ap.add_argument("uniform_lp_dir")
    ap.add_argument("blind_lp_dir")
    ap.add_argument("--out", default="results/full/stacking")
    ap.add_argument("--meta", default="gbm", choices=["gbm", "logreg"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--data-path", default=None)
    a = ap.parse_args(argv)
    questions = _resolve_questions(a.kmeans_lp_dir, a.questions, a.data_path)
    res = analyse_stacking(a.kmeans_lp_dir, a.uniform_lp_dir, a.blind_lp_dir, questions, a.out, a.meta, a.seed)
    write_report(res, a.out)


if __name__ == "__main__":
    main()
