"""Supervised answer-or-abstain classifier on top of Gemma's option log-probabilities.

Gemma over-abstains (it picks "can not be answered" for ~75% of answerable questions) although it is right ~70% of
the time when it does commit. This module learns *when to answer* instead of trusting the model's own choice:

  features   from the log-probabilities Gemma gave the option letters (raw_predictions.jsonl, ``option_logprobs``):
             P(abstain option), best/second real option, margin, entropy, log-odds best-vs-abstain, letter mass,
             position of the abstain option, Gemma's generated decision, question time. No answer-evidence or
             session-count features (those would leak answerability from the oracle session choice).
  label      1 if answering with Gemma's most likely real (non-abstain) option is correct, else 0
  policy     answer with that option if P(label=1) >= tau, else abstain
  model      LogisticRegression (standardised features), against a tuned threshold on the log-odds (1 parameter,
             no learning) as the baseline
  protocol   leave-one-participant-out (10 folds); tau is picked on the training participants only, from inner
             leave-one-participant-out predictions, to maximise accuracy. Every question gets exactly one test
             decision, from a model that never saw its participant.

Scoring reuses hindsight.evaluation.metrics, so the table matches the VLM runs, plus selective-prediction metrics
(coverage, AURC, coverage at 80% selective accuracy) and Gemma's QA-MRR from the log-probs.

    python -m hindsight.evaluation.abstention results/full/vlm/curated_kmeans_K100_lp \\
        results/full/vlm/curated_uniform_K100_lp --out results/full/abstention
"""
import argparse
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

from hindsight.data.schema import QuestionRecord, VLMAnswerOutput, is_unanswerable_text
from hindsight.evaluation.metrics import aggregate_metrics, compute_sample_evaluation

FEATURES = ["p_abstain", "p_top1", "p_top2", "margin", "p_top1_renorm", "entropy", "log_odds", "letter_mass",
            "abstain_pos_0", "abstain_pos_1", "abstain_pos_2", "abstain_pos_3", "gemma_abstained", "log_q_time"]
MODELS = ["threshold", "logreg"]
NAMES = {"gemma": "Gemma (own choice)", "threshold": "Tuned threshold (baseline)", "logreg": "Logistic regression"}


# ----------------------------------------------------------------------------------------------- data
def build_rows(questions: Dict[str, QuestionRecord], raw: List[dict]) -> List[dict]:
    """One row per question with log-probs: features, label, and what answering would pick."""
    rows = []
    for r in raw:
        q = questions.get(str(r["question_id"]))
        lp = r.get("option_logprobs")
        if q is None or not lp:
            continue
        letters = [chr(ord("A") + i) for i in range(len(q.choices))]
        if any(c not in lp for c in letters):
            continue
        ab = [i for i, c in enumerate(q.choices) if is_unanswerable_text(c)]
        if len(ab) != 1:
            continue
        a = ab[0]
        logp = np.array([lp[c] for c in letters], dtype=float)
        mass = float(np.logaddexp.reduce(logp))
        p = np.exp(logp - mass)                                   # renormalised over the option letters
        real = [i for i in range(len(p)) if i != a]
        order = sorted(real, key=lambda i: -p[i])
        best = order[0]
        top1, top2 = p[order[0]], (p[order[1]] if len(order) > 1 else 0.0)
        f = {
            "p_abstain": p[a], "p_top1": top1, "p_top2": top2, "margin": top1 - top2,
            "p_top1_renorm": top1 / max(1e-9, 1 - p[a]),
            "entropy": float(-(p * np.log(np.clip(p, 1e-12, 1))).sum()),
            "log_odds": math.log(max(top1, 1e-12)) - math.log(max(p[a], 1e-12)),
            "letter_mass": mass,
            **{f"abstain_pos_{k}": float(a == k) for k in range(4)},
            "gemma_abstained": float(bool(r.get("abstain")) or is_unanswerable_text(r.get("predicted_text", ""))),
            "log_q_time": math.log1p(q.cutoff_timestamp or 0.0),
        }
        ranks = sorted(range(len(p)), key=lambda i: -p[i])
        rows.append({
            "question_id": q.question_id, "subject": q.subject_id or "?", "question": q,
            "features": [float(f[k]) for k in FEATURES], "best_idx": best, "abstain_idx": a,
            "y": int(best == q.answer_idx), "unanswerable": q.is_answerable is False,
            "gemma_idx": int(r.get("predicted_choice_idx", -1)), "gemma_abstain": bool(f["gemma_abstained"]),
            "rr": 1.0 / (ranks.index(q.answer_idx) + 1),
        })
    return rows


# ----------------------------------------------------------------------------------------------- models
def make_model(name: str, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    raise ValueError(name)


def fit_score(name: str, X_tr, y_tr, X_te, seed: int) -> np.ndarray:
    """Score = how much answering is preferred (higher -> answer)."""
    if name == "threshold":
        return X_te[:, FEATURES.index("log_odds")]
    if len(set(y_tr)) < 2:                                       # degenerate fold: constant score
        return np.full(len(X_te), float(np.mean(y_tr)))
    m = make_model(name, seed).fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def policy_correct(score, tau, y, unans) -> np.ndarray:
    """1 where the decision is right: answering is right if y=1, abstaining is right if the question is unanswerable."""
    return np.where(score >= tau, y, unans).astype(float)


def best_threshold(score, y, unans) -> float:
    cands = np.concatenate([np.unique(score), [np.inf]])
    accs = [policy_correct(score, t, y, unans).mean() for t in cands]
    return float(cands[int(np.argmax(accs))])


def lopo_scores(name: str, X, y, unans, groups, seed: int):
    """Outer leave-one-participant-out test scores and the tau used for each test question."""
    score, tau = np.zeros(len(y)), np.zeros(len(y))
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        inner = np.zeros(tr.sum())                               # inner LOPO -> out-of-fold scores to tune tau
        Xtr, ytr, utr, gtr = X[tr], y[tr], unans[tr], groups[tr]
        for h in np.unique(gtr):
            ite = gtr == h
            inner[ite] = fit_score(name, Xtr[~ite], ytr[~ite], Xtr[ite], seed)
        t = best_threshold(inner, ytr, utr)
        score[te] = fit_score(name, Xtr, ytr, X[te], seed)
        tau[te] = t
    return score, tau


# ----------------------------------------------------------------------------------------------- metrics
def selective_metrics(score, y) -> Dict[str, float]:
    """Risk-coverage over all questions ranked by score; answering an unanswerable question counts as an error."""
    order = np.argsort(-score, kind="stable")
    hits = y[order].astype(float)
    k = np.arange(1, len(hits) + 1)
    acc = np.cumsum(hits) / k
    ok = np.where(acc >= 0.8)[0]
    return {"aurc": float(np.mean(1 - acc)), "coverage_at_80": float((ok.max() + 1) / len(hits)) if len(ok) else 0.0,
            "curve_coverage": (k / len(hits)).tolist(), "curve_risk": (1 - acc).tolist()}


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired correctness."""
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarise(rows, idx, abstain) -> dict:
    samples = []
    for r, i, ab in zip(rows, idx, abstain):
        q = r["question"]
        i = int(r["abstain_idx"] if ab else i)
        pred = VLMAnswerOutput(question_id=q.question_id, predicted_choice=q.get_choice_label(i),
                               predicted_choice_idx=i, predicted_text=q.choices[i], reasoning="", abstain=bool(ab))
        samples.append(compute_sample_evaluation(q, pred))
    s = aggregate_metrics(samples)
    answered = [x for x, ab in zip(samples, abstain) if not ab]
    return {"selective_risk": (1 - np.mean([x.is_correct for x in answered])) if answered else 0.0,"accuracy": s.accuracy, "partial_credit": s.score_with_partial_credit,
            "answerable_accuracy": s.answerable_accuracy, "abstention_accuracy": s.abstention_accuracy,
            "answerability_f1": s.answerability_f1, "answerability_macro_f1": s.answerability_macro_f1,
            "hallucination_rate": s.hallucination_rate, "over_abstention_rate": s.over_abstention_rate,
            "answered": float(np.mean([not a for a in abstain])),
            "_correct": np.array([float(x.is_correct) for x in samples])}


# ----------------------------------------------------------------------------------------------- run
def analyse(run_dir: str, questions: Dict[str, QuestionRecord], out_dir: str, seed: int = 0,
            models=MODELS) -> Optional[dict]:
    raw_path = os.path.join(run_dir, "raw_predictions.jsonl")
    raw = [json.loads(l) for l in open(raw_path) if l.strip()]
    rows = build_rows(questions, raw)
    name = os.path.basename(os.path.normpath(run_dir))
    print(f"[abstention] {name}: {len(rows)}/{len(raw)} answers have option log-probs")
    if len(rows) < 20 or len({r["subject"] for r in rows}) < 3:
        print(f"[abstention] {name}: not enough data (need log-probs; rerun Gemma with this version)")
        return None
    X = np.array([r["features"] for r in rows])
    y = np.array([r["y"] for r in rows])
    unans = np.array([float(r["unanswerable"]) for r in rows])
    groups = np.array([r["subject"] for r in rows])
    best = [r["best_idx"] for r in rows]

    res = {"n": len(rows), "n_participants": int(len(set(groups))), "mrr": float(np.mean([r["rr"] for r in rows])),
           "label_rate": float(y.mean()), "methods": {}}
    gemma_score = -np.array([float(r["gemma_abstain"]) for r in rows])     # answered (0) ranks above abstained (-1)
    g = summarise(rows, [r["gemma_idx"] for r in rows], [r["gemma_abstain"] for r in rows])
    g.update({k: v for k, v in selective_metrics(gemma_score, y).items() if not k.startswith("curve")})
    res["methods"]["gemma"] = g
    decisions = {r["question_id"]: {"subject": r["subject"], "y": int(r["y"]), "gemma_abstain": r["gemma_abstain"]}
                 for r in rows}
    curves = {}
    for m in models:
        score, tau = lopo_scores(m, X, y, unans, groups, seed)
        answer = score >= tau
        s = summarise(rows, best, ~answer)
        sel = selective_metrics(score, y)
        curves[m] = (sel.pop("curve_coverage"), sel.pop("curve_risk"))
        s.update(sel)
        s["p_vs_gemma"] = mcnemar_p(g["_correct"], s["_correct"])
        res["methods"][m] = s
        for r, sc, t, a in zip(rows, score, tau, answer):
            decisions[r["question_id"]][m] = {"score": float(sc), "tau": float(t), "answer": bool(a)}

    if "logreg" in models:                                       # standardised coefficients on all data
        lr = make_model("logreg", seed).fit(X, y)
        res["logreg_coefficients"] = dict(zip(FEATURES, map(float, lr[-1].coef_[0])))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_decisions.jsonl"), "w") as f:
        for qid, d in decisions.items():
            f.write(json.dumps({"question_id": qid, **d}) + "\n")
    plot_risk_coverage(curves, g, name, os.path.join(out_dir, f"{name}_risk_coverage.png"))
    for v in res["methods"].values():
        v.pop("_correct", None)
    return res


def plot_risk_coverage(curves, gemma, name, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m, (cov, risk) in curves.items():
        ax.plot(cov, risk, lw=1.8, label=NAMES[m])
    # Gemma's own decision is a single point: coverage = share answered, risk = share of those that are wrong
    ax.scatter([gemma["answered"]], [gemma["selective_risk"]], color="black", zorder=5, label=NAMES["gemma"])
    ax.set(xlabel="Coverage (share of questions answered)", ylabel="Risk (share of answered that are wrong)",
           xlim=(0, 1), ylim=(0, 1), title=f"Risk vs coverage: {name}")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(results: Dict[str, dict], out_dir: str):
    L = ["# Abstention classifier (leave-one-participant-out)", "",
         "Answer with Gemma's most likely real option when the classifier says so, otherwise abstain. Every number is "
         "on held-out participants; the threshold is tuned on the training participants only.", ""]
    rows = [("QA accuracy", "accuracy"), ("Score w/ partial credit", "partial_credit"),
            ("Answerable accuracy", "answerable_accuracy"), ("Abstention accuracy", "abstention_accuracy"),
            ("Answerability F1 (answerable)", "answerability_f1"), ("Answerability macro-F1", "answerability_macro_f1"),
            ("Hallucination rate", "hallucination_rate"), ("Over-abstention rate", "over_abstention_rate"),
            ("Answered (coverage)", "answered"), ("Coverage at 80% selective accuracy", "coverage_at_80"),
            ("AURC (lower is better)", "aurc")]
    for run, res in results.items():
        ms = list(res["methods"])
        L += [f"## {run}", "", f"{res['n']} questions, {res['n_participants']} participants. Gemma QA-MRR from the "
              f"log-probs: {100 * res['mrr']:.1f}%. Answering with the best real option would be right on "
              f"{100 * res['label_rate']:.1f}% of questions.", "",
              "| metric | " + " | ".join(NAMES[m] for m in ms) + " |", "|---|" + "---|" * len(ms)]
        for label, k in rows:
            fmt = (lambda v: f"{v:.3f}") if k == "aurc" else (lambda v: f"{100 * v:.1f}%")
            L.append(f"| {label} | " + " | ".join(fmt(res["methods"][m][k]) for m in ms) + " |")
        L.append("| McNemar p vs Gemma (accuracy) | - | " +
                 " | ".join(f"{res['methods'][m]['p_vs_gemma']:.3g}" for m in ms[1:]) + " |")
        if "logreg_coefficients" in res:
            coef = sorted(res["logreg_coefficients"].items(), key=lambda kv: -abs(kv[1]))
            L += ["", "Logistic regression, standardised coefficients (positive -> answer): " +
                  ", ".join(f"`{k}` {v:+.2f}" for k, v in coef[:8])]
        L += ["", f"Risk-coverage plot: `{run}_risk_coverage.png`; per-question decisions: `{run}_decisions.jsonl`.", ""]
    open(os.path.join(out_dir, "ABSTENTION.md"), "w").write("\n".join(L) + "\n")
    json.dump(results, open(os.path.join(out_dir, "abstention_results.json"), "w"), indent=1)
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="VLM run dirs whose raw_predictions.jsonl has option_logprobs")
    ap.add_argument("--data-path", default=None, help="all_qa.json (default: downloaded from Hugging Face)")
    ap.add_argument("--out", default="results/full/abstention")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from hindsight.data.loader import SuperMemoryDataLoader
    from hindsight.evaluation.run import default_qa_path
    from hindsight.vlm.config import DataConfig
    path = a.data_path or default_qa_path()
    qs = SuperMemoryDataLoader(DataConfig(dataset_name_or_path=path)).load_questions(source_path=path)
    questions = {q.question_id: q for q in qs}
    results = {}
    for run in a.runs:
        res = analyse(run, questions, a.out, a.seed, a.models)
        if res:
            results[os.path.basename(os.path.normpath(run))] = res
    if not results:
        raise SystemExit("no run had option log-probs")
    write_report(results, a.out)


if __name__ == "__main__":
    main()
