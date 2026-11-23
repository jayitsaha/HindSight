"""Learned retriever (supervised): score "is this frame answer evidence for this question?" and compare its
windows with zero-shot CLIP and random anchors.

Same setting as phase 1 (hindsight.curation.grounded): for each Object & Location question, the candidates are
the cached 1-fps CLIP frame embeddings of its sessions, before the question time. A frame is labelled 1 if it is
inside an evidence span (+- tol seconds). Labels come from the benchmark's evidence spans; nothing is hand-labelled.

Features per (question, frame), all computable without the label:
  relevance   CLIP cosine with the three text queries (question / object phrase / "a photo of <object>"),
              smoothed over +-5 s and +-15 s, z-score and percentile rank within the question
  shape       is the smoothed cosine a local peak (+-10 s); its height over the +-30 s median
  scene       visual change from the previous frame (smoothed); distance from the session's mean frame
  time        sessions back from the question's session (one-hot 0/1/2/3+), log minutes before the question,
              position in the person's history and inside the session
  metric      (optional) PCA of the element-wise image x text product, fitted on the training folds only

Models: logistic regression and gradient-boosted trees (sklearn HistGradientBoosting), class-balanced,
trained with leave-one-participant-out (10 folds; each participant's questions are scored by a model that
never saw that person). Training uses every positive frame plus hard (highest-cosine) and random negatives.

    python -m hindsight.curation.learned_retriever --cache cache --out out --results results/full/retriever \
        --text-cache results/full/grounded/text_emb.npz

Writes RETRIEVER.md, retriever.json, windows_<model>.jsonl (anchors for every question; phase 2 input),
and recall_vs_X.png to --results.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hindsight.curation import grounded as gr
from hindsight.data.supermemory import SKILL, load_qa, spans_of

MODELS = ("logreg", "hgb")
NAMES = {"zeroshot": "Zero-shot CLIP (phase 1)", "random": "Random anchors",
         "logreg": "Learned: logistic regression", "hgb": "Learned: gradient-boosted trees"}
BASE_FEATURES = (["cos_question", "cos_object", "cos_prompt", "cos_s5", "cos_s15", "cos_z", "cos_s5_z", "cos_pct",
                  "peak10", "peak_height", "motion", "distinct"]
                 + [f"back_{k}" for k in ("0", "1", "2", "3plus")]
                 + ["log_min_before", "history_pos", "session_pos"])


# ---------------------------------------------------------------------------------------- time metadata
def video_starts(qa: Sequence[dict]) -> Dict[str, float]:
    """video_id -> wall-clock start (unix seconds), from every place the QA file records it."""
    st: Dict[str, float] = {}

    def put(v, t):
        if v and isinstance(t, (int, float)) and t > 1e9:
            st.setdefault(v, float(t))

    for r in qa:
        md = r.get("metadata", {})
        put(md.get("primary_video_id"), md.get("primary_video_start_time"))
        qe = r.get("question_evidence", {}) or {}
        put(qe.get("video_id"), qe.get("start_time"))
        for s in qe.get("time_spans") or []:
            put(s.get("video_id"), s.get("video_start_time_unix"))
        for e in (r.get("answer_evidence", {}) or {}).get("evidence_list", []) or []:
            put(e.get("video_id"), e.get("start_time"))
    return st


def _rolling(x: np.ndarray, half: int, fn) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d, median_filter
    size = 2 * half + 1
    if fn == "max":
        return maximum_filter1d(x, size=size, mode="nearest")
    return median_filter(x, size=size, mode="nearest")


# ---------------------------------------------------------------------------------------- features
def question_features(q: dict, cands, qemb: Dict[str, np.ndarray], starts: Dict[str, float],
                      first_start: Dict[str, float], tol: float = 2.0):
    """Feature matrix, labels and (vid, t) for every candidate frame of one question.
    Returns X (n, len(BASE_FEATURES)), prod (n, D) image*text product, y (n,), vids, ts."""
    feats, prods, ys, vids, ts = [], [], [], [], []
    prim_n = gr.session_number(q["primary"])
    q_wall = starts.get(q["primary"], 0.0) + (q["q_time"] if np.isfinite(q["q_time"]) else 0.0)
    person_start = first_start.get(gr.person_of(q["primary"]), starts.get(q["primary"], 0.0))
    history = max(q_wall - person_start, 1.0)
    for v, t, emb in cands:
        c = {k: emb @ qemb[k] for k in ("question", "object", "prompt")}
        s5, s15 = gr.smooth(c["prompt"], 5), gr.smooth(c["prompt"], 15)
        peak = (s5 >= _rolling(s5, 10, "max") - 1e-9).astype(float)
        height = s5 - _rolling(s5, 30, "median")
        motion = np.r_[0.0, 1.0 - np.sum(emb[1:] * emb[:-1], axis=1)] if len(emb) > 1 else np.zeros(len(emb))
        motion = gr.smooth(motion, 2)
        centre = emb.mean(0); centre /= np.linalg.norm(centre) + 1e-12
        distinct = 1.0 - emb @ centre
        back = prim_n - gr.session_number(v)
        onehot = np.zeros((len(t), 4)); onehot[:, min(max(back, 0), 3)] = 1
        wall = starts.get(v, starts.get(q["primary"], 0.0)) + t
        min_before = np.maximum(q_wall - wall, 0.0) / 60.0
        dur = max(float(t.max()), 1.0)
        f = np.column_stack([c["question"], c["object"], c["prompt"], s5, s15, np.zeros(len(t)), np.zeros(len(t)),
                             np.zeros(len(t)), peak, height, motion, distinct, onehot,
                             np.log1p(min_before), 1.0 - (min_before * 60.0) / history, t / dur])
        feats.append(f); prods.append(emb * qemb["prompt"])
        lab = np.zeros(len(t))
        for sv, s, e in q["spans"]:
            if sv == v:
                lab[(t >= s - tol) & (t <= e + tol)] = 1
        ys.append(lab); vids += [v] * len(t); ts.append(t)
    if not feats:
        return None
    X = np.concatenate(feats); y = np.concatenate(ys)
    i = BASE_FEATURES.index
    for col, src in (("cos_z", "cos_prompt"), ("cos_s5_z", "cos_s5")):
        x = X[:, i(src)]
        X[:, i(col)] = (x - x.mean()) / (x.std() + 1e-9)
    X[:, i("cos_pct")] = np.argsort(np.argsort(X[:, i("cos_prompt")])) / max(len(X) - 1, 1)
    return X.astype(np.float32), np.concatenate(prods).astype(np.float32), y, np.array(vids), np.concatenate(ts)


def training_rows(d, n_hard: int, n_random: int, rng, tol_gap: float = 10.0):
    """Indices of all positives + the highest-cosine negatives (away from positives) + random negatives."""
    X, _, y, vids, ts = d
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos):
        near = np.zeros(len(y), bool)
        for p in pos:
            near |= (vids == vids[p]) & (np.abs(ts - ts[p]) <= tol_gap)
        far = neg[~near[neg]]
    else:
        far = neg
    cos = X[:, BASE_FEATURES.index("cos_s5")]
    hard = far[np.argsort(-cos[far])[:n_hard]]
    rest = np.setdiff1d(neg, hard)
    rnd = rng.choice(rest, size=min(n_random, len(rest)), replace=False) if len(rest) else rest
    return np.concatenate([pos, hard, rnd]).astype(int)


def make_model(name: str, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced",
                                                                  random_state=seed))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, class_weight="balanced", random_state=seed)
    raise ValueError(name)


def evidence_rank_pct(score: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Percentile (0 = top) of the best-ranked evidence frame among a question's candidates."""
    if not y.any():
        return None
    order = np.argsort(-score, kind="stable")
    rank = np.where(y[order] == 1)[0][0]
    return 100.0 * rank / max(len(score) - 1, 1)


def build_data(questions: List[dict], videos: Dict[str, dict], qemb_by_text: Dict[str, np.ndarray],
               starts: Dict[str, float], sessions: str = "evidence", tol: float = 2.0):
    """Features for every question: {qid: (X, prod, y, vids, ts)} and {qid: person}."""
    first_start: Dict[str, float] = {}
    for v, s in starts.items():
        p = gr.person_of(v)
        first_start[p] = min(first_start.get(p, s), s)
    data, people = {}, {}
    for q in questions:
        cands = gr.candidates(q, videos, gr.sessions_for(q, videos, sessions), "all")
        texts = gr.query_texts(q["question"])
        qemb = {k: qemb_by_text[texts[k]] for k in ("question", "object", "prompt")}
        d = question_features(q, cands, qemb, starts, first_start, tol)
        if d is not None:
            data[q["qid"]] = d
            people[q["qid"]] = gr.person_of(q["primary"])
    return data, people


def lopo_scores(data, people, ans: Sequence[str], models: Sequence[str] = MODELS, product_dims: int = 16,
                n_hard: int = 100, n_random: int = 200, rng=None, seed: int = 0, log=print):
    """Leave-one-participant-out frame scores {model: {qid: scores}} (every question of a person is scored by a
    model trained on the other people's answerable questions) and the logistic-regression coefficients per fold."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    t0 = time.time()
    scores: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in models}
    coefs = []
    for p in sorted(set(people.values())):
        test = [qid for qid in data if people[qid] == p]
        train = [qid for qid in ans if people[qid] != p]
        rows = {qid: training_rows(data[qid], n_hard, n_random, rng) for qid in train}
        Xtr = np.concatenate([data[qid][0][rows[qid]] for qid in train])
        ytr = np.concatenate([data[qid][2][rows[qid]] for qid in train])
        pca = None
        if product_dims:
            from sklearn.decomposition import PCA
            Ptr = np.concatenate([data[qid][1][rows[qid]] for qid in train])
            pca = PCA(n_components=product_dims, random_state=seed).fit(Ptr)
            Xtr = np.hstack([Xtr, pca.transform(Ptr)])
        for m in models:
            model = make_model(m, seed).fit(Xtr, ytr)
            if m == "logreg":
                coefs.append(model[-1].coef_[0][:len(BASE_FEATURES)])
            for qid in test:
                X = data[qid][0]
                if pca is not None:
                    X = np.hstack([X, pca.transform(data[qid][1])])
                scores[m][qid] = model.predict_proba(X)[:, 1]
        log(f"  fold {p}: trained on {len(train)} questions ({len(ytr)} frames, {int(ytr.sum())} evidence), "
            f"scored {len(test)} ({time.time() - t0:.0f}s)")
    return scores, coefs


# ---------------------------------------------------------------------------------------- experiment
def run(questions: List[dict], videos: Dict[str, dict], qemb_by_text: Dict[str, np.ndarray], starts: Dict[str, float],
        sessions: str = "evidence", models: Sequence[str] = MODELS, xs: Sequence[int] = (1, 2, 4, 8),
        ws: Sequence[float] = (15, 30), min_gap: float = 60, tol: float = 2.0, product_dims: int = 16,
        n_hard: int = 100, n_random: int = 200, n_random_anchor_seeds: int = 5, seed: int = 0, log=print):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    data, people = build_data(questions, videos, qemb_by_text, starts, sessions, tol)
    log(f"features for {len(data)} questions ({time.time() - t0:.0f}s)")
    qbyid = {q["qid"]: q for q in questions}
    ans = [q["qid"] for q in questions if q["answerable"] and q["qid"] in data]
    scores, coefs = lopo_scores(data, people, ans, models, product_dims, n_hard, n_random, rng, seed, log)

    def anchors_from(qid, score):
        _, _, _, vids, ts = data[qid]
        scored = [(v, ts[vids == v], score[vids == v]) for v in dict.fromkeys(vids)]
        return gr.select_anchors(scored, max(xs), min_gap)

    zs = {qid: data[qid][0][:, BASE_FEATURES.index("cos_s5")] for qid in data}
    all_anchors = {"zeroshot": {qid: anchors_from(qid, zs[qid]) for qid in data}}
    for m in models:
        all_anchors[m] = {qid: anchors_from(qid, scores[m][qid]) for qid in data}
    rand = {qid: [anchors_from(qid, rng.random(len(data[qid][2]))) for _ in range(n_random_anchor_seeds)] for qid in ans}

    rows_out = []
    for method in ["zeroshot", *models, "random"]:
        for w in ws:
            for x in xs:
                hit = []
                for qid in ans:
                    sets = rand[qid] if method == "random" else [all_anchors[method][qid]]
                    hit.append(np.mean([gr.span_hits(qbyid[qid], gr.windows_of(a[:x], qbyid[qid], w), tol).any()
                                        for a in sets]))
                rows_out.append(dict(method=method, w=w, x=x, question_recall=round(100 * float(np.mean(hit)), 1)))

    from sklearn.metrics import average_precision_score, roc_auc_score
    frame_metrics = {}
    for method in ["zeroshot", *models]:
        aps, ranks = [], []
        for qid in ans:
            y = data[qid][2]
            if y.any() and not y.all():
                s = zs[qid] if method == "zeroshot" else scores[method][qid]
                aps.append(average_precision_score(y, s)); ranks.append(evidence_rank_pct(s, y))
        ally = np.concatenate([data[qid][2] for qid in ans])
        alls = np.concatenate([zs[qid] if method == "zeroshot" else scores[method][qid] for qid in ans])
        frame_metrics[method] = dict(mean_ap=round(100 * float(np.mean(aps)), 2),
                                     auc=round(float(roc_auc_score(ally, alls)), 3),
                                     median_evidence_rank_pct=round(float(np.median(ranks)), 1),
                                     evidence_in_top5pct=round(100 * float(np.mean(np.array(ranks) <= 5)), 1))
    info = dict(questions=len(questions), scored=len(data), answerable=len(ans), sessions=sessions,
                folds=len(set(people.values())),
                product_dims=product_dims, n_hard=n_hard, n_random=n_random, min_gap=min_gap, tol=tol,
                evidence_frame_share=round(100 * float(np.mean(np.concatenate([data[q][2] for q in ans]))), 2),
                median_frames=int(np.median([len(data[q][2]) for q in data])))
    if coefs:
        c = np.mean(coefs, axis=0)
        info["logreg_top_features"] = [(BASE_FEATURES[j], round(float(c[j]), 3)) for j in np.argsort(-np.abs(c))[:10]]

    windows = {}
    w0 = ws[0]
    for m in ["zeroshot", *models]:
        windows[m] = [dict(qid=qid, answerable=qbyid[qid]["answerable"],
                           windows=[dict(video_id=v, anchor_t=round(t, 2), score=round(s, 4),
                                         start=round(max(0.0, t - w0), 2),
                                         end=round(min(t + w0, qbyid[qid]["q_time"]) if v == qbyid[qid]["primary"]
                                                   else t + w0, 2)) for v, t, s in all_anchors[m][qid]])
                      for qid in data]
    return rows_out, frame_metrics, info, windows


# ---------------------------------------------------------------------------------------- report
def write_report(rows, frame_metrics, info, windows, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "retriever.json"), "w") as f:
        json.dump({"info": info, "recall": rows, "frame_metrics": frame_metrics}, f, indent=1)
    for m, ws_ in windows.items():
        with open(os.path.join(out, f"windows_{m}.jsonl"), "w") as f:
            for r in ws_:
                f.write(json.dumps(r) + "\n")
    xs = sorted({r["x"] for r in rows}); methods = list(dict.fromkeys(r["method"] for r in rows))
    lines = ["# Learned retriever vs zero-shot CLIP", "",
             f"{info['answerable']} answerable questions ({info['scored']} scored), sessions = {info['sessions']}, "
             f"median {info['median_frames']} candidate frames per question, {info['evidence_frame_share']}% of "
             f"candidate frames are evidence. Leave-one-participant-out ({info['folds']} folds).", "",
             "**Recall@X**: % of answerable questions with an evidence span (+-2 s) inside one of the first X windows "
             f"(anchors at least {info['min_gap']:g} s apart).", ""]
    for w in sorted({r["w"] for r in rows}):
        lines += [f"## window = +-{w:g} s", "", "| method | " + " | ".join(f"X={x}" for x in xs) + " |",
                  "|---|" + "---|" * len(xs)]
        for m in methods:
            v = {r["x"]: r["question_recall"] for r in rows if r["method"] == m and r["w"] == w}
            lines.append(f"| {NAMES.get(m, m)} | " + " | ".join(f"{v[x]:.1f}" for x in xs) + " |")
        lines.append("")
    lines += ["## Frame ranking (is the evidence frame scored high?)", "",
              "| method | mean AP | AUC | median rank of best evidence frame (percentile, 0 = top) | evidence in top 5% |",
              "|---|---|---|---|---|"]
    for m, fm in frame_metrics.items():
        lines.append(f"| {NAMES.get(m, m)} | {fm['mean_ap']:.2f} | {fm['auc']:.3f} | "
                     f"{fm['median_evidence_rank_pct']:.1f} | {fm['evidence_in_top5pct']:.1f}% |")
    if info.get("logreg_top_features"):
        lines += ["", "Logistic regression, largest standardised coefficients (mean over folds): "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in info["logreg_top_features"]) + "."]
    with open(os.path.join(out, "RETRIEVER.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        w = sorted({r["w"] for r in rows})[0]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for m, style in zip(methods, ["-o", "-s", "-^", ":x", "-d"]):
            pts = sorted((r["x"], r["question_recall"]) for r in rows if r["method"] == m and r["w"] == w)
            ax.plot(*zip(*pts), style, label=NAMES.get(m, m))
        ax.set_xlabel("windows per question (X)"); ax.set_ylabel("questions with evidence in a window (%)")
        ax.set_xticks(xs); ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Learned retriever vs zero-shot CLIP (window +-{w:g} s)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out, "recall_vs_X.png"), dpi=150); plt.close(fig)
    except ImportError:
        pass


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="out")
    ap.add_argument("--qa", default=None)
    ap.add_argument("--results", default="results/full/retriever")
    ap.add_argument("--text-cache", default="results/full/grounded/text_emb.npz",
                    help="CLIP text embeddings from phase 1 (encoded here if missing)")
    ap.add_argument("--sessions", default="evidence", choices=["evidence", "prior"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--x", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--window", type=float, nargs="+", default=[15, 30])
    ap.add_argument("--min-gap", type=float, default=60)
    ap.add_argument("--product-dims", type=int, default=16, help="PCA dims of image x text product (0 = off)")
    ap.add_argument("--n-hard", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    qa = load_qa(a.qa)
    questions = gr.load_questions(qa)
    print(f"loading embeddings from {a.cache}/ ...", flush=True)
    videos = gr.load_cache(a.cache, a.out)
    if not videos:
        raise SystemExit(f"No *.npz embeddings under {a.cache}")
    print(f"{len(questions)} questions, {len(videos)} cached sessions", flush=True)
    texts = sorted({t for q in questions for t in gr.query_texts(q["question"]).values()})
    emb = gr.encode_texts_open_clip(texts, gr.DEFAULT_EMBEDDER["model"], gr.DEFAULT_EMBEDDER["pretrained"],
                                    a.device, cache_file=a.text_cache)
    rows, fm, info, windows = run(questions, videos, dict(zip(texts, emb)), video_starts(qa), sessions=a.sessions,
                                  models=a.models, xs=a.x, ws=a.window, min_gap=a.min_gap,
                                  product_dims=a.product_dims, n_hard=a.n_hard, n_random=a.n_random,
                                  log=lambda s: print(s, flush=True))
    write_report(rows, fm, info, windows, a.results)
    print(f"done; wrote {a.results}/RETRIEVER.md", flush=True)
    print(open(os.path.join(a.results, "RETRIEVER.md")).read())


if __name__ == "__main__":
    main()
