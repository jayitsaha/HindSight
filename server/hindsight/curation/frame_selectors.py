"""Which 100 frames per session should Gemma see? Evidence coverage of question-agnostic and question-aware selectors.

Same budget and metric as the curation analysis: for each Object & Location question, every selector picks
100 frames per evidence session (so 100 x sessions in total), and an evidence span counts as covered if a kept
frame is inside it (+- 2 s) and recorded before the question. CPU only, on the cached 1-fps CLIP embeddings.

Selectors
  uniform, kmeans        the frames the curated videos actually hold (out/<video>/<method>_K100.json)
  random                 100 random frames per session (mean over seeds)
  agglomerative          Ward clustering into 100 clusters, frame nearest each cluster mean (question-agnostic)
  agglomerative_temporal the same, but only neighbouring frames may merge: each cluster is a contiguous stretch
  clip_top               the question's highest zero-shot CLIP frames ("a photo of <object>", smoothed +- 5 s),
                         at least --gap seconds apart
  logreg_top, hgb_top    the same with the learned retriever's scores (leave-one-participant-out)
  kmeans+clip, kmeans+hgb  half the budget from KMeans (every other kept frame), half from the question-aware scores

    python -m hindsight.curation.frame_selectors --cache cache --out out --results results/full/selectors \
        --text-cache results/full/grounded/text_emb.npz

Writes SELECTORS.md, selectors.json and selections_<selector>.jsonl (the chosen frames for every question,
input for a Gemma run) to --results.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from hindsight.curation import grounded as gr
from hindsight.curation import learned_retriever as lr
from hindsight.data.supermemory import load_qa

K = 100
AGNOSTIC = ("uniform", "kmeans", "random", "agglomerative", "agglomerative_temporal")
AWARE = ("clip_top", "logreg_top", "hgb_top", "kmeans+clip", "kmeans+hgb")
NAMES = {"uniform": "Uniform (curated)", "kmeans": "KMeans (curated)", "random": "Random",
         "agglomerative": "Agglomerative (Ward)", "agglomerative_temporal": "Agglomerative, temporal (contiguous)",
         "clip_top": "Zero-shot CLIP top frames", "logreg_top": "Learned top frames (logistic regression)",
         "hgb_top": "Learned top frames (gradient-boosted trees)", "kmeans+clip": "50% KMeans + 50% CLIP top",
         "kmeans+hgb": "50% KMeans + 50% learned (trees)"}


# ---------------------------------------------------------------------------------------- per-session selectors
def manifest_times(out_dir: Optional[str], vid: str, method: str, k: int = K) -> Optional[np.ndarray]:
    if not out_dir:
        return None
    f = os.path.join(out_dir, vid, f"{method}_K{k}.json")
    if not os.path.isfile(f):
        return None
    with open(f) as fh:
        return np.sort(np.array([fr["t_sec"] for fr in json.load(fh)["frames"]], dtype=float))


def kmeans_times(emb: np.ndarray, t: np.ndarray, k: int = K, seed: int = 0) -> np.ndarray:
    """Same rule as the curation (KMeans, n_init=3, frame nearest each centre); used when no manifest exists."""
    from sklearn.cluster import KMeans
    k = min(k, len(t))
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(emb)
    return np.sort(_nearest_to_means(emb, t, km.labels_, k))


def _nearest_to_means(emb, t, labels, k):
    picks = []
    for c in range(k):
        m = np.where(labels == c)[0]
        if len(m):
            centre = emb[m].mean(0)
            picks.append(t[m[np.argmin(np.linalg.norm(emb[m] - centre, axis=1))]])
    return np.array(picks)


def agglomerative_times(emb: np.ndarray, t: np.ndarray, k: int = K, temporal: bool = False) -> np.ndarray:
    from scipy.sparse import diags
    from sklearn.cluster import AgglomerativeClustering
    k = min(k, len(t))
    conn = diags([1.0, 1.0], [-1, 1], shape=(len(t), len(t))) if temporal else None
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=conn).fit_predict(emb)
    return np.sort(_nearest_to_means(emb, t, labels, k))


def top_spaced(vids: np.ndarray, ts: np.ndarray, score: np.ndarray, n: int, gap: float,
               taken: Optional[List[tuple]] = None) -> List[tuple]:
    """Greedy highest-score frames, no two in one session closer than `gap` seconds (nor to `taken`)."""
    taken = list(taken or [])
    by_vid: Dict[str, List[float]] = {}
    for v, t in taken:
        by_vid.setdefault(v, []).append(t)
    out = []
    for j in np.argsort(-score, kind="stable"):
        v, t = vids[j], float(ts[j])
        if all(abs(t - u) >= gap for u in by_vid.get(v, [])):
            out.append((v, t)); by_vid.setdefault(v, []).append(t)
            if len(out) == n:
                break
    return out


# ---------------------------------------------------------------------------------------- coverage
def span_coverage(q: dict, frames: Sequence[tuple], tol: float = 2.0) -> np.ndarray:
    """Per evidence span: is a kept frame (vid, t) inside [s - tol, e + tol] and not after the question?"""
    by_vid: Dict[str, np.ndarray] = {}
    for v, t in frames:
        by_vid.setdefault(v, []).append(t)
    res = []
    for v, s, e in q["spans"]:
        t = np.asarray(by_vid.get(v, []), dtype=float)
        cut = q["q_time"] if v == q["primary"] else np.inf
        res.append(bool(((t >= s - tol) & (t <= e + tol) & (t <= cut)).any()))
    return np.array(res, dtype=bool)


# ---------------------------------------------------------------------------------------- experiment
def run(questions: List[dict], videos: Dict[str, dict], qemb_by_text: Dict[str, np.ndarray], starts: Dict[str, float],
        out_dir: Optional[str] = None, selectors: Sequence[str] = AGNOSTIC + AWARE, k: int = K, gap: float = 5.0,
        n_random: int = 3, tol: float = 2.0, product_dims: int = 16, seed: int = 0, log=print):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    # question-agnostic frames per session (computed once per session)
    sess_frames: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in selectors if s in AGNOSTIC or s.startswith("kmeans+")}
    needed = sorted({v for q in questions for v in gr.sessions_for(q, videos, "evidence") if v in videos})
    from_manifest = {"uniform": 0, "kmeans": 0}
    for v in needed:
        emb, t = videos[v]["emb"], videos[v]["t"]
        base_cache: Dict[str, np.ndarray] = {}
        for s in sess_frames:
            if s in ("uniform", "kmeans") or s.startswith("kmeans+"):
                base = "uniform" if s == "uniform" else "kmeans"
                if base not in base_cache:
                    got = manifest_times(out_dir, v, base, k)
                    if got is None:
                        got = (t[np.unique(np.linspace(0, len(t) - 1, min(k, len(t))).round().astype(int))]
                               if base == "uniform" else kmeans_times(emb, t, k, seed))
                    else:
                        from_manifest[base] += 1
                    base_cache[base] = got
                sess_frames[s][v] = base_cache[base]
            elif s == "agglomerative":
                sess_frames[s][v] = agglomerative_times(emb, t, k, temporal=False)
            elif s == "agglomerative_temporal":
                sess_frames[s][v] = agglomerative_times(emb, t, k, temporal=True)
    log(f"question-agnostic selections for {len(needed)} sessions; curated manifests used: {from_manifest} "
        f"({time.time() - t0:.0f}s)")

    # question-aware scores
    data = scores = None
    if any(s in AWARE for s in selectors):
        data, people = lr.build_data(questions, videos, qemb_by_text, starts, "evidence", tol)
        ans_ids = [q["qid"] for q in questions if q["answerable"] and q["qid"] in data]
        models = [m for m in lr.MODELS if any(s in (f"{m}_top", f"kmeans+{m}") for s in selectors)]
        scores = {}
        if models:
            scores, _ = lr.lopo_scores(data, people, ans_ids, models, product_dims, rng=rng, seed=seed, log=log)
        scores["clip"] = {qid: data[qid][0][:, lr.BASE_FEATURES.index("cos_s5")] for qid in data}
        log(f"question-aware scores ready ({time.time() - t0:.0f}s)")

    selections: Dict[str, Dict[str, List[tuple]]] = {s: {} for s in selectors}
    for q in questions:
        vids = [v for v in gr.sessions_for(q, videos, "evidence") if v in videos]
        if not vids:
            continue
        budget = k * len(vids)
        for s in selectors:
            if s in AGNOSTIC and s != "random":
                selections[s][q["qid"]] = [(v, float(t)) for v in vids for t in sess_frames[s][v]]
            elif s == "random":
                selections[s][q["qid"]] = [[(v, float(t)) for v in vids
                                            for t in rng.choice(videos[v]["t"], size=min(k, len(videos[v]["t"])),
                                                                replace=False)] for _ in range(n_random)]
            elif q["qid"] in (data or {}):
                _, _, _, dv, dt = data[q["qid"]]
                key = {"clip_top": "clip", "logreg_top": "logreg", "hgb_top": "hgb",
                       "kmeans+clip": "clip", "kmeans+hgb": "hgb"}[s]
                sc = scores[key][q["qid"]]
                if s.startswith("kmeans+"):
                    half = [(v, float(t)) for v in vids for t in sess_frames[s][v][::2]]
                    selections[s][q["qid"]] = half + top_spaced(dv, dt, sc, budget - len(half), gap, taken=half)
                else:
                    selections[s][q["qid"]] = top_spaced(dv, dt, sc, budget, gap)

    ans = [q for q in questions if q["answerable"]]
    rows = []
    for s in selectors:
        span_hits, q_hits, n_frames = [], [], []
        for q in ans:
            sel = selections[s].get(q["qid"])
            if sel is None:
                h = np.zeros(len(q["spans"]), bool); span_hits.append(h); q_hits.append(False)
                continue
            sets = sel if s == "random" else [sel]
            hs = [span_coverage(q, f, tol) for f in sets]
            span_hits.append(np.mean(hs, axis=0)); q_hits.append(np.mean([h.any() for h in hs]))
            n_frames.append(np.mean([sum(1 for v, t in f if v != q["primary"] or t <= q["q_time"]) for f in sets]))
        allspans = np.concatenate(span_hits) if span_hits else np.zeros(0)
        rows.append(dict(selector=s, span_coverage=round(100 * float(allspans.mean()), 1),
                         question_coverage=round(100 * float(np.mean(q_hits)), 1),
                         usable_frames_median=int(np.median(n_frames)) if n_frames else 0))
    info = dict(questions=len(questions), answerable=len(ans), spans=int(sum(len(q["spans"]) for q in ans)),
                sessions=len(needed), k=k, gap=gap, tol=tol, curated_manifests=from_manifest)
    log(f"coverage done ({time.time() - t0:.0f}s)")
    return rows, info, selections


def write_report(rows, info, selections, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "selectors.json"), "w") as f:
        json.dump({"info": info, "rows": rows}, f, indent=1)
    for s, sel in selections.items():
        if s == "random":
            continue
        with open(os.path.join(out, f"selections_{s}.jsonl"), "w") as f:
            for qid, frames in sel.items():
                f.write(json.dumps({"qid": qid, "frames": [[v, round(t, 2)] for v, t in frames]}) + "\n")
    lines = ["# Which 100 frames per session? Evidence coverage by selector", "",
             f"{info['answerable']} answerable questions, {info['spans']} evidence spans, {info['sessions']} sessions. "
             f"Every selector keeps {info['k']} frames per evidence session. A span is covered if a kept frame is inside "
             f"it (+-{info['tol']:g} s) and recorded before the question. Question-aware selectors keep frames at least "
             f"{info['gap']:g} s apart and only before the question; their scores are leave-one-participant-out.", "",
             "| selector | spans covered (%) | questions with a covered span (%) | usable frames per question (median) |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {NAMES.get(r['selector'], r['selector'])} | {r['span_coverage']:.1f} | "
                     f"{r['question_coverage']:.1f} | {r['usable_frames_median']} |")
    lines += ["", "Usable frames = kept frames recorded before the question (frames after it are never shown to Gemma)."]
    with open(os.path.join(out, "SELECTORS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="out", help="curation outputs (fps and the curated KMeans/uniform frames)")
    ap.add_argument("--qa", default=None)
    ap.add_argument("--results", default="results/full/selectors")
    ap.add_argument("--text-cache", default="results/full/grounded/text_emb.npz")
    ap.add_argument("--selectors", nargs="+", default=list(AGNOSTIC + AWARE), choices=list(AGNOSTIC + AWARE))
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--gap", type=float, default=5.0, help="min seconds between question-aware frames")
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
    rows, info, sel = run(questions, videos, dict(zip(texts, emb)), lr.video_starts(qa), out_dir=a.out,
                          selectors=a.selectors, k=a.k, gap=a.gap, log=lambda s: print(s, flush=True))
    write_report(rows, info, sel, a.results)
    print(f"done; wrote {a.results}/SELECTORS.md", flush=True)
    print(open(os.path.join(a.results, "SELECTORS.md")).read())


if __name__ == "__main__":
    main()
