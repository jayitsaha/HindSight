"""Question-grounded retrieval, phase 1: how often do CLIP-picked windows contain the answer evidence?

For every Object & Location Memory question:
  1. score the cached 1-fps CLIP frame embeddings of its sessions against a CLIP text query
     (the full question, the object phrase pulled out of it, or "a photo of <object>"),
     using only frames recorded before the question time;
  2. pick the top X anchors, at least --min-gap seconds apart within a session;
  3. a window is anchor +- W seconds (cut at the question time). A question is "hit" at X if any
     of its evidence spans (+- tol) overlaps one of its first X windows.

Compared against random anchors with the same X, W and spacing. CPU only; reads the curation cache
(cache/<video_id>_<hash>.npz with `emb` and `indices`) and the manifests in out/ for each video's fps.

    python -m hindsight.curation.grounded --cache cache --out out --results results/grounded

Writes to --results: RECALL.md, recall.json, windows.jsonl (anchors for every question, default
setting; input for phase 2) and recall_vs_X.png.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hindsight.data.supermemory import SKILL, load_qa, spans_of

DEFAULT_EMBEDDER = {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}
QUERIES = ("question", "object", "prompt")
POOLS = ("all", "uniform100")

# ---------------------------------------------------------------------------------------- questions
_PRONOUNS = {"it", "them", "they", "those", "these", "this", "that", "him", "her", "one", "ones"}
_VERBS = (r"leave|left|put|place|placed|keep|kept|set|drop|dropped|store|stored|stash|stashed|hide|hid|"
          r"move|moved|toss|tossed|throw|threw|grab|grabbed|take|took|see|saw|find|found|get|got|last see")
_TAIL = r"(?:\s+(?:after|before|when|while|from|earlier|originally|last|first|just|today|yesterday|this|during|once|in order|so|back)\b.*)?"
_PAT_DID = re.compile(r"where\b[^?]*?\b(?:did|do|does|have|has|had|would|could)\b\s+(?:\w+(?:\s+[A-Z])?)\s+"
                      r"(?:\w+ly\s+|last\s+|first\s+|just\s+)?(?:" + _VERBS + r")\s+(.+?)" + _TAIL + r"\?", re.I)
_PAT_WAS = re.compile(r"where(?:\s+|'s\s+)(?:exactly\s+)?(?:(?:is|are|was|were)\s+)?(.+?)(?:\s+(?:\w+ly\s+|last\s+)?"
                      r"(?:left|put|placed|kept|stored|stashed|hidden|moved|set|located|seen|sitting|now))?"
                      + _TAIL + r"\?", re.I)
_PAT_WHERE_I = re.compile(r"where\s+(?:i|we|he|she|they|person \w+)\s+(?:\w+ly\s+|last\s+|first\s+|just\s+)?"
                          r"(?:" + _VERBS + r")\s+(.+?)" + _TAIL + r"\?", re.I)
_ANTECEDENT = re.compile(r"\b(?:the|my|our|your|his|her|their)\s+([A-Za-z][\w'\- ]*?)"
                         r"(?=\s+(?:again|from|to|for|that|which|is|are|was|were|in|on|at|now|later|so|and|but|back|out|away|today)\b|[.,!?;]|$)")
_DET = re.compile(r"^(?:the|my|our|your|his|her|their|that|those|this|these|a|an|some)\s+", re.I)


_TRAILING = {"down", "up", "away", "back", "to", "off", "in", "into", "moved", "kept", "stored", "placed", "left", "put"}


def _clean(text: str) -> str:
    text = re.sub(r"^(?:down|up|away|back|aside)\s+", "", text.strip(" ,.'\""), flags=re.I)
    words = _DET.sub("", text).split()
    while words and words[-1].lower() in _TRAILING:
        words.pop()
    if len(words) > 1 and words[0].lower() in _PRONOUNS:     # "put it down" -> pronoun
        words = words[:1]
    return " ".join(words)


def object_phrase(question: str) -> Optional[str]:
    """The object a 'where did I leave X?' question asks about, or None if it can't be pulled out
    (pronouns such as 'them' need the earlier sentence; those questions fall back to the full question)."""
    for pat in (_PAT_DID, _PAT_WHERE_I, _PAT_WAS):
        m = pat.search(question)
        if m:
            obj = _clean(m.group(1))
            if not obj or re.match(r"(?:did|do|does|have|has|had)\b", obj, re.I):
                continue
            if obj and obj.lower() not in _PRONOUNS and len(obj.split()) <= 10:
                return obj
            if obj.lower() in _PRONOUNS:                  # "... the black spatula again. Where did I leave it?"
                before = question[:m.start()]
                found = _ANTECEDENT.findall(before)
                if found and len(found[-1].split()) <= 8:
                    return found[-1].strip()
    return None


def query_texts(question: str) -> Dict[str, str]:
    obj = object_phrase(question)
    return {"question": question,
            "object": obj or question,
            "prompt": f"a photo of {obj}" if obj else question}


def session_number(vid: str) -> int:
    m = re.search(r"session_(\d+)", vid)
    return int(m.group(1)) if m else 0


def person_of(vid: str) -> str:
    m = re.match(r"(Person_\d+)_", vid)
    return m.group(1) if m else vid


def load_questions(qa: Sequence[dict], skill: str = SKILL) -> List[dict]:
    """One dict per question: id, text, primary video, question time, evidence spans (video, s, e)."""
    out = []
    for r in qa:
        if r["metadata"]["skill"] != skill:
            continue
        prim = r["metadata"]["primary_video_id"]
        qs = spans_of(r["question_evidence"])
        q_time = min(s for s, _ in qs) if qs else float("inf")
        spans = []
        if r["is_answerable"]:
            for e in r["answer_evidence"]["evidence_list"]:
                spans += [(e["video_id"], s, en) for s, en in spans_of(e)]
        out.append(dict(qid=str(r["question_id"]), question=r["question"], primary=prim, q_time=q_time,
                        answerable=bool(r["is_answerable"]), spans=spans,
                        evidence_videos=sorted({v for v, _, _ in spans}, key=session_number)))
    return out


# ---------------------------------------------------------------------------------------- cache
def load_cache(cache_dir: str, out_dir: Optional[str] = None, default_fps: float = 30.0) -> Dict[str, dict]:
    """video_id -> {"t": seconds (sorted), "emb": L2-normalised (n, D)}. If a video has several cache
    files, the one with the most frames wins. fps comes from out/<video_id>/*.json (source.fps)."""
    best: Dict[str, str] = {}
    for f in glob.glob(os.path.join(cache_dir, "*.npz")):
        vid = os.path.basename(f)[:-4].rsplit("_", 1)[0]
        if vid not in best or os.path.getsize(f) > os.path.getsize(best[vid]):
            best[vid] = f
    videos = {}
    for vid, f in best.items():
        d = np.load(f)
        fps = default_fps
        if out_dir:
            for j in sorted(glob.glob(os.path.join(out_dir, vid, "*.json"))):
                with open(j) as fh:
                    fps = float(json.load(fh).get("source", {}).get("fps", default_fps))
                break
        order = np.argsort(d["indices"])
        emb = d["emb"][order].astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        videos[vid] = {"t": d["indices"][order] / fps, "emb": emb}
    return videos


# ---------------------------------------------------------------------------------------- retrieval
def sessions_for(q: dict, videos: Dict[str, dict], mode: str) -> List[str]:
    """'evidence': the sessions holding the answer evidence (own session if unanswerable), the same
    choice as the curated Gemma runs. 'prior': every cached session of that person up to the question's."""
    if mode == "evidence":
        vids = q["evidence_videos"] or [q["primary"]]
    elif mode == "prior":
        p, n = person_of(q["primary"]), session_number(q["primary"])
        vids = [v for v in videos if person_of(v) == p and session_number(v) <= n]
        vids = vids or [q["primary"]]
    else:
        raise ValueError(mode)
    return sorted(set(vids), key=session_number)


def smooth(x: np.ndarray, half: int) -> np.ndarray:
    if half <= 0 or len(x) < 2:
        return x
    k = np.ones(2 * half + 1) / (2 * half + 1)
    pad = np.pad(x, half, mode="edge")
    return np.convolve(pad, k, mode="valid")


def candidates(q: dict, videos: Dict[str, dict], vids: Sequence[str], pool: str):
    """Per session (vid, t, emb) of the frames that may be searched: before the question time in
    its own session; 'uniform100' keeps 100 evenly spaced frames per session (before the cut)."""
    out = []
    for v in vids:
        if v not in videos:
            continue
        t, emb = videos[v]["t"], videos[v]["emb"]
        if pool == "uniform100" and len(t) > 100:
            idx = np.unique(np.linspace(0, len(t) - 1, 100).round().astype(int))
            t, emb = t[idx], emb[idx]
        if v == q["primary"]:
            keep = t <= q["q_time"]
            t, emb = t[keep], emb[keep]
        if len(t):
            out.append((v, t, emb))
    return out


def select_anchors(scored: Sequence[Tuple[str, np.ndarray, np.ndarray]], x: int, min_gap: float) -> List[Tuple[str, float, float]]:
    """Greedy top-x (vid, t, score), no two anchors in one session closer than min_gap seconds."""
    scored = [(v, ts, ss) for v, ts, ss in scored if len(ts)]
    if not scored:
        return []
    vid = np.concatenate([np.full(len(ts), i) for i, (_, ts, _) in enumerate(scored)])
    t_all = np.concatenate([ts for _, ts, _ in scored])
    s_all = np.concatenate([ss for _, _, ss in scored])
    picked: List[Tuple[str, float, float]] = []
    for j in np.argsort(-s_all, kind="stable"):
        v, t = scored[vid[j]][0], float(t_all[j])
        if all(not (pv == v and abs(pt - t) < min_gap) for pv, pt, _ in picked):
            picked.append((v, t, float(s_all[j])))
            if len(picked) == x:
                break
    return picked


def windows_of(anchors, q: dict, w: float):
    return [(v, t - w, min(t + w, q["q_time"]) if v == q["primary"] else t + w) for v, t, _ in anchors]


def span_hits(q: dict, wins, tol: float) -> np.ndarray:
    """For each evidence span: does any window overlap [s - tol, e + tol] (part before the question)?"""
    res = []
    for v, s, e in q["spans"]:
        hi_cut = q["q_time"] if v == q["primary"] else float("inf")
        lo, hi = s - tol, min(e + tol, hi_cut)
        res.append(any(wv == v and a <= hi and b >= lo for wv, a, b in wins) if lo <= hi else False)
    return np.array(res, dtype=bool)


def first_hit_rank(q: dict, anchors, w: float, tol: float) -> Optional[int]:
    for i in range(len(anchors)):
        if span_hits(q, windows_of(anchors[i:i + 1], q, w), tol).any():
            return i + 1
    return None


# ---------------------------------------------------------------------------------------- evaluation
def encode_texts_open_clip(texts: List[str], model: str, pretrained: str, device: str = "cpu",
                           batch_size: int = 256) -> np.ndarray:
    import open_clip
    import torch
    m, _, _ = open_clip.create_model_and_transforms(model, pretrained=pretrained)
    m = m.eval().to(device)
    tok = open_clip.get_tokenizer(model)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            out.append(m.encode_text(tok(texts[i:i + batch_size]).to(device)).float().cpu().numpy())
    e = np.concatenate(out)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def evaluate(questions: List[dict], videos: Dict[str, dict], encode: Callable[[List[str]], np.ndarray],
             xs: Sequence[int] = (1, 2, 4, 8), ws: Sequence[float] = (15, 30), queries: Sequence[str] = QUERIES,
             pools: Sequence[str] = POOLS, session_modes: Sequence[str] = ("evidence",), min_gap: float = 60,
             smooth_s: int = 5, tol: float = 2.0, n_random: int = 5, seed: int = 0,
             default: Optional[dict] = None) -> Tuple[List[dict], List[dict], dict]:
    """Returns (rows of recall results, windows for every question under `default`, info)."""
    default = default or {"sessions": session_modes[0], "pool": "all", "query": "prompt", "w": ws[0]}
    texts = {q["qid"]: query_texts(q["question"]) for q in questions}
    uniq = sorted({t for d in texts.values() for t in d.values()})
    qemb = dict(zip(uniq, encode(uniq)))
    xmax = max(xs)
    rows, windows = [], []
    rng = np.random.default_rng(seed)
    ans = [q for q in questions if q["answerable"]]
    info = {"questions": len(questions), "answerable": len(ans), "videos_cached": len(videos),
            "object_extracted": sum(object_phrase(q["question"]) is not None for q in questions)}
    missing = sum(any(v not in videos for v, _, _ in q["spans"]) for q in ans)
    info["answerable_with_uncached_evidence"] = missing

    for mode in session_modes:
        for pool in pools:
            cands = {q["qid"]: candidates(q, videos, sessions_for(q, videos, mode), pool) for q in questions}
            frames = [sum(len(t) for _, t, _ in cands[q["qid"]]) for q in questions]
            # random anchors (query-independent), shared across query types
            rand = {}
            for q in ans:
                c = cands[q["qid"]]
                rand[q["qid"]] = [select_anchors([(v, t, rng.random(len(t))) for v, t, _ in c], xmax, min_gap)
                                  for _ in range(n_random)]
            for query in list(queries) + ["random"]:
                anchors = {}
                for q in questions:
                    if query == "random":
                        continue
                    qe = qemb[texts[q["qid"]][query]]
                    sc = [(v, t, smooth(e @ qe, smooth_s if pool == "all" else 0)) for v, t, e in cands[q["qid"]]]
                    anchors[q["qid"]] = select_anchors(sc, xmax, min_gap)
                for w in ws:
                    for x in xs:
                        qh, sh, ns = [], [], 0
                        for q in ans:
                            sets = rand[q["qid"]] if query == "random" else [anchors[q["qid"]]]
                            hq, hs = [], []
                            for a in sets:
                                h = span_hits(q, windows_of(a[:x], q, w), tol)
                                hq.append(h.any()); hs.append(h.mean() if len(h) else 0.0)
                            qh.append(np.mean(hq)); sh.append(np.mean(hs) * len(q["spans"])); ns += len(q["spans"])
                        rows.append(dict(sessions=mode, pool=pool, query=query, w=w, x=x,
                                         question_recall=round(100 * float(np.mean(qh)), 1),
                                         span_recall=round(100 * float(np.sum(sh)) / max(ns, 1), 1),
                                         frames_searched_median=int(np.median(frames)) if frames else 0))
                if (mode, pool, query) == (default["sessions"], default["pool"], default["query"]):
                    for q in questions:
                        a = anchors[q["qid"]]
                        windows.append(dict(qid=q["qid"], answerable=q["answerable"], query=texts[q["qid"]][query],
                                            first_hit_rank=first_hit_rank(q, a, default["w"], tol) if q["answerable"] else None,
                                            windows=[dict(video_id=v, anchor_t=round(t, 2), score=round(s, 4),
                                                          start=round(max(0.0, t - default["w"]), 2),
                                                          end=round(min(t + default["w"], q["q_time"]) if v == q["primary"]
                                                                    else t + default["w"], 2))
                                                     for v, t, s in a]))
    info["default"] = default
    return rows, windows, info


# ---------------------------------------------------------------------------------------- report
def write_report(rows: List[dict], windows: List[dict], info: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "recall.json"), "w") as f:
        json.dump({"info": info, "rows": rows}, f, indent=1)
    with open(os.path.join(out, "windows.jsonl"), "w") as f:
        for r in windows:
            f.write(json.dumps(r) + "\n")

    xs = sorted({r["x"] for r in rows})
    lines = ["# Question-grounded retrieval: recall of the answer evidence", "",
             f"{info['answerable']} answerable questions ({info['questions']} in total), "
             f"{info['videos_cached']} sessions with cached 1-fps CLIP embeddings. "
             f"Object phrase pulled out of {info['object_extracted']}/{info['questions']} questions "
             f"(the rest use the full question). "
             f"{info['answerable_with_uncached_evidence']} answerable questions have evidence in a session "
             f"without cached embeddings (counted as misses).", "",
             "**Recall@X** = % of answerable questions with an evidence span (+-2 s) inside one of the first X "
             "windows (anchor +- W s). Random = random anchors with the same spacing, averaged over seeds.", ""]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["sessions"], r["pool"], r["w"])].append(r)
    for (mode, pool, w), rs in groups.items():
        lines += [f"## sessions = {mode}, frames searched = {pool}, window = +-{w:g} s "
                  f"(median {rs[0]['frames_searched_median']} frames per question)", "",
                  "| query | " + " | ".join(f"X={x}" for x in xs) + " |",
                  "|---|" + "---|" * len(xs)]
        for query in [q for q in QUERIES if any(r["query"] == q for r in rs)] + ["random"]:
            vals = {r["x"]: r for r in rs if r["query"] == query}
            lines.append(f"| {query} | " + " | ".join(f"{vals[x]['question_recall']:.1f}" for x in xs) + " |")
        lines.append("")
    ranks = [r["first_hit_rank"] for r in windows if r["answerable"]]
    if ranks:
        hit = [k for k in ranks if k]
        lines += [f"Default setting {info['default']}: evidence found by one of the {max(xs)} windows for "
                  f"{100 * len(hit) / len(ranks):.1f}% of answerable questions; median rank of the first hit "
                  f"{int(np.median(hit)) if hit else '-'}.", ""]
    with open(os.path.join(out, "RECALL.md"), "w") as f:
        f.write("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = info["default"]
        sel = [r for r in rows if r["sessions"] == d["sessions"] and r["pool"] == d["pool"] and r["w"] == d["w"]]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for query, style in zip(list(QUERIES) + ["random"], ["-o", "-s", "-^", ":x"]):
            pts = sorted((r["x"], r["question_recall"]) for r in sel if r["query"] == query)
            if pts:
                ax.plot(*zip(*pts), style, label=query)
        ax.set_xlabel("windows per question (X)"); ax.set_ylabel("questions with evidence in a window (%)")
        ax.set_xticks(xs); ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.legend(frameon=False)
        ax.set_title(f"Retrieval recall (window +-{d['w']:g} s, {d['pool']} frames)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out, "recall_vs_X.png"), dpi=150); plt.close(fig)
    except ImportError:
        pass


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="cache", help="curation embedding cache (cache/<video>_<hash>.npz)")
    ap.add_argument("--out", default="out", help="curation output folder (for each video's fps)")
    ap.add_argument("--qa", default=None, help="all_qa.json (default: download from the HF dataset)")
    ap.add_argument("--results", default="results/grounded")
    ap.add_argument("--sessions", nargs="+", default=["evidence", "prior"], choices=["evidence", "prior"])
    ap.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    ap.add_argument("--queries", nargs="+", default=list(QUERIES), choices=list(QUERIES))
    ap.add_argument("--x", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--window", type=float, nargs="+", default=[15, 30], help="half-width W in seconds")
    ap.add_argument("--min-gap", type=float, default=60)
    ap.add_argument("--smooth", type=int, default=5, help="moving-average half-width in frames (pool=all)")
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    questions = load_questions(load_qa(a.qa))
    videos = load_cache(a.cache, a.out)
    if not videos:
        raise SystemExit(f"No *.npz embeddings under {a.cache}")
    print(f"{len(questions)} questions, {len(videos)} cached sessions", flush=True)
    enc = lambda texts: encode_texts_open_clip(texts, DEFAULT_EMBEDDER["model"], DEFAULT_EMBEDDER["pretrained"], a.device)
    rows, windows, info = evaluate(questions, videos, enc, xs=a.x, ws=a.window, queries=a.queries, pools=a.pools,
                                   session_modes=a.sessions, min_gap=a.min_gap, smooth_s=a.smooth, tol=a.tol)
    write_report(rows, windows, info, a.results)
    print(open(os.path.join(a.results, "RECALL.md")).read())


if __name__ == "__main__":
    main()
