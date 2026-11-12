"""Turn the array-job outputs into the comparison tables and graphs for the report.

    python -m hindsight.curation.analyze --out out --results results --k-bar 100

Reads out/<video_id>/{kmeans,uniform}_K<k>.json, matches the kept frames' timestamps against the
answer-evidence spans of Object & Location Memory questions, and adds a random baseline.

Writes to results/:
  coverage_per_session.csv     every session x method x K
  coverage_summary.csv         averaged over sessions (spans and unique moments)
  coverage_vs_K.png            headline plot: % of evidence covered vs number of frames
  coverage_by_session_K<k>.png per-session bars at one K (default 100)
  timelines/<video>_K<k>.png   when KMeans vs uniform frames fall, with evidence spans shaded
  grids/<video>_kmeans_K<k>.png keyframe grid from the KMeans output video
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hindsight.data.supermemory import evidence_rows, load_qa  # noqa: E402

METHOD_ORDER = ["Random", "Uniform", "KMeans"]
COLORS = {"Random": "#9aa0a6", "Uniform": "#1a73e8", "KMeans": "#e8710a"}
STYLES = {"Random": ":", "Uniform": "--", "KMeans": "-"}   # distinct styles so tied lines stay visible
MARKERS = {"Random": "s", "Uniform": "^", "KMeans": "o"}


def hits(times, spans, tol):
    """For each (s, e, q_cut) span: is any kept time inside [s - tol, e + tol] and before q_cut?"""
    t = np.asarray(times, dtype=float)
    return np.array([bool(((t >= s - tol) & (t <= e + tol) & (t <= q)).any()) for s, e, q in spans])


def load_runs(out_dir):
    runs = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*", "*_K*.json"))):
        m = re.match(r"(kmeans|uniform)_K(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        man = json.load(open(f))
        runs.append(dict(video_id=os.path.basename(os.path.dirname(f)),
                         method={"kmeans": "KMeans", "uniform": "Uniform"}[m.group(1)], K=int(m.group(2)),
                         times=[fr["t_sec"] for fr in man["frames"]],
                         duration=man["source"]["duration_s"], manifest=f,
                         video_path=os.path.join(os.path.dirname(f), os.path.basename(man["output"]["path"]))))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--results", default="results")
    ap.add_argument("--qa", default=None, help="path to all_qa.json (default: download)")
    ap.add_argument("--tol", type=float, default=2.0, help="seconds of slack around each evidence span")
    ap.add_argument("--k-bar", type=int, default=100, help="K used for the per-session bars, timelines and grids")
    ap.add_argument("--random-seeds", type=int, default=20)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.results, "timelines"), exist_ok=True)
    os.makedirs(os.path.join(a.results, "grids"), exist_ok=True)

    runs = load_runs(a.out)
    if not runs:
        sys.exit(f"No *_K*.json manifests found under {a.out}/<video_id>/")
    ev = pd.DataFrame(evidence_rows(load_qa(a.qa)))

    rows = []
    for vid in sorted({r["video_id"] for r in runs}):
        e = ev[ev.video_id == vid]
        if e.empty:
            print(f"skip {vid}: no object-question evidence")
            continue
        spans = list(zip(e.s, e.e, e.q_cut))
        moments = sorted(set(spans))                 # rephrased questions often share the same span
        mine = [r for r in runs if r["video_id"] == vid]
        duration = mine[0]["duration"]
        for r in mine:
            rows.append(dict(video_id=vid, method=r["method"], K=r["K"], frames=len(r["times"]),
                             spans=len(spans), span_hits=int(hits(r["times"], spans, a.tol).sum()),
                             moments=len(moments), moment_hits=int(hits(r["times"], moments, a.tol).sum())))
        for K in sorted({r["K"] for r in mine}):     # random baseline at each K
            sh, mh = [], []
            for seed in range(a.random_seeds):
                t = np.sort(np.random.default_rng(seed).uniform(0, duration, K))
                sh.append(hits(t, spans, a.tol).sum()); mh.append(hits(t, moments, a.tol).sum())
            rows.append(dict(video_id=vid, method="Random", K=K, frames=K, spans=len(spans), span_hits=float(np.mean(sh)),
                             moments=len(moments), moment_hits=float(np.mean(mh))))

    df = pd.DataFrame(rows)
    df["span_cov"] = df.span_hits / df.spans
    df["moment_cov"] = df.moment_hits / df.moments
    df.to_csv(os.path.join(a.results, "coverage_per_session.csv"), index=False)

    # micro-average over all spans / moments of all sessions
    summ = (df.groupby(["method", "K"])[["span_hits", "spans", "moment_hits", "moments"]].sum().reset_index())
    summ["span_cov_%"] = 100 * summ.span_hits / summ.spans
    summ["moment_cov_%"] = 100 * summ.moment_hits / summ.moments
    summ.to_csv(os.path.join(a.results, "coverage_summary.csv"), index=False)
    n_sessions = df.video_id.nunique()
    print(f"\n{n_sessions} sessions, {int(df.drop_duplicates('video_id').moments.sum())} unique evidence moments, "
          f"{int(df.drop_duplicates('video_id').spans.sum())} evidence spans\n")
    print(summ.pivot(index="K", columns="method", values="moment_cov_%")[
        [m for m in METHOD_ORDER if m in summ.method.unique()]].round(1).to_string())

    # ---- plot 1: coverage vs K ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, (col, title) in enumerate([("moment_cov_%", "Unique evidence moments covered"),
                                      ("span_cov_%", "Evidence spans covered (all questions)")]):
        for m in METHOD_ORDER:
            g = summ[summ.method == m].sort_values("K")
            if len(g):
                ax[i].plot(g.K, g[col], linestyle=STYLES[m], marker=MARKERS[m], color=COLORS[m], label=m, lw=2, ms=7, alpha=.9)
        ax[i].set(title=title, xlabel="Frames kept (K)", xticks=sorted(summ.K.unique()))
        ax[i].grid(alpha=.3)
    ax[0].set_ylabel("% covered"); ax[0].legend()
    fig.suptitle(f"Object & Location Memory: evidence coverage vs frame budget ({n_sessions} sessions)")
    fig.tight_layout(); fig.savefig(os.path.join(a.results, "coverage_vs_K.png"), dpi=160); plt.close(fig)

    # ---- plot 2: per-session bars at one K ----
    kb = a.k_bar if a.k_bar in set(df.K) else int(sorted(df.K.unique())[-1])
    d = df[df.K == kb].pivot(index="video_id", columns="method", values="moment_cov")
    d = d[[m for m in METHOD_ORDER if m in d.columns]] * 100
    labels = [v.split("_glasses")[0].replace("Person_", "P").replace("_session_", " s") for v in d.index]
    x = np.arange(len(d)); w = 0.8 / len(d.columns)
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(d) + 3), 4.2))
    for j, m in enumerate(d.columns):
        ax.bar(x + (j - (len(d.columns) - 1) / 2) * w, d[m], w, label=m, color=COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set(ylabel="% unique evidence moments covered", title=f"Per-session coverage at K={kb}", ylim=(0, 105))
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.results, f"coverage_by_session_K{kb}.png"), dpi=160); plt.close(fig)

    # ---- plot 3: timelines + keyframe grids ----
    try:
        import cv2
    except ImportError:
        cv2 = None
    for vid in d.index:
        km = next((r for r in runs if r["video_id"] == vid and r["method"] == "KMeans" and r["K"] == kb), None)
        un = next((r for r in runs if r["video_id"] == vid and r["method"] == "Uniform" and r["K"] == kb), None)
        e = ev[ev.video_id == vid]
        fig, ax = plt.subplots(figsize=(12, 2.2))
        for s, en in set(zip(e.s, e.e)):
            ax.axvspan(s / 60, en / 60, color="#34a853", alpha=.3, lw=0)
        for y, r, m in [(1, km, "KMeans"), (0, un, "Uniform")]:
            if r:
                ax.vlines(np.array(r["times"]) / 60, y - .35, y + .35, color=COLORS[m], lw=1)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Uniform", "KMeans"]); ax.set_xlabel("minutes")
        ax.set_xlim(0, (km or un)["duration"] / 60)
        ax.set_title(f"{vid}: kept frames (K={kb}); green = answer-evidence spans")
        fig.tight_layout(); fig.savefig(os.path.join(a.results, "timelines", f"{vid}_K{kb}.png"), dpi=150); plt.close(fig)

        if cv2 is not None and km and os.path.exists(km["video_path"]):
            cap = cv2.VideoCapture(km["video_path"]); frames = []
            while True:                               # K frames at output size: small enough to hold
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
            if frames:
                idx = np.linspace(0, len(frames) - 1, min(24, len(frames))).astype(int)
                cols = 8; rws = int(np.ceil(len(idx) / cols))
                fig, axes = plt.subplots(rws, cols, figsize=(2 * cols, 2.1 * rws))
                for axx in np.ravel(axes):
                    axx.axis("off")
                for axx, i in zip(np.ravel(axes), idx):
                    axx.imshow(frames[i]); axx.set_title(f"{km['times'][i] / 60:.1f} min", fontsize=8)
                fig.suptitle(f"{vid}: {len(idx)} of the K={kb} KMeans keyframes (evenly spaced)")
                fig.tight_layout(); fig.savefig(os.path.join(a.results, "grids", f"{vid}_kmeans_K{kb}.png"), dpi=120)
                plt.close(fig)

    print(f"\nWrote tables and figures to {a.results}/")


if __name__ == "__main__":
    main()
