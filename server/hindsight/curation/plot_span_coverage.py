"""Single-panel figure for the report: % of evidence spans covered vs frame budget K.

Reads coverage_summary.csv written by ``hindsight.curation.analyze``; needs only matplotlib (no GPU).

    python -m hindsight.curation.plot_span_coverage results/full/curation/coverage_summary.csv \
        --out results/full/curation/span_coverage_vs_K.png
"""
import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"Random": dict(color="#9aa0a6", ls=":", marker="s"),
         "Uniform": dict(color="#1a73e8", ls="--", marker="^"),
         "KMeans": dict(color="#e8710a", ls="-", marker="o")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary", help="coverage_summary.csv from hindsight.curation.analyze")
    ap.add_argument("--out", default="span_coverage_vs_K.png")
    ap.add_argument("--sessions", type=int, default=None, help="number of sessions, shown in the title")
    a = ap.parse_args(argv)

    pts, n_spans = defaultdict(list), None
    for r in csv.DictReader(open(a.summary)):
        pts[r["method"]].append((int(r["K"]), float(r["span_cov_%"])))
        n_spans = int(r["spans"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ks = sorted({k for v in pts.values() for k, _ in v})
    top = {k: max(((y, m) for m, v in pts.items() for kk, y in v if kk == k))[1] for k in ks}
    for m in ["Random", "Uniform", "KMeans"]:
        if m not in pts:
            continue
        k, y = zip(*sorted(pts[m]))
        ax.plot(k, y, lw=2.2, ms=7, label=m, **STYLE[m])
        for kk, yy in zip(k, y):
            dy = 9 if top[kk] == m else -16                        # best method's label above, the others below
            ax.annotate(f"{yy:.1f}", (kk, yy), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=8, color=STYLE[m]["color"])
    ax.set_xticks(ks)
    ax.set(xlabel="Frames kept (K)", ylabel="% of evidence spans covered", ylim=(25, 102))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    where = f"{a.sessions} sessions, " if a.sessions else ""
    ax.set_title(f"Evidence coverage vs frame budget\n(Object & Location Memory, {where}{n_spans} evidence spans)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
