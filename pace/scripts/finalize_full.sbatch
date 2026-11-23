#!/bin/bash
# Last step of run_full_pipeline.sh: coverage over all curated sessions + the VLM comparison table, in one file.
# Submitted by run_full_pipeline.sh (needs VLM_DIRS); can also be run by hand on the login node:
#   VLM_DIRS="results/full/vlm/curated_kmeans_K100 results/full/vlm/curated_uniform_K100 results/full/vlm/blind" \
#     bash $HINDSIGHT_REPO/pace/scripts/finalize_full.sbatch
# Runs on a GPU partition only because that is where ICE jobs go; it does not use the GPU (a few minutes).
#SBATCH -J hs-final
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N1 --ntasks-per-node=2 --mem=16G
#SBATCH -t 0:45:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
KS="${KS:-32 64 100 150}"
OUT=results/full
SUMMARY=$OUT/SUMMARY.md
mkdir -p $OUT

python -m hindsight.curation.analyze --out out --results $OUT/curation --k-bar 100 > $OUT/curation_analyze.log 2>&1
ANALYZE_OK=$?

python - "$SUMMARY" "$KS" ${VLM_DIRS:-} <<'EOF'
import json, os, subprocess, sys
summary, ks, dirs = sys.argv[1], [int(k) for k in sys.argv[2].split()], sys.argv[3:]
L = ["# HindSight full run", ""]

# --- curation
done, partial = [], []
for vid in sorted(os.listdir("out")):
    if not os.path.isdir(os.path.join("out", vid)):
        continue
    have = all(os.path.isfile(f"out/{vid}/{m}_K{k}.json") for k in ks for m in ("kmeans", "uniform"))
    (done if have else partial).append(vid)
L += ["## Frame curation", f"- sessions fully curated (K = {' '.join(map(str, ks))}): **{len(done)}**"]
if partial:
    L.append(f"- incomplete (check logs/hs-curate-rest-*): {', '.join(partial)}")
if os.path.isfile("sessions_rest.txt"):
    todo = [l.split("\t")[1] for l in open("sessions_rest.txt") if l.strip()]
    missing = [v for v in todo if v not in done]
    L.append(f"- this run's sessions: {len(todo) - len(missing)}/{len(todo)} finished"
             + (f"; not finished: {', '.join(missing)}" if missing else ""))
cov = "results/full/curation/coverage_summary.csv"
if os.path.isfile(cov):
    import pandas as pd
    s = pd.read_csv(cov)
    for col, title in [("moment_cov_%", "unique evidence moments covered (%)"), ("span_cov_%", "evidence spans covered (%)")]:
        t = s.pivot(index="K", columns="method", values=col)
        t = t[[m for m in ["Random", "Uniform", "KMeans"] if m in t.columns]].round(1)
        L += ["", f"**{title}**", "", "| K | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
        L += [f"| {k} | " + " | ".join(f"{v:.1f}" for v in row) + " |" for k, row in t.iterrows()]
    L += ["", "Plots and per-session tables: `results/full/curation/` (coverage_vs_K.png, timelines/, grids/)"]
else:
    L.append("- coverage analysis failed: see results/full/curation_analyze.log")

# --- VLM
L += ["", "## Gemma 4 answers", ""]
ok = [d for d in dirs if os.path.isfile(os.path.join(d, "metrics.json"))]
for d in dirs:
    if d not in ok:
        L.append(f"- **{d}: no metrics.json** (job failed or did not finish: see logs/hs-vlm-*)")
for d in ok:
    raw = os.path.join(d, "raw_predictions.jsonl")
    cfg = os.path.join(d, "run_config.json")
    n_raw = sum(1 for _ in open(raw)) if os.path.isfile(raw) else 0
    n_q = json.load(open(cfg)).get("n_questions") if os.path.isfile(cfg) else "?"
    if isinstance(n_q, int) and n_raw < n_q:
        L.append(f"- {d}: only {n_raw}/{n_q} questions answered (rerun run_full_pipeline.sh to resume)")
if ok:
    table = subprocess.run([sys.executable, "-m", "hindsight.evaluation.compare", *ok], capture_output=True, text=True)
    L += ["", table.stdout.strip() or table.stderr.strip()]
L += ["", "Per-question predictions: `<dir>/predictions.jsonl`; model outputs: `<dir>/raw_predictions.jsonl`."]
open(summary, "w").write("\n".join(L) + "\n")
print("\n".join(L))
EOF
[ $ANALYZE_OK -eq 0 ] || echo "coverage analysis exited with $ANALYZE_OK (see $OUT/curation_analyze.log)"
echo "== wrote $SUMMARY"
