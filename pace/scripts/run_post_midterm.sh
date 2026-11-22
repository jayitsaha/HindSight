#!/bin/bash
# =====================================================================================================
# Post-midterm models in one go: temporal coverage (4.1), frame scorer (5.1), stacking (5.3).
#
#   cd ~/scratch/hindsight && bash "$HINDSIGHT_REPO/pace/scripts/run_post_midterm.sh"
#
# Run on the LOGIN node from your work dir (the one holding out/ and cache/ from the earlier
# curation runs). All three jobs are CPU; the GPU work they depend on must already exist:
#
#   1. hs-temporal  re-curates sessions so temporal_K<k> exists (CLIP embeddings reused from
#                   cache/), then hindsight.curation.analyze -> results/post_midterm/curation
#                   (coverage table now includes Temporal)
#   2. hs-scorer    frame_scorer eval on the cached CLIP frame embeddings
#                   -> results/post_midterm/frame_scorer/eval_<model>.txt
#   3. hs-stack     hindsight.evaluation.stacking on the three _lp Gemma runs
#                   -> results/post_midterm/stacking/STACKING.md
#
# 1 and 2 only need the curation embedding cache and run in parallel. 3 needs the
# results/full/vlm/*_lp dirs from submit_abstention.sh (GPU Gemma re-runs): if they are missing
# it is chained after any queued hs-lp-* jobs, or skipped with a warning.
#
# Settings (environment variables, defaults in brackets):
#   HINDSIGHT_REPO  clone on the post-midterm branch     [~/scratch/hindsight/HindSight]
#   CURATION_ENV    conda env used for curation          [~/scratch/hindsight/env]
#   KS              K values for the temporal pass       ["32 64 100 150"]
#   SCORER_MODEL    frame scorer model                   [logreg]  (logreg | mlp | gbm)
#   K               K of the _lp runs for stacking       [100]
#   DRY_RUN=1       print the sbatch commands, submit nothing
#   FORCE=1         submit even without cache/, or while hs-* post-midterm jobs are queued
# Re-running is safe: finished sessions/K values are skipped; analyze and the evals just rerun.
# =====================================================================================================
set -euo pipefail

WORK="$(pwd)"
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
CURATION_ENV="${CURATION_ENV:-$HOME/scratch/hindsight/env}"
KS="${KS:-32 64 100 150}"
SCORER_MODEL="${SCORER_MODEL:-logreg}"
K="${K:-100}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
S="$REPO/pace/scripts"
LP_DIRS=(results/full/vlm/curated_kmeans_K${K}_lp results/full/vlm/curated_uniform_K${K}_lp results/full/vlm/blind_lp)

die() { echo "ERROR: $*" >&2; exit 1; }
submit() {   # prints the job id; in DRY_RUN prints the command and a fake id
    if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}

echo "== HindSight post-midterm | work dir $WORK | repo $REPO ($(git -C "$REPO" branch --show-current 2>/dev/null || echo '?'))"

# ---------------------------------------------------------------- pre-flight (fail now, not at 3 am)
if command -v squeue >/dev/null && [ -z "${FORCE:-}" ]; then
    OLD=$(squeue --me -h -o "%i %j" | grep -E " hs-(temporal|scorer|stack)" || true)
    [ -z "$OLD" ] || die "post-midterm jobs from an earlier run are still queued (scancel them, or FORCE=1):
$OLD"
fi
[ -f "$REPO/server/hindsight/video/curation_temporal.py" ] && [ -f "$REPO/server/hindsight/evaluation/stacking.py" ] \
    || die "$REPO has no temporal/stacking code: git -C $REPO checkout post-midterm && git pull"
grep -q '"temporal"' "$REPO/server/hindsight/curation/run_session.py" \
    || die "$REPO's run_session does not write temporal_K<k>: git -C $REPO checkout post-midterm && git pull"
[ -d "$CURATION_ENV" ] || die "curation env not found: $CURATION_ENV (set CURATION_ENV)"
[ -d out ] || die "run from the work dir that holds out/ (curated sessions)"
if [ ! -d cache ] || [ -z "$(find cache -name '*.npz' 2>/dev/null | head -1)" ]; then
    [ -n "${FORCE:-}" ] || die "no curation embedding cache in $WORK/cache - temporal and the frame scorer would
re-embed every frame with CLIP on the CPU. Run curation first (curate_array.sbatch /
run_full_pipeline.sh, docs/PACE.md sections 7 and 9), or FORCE=1 to accept the slow path."
    echo "WARNING: cache/ is empty; proceeding because FORCE=1 (CPU re-embedding will be slow)"
fi

module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CURATION_ENV"
python -c "import sklearn, open_clip, cv2, pandas, matplotlib" 2>/dev/null \
    || die "$CURATION_ENV is missing sklearn/open_clip/cv2/pandas/matplotlib"
PYTHONPATH="$REPO/server" python -c "import hindsight.curation.frame_scorer, hindsight.evaluation.stacking" \
    || die "$REPO/server does not import (branch post-midterm?)"
conda deactivate
mkdir -p logs results/post_midterm

EXP=(--export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$CURATION_ENV")

# ---------------------------------------------------------------- 1 + 2: cache-dependent, in parallel
T_ID=$(submit --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$CURATION_ENV",KS="$KS" \
    "$S/temporal_coverage.sbatch")
echo "== [1] temporal coverage -> job $T_ID"
F_ID=$(submit --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$CURATION_ENV",SCORER_MODEL="$SCORER_MODEL" \
    "$S/frame_scorer.sbatch")
echo "== [2] frame scorer ($SCORER_MODEL) -> job $F_ID"

# ---------------------------------------------------------------- 3: stacking, needs the _lp Gemma runs
MISSING=()
for d in "${LP_DIRS[@]}"; do [ -f "$d/raw_predictions.jsonl" ] || MISSING+=("$d"); done
S_ID=
if [ ${#MISSING[@]} -eq 0 ]; then
    S_ID=$(submit "${EXP[@]}" "$S/stacking.sbatch" "${LP_DIRS[@]}" --out results/post_midterm/stacking)
    echo "== [3] stacking -> job $S_ID"
else
    LP_JOBS=$(command -v squeue >/dev/null && squeue --me -h -o "%i %j" | awk '$2 ~ /^hs-lp-/ {print $1}' | paste -sd: - || true)
    if [ -n "$LP_JOBS" ]; then
        S_ID=$(submit --dependency=afterany:"$LP_JOBS" "${EXP[@]}" "$S/stacking.sbatch" \
            "${LP_DIRS[@]}" --out results/post_midterm/stacking)
        echo "== [3] stacking -> job $S_ID (after queued hs-lp jobs $LP_JOBS)"
    else
        echo "WARNING: [3] stacking SKIPPED - missing _lp runs (no raw_predictions.jsonl):" >&2
        printf '         %s\n' "${MISSING[@]}" >&2
        echo "         Run  bash \"$S/submit_abstention.sh\"  (GPU Gemma re-runs, docs/PACE.md section 10)," >&2
        echo "         then  sbatch \"$S/stacking.sbatch\"  or re-run this script." >&2
    fi
fi

{
    echo "submitted $(date)"; echo "temporal $T_ID"; echo "scorer $F_ID"; echo "stacking ${S_ID:-skipped}"
} > results/post_midterm/jobs.txt
cat <<EOF

Queued (ids in results/post_midterm/jobs.txt). Check with:  squeue --me
When done:
  results/post_midterm/curation/coverage_summary.csv   (Temporal vs KMeans/Uniform/Random)
  results/post_midterm/frame_scorer/eval_${SCORER_MODEL}.txt
  results/post_midterm/stacking/STACKING.md
Logs: logs/hs-temporal-*  logs/hs-scorer-*  logs/hs-stack-*
If something died, just run this script again: finished work is skipped.
EOF
