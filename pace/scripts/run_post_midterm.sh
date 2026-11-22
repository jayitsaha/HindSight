#!/bin/bash
# =====================================================================================================
# Post-midterm models in one go: temporal coverage (4.1), frame scorer (5.1), stacking (5.3).
#
#   cd ~/scratch/hindsight && bash "$HINDSIGHT_REPO/pace/scripts/run_post_midterm.sh"
#
# Run on the LOGIN node from your work dir. All three jobs are CPU:
#
#   1. hs-temporal  downloads a balanced ~N-session subset (make_sessions), curates each raw
#                   session at 1 fps (kmeans+temporal+uniform from the same pool) deleting the
#                   mp4 after, then hindsight.curation.analyze -> results/post_midterm/curation
#                   (coverage table including Temporal)
#   2. hs-scorer    frame_scorer eval on the SAME subset (--sessions), raw 1 fps embeddings
#                   -> results/post_midterm/frame_scorer/eval_<model>.txt
#   3. hs-stack     hindsight.evaluation.stacking on the three _lp Gemma runs
#                   -> results/post_midterm/stacking/STACKING.md
#
# 1 and 2 share one subset list and run in parallel (each downloads+embeds the subset itself,
# one mp4 at a time, ~5 GB peak; KEEP_VIDEO=1 keeps the mp4s if you have ~70 GB and want to
# avoid the double download). 3 needs results/full/vlm/*_lp from submit_abstention.sh (GPU Gemma
# re-runs); if missing it is chained after any queued hs-lp-* jobs, or skipped with a warning.
# The 1.34 GB curated sample (hf:hindsight-cs7641/hindsight-curated-frames) is the VLM input and
# the kmeans/uniform baselines, NOT the input to temporal/scorer (those need raw 1 fps frames).
#
# Settings (environment variables, defaults in brackets):
#   HINDSIGHT_REPO  clone on the post-midterm branch     [~/scratch/hindsight/HindSight]
#   CURATION_ENV    conda env used for curation          [~/scratch/hindsight/env]
#   KS              K values for the temporal pass       ["32 64 100 150"]
#   SCORER_MODEL    frame scorer model                   [logreg]  (logreg | mlp | gbm)
#   K               K of the _lp runs for stacking       [100]
#   N / PER         subset size / max sessions per person [25 / 3]
#   SESSIONS        shared subset list                   [sessions_post_midterm.txt]
#   KEEP_VIDEO=1    keep each mp4 after embedding (needs ~70 GB; avoids the double download)
#   DRY_RUN=1       print the sbatch commands, submit nothing
#   FORCE=1         submit while hs-* post-midterm jobs are still queued
# Re-running is safe: finished sessions/K values are skipped; analyze and the evals just rerun.
# =====================================================================================================
set -euo pipefail

WORK="$(pwd)"
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
CURATION_ENV="${CURATION_ENV:-$HOME/scratch/hindsight/env}"
KS="${KS:-32 64 100 150}"
SCORER_MODEL="${SCORER_MODEL:-logreg}"
K="${K:-100}"
N="${N:-25}"; PER="${PER:-3}"
SESSIONS="${SESSIONS:-sessions_post_midterm.txt}"
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
mkdir -p out
# cache/ is optional now: temporal + scorer run on a balanced ~25-session subset that is
# downloaded fresh at 1 fps (one mp4 at a time, deleted after embedding, ~5 GB peak). A warm
# cache/ only speeds things up; a cold cache just means CLIP embeds the subset on the CPU.
if [ ! -d cache ] || [ -z "$(find cache -name '*.npz' 2>/dev/null | head -1)" ]; then
    echo "NOTE: no warm cache/ in $WORK - the ~$N-session subset will be downloaded from Hugging Face"
    echo "      and embedded with CLIP (one mp4 at a time, deleted after; peak ~5 GB, but slow on CPU)."
fi

module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CURATION_ENV"
python -c "import sklearn, open_clip, cv2, pandas, matplotlib" 2>/dev/null \
    || die "$CURATION_ENV is missing sklearn/open_clip/cv2/pandas/matplotlib"
PYTHONPATH="$REPO/server" python -c "import hindsight.curation.frame_scorer, hindsight.evaluation.stacking" \
    || die "$REPO/server does not import (branch post-midterm?)"
# One shared subset list so temporal and the frame scorer touch the SAME ~25 sessions.
if [ ! -s "$SESSIONS" ]; then
    PYTHONPATH="$REPO/server" python -m hindsight.curation.make_sessions --n "$N" --per-person "$PER" --out "$SESSIONS" \
        || die "could not build the session subset (needs Hugging Face access for all_qa.json)"
fi
echo "== subset: $(wc -l < "$SESSIONS") sessions in $SESSIONS (<= $PER per participant)"
conda deactivate
mkdir -p logs results/post_midterm

EXP=(--export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$CURATION_ENV")

# ---------------------------------------------------------------- 1 + 2: subset runs, in parallel
COMMON="HINDSIGHT_REPO=$REPO,HINDSIGHT_ENV=$CURATION_ENV,SESSIONS=$SESSIONS,N=$N,PER=$PER${KEEP_VIDEO:+,KEEP_VIDEO=$KEEP_VIDEO}"
T_ID=$(submit --export=ALL,"$COMMON",KS="$KS" "$S/temporal_coverage.sbatch")
echo "== [1] temporal coverage -> job $T_ID"
F_ID=$(submit --export=ALL,"$COMMON",SCORER_MODEL="$SCORER_MODEL" "$S/frame_scorer.sbatch")
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
