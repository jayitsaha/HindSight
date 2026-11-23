#!/bin/bash
# Temporal agglomerative clustering, end to end: curate 100 frames per session, then Gemma on them.
#
#   cd ~/scratch/hindsight && bash "$HINDSIGHT_REPO/pace/scripts/submit_agglomerative.sh"
#
# Run on the LOGIN node from the work dir of the full run (it needs cache/ and out/). Submits:
#   1. curate   array job: re-downloads each session video, writes out/<video>/agglomerative_K100.mp4 + .json
#               (the CLIP embeddings come from cache/, so nothing is re-embedded), deletes the video
#   2. vlm      Gemma 4 on every question, same settings as the KMeans/uniform runs
#               -> results/full/vlm/curated_agglomerative_K100 (+ a continuation job that resumes if it hits 8 h)
# Compare afterwards:
#   python -m hindsight.evaluation.compare results/full/vlm/curated_kmeans_K100 \
#       results/full/vlm/curated_uniform_K100 results/full/vlm/curated_agglomerative_K100 results/full/vlm/blind
#
# Settings: HINDSIGHT_REPO, CURATION_ENV [~/scratch/hindsight/env], HINDSIGHT_ENV (Gemma 4) [~/scratch/hindsight/env-gemma4],
# CUR_TASKS (array size) [10], IMAGE_TOKENS [140], MODEL [google/gemma-4-E4B-it], DRY_RUN=1 (print, submit nothing).
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
CURATION_ENV="${CURATION_ENV:-$HOME/scratch/hindsight/env}"
VLM_ENV="${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env-gemma4}"
CUR_TASKS="${CUR_TASKS:-10}"
IMAGE_TOKENS="${IMAGE_TOKENS:-140}"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
S="$REPO/pace/scripts"
OUT_DIR=results/full/vlm/curated_agglomerative_K100

die() { echo "ERROR: $*" >&2; exit 1; }
submit() { if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }

[ -d cache ] && [ -d out ] || die "run from the work dir of the full run (needs cache/ and out/)"
[ "$(ls cache/*.npz 2>/dev/null | wc -l)" -gt 0 ] || die "cache/ has no embeddings; the agglomerative run would re-embed every frame"
module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CURATION_ENV"
mkdir -p logs
python -m hindsight.curation.make_sessions --all --skip-done out --ks 100 --methods agglomerative --out sessions_agglo.txt
N=$(grep -c . sessions_agglo.txt || true)
conda deactivate

DEP=""
if [ "$N" -gt 0 ]; then
    T=$(( N < CUR_TASKS ? N : CUR_TASKS ))
    CUR_ID=$(submit --array=0-$((T - 1)) --time=8:00:00 \
        --export=ALL,SESSIONS=sessions_agglo.txt,STRIDE="$T",KS=100,METHODS=agglomerative,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$CURATION_ENV" \
        --job-name=hs-curate-agglo "$S/curate_array.sbatch")
    DEP="--dependency=afterany:$CUR_ID"
    echo "== [1] curation: $N sessions in $T array tasks -> job $CUR_ID"
else
    echo "== [1] curation: every session already has agglomerative_K100"
fi

ARGS=(--model-name "$MODEL" --output-dir "$OUT_DIR" --condition curated --curated-dir out --method agglomerative
      --k 100 --image-tokens "$IMAGE_TOKENS")
A=$(submit $DEP --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$VLM_ENV" --job-name=hs-vlm-agglo "$S/vlm_eval.sbatch" "${ARGS[@]}")
B=$(submit --dependency=afterany:"$A" --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$VLM_ENV" \
    --job-name=hs-vlm-agglo-cont "$S/vlm_eval.sbatch" "${ARGS[@]}")
echo "== [2] Gemma on agglomerative frames -> jobs $A (+ continuation $B), results in $OUT_DIR"
echo "Watch: squeue --me   |   tail -f logs/hs-curate-agglo-*.out   |   tail -f logs/hs-vlm-agglo-*.out"
