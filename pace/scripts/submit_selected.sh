#!/bin/bash
# Learned frame picker, end to end: cut each question's chosen frames from the raw videos, then Gemma on them.
#
#   cd ~/scratch/hindsight && bash "$HINDSIGHT_REPO/pace/scripts/submit_selected.sh"
#
# Run on the LOGIN node from the work dir of the full run. Needs the selections from section 13
# (results/full/selectors/selections_<SELECTOR>.jsonl). Submits:
#   1. cut   CPU array (CUT_TASKS tasks): downloads each needed session video into data_selected/ (separate from the
#            curation's data/, so it can run alongside the agglomerative job), writes selected/<SELECTOR>/frames/,
#            deletes the video
#   2. vlm   Gemma 4 on every question, same settings as the curated runs -> results/full/vlm/selected_<SELECTOR>
#            (+ a continuation job that resumes if it hits 8 h)
#
# Settings: SELECTOR [logreg_top], CUT_TASKS [3] (ICE caps jobs per user and counts array tasks; 3 + 2 here plus
# the 12 of the agglomerative run stays at 17), HINDSIGHT_REPO, CURATION_ENV [~/scratch/hindsight/env],
# HINDSIGHT_ENV (Gemma 4) [~/scratch/hindsight/env-gemma4], IMAGE_TOKENS [140], MODEL, DRY_RUN=1.
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
CURATION_ENV="${CURATION_ENV:-$HOME/scratch/hindsight/env}"
VLM_ENV="${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env-gemma4}"
SELECTOR="${SELECTOR:-logreg_top}"
CUT_TASKS="${CUT_TASKS:-3}"
IMAGE_TOKENS="${IMAGE_TOKENS:-140}"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
S="$REPO/pace/scripts"
SEL="results/full/selectors/selections_${SELECTOR}.jsonl"
OUT="selected/${SELECTOR}"
RES="results/full/vlm/selected_${SELECTOR}"

die() { echo "ERROR: $*" >&2; exit 1; }
submit() { if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }
[ -f "$SEL" ] || die "$SEL not found: run section 13 (selectors.sbatch) first, from this work dir"
mkdir -p logs "$OUT"
echo "== $(wc -l < "$SEL") questions in $SEL"

CUT=$(submit --array=0-$((CUT_TASKS - 1)) \
    --export=ALL,SELECTIONS="$SEL",OUT="$OUT",TASKS="$CUT_TASKS",HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$CURATION_ENV" \
    "$S/prepare_selected.sbatch")
echo "== [1] cut frames: $CUT_TASKS tasks -> job $CUT ($OUT)"

ARGS=(--model-name "$MODEL" --output-dir "$RES" --condition selected --selected-dir "$OUT" --image-tokens "$IMAGE_TOKENS")
A=$(submit --dependency=afterany:"$CUT" --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$VLM_ENV" \
    --job-name=hs-vlm-selected "$S/vlm_eval.sbatch" "${ARGS[@]}")
B=$(submit --dependency=afterany:"$A" --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$VLM_ENV" \
    --job-name=hs-vlm-selected-cont "$S/vlm_eval.sbatch" "${ARGS[@]}")
echo "== [2] Gemma on $SELECTOR frames -> jobs $A (+ continuation $B), results in $RES"
echo "Watch: squeue --me   |   tail -f logs/hs-cut-*.out   |   tail -f logs/hs-vlm-selected-*.out"
