#!/bin/bash
# Supervised abstention classifier, end to end: queue it and walk away.
#
#   cd ~/scratch/hindsight && bash "$HINDSIGHT_REPO/pace/scripts/submit_abstention.sh"
#
# 1. Gemma 4 again on curated KMeans K=100, curated uniform K=100 and blind, this time recording the option
#    log-probs (results/full/vlm/*_lp; same prompts and frames as the full run, ~2 h, ~2 h and ~35 min, in parallel)
# 2. hindsight.evaluation.abstention on all three -> results/full/abstention/ABSTENTION.md (+ risk-coverage plots)
# Blind is the control: if the classifier gains as much without frames, it is only learning option priors.
# 4 jobs in all. Settings: K [100], IMAGE_TOKENS [140], MODEL, CURATION_ENV, HINDSIGHT_ENV as in run_full_pipeline.sh;
# DRY_RUN=1 prints the sbatch commands. Re-running resumes unfinished Gemma runs.
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
CURATION_ENV="${CURATION_ENV:-$HOME/scratch/hindsight/env}"
VLM_ENV="${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env-gemma4}"
K="${K:-100}"
IMAGE_TOKENS="${IMAGE_TOKENS:-140}"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
S="$REPO/pace/scripts"
OUT=results/full/vlm
submit() { if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }

[ -d out ] || { echo "ERROR: run from the work dir that holds out/ (curated frames)" >&2; exit 1; }
grep -q "option_logprobs" "$REPO/server/hindsight/vlm/hf_vlm.py" || { echo "ERROR: $REPO is too old: git pull" >&2; exit 1; }
mkdir -p logs results/full
VLM=(--export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$VLM_ENV")
IDS=()
for M in kmeans uniform; do
    IDS+=("$(submit "${VLM[@]}" --job-name="hs-lp-$M-K$K" "$S/vlm_eval.sbatch" --model-name "$MODEL" \
        --condition curated --curated-dir out --method "$M" --k "$K" --image-tokens "$IMAGE_TOKENS" \
        --output-dir "$OUT/curated_${M}_K${K}_lp")")
done
IDS+=("$(submit "${VLM[@]}" --job-name=hs-lp-blind "$S/vlm_eval.sbatch" --model-name "$MODEL" \
    --condition blind --curated-dir out --method kmeans --k "$K" --output-dir "$OUT/blind_lp")")
FIN=$(submit --dependency=afterany:$(IFS=:; echo "${IDS[*]}") \
    --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$CURATION_ENV" "$S/abstention.sbatch" \
    "$OUT/curated_kmeans_K${K}_lp" "$OUT/curated_uniform_K${K}_lp" "$OUT/blind_lp" --out results/full/abstention)
echo "== Gemma with log-probs: ${IDS[*]}  -> classifier: $FIN"
echo "Check: squeue --me | first log line to look for:  grep 'first option log-probs' logs/hs-lp-kmeans-*"
echo "Result: cat results/full/abstention/ABSTENTION.md"
