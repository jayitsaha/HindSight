#!/bin/bash
# Gemma 4 (or any HF VLM) on SuperMemory-VQA Object & Location Memory, one condition per job.
# All arguments are passed to `python -m hindsight.evaluation.run` (backend hf is set here).
#
# Submit FROM your work dir (results/ and logs/ are created there), e.g. KMeans vs uniform vs blind at K=100:
#   cd ~/scratch/hindsight && mkdir -p logs
#   CUR=hf:hindsight-cs7641/hindsight-curated-frames      # or the local out/ folder from curate_array.sbatch
#   for M in kmeans uniform; do
#     sbatch "$HINDSIGHT_REPO/pace/scripts/vlm_eval.sbatch" --condition curated --curated-dir $CUR --method $M --k 100 --image-tokens 140
#   done
#   sbatch "$HINDSIGHT_REPO/pace/scripts/vlm_eval.sbatch" --condition blind --curated-dir $CUR --method kmeans --k 100
# Then: python -m hindsight.evaluation.compare results/vlm/curated_kmeans_K100 results/vlm/curated_uniform_K100 results/vlm/blind
#
# Needs the Gemma 4 env (Transformers 5, see docs/PACE.md section 8); set HINDSIGHT_ENV to it.
# Gemma 4 E4B in bf16 is ~18 GB + frames, which fits an L40S (48 GB). Re-submitting resumes a killed run.
#SBATCH -J hs-vlm
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N1 --ntasks-per-node=4 --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env-gemma4}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"

echo "== $(hostname) at $(date): $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m hindsight.evaluation.run --backend hf --device auto "$@"
echo "== finished at $(date)"
