#!/bin/bash
# Frame curation on SuperMemory-VQA: one Slurm array task per line of sessions.txt (up to 10 at once).
# Each task downloads one session and writes KMeans + same-size uniform keyframe videos for every K.
#
# Submit FROM your work dir (data/, out/, cache/, logs/ are created there):
#   cd ~/scratch/hindsight
#   python -m hindsight.curation.make_sessions --n 30 --per-person 3 --out sessions.txt
#   mkdir -p logs && sbatch --array=0-$(( $(wc -l < sessions.txt) - 1 ))%10 "$HINDSIGHT_REPO/pace/scripts/curate_array.sbatch"
# Then (login node, no GPU):
#   python -m hindsight.curation.analyze --out out --results results --k-bar 100
#
# Set HINDSIGHT_REPO to your clone if it isn't ~/scratch/hindsight/HindSight.
# GPU: request an NVIDIA type explicitly; mi210 / r9700 are AMD and PyTorch would fall back to CPU.
#SBATCH -J hs-curate
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N1 --ntasks-per-node=4 --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"

echo "== task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python -m hindsight.curation.run_session --index "${SLURM_ARRAY_TASK_ID}" --sessions sessions.txt \
  --ks 32 64 100 150 --delete-video

echo "== finished at $(date)"
