#!/bin/bash
# Cut per-question frames (e.g. the learned frame picker's) from the raw session videos: one array task per slice
# of videos (task i takes every N-th video). CPU only. Submitted by submit_selected.sh; needs SELECTIONS, OUT, TASKS.
#SBATCH -J hs-cut
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=2 --mem=8G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
echo "== task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
python -m hindsight.evaluation.prepare_selected --selections "$SELECTIONS" --out "$OUT" \
    --task "${SLURM_ARRAY_TASK_ID}" --tasks "$TASKS"
echo "== finished at $(date)"
