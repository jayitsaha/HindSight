#!/bin/bash
# Question-grounded retrieval recall (phase 1): CPU only. Login nodes kill long CPU-heavy processes, so run it here.
#
# Submit FROM the work dir of the full run (it needs cache/ and out/):
#   cd ~/scratch/hindsight && mkdir -p logs && sbatch "$HINDSIGHT_REPO/pace/scripts/grounded.sbatch"
# Watch:   tail -f logs/hs-grounded-<jobid>.out
# Result:  results/full/grounded/RECALL.md
# Extra arguments are passed through, e.g.  sbatch .../grounded.sbatch --sessions evidence --pools all
#SBATCH -J hs-grounded
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "== $(hostname) at $(date), ${SLURM_CPUS_PER_TASK:-?} CPUs"
python -m hindsight.curation.grounded --cache cache --out out --results results/full/grounded --device cpu "$@"
echo "== finished at $(date)"
