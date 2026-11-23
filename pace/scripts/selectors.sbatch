#!/bin/bash
# Which 100 frames per session? Coverage of uniform, KMeans, agglomerative and question-aware selectors. CPU only.
# Needs cache/ and out/ from the full run; reuses results/full/grounded/text_emb.npz.
#
# Submit FROM the work dir of the full run:
#   cd ~/scratch/hindsight && mkdir -p logs && sbatch "$HINDSIGHT_REPO/pace/scripts/selectors.sbatch"
# Watch:   tail -f logs/hs-selectors-<jobid>.out
# Result:  results/full/selectors/SELECTORS.md
# Extra arguments are passed through, e.g.  sbatch ./selectors.sbatch --gap 10
#SBATCH -J hs-selectors
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=4 --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "== $(hostname) at $(date), ${SLURM_CPUS_PER_TASK:-?} CPUs"
python -m hindsight.curation.frame_selectors --cache cache --out out --results results/full/selectors \
    --text-cache results/full/grounded/text_emb.npz --device cpu "$@"
echo "== finished at $(date)"
