#!/bin/bash
#SBATCH -J hs-vq2d
#SBATCH -N1 --ntasks-per-node=8 --mem=64G --gres=gpu:H100:1
#SBATCH -t8:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# CS 7641 HINDSIGHT — train the vq2d head on Ego4D features.
# Resubmit after the 8h wall limit; the run resumes from ckpt/vq2d/latest.pt.
set -euo pipefail
module load anaconda3 cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/hindsight/envs/hs
cd ~/scratch/hindsight/HINDSIGHT
export EGO4D_ROOT=~/scratch/hindsight/ego4d
export HINDSIGHT_RUNS=~/scratch/hindsight/runs
export HINDSIGHT_CKPT=~/scratch/hindsight/ckpt
nvidia-smi
python -m hindsight.benchmarks.train vq2d
python -m hindsight.benchmarks.eval vq2d --checkpoint $HINDSIGHT_CKPT/vq2d/best.pt
