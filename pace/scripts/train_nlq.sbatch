#!/bin/bash
#SBATCH -J hs-nlq
#SBATCH -N1 --ntasks-per-node=8 --mem=64G --gres=gpu:H100:1
#SBATCH -t8:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# CS 7641 HINDSIGHT — train the nlq head on Ego4D features.
# Resubmit after the 8h wall limit; the run resumes from ckpt/nlq/latest.pt.
set -euo pipefail
module load anaconda3 cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/hindsight/envs/hs
cd ~/scratch/hindsight/HINDSIGHT
export EGO4D_ROOT=~/scratch/hindsight/ego4d
export HINDSIGHT_RUNS=~/scratch/hindsight/runs
export HINDSIGHT_CKPT=~/scratch/hindsight/ckpt
nvidia-smi
python -m hindsight.benchmarks.train nlq
python -m hindsight.benchmarks.eval nlq --checkpoint $HINDSIGHT_CKPT/nlq/best.pt
