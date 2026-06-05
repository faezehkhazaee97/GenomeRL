#!/bin/bash
#SBATCH --job-name=boundary-analysis
#SBATCH --partition=gpu-a100-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
CONDA_BASE="/group/gquongrp/workspaces/baran/miniconda3"
ENV_NAME="genomerl"

mkdir -p "${REPO_DIR}/results/slurm"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
cd "${REPO_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Running boundary interpretability analysis"
echo "Python: $(which python)"
nvidia-smi || true

if [ "$#" -gt 0 ]; then
  echo "Running command:"
  printf ' %q' "$@"
  echo
  "$@"
else
  python -m src.eval.analyze_boundary_interpretability
fi
