#!/bin/bash
#SBATCH --job-name=baseline-cnn
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
export TOKENIZERS_PARALLELISM=false

cd "${REPO_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Running CNN baseline on gpu-a100-h"
nvidia-smi || true

python -m src.eval.evaluate_baseline --config configs/baseline.yaml --model-type cnn "$@"
