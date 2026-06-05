#!/bin/bash
#SBATCH --job-name=dnabert2-debug
#SBATCH --partition=gpu-a100-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
CONDA_BASE="/group/gquongrp/workspaces/baran/miniconda3"
ENV_NAME="genomerl"

mkdir -p "${REPO_DIR}/results/slurm"
mkdir -p "${REPO_DIR}/.hf-cache"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${REPO_DIR}/.hf-cache"
export TRANSFORMERS_CACHE="${REPO_DIR}/.hf-cache/transformers"
export HF_HUB_DISABLE_TELEMETRY=1

cd "${REPO_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Running DNABERT-2 debug smoke test on gpu-a100-h"
echo "HF_HOME=${HF_HOME}"
nvidia-smi || true

python -m src.eval.evaluate_baseline \
  --config configs/baseline.yaml \
  --model-type dnabert2 \
  --epochs 1 \
  --batch-size 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  "$@"
