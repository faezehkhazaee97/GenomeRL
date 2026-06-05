#!/bin/bash
#SBATCH --job-name=token-len-dist
#SBATCH --partition=gpu-a100-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
CONDA_BASE="/group/gquongrp/workspaces/baran/miniconda3"
ENV_NAME="dnabert2-compat"

mkdir -p "${REPO_DIR}/results/slurm"
mkdir -p "${REPO_DIR}/.hf-cache-compat"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${REPO_DIR}/.hf-cache-compat"
export HF_HUB_DISABLE_TELEMETRY=1

cd "${REPO_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Running token length distribution collection in dnabert2-compat on gpu-a100-h"
echo "Python: $(which python)"
nvidia-smi || true

if [ "$#" -gt 0 ]; then
  echo "Running command:"
  printf ' %q' "$@"
  echo
  "$@"
else
  python -m src.eval.collect_token_length_distribution \
    --config configs/grpo_finetuned_env.yaml \
    --checkpoint results/rl_runs/grpo_finetuned_env_seed_42/agent_env_last.pt \
    --split test \
    --threshold 0.11 \
    --output-path results/analysis/token_length_distributions.json
fi
