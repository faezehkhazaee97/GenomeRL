#!/bin/bash
#SBATCH --job-name=finetuned-env-eval
#SBATCH --partition=gpu-a100-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
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
echo "Running fine-tuned-env evaluation in dnabert2-compat on gpu-a100-h"
echo "Python: $(which python)"
echo "HF_HOME=${HF_HOME}"
nvidia-smi || true

if [ "$#" -gt 0 ]; then
  echo "Running command:"
  printf ' %q' "$@"
  echo
  "$@"
else
  python -m src.eval.evaluate_finetuned_env --config configs/grpo_finetuned_env.yaml --checkpoint results/rl_runs/grpo_finetuned_env/agent_env_last.pt --split valid --mode sampled
fi
