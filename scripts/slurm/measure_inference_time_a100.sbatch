#!/bin/bash
#SBATCH --job-name=inference-timing
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
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m src.eval.measure_inference_time \
    --config configs/grpo_finetuned_env.yaml \
    --checkpoint results/rl_runs/grpo_finetuned_env/agent_env_last.pt \
    --output-path results/analysis/inference_timing.json \
    --split test \
    --batch-size 16 \
    --n-warmup 10 \
    --n-batches 100
