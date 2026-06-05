#!/bin/bash
#SBATCH --job-name=all-stride-pool
#SBATCH --partition=gpu-a100-h
#SBATCH --account=quonbiogrp
#SBATCH --nodelist=gpu-3-38
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
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
export PYTHONUNBUFFERED=1
export HF_HOME="${REPO_DIR}/.hf-cache-compat"
export HF_HUB_DISABLE_TELEMETRY=1

cd "${REPO_DIR}"

nvidia-smi || true

run() {
    echo "========================================"
    echo "Running: $1"
    echo "========================================"
    python -m src.rl.train_stride_pooling --config "$1"
}

run configs/stride_pooling_promoter.yaml
run configs/generated/gue_human_tf_0/stride_pooling.yaml
run configs/generated/gue_human_tf_1/stride_pooling.yaml
run configs/generated/gue_human_tf_2/stride_pooling.yaml
run configs/generated/gue_human_tf_3/stride_pooling.yaml
run configs/generated/gue_human_tf_4/stride_pooling.yaml
run configs/generated/gue_splice_site/stride_pooling.yaml

echo "All stride-pooling runs complete."
