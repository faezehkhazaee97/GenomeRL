#!/bin/bash
#SBATCH --job-name=gumbel-tf1long
#SBATCH --partition=gpu-a100-h
#SBATCH --account=quonbiogrp
#SBATCH --nodelist=gpu-3-38
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
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

run_gumbel() {
    echo "========================================"
    echo "Gumbel segmenter: $1"
    echo "========================================"
    python -m src.rl.train_gumbel_segmenter --config "$1"
}

run_baseline() {
    echo "========================================"
    echo "Baseline (long): $1"
    echo "========================================"
    python -m src.eval.evaluate_baseline --config "$1"
}

# --- Gumbel-Softmax segmenter: promoter + all transfer tasks ---
run_gumbel configs/gumbel_segmenter_promoter.yaml
run_gumbel configs/generated/gue_human_tf_0/gumbel_segmenter.yaml
run_gumbel configs/generated/gue_human_tf_1/gumbel_segmenter.yaml
run_gumbel configs/generated/gue_human_tf_2/gumbel_segmenter.yaml
run_gumbel configs/generated/gue_human_tf_3/gumbel_segmenter.yaml
run_gumbel configs/generated/gue_human_tf_4/gumbel_segmenter.yaml
run_gumbel configs/generated/gue_splice_site/gumbel_segmenter.yaml

# --- TF1 longer baseline (20 epochs) ---
run_baseline configs/generated/gue_human_tf_1/baseline_finetuned_dnabert2_long.yaml

echo "All jobs complete."
