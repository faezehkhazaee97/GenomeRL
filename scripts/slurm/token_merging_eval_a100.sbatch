#!/bin/bash
#SBATCH --job-name=token-merging-eval
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
mkdir -p "${REPO_DIR}/results/token_merging"
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
echo "Python: $(which python)"
nvidia-smi || true

echo ""
echo "=== Token Merging: Promoter ==="
python -m src.eval.evaluate_token_merging \
    --task promoter \
    --batch-size 32 \
    --output results/token_merging/token_merging_promoter.json

echo ""
echo "=== Token Merging: Splice Site ==="
python -m src.eval.evaluate_token_merging \
    --task splice_site \
    --batch-size 32 \
    --output results/token_merging/token_merging_splice_site.json

echo ""
echo "=== All done ==="
cat results/token_merging/token_merging_promoter.json
echo ""
cat results/token_merging/token_merging_splice_site.json
