#!/bin/bash
#SBATCH --job-name=budget-parity-20ep
#SBATCH --partition=gpu-a100-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
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
echo "Python: $(which python)"
nvidia-smi || true

echo ""
echo "=== Training 20-epoch fine-tuned DNABERT-2 baselines for budget parity ==="
echo "=== These match the 20-epoch human_tf_1 analysis already in the paper ==="

for TASK in human_tf_0 human_tf_2 human_tf_3 human_tf_4 splice_site; do
    echo ""
    echo "--- Task: ${TASK} ---"
    python -m src.eval.evaluate_dnabert2_finetuned_baseline \
        --config configs/generated/gue_${TASK}/baseline_finetuned_dnabert2.yaml \
        --epochs 20 \
        --output-dir results/baselines/dnabert2_bpe_finetuned_${TASK}_20epochs
    echo "Done: ${TASK}"
done

echo ""
echo "=== All tasks complete ==="
echo ""
echo "=== Results Summary ==="
for TASK in human_tf_0 human_tf_2 human_tf_3 human_tf_4 splice_site; do
    echo -n "${TASK} 20ep: "
    python3 -c "
import json
d = json.load(open('results/baselines/dnabert2_bpe_finetuned_${TASK}_20epochs/metrics.json'))
print(f'acc={d[\"accuracy\"]:.4f} f1={d[\"macro_f1\"]:.4f}')
" 2>/dev/null || echo "no result"
done
