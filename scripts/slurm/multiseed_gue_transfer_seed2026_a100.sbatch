#!/bin/bash
#SBATCH --job-name=gue-transfer-s2026
#SBATCH --partition=gpu-a100-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
CONDA_BASE="/group/gquongrp/workspaces/baran/miniconda3"
ENV_NAME="dnabert2-compat"
SEED=2026

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
echo "Seed: ${SEED}"
echo "Python: $(which python)"
nvidia-smi || true

TASKS=(
  "gue_human_tf_0:human_tf_0"
  "gue_human_tf_1:human_tf_1"
  "gue_human_tf_2:human_tf_2"
  "gue_human_tf_3:human_tf_3"
  "gue_human_tf_4:human_tf_4"
  "gue_splice_site:splice_site"
)

for TASK_PAIR in "${TASKS[@]}"; do
  CONFIG_KEY="${TASK_PAIR%%:*}"
  RESULT_KEY="${TASK_PAIR##*:}"
  CONFIG="configs/generated/${CONFIG_KEY}/grpo_finetuned_env.yaml"
  OUTPUT_DIR="results/rl_runs/grpo_finetuned_env_${RESULT_KEY}_seed_${SEED}"
  EVAL_OUTPUT="${OUTPUT_DIR}/eval_test_sampled.json"

  echo "========================================================"
  echo "Task: ${RESULT_KEY}  Seed: ${SEED}"
  echo "========================================================"

  python -m src.rl.train_agent_finetuned_env \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}"

  python -m src.eval.evaluate_finetuned_env \
    --config "${CONFIG}" \
    --checkpoint "${OUTPUT_DIR}/agent_env_last.pt" \
    --split test \
    --mode sampled \
    --output-path "${EVAL_OUTPUT}"

  echo "Done: ${RESULT_KEY}  →  ${EVAL_OUTPUT}"
done

echo "All tasks complete for seed ${SEED}."
