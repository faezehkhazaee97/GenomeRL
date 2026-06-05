#!/bin/bash
# Submit multi-seed GUE transfer jobs (seeds 1337 and 2026) for all 6 tasks.
# Seed 42 runs already exist. This adds the two missing seeds so each task has 3 seeds.
# Usage: bash scripts/submit_multiseed_gue_transfer.sh

set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"

TASKS=(
  "gue_human_tf_0:human_tf_0"
  "gue_human_tf_1:human_tf_1"
  "gue_human_tf_2:human_tf_2"
  "gue_human_tf_3:human_tf_3"
  "gue_human_tf_4:human_tf_4"
  "gue_splice_site:splice_site"
)

SEEDS=(1337 2026)

for TASK_PAIR in "${TASKS[@]}"; do
  CONFIG_KEY="${TASK_PAIR%%:*}"   # e.g. gue_human_tf_0
  RESULT_KEY="${TASK_PAIR##*:}"   # e.g. human_tf_0
  CONFIG="configs/generated/${CONFIG_KEY}/grpo_finetuned_env.yaml"

  for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="results/rl_runs/grpo_finetuned_env_${RESULT_KEY}_seed_${SEED}"
    EVAL_OUTPUT="results/rl_runs/grpo_finetuned_env_${RESULT_KEY}_seed_${SEED}/eval_test_sampled.json"

    echo "=== Submitting train: ${RESULT_KEY} seed=${SEED} ==="
    TRAIN_JOB=$(sbatch \
      --job-name="ft-train-${RESULT_KEY}-s${SEED}" \
      --parsable \
      "${REPO_DIR}/scripts/slurm/train_agent_finetuned_env_compat_a100.sbatch" \
      python -m src.rl.train_agent_finetuned_env \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}")
    echo "  Train job ID: ${TRAIN_JOB}"

    echo "=== Submitting eval: ${RESULT_KEY} seed=${SEED} (after job ${TRAIN_JOB}) ==="
    EVAL_JOB=$(sbatch \
      --job-name="ft-eval-${RESULT_KEY}-s${SEED}" \
      --parsable \
      --dependency="afterok:${TRAIN_JOB}" \
      "${REPO_DIR}/scripts/slurm/evaluate_finetuned_env_compat_a100.sbatch" \
      python -m src.eval.evaluate_finetuned_env \
        --config "${CONFIG}" \
        --checkpoint "${OUTPUT_DIR}/agent_env_last.pt" \
        --split test \
        --mode sampled \
        --output-path "${EVAL_OUTPUT}")
    echo "  Eval job ID: ${EVAL_JOB}"
    echo ""
  done
done

echo "All multi-seed GUE transfer jobs submitted."
echo "Results will appear in: results/rl_runs/grpo_finetuned_env_<task>_seed_<seed>/"
