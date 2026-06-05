#!/bin/bash
set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
cd "${REPO_DIR}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-500}"
SEED="${SEED:-42}"
BETAS="${BETAS:-0.01 0.02 0.05}"

for beta in ${BETAS}; do
  beta_tag="${beta/./p}"
  output_dir="results/rl_runs/grpo_finetuned_env_beta_${beta_tag}_seed_${SEED}"
  echo "Submitting beta=${beta}, seed=${SEED} -> ${output_dir}"
  bash scripts/submit_train_agent_finetuned_env_compat_a100.sh \
    python -m src.rl.train_agent_finetuned_env \
      --config configs/grpo_finetuned_env.yaml \
      --max-steps "${MAX_STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --beta "${beta}" \
      --seed "${SEED}" \
      --output-dir "${output_dir}"
done
