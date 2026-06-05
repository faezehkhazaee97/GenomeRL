#!/bin/bash
set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
cd "${REPO_DIR}"

SEEDS="${SEEDS:-42 1337 2026}"
PROTOTYPE_STEPS="${PROTOTYPE_STEPS:-1000}"
FROZEN_STEPS="${FROZEN_STEPS:-500}"
FINETUNED_STEPS="${FINETUNED_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FROZEN_BETA="${FROZEN_BETA:-0.05}"
FINETUNED_BETA="${FINETUNED_BETA:-0.02}"
RUNS="${RUNS:-prototype frozen finetuned_baseline finetuned_rl}"

for seed in ${SEEDS}; do
  echo "=== Seed ${seed} ==="

  if [[ " ${RUNS} " == *" prototype "* ]]; then
    bash scripts/submit_train_agent_a100.sh \
      python -m src.rl.train_agent \
        --config configs/grpo.yaml \
        --max-steps "${PROTOTYPE_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --seed "${seed}" \
        --output-dir "results/rl_runs/grpo_seed_${seed}"
  fi

  if [[ " ${RUNS} " == *" frozen "* ]]; then
    bash scripts/submit_train_agent_frozen_env_compat_a100.sh \
      python -m src.rl.train_agent_frozen_env \
        --config configs/grpo_frozen_env.yaml \
        --max-steps "${FROZEN_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --beta "${FROZEN_BETA}" \
        --seed "${seed}" \
        --output-dir "results/rl_runs/grpo_frozen_env_seed_${seed}"
  fi

  if [[ " ${RUNS} " == *" finetuned_baseline "* ]]; then
    bash scripts/submit_dnabert2_finetuned_baseline_compat_a100.sh \
      python -m src.eval.evaluate_dnabert2_finetuned_baseline \
        --config configs/baseline_finetuned_dnabert2.yaml \
        --seed "${seed}" \
        --output-dir "results/baselines/dnabert2_bpe_finetuned_seed_${seed}"
  fi

  if [[ " ${RUNS} " == *" finetuned_rl "* ]]; then
    bash scripts/submit_train_agent_finetuned_env_compat_a100.sh \
      python -m src.rl.train_agent_finetuned_env \
        --config configs/grpo_finetuned_env.yaml \
        --max-steps "${FINETUNED_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --beta "${FINETUNED_BETA}" \
        --seed "${seed}" \
        --output-dir "results/rl_runs/grpo_finetuned_env_seed_${seed}"
  fi
done
