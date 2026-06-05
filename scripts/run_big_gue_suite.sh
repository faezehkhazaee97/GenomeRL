#!/bin/bash
set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
cd "${REPO_DIR}"

MODELS="${MODELS:-baseline frozen finetuned}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-500}"

tasks=(
  "gue_human_tf_0:human_tf_0"
  "gue_human_tf_1:human_tf_1"
  "gue_human_tf_2:human_tf_2"
  "gue_human_tf_3:human_tf_3"
  "gue_human_tf_4:human_tf_4"
  "gue_splice_site:splice_reconstructed"
)

for task_spec in "${tasks[@]}"; do
  task_name="${task_spec%%:*}"
  hf_config="${task_spec##*:}"
  echo "Submitting task ${task_name} (${hf_config})"
  TASK_NAME="${task_name}" \
  HF_CONFIG="${hf_config}" \
  MODELS="${MODELS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  MAX_STEPS="${MAX_STEPS}" \
  bash scripts/run_extra_gue_task_panel.sh
done
