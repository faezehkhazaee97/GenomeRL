#!/bin/bash
set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
cd "${REPO_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/group/gquongrp/workspaces/baran/miniconda3/envs/genomerl/bin/python}"

TASK_NAME="${TASK_NAME:?Set TASK_NAME, e.g. gue_tf_binding or gue_splice_site}"
HF_CONFIG="${HF_CONFIG:?Set HF_CONFIG to the GUE Hugging Face config name}"
HF_DATASET="${HF_DATASET:-leannmlindsey/GUE}"
SEQUENCE_COL="${SEQUENCE_COL:-sequence}"
LABEL_COL="${LABEL_COL:-label}"
MODELS="${MODELS:-baseline frozen finetuned}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-500}"

"${PYTHON_BIN}" -m src.data.prepare_gue_task \
  --task-name "${TASK_NAME}" \
  --hf-dataset "${HF_DATASET}" \
  --hf-config "${HF_CONFIG}" \
  --sequence-col "${SEQUENCE_COL}" \
  --label-col "${LABEL_COL}"

generated_dir="configs/generated/${TASK_NAME}"
processed_dir="data/processed/${TASK_NAME}"
task_suffix="${TASK_NAME#gue_}"

baseline_config="${generated_dir}/baseline_finetuned_dnabert2.yaml"
frozen_config="${generated_dir}/grpo_frozen_env.yaml"
finetuned_config="${generated_dir}/grpo_finetuned_env.yaml"

"${PYTHON_BIN}" -m src.utils.render_task_config \
  --template configs/baseline_finetuned_dnabert2.yaml \
  --output "${baseline_config}" \
  --task-name "${TASK_NAME}" \
  --project-name "GenomeRLFineTunedBaseline_${task_suffix}" \
  --processed-dir "${processed_dir}" \
  --output-dir "results/baselines/dnabert2_bpe_finetuned_${task_suffix}"

"${PYTHON_BIN}" -m src.utils.render_task_config \
  --template configs/grpo_frozen_env.yaml \
  --output "${frozen_config}" \
  --task-name "${TASK_NAME}" \
  --project-name "GenomeRLFrozenEnv_${task_suffix}" \
  --processed-dir "${processed_dir}" \
  --output-dir "results/rl_runs/grpo_frozen_env_${task_suffix}"

"${PYTHON_BIN}" -m src.utils.render_task_config \
  --template configs/grpo_finetuned_env.yaml \
  --output "${finetuned_config}" \
  --task-name "${TASK_NAME}" \
  --project-name "GenomeRLFineTunedEnv_${task_suffix}" \
  --processed-dir "${processed_dir}" \
  --output-dir "results/rl_runs/grpo_finetuned_env_${task_suffix}"

if [[ " ${MODELS} " == *" baseline "* ]]; then
  echo "Submitting fine-tuned DNABERT-2 baseline for ${TASK_NAME}"
  bash scripts/submit_dnabert2_finetuned_baseline_compat_a100.sh \
    python -m src.eval.evaluate_dnabert2_finetuned_baseline \
      --config "${baseline_config}"
fi

if [[ " ${MODELS} " == *" frozen "* ]]; then
  echo "Submitting frozen-env RL grouping for ${TASK_NAME}"
  bash scripts/submit_train_agent_frozen_env_compat_a100.sh \
    python -m src.rl.train_agent_frozen_env \
      --config "${frozen_config}" \
      --max-steps "${MAX_STEPS}" \
      --batch-size "${BATCH_SIZE}"
fi

if [[ " ${MODELS} " == *" finetuned "* ]]; then
  echo "Submitting fine-tuned RL grouping for ${TASK_NAME}"
  bash scripts/submit_train_agent_finetuned_env_compat_a100.sh \
    python -m src.rl.train_agent_finetuned_env \
      --config "${finetuned_config}" \
      --max-steps "${MAX_STEPS}" \
      --batch-size "${BATCH_SIZE}"
fi
