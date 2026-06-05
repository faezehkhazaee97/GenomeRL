#!/bin/bash
set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
cd "${REPO_DIR}"

CHECKPOINT="${CHECKPOINT:-results/rl_runs/grpo/agent_classifier_last.pt}"
CONFIG="${CONFIG:-configs/grpo.yaml}"
SPLIT="${SPLIT:-test}"
THRESHOLD="${THRESHOLD:-0.11}"
MOTIFS="${MOTIFS:-TATA CAAT GCGC}"

for motif in ${MOTIFS}; do
  motif_tag="$(echo "${motif}" | tr '[:upper:]' '[:lower:]')"
  output_path="results/analysis/boundary_interpretability_${motif_tag}_t${THRESHOLD/./}.json"
  echo "Submitting motif=${motif}, threshold=${THRESHOLD} -> ${output_path}"
  bash scripts/submit_boundary_analysis_a100.sh \
    python -m src.eval.analyze_boundary_interpretability \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --split "${SPLIT}" \
      --motif "${motif}" \
      --threshold "${THRESHOLD}" \
      --output-path "${output_path}"
done
