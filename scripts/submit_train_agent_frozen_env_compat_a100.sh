#!/bin/bash
set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"

cd "${REPO_DIR}"
mkdir -p results/slurm

sbatch scripts/slurm/train_agent_frozen_env_compat_a100.sbatch "$@"
