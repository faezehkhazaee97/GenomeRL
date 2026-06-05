#!/usr/bin/env bash

set -euo pipefail

cd /group/gquongrp/workspaces/baran/GenomeRL
mkdir -p results/slurm

sbatch scripts/slurm/train_agent_smoke_a100.sbatch "$@"
