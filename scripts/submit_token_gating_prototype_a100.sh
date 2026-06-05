#!/usr/bin/env bash

set -euo pipefail

cd /group/gquongrp/workspaces/baran/GenomeRL
mkdir -p results/slurm

sbatch scripts/slurm/token_gating_prototype_a100.sbatch "$@"
