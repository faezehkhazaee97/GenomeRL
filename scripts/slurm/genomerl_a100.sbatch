#!/bin/bash
#SBATCH --job-name=genomerl
#SBATCH --partition=gpu-a100-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage:"
  echo "  sbatch scripts/slurm/genomerl_a100.sbatch <command> [args...]"
  exit 1
fi

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
CONDA_BASE="/group/gquongrp/workspaces/baran/miniconda3"
ENV_NAME="genomerl"

mkdir -p "${REPO_DIR}/results/slurm"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || true

echo "Running command:"
printf ' %q' "$@"
echo

"$@"
