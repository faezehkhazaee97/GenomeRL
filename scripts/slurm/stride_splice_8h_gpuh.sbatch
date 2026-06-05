#!/bin/bash
#SBATCH --job-name=stride-splice
#SBATCH --partition=gpuh
#SBATCH --account=quonbiogrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
set -euo pipefail
source /group/gquongrp/workspaces/baran/miniconda3/etc/profile.d/conda.sh
conda activate dnabert2-compat
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/group/gquongrp/workspaces/baran/GenomeRL/.hf-cache-compat HF_HUB_DISABLE_TELEMETRY=1
cd /group/gquongrp/workspaces/baran/GenomeRL
python -m src.rl.train_stride_pooling --config configs/generated/gue_splice_site/stride_pooling.yaml
