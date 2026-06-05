#!/bin/bash
#SBATCH --job-name=eval-4layer
#SBATCH --partition=gpuh
#SBATCH --account=quonbiogrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
set -euo pipefail
source /group/gquongrp/workspaces/baran/miniconda3/etc/profile.d/conda.sh
conda activate dnabert2-compat
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/group/gquongrp/workspaces/baran/GenomeRL/.hf-cache-compat HF_HUB_DISABLE_TELEMETRY=1
cd /group/gquongrp/workspaces/baran/GenomeRL
python -m src.eval.evaluate_finetuned_env \
  --config configs/grpo_finetuned_env_4layers.yaml \
  --checkpoint results/rl_runs/grpo_finetuned_env_4layers/agent_env_last.pt \
  --split test --mode sampled \
  --output-path results/rl_runs/grpo_finetuned_env_4layers/eval_test_sampled.json
