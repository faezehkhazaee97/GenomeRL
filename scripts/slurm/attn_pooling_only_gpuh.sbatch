#!/bin/bash
#SBATCH --job-name=attn-pool
#SBATCH --partition=gpuh
#SBATCH --account=quonbiogrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
source /group/gquongrp/workspaces/baran/miniconda3/etc/profile.d/conda.sh
conda activate dnabert2-compat
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/group/gquongrp/workspaces/baran/GenomeRL/.hf-cache-compat HF_HUB_DISABLE_TELEMETRY=1
cd /group/gquongrp/workspaces/baran/GenomeRL

echo "======== Attention-based Segment Pooling Training ========"
python -m src.rl.train_agent_finetuned_env --config configs/grpo_finetuned_env_attn_pool.yaml

echo "======== Attention Pooling Evaluation ========"
python -m src.eval.evaluate_finetuned_env \
    --config configs/grpo_finetuned_env_attn_pool.yaml \
    --checkpoint results/rl_runs/grpo_finetuned_env_attn_pool/agent_env_last.pt \
    --split test --mode sampled \
    --output-path results/rl_runs/grpo_finetuned_env_attn_pool/eval_test_sampled.json

echo "Done."
