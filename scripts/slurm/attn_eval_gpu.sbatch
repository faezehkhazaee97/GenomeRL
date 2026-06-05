#!/bin/bash
#SBATCH --job-name=attn-eval
#SBATCH --partition=gpuh
#SBATCH --account=quonbiogrp
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=60G --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x-%j.out

source /group/gquongrp/workspaces/baran/miniconda3/etc/profile.d/conda.sh
conda activate dnabert2-compat
export HF_HOME=/group/gquongrp/workspaces/baran/GenomeRL/.hf-cache-compat PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/GenomeRL

python -m src.eval.evaluate_finetuned_env \
    --config configs/grpo_finetuned_env_attn_pool.yaml \
    --checkpoint results/rl_runs/grpo_finetuned_env_attn_pool/agent_env_last.pt \
    --split test --mode sampled \
    --output-path results/rl_runs/grpo_finetuned_env_attn_pool/eval_test_sampled.json
