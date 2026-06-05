#!/bin/bash
#SBATCH --job-name=new-exps
#SBATCH --partition=gpuh
#SBATCH --account=quonbiogrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
source /group/gquongrp/workspaces/baran/miniconda3/etc/profile.d/conda.sh
conda activate dnabert2-compat
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/group/gquongrp/workspaces/baran/GenomeRL/.hf-cache-compat HF_HUB_DISABLE_TELEMETRY=1
cd /group/gquongrp/workspaces/baran/GenomeRL

run_train() {
    echo "======== Training: $1 ========"
    python -m src.rl.train_agent_finetuned_env --config "$1"
}

run_eval() {
    local cfg=$1 ckpt=$2 out=$3
    echo "======== Eval: $ckpt ========"
    python -m src.eval.evaluate_finetuned_env \
        --config "$cfg" --checkpoint "$ckpt" \
        --split test --mode sampled --output-path "$out"
}

run_lagrangian() {
    echo "======== Lagrangian: $1 ========"
    python -m src.rl.train_agent_lagrangian --config "$1"
}

# --- 1. RL steps sensitivity curve ---
for steps in 100 200 300 400 500; do
    run_train "configs/grpo_finetuned_env_steps_${steps}.yaml"
    run_eval "configs/grpo_finetuned_env_steps_${steps}.yaml" \
             "results/rl_runs/grpo_finetuned_env_steps_${steps}/agent_env_last.pt" \
             "results/rl_runs/grpo_finetuned_env_steps_${steps}/eval_test_sampled.json"
done

# --- 2. Lagrangian constrained RL ---
run_lagrangian configs/grpo_finetuned_env_lagrangian.yaml
run_eval configs/grpo_finetuned_env_lagrangian.yaml \
         results/rl_runs/grpo_finetuned_env_lagrangian/agent_env_last.pt \
         results/rl_runs/grpo_finetuned_env_lagrangian/eval_test_sampled.json

# --- 3. Attention-based segment pooling ---
run_train configs/grpo_finetuned_env_attn_pool.yaml
run_eval configs/grpo_finetuned_env_attn_pool.yaml \
         results/rl_runs/grpo_finetuned_env_attn_pool/agent_env_last.pt \
         results/rl_runs/grpo_finetuned_env_attn_pool/eval_test_sampled.json

echo "All new experiments complete."
