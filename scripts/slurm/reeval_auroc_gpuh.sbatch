#!/bin/bash
#SBATCH --job-name=reeval-auroc
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

run() {
    local cfg=$1 ckpt=$2 out=$3
    echo "=== Evaluating: $ckpt ==="
    python -m src.eval.evaluate_finetuned_env \
        --config "$cfg" --checkpoint "$ckpt" \
        --split test --mode sampled \
        --output-path "$out"
}

# Main promoter runs (seed 42)
run configs/grpo_finetuned_env.yaml \
    results/rl_runs/grpo_finetuned_env/agent_env_last.pt \
    results/rl_runs/grpo_finetuned_env/eval_test_sampled.json

run configs/grpo_finetuned_env.yaml \
    results/rl_runs/grpo_finetuned_env_seed_1337/agent_env_last.pt \
    results/rl_runs/grpo_finetuned_env_seed_1337/eval_test_sampled.json

run configs/grpo_finetuned_env.yaml \
    results/rl_runs/grpo_finetuned_env_seed_2026/agent_env_last.pt \
    results/rl_runs/grpo_finetuned_env_seed_2026/eval_test_sampled.json

# TF transfer tasks (seed 42)
for task in human_tf_0 human_tf_1 human_tf_2 human_tf_3 human_tf_4; do
    run "configs/generated/gue_${task}/grpo_finetuned_env.yaml" \
        "results/rl_runs/grpo_finetuned_env_${task}/agent_env_last.pt" \
        "results/rl_runs/grpo_finetuned_env_${task}/eval_test_sampled.json"
done

# Splice site
run configs/generated/gue_splice_site/grpo_finetuned_env.yaml \
    results/rl_runs/grpo_finetuned_env_splice_site/agent_env_last.pt \
    results/rl_runs/grpo_finetuned_env_splice_site/eval_test_sampled.json

# 4-layer ablation
run configs/grpo_finetuned_env_4layers.yaml \
    results/rl_runs/grpo_finetuned_env_4layers/agent_env_last.pt \
    results/rl_runs/grpo_finetuned_env_4layers/eval_test_sampled.json

echo "All AUROC re-evaluations complete."
