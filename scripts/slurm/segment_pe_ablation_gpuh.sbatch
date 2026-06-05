#!/bin/bash
#SBATCH --job-name=segment-pe
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

echo "======== Segment-Aware Positional Encoding Ablation ========"
echo "Comparing: Standard mean-pooling vs. segment-PE on boundary-sensitive tasks"
echo ""

# Test 1: Promoter detection (main task)
echo "======== Task 1: Promoter Detection (with segment PE) ========"
python -m src.rl.train_agent_finetuned_env --config configs/grpo_finetuned_env_segment_pe_promoter.yaml

echo "======== Evaluating Promoter ========"
python -m src.eval.evaluate_finetuned_env \
    --config configs/grpo_finetuned_env_segment_pe_promoter.yaml \
    --checkpoint results/rl_runs/grpo_finetuned_env_segment_pe_promoter/agent_env_last.pt \
    --split test --mode sampled \
    --output-path results/rl_runs/grpo_finetuned_env_segment_pe_promoter/eval_test_sampled.json

echo ""
echo "======== Task 2: Splice-site Prediction (with segment PE) ========"
python -m src.rl.train_agent_finetuned_env --config configs/grpo_finetuned_env_segment_pe_splice_site.yaml

echo "======== Evaluating Splice-site ========"
python -m src.eval.evaluate_finetuned_env \
    --config configs/grpo_finetuned_env_segment_pe_splice_site.yaml \
    --checkpoint results/rl_runs/grpo_finetuned_env_segment_pe_splice_site/agent_env_last.pt \
    --split test --mode sampled \
    --output-path results/rl_runs/grpo_finetuned_env_segment_pe_splice_site/eval_test_sampled.json

echo ""
echo "======== Segment PE Ablation Complete ========"
echo ""
echo "Comparison (segment PE vs. standard mean-pooling):"
echo ""
python3 << 'PYTHON_EOF'
import json
from pathlib import Path

tasks = [
    ("promoter", "Promoter Detection"),
    ("splice_site", "Splice-site Prediction"),
]

print("Task".ljust(25) + "Standard RL".ljust(20) + "Segment PE RL".ljust(20) + "Improvement")
print("-" * 85)

for task_key, task_name in tasks:
    # Load standard RL result
    std_path = Path(f"results/rl_runs/grpo_finetuned_env_{task_key}/eval_test_sampled.json")
    seg_path = Path(f"results/rl_runs/grpo_finetuned_env_segment_pe_{task_key}/eval_test_sampled.json")

    std_f1 = None
    seg_f1 = None

    if std_path.exists():
        with open(std_path) as f:
            std_f1 = json.load(f).get("f1")

    if seg_path.exists():
        with open(seg_path) as f:
            seg_f1 = json.load(f).get("f1")

    if std_f1 and seg_f1:
        improvement = seg_f1 - std_f1
        pct_change = (improvement / std_f1) * 100 if std_f1 > 0 else 0
        print(
            f"{task_name:<25} "
            f"{std_f1:.4f}".ljust(20) +
            f"{seg_f1:.4f}".ljust(20) +
            f"{improvement:+.4f} ({pct_change:+.1f}%)"
        )
PYTHON_EOF

echo ""
echo "Results saved to:"
echo "  - results/rl_runs/grpo_finetuned_env_segment_pe_promoter/"
echo "  - results/rl_runs/grpo_finetuned_env_segment_pe_splice_site/"
