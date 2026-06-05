#!/bin/bash
#SBATCH --job-name=lagrangian-all
#SBATCH --partition=gpuh
#SBATCH --account=quonbiogrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
source /group/gquongrp/workspaces/baran/miniconda3/etc/profile.d/conda.sh
conda activate dnabert2-compat
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/group/gquongrp/workspaces/baran/GenomeRL/.hf-cache-compat HF_HUB_DISABLE_TELEMETRY=1
cd /group/gquongrp/workspaces/baran/GenomeRL

# Run Lagrangian RL on remaining 6 GUE tasks
# Note: Promoter already done (grpo_finetuned_env_lagrangian)

echo "========== Starting Lagrangian RL Training on GUE Tasks =========="

TASKS=("human_tf_0" "human_tf_1" "human_tf_2" "human_tf_3" "human_tf_4" "splice_site")

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "======== Lagrangian RL: $TASK ========"
    CONFIG="configs/generated/gue_${TASK}/grpo_finetuned_env_lagrangian.yaml"

    if [ ! -f "$CONFIG" ]; then
        echo "Config not found: $CONFIG - creating it..."
        # Create config by copying promoter lagrangian and changing dataset paths
        python3 << 'PYTHON_EOF'
import yaml
import sys

# Load promoter config as template
with open('configs/grpo_finetuned_env_lagrangian.yaml') as f:
    config = yaml.safe_load(f)

task = sys.argv[1] if len(sys.argv) > 1 else 'human_tf_0'

# Update dataset paths
config['dataset']['train_path'] = f'data/processed/gue_{task}/train.jsonl'
config['dataset']['valid_path'] = f'data/processed/gue_{task}/valid.jsonl'
config['dataset']['test_path'] = f'data/processed/gue_{task}/test.jsonl'

# Determine num_labels based on task
if task == 'splice_site':
    config['env']['num_labels'] = 3  # donor/acceptor/neither
else:
    config['env']['num_labels'] = 2  # binary classification

# Adjust output directory
config['training']['output_dir'] = f'results/rl_runs/grpo_finetuned_env_{task}_lagrangian'

# Save config
import os
os.makedirs(f'configs/generated/gue_{task}', exist_ok=True)
with open(f'configs/generated/gue_{task}/grpo_finetuned_env_lagrangian.yaml', 'w') as f:
    yaml.dump(config, f)

print(f"Created config: configs/generated/gue_{task}/grpo_finetuned_env_lagrangian.yaml")
PYTHON_EOF
    fi

    python -m src.rl.train_agent_lagrangian --config "$CONFIG"

    echo "======== Lagrangian RL Eval: $TASK ========"
    python -m src.eval.evaluate_finetuned_env \
        --config "$CONFIG" \
        --checkpoint "results/rl_runs/grpo_finetuned_env_${TASK}_lagrangian/agent_env_last.pt" \
        --split test --mode sampled \
        --output-path "results/rl_runs/grpo_finetuned_env_${TASK}_lagrangian/eval_test_sampled.json"

    echo "Completed $TASK"
done

echo ""
echo "======== All Lagrangian RL tasks completed ========"
echo "Results saved to results/rl_runs/grpo_finetuned_env_*_lagrangian/"
