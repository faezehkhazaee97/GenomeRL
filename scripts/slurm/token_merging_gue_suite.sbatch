#!/bin/bash
#SBATCH --job-name=token_merging_gue
#SBATCH --partition=gpuh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --output=results/slurm/token_merging_gue_%j.out
#SBATCH --error=results/slurm/token_merging_gue_%j.err

set -e

# Change to repo directory
cd /group/gquongrp/workspaces/baran/GenomeRL

# Create output directory
mkdir -p results/token_merging

# Token merging ratios to test (to match standard RL compression ~0.18)
# Standard RL achieves ~0.18 compression with 5 tokens on ~30-token sequences
# So we test merge_ratio values that achieve similar compression
MERGE_RATIOS=(0.25 0.33 0.5)  # Different compression targets

# GUE tasks
TASKS=("promoter" "human_tf_0" "human_tf_1" "human_tf_2" "human_tf_3" "human_tf_4" "splice_site")

echo "Starting token merging baseline evaluation on GUE tasks"
echo "Merge ratios: ${MERGE_RATIOS[@]}"
echo "Tasks: ${TASKS[@]}"
echo "==============================================="

# Run each task with different merge ratios
for task in "${TASKS[@]}"; do
    echo "Processing task: $task"

    for merge_ratio in "${MERGE_RATIOS[@]}"; do
        echo "  Running with merge_ratio=$merge_ratio"

        python3 src/rl/train_token_merging.py \
            --task "$task" \
            --data_dir ./data \
            --merge_ratio "$merge_ratio" \
            --merge_start_layer 6 \
            --num_epochs 20 \
            --batch_size 32 \
            --lr 1e-4 \
            --seed 42 \
            --output_dir results/token_merging

        echo "  Completed merge_ratio=$merge_ratio"
    done

    echo "Task $task completed"
done

echo "==============================================="
echo "All token merging experiments completed!"
echo "Results saved to results/token_merging/"
