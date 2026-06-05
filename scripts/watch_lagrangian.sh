#!/bin/bash
# Real-time monitor for Lagrangian RL results
# Run: watch -n 10 scripts/watch_lagrangian.sh

clear
echo "========================================================================================================"
echo "LAGRANGIAN RL PROGRESS MONITOR - Updated $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================================================"
echo ""

TASKS=("promoter" "human_tf_0" "human_tf_1" "human_tf_2" "human_tf_3" "human_tf_4" "splice_site")
COMPLETED=0
PENDING=0

echo "TASK RESULTS:"
echo ""
printf "%-20s %-12s %-12s %-15s %-12s\n" "Task" "F1" "Accuracy" "Compression" "Status"
echo "------------------------------------------------------------------------------------------------------------"

for task in "${TASKS[@]}"; do
    path="results/rl_runs/grpo_finetuned_env_${task}_lagrangian/eval_test_sampled.json"

    if [ -f "$path" ]; then
        f1=$(python3 -c "import json; m=json.load(open('$path')); print(f\"{m.get('macro_f1', 0):.4f}\")" 2>/dev/null)
        acc=$(python3 -c "import json; m=json.load(open('$path')); print(f\"{m.get('accuracy', 0):.4f}\")" 2>/dev/null)
        comp=$(python3 -c "import json; m=json.load(open('$path')); print(f\"{m.get('compression_ratio', 0):.4f}\")" 2>/dev/null)

        printf "%-20s %-12s %-12s %-15s ✅ DONE\n" "$task" "$f1" "$acc" "$comp"
        ((COMPLETED++))
    else
        printf "%-20s %-12s %-12s %-15s ⏳ Training...\n" "$task" "—" "—" "—"
        ((PENDING++))
    fi
done

echo ""
echo "========================================================================================================"
echo "PROGRESS: $COMPLETED/7 completed  |  $PENDING/7 pending"
echo "========================================================================================================"

# Show job status
echo ""
echo "SLURM JOB STATUS:"
squeue -j 34526906 -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %N" 2>/dev/null || echo "Job not found in queue (may be completed)"

echo ""
echo "To stop monitoring: Press Ctrl+C"
echo "To auto-refresh every 10 seconds: watch -n 10 scripts/watch_lagrangian.sh"
