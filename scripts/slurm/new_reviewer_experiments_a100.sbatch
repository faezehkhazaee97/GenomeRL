#!/bin/bash
#SBATCH --job-name=new-reviewer-exps
#SBATCH --partition=gpuh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

REPO_DIR="/group/gquongrp/workspaces/baran/GenomeRL"
CONDA_BASE="/group/gquongrp/workspaces/baran/miniconda3"
ENV_NAME="dnabert2-compat"

mkdir -p "${REPO_DIR}/results/slurm"
mkdir -p "${REPO_DIR}/.hf-cache-compat"
mkdir -p "${REPO_DIR}/results/analysis"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${REPO_DIR}/.hf-cache-compat"
export HF_HUB_DISABLE_TELEMETRY=1

cd "${REPO_DIR}"
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname)"
nvidia-smi || true

echo ""
echo "============================================================"
echo "Exp 1: Per-class splice-site analysis (Q1/Q2)"
echo "  Confusion matrix, per-class P/R/F1, ECE, length strat."
echo "============================================================"
python -m src.eval.evaluate_splice_perclass

echo ""
echo "============================================================"
echo "Exp 2: GT/AG proximity reward — gamma=0.1 (Q5)"
echo "  Curriculum β + asymmetric recall + motif proximity bonus"
echo "============================================================"
python -m src.rl.train_agent_splice_motif_reward \
    --config configs/grpo_curriculum_splice.yaml \
    --gamma 0.1 \
    --window 3 \
    --seed 42 \
    --output-dir results/rl_runs/grpo_splice_motif_gamma0.1

echo ""
echo "============================================================"
echo "Exp 3: GT/AG proximity reward — gamma=0.3 (stronger signal)"
echo "============================================================"
python -m src.rl.train_agent_splice_motif_reward \
    --config configs/grpo_curriculum_splice.yaml \
    --gamma 0.3 \
    --window 5 \
    --seed 42 \
    --output-dir results/rl_runs/grpo_splice_motif_gamma0.3

echo ""
echo "============================================================"
echo "All done. Results:"
echo "============================================================"
echo ""
echo "=== Per-class analysis ==="
cat results/analysis/splice_site_perclass_analysis.json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data:
    print(f\"\n{m['name']}\")
    print(f\"  Macro-F1: {m['macro_f1']}  ECE: {m['ece']}\")
    for cls, stats in m['per_class'].items():
        print(f\"  {cls}: P={stats['precision']} R={stats['recall']} F1={stats['f1-score']}\")
" 2>/dev/null || echo "no analysis result"

echo ""
echo "=== GT/AG reward results ==="
for dir in results/rl_runs/grpo_splice_motif_gamma*; do
    echo -n "  $(basename $dir): "
    python3 -c "
import json
d = json.load(open('$dir/eval_test_sampled.json'))
print(f'F1={d[\"macro_f1\"]:.4f}  acc={d[\"accuracy\"]:.4f}  comp={d[\"compression_ratio\"]:.4f}  gamma={d[\"gamma\"]}')
" 2>/dev/null || echo "no result"
done

echo ""
echo "Standard RL baseline (for comparison): F1=0.241, acc=0.566"
