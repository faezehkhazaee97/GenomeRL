#!/bin/bash
#SBATCH --job-name=curriculum-splice
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

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${REPO_DIR}/.hf-cache-compat"
export HF_HUB_DISABLE_TELEMETRY=1

cd "${REPO_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(which python)"
nvidia-smi || true

echo ""
echo "============================================================"
echo "Exp 1: Curriculum RL — Splice site (BiLSTM)"
echo "  β: 0.005 → 0.05, recall_weight=2.0, min_ratio=0.15"
echo "============================================================"
python -m src.rl.train_agent_curriculum_splice \
    --config configs/grpo_curriculum_splice.yaml \
    --seed 42

echo ""
echo "============================================================"
echo "Exp 2: Curriculum RL — Splice site (Transformer policy)"
echo "  β: 0.005 → 0.05, recall_weight=2.0, [CLS] global context"
echo "============================================================"
python -m src.rl.train_agent_curriculum_splice \
    --config configs/grpo_curriculum_splice_transformer.yaml \
    --seed 42 \
    --use-transformer

echo ""
echo "============================================================"
echo "Exp 3: Curriculum RL — Splice site (BiLSTM, seed 1337)"
echo "============================================================"
python -m src.rl.train_agent_curriculum_splice \
    --config configs/grpo_curriculum_splice.yaml \
    --seed 1337 \
    --output-dir results/rl_runs/grpo_curriculum_splice_seed_1337

echo ""
echo "============================================================"
echo "All curriculum splice-site experiments done."
echo "============================================================"
echo ""
echo "Results:"
for dir in results/rl_runs/grpo_curriculum_splice*; do
    echo -n "  $(basename $dir): "
    python3 -c "
import json, pathlib
f = pathlib.Path('$dir/eval_test_sampled.json')
if f.exists():
    d = json.load(open(f))
    print(f'acc={d[\"accuracy\"]:.4f}  f1={d[\"macro_f1\"]:.4f}  comp={d.get(\"compression_ratio\",\"?\"):.4f}')
else:
    print('no result')
" 2>/dev/null || echo "no result"
done

echo ""
echo "=== COMPARISON TO STANDARD RL ==="
echo "Standard RL splice-site (3 seeds): F1=0.241 (all seeds identical)"
echo "Standard RL: severely over-compresses; biased predictions (acc=0.566, F1=0.241)"
