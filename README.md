# GenomeRL

A reinforcement learning framework for learning adaptive DNA tokenization strategies for genomic sequence classification.

## Overview

GenomeRL trains a **boundary policy** — a neural agent that learns, at each nucleotide position in a DNA sequence, whether to start a new token. Instead of using fixed tokenization schemes (single nucleotides, k-mers, or BPE), the policy is optimized end-to-end with **GRPO (Grouped Reward Policy Optimization)** using downstream classification accuracy as the reward signal.

**Research question:** Can a learned tokenizer preserve classification performance while using fewer tokens than standard genomic tokenization schemes?

**Key tasks supported:**
- Human promoter detection (GUE benchmark)
- Splice site detection
- Transcription factor (TF) binding prediction

---

## Repository Structure

```
GenomeRL/
├── src/
│   ├── data/           # Dataset loading and preprocessing
│   ├── models/         # Model architectures (backbone, agent, classifier)
│   ├── rl/             # GRPO algorithm and training loops
│   ├── tokenization/   # Tokenization strategies (1-nt, k-mer, BPE, random, learned)
│   ├── eval/           # Evaluation scripts and analysis tools
│   └── utils/          # Config loading, seeding, logging
├── configs/            # YAML experiment configurations
├── scripts/            # Shell scripts for running experiments (Slurm + local)
├── data/               # Preprocessed GUE datasets (JSONL format)
├── OpenRLHF/           # RL training library (included)
└── requirements.txt
```

---

## Setup

### 1. Create and activate a conda environment

```bash
conda create -n genomerl python=3.10 -y
conda activate genomerl
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The DNABERT-2 model (`zhihan1996/DNABERT-2-117M`) is downloaded automatically from Hugging Face on first use. Make sure you have internet access or the model cached.

### 3. (Optional) Set Hugging Face cache directory

```bash
export HF_HOME=/your/preferred/cache/dir
```

---

## Data

Preprocessed datasets are included in `data/processed/` in JSONL format. Each line is a JSON object with fields:

| Field      | Description                        |
|------------|------------------------------------|
| `id`       | Example ID                         |
| `sequence` | DNA sequence string                |
| `label`    | Integer class label                |
| `split`    | `train`, `valid`, or `test`        |

**Datasets:**
- `data/processed/gue_promoter/` — Human promoter detection (47,356 train / 5,920 val / 5,920 test, 300 bp)
- `data/processed/gue_splice_site/` — Splice site detection
- `data/processed/gue_human_tf_0/` through `gue_human_tf_4/` — TF binding prediction

To re-download and re-preprocess from Hugging Face:

```bash
python -m src.data.download_gue
python -m src.data.preprocess_gue
```

---

## Running Experiments

All commands should be run from the repository root with the conda environment active.

### Tokenizer Comparison (Baselines)

Compare all tokenization strategies (single-nt, k-mer, BPE, random):

```bash
python -m src.eval.compare_tokenizers
# Output: results/baselines/tokenizer_comparison.json
```

### Supervised Baselines

**DNABERT-2 BPE baseline (frozen backbone):**

```bash
python -m src.eval.evaluate_baseline --config configs/baseline.yaml
```

**DNABERT-2 finetuned baseline:**

```bash
python -m src.eval.evaluate_dnabert2_finetuned_baseline --config configs/baseline_finetuned_dnabert2.yaml
```

**CNN baseline:**

```bash
python -m src.eval.evaluate_baseline --config configs/baseline.yaml  # set model.backbone_name to cnn in config
```

### GRPO Agent Training

**Train agent with frozen DNABERT-2 environment:**

```bash
python -m src.rl.train_agent_frozen_env --config configs/grpo_frozen_env.yaml
```

**Train agent with finetuned DNABERT-2 environment:**

```bash
python -m src.rl.train_agent_finetuned_env --config configs/grpo_finetuned_env.yaml
```

**Train on other tasks (splice site, TF binding):**

```bash
python -m src.rl.train_agent_frozen_env --config configs/grpo_frozen_env_splice_site.yaml
python -m src.rl.train_agent_frozen_env --config configs/grpo_frozen_env_tf_binding.yaml
```

### Evaluation

**Evaluate trained agent (frozen environment):**

```bash
python -m src.eval.evaluate_frozen_env --config configs/grpo_frozen_env.yaml
```

**Evaluate trained agent (finetuned environment):**

```bash
python -m src.eval.evaluate_finetuned_env --config configs/grpo_finetuned_env.yaml
```

**Analyze learned boundary interpretability:**

```bash
python -m src.eval.analyze_boundary_interpretability
```

**Export example tokenizations:**

```bash
python -m src.eval.export_boundary_examples
```

---

## Slurm (HPC Cluster)

All submission scripts target the `gpu-a100-h` partition. Before using them, update the paths inside `scripts/slurm/genomerl_a100.sbatch`:

```bash
REPO_DIR="/path/to/your/GenomeRL"
CONDA_BASE="/path/to/your/miniconda3"
ENV_NAME="genomerl"
```

Then submit jobs with:

```bash
# Generic submission
sbatch scripts/slurm/genomerl_a100.sbatch python -m src.rl.train_agent_frozen_env --config configs/grpo_frozen_env.yaml

# Convenience wrappers
bash scripts/submit_baseline_a100.sh
bash scripts/submit_train_agent_frozen_env_compat_a100.sh
bash scripts/submit_evaluate_frozen_env_compat_a100.sh
```

Slurm logs are written to `results/slurm/`.

---

## Configuration

Experiments are controlled by YAML config files in `configs/`. Key fields:

```yaml
project_name: GenomeRL
seed: 42

dataset:
  train_path: data/processed/gue_promoter/train.jsonl
  valid_path: data/processed/gue_promoter/valid.jsonl
  test_path:  data/processed/gue_promoter/test.jsonl

model:
  backbone_name: zhihan1996/DNABERT-2-117M
  freeze_backbone: true
  hidden_size: 768
  num_labels: 2

tokenizer:
  name: dnabert2_bpe   # one of: single_nt, fixed_kmer, random_policy, dnabert2_bpe
  max_length: 512

training:
  batch_size: 16
  learning_rate: 3e-5
  epochs: 5

grpo:                  # Only for RL configs
  group_size: 8
  beta: 0.02
```

Available configs:

| Config file | Description |
|-------------|-------------|
| `baseline.yaml` | DNABERT-2 BPE baseline, frozen backbone |
| `baseline_finetuned_dnabert2.yaml` | DNABERT-2 finetuned baseline |
| `grpo_frozen_env.yaml` | GRPO agent, frozen DNABERT-2 env, promoter |
| `grpo_frozen_env_splice_site.yaml` | GRPO agent, frozen env, splice site |
| `grpo_frozen_env_tf_binding.yaml` | GRPO agent, frozen env, TF binding |
| `grpo_finetuned_env.yaml` | GRPO agent, finetuned DNABERT-2 env, promoter |
| `grpo_finetuned_env_splice_site.yaml` | GRPO agent, finetuned env, splice site |
| `grpo_finetuned_env_tf_binding.yaml` | GRPO agent, finetuned env, TF binding |
| `controlled_ablation.yaml` | Controlled tokenizer ablation study |

---

## Tokenization Strategies

| Strategy | Description | Compression ratio |
|----------|-------------|-------------------|
| `single_nt` | One token per nucleotide | 1.0 (baseline) |
| `fixed_kmer` | Non-overlapping 6-mers | ~0.167 |
| `dnabert2_bpe` | DNABERT-2 BPE tokenizer | ~0.211 |
| `random_policy` | Bernoulli boundary sampling | ~0.193 |
| Learned (GRPO) | RL-trained boundary policy | ~0.19 (adaptive) |

*Compression ratio = num_tokens / sequence_length*

---

## Key Concepts

**Boundary mask:** A binary vector of length equal to the DNA sequence, where `1` marks the start of a new token. The first position is always `1`.

**GRPO training loop:**
1. Sample `group_size` tokenizations from the current policy for each sequence.
2. Score each tokenization using the downstream classifier.
3. Normalize rewards within each group to compute advantages.
4. Update the policy with a clipped policy gradient loss.

**Token-gating agent:** An LSTM-based network that processes the one-hot encoded DNA sequence and outputs boundary probabilities at each position.

---

## Dependencies

- Python 3.10+
- PyTorch
- Hugging Face `transformers` and `datasets`
- scikit-learn
- pandas, pyyaml, tqdm
- DNABERT-2 pretrained model: `zhihan1996/DNABERT-2-117M` (auto-downloaded)
