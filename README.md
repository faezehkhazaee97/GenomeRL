# GenomeRL: Reinforcement Learning for Adaptive DNA Tokenization

**GenomeRL** frames adaptive DNA tokenization as a reinforcement learning problem. A boundary policy learns where to place token boundaries along a nucleotide sequence, trading off downstream classification performance against compression. The system is evaluated against strong non-RL baselines (fixed-stride pooling, Gumbel-Softmax segmentation, in-encoder token merging) across seven GUE benchmark tasks.

> **Key finding:** In co-adapted settings, compression itself — not learned boundary placement — is the primary driver of performance. The RL approach's distinctive value is biological interpretability: learned boundaries are specifically enriched at TATA-box motifs (5.5×, p<0.001), a pattern absent from fixed-stride pooling.

---

## Method

A BiLSTM boundary agent reads DNABERT-2 BPE token hidden states and predicts per-position boundary probabilities. Consecutive token states between boundaries are merged via mean-pooling to form segments, which are passed to a fine-tuned classifier. The policy is optimized with **GRPO** (group-relative policy optimization) using a reward that combines classification loss and a compression penalty:

```
R = -L_cls - β · C(b)
```

where `C(b) = T(b) / L` is the compression ratio (segments / BPE tokens). A **Lagrangian constrained** variant enforces hard compression budgets via dual ascent.

---

## Repository Structure

```
GenomeRL/
├── src/
│   ├── data/               # Dataset loading (GUE JSONL format)
│   ├── models/             # Model architectures
│   │   ├── backbone.py                       # Frozen DNABERT-2 classifier
│   │   ├── finetuned_dnabert2_env.py         # Fine-tunable DNABERT-2 environment
│   │   ├── token_state_boundary_agent.py     # BiLSTM boundary policy
│   │   ├── transformer_boundary_agent.py     # Transformer [CLS] boundary policy
│   │   ├── attention_pooling_env.py          # Attention-weighted segment pooling
│   │   └── segment_aware_env.py              # Segment-aware positional encodings
│   ├── rl/                 # Training scripts
│   │   ├── grpo.py                           # GRPO advantage + policy loss
│   │   ├── train_agent_finetuned_env.py      # Main fine-tuned RL training
│   │   ├── train_agent_lagrangian.py         # Lagrangian constrained RL
│   │   ├── train_agent_curriculum_splice.py  # Curriculum RL for splice-site
│   │   ├── train_agent_transformer_policy.py # Transformer policy training
│   │   └── train_agent_splice_motif_reward.py# GT/AG proximity reward
│   ├── eval/               # Evaluation and analysis
│   │   ├── evaluate_splice_perclass.py       # Per-class splice-site diagnostics
│   │   └── evaluate_token_merging.py         # In-encoder token merging baseline
│   ├── tokenization/       # Tokenization strategies (1-nt, k-mer, BPE, learned)
│   └── utils/              # Config loading, seeding
├── configs/                # YAML experiment configurations
│   ├── grpo_finetuned_env.yaml               # Main RL config (promoter)
│   ├── grpo_transformer_policy_promoter.yaml
│   ├── grpo_curriculum_splice.yaml
│   └── generated/                            # Per-task auto-generated configs
├── scripts/
│   └── slurm/              # SLURM sbatch scripts for HPC
└── paper/                  # LaTeX source (main.tex, refs.bib)
```

---

## Setup

```bash
git clone https://github.com/faezehkhazaee97/GenomeRL.git
cd GenomeRL
pip install -r requirements.txt

# Download and preprocess GUE benchmark data
python -m src.data.download_gue
python -m src.data.preprocess_gue
```

**Requirements:** Python 3.10+, PyTorch 2.2+, Transformers 4.38+, CUDA GPU recommended.

---

## Running Experiments

### Fine-tuned RL (main experiment)
```bash
python -m src.rl.train_agent_finetuned_env \
    --config configs/grpo_finetuned_env.yaml
```

### Lagrangian constrained RL
```bash
python -m src.rl.train_agent_lagrangian \
    --config configs/generated/gue_promoter/grpo_finetuned_env_lagrangian.yaml
```

### Transformer boundary policy
```bash
python -m src.rl.train_agent_transformer_policy \
    --config configs/grpo_transformer_policy_promoter.yaml
```

### Curriculum RL for splice-site
```bash
python -m src.rl.train_agent_curriculum_splice \
    --config configs/grpo_curriculum_splice.yaml
```

### SLURM (HPC cluster)
```bash
sbatch scripts/slurm/train_agent_finetuned_env_compat_a100.sbatch
```

---

## Baselines

| Method | Description |
|--------|-------------|
| Fine-tuned DNABERT-2 | Full fine-tuning, no compression |
| Fixed-stride pooling | Mean-pool every s=4 BPE tokens |
| Random grouping | Random boundaries at matched compression |
| Gumbel-Softmax | Differentiable segmentation (straight-through gradients) |
| In-encoder token merging | ToMe-style bipartite similarity merging on final hidden states |
| RL frozen backbone | Policy against frozen DNABERT-2 |
| RL fine-tuned | Policy with last 2 DNABERT-2 layers unfrozen |
| Lagrangian RL | Hard compression budget via dual ascent |

---

## Key Results

**Promoter detection** (300 bp, 5,920 test sequences):

| Method | F1 | Compression |
|--------|-----|-------------|
| Fine-tuned DNABERT-2 | 0.939 | 1.00 |
| Stride pooling (s=4) | 0.920 | 0.21 |
| Fine-tuned RL | 0.886 ± 0.007 | 0.18 |
| In-encoder token merging | 0.869 | 0.24 |
| Transformer policy RL | 0.805 ± 0.000 | — |
| Frozen RL | 0.739 | 0.07 |

**Splice-site** (3-class, 400 bp): Fine-tuned RL collapses to predicting only the majority class (macro-F1=0.241 vs baseline 0.924). Per-class analysis confirms recall=0 for no-splice and acceptor classes — five reward-shaping variants (curriculum, asymmetric recall, GT/AG proximity) all reproduce the same collapse, confirming a representational bottleneck.

**Biological interpretability:** RL boundaries are enriched 5.5× at TATA-box motifs (p<0.001, Cohen's h=1.04, 95% CI [5.1, 5.9]), a pattern absent from stride pooling or token merging.

---

## Citation

```bibtex
@article{khazaee2025genomerl,
  title={{GenomeRL}: Reinforcement Learning for Adaptive DNA Tokenization
         in Genomic Sequence Classification},
  author={Khazaee, Baran},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
