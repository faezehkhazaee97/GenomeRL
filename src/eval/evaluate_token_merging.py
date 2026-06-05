"""Evaluate in-encoder token merging on DNABERT-2 fine-tuned environments.

Token merging (ToMe-style) is applied at the final hidden state level:
  1. Run all 12 DNABERT-2 layers to get token hidden states.
  2. Merge tokens by cosine similarity (most similar pairs averaged).
  3. Mean-pool merged tokens; classify with the fine-tuned head.

This is the correct in-encoder pruning baseline that the reviewer requested.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment


def bipartite_merge(
    token_states: torch.Tensor,
    token_mask: torch.Tensor,
    target_compression: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge tokens iteratively using ToMe-style bipartite matching.

    Most similar token pairs (by cosine similarity) are merged by averaging
    until the compression ratio reaches the target.

    Args:
        token_states: (batch, seq_len, hidden_dim)
        token_mask:   (batch, seq_len) float mask (1=valid, 0=pad)
        target_compression: fraction of original valid tokens to keep

    Returns:
        merged_states: (batch, max_merged_len, hidden_dim) zero-padded
        merged_mask:   (batch, max_merged_len) float mask
    """
    batch_size, seq_len, hidden_dim = token_states.shape
    device = token_states.device

    all_states: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []

    for b in range(batch_size):
        mask_b = token_mask[b].bool()
        valid = token_states[b][mask_b]  # (n_valid, hidden_dim)
        n_valid = valid.size(0)
        target_n = max(1, int(n_valid * target_compression))

        states = valid.clone()

        while states.size(0) > target_n:
            n = states.size(0)
            if n <= 1:
                break
            # Split into set A (even indices) and B (odd indices)
            n_a = (n + 1) // 2
            n_b = n // 2
            a = states[:n_a]
            b_set = states[n_a:]

            if b_set.size(0) == 0:
                break

            # Cosine similarity between every a token and every b token
            a_norm = F.normalize(a, dim=-1)        # (n_a, d)
            b_norm = F.normalize(b_set, dim=-1)    # (n_b, d)
            sim = a_norm @ b_norm.T                # (n_a, n_b)

            # Each a merges with its most similar b
            best_b_idx = sim.argmax(dim=-1)        # (n_a,)
            best_b_states = b_set[best_b_idx]      # (n_a, d)

            # Determine which b tokens are matched
            matched_b = torch.zeros(n_b, dtype=torch.bool, device=device)
            matched_b[best_b_idx.unique()] = True

            # Merge: average matched b into its a
            merged = (a + best_b_states) / 2.0

            # Keep unmatched b tokens as-is
            unmatched = b_set[~matched_b]
            states = torch.cat([merged, unmatched], dim=0)

        all_states.append(states)
        all_masks.append(torch.ones(states.size(0), device=device))

    max_len = max(s.size(0) for s in all_states)
    out_states = torch.zeros(batch_size, max_len, hidden_dim, device=device)
    out_mask = torch.zeros(batch_size, max_len, device=device)
    for b, (s, m) in enumerate(zip(all_states, all_masks)):
        n = s.size(0)
        out_states[b, :n] = s
        out_mask[b, :n] = m

    return out_states, out_mask


def evaluate_token_merging(
    env: FineTunedDNABERT2Environment,
    dataloader,
    device: torch.device,
    compression: float,
) -> Dict[str, float]:
    """Run evaluation with token merging at a given compression ratio."""
    env.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    total_original_tokens = 0
    total_merged_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Compression={compression:.2f}", leave=False):
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)

            features = env.encode_sequences(sequences, device)
            token_states = features["token_states"]
            token_mask = features["token_mask"]
            cls_states = features["cls_states"]

            original_counts = token_mask.sum(dim=1)
            total_original_tokens += original_counts.sum().item()

            if compression >= 1.0:
                merged_states = token_states
                merged_mask = token_mask
            else:
                merged_states, merged_mask = bipartite_merge(
                    token_states, token_mask, target_compression=compression
                )

            merged_counts = merged_mask.sum(dim=1)
            total_merged_tokens += merged_counts.sum().item()

            logits = env.logits_from_segments(cls_states, merged_states, merged_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    achieved_compression = total_merged_tokens / max(1, total_original_tokens)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "target_compression": compression,
        "achieved_compression": round(achieved_compression, 4),
        "n_samples": len(all_labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="promoter",
                        choices=["promoter", "splice_site"])
    parser.add_argument("--env-checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    task_configs = {
        "promoter": {
            "test_path": "data/processed/gue_promoter/test.jsonl",
            "baseline_checkpoint": "results/baselines/dnabert2_bpe_finetuned/best_model.pt",
            "rl_checkpoint": "results/rl_runs/grpo_finetuned_env_seed_42/agent_env_last.pt",
        },
        "splice_site": {
            "test_path": "data/processed/gue_splice_site/test.jsonl",
            "baseline_checkpoint": "results/baselines/dnabert2_bpe_finetuned_splice_site/best_model.pt",
            "rl_checkpoint": "results/rl_runs/grpo_finetuned_env_splice_site_seed_2026/agent_env_last.pt",
        },
    }

    config = task_configs[args.task]
    checkpoint_path = args.env_checkpoint or config["rl_checkpoint"]

    print(f"Task: {args.task}")
    print(f"Loading env checkpoint: {checkpoint_path}")

    env = FineTunedDNABERT2Environment(
        backbone_name="zhihan1996/DNABERT-2-117M",
        checkpoint_path="results/baselines/dnabert2_bpe/best_model.pt",
        hidden_size=768,
        num_labels=2,
        max_length=512,
        cls_mix_weight=0.0,
        trainable_encoder_layers=2,
        freeze_embeddings=True,
    ).to(device)

    # Load the RL-fine-tuned env weights (backbone last 2 layers + classifier)
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "env_state_dict" in raw:
        ckpt = raw["env_state_dict"]
    elif isinstance(raw, dict) and "env" in raw:
        ckpt = raw["env"]
    elif isinstance(raw, dict) and all(k.startswith("env.") for k in list(raw.keys())[:3]):
        ckpt = {k[4:]: v for k, v in raw.items()}
    else:
        ckpt = raw

    env_state = env.state_dict()
    compatible = {k: v for k, v in ckpt.items()
                  if k in env_state and env_state[k].shape == v.shape}
    missing = env.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(env_state)} keys from checkpoint")

    dataset = GUEPromoterDataset(config["test_path"])
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    compressions = [1.0, 0.50, 0.30, 0.25, 0.20, 0.15]
    results = []

    print(f"\n{'Compression':>12} {'F1':>8} {'Accuracy':>10} {'Achieved':>10}")
    print("-" * 45)

    for c in compressions:
        metrics = evaluate_token_merging(env, dataloader, device, c)
        results.append({"task": args.task, **metrics})
        print(f"{c:>12.2f} {metrics['f1']:>8.4f} {metrics['accuracy']:>10.4f} {metrics['achieved_compression']:>10.4f}")

    output_path = args.output or f"results/token_merging/token_merging_{args.task}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
