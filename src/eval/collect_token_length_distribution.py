"""Collect token-length distributions for the finetuned-env model and fixed baselines.

For each tokenization strategy, records the per-token lengths across all test
sequences and saves a JSON suitable for histogram / violin-plot figures.

Output schema (results/analysis/token_length_distributions.json):
  {
    "<strategy>": {
      "lengths": [int, ...],          # individual token lengths (all seqs, all tokens)
      "avg_token_length": float,
      "avg_num_tokens":   float,
      "compression_ratio": float,
    },
    ...
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment
from src.models.token_gating_agent import threshold_boundary_mask
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.rl.rollout import group_token_states
from src.tokenization.boundary_utils import apply_boundary_mask
from src.utils.config import load_config


def masks_to_lengths(boundary_masks: torch.Tensor, token_mask: torch.Tensor) -> List[List[int]]:
    """Convert boundary masks to per-sequence lists of token lengths."""
    all_lengths: List[List[int]] = []
    for seq_idx in range(boundary_masks.shape[0]):
        valid_len = int(token_mask[seq_idx].sum().item())
        mask = boundary_masks[seq_idx, :valid_len].cpu().tolist()
        # find starts of tokens
        starts = [i for i, v in enumerate(mask) if int(v) == 1]
        lengths: List[int] = []
        for k, start in enumerate(starts):
            end = starts[k + 1] if k + 1 < len(starts) else valid_len
            lengths.append(end - start)
        all_lengths.append(lengths)
    return all_lengths


def collect_learned_lengths(
    agent: torch.nn.Module,
    env: FineTunedDNABERT2Environment,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    all_sequences: List[str],
) -> Dict[str, object]:
    agent.eval()
    env.eval()
    all_lengths: List[int] = []
    total_tokens = 0
    total_seqs = 0
    # Use DNA sequence length (not DNABERT-2 token count) for compression ratio
    total_dna_len = sum(len(s) for s in all_sequences)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Learned (threshold)"):
            sequences = batch["sequences"]
            features = env.encode_sequences(sequences, device)
            token_states = features["token_states"]
            token_mask   = features["token_mask"]
            _, boundary_probs = agent(token_states, token_mask)
            masks = threshold_boundary_mask(boundary_probs, token_mask, threshold=threshold)
            batch_lengths = masks_to_lengths(masks, token_mask)
            for lengths in batch_lengths:
                all_lengths.extend(lengths)
                total_tokens += len(lengths)
                total_seqs += 1

    avg_num_tokens = total_tokens / max(total_seqs, 1)
    avg_dna_len    = total_dna_len / max(total_seqs, 1)
    return {
        "lengths": all_lengths,
        "avg_token_length_dnabert2": sum(all_lengths) / max(len(all_lengths), 1),
        "avg_num_tokens": avg_num_tokens,
        "compression_ratio": avg_num_tokens / max(avg_dna_len, 1e-8),
        "note": "Lengths are in DNABERT-2 token units; compression_ratio = groups / DNA_length.",
    }


def collect_fixed_kmer_lengths(sequences: List[str], k: int) -> Dict[str, object]:
    """Non-overlapping k-mer tokenization lengths."""
    all_lengths: List[int] = []
    total_tokens = 0
    total_seqs = 0
    for seq in sequences:
        n = len(seq)
        num_tokens = (n + k - 1) // k
        for i in range(num_tokens):
            tok_len = min(k, n - i * k)
            all_lengths.append(tok_len)
        total_tokens += num_tokens
        total_seqs += 1
    avg_num_tokens = total_tokens / max(total_seqs, 1)
    avg_seq_len = sum(len(s) for s in sequences) / max(len(sequences), 1)
    return {
        "lengths": all_lengths,
        "avg_token_length": sum(all_lengths) / max(len(all_lengths), 1),
        "avg_num_tokens": avg_num_tokens,
        "compression_ratio": avg_num_tokens / max(avg_seq_len, 1e-8),
    }


def collect_single_nt_lengths(sequences: List[str]) -> Dict[str, object]:
    total_len = sum(len(s) for s in sequences)
    n = total_len
    return {
        "lengths": [1] * n,
        "avg_token_length": 1.0,
        "avg_num_tokens": total_len / max(len(sequences), 1),
        "compression_ratio": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_finetuned_env.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/rl_runs/grpo_finetuned_env_seed_42/agent_env_last.pt",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.11)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/analysis/token_length_distributions.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", load_config(args.config))

    env_config = config["env"]
    env = FineTunedDNABERT2Environment(
        backbone_name=env_config["backbone_name"],
        checkpoint_path=env_config["checkpoint_path"],
        hidden_size=env_config["hidden_size"],
        num_labels=env_config["num_labels"],
        max_length=env_config.get("max_length", 512),
        cls_mix_weight=env_config.get("cls_mix_weight", 0.0),
        trainable_encoder_layers=env_config.get("trainable_encoder_layers", 4),
        freeze_embeddings=env_config.get("freeze_embeddings", True),
    ).to(device)
    env.load_state_dict(checkpoint["env_state_dict"])

    agent_config = config["agent"]
    agent = TokenStateBoundaryAgent(
        input_dim=agent_config["input_dim"],
        model_dim=agent_config.get("model_dim", 256),
        hidden_dim=agent_config.get("hidden_dim", 128),
        num_layers=agent_config.get("num_layers", 1),
        dropout=agent_config.get("dropout", 0.1),
        initial_boundary_prob=agent_config.get("initial_boundary_prob", 0.2),
    ).to(device)
    agent.load_state_dict(checkpoint["agent_state_dict"])

    dataset = GUEPromoterDataset(config["dataset"][f"{args.split}_path"])
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    all_sequences = [item["sequence"] for item in dataset.items]

    print(f"Device: {device} | Split: {args.split} | N={len(dataset)}")

    results: Dict[str, object] = {}

    print("Collecting: Single nucleotide")
    results["Single nt"] = collect_single_nt_lengths(all_sequences)

    print("Collecting: Fixed 6-mer")
    results["Fixed 6-mer"] = collect_fixed_kmer_lengths(all_sequences, k=6)

    print("Collecting: Learned (finetuned RL, threshold=%.2f)" % args.threshold)
    results["Learned RL"] = collect_learned_lengths(agent, env, dataloader, device, args.threshold, all_sequences)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nSummary:")
    for name, stats in results.items():
        avg_len_key = "avg_token_length_dnabert2" if "avg_token_length_dnabert2" in stats else "avg_token_length"
        print(
            f"  {name:20s}  avg_len={stats[avg_len_key]:.2f}  "
            f"avg_tokens={stats['avg_num_tokens']:.1f}  "
            f"compression={stats['compression_ratio']:.3f}"
        )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
