"""Batch-level prototype rollout test for the Token-Gating Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.agent_token_stats import summarize_agent_tokens
from src.models.build_agent import build_token_gating_agent
from src.models.token_gating_agent import (
    encode_dna_batch,
    masks_to_token_lists,
    sample_boundary_mask,
    threshold_boundary_mask,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    path = config["dataset"][f"{args.split}_path"]
    dataset = GUEPromoterDataset(path)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(loader))
    sequences = batch["sequences"]
    input_ids, attention_mask = encode_dna_batch(sequences)
    agent = build_token_gating_agent(config)

    with torch.no_grad():
        logits, probs = agent(input_ids, attention_mask)

    sampled_masks, log_probs = sample_boundary_mask(probs, attention_mask)
    sampled_tokens = masks_to_token_lists(sequences, sampled_masks)
    threshold = config["evaluation"].get("threshold", 0.5)
    threshold_masks = threshold_boundary_mask(probs, attention_mask, threshold)
    threshold_tokens = masks_to_token_lists(sequences, threshold_masks)

    for sequence, tokens in zip(sequences, sampled_tokens):
        assert "".join(tokens) == sequence
    for sequence, tokens in zip(sequences, threshold_tokens):
        assert "".join(tokens) == sequence

    sampled_stats = summarize_agent_tokens(sequences, sampled_tokens)
    threshold_stats = summarize_agent_tokens(sequences, threshold_tokens)

    print("=" * 80)
    print("Prototype Token-Gating Agent Rollout")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Batch size: {len(sequences)}")
    print()
    print("Sampled token stats:")
    print(json.dumps(sampled_stats, indent=2))
    print()
    print("Threshold token stats:")
    print(json.dumps(threshold_stats, indent=2))
    print()
    print("Example tokenizations:")
    for index in range(min(3, len(sequences))):
        print("-" * 80)
        print(f"Example {index}")
        print("Sequence length:", len(sequences[index]))
        print("Sequence prefix:", sequences[index][:80])
        print("Sampled tokens:", sampled_tokens[index][:20])
        print("Threshold tokens:", threshold_tokens[index][:20])

    out_dir = Path(config["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prototype_rollout_stats.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sampled_stats": sampled_stats,
                "threshold_stats": threshold_stats,
            },
            handle,
            indent=2,
        )
    print()
    print(f"Saved rollout stats to {out_path}")


if __name__ == "__main__":
    main()
