"""Compute position-wise average boundary probability for the finetuned-env agent.

Sweeps the full test set and records, at each sequence position, the mean
boundary probability and hard boundary rate (at the given threshold). Outputs
a JSON suitable for plotting a position-wise heatmap / profile.

Usage:
  python -m src.eval.analyze_positionwise_boundaries \\
      --config configs/grpo_finetuned_env.yaml \\
      --checkpoint results/rl_runs/grpo_finetuned_env_seed_42/agent_env_last.pt \\
      --split test \\
      --threshold 0.11 \\
      --output-path results/analysis/positionwise_finetuned_env.json
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
from src.utils.config import load_config


def _empty_profile(length: int) -> Dict[str, List[float]]:
    return {
        "prob_sum": [0.0] * length,
        "rate_sum": [0.0] * length,
        "count":    [0.0] * length,
    }


def _finalize(profile: Dict[str, List[float]]) -> Dict[str, List[float]]:
    boundary_probability: List[float] = []
    boundary_rate: List[float] = []
    for prob_sum, rate_sum, count in zip(
        profile["prob_sum"], profile["rate_sum"], profile["count"]
    ):
        if count > 0:
            boundary_probability.append(prob_sum / count)
            boundary_rate.append(rate_sum / count)
        else:
            boundary_probability.append(0.0)
            boundary_rate.append(0.0)
    return {
        "boundary_probability": boundary_probability,
        "boundary_rate": boundary_rate,
        "counts": profile["count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_finetuned_env.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/rl_runs/grpo_finetuned_env_seed_42/agent_env_last.pt",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--threshold", type=float, default=0.11)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/analysis/positionwise_finetuned_env.json",
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
    env.eval()

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
    agent.eval()

    dataset = GUEPromoterDataset(config["dataset"][f"{args.split}_path"])
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Max length in DNABERT-2 token space — determined during first pass.
    # We use 512 as a safe upper bound (model max_length) and trim after.
    MAX_DNABERT_TOKENS = 512

    profiles: Dict[str, Dict] = {
        "all":      _empty_profile(MAX_DNABERT_TOKENS),
        "positive": _empty_profile(MAX_DNABERT_TOKENS),
        "negative": _empty_profile(MAX_DNABERT_TOKENS),
    }

    max_observed_len = 0

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}, Dataset size: {len(dataset)}")
    print(f"Threshold: {args.threshold}")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Position-wise analysis"):
            sequences = batch["sequences"]
            labels = batch["labels"].tolist()

            features = env.encode_sequences(sequences, device)
            token_states = features["token_states"]
            token_mask   = features["token_mask"]

            _, boundary_probs = agent(token_states, token_mask)
            masks = threshold_boundary_mask(boundary_probs, token_mask, threshold=args.threshold)

            for idx, label in enumerate(labels):
                # valid_len is the number of DNABERT-2 tokens for this sequence
                valid_len = int(token_mask[idx].sum().item())
                max_observed_len = max(max_observed_len, valid_len)

                probs = boundary_probs[idx, :valid_len].cpu().tolist()
                hard  = masks[idx, :valid_len].cpu().tolist()
                label_key = "positive" if int(label) == 1 else "negative"

                for pos, (prob, h) in enumerate(zip(probs, hard)):
                    for key in ("all", label_key):
                        profiles[key]["prob_sum"][pos] += prob
                        profiles[key]["rate_sum"][pos] += float(h)
                        profiles[key]["count"][pos]    += 1.0

    # Trim profiles to the max observed DNABERT-2 token length
    for key in profiles:
        for field in ("prob_sum", "rate_sum", "count"):
            profiles[key][field] = profiles[key][field][:max_observed_len]

    result = {
        "split": args.split,
        "threshold": args.threshold,
        "max_dnabert2_tokens": max_observed_len,
        "note": "Positions are DNABERT-2 token indices (~63 per 300bp sequence), not nucleotide positions.",
        "total_sequences": len(dataset),
        "positionwise_profiles": {k: _finalize(v) for k, v in profiles.items()},
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"Saved position-wise profile to {output_path}")


if __name__ == "__main__":
    main()
