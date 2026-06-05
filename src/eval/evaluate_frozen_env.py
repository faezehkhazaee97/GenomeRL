"""Evaluate a frozen-environment GenomeRL boundary-policy checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.metrics import compute_classification_metrics
from src.models.frozen_dnabert2_env import FrozenDNABERT2Environment
from src.models.token_gating_agent import sample_boundary_mask, threshold_boundary_mask
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.rl.rollout import group_token_states
from src.utils.config import load_config


def maybe_limit_batches(dataloader, max_batches: int | None):
    if max_batches is None:
        for batch in dataloader:
            yield batch
        return
    count = 0
    for batch in dataloader:
        if count >= max_batches:
            break
        yield batch
        count += 1


def evaluate_checkpoint(
    agent: torch.nn.Module,
    env: FrozenDNABERT2Environment,
    dataloader,
    device: torch.device,
    mode: str,
    threshold: float,
    random_boundary_prob: float = 0.2,
    seed: int = 42,
    max_batches: int | None = None,
) -> Dict[str, float]:
    agent.eval()
    env.eval()

    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_num_segments: List[float] = []
    all_token_counts: List[float] = []

    with torch.no_grad():
        rng = random.Random(seed)
        for batch in tqdm(maybe_limit_batches(dataloader, max_batches), desc="Evaluating"):
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)

            features = env.encode_sequences(sequences, device)
            cls_states = features["cls_states"]
            token_states = features["token_states"]
            token_mask = features["token_mask"]

            _, boundary_probs = agent(token_states, token_mask)
            if mode == "sampled":
                masks, _ = sample_boundary_mask(boundary_probs, token_mask)
            elif mode == "threshold":
                masks = threshold_boundary_mask(boundary_probs, token_mask, threshold=threshold)
            elif mode == "random":
                masks = torch.zeros_like(token_mask, dtype=torch.long)
                for batch_index in range(token_mask.shape[0]):
                    valid_len = int(token_mask[batch_index].sum().item())
                    if valid_len <= 0:
                        continue
                    mask = [1]
                    for _ in range(1, valid_len):
                        mask.append(1 if rng.random() < random_boundary_prob else 0)
                    masks[batch_index, :valid_len] = torch.tensor(mask, dtype=torch.long, device=device)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            grouped = group_token_states(token_states, masks, token_mask)
            logits = env.logits_from_segments(
                cls_states=cls_states,
                segment_states=grouped["segment_states"],
                segment_mask=grouped["segment_mask"],
            )
            predictions = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_num_segments.extend(grouped["num_segments"].cpu().tolist())
            all_token_counts.extend(features["token_counts"].cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_predictions)
    avg_tokens = sum(all_num_segments) / len(all_num_segments)
    avg_sequence_length = sum(all_token_counts) / len(all_token_counts)
    metrics.update(
        {
            "avg_tokens": float(avg_tokens),
            "avg_sequence_length": float(avg_sequence_length),
            "compression_ratio": float(avg_tokens / max(avg_sequence_length, 1e-8)),
            "avg_token_length": float(avg_sequence_length / max(avg_tokens, 1e-8)),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_frozen_env.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/rl_runs/grpo_frozen_env/agent_last.pt")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--mode", type=str, default="sampled", choices=["sampled", "threshold", "random"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-boundary-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", load_config(args.config))

    env_config = config["env"]
    env = FrozenDNABERT2Environment(
        backbone_name=env_config["backbone_name"],
        checkpoint_path=env_config["checkpoint_path"],
        hidden_size=env_config["hidden_size"],
        num_labels=env_config["num_labels"],
        max_length=env_config.get("max_length", 512),
        cls_mix_weight=env_config.get("cls_mix_weight", 0.5),
    ).to(device)

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

    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 8)
    dataset = GUEPromoterDataset(config["dataset"][f"{args.split}_path"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Mode: {args.mode}")
    print(f"Dataset size: {len(dataset)}")

    metrics = evaluate_checkpoint(
        agent=agent,
        env=env,
        dataloader=dataloader,
        device=device,
        mode=args.mode,
        threshold=args.threshold,
        random_boundary_prob=args.random_boundary_prob,
        seed=args.seed,
        max_batches=args.max_batches,
    )

    print("=" * 80)
    print("Frozen-env GenomeRL evaluation metrics")
    print(json.dumps(metrics, indent=2))

    output_dir = Path(config.get("training", {}).get("output_dir", "results/rl_runs/grpo_frozen_env"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_path) if args.output_path else output_dir / f"eval_{args.split}_{args.mode}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
