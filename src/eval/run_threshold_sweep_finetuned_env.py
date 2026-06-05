"""Sweep boundary decision thresholds for a fine-tuned-env checkpoint.

Evaluates the same trained model at multiple hard-threshold values and saves
one JSON per threshold plus a summary JSON, giving a dense compression–performance
tradeoff curve without any additional training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.evaluate_finetuned_env import evaluate_checkpoint
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.utils.config import load_config

THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_finetuned_env.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/rl_runs/grpo_finetuned_env/agent_env_last.pt",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

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

    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 8)
    dataset = GUEPromoterDataset(config["dataset"][f"{args.split}_path"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    default_output_dir = Path(
        config.get("training", {}).get("output_dir", "results/rl_runs/grpo_finetuned_env")
    ) / "threshold_sweep"
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}, Dataset size: {len(dataset)}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Output dir: {output_dir}")

    summary = []
    for threshold in THRESHOLDS:
        print(f"\n--- threshold={threshold:.2f} ---")
        metrics = evaluate_checkpoint(
            agent=agent,
            env=env,
            dataloader=dataloader,
            device=device,
            mode="threshold",
            threshold=threshold,
        )
        metrics["threshold"] = threshold
        print(json.dumps(metrics, indent=2))

        per_threshold_path = output_dir / f"threshold_{threshold:.2f}.json".replace(".", "p")
        with per_threshold_path.open("w") as fh:
            json.dump(metrics, fh, indent=2)

        summary.append(metrics)

    summary_path = output_dir / "sweep_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved {len(summary)} threshold results to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
