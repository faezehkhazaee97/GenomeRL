"""Sweep deterministic thresholds for frozen-env boundary decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.evaluate_frozen_env import evaluate_checkpoint
from src.models.frozen_dnabert2_env import FrozenDNABERT2Environment
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_frozen_env.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/rl_runs/grpo_frozen_env/agent_last.pt")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--select-metric", type=str, default="macro_f1", choices=["accuracy", "macro_f1"])
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", load_config(args.config))

    env_config = config["env"]
    env = FrozenDNABERT2Environment(
        backbone_name=env_config["backbone_name"],
        checkpoint_path=env_config["checkpoint_path"],
        hidden_size=env_config["hidden_size"],
        num_labels=env_config["num_labels"],
        max_length=env_config.get("max_length", 512),
        cls_mix_weight=env_config.get("cls_mix_weight", 0.0),
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

    results = []
    best_result = None
    best_value = None
    for threshold in args.thresholds:
        metrics = evaluate_checkpoint(
            agent=agent,
            env=env,
            dataloader=dataloader,
            device=device,
            mode="threshold",
            threshold=threshold,
            max_batches=args.max_batches,
        )
        record = {"threshold": threshold, **metrics}
        results.append(record)
        value = record[args.select_metric]
        if best_result is None or value > best_value:
            best_result = record
            best_value = value

    payload = {
        "split": args.split,
        "select_metric": args.select_metric,
        "best": best_result,
        "results": results,
    }
    print(json.dumps(payload, indent=2))

    output_dir = Path(config.get("training", {}).get("output_dir", "results/rl_runs/grpo_frozen_env"))
    output_path = Path(args.output_path) if args.output_path else output_dir / f"threshold_sweep_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved threshold sweep to {output_path}")


if __name__ == "__main__":
    main()
