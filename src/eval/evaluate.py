"""Evaluate a trained GenomeRL checkpoint on validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.agent_token_stats import summarize_agent_tokens
from src.eval.metrics import compute_classification_metrics
from src.models.build_agent import build_token_gating_agent
from src.models.build_classifier import build_learned_token_classifier
from src.models.learned_token_classifier import encode_token_lists
from src.models.token_gating_agent import (
    encode_dna_batch,
    masks_to_token_lists,
    sample_boundary_mask,
    threshold_boundary_mask,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


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
    classifier: torch.nn.Module,
    dataloader,
    device: torch.device,
    mode: str,
    threshold: float,
    max_batches: int | None = None,
) -> Dict[str, float]:
    agent.eval()
    classifier.eval()

    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_sequences: List[str] = []
    all_token_lists: List[List[str]] = []

    with torch.no_grad():
        for batch in tqdm(maybe_limit_batches(dataloader, max_batches), desc="Evaluating"):
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)

            input_ids, attention_mask = encode_dna_batch(sequences)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            _, boundary_probs = agent(input_ids, attention_mask)
            if mode == "sampled":
                masks, _ = sample_boundary_mask(boundary_probs, attention_mask)
            elif mode == "threshold":
                masks = threshold_boundary_mask(boundary_probs, attention_mask, threshold=threshold)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            token_lists = masks_to_token_lists(sequences, masks)
            token_ids, token_mask, base_mask = encode_token_lists(token_lists)
            token_ids = token_ids.to(device)
            token_mask = token_mask.to(device)
            base_mask = base_mask.to(device)

            logits = classifier(token_ids, token_mask, base_mask)
            predictions = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_sequences.extend(sequences)
            all_token_lists.extend(token_lists)

    if not all_sequences:
        raise RuntimeError("Evaluation produced no batches; check dataloader and max_batches settings.")

    metrics = compute_classification_metrics(all_labels, all_predictions)
    token_stats = summarize_agent_tokens(all_sequences, all_token_lists)
    metrics.update(
        {
            "avg_tokens": float(token_stats["avg_num_tokens"]),
            "avg_sequence_length": float(sum(len(seq) for seq in all_sequences) / len(all_sequences)),
            "compression_ratio": float(token_stats["avg_compression_ratio"]),
            "avg_token_length": float(token_stats["avg_token_length"]),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/rl_runs/grpo/agent_classifier_last.pt",
    )
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--mode", type=str, default="sampled", choices=["sampled", "threshold"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)

    agent = build_token_gating_agent(checkpoint_config).to(device)
    classifier = build_learned_token_classifier(checkpoint_config).to(device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])

    batch_size = args.batch_size or checkpoint_config.get("training", {}).get("batch_size", 8)
    dataset_path = checkpoint_config["dataset"][f"{args.split}_path"]
    dataset = GUEPromoterDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    threshold = args.threshold
    if threshold is None:
        threshold = checkpoint_config.get("evaluation", {}).get("threshold", 0.5)

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {batch_size}")
    print(f"Dataset size: {len(dataset)}")

    metrics = evaluate_checkpoint(
        agent=agent,
        classifier=classifier,
        dataloader=dataloader,
        device=device,
        mode=args.mode,
        threshold=threshold,
        max_batches=args.max_batches,
    )

    print("=" * 80)
    print("GenomeRL evaluation metrics")
    print(json.dumps(metrics, indent=2))

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(checkpoint_config.get("training", {}).get("output_dir", "results/rl_runs/grpo"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_{args.split}_{args.mode}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
