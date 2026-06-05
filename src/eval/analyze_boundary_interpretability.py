"""Analyze DNA-level learned boundary behavior around promoter motifs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.build_agent import build_token_gating_agent
from src.models.token_gating_agent import encode_dna_batch, threshold_boundary_mask
from src.utils.config import load_config


def find_motif_spans(sequence: str, motif: str) -> List[tuple[int, int]]:
    spans = []
    start = 0
    while True:
        index = sequence.find(motif, start)
        if index == -1:
            break
        spans.append((index, index + len(motif)))
        start = index + 1
    return spans


def summarize(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def initialize_label_stats() -> Dict[str, List[float] | int]:
    return {
        "motif_probs": [],
        "non_motif_probs": [],
        "motif_boundary_rates": [],
        "non_motif_boundary_rates": [],
        "core_probs": [],
        "non_core_probs": [],
        "core_token_lengths": [],
        "non_core_token_lengths": [],
        "motif_sequence_count": 0,
        "total_sequences": 0,
    }


def initialize_profile(max_length: int) -> Dict[str, List[float]]:
    return {
        "prob_sum": [0.0] * max_length,
        "rate_sum": [0.0] * max_length,
        "count": [0.0] * max_length,
    }


def mask_to_token_spans(mask: List[int], valid_length: int) -> List[tuple[int, int]]:
    starts = [index for index, value in enumerate(mask[:valid_length]) if int(value) == 1]
    if not starts:
        return [(0, valid_length)]

    spans = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else valid_length
        spans.append((start, end))
    return spans


def update_label_statistics(
    stats: Dict[str, List[float] | int],
    sequence: str,
    probs: List[float],
    hard: List[int],
    motif: str,
    core_start: int,
    core_end: int,
) -> None:
    stats["total_sequences"] += 1

    spans = find_motif_spans(sequence, motif)
    motif_positions = set()
    for start, end in spans:
        motif_positions.update(range(start, end))
    if spans:
        stats["motif_sequence_count"] += 1

    for position, prob in enumerate(probs):
        if position in motif_positions:
            stats["motif_probs"].append(prob)
            stats["motif_boundary_rates"].append(float(hard[position]))
        else:
            stats["non_motif_probs"].append(prob)
            stats["non_motif_boundary_rates"].append(float(hard[position]))

        if core_start <= position < core_end:
            stats["core_probs"].append(prob)
        else:
            stats["non_core_probs"].append(prob)

    token_spans = mask_to_token_spans(hard, len(sequence))
    for start, end in token_spans:
        token_length = float(end - start)
        for position in range(start, end):
            if core_start <= position < core_end:
                stats["core_token_lengths"].append(token_length)
            else:
                stats["non_core_token_lengths"].append(token_length)


def finalize_label_statistics(stats: Dict[str, List[float] | int]) -> Dict[str, float]:
    total_sequences = int(stats["total_sequences"])
    motif_sequence_count = int(stats["motif_sequence_count"])
    return {
        "total_sequences": total_sequences,
        "motif_sequence_count": motif_sequence_count,
        "motif_sequence_fraction": motif_sequence_count / max(total_sequences, 1),
        "avg_boundary_prob_on_motif": summarize(stats["motif_probs"]),
        "avg_boundary_prob_off_motif": summarize(stats["non_motif_probs"]),
        "avg_boundary_rate_on_motif": summarize(stats["motif_boundary_rates"]),
        "avg_boundary_rate_off_motif": summarize(stats["non_motif_boundary_rates"]),
        "avg_boundary_prob_core_promoter": summarize(stats["core_probs"]),
        "avg_boundary_prob_outside_core": summarize(stats["non_core_probs"]),
        "avg_token_length_core_promoter": summarize(stats["core_token_lengths"]),
        "avg_token_length_outside_core": summarize(stats["non_core_token_lengths"]),
    }


def finalize_profile(profile: Dict[str, List[float]]) -> Dict[str, List[float]]:
    boundary_probability = []
    boundary_rate = []
    counts = []
    for prob_sum, rate_sum, count in zip(profile["prob_sum"], profile["rate_sum"], profile["count"]):
        if count > 0:
            boundary_probability.append(prob_sum / count)
            boundary_rate.append(rate_sum / count)
        else:
            boundary_probability.append(0.0)
            boundary_rate.append(0.0)
        counts.append(count)

    return {
        "boundary_probability": boundary_probability,
        "boundary_rate": boundary_rate,
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/rl_runs/grpo/agent_classifier_last.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--motif", type=str, default="TATA")
    parser.add_argument("--core-start", type=int, default=220)
    parser.add_argument("--core-end", type=int, default=280)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-path", type=str, default="results/analysis/boundary_interpretability.json")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", load_config(args.config))

    agent = build_token_gating_agent(config).to(device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    dataset = GUEPromoterDataset(config["dataset"][f"{args.split}_path"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    max_sequence_length = max(len(item["sequence"]) for item in dataset.items)

    label_stats = {
        "all": initialize_label_stats(),
        "positive": initialize_label_stats(),
        "negative": initialize_label_stats(),
    }
    label_profiles = {
        "all": initialize_profile(max_sequence_length),
        "positive": initialize_profile(max_sequence_length),
        "negative": initialize_profile(max_sequence_length),
    }

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(dataloader, desc="Analyzing boundaries")):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            sequences = batch["sequences"]
            labels = batch["labels"].tolist()
            input_ids, attention_mask = encode_dna_batch(sequences)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            _, boundary_probs = agent(input_ids, attention_mask)
            masks = threshold_boundary_mask(boundary_probs, attention_mask, threshold=args.threshold)

            for sequence_index, (sequence, label) in enumerate(zip(sequences, labels)):
                label_key = "positive" if int(label) == 1 else "negative"
                valid_len = len(sequence)
                probs = boundary_probs[sequence_index, :valid_len].detach().cpu().tolist()
                hard = masks[sequence_index, :valid_len].detach().cpu().tolist()

                update_label_statistics(
                    label_stats["all"],
                    sequence,
                    probs,
                    hard,
                    motif=args.motif,
                    core_start=args.core_start,
                    core_end=args.core_end,
                )
                update_label_statistics(
                    label_stats[label_key],
                    sequence,
                    probs,
                    hard,
                    motif=args.motif,
                    core_start=args.core_start,
                    core_end=args.core_end,
                )

                for position, prob in enumerate(probs):
                    hard_value = float(hard[position])
                    for profile_key in ("all", label_key):
                        label_profiles[profile_key]["prob_sum"][position] += prob
                        label_profiles[profile_key]["rate_sum"][position] += hard_value
                        label_profiles[profile_key]["count"][position] += 1.0

    all_stats = finalize_label_statistics(label_stats["all"])
    result: Dict[str, object] = {
        "split": args.split,
        "motif": args.motif,
        "total_sequences": all_stats["total_sequences"],
        "motif_sequence_count": all_stats["motif_sequence_count"],
        "motif_sequence_fraction": all_stats["motif_sequence_fraction"],
        "avg_boundary_prob_on_motif": all_stats["avg_boundary_prob_on_motif"],
        "avg_boundary_prob_off_motif": all_stats["avg_boundary_prob_off_motif"],
        "avg_boundary_rate_on_motif": all_stats["avg_boundary_rate_on_motif"],
        "avg_boundary_rate_off_motif": all_stats["avg_boundary_rate_off_motif"],
        "avg_boundary_prob_core_promoter": all_stats["avg_boundary_prob_core_promoter"],
        "avg_boundary_prob_outside_core": all_stats["avg_boundary_prob_outside_core"],
        "avg_token_length_core_promoter": all_stats["avg_token_length_core_promoter"],
        "avg_token_length_outside_core": all_stats["avg_token_length_outside_core"],
        "core_window_start": args.core_start,
        "core_window_end": args.core_end,
        "threshold": args.threshold,
        "by_label": {
            "all": all_stats,
            "positive": finalize_label_statistics(label_stats["positive"]),
            "negative": finalize_label_statistics(label_stats["negative"]),
        },
        "positionwise_profiles": {
            "all": finalize_profile(label_profiles["all"]),
            "positive": finalize_profile(label_profiles["positive"]),
            "negative": finalize_profile(label_profiles["negative"]),
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved interpretability analysis to {output_path}")


if __name__ == "__main__":
    main()
