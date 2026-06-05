"""Export qualitative boundary/tokenization examples from the DNA-level policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.build_agent import build_token_gating_agent
from src.models.token_gating_agent import (
    encode_dna_batch,
    masks_to_token_lists,
    threshold_boundary_mask,
)
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


def token_spans_from_tokens(tokens: List[str]) -> List[tuple[int, int]]:
    spans = []
    cursor = 0
    for token in tokens:
        spans.append((cursor, cursor + len(token)))
        cursor += len(token)
    return spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/rl_runs/grpo/agent_classifier_last.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--threshold", type=float, default=0.11)
    parser.add_argument("--motif", type=str, default="TATA")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--label-filter", type=str, default="all", choices=["all", "positive", "negative"])
    parser.add_argument("--motif-only", action="store_true")
    parser.add_argument("--motif-absent", action="store_true",
                        help="Only include sequences that do NOT contain the motif.")
    parser.add_argument("--output-path", type=str, default="results/analysis/boundary_examples.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", load_config(args.config))

    agent = build_token_gating_agent(config).to(device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    dataset = GUEPromoterDataset(config["dataset"][f"{args.split}_path"])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    requested_label = None
    if args.label_filter == "positive":
        requested_label = 1
    elif args.label_filter == "negative":
        requested_label = 0

    exported = []
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequences"]
            labels = batch["labels"].tolist()
            ids = batch["ids"]

            input_ids, attention_mask = encode_dna_batch(sequences)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            _, boundary_probs = agent(input_ids, attention_mask)
            masks = threshold_boundary_mask(boundary_probs, attention_mask, threshold=args.threshold)
            token_lists = masks_to_token_lists(sequences, masks)

            for sequence_index, (seq_id, sequence, label, tokens) in enumerate(zip(ids, sequences, labels, token_lists)):
                if requested_label is not None and int(label) != requested_label:
                    continue

                motif_spans = find_motif_spans(sequence, args.motif)
                if args.motif_only and not motif_spans:
                    continue
                if args.motif_absent and motif_spans:
                    continue

                valid_length = len(sequence)
                probs = boundary_probs[sequence_index, :valid_length].detach().cpu().tolist()
                hard_mask = masks[sequence_index, :valid_length].detach().cpu().tolist()
                exported.append(
                    {
                        "id": seq_id,
                        "label": int(label),
                        "sequence": sequence,
                        "motif": args.motif,
                        "motif_spans": motif_spans,
                        "threshold": args.threshold,
                        "boundary_probabilities": probs,
                        "hard_boundary_mask": hard_mask,
                        "tokens": tokens,
                        "token_spans": token_spans_from_tokens(tokens),
                    }
                )
                if len(exported) >= args.num_examples:
                    output_path = Path(args.output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(json.dumps(exported, indent=2), encoding="utf-8")
                    print(json.dumps(exported, indent=2))
                    print(f"Saved boundary examples to {output_path}")
                    return

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(exported, indent=2), encoding="utf-8")
    print(json.dumps(exported, indent=2))
    print(f"Saved boundary examples to {output_path}")


if __name__ == "__main__":
    main()
