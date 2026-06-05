"""Smoke test for the learned-token classifier on real GUE sequences."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.build_agent import build_token_gating_agent
from src.models.build_classifier import build_learned_token_classifier
from src.models.learned_token_classifier import encode_token_lists
from src.models.token_gating_agent import (
    encode_dna_batch,
    masks_to_token_lists,
    sample_boundary_mask,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    config = load_config("configs/grpo.yaml")
    set_seed(config["seed"])

    dataset = GUEPromoterDataset(config["dataset"]["train_path"])
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    sequences = batch["sequences"]
    labels = batch["labels"]

    input_ids, attention_mask = encode_dna_batch(sequences)
    agent = build_token_gating_agent(config)
    classifier = build_learned_token_classifier(config)

    with torch.no_grad():
        _, probs = agent(input_ids, attention_mask)
    sampled_masks, _ = sample_boundary_mask(probs, attention_mask)
    token_lists = masks_to_token_lists(sequences, sampled_masks)

    for sequence, tokens in zip(sequences, token_lists):
        assert "".join(tokens) == sequence

    token_ids, token_mask, base_mask = encode_token_lists(token_lists)
    logits = classifier(token_ids, token_mask, base_mask)
    loss = F.cross_entropy(logits, labels)
    predictions = logits.argmax(dim=-1)

    print("Learned-token classifier smoke test")
    print("=" * 80)
    print("Batch size:", len(sequences))
    print("Logits shape:", tuple(logits.shape))
    print("Cross-entropy loss:", float(loss.item()))
    print("Predictions:", predictions.tolist())
    print("Labels:", labels.tolist())
    print("Token counts:", [len(tokens) for tokens in token_lists])
    print("First example tokens:", token_lists[0][:20])


if __name__ == "__main__":
    main()
