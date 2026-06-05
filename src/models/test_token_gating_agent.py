"""Quick test of the Token-Gating Agent on real GUE sequences."""

from __future__ import annotations

from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.agent_token_stats import summarize_agent_tokens
from src.models.token_gating_agent import (
    TokenGatingAgent,
    encode_dna_batch,
    masks_to_token_lists,
    sample_boundary_mask,
    threshold_boundary_mask,
)


def main() -> None:
    dataset = GUEPromoterDataset("data/processed/gue_promoter/train.jsonl")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(loader))
    sequences = batch["sequences"]
    input_ids, attention_mask = encode_dna_batch(sequences)

    agent = TokenGatingAgent()
    logits, probs = agent(input_ids, attention_mask)
    sampled_masks, log_probs = sample_boundary_mask(probs, attention_mask)
    sampled_tokens = masks_to_token_lists(sequences, sampled_masks)

    threshold_masks = threshold_boundary_mask(probs, attention_mask, threshold=0.5)
    threshold_tokens = masks_to_token_lists(sequences, threshold_masks)

    print("=" * 80)
    print("Sampled tokenization")
    print("=" * 80)
    for sequence, tokens in zip(sequences, sampled_tokens):
        print("Sequence length:", len(sequence))
        print("Number of tokens:", len(tokens))
        print("First 10 tokens:", tokens[:10])
        print()

    sampled_stats = summarize_agent_tokens(sequences, sampled_tokens)
    print("Sampled stats:")
    print(sampled_stats)
    print()

    print("=" * 80)
    print("Threshold tokenization")
    print("=" * 80)
    for sequence, tokens in zip(sequences, threshold_tokens):
        print("Sequence length:", len(sequence))
        print("Number of tokens:", len(tokens))
        print("First 10 tokens:", tokens[:10])
        print()

    threshold_stats = summarize_agent_tokens(sequences, threshold_tokens)
    print("Threshold stats:")
    print(threshold_stats)


if __name__ == "__main__":
    main()
