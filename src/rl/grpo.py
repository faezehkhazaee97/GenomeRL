"""GRPO algorithm components for GenomeRL.

This module implements the grouped sampling and policy-loss utilities used for
training the token-gating agent. The high-level idea is:

1. For each input sequence, sample multiple tokenizations from the boundary
   policy.
2. Score each sampled tokenization with a downstream reward.
3. Normalize rewards within each sequence-specific group.
4. Use the normalized advantages to scale the sampled log-probabilities.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from src.models.token_gating_agent import masks_to_token_lists, sample_boundary_mask


def sample_group_tokenizations(
    sequences: List[str],
    boundary_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    group_size: int,
) -> Dict[str, object]:
    """Sample multiple tokenizations per sequence from the boundary policy.

    Returns a dictionary containing:
    - `masks`: [batch, group_size, seq_len]
    - `log_probs`: [batch, group_size, seq_len]
    - `token_lists`: nested Python list of token sequences
    - `num_tokens`: [batch, group_size]
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    batch_masks = []
    batch_log_probs = []
    batch_token_lists: List[List[List[str]]] = []
    batch_num_tokens = []

    for _ in range(group_size):
        masks, log_probs = sample_boundary_mask(boundary_probs, attention_mask)
        token_lists = masks_to_token_lists(sequences, masks)
        num_tokens = torch.tensor(
            [len(tokens) for tokens in token_lists],
            dtype=torch.float,
            device=boundary_probs.device,
        )

        batch_masks.append(masks)
        batch_log_probs.append(log_probs)
        batch_token_lists.append(token_lists)
        batch_num_tokens.append(num_tokens)

    masks_tensor = torch.stack(batch_masks, dim=1)
    log_probs_tensor = torch.stack(batch_log_probs, dim=1)
    num_tokens_tensor = torch.stack(batch_num_tokens, dim=1)

    grouped_token_lists: List[List[List[str]]] = []
    for batch_index in range(len(sequences)):
        grouped_token_lists.append(
            [batch_token_lists[group_index][batch_index] for group_index in range(group_size)]
        )

    return {
        "masks": masks_tensor,
        "log_probs": log_probs_tensor,
        "token_lists": grouped_token_lists,
        "num_tokens": num_tokens_tensor,
    }


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize rewards within each sequence-specific group.

    Args:
        rewards: Tensor of shape [batch, group_size].
        eps: Numerical stability constant.
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch, group_size]")

    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - group_mean) / (group_std + eps)


def reduce_log_probs(
    log_probs: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum token-boundary log-probabilities over valid sequence positions.

    Args:
        log_probs: Tensor of shape [batch, group_size, seq_len].
        attention_mask: Tensor of shape [batch, seq_len].
    """
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape [batch, group_size, seq_len]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, seq_len]")

    expanded_mask = attention_mask.unsqueeze(1).float()
    return (log_probs * expanded_mask).sum(dim=-1)


def compute_policy_loss(
    log_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Compute the GRPO-style policy loss."""
    reduced_log_probs = reduce_log_probs(log_probs, attention_mask)
    if reduced_log_probs.shape != advantages.shape:
        raise ValueError(
            "Reduced log-probs and advantages must have the same shape, "
            f"got {tuple(reduced_log_probs.shape)} vs {tuple(advantages.shape)}"
        )
    return -(advantages.detach() * reduced_log_probs).mean()


def summarize_group_samples(grouped_samples: Dict[str, object]) -> Dict[str, float]:
    """Produce compact logging stats for grouped tokenization samples."""
    num_tokens = grouped_samples["num_tokens"]
    if not isinstance(num_tokens, torch.Tensor):
        raise TypeError("grouped_samples['num_tokens'] must be a tensor")
    return {
        "avg_num_tokens": float(num_tokens.mean().item()),
        "min_num_tokens": float(num_tokens.min().item()),
        "max_num_tokens": float(num_tokens.max().item()),
    }


def main() -> None:
    """Small smoke test for grouped sampling and policy loss."""
    from src.models.token_gating_agent import encode_dna_batch

    sequences = [
        "ATATAAGCGT",
        "ACGTACGTAA",
    ]
    input_ids, attention_mask = encode_dna_batch(sequences)

    boundary_probs = torch.full_like(attention_mask, 0.2)
    grouped_samples = sample_group_tokenizations(
        sequences=sequences,
        boundary_probs=boundary_probs,
        attention_mask=attention_mask,
        group_size=4,
    )
    rewards = torch.tensor(
        [
            [-0.41, -0.39, -0.44, -0.40],
            [-0.52, -0.48, -0.55, -0.50],
        ],
        dtype=torch.float,
    )
    advantages = compute_group_advantages(rewards)
    policy_loss = compute_policy_loss(
        log_probs=grouped_samples["log_probs"],
        attention_mask=attention_mask,
        advantages=advantages,
    )

    print("GenomeRL GRPO smoke test")
    print("=" * 80)
    print("Masks shape:", tuple(grouped_samples["masks"].shape))
    print("Log-probs shape:", tuple(grouped_samples["log_probs"].shape))
    print("Num-tokens shape:", tuple(grouped_samples["num_tokens"].shape))
    print("Example grouped tokens:", grouped_samples["token_lists"][0][0])
    print("Rewards:")
    print(rewards)
    print("Advantages:")
    print(advantages)
    print("Policy loss:", float(policy_loss.item()))
    print("Sample summary:", summarize_group_samples(grouped_samples))


if __name__ == "__main__":
    main()
