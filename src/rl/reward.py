"""Reward computation for RL training.

The first reward used in GenomeRL is intentionally simple:

    reward = -classification_loss - beta * (num_tokens / sequence_length)

This encourages the policy to preserve downstream prediction quality while
penalizing overly long tokenizations.
"""

from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn.functional as F


ScalarLike = Union[float, int, torch.Tensor]


def _to_float_tensor(value: ScalarLike) -> torch.Tensor:
    """Convert a scalar-like input into a float tensor."""
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.tensor(value, dtype=torch.float)


def compute_compression_ratio(
    num_tokens: ScalarLike,
    sequence_length: ScalarLike,
) -> torch.Tensor:
    """Compute the normalized token count."""
    num_tokens_tensor = _to_float_tensor(num_tokens)
    sequence_length_tensor = _to_float_tensor(sequence_length).clamp_min(1.0)
    return num_tokens_tensor / sequence_length_tensor


def compute_reward_from_loss(
    classification_loss: ScalarLike,
    num_tokens: ScalarLike,
    sequence_length: ScalarLike,
    beta: float,
) -> Dict[str, torch.Tensor]:
    """Compute reward from a pre-computed classification loss.

    Args:
        classification_loss: Scalar or tensor of cross-entropy-style losses.
        num_tokens: Number of tokens produced by the tokenizer.
        sequence_length: Original sequence length before tokenization.
        beta: Compression penalty strength.

    Returns:
        Dictionary containing the total reward and its components.
    """
    loss_tensor = _to_float_tensor(classification_loss)
    compression_ratio = compute_compression_ratio(num_tokens, sequence_length)
    prediction_term = -loss_tensor
    compression_term = beta * compression_ratio
    reward = prediction_term - compression_term
    return {
        "reward": reward,
        "prediction_term": prediction_term,
        "compression_term": compression_term,
        "compression_ratio": compression_ratio,
    }


def compute_reward_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_tokens: ScalarLike,
    sequence_length: ScalarLike,
    beta: float,
) -> Dict[str, torch.Tensor]:
    """Compute reward directly from classifier logits and labels."""
    losses = F.cross_entropy(logits, labels, reduction="none")
    return compute_reward_from_loss(
        classification_loss=losses,
        num_tokens=num_tokens,
        sequence_length=sequence_length,
        beta=beta,
    )


def summarize_reward_dict(reward_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Convert tensor outputs into scalar logging values."""
    summary: Dict[str, float] = {}
    for key, value in reward_dict.items():
        if isinstance(value, torch.Tensor):
            summary[key] = float(value.mean().item())
        else:
            summary[key] = float(value)
    return summary


def main() -> None:
    """Small smoke test for reward computation."""
    beta = 0.05
    classification_losses = torch.tensor([0.40, 0.42, 0.60], dtype=torch.float)
    num_tokens = torch.tensor([60, 40, 120], dtype=torch.float)
    sequence_length = torch.tensor([300, 300, 300], dtype=torch.float)

    reward_dict = compute_reward_from_loss(
        classification_loss=classification_losses,
        num_tokens=num_tokens,
        sequence_length=sequence_length,
        beta=beta,
    )
    summary = summarize_reward_dict(reward_dict)

    print("GenomeRL reward smoke test")
    print("=" * 80)
    print(f"beta: {beta}")
    print("classification losses:", classification_losses.tolist())
    print("num tokens:", num_tokens.tolist())
    print("sequence lengths:", sequence_length.tolist())
    print()
    for key, value in reward_dict.items():
        print(f"{key}: {value.tolist()}")
    print()
    print("Mean summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
