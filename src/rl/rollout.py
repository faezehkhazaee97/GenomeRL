"""Rollout generation utilities for frozen-token-state grouping."""

from __future__ import annotations

from typing import Dict, List

import torch


def group_token_states(
    token_states: torch.Tensor,
    boundary_masks: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Group contiguous frozen token states into learned segments.

    Args:
        token_states: [batch, seq_len, hidden_size]
        boundary_masks: [batch, seq_len] with 1 marking segment starts
        attention_mask: [batch, seq_len] with 1 for valid token positions
    """
    batch_size, _, hidden_size = token_states.shape
    grouped_segments: List[torch.Tensor] = []
    num_segments: List[int] = []

    for batch_index in range(batch_size):
        valid_len = int(attention_mask[batch_index].sum().item())
        valid_states = token_states[batch_index, :valid_len]
        valid_boundaries = boundary_masks[batch_index, :valid_len].bool()

        segments = []
        current_segment = []
        for position in range(valid_len):
            if position == 0 or valid_boundaries[position]:
                if current_segment:
                    segments.append(torch.stack(current_segment, dim=0).mean(dim=0))
                    current_segment = []
            current_segment.append(valid_states[position])

        if current_segment:
            segments.append(torch.stack(current_segment, dim=0).mean(dim=0))

        if not segments:
            segments = [torch.zeros(hidden_size, device=token_states.device)]

        grouped = torch.stack(segments, dim=0)
        grouped_segments.append(grouped)
        num_segments.append(grouped.shape[0])

    max_segments = max(num_segments)
    segment_states = torch.zeros(
        batch_size,
        max_segments,
        hidden_size,
        dtype=token_states.dtype,
        device=token_states.device,
    )
    segment_mask = torch.zeros(
        batch_size,
        max_segments,
        dtype=torch.float,
        device=token_states.device,
    )

    for batch_index, grouped in enumerate(grouped_segments):
        length = grouped.shape[0]
        segment_states[batch_index, :length] = grouped
        segment_mask[batch_index, :length] = 1.0

    return {
        "segment_states": segment_states,
        "segment_mask": segment_mask,
        "num_segments": torch.tensor(num_segments, dtype=torch.float, device=token_states.device),
    }


def summarize_segment_samples(num_segments: torch.Tensor) -> Dict[str, float]:
    """Compact logging stats for grouped segment counts."""
    return {
        "avg_num_segments": float(num_segments.mean().item()),
        "min_num_segments": float(num_segments.min().item()),
        "max_num_segments": float(num_segments.max().item()),
    }
