"""DNABERT-2 environment with learned attention-weighted segment pooling.

Replaces uniform mean-pooling with a single-head attention mechanism:
    score_i = MLP(h_i) -> softmax over segment positions
    segment_repr = sum(score_i * h_i)

This allows the model to weight token states within a segment by relevance,
rather than treating all positions equally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment


class AttentionPoolingEnv(FineTunedDNABERT2Environment):
    """Extends FineTunedDNABERT2Environment with attention-weighted segment pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.hidden_size
        # Learned attention scorer: hidden -> scalar
        self.attn_scorer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def logits_from_segments(
        self,
        cls_states: torch.Tensor,
        segment_states: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-weighted mean-pool over segments, then classify."""
        # segment_states: (B, S, H), segment_mask: (B, S)
        scores = self.attn_scorer(segment_states).squeeze(-1)  # (B, S)
        # Mask padding segments with -inf before softmax
        scores = scores.masked_fill(segment_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, S)
        # Weighted sum
        seq_repr = (weights.unsqueeze(-1) * segment_states).sum(dim=1)  # (B, H)
        return self.classifier(seq_repr)
