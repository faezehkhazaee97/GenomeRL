"""
DNABERT-2 environment with segment-aware positional encodings.
Tests whether explicit segment position embeddings improve boundary-sensitive tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment


class SegmentAwareDNABERT2Environment(FineTunedDNABERT2Environment):
    """
    Extends FineTunedDNABERT2Environment with segment-aware positional encodings.

    Instead of mean-pooling token states and losing positional signal, this adds
    explicit learned segment position embeddings that indicate where each segment
    appears in the compressed sequence.
    """

    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: str,
        hidden_size: int,
        num_labels: int,
        max_length: int = 512,
        cls_mix_weight: float = 0.0,
        trainable_encoder_layers: int = 2,
        freeze_embeddings: bool = True,
        segment_pe_dim: int = 128,
        max_segments: int = 100,
    ):
        super().__init__(
            backbone_name=backbone_name,
            checkpoint_path=checkpoint_path,
            hidden_size=hidden_size,
            num_labels=num_labels,
            max_length=max_length,
            cls_mix_weight=cls_mix_weight,
            trainable_encoder_layers=trainable_encoder_layers,
            freeze_embeddings=freeze_embeddings,
        )

        # Segment positional encoding
        self.segment_pe_dim = segment_pe_dim
        self.max_segments = max_segments

        # Learnable segment position embeddings
        self.segment_position_embeddings = nn.Embedding(max_segments, segment_pe_dim)

        # Project segment PE to match hidden_size for addition
        self.segment_pe_projection = nn.Linear(segment_pe_dim, hidden_size)

    def logits_from_segments(
        self,
        cls_states: torch.Tensor,
        segment_states: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits from segments, with segment-aware positional encodings.

        Adds explicit learned segment position embeddings before final classification,
        preserving positional signal that mean-pooling alone would discard.

        Args:
            cls_states: (batch, hidden_size) pooled representation
            segment_states: (batch, max_segments, hidden_size) segment representations
            segment_mask: (batch, max_segments) boolean mask for valid segments

        Returns:
            logits: (batch, num_labels)
        """
        batch_size, max_segs, hidden_size = segment_states.shape

        # Create positional indices for each segment
        segment_positions = torch.arange(max_segs, device=segment_states.device)
        segment_positions = segment_positions.unsqueeze(0).expand(batch_size, -1)

        # Get learnable segment position embeddings
        segment_pe = self.segment_position_embeddings(segment_positions)  # (batch, max_segs, pe_dim)
        segment_pe = self.segment_pe_projection(segment_pe)  # (batch, max_segs, hidden_size)

        # Add segment PE to segment states, preserving positional signal
        segment_states_with_pe = segment_states + segment_pe  # (batch, max_segs, hidden_size)

        # Use parent class's masked_mean pooling on enhanced segment states
        from src.models.finetuned_dnabert2_env import masked_mean
        pooled_segments = masked_mean(segment_states_with_pe, segment_mask)

        # Apply cls_mix_weight if configured
        if self.cls_mix_weight <= 0.0:
            sequence_repr = pooled_segments
        else:
            sequence_repr = (
                self.cls_mix_weight * cls_states
                + (1.0 - self.cls_mix_weight) * pooled_segments
            )

        return self.classifier(sequence_repr)


if __name__ == "__main__":
    # Test instantiation
    env = SegmentAwareDNABERT2Environment(
        backbone_name="zhihan1996/DNABERT-2-117M",
        checkpoint_path="results/baselines/dnabert2_bpe/best_model.pt",
        hidden_size=768,
        num_labels=2,
        segment_pe_dim=128,
    )

    # Test forward pass
    batch_size = 4
    max_segs = 20
    hidden_size = 768

    cls_states = torch.randn(batch_size, hidden_size)
    segment_states = torch.randn(batch_size, max_segs, hidden_size)
    segment_mask = torch.ones(batch_size, max_segs, dtype=torch.bool)

    logits = env.logits_from_segments(cls_states, segment_states, segment_mask)
    print(f"Input cls shape: {cls_states.shape}")
    print(f"Input segments shape: {segment_states.shape}")
    print(f"Output logits shape: {logits.shape}")
    print("✓ SegmentAwareDNABERT2Environment test passed")
