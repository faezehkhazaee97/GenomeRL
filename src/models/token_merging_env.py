"""Token Merging baseline: in-encoder token pruning/merging during transformer layers.

Implements a differentiable token merging strategy during transformer forward pass.
Tokens with highest similarity to their neighbors are merged, reducing sequence length
while preserving task-relevant information. This provides an in-encoder alternative
to pre-encoding RL-based boundary decisions.

References:
- Token Merging for Fast Stable Diffusion (Bolya et al., 2023)
- Dynamic Token Pooling (Nawrot et al., 2022)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.models.backbone import FrozenBackboneClassifier
from src.models.frozen_dnabert2_env import filter_compatible_state_dict


class TokenMergingAttention(nn.Module):
    """Wrapper around transformer attention that merges similar tokens."""

    def __init__(self, merge_ratio: float = 0.5):
        """
        Args:
            merge_ratio: Fraction of tokens to merge at each layer (0.0-1.0).
                        Merges (1 - merge_ratio) fraction of least-important tokens.
        """
        super().__init__()
        self.merge_ratio = merge_ratio

    def compute_token_importance(
        self, attention_weights: torch.Tensor, token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute token importance from attention patterns and embeddings.

        Args:
            attention_weights: (batch, heads, seq_len, seq_len)
            token_embeddings: (batch, seq_len, hidden)

        Returns:
            importance: (batch, seq_len) normalized importance scores
        """
        # Aggregate attention: how much each token is attended to
        attn_importance = attention_weights.mean(dim=1).sum(dim=1)  # (batch, seq_len)

        # Norm-based importance: tokens with larger embeddings are important
        emb_importance = torch.norm(token_embeddings, dim=-1)  # (batch, seq_len)

        # Combined importance
        importance = (attn_importance + emb_importance) / 2.0
        # Normalize
        importance = importance / (importance.max(dim=1, keepdim=True)[0] + 1e-8)
        return importance

    def merge_tokens(
        self,
        token_embeds: torch.Tensor,
        importance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge tokens with low importance scores.

        Args:
            token_embeds: (batch, seq_len, hidden)
            importance: (batch, seq_len)
            mask: (batch, seq_len) optional attention mask

        Returns:
            merged_embeds: (batch, new_seq_len, hidden)
            merge_map: (batch, new_seq_len, original_seq_len) sparse merge matrix
        """
        batch_size, seq_len, hidden_dim = token_embeds.shape

        # Determine number of tokens to keep
        if self.merge_ratio >= 1.0:
            return token_embeds, None

        keep_ratio = self.merge_ratio
        num_keep = max(1, int(seq_len * keep_ratio))

        # Find tokens to keep (highest importance)
        if mask is not None:
            # Mask out padding tokens
            masked_importance = importance.clone()
            masked_importance[mask == 0] = -1e8
        else:
            masked_importance = importance

        # Always keep first and last tokens (CLS and EOS)
        keep_indices = torch.topk(masked_importance[:, 1:-1], k=max(1, num_keep - 2), dim=1)[1] + 1

        # Add first and last
        keep_indices = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=token_embeds.device),
                keep_indices,
                torch.full((batch_size, 1), seq_len - 1, dtype=torch.long, device=token_embeds.device),
            ],
            dim=1,
        )

        # Gather kept tokens
        keep_indices_flat = keep_indices + torch.arange(batch_size, device=token_embeds.device).unsqueeze(1) * seq_len
        merged_embeds = token_embeds.reshape(batch_size * seq_len, hidden_dim)[keep_indices_flat]
        merged_embeds = merged_embeds.reshape(batch_size, -1, hidden_dim)

        return merged_embeds, keep_indices


class TokenMergingDNABERT2Environment(nn.Module):
    """DNABERT-2 with in-layer token merging for compression.

    Token merging happens during forward pass at specified layers,
    reducing sequence length as it passes through the transformer.
    """

    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: str,
        hidden_size: int = 768,
        num_labels: int = 2,
        merge_start_layer: int = 6,
        merge_ratio: float = 0.5,
        trainable_layers: int = 2,
        max_length: int = 512,
    ):
        """
        Args:
            backbone_name: HuggingFace model ID (e.g., 'zhihan1996/DNABERT-2-117M')
            checkpoint_path: Path to pretrained checkpoint
            hidden_size: Hidden dimension of DNABERT-2
            num_labels: Number of classification labels
            merge_start_layer: First layer to apply merging (0-indexed)
            merge_ratio: Fraction of tokens to keep after merging
            trainable_layers: Number of final layers to unfreeze for fine-tuning
            max_length: Max sequence length
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.merge_start_layer = merge_start_layer
        self.merge_ratio = merge_ratio
        self.trainable_layers = trainable_layers
        self.max_length = max_length

        # Load backbone
        self.backbone = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)

        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            compat = filter_compatible_state_dict(self.backbone, checkpoint)
            self.backbone.load_state_dict(compat, strict=False)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

        # Freeze backbone, unfreeze final layers
        self._freeze_and_unfreeze_layers()

    def _freeze_and_unfreeze_layers(self) -> None:
        """Freeze all layers except final trainable_layers."""
        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze final N layers
        if self.trainable_layers > 0:
            encoder_layers = list(self.backbone.encoder.layer)
            for layer in encoder_layers[-self.trainable_layers :]:
                for param in layer.parameters():
                    param.requires_grad = True

    def encode_with_merging(
        self, sequences: list[str], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Encode sequences with in-layer token merging.

        Returns:
            hidden: (batch, num_kept_tokens, hidden_size)
            stats: Compression statistics
        """
        # Tokenize
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        batch_size, seq_len = input_ids.shape
        original_token_count = attention_mask.sum(dim=1).float()

        # Forward through encoder with merging
        embeddings = self.backbone.embeddings(input_ids)

        # Track sequence length as we merge
        current_embeds = embeddings
        current_mask = attention_mask.float()
        seq_lengths = [seq_len] * batch_size

        for layer_idx, layer in enumerate(self.backbone.encoder.layer):
            # Standard layer forward
            current_embeds = layer(
                current_embeds,
                attention_mask=(current_mask > 0).unsqueeze(1).unsqueeze(2).bool() if current_mask is not None else None,
            )[0]

            # Apply token merging at specified layers
            if layer_idx >= self.merge_start_layer and self.merge_ratio < 1.0:
                # Simple merging: keep top-k important tokens
                importance = current_embeds.norm(dim=-1)  # (batch, seq_len)

                # Mask out padding
                if current_mask is not None:
                    importance[current_mask == 0] = -1e8

                # Keep CLS, keep top tokens, keep EOS
                keep_ratio = self.merge_ratio
                keep_count = max(2, int(current_embeds.shape[1] * keep_ratio))

                # Always keep first token (CLS)
                importance_interior = importance[:, 1:]
                keep_interior = torch.topk(importance_interior, k=keep_count - 1, dim=1)[1]
                keep_indices = torch.cat(
                    [
                        torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                        keep_interior + 1,
                    ],
                    dim=1,
                )

                # Gather
                batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
                current_embeds = current_embeds[batch_idx, keep_indices]
                if current_mask is not None:
                    current_mask = current_mask[batch_idx, keep_indices]

                seq_lengths = [min(keep_count, s) for s in seq_lengths]

        # Global mean-pooling
        if current_mask is not None:
            denom = current_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            pooled = (current_embeds * current_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = current_embeds.mean(dim=1)

        # Statistics
        final_token_count = current_mask.sum(dim=1).float() if current_mask is not None else torch.tensor([current_embeds.shape[1]] * batch_size)
        stats = {
            "avg_tokens_original": original_token_count.mean().item(),
            "avg_tokens_final": final_token_count.mean().item(),
            "compression_ratio": (final_token_count.mean() / original_token_count.mean()).item(),
        }

        return pooled, stats

    def forward(self, sequences: list[str], device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Classify sequences with token merging.

        Args:
            sequences: List of DNA sequences
            device: torch device

        Returns:
            logits: (batch, num_labels)
            stats: Compression and performance statistics
        """
        hidden, stats = self.encode_with_merging(sequences, device)
        logits = self.classifier(hidden)
        return logits, stats
