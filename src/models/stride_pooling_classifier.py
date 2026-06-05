"""Fixed-stride mean-pooling classifier over DNABERT-2 token states.

Non-RL baseline: pools every `stride` consecutive DNABERT-2 token embeddings
with mean pooling, then trains a linear head.  Serves as a simple differentiable
alternative to RL-learned adaptive tokenization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.models.backbone import FrozenBackboneClassifier
from src.models.frozen_dnabert2_env import filter_compatible_state_dict


class StridedPoolingClassifier(nn.Module):
    """DNABERT-2 backbone + fixed-stride mean-pooling + linear classifier.

    Args:
        backbone_name: HuggingFace model identifier.
        checkpoint_path: Path to pretrained DNABERT-2 checkpoint.
        hidden_size: DNABERT-2 hidden dimension (768).
        num_labels: Number of output classes.
        stride: Pool every this many consecutive token states.
        trainable_encoder_layers: Number of final transformer layers to unfreeze.
        max_length: Tokenizer max length.
    """

    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: str,
        hidden_size: int,
        num_labels: int,
        stride: int = 5,
        trainable_encoder_layers: int = 2,
        max_length: int = 512,
    ):
        super().__init__()
        self.stride = stride
        self.max_length = max_length

        self.model = FrozenBackboneClassifier(
            backbone_name=backbone_name,
            hidden_size=hidden_size,
            num_labels=num_labels,
            freeze_backbone=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)

        # Load pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        compat = filter_compatible_state_dict(self.model, checkpoint)
        self.model.load_state_dict(compat, strict=False)

        # Selectively unfreeze the last N encoder layers
        self._unfreeze_last_layers(trainable_encoder_layers)

        # Replace classifier head: pool → hidden → output
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def _unfreeze_last_layers(self, n: int) -> None:
        if n <= 0:
            return
        encoder_layers = list(self.model.backbone.encoder.layer)
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def encode_and_pool(
        self, sequences: list[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Encode sequences and apply stride mean-pooling.

        Returns:
            pooled: (batch, num_segments, hidden)
            seg_mask: (batch, num_segments) float mask
            stats: compression statistics
        """
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attn_mask = encoded["attention_mask"].to(device)
        special_mask = encoded["special_tokens_mask"].to(device).bool()

        outputs = self.model.backbone(input_ids=input_ids, attention_mask=attn_mask)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # Strip CLS/SEP: use positions 1: with valid token mask
        token_states = hidden[:, 1:, :]
        token_mask = (attn_mask[:, 1:].bool() & ~special_mask[:, 1:]).float()

        B, T, H = token_states.shape
        stride = self.stride
        # Pad T to a multiple of stride
        pad_len = (stride - T % stride) % stride
        if pad_len > 0:
            token_states = torch.cat([token_states, torch.zeros(B, pad_len, H, device=device)], dim=1)
            token_mask = torch.cat([token_mask, torch.zeros(B, pad_len, device=device)], dim=1)

        T_padded = token_states.shape[1]
        num_segs = T_padded // stride

        # Reshape to (B, num_segs, stride, H) and mean-pool over stride dimension
        reshaped = token_states.view(B, num_segs, stride, H)
        mask_reshaped = token_mask.view(B, num_segs, stride)

        denom = mask_reshaped.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        pooled = (reshaped * mask_reshaped.unsqueeze(-1)).sum(dim=2) / denom

        # Segment is valid if at least one of its positions was a real token
        seg_mask = (mask_reshaped.sum(dim=2) > 0).float()

        # Compression stats
        avg_real_tokens = token_mask.sum(dim=1).mean().item()
        avg_segments = seg_mask.sum(dim=1).mean().item()
        stats = {
            "avg_tokens": avg_real_tokens,
            "avg_segments": avg_segments,
            "compression_ratio": avg_segments / max(avg_real_tokens, 1e-6),
        }

        return pooled, seg_mask, stats

    def forward(
        self, sequences: list[str], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return logits and compression stats."""
        pooled, seg_mask, stats = self.encode_and_pool(sequences, device)
        # Global mean-pool over segments
        denom = seg_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        seq_repr = (pooled * seg_mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.proj(seq_repr)
        return logits, stats
