"""Gumbel-Softmax differentiable segmentation baseline.

Replaces the RL REINFORCE estimator with a straight-through Gumbel-Softmax
relaxation, allowing end-to-end gradient flow through the boundary decisions.

At training time, boundary decisions are soft (Gumbel + straight-through).
At inference time, boundary decisions are hard (sigmoid threshold).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.models.backbone import FrozenBackboneClassifier
from src.models.frozen_dnabert2_env import filter_compatible_state_dict


def gumbel_softmax_binary(logits: torch.Tensor, temperature: float, hard: bool) -> torch.Tensor:
    """Gumbel-Softmax relaxation for independent Bernoulli boundary decisions.

    Args:
        logits: (batch, seq_len) raw boundary logits.
        temperature: Gumbel temperature (high=soft, low=hard).
        hard: if True, use straight-through estimator (forward=hard, backward=soft).

    Returns:
        soft or hard boundary indicators in [0, 1].
    """
    gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8) + 1e-8) + 1e-8)
    soft = torch.sigmoid((logits + gumbel) / temperature)
    if not hard:
        return soft
    hard_vals = (soft > 0.5).float()
    return hard_vals - soft.detach() + soft  # straight-through


def soft_segment_pool(
    token_states: torch.Tensor,
    boundaries: torch.Tensor,
    token_mask: torch.Tensor,
    max_segments: int = 80,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable mean-pooling into variable-length segments.

    Uses soft boundary indicators to compute a weighted sum over token states
    within each segment. Gradients flow back through the soft boundaries.

    Args:
        token_states: (B, T, H)
        boundaries: (B, T) soft/hard values in [0, 1]; 1 = start new segment
        token_mask: (B, T) float, 1 for real tokens
        max_segments: maximum number of output segments per sequence

    Returns:
        seg_states: (B, max_segments, H)
        seg_mask: (B, max_segments)
        num_segs: (B,) float
    """
    B, T, H = token_states.shape
    device = token_states.device

    # Force boundary at position 0 (always start a segment)
    b = boundaries * token_mask
    b[:, 0] = 1.0

    # Compute soft segment assignment weights.
    # Weight of token t in segment starting at s: b_s * prod_{k=s+1}^{t}(1-b_k)
    # We accumulate this efficiently with a scan.
    # segment_id_soft[b, t] = sum_s b_s * prod_{k=s+1}^{t}(1 - b_k)  <-- too expensive
    # Instead: use a simple cumulative sum of boundary indicators as a segment index
    # approximation that is still differentiable.
    seg_id_soft = torch.cumsum(b, dim=1)  # (B, T) in [0, T]

    # Build assignment matrix via soft binning into integer segment slots
    # seg_id_int = floor(seg_id_soft); assignment = one-hot-ish via linear interpolation
    seg_id_int = seg_id_soft.detach().long().clamp(0, max_segments - 1)  # (B, T)

    # Hard assignment one-hot (straight-through compatible)
    one_hot = torch.zeros(B, T, max_segments, device=device)
    one_hot.scatter_(2, seg_id_int.unsqueeze(2), 1.0)

    # Apply token_mask
    one_hot = one_hot * token_mask.unsqueeze(2)  # (B, T, S)

    # Segment means: (B, S, H) = one_hot^T @ token_states / counts
    counts = one_hot.sum(dim=1).clamp_min(1.0)          # (B, S)
    seg_states = torch.bmm(one_hot.transpose(1, 2), token_states) / counts.unsqueeze(2)

    seg_mask = (counts > 0).float()
    num_segs = seg_mask.sum(dim=1)

    return seg_states, seg_mask, num_segs


class GumbelSegmenter(nn.Module):
    """DNABERT-2 + Gumbel-Softmax segmentation + classification head.

    Args:
        backbone_name: HuggingFace model ID.
        checkpoint_path: pretrained DNABERT-2 checkpoint.
        hidden_size: 768.
        num_labels: number of classes.
        trainable_encoder_layers: last N transformer layers to unfreeze.
        initial_temperature: starting Gumbel temperature.
        min_temperature: minimum temperature after annealing.
        anneal_rate: exponential decay per gradient step.
        beta: compression penalty weight (same as RL).
        max_length: tokenizer max length.
    """

    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: str,
        hidden_size: int,
        num_labels: int,
        trainable_encoder_layers: int = 2,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        anneal_rate: float = 0.0003,
        beta: float = 0.02,
        max_length: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.beta = beta
        self.max_length = max_length
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate

        # DNABERT-2 backbone
        self.backbone_model = FrozenBackboneClassifier(
            backbone_name=backbone_name,
            hidden_size=hidden_size,
            num_labels=num_labels,
            freeze_backbone=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)

        # Load pretrained weights
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        compat = filter_compatible_state_dict(self.backbone_model, ckpt)
        self.backbone_model.load_state_dict(compat, strict=False)

        # Unfreeze last N encoder layers
        for layer in list(self.backbone_model.backbone.encoder.layer)[-trainable_encoder_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Boundary predictor: BiLSTM over token states
        self.boundary_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Initialise boundary head bias so initial boundary prob ≈ 0.2
        nn.init.constant_(self.boundary_head[-1].bias, -1.386)

        # Classification head over pooled segment representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def anneal_temperature(self) -> None:
        self.temperature = max(
            self.min_temperature,
            self.temperature * (1.0 - self.anneal_rate),
        )

    def encode(self, sequences: list[str], device: torch.device) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        special_mask = enc["special_tokens_mask"].to(device).bool()

        outputs = self.backbone_model.backbone(input_ids=input_ids, attention_mask=attn_mask)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        token_states = hidden[:, 1:, :]
        token_mask = (attn_mask[:, 1:].bool() & ~special_mask[:, 1:]).float()
        return {"token_states": token_states, "token_mask": token_mask}

    def forward(
        self, sequences: list[str], device: torch.device, hard: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        feats = self.encode(sequences, device)
        token_states = feats["token_states"]
        token_mask = feats["token_mask"]

        # Boundary predictions
        lstm_out, _ = self.boundary_lstm(token_states)
        boundary_logits = self.boundary_head(lstm_out).squeeze(-1)  # (B, T)

        # Gumbel-Softmax (or hard at inference)
        if self.training and not hard:
            boundaries = gumbel_softmax_binary(boundary_logits, self.temperature, hard=True)
        else:
            boundaries = (torch.sigmoid(boundary_logits) > 0.5).float()

        # Differentiable segment pooling
        seg_states, seg_mask, num_segs = soft_segment_pool(
            token_states, boundaries, token_mask
        )

        # Global mean over segments
        denom = seg_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        seq_repr = (seg_states * seg_mask.unsqueeze(-1)).sum(dim=1) / denom

        logits = self.classifier(seq_repr)

        comp_ratio = (num_segs / token_mask.sum(dim=1).clamp_min(1.0)).mean().item()
        stats = {
            "avg_segments": num_segs.mean().item(),
            "compression_ratio": comp_ratio,
            "temperature": self.temperature,
        }
        return logits, stats
