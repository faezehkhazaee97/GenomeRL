"""Transformer-based boundary policy with learnable [CLS] global context token.

Addresses reviewer Q2: replaces BiLSTM with a small transformer encoder that has
an explicit global summary token, allowing boundary decisions to condition on a
distilled sequence representation rather than only local LSTM hidden states.

Architecture:
  1. Linear(768 → model_dim) input projection
  2. Prepend learnable [CLS] token (position 0)
  3. Sinusoidal + learnable positional encodings
  4. TransformerEncoder (num_layers, num_heads, dim=model_dim)
  5. Boundary head: for each position i>0, cat(h_i, h_cls) → Linear → sigmoid
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerBoundaryAgent(nn.Module):
    """Transformer boundary policy with explicit global [CLS] context."""

    def __init__(
        self,
        input_dim: int = 768,
        model_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        initial_boundary_prob: float = 0.2,
    ):
        super().__init__()
        self.model_dim = model_dim

        # Project DNABERT-2 hidden states to model_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
        )

        # Learnable [CLS] token prepended to every sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional encoding (applied after prepending [CLS])
        self.pos_embedding = nn.Embedding(max_seq_len + 1, model_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Boundary head: fuses per-position repr with [CLS] global repr
        self.boundary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

        # Bias final layer toward initial_boundary_prob
        logit = math.log(initial_boundary_prob / (1.0 - initial_boundary_prob))
        nn.init.constant_(self.boundary_head[-1].bias, logit)

    def forward(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            token_states: (batch, seq_len, input_dim)
            attention_mask: (batch, seq_len) float mask (1=valid, 0=pad)

        Returns:
            boundary_logits: (batch, seq_len)
            boundary_probs:  (batch, seq_len)
        """
        B, T, _ = token_states.shape

        # Project input
        x = self.input_projection(token_states)  # (B, T, D)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat([cls, x], dim=1)           # (B, T+1, D)

        # Positional encoding
        positions = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Build transformer padding mask: True = ignore (opposite of attention_mask)
        if attention_mask is not None:
            # pad_mask shape: (B, T+1); [CLS] position always attended
            cls_mask = torch.zeros(B, 1, device=x.device)
            pad_mask = torch.cat([cls_mask, 1.0 - attention_mask], dim=1).bool()
        else:
            pad_mask = None

        # Transformer encode
        out = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, T+1, D)

        # Split [CLS] from sequence tokens
        cls_out = out[:, 0:1, :].expand(-1, T, -1)  # (B, T, D)
        tok_out = out[:, 1:, :]                      # (B, T, D)

        # Boundary head: fuse local + global
        fused = torch.cat([tok_out, cls_out], dim=-1)  # (B, T, 2D)
        boundary_logits = self.boundary_head(fused).squeeze(-1)  # (B, T)

        # Mask padding positions
        if attention_mask is not None:
            boundary_logits = boundary_logits.masked_fill(attention_mask == 0, -1e9)

        boundary_probs = torch.sigmoid(boundary_logits)
        return boundary_logits, boundary_probs


def main() -> None:
    B, T, D = 2, 20, 768
    token_states = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[1, -4:] = 0

    agent = TransformerBoundaryAgent(input_dim=D)
    logits, probs = agent(token_states, mask)

    print("Transformer boundary agent smoke test")
    print("=" * 60)
    total = sum(p.numel() for p in agent.parameters())
    print(f"Parameters: {total:,}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Probs  shape: {tuple(probs.shape)}")
    print(f"Sample probs: {probs[0, :8].tolist()}")


if __name__ == "__main__":
    main()
