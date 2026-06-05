"""Boundary policy that operates over frozen token-state sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenStateBoundaryAgent(nn.Module):
    """Predict boundary probabilities from contextual token representations."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        initial_boundary_prob: float = 0.2,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
        )
        self.encoder = nn.LSTM(
            input_size=model_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.boundary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        final_layer = self.boundary_head[-1]
        logit = torch.log(torch.tensor(initial_boundary_prob / (1.0 - initial_boundary_prob)))
        nn.init.constant_(final_layer.bias, float(logit))

    def forward(self, token_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.input_projection(token_states)
        encoded, _ = self.encoder(x)
        boundary_logits = self.boundary_head(encoded).squeeze(-1)

        if attention_mask is not None:
            boundary_logits = boundary_logits.masked_fill(attention_mask == 0, -1e9)

        boundary_probs = torch.sigmoid(boundary_logits)
        return boundary_logits, boundary_probs


def main() -> None:
    batch_size = 2
    seq_len = 12
    hidden_size = 768
    token_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[1, -3:] = 0

    agent = TokenStateBoundaryAgent(input_dim=hidden_size)
    logits, probs = agent(token_states, attention_mask)

    print("Token-state boundary agent smoke test")
    print("=" * 80)
    print("Input shape:", tuple(token_states.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Probs shape:", tuple(probs.shape))
    print("First example probs:", probs[0, :8].tolist())


if __name__ == "__main__":
    main()
